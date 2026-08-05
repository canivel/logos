"""export_parity: the .lpack artifact vs an independent bit-unpacker, plus the
round-trip logit-parity gate (PLAN.md s5: a P0 exit criterion).

The reference path reads manifest.json + tensors.bin directly and decodes the
little-endian bitstreams with its own shift-and-mask decoder (format spec:
value i occupies bits [i*b, (i+1)*b); ternary stores code+1 in 2 bits, k-bit
stores code+2^(k-1)); load_artifact is only exercised as the round-trip under
test, never as the reference.
"""

from __future__ import annotations

import glob
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from validation.base import GateOutcome, Probe

RUNS_GLOB = str(Path(__file__).resolve().parents[2] / "runs" / "local_p0" / "*" / "parity.json")

TINY = dict(
    d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=256,
    vocab_size=512, max_seq_len=64,
)

# Exact per-format code-stream byte math (bit-contiguous little-endian pack).
EXPECT_NBYTES = {
    "i2_s": lambda n: math.ceil(n / 4),
    "gint2": lambda n: math.ceil(n / 4),
    "gint3": lambda n: math.ceil(n / 8) * 3,
    "gint4": lambda n: math.ceil(n / 2),
}


def _decode_bits(raw: bytes, bits: int, n: int) -> np.ndarray:
    """Own little-endian bitstream decoder: value i = bits [i*bits, (i+1)*bits)."""
    buf = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    bit_idx = np.arange(n, dtype=np.int64)[:, None] * bits + np.arange(bits, dtype=np.int64)
    bit = (buf[bit_idx // 8] >> (bit_idx % 8)) & 1
    return (bit << np.arange(bits, dtype=np.int64)).sum(axis=1)


class ExportParityProbe(Probe):
    name = "export_parity"
    description = (
        ".lpack export vs an independent manifest+bitstream unpacker; "
        "load_artifact round-trip logit parity; packed byte math"
    )
    gate_specs = [
        ("G1", "independent unpack of export_artifact output reproduces the live "
               "model's quantize_weight() for arms {1.58, 2, 3, 4} "
               "(max |diff| <= 1e-6)"),
        ("G2", "load_artifact round-trip: re-derived codes bitwise-identical to "
               "the artifact (gint scales bitwise, ternary gamma within 1e-7, "
               "bf16 masters bitwise) and max KL < 1e-4 on 8 synthetic prompts, "
               "all 4 quantized arms + bf16; max |dlogit| reported non-gating"),
        ("G3", "packed code streams have exact per-format byte lengths: ternary "
               "ceil(n/4); 2b ceil(n/4); 3b ceil(n/8)*3; 4b ceil(n/2)"),
        ("G4", "every runs/local_p0/*/parity.json reports pass=true "
               "(vacuously true if none exist)"),
    ]

    def collect(self) -> dict:
        from logos.config import ModelConfig, Precision
        from logos.export.artifact import export_artifact, load_artifact
        from logos.export.pack import FMT_BY_PRECISION
        from logos.model.transformer import Transformer

        m: dict = {"unpack_max_diff": {}, "logit_max_diff": {}, "kl_max": {},
                   "roundtrip_ok": {}, "roundtrip_detail": "",
                   "nbytes_ok": True, "nbytes_detail": ""}
        tmp = Path(tempfile.mkdtemp(prefix="lpack_probe_"))
        g = torch.Generator().manual_seed(7)
        prompts = torch.randint(0, TINY["vocab_size"], (8, 48), generator=g)

        for prec in (Precision.W1_58, Precision.W2, Precision.W3, Precision.W4,
                     Precision.BF16):
            cfg = ModelConfig(precision=prec, **TINY)
            torch.manual_seed(11)
            model = Transformer(cfg).eval()
            out_dir = export_artifact(model, tmp / f"arm_{prec.value.replace('.', '_')}")

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            raw = (out_dir / "tensors.bin").read_bytes()
            modules = dict(model.named_modules())
            fmt = FMT_BY_PRECISION[prec.value]

            max_diff = 0.0
            decoded: dict[str, tuple] = {}  # name -> (codes, scale array, kind, bits)
            for e in manifest["tensors"]:
                if e["kind"] == "ternary":
                    n = int(np.prod(e["shape"]))
                    codes = _decode_bits(
                        raw[e["codes_offset"]: e["codes_offset"] + e["codes_nbytes"]], 2, n
                    ) - 1  # stored {0,1,2} -> {-1,0,1}
                    assert set(np.unique(codes)) <= {-1, 0, 1}
                    gamma = np.frombuffer(
                        raw[e["scale_offset"]: e["scale_offset"] + e["scale_nbytes"]],
                        dtype="<f4",
                    )[0]
                    eff = (codes.astype(np.float32) * gamma).reshape(e["shape"])
                    decoded[e["name"]] = (codes.reshape(e["shape"]), np.array([gamma]), "ternary", 2)
                    bits = 2
                elif e["kind"] == "gint":
                    bits = e["bits"]
                    n = int(np.prod(e["shape"]))
                    codes = _decode_bits(
                        raw[e["codes_offset"]: e["codes_offset"] + e["codes_nbytes"]],
                        bits, n,
                    ) - (1 << (bits - 1))  # offset-binary -> signed
                    scales = np.frombuffer(
                        raw[e["scale_offset"]: e["scale_offset"] + e["scale_nbytes"]],
                        dtype="<f4",
                    ).reshape(e["scales_shape"])
                    out_f, groups, gs = e["shape"]
                    eff = (
                        codes.reshape(out_f, groups, gs).astype(np.float32)
                        * scales[:, :, None]
                    ).reshape(out_f, groups * gs)
                    decoded[e["name"]] = (codes.reshape(out_f, groups, gs), scales, "gint", bits)
                else:
                    continue
                exp_n = EXPECT_NBYTES[fmt](n)
                if e["codes_nbytes"] != exp_n:
                    m["nbytes_ok"] = False
                    m["nbytes_detail"] += (
                        f"{prec.value}:{e['name']} nbytes={e['codes_nbytes']} expected={exp_n}; "
                    )
                with torch.no_grad():
                    live = modules[e["name"]].quantize_weight().float().numpy()
                max_diff = max(max_diff, float(np.abs(live.reshape(eff.shape) - eff).max()))
            if prec is not Precision.BF16:
                m["unpack_max_diff"][prec.value] = max_diff

            # round-trip: reloaded masters must re-derive to the artifact's
            # exact codes/scales (numpy re-derivation, not the stack's), plus
            # KL parity; max-abs logit diff reported non-gating (the int8
            # activation round discontinuity amplifies ulp master diffs with
            # depth on trained models — see parity.py's weights_parity note).
            model2 = load_artifact(out_dir)
            modules2 = dict(model2.named_modules())
            rt_ok = True
            for name, (a_codes, a_scales, kind, bits) in decoded.items():
                mod2 = modules2[name]
                w2 = mod2.weight.detach().float().numpy().astype(np.float32)
                if kind == "ternary":
                    gamma2 = np.float32(max(np.abs(w2).mean(dtype=np.float32),
                                            np.float32(1e-8)))
                    codes2 = np.clip(np.rint((w2 / gamma2).astype(np.float32)),
                                     -1, 1).astype(np.int64)
                    ok = bool(np.array_equal(codes2, a_codes)
                              and abs(float(gamma2) - float(a_scales[0])) <= 1e-7)
                else:
                    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
                    s2 = mod2.scale.detach().float().numpy().astype(np.float32)
                    gfac = np.float32(1.0 / math.sqrt(w2.size * qmax))
                    s_ = np.maximum(s2, np.float32(1e-8))
                    sg = (s_ * gfac).astype(np.float32)
                    s_eff = (sg + (s_ - sg).astype(np.float32)).astype(np.float32)
                    wg = w2.reshape(a_codes.shape)
                    codes2 = np.clip(
                        np.rint((wg / s_eff[:, :, None]).astype(np.float32)),
                        qmin, qmax,
                    ).astype(np.int64)
                    ok = bool(np.array_equal(codes2, a_codes)
                              and np.array_equal(s2, a_scales.astype(np.float32)))
                if not ok:
                    rt_ok = False
                    m["roundtrip_detail"] += f"{prec.value}:{name}; "
            # raw fp32 masters (embed, norms, whole model on the bf16 arm)
            params2 = dict(model2.named_parameters())
            for e in manifest["tensors"]:
                if e["kind"] != "raw":
                    continue
                buf = np.frombuffer(raw[e["offset"]: e["offset"] + e["nbytes"]],
                                    dtype="<f4").reshape(e["shape"])
                p2 = params2[e["name"]].detach().float().numpy()
                if not np.array_equal(p2, buf):
                    rt_ok = False
                    m["roundtrip_detail"] += f"{prec.value}:{e['name']} (raw); "
            m["roundtrip_ok"][prec.value] = rt_ok
            with torch.no_grad():
                la = model(prompts)[0].float()
                lb = model2(prompts)[0].float()
            m["logit_max_diff"][prec.value] = float((la - lb).abs().max().item())
            p = F.softmax(la, dim=-1)
            kl = (p * (F.log_softmax(la, dim=-1) - F.log_softmax(lb, dim=-1))).sum(-1)
            m["kl_max"][prec.value] = float(kl.max().item())

        # G4: background micro-P0 parity artifacts
        files = sorted(glob.glob(RUNS_GLOB))
        m["parity_files"] = len(files)
        m["parity_all_pass"] = True
        m["parity_fail_detail"] = ""
        for f in files:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            if not d.get("pass", False):
                m["parity_all_pass"] = False
                m["parity_fail_detail"] += f"{f}: pass={d.get('pass')}; "
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)
        worst_unpack = max(m["unpack_max_diff"].values())
        worst_logit = max(m["logit_max_diff"].values())
        worst_kl = max(m["kl_max"].values())
        if m["parity_files"] == 0:
            g4_detail = "no run artifacts present (gate vacuous)"
        else:
            g4_detail = (f"{m['parity_files']} parity.json checked; "
                         f"{m['parity_fail_detail'] or 'all pass=true'}")
        return [
            GateOutcome(
                "G1", specs["G1"], worst_unpack <= 1e-6,
                detail="max |unpack - quantize_weight| per arm: "
                       + ", ".join(f"{k}={v:.3e}" for k, v in m["unpack_max_diff"].items()),
            ),
            GateOutcome(
                "G2", specs["G2"],
                all(m["roundtrip_ok"].values()) and worst_kl < 1e-4,
                detail=(f"bitwise round-trip ok={m['roundtrip_ok']}; "
                        f"max KL = {worst_kl:.3e}; "
                        f"max |dlogit| = {worst_logit:.3e} (non-gating); "
                        f"{m['roundtrip_detail'] or ''}")
            ),
            GateOutcome("G3", specs["G3"], m["nbytes_ok"],
                        detail=m["nbytes_detail"] or "all code streams match exact byte math"),
            GateOutcome("G4", specs["G4"], m["parity_all_pass"], detail=g4_detail),
        ]
