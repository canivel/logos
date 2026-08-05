"""Checkpoint -> deployable artifact (.lpack) and back (PLAN.md s5: export
parity is a P0 exit criterion).

.lpack layout: out_dir/manifest.json (config, per-tensor entries with byte
offsets) + out_dir/tensors.bin (concatenated raw little-endian buffers).
Quantized linears store packed codes + fp32 scales; everything else (embed,
norms, untied head) stores exact fp32 masters.

`load_artifact` rebuilds a Transformer whose forward is the DEQUANTIZED
equivalent of the exported model: masters are replaced by values on which
fake-quant is idempotent, so re-quantization reproduces the exported codes
and scales exactly.

Why parity holds (up to float associativity):
- GroupIntLinear: the `scale` Parameter is restored bitwise from the artifact
  and masters are set to codes*scale, so round(masters/scale) = codes exactly.
- BitLinear recomputes gamma = mean|W| on forward, so masters = codes*gamma
  would NOT round-trip (recomputed gamma' = gamma * mean|codes| < gamma).
  Instead masters = codes * (gamma / mean|codes|): then mean|masters| = gamma,
  and RoundClip(masters/gamma) = RoundClip(codes / mean|codes|) = codes, since
  nonzero entries have magnitude >= 1 and clip back to +/-1.

GGUF export (`export_gguf`) is behind a guarded `import gguf`. Ternary maps
to TQ1_0-style packing which requires the bitnet.cpp conversion toolchain;
real bitnet.cpp / llama.cpp parity runs happen on the RunPod image. Local
tests use .lpack only.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from logos.config import ModelConfig, Precision
from logos.export.pack import pack_int, pack_ternary, unpack_int, unpack_ternary
from logos.model.transformer import Transformer
from logos.quant.bitlinear import BitLinear
from logos.quant.paretoq import GroupIntLinear

LPACK_FORMAT = "lpack-v1"


@torch.no_grad()
def export_artifact(model: Transformer, out_dir) -> Path:
    """Write packed tensors + manifest.json to out_dir. Returns out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffers: list[bytes] = []
    tensors: list[dict] = []
    offset = 0

    def put(b: bytes) -> tuple[int, int]:
        nonlocal offset
        buffers.append(b)
        o, n = offset, len(b)
        offset += n
        return o, n

    handled: set[int] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            codes, gamma = mod.packed_weight()
            co, cn = put(pack_ternary(codes))
            so, sn = put(np.array([float(gamma)], dtype="<f4").tobytes())
            tensors.append(
                {
                    "name": name,
                    "kind": "ternary",
                    "shape": list(codes.shape),
                    "codes_offset": co,
                    "codes_nbytes": cn,
                    "scale_offset": so,
                    "scale_nbytes": sn,
                }
            )
            handled.add(id(mod.weight))
        elif isinstance(mod, GroupIntLinear):
            codes, scales = mod.packed_weight()
            co, cn = put(pack_int(codes, mod.bits))
            so, sn = put(scales.cpu().numpy().astype("<f4").tobytes())
            tensors.append(
                {
                    "name": name,
                    "kind": "gint",
                    "bits": mod.bits,
                    "group_size": mod.group_size,
                    "shape": list(codes.shape),  # [out, groups, gs]
                    "scales_shape": list(scales.shape),  # [out, groups]
                    "codes_offset": co,
                    "codes_nbytes": cn,
                    "scale_offset": so,
                    "scale_nbytes": sn,
                }
            )
            handled.add(id(mod.weight))
            handled.add(id(mod.scale))

    seen: set[int] = set()
    for name, p in model.named_parameters():  # tied head appears once
        if id(p) in handled or id(p) in seen:
            continue
        seen.add(id(p))
        o, n = put(p.detach().float().cpu().numpy().astype("<f4").tobytes())
        tensors.append(
            {"name": name, "kind": "raw", "dtype": "float32", "shape": list(p.shape),
             "offset": o, "nbytes": n}
        )

    cfg = asdict(model.cfg)  # Precision is a str-enum -> json-safe
    manifest = {
        "format": LPACK_FORMAT,
        "config": cfg,
        "tensors": tensors,
        "total_bytes": offset,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "tensors.bin").write_bytes(b"".join(buffers))
    return out_dir


@torch.no_grad()
def load_artifact(out_dir) -> Transformer:
    """Rebuild a Transformer from an .lpack dir. Masters are set so that
    fake-quant is idempotent (see module docstring); forward equals the
    exported model's forward up to float associativity."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("format") == LPACK_FORMAT, f"unknown format {manifest.get('format')!r}"
    cfg_d = dict(manifest["config"])
    cfg_d["precision"] = Precision(cfg_d["precision"])
    cfg = ModelConfig(**cfg_d)
    model = Transformer(cfg)
    raw = (out_dir / "tensors.bin").read_bytes()
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())

    def f32(o: int, n: int) -> np.ndarray:
        return np.frombuffer(raw[o : o + n], dtype="<f4").copy()

    for e in manifest["tensors"]:
        if e["kind"] == "ternary":
            m = modules[e["name"]]
            codes = torch.from_numpy(
                unpack_ternary(raw[e["codes_offset"] : e["codes_offset"] + e["codes_nbytes"]],
                               tuple(e["shape"])).copy()
            ).float()
            gamma = float(f32(e["scale_offset"], e["scale_nbytes"])[0])
            rho = codes.abs().mean().item()  # mean|codes| in [0, 1]
            w = codes * (gamma / rho) if rho > 0 else torch.zeros_like(codes)
            m.weight.copy_(w)
        elif e["kind"] == "gint":
            m = modules[e["name"]]
            codes = torch.from_numpy(
                unpack_int(raw[e["codes_offset"] : e["codes_offset"] + e["codes_nbytes"]],
                           e["bits"], tuple(e["shape"])).copy()
            ).float()  # [out, groups, gs]
            scales = torch.from_numpy(
                f32(e["scale_offset"], e["scale_nbytes"]).reshape(e["scales_shape"])
            )
            m.scale.copy_(scales)  # restore the Parameter bitwise
            m.weight.copy_((codes * scales.unsqueeze(-1)).reshape(m.out_features, m.in_features))
        else:  # raw fp32 master (embed, norms, untied head)
            params[e["name"]].copy_(
                torch.from_numpy(f32(e["offset"], e["nbytes"]).reshape(e["shape"]))
            )
    return model.eval()


def export_gguf(model: Transformer, path) -> Path:
    """GGUF export, optional dependency. Ternary arms target TQ1_0-style
    packing, which requires the bitnet.cpp conversion toolchain; 2-4 bit arms
    target llama.cpp/torchao formats. Real bitnet.cpp/llama.cpp logit-parity
    runs execute on the RunPod image (PLAN.md s5); local tests use .lpack.

    This writer emits an F32 GGUF of the dequantized weights as the neutral
    interchange baseline; downstream toolchains requantize to their native
    packing (i2_s for bitnet.cpp).
    """
    try:
        import gguf
    except ImportError as e:
        raise ImportError(
            "gguf is not installed; GGUF export is only needed on the RunPod "
            "export image. `uv pip install gguf` there. Local tests use the "
            ".lpack path (export_artifact/load_artifact)."
        ) from e

    path = Path(path)
    cfg = model.cfg
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.d_model)
    writer.add_block_count(cfg.n_layers)
    writer.add_head_count(cfg.n_heads)
    writer.add_head_count_kv(cfg.n_kv_heads)
    writer.add_layer_norm_rms_eps(cfg.norm_eps)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, (BitLinear, GroupIntLinear)):
                w = mod.quantize_weight().detach().float().cpu().numpy()
                writer.add_tensor(name + ".weight", w)
        seen: set[int] = set()
        handled = {
            id(m.weight)
            for m in model.modules()
            if isinstance(m, (BitLinear, GroupIntLinear))
        }
        for name, p in model.named_parameters():
            if id(p) in handled or id(p) in seen or name.endswith(".scale"):
                continue
            seen.add(id(p))
            writer.add_tensor(name, p.detach().float().cpu().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
