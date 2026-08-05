"""quant_numerics: independent numpy reimplementation of the LOGOS quantizer
ladder from the source equations, compared against src/logos/quant.

References (re-derived here, never imported from the code under test):
- BitNet b1.58 absmean: gamma = mean|W|; Wq = clip(round(W/gamma), -1, 1)*gamma.
- LSQ-style per-group symmetric k-bit: q = clip(round(W/s), -2^(k-1), 2^(k-1)-1)*s
  with the layer's OWN scale values read as plain numbers. The LSQ grad-rescale
  trick s*g + (s - s*g).detach() is value-identity in exact arithmetic; we
  replicate the same fp32 expression so the reference is bitwise-faithful.
- Per-token absmax int8 activations: s = 127/max|x|; xq = clamp(round(x*s),-128,127)/s.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from validation.base import GateOutcome, Probe

TERNARY_SHAPES = [(256, 512), (128, 384), (64, 64)]  # (in_features, out_features)


def _np_ternary(w: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """Absmean ternary reference, all-fp32."""
    gamma = np.float32(max(np.abs(w).mean(dtype=np.float32), np.float32(1e-8)))
    q = np.clip(np.rint((w / gamma).astype(np.float32)), -1.0, 1.0).astype(np.float32)
    return (q * gamma).astype(np.float32), gamma


def _np_gint(w: np.ndarray, s: np.ndarray, bits: int, numel: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group symmetric k-bit reference. w: [out, groups, gs], s: [out, groups].
    Returns (effective weights [out, groups, gs], integer codes)."""
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    g = np.float32(1.0 / math.sqrt(numel * qmax))
    s = np.maximum(s.astype(np.float32), np.float32(1e-8))
    sg = (s * g).astype(np.float32)
    s_eff = (sg + (s - sg).astype(np.float32)).astype(np.float32)[:, :, None]
    q = np.clip(np.rint((w / s_eff).astype(np.float32)), qmin, qmax).astype(np.float32)
    return (q * s_eff).astype(np.float32), q


class QuantNumericsProbe(Probe):
    name = "quant_numerics"
    description = (
        "Ternary absmean + 2/3/4-bit per-group quantizers vs an independent "
        "numpy reimplementation of the BitNet b1.58 / LSQ equations"
    )
    gate_specs = [
        ("G1", "ternary effective weights match the numpy absmean reference on "
               "seeded masters (max |diff| <= 1e-7)"),
        ("G2", "ternary levels are exactly {-gamma, 0, +gamma} and gamma equals "
               "numpy mean|W| within 1e-7"),
        ("G3", "2/3/4-bit effective weights match the numpy per-group reference "
               "given the layer's own scale values (max |diff| <= 1e-7); "
               "<= 2^k distinct levels per group"),
        ("G4", "ternary ||W - Wq|| matches the numpy-predicted error on the same "
               "tensor (rel diff <= 1e-5) and error decreases monotonically "
               "2b -> 3b -> 4b on identical masters with each layer's init scales"),
        ("G5", "activation quant: per-token codes are integers in [-128, 127] and "
               "|x - xq| <= 0.5/scale per element"),
    ]

    def collect(self) -> dict:
        from logos.quant.activations import ActQuant
        from logos.quant.bitlinear import BitLinear
        from logos.quant.paretoq import GroupIntLinear

        m: dict = {}

        # ---- ternary (G1, G2, part of G4) ----
        tern_diff, gamma_diff, level_ok, err_rel = [], [], [], []
        for i, (fin, fout) in enumerate(TERNARY_SHAPES):
            torch.manual_seed(101 + i)
            bl = BitLinear(fin, fout)
            with torch.no_grad():
                wq = bl.quantize_weight().numpy()
            w = bl.weight.detach().numpy().astype(np.float32)
            ref, gamma = _np_ternary(w)
            tern_diff.append(float(np.abs(wq - ref).max()))
            # stack recomputes gamma = mean|W| on every forward; compare torch
            # reduction against the numpy reference value
            gamma_torch = bl.weight.detach().float().abs().mean().item()
            gamma_diff.append(float(abs(gamma_torch - float(gamma))))
            # level set: exactly one nonzero magnitude g with values in
            # {-g, 0, +g}, and g equal to numpy gamma within 1e-7
            mags = np.unique(np.abs(wq))
            nonzero = mags[mags > 0]
            level_ok.append(
                len(nonzero) == 1 and abs(float(nonzero[0]) - float(gamma)) <= 1e-7
            )
            e_stack = float(np.linalg.norm(w - wq))
            e_np = float(np.linalg.norm(w - ref))
            err_rel.append(abs(e_stack - e_np) / max(e_np, 1e-12))
        m["ternary_max_diff"] = max(tern_diff)
        m["gamma_max_diff"] = max(gamma_diff)
        m["ternary_levels_ok"] = all(level_ok)
        m["ternary_err_rel"] = max(err_rel)

        # ---- 2/3/4-bit (G3 + G4 monotonicity) ----
        gint_diff, group_levels_ok, errs = [], [], {}
        for bits in (2, 3, 4):
            torch.manual_seed(202)  # same seed -> identical masters across bits
            gl = GroupIntLinear(256, 512, bits=bits, group_size=128)
            with torch.no_grad():
                wq = gl.quantize_weight().numpy()
            w = gl.weight.detach().numpy().astype(np.float32).reshape(
                gl.out_features, gl.n_groups, gl.group_size
            )
            s = gl.scale.detach().numpy().astype(np.float32)  # layer's own scales
            ref, codes = _np_gint(w, s, bits, gl.weight.numel())
            gint_diff.append(float(np.abs(wq - ref.reshape(wq.shape)).max()))
            per_group = codes.reshape(-1, gl.group_size)
            n_lv = max(len(np.unique(row)) for row in per_group)
            group_levels_ok.append(n_lv <= 2 ** bits)
            errs[bits] = float(np.linalg.norm(w.reshape(wq.shape) - ref.reshape(wq.shape)))
        m["gint_max_diff"] = max(gint_diff)
        m["group_levels_ok"] = all(group_levels_ok)
        m["err_2b"], m["err_3b"], m["err_4b"] = errs[2], errs[3], errs[4]

        # ---- activations (G5) ----
        aq = ActQuant(bits=8)
        torch.manual_seed(303)
        x = torch.randn(32, 128)
        x[::4] *= 10.0  # mixed-magnitude tokens
        with torch.no_grad():
            xq = aq(x).numpy().astype(np.float32)
        xn = x.numpy().astype(np.float32)
        scale = np.float32(127.0) / np.maximum(
            np.abs(xn).max(axis=-1, keepdims=True), np.float32(1e-5)
        )
        codes = xq * scale
        m["act_codes_int"] = float(np.abs(codes - np.rint(codes)).max())
        m["act_codes_in_range"] = bool(
            (np.rint(codes) >= -128).all() and (np.rint(codes) <= 127).all()
        )
        bound = 0.5 / scale
        m["act_err_excess"] = float((np.abs(xn - xq) - bound * (1 + 1e-5)).max())
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)
        return [
            GateOutcome(
                "G1", specs["G1"], m["ternary_max_diff"] <= 1e-7,
                detail=f"max |wq - ref| = {m['ternary_max_diff']:.3e}",
            ),
            GateOutcome(
                "G2", specs["G2"],
                m["ternary_levels_ok"] and m["gamma_max_diff"] <= 1e-7,
                detail=(f"levels_ok={m['ternary_levels_ok']} "
                        f"|gamma_torch - gamma_np| = {m['gamma_max_diff']:.3e}"),
            ),
            GateOutcome(
                "G3", specs["G3"], m["gint_max_diff"] <= 1e-7 and m["group_levels_ok"],
                detail=(f"max |wq - ref| = {m['gint_max_diff']:.3e} "
                        f"levels_ok={m['group_levels_ok']}"),
            ),
            GateOutcome(
                "G4", specs["G4"],
                m["ternary_err_rel"] <= 1e-5
                and m["err_2b"] > m["err_3b"] > m["err_4b"],
                detail=(f"ternary err rel={m['ternary_err_rel']:.3e}; "
                        f"||err|| 2b={m['err_2b']:.4f} 3b={m['err_3b']:.4f} "
                        f"4b={m['err_4b']:.4f}"),
            ),
            GateOutcome(
                "G5", specs["G5"],
                m["act_codes_int"] <= 1e-3 and m["act_codes_in_range"]
                and m["act_err_excess"] <= 1e-9,
                detail=(f"code int dev={m['act_codes_int']:.3e} "
                        f"in_range={m['act_codes_in_range']} "
                        f"bound excess={m['act_err_excess']:.3e}"),
            ),
        ]
