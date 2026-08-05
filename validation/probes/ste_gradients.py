"""ste_gradients: autograd gradients of the STE primitives and the two
fake-quant linears vs hand-derived chain-rule expectations.

Every expectation below is derived on paper (in the comments) from the STE
definition y = x + (q(x) - x).detach(); nothing is computed with the code
under test.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from validation.base import GateOutcome, Probe

# G3 fixed 4x3 BitLinear example. Values chosen so u = w/gamma stays clear of
# the rounding boundaries {+-0.5, +-1} by > 0.03 and w has no zeros.
W43 = [
    [0.30, -0.12, 0.45],
    [-0.60, 0.05, -0.28],
    [0.18, 0.40, -0.07],
    [0.22, -0.35, 0.09],
]
X23 = [[0.9, -1.4, 0.33], [0.05, 0.71, -0.2]]

# G4 1-group GroupIntLinear toy: 3 out channels, in_features = group_size = 8,
# bits=3 (levels [-4, 3]). Rows mix in-range and clipped codes; every v = w/s
# stays > 0.03 away from the half-integer rounding boundaries so fp32
# (autograd) and fp64 (hand) round identically.
WG = [
    [0.20, -0.85, 1.30, -2.10, 0.55, 2.40, -0.15, 1.05],
    [-1.60, 0.30, 2.90, -0.70, 1.25, -2.50, 0.10, 0.90],
    [0.44, -0.20, 0.80, 1.60, -1.10, 0.35, -0.60, 2.20],
]
SG = [0.5, 0.8, 0.3]


class SteGradientsProbe(Probe):
    name = "ste_gradients"
    description = (
        "STE backward = identity on the clip region, zero outside; BitLinear "
        "and GroupIntLinear gradients vs hand chain-rule derivations"
    )
    gate_specs = [
        ("G1", "d ste_round(x)/dx == 1 exactly (autograd equals ones, incl. "
               "half-integer points)"),
        ("G2", "ste_round_clip gradient equals the [lo, hi] clip-region "
               "indicator exactly (boundary inclusive), zero outside"),
        ("G3", "BitLinear master-weight autograd gradient on L = sum(BitLinear(x)) "
               "matches the hand chain rule c_l*m_kl + sign(w_kl)/n * "
               "sum_ij c_j (R_ij - m_ij u_ij) on a fixed 4x3 example (rtol 1e-4)"),
        ("G4", "GroupIntLinear scale autograd gradient equals the hand LSQ "
               "formula g * sum_j (q_ij - ind_ij v_ij) with grad-rescale "
               "g = 1/sqrt(numel*qmax) on a 1-group toy (rtol 1e-4)"),
    ]

    def collect(self) -> dict:
        from logos.quant.bitlinear import BitLinear
        from logos.quant.paretoq import GroupIntLinear
        from logos.quant.ste import ste_round, ste_round_clip

        m: dict = {}

        # ---- G1: ste_round grad == ones, including exact half-integers ----
        x = torch.tensor([-2.5, -1.5, -0.7, -0.5, 0.0, 0.3, 0.5, 1.5, 2.4],
                         requires_grad=True)
        ste_round(x).sum().backward()
        m["g1_exact"] = bool(torch.equal(x.grad, torch.ones_like(x)))

        # ---- G2: ste_round_clip grad == indicator(lo <= x <= hi) ----
        lo, hi = -2.0, 1.0
        x = torch.tensor([-3.0, -2.0, -1.2, -0.5, 0.0, 0.7, 1.0, 1.5, 3.0],
                         requires_grad=True)
        ste_round_clip(x, lo, hi).sum().backward()
        ind = ((x.detach() >= lo) & (x.detach() <= hi)).float()
        m["g2_exact"] = bool(torch.equal(x.grad, ind))

        # ---- G3: BitLinear end-to-end weight gradient, hand chain rule ----
        # Forward: gamma = mean|w| (n=12 entries, autograd flows through |.|);
        # u = w/gamma; wq = f(u)*gamma with f = RoundClip under STE, so
        # forward f(u) = R = round(clamp(u,-1,1)), backward df/du = m = 1{|u|<=1}.
        # L = sum_b sum_i (xq_b . wq_i) = sum_ij wq_ij c_j with c_j = sum_b xq_bj
        # (xq is the int8-fake-quantized activation, a constant wrt w).
        #   dL/dwq_ij     = c_j
        #   dgamma/dw_kl  = sign(w_kl)/n
        #   du_ij/dw_kl   = delta_ik delta_jl / gamma - w_ij sign(w_kl)/(n gamma^2)
        #   dwq_ij/dw_kl  = gamma m_ij du_ij/dw_kl + R_ij dgamma/dw_kl
        # =>  dL/dw_kl = c_l m_kl + sign(w_kl)/n * sum_ij c_j (R_ij - m_ij u_ij)
        bl = BitLinear(3, 4)
        with torch.no_grad():
            bl.weight.copy_(torch.tensor(W43))
        xt = torch.tensor(X23)
        loss = bl(xt).sum()
        loss.backward()
        auto = bl.weight.grad.numpy().astype(np.float64)

        w = np.array(W43, dtype=np.float64)
        n = w.size
        gamma = np.abs(w).mean()
        u = w / gamma
        R = np.clip(np.rint(u), -1.0, 1.0)
        mask = (np.abs(u) <= 1.0).astype(np.float64)
        # xq re-derived from the activation equation (per-token absmax int8)
        xn = np.array(X23, dtype=np.float64)
        s = 127.0 / np.maximum(np.abs(xn).max(axis=-1, keepdims=True), 1e-5)
        xq = np.clip(np.rint(xn * s), -128, 127) / s
        c = xq.sum(axis=0)  # [3]
        t = float((c[None, :] * (R - mask * u)).sum())
        hand = c[None, :] * mask + np.sign(w) / n * t
        m["g3_max_rel"] = float(
            np.abs(auto - hand).max() / max(np.abs(hand).max(), 1e-12)
        )

        # ---- G4: GroupIntLinear scale gradient, hand LSQ formula ----
        # s_eff = s*g + (s - s*g).detach(): value s, ds_eff/ds = g (grad rescale),
        # g = 1/sqrt(numel*qmax). v = w/s; q = clip(round(v), qmin, qmax) with
        # round under STE and clamp in autograd (indicator ind = 1{qmin <= round(v)
        # <= qmax}, boundary inclusive). wq = q*s_eff. For L = sum(wq):
        #   dwq_ij/ds_i = g * (q_ij + s * ind_ij * d(w_ij/s)/ds)
        #              = g * (q_ij - ind_ij * v_ij)
        # => dL/ds_i = g * sum_j (q_ij - ind_ij v_ij)  -- the LSQ gradient:
        # (round(v)-v) in range, qmin/qmax when clipped.
        gl = GroupIntLinear(8, 3, bits=3, group_size=8)
        with torch.no_grad():
            gl.weight.copy_(torch.tensor(WG))
            gl.scale.copy_(torch.tensor(SG).view(3, 1))
        gl.quantize_weight().sum().backward()
        auto_s = gl.scale.grad.numpy().astype(np.float64).ravel()

        wg = np.array(WG, dtype=np.float64)
        sg = np.array(SG, dtype=np.float64)[:, None]
        qmin, qmax = -4, 3
        g = 1.0 / math.sqrt(wg.size * qmax)
        v = wg / sg
        r = np.rint(v)
        q = np.clip(r, qmin, qmax)
        ind = ((r >= qmin) & (r <= qmax)).astype(np.float64)
        hand_s = g * (q - ind * v).sum(axis=1)
        m["g4_max_rel"] = float(
            np.abs(auto_s - hand_s).max() / max(np.abs(hand_s).max(), 1e-12)
        )
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)
        return [
            GateOutcome("G1", specs["G1"], m["g1_exact"],
                        detail=f"grad == ones exactly: {m['g1_exact']}"),
            GateOutcome("G2", specs["G2"], m["g2_exact"],
                        detail=f"grad == clip indicator exactly: {m['g2_exact']}"),
            GateOutcome("G3", specs["G3"], m["g3_max_rel"] <= 1e-4,
                        detail=f"max rel dev vs hand chain rule = {m['g3_max_rel']:.3e}"),
            GateOutcome("G4", specs["G4"], m["g4_max_rel"] <= 1e-4,
                        detail=f"max rel dev vs hand LSQ formula = {m['g4_max_rel']:.3e}"),
        ]
