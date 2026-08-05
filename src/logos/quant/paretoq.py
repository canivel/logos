"""2/3/4-bit per-group integer weight fake-quant, after the ParetoQ ladder
(Liu et al. 2025).

Scheme: symmetric per-group integer quantization with a learnable scale per
group (LSQ-style gradient with the standard 1/sqrt(numel * qmax) gradient
rescale), group size along the input dimension. ParetoQ's finding that <=2-bit
sits in a "compensation" regime and >=3-bit in a "reconstruction" regime is a
*result we test* (RQ2), not something the quantizer assumes; the same code
path serves 2, 3 and 4 bits with only the level count changing.

Levels: k-bit -> q in [-(2^(k-1)) .. 2^(k-1)-1]  (4b: [-8,7], 3b: [-4,3],
2b: [-2,1]). Activations: per-token int8 (WxA8), same as the ternary arm.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from logos.quant.activations import ActQuant
from logos.quant.ste import ste_round


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale + (x - x * scale).detach()


class GroupIntLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        bias: bool = False,
        group_size: int = 128,
        act_bits: int = 8,
    ):
        super().__init__(in_features, out_features, bias=bias)
        assert bits in (2, 3, 4)
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.group_size = min(group_size, in_features)
        assert in_features % self.group_size == 0, "in_features must divide by group_size"
        self.n_groups = in_features // self.group_size
        self.act_quant = ActQuant(bits=act_bits)
        # Learnable per-(out_channel, group) scale, LSQ init from weight stats.
        with torch.no_grad():
            wg = self.weight.detach().float().view(out_features, self.n_groups, self.group_size)
            init = 2.0 * wg.abs().mean(dim=-1) / math.sqrt(self.qmax)
        self.scale = nn.Parameter(init.clamp_min(1e-8))

    def quantize_weight(self) -> torch.Tensor:
        w = self.weight.float().view(self.out_features, self.n_groups, self.group_size)
        g = 1.0 / math.sqrt(self.weight.numel() * self.qmax)
        s = _grad_scale(self.scale.float().clamp_min(1e-8), g).unsqueeze(-1)
        q = torch.clamp(ste_round(w / s), self.qmin, self.qmax)
        return (q * s).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self.quantize_weight().to(x.dtype)
        return F.linear(self.act_quant(x), wq, self.bias)

    @torch.no_grad()
    def packed_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(int8 codes in [qmin,qmax] shaped [out, groups, gs], fp32 scales
        [out, groups]) for export/parity.

        Codes are derived with bit-identical arithmetic to quantize_weight():
        the LSQ grad-rescale round-trip (s*g + (s - s*g)) can differ from s
        by 1 ulp in fp32, and a master within an ulp of a rounding boundary
        would otherwise export a different code than the forward pass uses."""
        w = self.weight.float().view(self.out_features, self.n_groups, self.group_size)
        g = 1.0 / math.sqrt(self.weight.numel() * self.qmax)
        s0 = self.scale.float().clamp_min(1e-8)
        s = (s0 * g + (s0 - s0 * g)).unsqueeze(-1)  # forward-path fp32 value
        codes = torch.clamp(torch.round(w / s), self.qmin, self.qmax).to(torch.int8)
        return codes, s.squeeze(-1)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, bits={self.bits}, group={self.group_size}"
