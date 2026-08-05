"""BitLinear: 1.58-bit (ternary) weight quantization per the BitNet b1.58
2B4T report.

Weight path (absmean scaling):
    gamma = mean(|W|)
    W_q   = RoundClip(W / (gamma + eps), -1, 1) * gamma      # in {-g, 0, +g}
Activation path: per-token absmax int8 (ActQuant).
Master weights live in the module parameter (bf16 under autocast); the
quantization computation runs in fp32; gradients flow via STE.

The required sub-layer norm (subln) is NOT inside this module — the
architecture places RMSNorm before o_proj / down_proj inputs when the model
is built with a quantized precision (see model/transformer.py), keeping the
module a drop-in replacement for nn.Linear.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from logos.quant.activations import ActQuant
from logos.quant.ste import ste_round_clip


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, act_bits: int = 8):
        super().__init__(in_features, out_features, bias=bias)
        self.act_quant = ActQuant(bits=act_bits)

    def quantize_weight(self) -> torch.Tensor:
        w = self.weight.float()
        gamma = w.abs().mean().clamp_min(1e-8)
        return ste_round_clip(w / gamma, -1.0, 1.0) * gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self.quantize_weight().to(x.dtype)
        return F.linear(self.act_quant(x), wq, self.bias)

    @torch.no_grad()
    def packed_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(int8 ternary codes in {-1,0,1}, scalar gamma) for export/parity."""
        w = self.weight.float()
        gamma = w.abs().mean().clamp_min(1e-8)
        codes = torch.clamp(torch.round(w / gamma), -1, 1).to(torch.int8)
        return codes, gamma
