"""8-bit activation fake-quant (the A8 in WxA8).

Per-token absmax int8, as in BitNet b1.58: for each token vector x,
s = Q / max|x|, x_q = clamp(round(x * s), -Q-1, Q) / s, with STE gradients.
Applied to the input of every quantized linear layer. bf16 arms bypass this
entirely (principle 1: precision is the only difference between arms).
"""

from __future__ import annotations

import torch
from torch import nn


class ActQuant(nn.Module):
    def __init__(self, bits: int = 8, eps: float = 1e-5):
        super().__init__()
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        scale = self.qmax / x32.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        q = torch.clamp(torch.round(x32 * scale), -self.qmax - 1, self.qmax) / scale
        return (x32 + (q - x32).detach()).to(dtype)

    def extra_repr(self) -> str:
        return f"bits={self.bits}"
