"""Straight-through estimator primitives.

All fake-quant in LOGOS is expressed as `x + (q(x) - x).detach()` so the
backward pass is exactly identity on the clipped region. The unit tests
(tests/test_quant.py) assert this gradient structure directly; the validation
panel's ste_probe re-verifies it against finite differences.
"""

from __future__ import annotations

import torch


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """round() with identity gradient."""
    return x + (torch.round(x) - x).detach()


def ste_round_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """RoundClip with STE. Gradient is identity inside [lo, hi] and zero
    outside (clip participates in autograd, round does not)."""
    y = torch.clamp(x, lo, hi)
    return y + (torch.round(y) - y).detach()
