"""LR / weight-decay schedules, computed per-step from the token count.

Single-stage cosine with linear warmup is THE grid schedule: comparability
beats per-arm maximum performance (PLAN.md principle 6). The BitNet-style
two-stage recipe (mid-run LR drop + weight-decay switch-off) is gated behind
schedule="bitnet2stage" and reserved for the P5 capstone, where absolute
quality matters.
"""

from __future__ import annotations

import math

from logos.config import TrainConfig

# bitnet2stage shape (capstone-only; BitNet b1.58 recipe, simplified):
STAGE_SPLIT = 0.5  # stage boundary as a fraction of total tokens
STAGE1_FLOOR = 0.5  # stage 1 cosine decays lr -> STAGE1_FLOOR * lr
STAGE2_PEAK = 0.25  # abrupt drop at the boundary; stage 2 restarts here


def _warmup_cosine(peak: float, floor: float, warmup: int, total: int, tokens: int) -> float:
    """Linear warmup to `peak` over `warmup` tokens, then cosine to `floor`."""
    if warmup > 0 and tokens < warmup:
        return peak * tokens / warmup
    prog = min(1.0, max(0.0, (tokens - warmup) / max(1, total - warmup)))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * prog))


def lr_at(cfg: TrainConfig, tokens: int) -> float:
    if cfg.schedule == "cosine":
        return _warmup_cosine(
            cfg.lr, cfg.min_lr_ratio * cfg.lr, cfg.warmup, cfg.total_tokens, tokens
        )
    if cfg.schedule == "bitnet2stage":
        split = int(STAGE_SPLIT * cfg.total_tokens)
        if tokens <= split:
            return _warmup_cosine(cfg.lr, STAGE1_FLOOR * cfg.lr, cfg.warmup, split, tokens)
        prog = min(1.0, (tokens - split) / max(1, cfg.total_tokens - split))
        peak2, floor2 = STAGE2_PEAK * cfg.lr, cfg.min_lr_ratio * cfg.lr
        return floor2 + 0.5 * (peak2 - floor2) * (1.0 + math.cos(math.pi * prog))
    raise ValueError(f"unknown schedule: {cfg.schedule!r}")


def wd_at(cfg: TrainConfig, tokens: int) -> float:
    """Weight decay at a token count. bitnet2stage switches decay off in
    stage 2 (BitNet recipe); the grid schedule keeps it constant."""
    if cfg.schedule == "bitnet2stage" and tokens > int(STAGE_SPLIT * cfg.total_tokens):
        return 0.0
    return cfg.weight_decay
