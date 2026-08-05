"""Logit-parity gate between two models (PLAN.md s5: "logit parity within
tolerance on 64 prompts" is a P0 exit criterion).

Canonical use: logit_parity(trained, load_artifact(export_artifact(trained))).
Exact equality holds up to float associativity because the reloaded masters
re-quantize to identical codes and scales (see export/artifact.py); in fp32
on CPU the observed max_abs_diff is ~1e-6, gated at 1e-2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def synthetic_prompts(
    vocab_size: int, n_prompts: int = 64, seq_len: int = 128, seed: int = 0
) -> torch.Tensor:
    """Deterministic synthetic token windows for the parity gate."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n_prompts, seq_len), generator=g)


@torch.no_grad()
def logit_parity(
    model_a: nn.Module,
    model_b: nn.Module,
    prompts_tokens: torch.Tensor,
    device: str = "cpu",
    atol: float = 1e-2,
    rtol: float = 0.0,
    batch_size: int = 8,
) -> dict:
    """Compare logits of two models on [N, T] prompt tokens.

    Returns dict(max_abs_diff, mean_abs_diff, kl_div_max, pass) where pass
    means |a - b| <= atol + rtol * |a| everywhere and kl_div_max is the max
    over positions of KL(softmax(a) || softmax(b)).
    """
    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()
    max_abs = 0.0
    abs_sum = 0.0
    n_elem = 0
    kl_max = 0.0
    ok = True
    for i in range(0, prompts_tokens.size(0), batch_size):
        x = prompts_tokens[i : i + batch_size].to(device)
        out_a, out_b = model_a(x), model_b(x)
        la = (out_a[0] if isinstance(out_a, tuple) else out_a).float()
        lb = (out_b[0] if isinstance(out_b, tuple) else out_b).float()
        diff = (la - lb).abs()
        max_abs = max(max_abs, diff.max().item())
        abs_sum += diff.sum().item()
        n_elem += diff.numel()
        ok = ok and bool((diff <= atol + rtol * la.abs()).all().item())
        p = F.softmax(la, dim=-1)
        kl = (p * (F.log_softmax(la, dim=-1) - F.log_softmax(lb, dim=-1))).sum(-1)
        kl_max = max(kl_max, kl.max().item())
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": abs_sum / max(n_elem, 1),
        "kl_div_max": kl_max,
        "pass": ok,
    }
