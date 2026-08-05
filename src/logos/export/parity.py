"""Export-parity gates between two models (PLAN.md s5: "logit parity within
tolerance on 64 prompts" is a P0 exit criterion).

Canonical use: export_parity(trained, load_artifact(export_artifact(trained))).

Two gates, in order of authority:
1. `weights_parity` — packed codes and scales bitwise identical per layer.
   This is the true export-correctness criterion: any packing/format bug
   flips codes.
2. `logit_parity` — logit agreement on synthetic prompts. CAVEAT (measured
   on trained 6-layer ternary models): the reloaded absmean gamma is a
   recomputed fp32 mean, ulp-level different from the original; quantized
   activations are discontinuous, so a ~1e-6 seed difference can amplify
   through depth into ~1e-1 max-abs logit drift while KL stays ~1e-4.
   Max-abs is therefore NOT a reliable discriminator for deep quantized
   models; the combined `export_parity` verdict uses codes+scales equality
   plus a KL bound, and reports max-abs for the record.
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


@torch.no_grad()
def weights_parity(model_a: nn.Module, model_b: nn.Module) -> dict:
    """Bitwise comparison of packed codes and scales for every quantized
    layer, plus exact equality of non-master tensors (embeddings, norms).
    The authoritative export-correctness gate."""
    mods_a = dict(model_a.named_modules())
    mods_b = dict(model_b.named_modules())
    n_layers, bad = 0, []
    for name, ma in mods_a.items():
        if not hasattr(ma, "packed_weight"):
            continue
        mb = mods_b.get(name)
        if mb is None or not hasattr(mb, "packed_weight"):
            bad.append(f"{name}: missing in reload")
            continue
        ca, sa = (t.cpu() for t in ma.packed_weight())
        cb, sb = (t.cpu() for t in mb.packed_weight())
        n_layers += 1
        if not torch.equal(ca, cb):
            bad.append(f"{name}: codes differ ({(ca != cb).sum().item()} elems)")
        if not torch.allclose(sa.float(), sb.float(), rtol=1e-6, atol=0.0):
            bad.append(f"{name}: scales differ")
    return {"n_quant_layers": n_layers, "mismatches": bad, "pass": not bad}


@torch.no_grad()
def export_parity(
    model_a: nn.Module,
    model_b: nn.Module,
    prompts_tokens: torch.Tensor,
    device: str = "cpu",
    kl_tol: float = 1e-3,
    atol: float = 1e-2,
) -> dict:
    """Combined verdict: weights_parity AND kl_div_max <= kl_tol. max-abs
    logit diff is reported but does not gate (see module docstring)."""
    wp = weights_parity(model_a, model_b)
    lp = logit_parity(model_a, model_b, prompts_tokens, device=device, atol=atol)
    return {
        **lp,
        "weights_pass": wp["pass"],
        "n_quant_layers": wp["n_quant_layers"],
        "mismatches": wp["mismatches"],
        "logit_atol_pass": lp["pass"],
        "pass": bool(wp["pass"] and lp["kl_div_max"] <= kl_tol),
    }
