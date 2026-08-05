"""Chinchilla-standard fitting (PLAN.md section 8): Huber loss on residuals in
log-loss space, multi-start L-BFGS-B from a deterministic-seeded LHS of inits.
Deterministic given seed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from logos.config import Precision
from logos.fitting.forms import Form

#: Columns of the fitting-input frame (decoupled from the results store).
COLUMNS = ["run_id", "size", "n_nonemb", "tokens", "precision", "bits", "seed", "bpb"]

_BAD = 1e10  # objective value for non-finite / non-positive predictions


def records_from_jsonl(path: str | Path, target: str = "mean") -> pd.DataFrame:
    """Build the fitting frame from a results JSONL whose rows have keys
    {run_id, size, n_nonemb, tokens, precision, seed, bpb_val1, bpb_val2}.
    `target`: 'mean' (default) averages the two held-out val sets (PLAN.md
    principle 3), or 'val1' / 'val2' to fit on one of them."""
    picker = {
        "mean": lambda r: 0.5 * (float(r["bpb_val1"]) + float(r["bpb_val2"])),
        "val1": lambda r: float(r["bpb_val1"]),
        "val2": lambda r: float(r["bpb_val2"]),
    }[target]
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            prec = str(r["precision"])
            rows.append(
                dict(
                    run_id=str(r["run_id"]),
                    size=str(r["size"]),
                    n_nonemb=int(r["n_nonemb"]),
                    tokens=int(r["tokens"]),
                    precision=prec,
                    bits=Precision(prec).bits,
                    seed=int(r["seed"]),
                    bpb=picker(r),
                )
            )
    return pd.DataFrame(rows, columns=COLUMNS)


@dataclass
class FitResult:
    """Best-of-starts fit of one form on one frame."""

    form_name: str
    params: dict[str, float]
    train_loss: float
    n_runs: int
    converged_frac: float
    form: Form = field(repr=False, compare=False, default=None)

    def predict(self, N, D, bits) -> np.ndarray:
        return self.form.predict(self.params, N, D, bits)


def huber(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def _objective(
    theta: np.ndarray,
    form: Form,
    N: np.ndarray,
    D: np.ndarray,
    bits: np.ndarray,
    log_obs: np.ndarray,
    delta: float,
) -> float:
    params = form.to_natural(theta)
    pred = form.predict(params, N, D, bits)
    if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
        return _BAD
    val = float(np.mean(huber(np.log(pred) - log_obs, delta)))
    if form.has_penalty:
        val += form.penalty(params)
    return val if np.isfinite(val) else _BAD


def fit_form(
    form: Form,
    df: pd.DataFrame,
    *,
    n_starts: int = 256,
    seed: int = 0,
    huber_delta: float = 1e-3,
    coarse_iter: int = 40,
    top_k: int = 8,
    max_iter: int = 500,
    extra_starts: list[dict[str, float]] | None = None,
) -> FitResult:
    """Multi-start L-BFGS-B (PLAN.md s8). Huber (delta=1e-3 default) on
    log L_pred - log L_obs. Two-stage for speed: a coarse pass (`coarse_iter`
    L-BFGS-B iterations) over all starts, then the `top_k` best are polished
    to convergence. `extra_starts` prepends warm starts (natural-space param
    dicts, clipped into bounds) — used by bootstrap/LOSO refits — and warm
    starts always make the polish stage. Deterministic given seed."""
    N = df["n_nonemb"].to_numpy(dtype=float)
    D = df["tokens"].to_numpy(dtype=float)
    bits = df["bits"].to_numpy(dtype=float)
    log_obs = np.log(df["bpb"].to_numpy(dtype=float))

    rng = np.random.default_rng(seed)
    inits = form.sample_inits(rng, n_starts)
    bounds = form.internal_bounds()
    n_warm = 0
    if extra_starts:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        warm = np.array([np.clip(form.to_internal(p), lo, hi) for p in extra_starts])
        inits = np.vstack([warm, inits])
        n_warm = len(warm)

    args = (form, N, D, bits, log_obs, huber_delta)
    coarse = [
        minimize(_objective, x0, args=args, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": coarse_iter})
        for x0 in inits
    ]
    order = list(np.argsort([r.fun for r in coarse], kind="stable")[:top_k])
    order += [i for i in range(n_warm) if i not in order]  # warm starts always polish

    best = None
    n_conv = 0
    for i in order:
        res = minimize(
            _objective, coarse[i].x, args=args, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-13, "gtol": 1e-10},
        )
        if res.success:
            n_conv += 1
        if best is None or res.fun < best.fun:  # strict <: first-best wins, deterministic
            best = res
    return FitResult(
        form_name=form.name,
        params=form.to_natural(best.x),
        train_loss=float(best.fun),
        n_runs=len(df),
        converged_frac=n_conv / len(order),
        form=form,
    )
