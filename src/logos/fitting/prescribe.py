"""Cashing in the law (PLAN.md s8 memory transform + RQ3/RQ5): given the
fitted L(N, D, b), prescribe (N*, b*) under a weight-byte budget, the b*(M, D)
phase diagram, effective capacity per bit, and the P4 total-footprint
extension with the KV cache in the budget (PLAN.md section 10)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from logos.config import LADDER, make_model
from logos.fitting.fit import FitResult
from logos.fitting.forms import BF16_BITS

GB = float(1 << 30)
DEFAULT_BITS: tuple[float, ...] = (1.58, 2.0, 3.0, 4.0, 16.0)

#: bytes(N, bits) — theoretical packed size. PLAN.md s8: real deployments use
#: measured GGUF/lpack sizes, passed as `measured_bytes`, not this theory value.
WeightBytesFn = Callable[[float, float], float]


def default_weight_bytes(N: float, bits: float) -> float:
    """M_w = N * b_w / 8 (theory; swap in measured packed sizes when known)."""
    return N * bits / 8.0


def _max_n(budget: float, bytes_fn: Callable[[float], float], n_min: float, n_max: float) -> float:
    """Largest N with bytes_fn(N) <= budget, by bisection (bytes monotone in N)."""
    if bytes_fn(n_min) > budget:
        return 0.0
    if bytes_fn(n_max) <= budget:
        return n_max
    lo, hi = np.log(n_min), np.log(n_max)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bytes_fn(np.exp(mid)) <= budget:
            lo = mid
        else:
            hi = mid
    return float(np.exp(lo))


def _best_n(fit: FitResult, D: float, bits: float, n_lo: float, n_hi: float) -> tuple[float, float]:
    """1-D minimize of fitted L over N in [n_lo, n_hi] (log-N search)."""
    obj = lambda x: float(fit.predict(np.exp(x), D, bits))
    res = minimize_scalar(obj, bounds=(np.log(n_lo), np.log(n_hi)), method="bounded",
                          options={"xatol": 1e-6})
    # L is usually monotone decreasing in N; make sure the boundary is considered.
    cands = [(obj(np.log(n_hi)), n_hi), (float(res.fun), float(np.exp(res.x)))]
    L, n = min(cands)
    return n, L


def optimal_config(
    fit: FitResult,
    M_budget_bytes: float,
    D: float,
    bit_options: Sequence[float] = DEFAULT_BITS,
    *,
    measured_bytes: WeightBytesFn | None = None,
    n_min: float = 1e6,
    n_cap: float = 1e13,
) -> tuple[float, float, float]:
    """(N*, b*, L*) minimizing fitted L s.t. M_w(N, b) <= M, closed over N by
    1-D optimization per bit option (PLAN.md s8 optimal prescription)."""
    wbytes = measured_bytes or default_weight_bytes
    best: tuple[float, float, float] | None = None
    for b in bit_options:
        n_max = _max_n(M_budget_bytes, lambda n: wbytes(n, b), n_min, n_cap)
        if n_max < n_min:
            continue
        n_star, L = _best_n(fit, D, b, n_min, n_max)
        if best is None or L < best[2]:
            best = (n_star, b, L)
    if best is None:
        raise ValueError(f"budget {M_budget_bytes:.3g} B fits no model above N={n_min:.0g}")
    return best


def phase_diagram(
    fit: FitResult,
    M_grid: Sequence[float],
    D_grid: Sequence[float],
    bits: Sequence[float] = DEFAULT_BITS,
    *,
    measured_bytes: WeightBytesFn | None = None,
) -> np.ndarray:
    """b*(M, D) array of shape (len(M_grid), len(D_grid)) — the RQ3 phase
    diagram."""
    out = np.empty((len(M_grid), len(D_grid)))
    for i, M in enumerate(M_grid):
        for j, D in enumerate(D_grid):
            _, b_star, _ = optimal_config(fit, M, D, bits, measured_bytes=measured_bytes)
            out[i, j] = b_star
    return out


def effective_capacity_summary(
    fit: FitResult, bit_options: Sequence[float] = (1.58, 2.0, 3.0, 4.0, 16.0)
) -> dict[float, dict[str, float]]:
    """f(b) per bit width with distance from the ~2 bits/param knowledge-
    capacity bound (RQ3). Per width: f (effective params per physical param),
    f_per_byte = f * 16 / b (capacity per byte relative to bf16),
    knowledge_bits = 2 * f (a bf16 param stores ~2 bits), bound = min(b, 2)
    (a b-bit weight cannot store more than b raw bits), and
    fraction_of_bound = knowledge_bits / bound."""
    out: dict[float, dict[str, float]] = {}
    for b in bit_options:
        f = float(fit.form.f_of_b(fit.params, np.array([b]))[0])
        kb = 2.0 * f
        bound = min(b, 2.0)
        out[b] = dict(
            f=f,
            f_per_byte=f * BF16_BITS / b,
            knowledge_bits=kb,
            bound=bound,
            fraction_of_bound=kb / bound,
        )
    return out


# ---- P4 extension: weights + KV cache in one budget (PLAN.md section 10) ----


def kv_bytes_interpolator() -> Callable[[float, float, float], float]:
    """Smooth kv(N, context_len, kv_bits) built from the LADDER shapes:
    log-log linear interpolation (with edge extrapolation) of the per-token
    per-bit KV coefficient 2 * layers * kv_heads * head_dim against n_nonemb;
    exact ModelConfig.kv_bytes at the ladder points."""
    pts = sorted(
        (float(m.n_nonemb), float(m.kv_bytes(context_len=1, kv_bits=8.0)))
        for m in (make_model(s) for s in LADDER)
    )
    ln = np.array([np.log(n) for n, _ in pts])
    lc = np.array([np.log(c) for _, c in pts])

    def kv(N: float, context_len: float, kv_bits: float) -> float:
        x = np.log(float(N))
        if x <= ln[0]:
            coeff = lc[0] + (lc[1] - lc[0]) * (x - ln[0]) / (ln[1] - ln[0])
        elif x >= ln[-1]:
            coeff = lc[-1] + (lc[-1] - lc[-2]) * (x - ln[-1]) / (ln[-1] - ln[-2])
        else:
            coeff = float(np.interp(x, ln, lc))
        return float(np.exp(coeff)) * context_len * kv_bits / 8.0

    return kv


def total_footprint_optimal(
    fit: FitResult,
    M_budget: float,
    context_len: int,
    kv_bits_options: Sequence[float] = (16.0, 8.0, 4.0, 3.0, 2.0),
    size_key_fn: Callable[[float], str] | None = None,
    *,
    D: float,
    bit_options: Sequence[float] = DEFAULT_BITS,
    measured_bytes: WeightBytesFn | None = None,
    n_min: float = 1e6,
    n_cap: float = 1e13,
) -> tuple[float, float, float, float]:
    """(N*, b_w*, b_kv*, L*) minimizing fitted L s.t.
    footprint = M_w(N, b_w) + KV_bytes(N, context, b_kv) <= M_budget (RQ5).
    KV bytes come from config.make_model shapes: via `size_key_fn(N) -> ladder
    key` for exact shapes, else the smooth LADDER interpolator. The fitted L
    does not (yet) depend on b_kv — P4's native KV-QAT arms will add that —
    so today b_kv* is simply the cheapest cache that buys the most N."""
    wbytes = measured_bytes or default_weight_bytes
    kv_interp = kv_bytes_interpolator()

    def kv_term(N: float, b_kv: float) -> float:
        if size_key_fn is not None:
            return float(make_model(size_key_fn(N)).kv_bytes(context_len, b_kv))
        return kv_interp(N, context_len, b_kv)

    best: tuple[float, float, float, float] | None = None
    for b_w in bit_options:
        for b_kv in kv_bits_options:
            footprint = lambda n: wbytes(n, b_w) + kv_term(n, b_kv)
            n_max = _max_n(M_budget, footprint, n_min, n_cap)
            if n_max < n_min:
                continue
            n_star, L = _best_n(fit, D, b_w, n_min, n_max)
            if best is None or L < best[3]:
                best = (n_star, b_w, b_kv, L)
    if best is None:
        raise ValueError(
            f"budget {M_budget:.3g} B at context {context_len} fits no model above N={n_min:.0g}"
        )
    return best
