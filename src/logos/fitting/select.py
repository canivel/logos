"""Model selection + uncertainty (PLAN.md section 8): leave-one-size-out
extrapolation error (never in-sample fit — principle 7), stratified bootstrap
CIs, and the RQ4 extrapolation-acceptance checks (PLAN.md section 9)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from logos.config import LADDER, make_model
from logos.fitting.fit import FitResult, fit_form
from logos.fitting.forms import Form


def _sizes_by_n(df: pd.DataFrame) -> list[str]:
    """Ladder sizes present, ordered by n_nonemb ascending."""
    return list(df.groupby("size")["n_nonemb"].mean().sort_values().index)


def _abs_log_err(fit: FitResult, held: pd.DataFrame) -> float:
    pred = fit.predict(held["n_nonemb"], held["tokens"], held["bits"])
    return float(np.mean(np.abs(np.log(pred) - np.log(held["bpb"].to_numpy(dtype=float)))))


@dataclass
class LOSOResult:
    """Held-out extrapolation errors, mean |log L_pred - log L_obs|."""

    per_size: dict[str, float]
    mean_error: float
    upward_error: float  # fit on sizes below the largest, predict the largest (RQ4 protocol)


def loso(
    form: Form,
    df: pd.DataFrame,
    *,
    n_starts: int = 64,
    seed: int = 0,
    huber_delta: float = 1e-3,
    top_k: int = 6,
) -> LOSOResult:
    """Leave-one-size-out: for each ladder size present, fit on the rest and
    score on the held-out size. The largest-size fold doubles as the
    UPWARD-only variant (train strictly below, predict above)."""
    sizes = _sizes_by_n(df)
    per_size: dict[str, float] = {}
    for i, s in enumerate(sizes):
        train, held = df[df["size"] != s], df[df["size"] == s]
        fit = fit_form(form, train, n_starts=n_starts, seed=seed + i,
                       huber_delta=huber_delta, top_k=top_k)
        per_size[s] = _abs_log_err(fit, held)
    return LOSOResult(
        per_size=per_size,
        mean_error=float(np.mean(list(per_size.values()))),
        upward_error=per_size[sizes[-1]],
    )


@dataclass
class SelectionResult:
    best: str  # form name with the lowest mean LOSO error
    loso: dict[str, LOSOResult]
    fits: dict[str, FitResult]  # full-data fits, for downstream use


def select_by_loso(
    forms: dict[str, Form],
    df: pd.DataFrame,
    *,
    n_starts: int = 64,
    full_n_starts: int = 256,
    seed: int = 0,
    huber_delta: float = 1e-3,
    top_k: int = 6,
) -> SelectionResult:
    """Fit every candidate form; pick by LOSO error, NEVER in-sample
    (PLAN.md principle 7)."""
    results = {
        name: loso(f, df, n_starts=n_starts, seed=seed, huber_delta=huber_delta, top_k=top_k)
        for name, f in forms.items()
    }
    fits = {
        name: fit_form(f, df, n_starts=full_n_starts, seed=seed, huber_delta=huber_delta)
        for name, f in forms.items()
    }
    best = min(results, key=lambda k: results[k].mean_error)
    return SelectionResult(best=best, loso=results, fits=fits)


@dataclass
class BootstrapResult:
    point: dict[str, float]  # full-data point estimates (params + derived)
    ci: dict[str, tuple[float, float]]  # percentile CIs
    samples: dict[str, np.ndarray] = field(repr=False)
    n_boot: int = 0


def bootstrap_ci(
    form: Form,
    df: pd.DataFrame,
    *,
    n_boot: int = 100,
    seed: int = 0,
    n_starts: int = 16,
    top_k: int = 4,
    ci_level: float = 0.95,
    derived: dict[str, Callable[[dict[str, float]], float]] | None = None,
    base_fit: FitResult | None = None,
    huber_delta: float = 1e-3,
) -> BootstrapResult:
    """Resample runs with replacement, stratified by (size, precision) so every
    grid cell keeps its weight; refit each replicate (warm-started from the
    full-data fit); percentile CIs on every param and on derived quantities
    via callables params -> float."""
    if base_fit is None:
        base_fit = fit_form(form, df, seed=seed, huber_delta=huber_delta)
    derived = derived or {}
    rng = np.random.default_rng(seed)
    strata = [idx.to_numpy() for _, idx in df.groupby(["size", "precision"]).groups.items()]

    names = list(base_fit.params) + list(derived)
    samples: dict[str, list[float]] = {k: [] for k in names}
    for b in range(n_boot):
        take = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])
        rep = df.loc[take]
        fit = fit_form(
            form, rep, n_starts=n_starts, seed=seed + 1 + b, top_k=top_k,
            extra_starts=[base_fit.params], huber_delta=huber_delta,
        )
        for k, v in fit.params.items():
            samples[k].append(v)
        for k, fn in derived.items():
            samples[k].append(float(fn(fit.params)))

    lo_q, hi_q = 100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2
    arr = {k: np.array(v) for k, v in samples.items()}
    ci = {k: (float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))) for k, v in arr.items()}
    point = dict(base_fit.params)
    point.update({k: float(fn(base_fit.params)) for k, fn in derived.items()})
    return BootstrapResult(point=point, ci=ci, samples=arr, n_boot=n_boot)


# ---- RQ4: predict-then-check (PLAN.md section 9) ----


def seed_sigma_by_size(df: pd.DataFrame) -> dict[str, float]:
    """Seed sigma per ladder size: std of bpb across seeds within each
    (size, precision, tokens) cell, averaged per size (RQ1 sigma table).
    Sizes with a single seed get no entry — extrapolate sigma upward
    (PLAN.md principle 4)."""
    cell = df.groupby(["size", "precision", "tokens"])["bpb"].std(ddof=1).dropna()
    return {s: float(v) for s, v in cell.groupby(level="size").mean().items()}


def _sigma_for_n(sigma_by_size: dict | float, N: np.ndarray) -> np.ndarray:
    """Resolve a sigma per row: scalar, or nearest-in-log-N lookup from a dict
    keyed by ladder size name or by n_nonemb."""
    N = np.asarray(N, dtype=float)
    if np.isscalar(sigma_by_size):
        return np.full(N.shape, float(sigma_by_size))
    keys, vals = [], []
    for k, v in sigma_by_size.items():
        keys.append(float(make_model(k).n_nonemb) if isinstance(k, str) and k in LADDER else float(k))
        vals.append(float(v))
    keys, vals = np.array(keys), np.array(vals)
    nearest = np.argmin(np.abs(np.log(N)[:, None] - np.log(keys)[None, :]), axis=1)
    return vals[nearest]


def predict_with_band(
    fit: FitResult, N, D, bits, sigma_by_size: dict | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pred, lo, hi) with the seed-sigma-implied 1-sigma band. The RQ4
    acceptance bar is 2x this band (check_extrapolation)."""
    N = np.atleast_1d(np.asarray(N, dtype=float))
    pred = np.atleast_1d(fit.predict(N, D, bits))
    sig = _sigma_for_n(sigma_by_size, N)
    return pred, pred - sig, pred + sig


@dataclass
class ExtrapolationVerdict:
    """Both RQ4 acceptance verdicts (PLAN.md s8/s9)."""

    within_band: bool  # every holdout cell within 2x the seed-sigma band
    ordering_correct: bool  # precision ORDERING at the holdout size correct (per D)
    max_abs_err: float  # worst |pred - obs| over holdout cells, BPB
    band: float  # the 2-sigma acceptance band, BPB


def check_extrapolation(
    fits: dict[str, FitResult] | FitResult,
    df_holdout: pd.DataFrame,
    sigma: dict | float,
) -> dict[str, ExtrapolationVerdict]:
    """RQ4 acceptance: predicted holdout loss within 2x the seed-sigma-implied
    band, and predicted precision ordering correct at each token budget."""
    if isinstance(fits, FitResult):
        fits = {fits.form_name: fits}
    out: dict[str, ExtrapolationVerdict] = {}
    for name, fit in fits.items():
        cells = (
            df_holdout.groupby(["tokens", "precision"])
            .agg(n_nonemb=("n_nonemb", "mean"), bits=("bits", "mean"), bpb=("bpb", "mean"))
            .reset_index()
        )
        pred = fit.predict(cells["n_nonemb"], cells["tokens"], cells["bits"])
        sig = _sigma_for_n(sigma, cells["n_nonemb"].to_numpy())
        err = np.abs(pred - cells["bpb"].to_numpy())
        within = bool(np.all(err <= 2 * sig))
        ordering = True
        cells = cells.assign(pred=pred)
        for _, grp in cells.groupby("tokens"):
            by_pred = grp.sort_values("pred")["precision"].tolist()
            by_obs = grp.sort_values("bpb")["precision"].tolist()
            ordering = ordering and (by_pred == by_obs)
        out[name] = ExtrapolationVerdict(
            within_band=within,
            ordering_correct=bool(ordering),
            max_abs_err=float(err.max()),
            band=float(2 * sig.max()),
        )
    return out
