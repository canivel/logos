"""Tests for the scaling-law fitting subsystem (PLAN.md sections 8-10):
synthetic recovery, LOSO form selection, bootstrap CI coverage, prescription
sanity (the crossover), and determinism."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from logos.config import ALL_PRECISIONS, Precision, make_model
from logos.fitting.fit import FitResult, fit_form, records_from_jsonl
from logos.fitting.forms import FormAFree, FormB, FormC, fmt_bits
from logos.fitting.prescribe import GB, optimal_config, phase_diagram
from logos.fitting.select import (
    bootstrap_ci,
    check_extrapolation,
    seed_sigma_by_size,
    select_by_loso,
)

# Ground truth: Form A (effective capacity) with realistic params (PLAN.md s8).
TRUE = dict(E=0.6, A=300.0, alpha=0.32, B=250.0, beta=0.28)
TRUE_F = {1.58: 0.55, 2.0: 0.62, 3.0: 0.78, 4.0: 0.88}
TRUE_ALL = {**TRUE, **{f"f_{fmt_bits(b)}": v for b, v in TRUE_F.items()}}

# The P2 run matrix (PLAN.md s7): size -> (tokens/param tiers, seeds).
P2_GRID = {
    "25m": ((20, 80, 320), 3),
    "60m": ((20, 80, 320), 3),
    "125m": ((20, 80, 320), 1),
    "250m": ((20, 80, 320), 1),
    "490m": ((20, 80), 1),
}
P2_EXT = ("1.58", "4", "bf16")  # 490m extended tier at 320x


def make_grid(seed: int = 0, sigma: float = 0.005) -> pd.DataFrame:
    """Synthetic P2 grid from the ground-truth Form A + lognormal noise."""
    form = FormAFree()
    rng = np.random.default_rng(seed)
    rows = []

    def add(size: str, tpp: int, prec: Precision, s: int) -> None:
        n = make_model(size).n_nonemb
        D = int(tpp * n)
        L = float(form.predict(TRUE_ALL, n, D, prec.bits))
        L *= float(np.exp(rng.normal(0.0, sigma)))
        rows.append(
            dict(run_id=f"p2-{size}-{prec.value}-{tpp}x-s{s}", size=size, n_nonemb=n,
                 tokens=D, precision=prec.value, bits=prec.bits, seed=s, bpb=L)
        )

    for size, (tpps, n_seeds) in P2_GRID.items():
        for tpp in tpps:
            for prec in ALL_PRECISIONS:
                for s in range(n_seeds):
                    add(size, tpp, prec, s)
    for pv in P2_EXT:
        add("490m", 320, Precision(pv), 0)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return make_grid(seed=0)


@pytest.fixture(scope="module")
def fit_a(df: pd.DataFrame) -> FitResult:
    return fit_form(FormAFree(), df, n_starts=128, seed=0)


def test_synthetic_recovery(fit_a: FitResult) -> None:
    """Fitted params recover alpha/beta within 15% rel., f(b) within 10%."""
    p = fit_a.params
    assert fit_a.converged_frac > 0.5
    assert abs(p["alpha"] - TRUE["alpha"]) / TRUE["alpha"] < 0.15
    assert abs(p["beta"] - TRUE["beta"]) / TRUE["beta"] < 0.15
    for b, f_true in TRUE_F.items():
        assert abs(p[f"f_{fmt_bits(b)}"] - f_true) / f_true < 0.10, f"f({b})"


def test_loso_prefers_true_form(df: pd.DataFrame) -> None:
    """Data from Form A: selection by LOSO extrapolation (never in-sample,
    PLAN.md principle 7) must pick A over B."""
    sel = select_by_loso(
        {"A-free": FormAFree(), "B": FormB()}, df,
        n_starts=32, full_n_starts=64, seed=7, top_k=4,
    )
    assert sel.best == "A-free"
    assert sel.loso["A-free"].mean_error < sel.loso["B"].mean_error
    # the upward-only (RQ4) variant agrees
    assert sel.loso["A-free"].upward_error < sel.loso["B"].upward_error


def test_bootstrap_ci_contains_truth(df: pd.DataFrame, fit_a: FitResult) -> None:
    """95% stratified-bootstrap CIs contain the generating truth for >=80%
    of params."""
    boot = bootstrap_ci(
        FormAFree(), df, n_boot=50, seed=11, n_starts=8, top_k=3, base_fit=fit_a,
        derived={"f_per_byte_1.58": lambda p: p["f_1.58"] * 16 / 1.58},
    )
    hits = sum(lo <= TRUE_ALL[k] <= hi for k, (lo, hi) in boot.ci.items() if k in TRUE_ALL)
    n = len(TRUE_ALL)
    assert hits / n >= 0.8, {k: boot.ci[k] for k in TRUE_ALL}
    d_lo, d_hi = boot.ci["f_per_byte_1.58"]
    assert d_lo <= TRUE_F[1.58] * 16 / 1.58 <= d_hi


def test_prescription_crossover() -> None:
    """With f(b) making low-bit capacity-efficient per byte
    (f(1.58)*16/1.58 > 1) plus an Ouyang-style degradation term, a tight
    budget prescribes a lower bit width than a loose one, and the crossover
    exists in between (the core RQ3 claim)."""
    form = FormC()
    params = dict(E=0.6, A=400.0, alpha=0.30, B=250.0, beta=0.28,
                  b0=0.0, gamma_f=2.0, delta=0.30, eta=0.08)
    params.update({"g_1.58": 1.3e-3, "g_2": 1.0e-3, "g_3": 6e-4, "g_4": 4e-4})
    fit = FitResult("C", params, 0.0, 0, 1.0, form)
    f158 = float(form.f_of_b(params, np.array([1.58]))[0])
    assert f158 * 16 / 1.58 > 1.0  # low-bit is capacity-efficient per byte

    D = 1e11
    _, b_tight, _ = optimal_config(fit, 0.25 * GB, D)
    _, b_loose, _ = optimal_config(fit, 32 * GB, D)
    assert b_tight < b_loose
    # the crossover exists: scanning budgets flips the prescription
    bstar = phase_diagram(fit, np.geomspace(0.25 * GB, 32 * GB, 12), [D])[:, 0]
    assert bstar[0] == b_tight and bstar[-1] == b_loose
    assert len(set(bstar)) >= 2


def test_extrapolation_check(df: pd.DataFrame) -> None:
    """RQ4 protocol: fit on sizes below the largest, predict the largest;
    with the true form both verdicts pass."""
    below = df[df["size"] != "490m"]
    held = df[df["size"] == "490m"]
    fit = fit_form(FormAFree(), below, n_starts=64, seed=3)
    sigma = max(seed_sigma_by_size(df).values())
    verdicts = check_extrapolation(fit, held, sigma)
    v = verdicts["A-free"]
    assert v.within_band
    assert v.ordering_correct


def test_determinism(df: pd.DataFrame) -> None:
    """Same seed -> identical FitResult."""
    a = fit_form(FormAFree(), df, n_starts=32, seed=42)
    b = fit_form(FormAFree(), df, n_starts=32, seed=42)
    assert a.params == b.params
    assert a.train_loss == b.train_loss
    assert a.converged_frac == b.converged_frac


def test_records_from_jsonl(tmp_path) -> None:
    rows = [
        dict(run_id="r0", size="25m", n_nonemb=25_000_000, tokens=500_000_000,
             precision="1.58", seed=0, bpb_val1=1.20, bpb_val2=1.30),
        dict(run_id="r1", size="60m", n_nonemb=60_000_000, tokens=1_200_000_000,
             precision="bf16", seed=1, bpb_val1=1.00, bpb_val2=1.10),
    ]
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = records_from_jsonl(p)
    assert list(out["bpb"]) == [1.25, 1.05]
    assert list(out["bits"]) == [1.58, 16.0]
    out1 = records_from_jsonl(p, target="val1")
    assert list(out1["bpb"]) == [1.20, 1.00]
