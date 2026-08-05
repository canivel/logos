"""Probe: the fitting pipeline recovers a hidden ground-truth law (PLAN.md s8).

Synthetic grids are generated from Form A's equation HAND-CODED here (never
`logos.fitting.forms`), with truth parameters drawn from plausible ranges the
fitter never sees, lognormal noise sigma=0.004, over the P2 ladder sizes and
token multiples. Checks: parameter recovery, LOSO form selection, upward
extrapolation, the b*(M, D) prescription, and seed determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from logos.config import make_model
from logos.fitting.fit import fit_form
from logos.fitting.forms import FormAFree, FormB
from logos.fitting.prescribe import phase_diagram
from logos.fitting.select import loso
from validation.base import GateOutcome, Probe

QBITS = (1.58, 2.0, 3.0, 4.0)
BITS5 = QBITS + (16.0,)
SIZES = ["25m", "60m", "125m", "250m", "490m"]
TPP = [20.0, 80.0, 320.0]
NOISE_SIGMA = 0.004
PROBE_SEED = 20260805
GB = float(1 << 30)


def _form_a_true(truth: dict, N, D, bits) -> np.ndarray:
    """Hand-coded Form A: L = E + A/(N*f(b))^alpha + B/D^beta, f(16) = 1."""
    N, D, bits = np.asarray(N, float), np.asarray(D, float), np.asarray(bits, float)
    f = np.where(np.isclose(bits, 16.0), 1.0, np.nan)
    for b, fv in zip(QBITS, truth["f"]):
        f = np.where(np.isclose(bits, b), fv, f)
    return truth["E"] + truth["A"] / (N * f) ** truth["alpha"] + truth["B"] / D ** truth["beta"]


def _draw_truth(rng: np.random.Generator) -> dict:
    """Hidden truth: exponents/E/f in plausible ranges, f(b) ordered < 1."""
    return dict(
        alpha=rng.uniform(0.25, 0.4),
        beta=rng.uniform(0.25, 0.4),
        E=rng.uniform(0.5, 0.8),
        A=float(np.exp(rng.uniform(np.log(100), np.log(400)))),
        B=float(np.exp(rng.uniform(np.log(200), np.log(800)))),
        f=np.sort(rng.uniform(0.35, 0.95, size=4)),
    )


def _make_grid(truth: dict, rng: np.random.Generator) -> pd.DataFrame:
    """P2-shaped grid (LADDER n_nonemb x tokens-per-param x 5 precisions)
    with lognormal observation noise."""
    rows = []
    for s in SIZES:
        n = float(make_model(s).n_nonemb)
        for tpp in TPP:
            d = tpp * n
            for b in BITS5:
                lt = float(_form_a_true(truth, n, d, b))
                rows.append(dict(
                    size=s, n_nonemb=n, tokens=d, bits=b,
                    precision=f"{b:g}" if b != 16.0 else "bf16",
                    bpb=lt * float(np.exp(rng.normal(0.0, NOISE_SIGMA))),
                ))
    return pd.DataFrame(rows)


def _true_bstar(truth: dict, M: float, D: float, n_min: float = 1e6, n_cap: float = 1e13) -> float:
    """b* from the TRUE law: L is decreasing in N, so per bit width the best N
    is the byte-budget boundary N = min(cap, M*8/b); argmin over widths."""
    best_b, best_l = None, np.inf
    for b in BITS5:
        n = min(n_cap, M * 8.0 / b)
        if n < n_min:
            continue
        l = float(_form_a_true(truth, n, D, b))
        if l < best_l:
            best_b, best_l = b, l
    return best_b


class FitRecoveryProbe(Probe):
    name = "fit_recovery"
    description = "fitting pipeline vs hidden hand-coded Form A ground truth"
    gate_specs = [
        ("G1", "recovery: fitted alpha/beta within 15% relative and f(b) within "
               "10% of hidden truth, on both independent draws"),
        ("G2", "LOSO selection prefers Form A-free over Form B on A-generated "
               "data, on both draws"),
        ("G3", "upward extrapolation: fit on <=250m rows, predict 490m rows; "
               "mean |log residual| < 3x the injected noise sigma, both draws"),
        ("G4", "prescription: fitted-law b*(M,D) matches true-law b* on >= 90% "
               "of a 12x6 grid spanning 0.125-4GB x 20x-320x"),
        ("G5", "determinism: same seed twice gives identical fitted params"),
    ]

    def collect(self) -> dict:
        m: dict = {"draws": []}
        first_fit = first_df = first_truth = None
        for d in range(2):
            rng = np.random.default_rng([PROBE_SEED, d])
            truth = _draw_truth(rng)
            df = _make_grid(truth, rng)

            fit = fit_form(FormAFree(QBITS), df, n_starts=96, seed=0, top_k=6)
            rel = {
                "alpha": abs(fit.params["alpha"] - truth["alpha"]) / truth["alpha"],
                "beta": abs(fit.params["beta"] - truth["beta"]) / truth["beta"],
            }
            f_rel = {
                f"f_{b:g}": abs(fit.params[f"f_{b:g}"] - fv) / fv
                for b, fv in zip(QBITS, truth["f"])
            }

            loso_a = loso(FormAFree(QBITS), df, n_starts=48, seed=0, top_k=4)
            loso_b = loso(FormB(QBITS), df, n_starts=48, seed=0, top_k=4)

            up = fit_form(FormAFree(QBITS), df[df["size"] != "490m"],
                          n_starts=96, seed=0, top_k=6)
            held = df[df["size"] == "490m"]
            up_res = float(np.mean(np.abs(
                np.log(up.predict(held["n_nonemb"], held["tokens"], held["bits"]))
                - np.log(held["bpb"].to_numpy(dtype=float))
            )))

            m["draws"].append({
                "rel": rel, "f_rel": f_rel,
                "loso_a": loso_a.mean_error, "loso_b": loso_b.mean_error,
                "upward": up_res,
            })
            if d == 0:
                first_fit, first_df, first_truth = fit, df, truth

        # G4: phase-diagram agreement, fitted law (real prescribe machinery)
        # vs true law (closed-form boundary argmin above).
        m_grid = np.geomspace(0.125 * GB, 4 * GB, 12)
        d_grid = np.geomspace(20.0 * (0.125 * GB / 2.0), 320.0 * (4 * GB / 2.0), 6)
        fitted_b = phase_diagram(first_fit, m_grid, d_grid, bits=BITS5)
        true_b = np.array([[_true_bstar(first_truth, M, D) for D in d_grid] for M in m_grid])
        m["bstar_agree"] = float(np.mean(np.isclose(fitted_b, true_b)))
        m["bstar_cells"] = int(fitted_b.size)

        # G5: refit draw 0 with identical arguments.
        refit = fit_form(FormAFree(QBITS), first_df, n_starts=96, seed=0, top_k=6)
        m["deterministic"] = refit.params == first_fit.params
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        out: list[GateOutcome] = []

        rec_ok, rec_det = True, []
        for i, d in enumerate(m["draws"]):
            worst_ab = max(d["rel"].values())
            worst_f = max(d["f_rel"].values())
            rec_ok &= worst_ab <= 0.15 and worst_f <= 0.10
            rec_det.append(f"draw{i}: max|alpha,beta| rel={worst_ab:.3f}, max f rel={worst_f:.3f}")
        out.append(GateOutcome("G1", self.gate_specs[0][1], passed=rec_ok,
                               detail="; ".join(rec_det)))

        loso_ok = all(d["loso_a"] < d["loso_b"] for d in m["draws"])
        out.append(GateOutcome(
            "G2", self.gate_specs[1][1], passed=loso_ok,
            detail="; ".join(f"draw{i}: A-free={d['loso_a']:.2e} B={d['loso_b']:.2e}"
                             for i, d in enumerate(m["draws"])),
        ))

        up_ok = all(d["upward"] < 3 * NOISE_SIGMA for d in m["draws"])
        out.append(GateOutcome(
            "G3", self.gate_specs[2][1], passed=up_ok,
            detail="; ".join(f"draw{i}: {d['upward']:.5f}" for i, d in enumerate(m["draws"]))
                   + f" (bar {3 * NOISE_SIGMA})",
        ))

        out.append(GateOutcome(
            "G4", self.gate_specs[3][1], passed=m["bstar_agree"] >= 0.90,
            detail=f"agreement {m['bstar_agree']:.1%} over {m['bstar_cells']} (M, D) cells",
        ))

        out.append(GateOutcome("G5", self.gate_specs[4][1], passed=m["deterministic"],
                               detail=f"identical params: {m['deterministic']}"))
        return out
