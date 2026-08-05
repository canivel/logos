"""End-to-end demo of the fitting subsystem on synthetic data — the skeleton
of the fitting notebook PLAN.md s7/s8 call for.

Generates the P2 run matrix (PLAN.md s7) from a ground-truth Form A with
realistic params, fits the competing forms, selects by LOSO (never in-sample,
principle 7), bootstraps CIs, runs the RQ4 upward-extrapolation check, and
saves the four paper figures to analysis/out/.

Run: python analysis/fit_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from logos.config import ALL_PRECISIONS, Precision, make_model
from logos.fitting.fit import fit_form
from logos.fitting.forms import FormAFree, FormAParam, FormB, fmt_bits
from logos.fitting.plots import (
    plot_extrapolation_check,
    plot_gap_curves,
    plot_iso_memory_frontier,
    plot_phase_diagram,
)
from logos.fitting.prescribe import GB, effective_capacity_summary, optimal_config
from logos.fitting.select import (
    bootstrap_ci,
    check_extrapolation,
    seed_sigma_by_size,
    select_by_loso,
)

OUT = Path(__file__).resolve().parent / "out"

# Ground-truth Form A (effective capacity), realistic params (PLAN.md s8).
TRUE = dict(E=0.6, A=300.0, alpha=0.32, B=250.0, beta=0.28)
TRUE_F = {1.58: 0.55, 2.0: 0.62, 3.0: 0.78, 4.0: 0.88, 16.0: 1.0}
TRUE_ALL = {**TRUE, **{f"f_{fmt_bits(b)}": v for b, v in TRUE_F.items() if b != 16.0}}
NOISE_SIGMA = 0.005  # lognormal seed noise, ~RQ1 scale

# The P2 run matrix (PLAN.md s7): size -> (tokens/param tiers, seeds).
P2_GRID = {
    "25m": ((20, 80, 320), 3),
    "60m": ((20, 80, 320), 3),
    "125m": ((20, 80, 320), 1),
    "250m": ((20, 80, 320), 1),
    "490m": ((20, 80), 1),
}
P2_EXT = ("1.58", "4", "bf16")  # 490m extended tier at 320x


def make_grid(seed: int = 0, sigma: float = NOISE_SIGMA) -> pd.DataFrame:
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


def main() -> None:
    t0 = time.time()
    df = make_grid()
    print(f"synthetic P2 grid: {len(df)} runs, "
          f"{df['size'].nunique()} sizes x {df['precision'].nunique()} precisions")

    # -- form competition, selected by LOSO extrapolation (principle 7) --
    forms = {"A-free": FormAFree(), "A-param": FormAParam(), "B": FormB()}
    sel = select_by_loso(forms, df, n_starts=48, full_n_starts=192, seed=0)
    print("\nLOSO extrapolation error (mean |log L_pred - log L_obs|):")
    for name, r in sel.loso.items():
        marker = " <== selected" if name == sel.best else ""
        print(f"  {name:8s} mean={r.mean_error:.5f}  upward(RQ4)={r.upward_error:.5f}{marker}")
    fit = sel.fits[sel.best]
    print(f"full-data fit: loss={fit.train_loss:.3e} converged={fit.converged_frac:.0%}")

    # -- bootstrap CIs (stratified by size x precision) --
    boot = bootstrap_ci(
        FormAFree(), df, n_boot=100, seed=1, n_starts=8, top_k=3,
        base_fit=sel.fits["A-free"],
        derived={f"f_per_byte_{fmt_bits(b)}": (lambda p, b=b: p[f"f_{fmt_bits(b)}"] * 16 / b)
                 for b in (1.58, 2.0, 3.0, 4.0)},
    )
    print(f"\nrecovered params vs ground truth (bootstrap {boot.n_boot}, 95% CI):")
    print(f"  {'param':14s} {'true':>9s} {'fit':>9s} {'ci_lo':>9s} {'ci_hi':>9s}  in")
    for k, tv in TRUE_ALL.items():
        v, (lo, hi) = boot.point[k], boot.ci[k]
        print(f"  {k:14s} {tv:9.4f} {v:9.4f} {lo:9.4f} {hi:9.4f}  {'y' if lo <= tv <= hi else 'N'}")

    print("\neffective capacity f(b) vs the ~2 bits/param bound:")
    for b, row in effective_capacity_summary(sel.fits["A-free"]).items():
        print(f"  b={b:<5g} f={row['f']:.3f} (true {TRUE_F[b]:.2f})  "
              f"per-byte x{row['f_per_byte']:.2f}  "
              f"knowledge {row['knowledge_bits']:.2f}/{row['bound']:.2f} bits "
              f"({row['fraction_of_bound']:.0%} of bound)")

    # -- RQ4 upward extrapolation: fit below 490m, predict 490m --
    below, held = df[df["size"] != "490m"], df[df["size"] == "490m"]
    fit_below = fit_form(FormAFree(), below, n_starts=192, seed=2)
    sigma = seed_sigma_by_size(df)
    sig_max = max(sigma.values())  # extrapolate sigma upward (principle 4)
    verdict = check_extrapolation(fit_below, held, sig_max)["A-free"]
    print(f"\nRQ4 check on held-out 490m: within 2x seed-sigma band={verdict.within_band} "
          f"(max err {verdict.max_abs_err:.4f} vs band {verdict.band:.4f}), "
          f"precision ordering correct={verdict.ordering_correct}")

    # -- cash in the law: the prescription at a phone-class budget --
    D_star = 100e9
    n_star, b_star, l_star = optimal_config(fit, 0.5 * GB, D_star)
    print(f"\nprescription at 0.5 GB weights, D={D_star:.0e}: "
          f"N*={n_star / 1e9:.2f}B params at {b_star:g}-bit -> BPB {l_star:.3f}")

    # -- the four figures --
    paths = [
        plot_iso_memory_frontier(fit, OUT / "iso_memory_frontier.png", D=D_star),
        plot_phase_diagram(fit, OUT / "phase_diagram.png",
                           M_grid=np.geomspace(0.125 * GB, 4 * GB, 24),
                           D_grid=np.geomspace(1e9, 1e12, 24)),
        plot_gap_curves(df, OUT / "gap_curves.png"),
        plot_extrapolation_check(fit_below, held, sig_max, OUT / "extrapolation_check.png"),
    ]
    print("\nfigures:")
    for p in paths:
        print(f"  {p}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
