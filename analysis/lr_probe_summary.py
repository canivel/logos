"""L1 LR-transfer probes: pick the per-precision LR multiplier, noise-aware.

The protocol freeze (plan v0.2 s6 / v0.3 L1) sets one LR multiplier per
precision for the rest of the project, so this decision must not be made by
eyeballing argmins of single-seed runs. The rule implemented here:

  1. The comparison bar is the MEASURED seed noise, not the eye. Two
     single-seed runs differ significantly only if |dBPB| > 2*sigma*sqrt(2),
     with sigma taken from the L0 seed-sigma table at the same size.
  2. Deviate from the prior multiplier only on significant evidence.
     Priors: bf16 -> 1x, quantized -> 2x (BitNet b1.58), and the l1ctl
     control arms already confirmed those two at 2 seeds x 2 D/N points.
  3. An argmin sitting at an EDGE of the probe grid means the optimum is
     unbracketed; that is a hole, not a result (l1lrx extends the grid).

Outputs the probe figure, the decision table, and the frozen-rule YAML body.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

ARM_COLOR = {"1.58": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "bf16": "#e87ba4"}
ARM_LABEL = {"1.58": "ternary", "2": "2-bit", "3": "3-bit", "4": "4-bit", "bf16": "bf16"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

PRIOR = {"1.58": 2.0, "2": 2.0, "3": 2.0, "4": 2.0, "bf16": 1.0}
PROBE_RE = re.compile(r"^l-lrp-6m-(?P<prec>[\d.]+|bf16)-x(?P<mult>[\d.]+)(?:-s(?P<seed>\d+))?$")


def load_rows() -> list[dict]:
    return [
        json.loads(x)
        for x in (ROOT / "results" / "results.jsonl").read_text().splitlines()
        if x.strip()
    ]


def probe_cells(rows: list[dict]) -> dict[tuple[str, float], list[float]]:
    cells: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        m = PROBE_RE.match(r["run_id"])
        if m and r["status"] == "complete":
            cells[(m["prec"], float(m["mult"]))].append(r["bpb_val1"])
    return cells


def seed_sigma_6m(rows: list[dict]) -> float:
    """Mean within-cell seed sigma at 6m from the L0 tier."""
    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["size"] == "6m" and r["phase"] == "l0" and r["status"] == "complete":
            cells[(r["precision"], r["tokens_per_param"])].append(r["bpb_val1"])
    sigmas = [st.stdev(v) for v in cells.values() if len(v) > 1]
    return float(st.mean(sigmas)) if sigmas else float("nan")


def decide(cells, sigma: float) -> tuple[list[str], dict[str, float]]:
    """Apply the rule. Returns (report lines, frozen multipliers)."""
    bar = 2 * sigma * math.sqrt(2)  # 2-sigma bar on a difference of two means
    lines = [
        f"seed sigma (6m, measured) = {sigma:.4f} BPB; "
        f"significance bar on a single-seed difference = {bar:.4f} BPB",
        "",
        "| arm | " + " | ".join(f"x{m:g}" for m in (0.25, 0.5, 1, 2, 4)) + " | argmin | prior | frozen | note |",
        "|-----|" + "------|" * 5 + "--------|-------|--------|------|",
    ]
    frozen: dict[str, float] = {}
    grid_edges = (min(m for _, m in cells), max(m for _, m in cells))
    for prec in ("1.58", "2", "3", "4", "bf16"):
        pts = {m: st.mean(v) for (p, m), v in cells.items() if p == prec}
        if not pts:
            continue
        cellstr = []
        best_m = min(pts, key=pts.get)
        for m in (0.25, 0.5, 1, 2, 4):
            if m in pts:
                n = len([v for (p, mm), v in cells.items() if p == prec and mm == m][0])
                mark = "**" if m == best_m else ""
                cellstr.append(f"{mark}{pts[m]:.4f}{mark}" + (f" (n={n})" if n > 1 else ""))
            else:
                cellstr.append("—")
        prior = PRIOR[prec]
        gain = pts.get(prior, float("inf")) - pts[best_m]
        significant = gain > bar
        pick = best_m if significant else prior
        notes = []
        if not significant:
            notes.append(f"argmin gain {gain:+.4f} < bar; keep prior")
        else:
            notes.append(f"argmin beats prior by {gain:.4f} > bar")
        if best_m in grid_edges and len(pts) < 5:
            notes.append(f"**argmin at grid edge x{best_m:g} — optimum unbracketed**")
        frozen[prec] = pick
        lines.append(
            f"| {ARM_LABEL[prec]} | " + " | ".join(cellstr) +
            f" | x{best_m:g} | x{prior:g} | **x{pick:g}** | {'; '.join(notes)} |"
        )
    return lines, frozen


def figure(cells, sigma: float) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.75)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for prec in ("1.58", "2", "3", "4", "bf16"):
        pts = sorted((m, st.mean(v)) for (p, m), v in cells.items() if p == prec)
        if not pts:
            continue
        xs = [m for m, _ in pts]
        ys = [y for _, y in pts]
        c = ARM_COLOR[prec]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=8)
        ax.annotate(
            ARM_LABEL[prec], (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
            color=c, fontsize=9.5, fontweight="bold", va="center",
        )
    ax.set_xscale("log")
    ticks = sorted({m for _, m in cells})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("learning-rate multiplier (relative to the size's base LR)", color=INK2, fontsize=10)
    ax.set_ylabel("val bits-per-byte", color=INK2, fontsize=10)
    ax.set_title("LR transfer at 6M params, 31 tokens/param", color=INK, fontsize=12, loc="left")
    ax.text(
        0.02, 0.04,
        f"differences under {2 * sigma * math.sqrt(2):.3f} BPB are inside measured seed noise",
        transform=ax.transAxes, color=MUTED, fontsize=8.5,
    )
    fig.tight_layout()
    out = FIG_DIR / "l1_lr_probes.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    cells = probe_cells(rows)
    sigma = seed_sigma_6m(rows)
    lines, frozen = decide(cells, sigma)
    print(f"{sum(len(v) for v in cells.values())} probe runs\n")
    print("\n".join(lines))
    print("\nfrozen multipliers:", json.dumps(frozen))
    print("figure:", figure(cells, sigma))
