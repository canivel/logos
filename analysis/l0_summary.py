"""L0 exit artifacts: the gap-vs-tokens/param figure, the BPB small
multiples, and the sigma table (markdown), regenerated from results.jsonl.

Idempotent: re-run any time; figures land in docs/figures/ and the table
prints to stdout (paste into the research note). Uses the project's fixed
entity->color mapping (validated categorical palette, slots in bits-ascending
order): 1.58->blue, 2->orange, 3->aqua, 4->yellow, bf16->magenta.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

# Fixed entity->slot assignment (never re-ranked). Light-mode steps.
ARM_COLOR = {"1.58": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "bf16": "#e87ba4"}
SIZE_COLOR = {"3m": "#2a78d6", "6m": "#eb6834", "12m": "#1baf7a", "25m": "#eda100"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

L0_SIZES = ("3m", "6m")


def load_cells() -> dict[tuple[str, str, float], list[float]]:
    rows = [
        json.loads(x)
        for x in (ROOT / "results" / "results.jsonl").read_text().splitlines()
        if x.strip()
    ]
    cells: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for r in rows:
        if r["size"] in L0_SIZES and r["run_id"].startswith("l-") and r["status"] == "complete":
            tpp = round(r["tokens_per_param"])
            cells[(r["size"], r["precision"], tpp)].append(r["bpb_val1"])
    return cells


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.75)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xscale("log")
    ax.set_xticks([20, 80, 320])
    ax.set_xticklabels(["20×", "80×", "320×"])
    ax.minorticks_off()


def fig_gap(cells) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=BASE, lw=1.25)
    for size in L0_SIZES:
        xs, gaps, bands = [], [], []
        for tpp in (20, 80, 320):
            lo = cells.get((size, "1.58", tpp), [])
            hi = cells.get((size, "bf16", tpp), [])
            if not lo or not hi:
                continue
            xs.append(tpp)
            gaps.append(st.mean(lo) - st.mean(hi))
            sig = 0.0
            if len(lo) > 1 or len(hi) > 1:
                sig = max(
                    st.stdev(lo) if len(lo) > 1 else 0.0,
                    st.stdev(hi) if len(hi) > 1 else 0.0,
                )
            bands.append(2 * sig)
        if not xs:
            continue
        c = SIZE_COLOR[size]
        ax.errorbar(
            xs, gaps, yerr=bands, color=c, lw=2, marker="o", ms=8,
            capsize=4, capthick=1.5, elinewidth=1.5,
        )
        ax.annotate(
            size, (xs[-1], gaps[-1]), xytext=(8, 0), textcoords="offset points",
            color=c, fontsize=10, fontweight="bold", va="center",
        )
    ax.set_xlabel("training tokens per parameter", color=INK2, fontsize=10)
    ax.set_ylabel("Δ BPB  (ternary − bf16)", color=INK2, fontsize=10)
    ax.set_title(
        "The crossover: ternary wins undertrained, loses overtrained",
        color=INK, fontsize=12, loc="left",
    )
    ax.text(
        0.02, 0.03, "below 0 = ternary better · bands are ±2σ seed noise",
        transform=ax.transAxes, color=MUTED, fontsize=8.5,
    )
    fig.tight_layout()
    out = FIG_DIR / "l0_gap_vs_dn.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_bpb(cells) -> Path:
    fig, axes = plt.subplots(1, len(L0_SIZES), figsize=(9, 4.2), sharey=True, facecolor=SURFACE)
    for ax, size in zip(axes, L0_SIZES):
        _style(ax)
        for prec in ("1.58", "bf16"):
            xs, ys = [], []
            for tpp in (20, 80, 320):
                v = cells.get((size, prec, tpp), [])
                if not v:
                    continue
                xs.append(tpp)
                ys.append(st.mean(v))
                ax.plot([tpp] * len(v), v, "o", color=ARM_COLOR[prec], ms=4, alpha=0.45)
            if xs:
                ax.plot(xs, ys, color=ARM_COLOR[prec], lw=2, marker="o", ms=8)
                label = "ternary" if prec == "1.58" else prec
                ax.annotate(
                    label, (xs[-1], ys[-1]), xytext=(6, -2), textcoords="offset points",
                    color=ARM_COLOR[prec], fontsize=9.5, fontweight="bold",
                )
        ax.set_title(f"{size} non-emb params", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("tokens per parameter", color=INK2, fontsize=10)
    axes[0].set_ylabel("val bits-per-byte", color=INK2, fontsize=10)
    fig.suptitle(
        "L0 replication tier: FineWeb-Edu validation BPB (dots = seeds)",
        color=INK, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "l0_bpb_vs_dn.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def sigma_table(cells) -> str:
    lines = [
        "| Cell | mean BPB | seed σ | n |",
        "|------|---------:|-------:|---|",
    ]
    for k in sorted(cells, key=lambda k: (k[0], k[2], k[1])):
        v = cells[k]
        s = f"{st.stdev(v):.4f}" if len(v) > 1 else "—"
        lines.append(f"| {k[0]} {k[1]} @{k[2]:.0f}× | {st.mean(v):.4f} | {s} | {len(v)} |")
    lines.append("")
    lines.append("| Cell | gap (ternary−bf16) | 2σ | verdict |")
    lines.append("|------|--------------------:|----:|---------|")
    bysize = defaultdict(dict)
    for (size, prec, tpp), v in cells.items():
        bysize[(size, tpp)][prec] = v
    for (size, tpp), d in sorted(bysize.items()):
        lo, hi = d.get("1.58", []), d.get("bf16", [])
        if len(lo) > 1 and len(hi) > 1:
            g = st.mean(lo) - st.mean(hi)
            two = 2 * max(st.stdev(lo), st.stdev(hi))
            verdict = "**significant**" if abs(g) > two else "within noise"
            lines.append(f"| {size} @{tpp:.0f}× | {g:+.4f} | {two:.4f} | {verdict} |")
    return "\n".join(lines)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    n = sum(len(v) for v in cells.values())
    print(f"{n} completed L0 runs\n")
    print(sigma_table(cells))
    print("\nfigures:", fig_gap(cells), fig_bpb(cells), sep="\n  ")
