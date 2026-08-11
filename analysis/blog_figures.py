"""Charts sized for a Substack post, from the same data as the book.

Blog figures are not book figures. They are viewed at roughly 700px wide,
often on a phone, with no surrounding chapter to explain them, and Substack
renders them on a light background regardless of the reader's theme. So:
bigger type, fewer elements, the key number written on the chart itself, and
light-mode only.

    python analysis/blog_figures.py [--only name1,name2]

Writes blog/figures/<name>.png.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog" / "figures"

SURFACE = "#ffffff"
INK, INK2, MUTED = "#101010", "#3d3d3a", "#7a7975"
GRID, BASE = "#e4e3dd", "#bfbeb4"
ARM = {"1.58": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "bf16": "#b3557f"}
ACCENT, WARN = "#2a78d6", "#c02f2f"

FIGS: dict[str, callable] = {}


def figure(name):
    def deco(fn):
        FIGS[name] = fn
        return fn
    return deco


def style(ax, *, ygrid=True, xgrid=False):
    ax.set_facecolor(SURFACE)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, lw=1.0, zorder=0)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def titles(ax, title, sub=None, xlabel=None, ylabel=None):
    ax.set_title(title, color=INK, fontsize=16.5, loc="left", pad=30 if sub else 12,
                 fontweight="bold")
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=MUTED, fontsize=11.5,
                va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=12.5, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=12.5, labelpad=8)


def results() -> list[dict]:
    p = ROOT / "results" / "results.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def grid_cells():
    cells = defaultdict(list)
    for r in results():
        rid = r["run_id"]
        if rid.startswith("l-") and r["status"] == "complete" and not any(
            t in rid for t in ("lrp", "ctl", "sqrelu")
        ):
            cells[(r["size"], r["precision"], round(r["tokens_per_param"]))].append(r["bpb_val1"])
    return cells


# ----------------------------------------------------------------- charts


@figure("memory-wall")
def f_memory_wall():
    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=SURFACE)
    style(ax)
    bars = ax.bar(["2023–24", "2026"], [8, 30], color=["#c9d9ef", ACCENT], width=0.5, zorder=3)
    for b, v in zip(bars, [8, 30]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v}%", ha="center",
                color=INK, fontsize=22, fontweight="bold")
    ax.set_ylim(0, 38)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.annotate("", xy=(1, 33.5), xytext=(0, 11.5),
                arrowprops=dict(arrowstyle="->", color=WARN, lw=2.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, 26.5, "nearly 4× in two years", ha="center", color=WARN,
            fontsize=13, fontweight="bold")
    titles(ax, "Memory is eating the AI infrastructure budget",
           sub="memory as a share of hyperscaler capital spending",
           ylabel="share of AI infrastructure spend")
    ax.text(0, -0.17, "datacenters now absorb ~70% of world memory output · "
            "server DRAM contract prices moved 60–90% in one quarter",
            transform=ax.transAxes, color=MUTED, fontsize=10.5)
    fig.tight_layout()
    return fig


@figure("crossover")
def f_crossover():
    c = grid_cells()
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    style(ax)
    ax.axhline(0, color=BASE, lw=1.8, zorder=2)
    colors = {"3m": ACCENT, "6m": "#eb6834", "12m": "#1baf7a"}
    for size in ("3m", "6m", "12m"):
        xs, gaps, bands = [], [], []
        for tpp in (20, 80, 320):
            lo, hi = c.get((size, "1.58", tpp), []), c.get((size, "bf16", tpp), [])
            if not lo or not hi:
                continue
            xs.append(tpp)
            gaps.append(st.mean(lo) - st.mean(hi))
            sig = max(st.stdev(lo) if len(lo) > 1 else 0.0,
                      st.stdev(hi) if len(hi) > 1 else 0.0)
            bands.append(2 * sig)
        if not xs:
            continue
        ax.errorbar(xs, gaps, yerr=bands, color=colors[size], lw=2.8, marker="o", ms=11,
                    capsize=6, capthick=2, elinewidth=2, zorder=4,
                    label=f"{size.upper()} params")
    ax.set_xscale("log")
    ax.set_xticks([20, 80, 320])
    ax.set_xticklabels(["20×", "80×", "320×"])
    ax.minorticks_off()
    ax.set_xlim(15, 480)
    leg = ax.legend(frameon=False, fontsize=12, loc="lower right", labelcolor=INK2)
    leg.set_title("model size", prop={"size": 11})
    leg.get_title().set_color(MUTED)
    ax.text(0.015, 0.94, "ternary is WORSE ↑", transform=ax.transAxes,
            color=MUTED, fontsize=11.5, fontweight="bold")
    ax.text(0.015, 0.06, "ternary is BETTER ↓", transform=ax.transAxes,
            color=MUTED, fontsize=11.5, fontweight="bold")
    titles(ax, "Ternary wins when undertrained, loses when overtrained",
           sub="1.58-bit weights vs 16-bit, same data, same everything else",
           xlabel="training tokens per parameter",
           ylabel="Δ bits per byte  (ternary − bf16)")
    ax.text(0, -0.20, "bars are ±2σ of measured run-to-run noise; a bar crossing zero "
            "means no claim", transform=ax.transAxes, color=MUTED, fontsize=10.5)
    fig.tight_layout()
    return fig


@figure("bit-regime")
def f_bit_regime():
    """The 12M @80x deficits by bit width, with the noise bar drawn to scale
    so the reader can see for themselves that the effect fits inside it."""
    c = grid_cells()
    ref = c.get(("12m", "bf16", 80))
    arms = [("4", "4-bit"), ("3", "3-bit"), ("2", "2-bit"), ("1.58", "ternary")]
    pts = [(lab, st.mean(c[("12m", p, 80)]) - st.mean(ref))
           for p, lab in arms if c.get(("12m", p, 80)) and ref]
    if len(pts) < 4:
        return None
    labels = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    xs = np.arange(len(pts))

    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    style(ax)
    key_for = {"4-bit": "4", "3-bit": "3", "2-bit": "2", "ternary": "1.58"}
    ax.plot(xs, vals, color=ACCENT, lw=2.6, zorder=3)
    for x, v, lab in zip(xs, vals, labels):
        ax.plot([x], [v], "o", ms=13, color=ARM[key_for[lab]], zorder=5)
        ax.text(x, v + 0.006, f"+{v:.3f}", ha="center", color=INK,
                fontsize=13, fontweight="bold", zorder=6)

    # Step labels sit in a clean row near the axis rather than beside the
    # line: the last segment is steep enough that a perpendicular offset
    # collides with the point label above it.
    lo_y = min(vals) - 0.022
    for i in range(len(vals) - 1):
        step = vals[i + 1] - vals[i]
        mid_x = (xs[i] + xs[i + 1]) / 2
        big = step > 0.03
        ax.text(mid_x, lo_y, f"+{step:.3f}", ha="center", va="center",
                color=WARN if big else MUTED,
                fontsize=13 if big else 11.5,
                fontweight="bold" if big else "normal")
    ax.text(-0.42, lo_y, "step:", ha="left", va="center", color=MUTED, fontsize=11)

    # The noise bar drawn to the same scale as the whole effect, so the
    # comparison is visual rather than asserted.
    bar = 0.1313
    spread = vals[-1] - vals[0]
    base_y = vals[0]
    # Labels sit *below* both arrows, at a shared height, so neither can
    # collide with the ternary point label at the top of the plot.
    for x_pos, height, color, label, bold in (
        (len(pts) - 0.18, spread, INK2, f"whole\neffect\n{spread:.3f}", False),
        (len(pts) + 0.48, bar, WARN, f"noise\nbar\n{bar:.3f}", True),
    ):
        ax.annotate("", xy=(x_pos, base_y), xytext=(x_pos, base_y + height),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2.2))
        ax.text(x_pos, base_y - 0.008, label, ha="center", va="top",
                color=color, fontsize=11, linespacing=1.35,
                fontweight="bold" if bold else "normal")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_xlim(-0.45, len(pts) + 1.0)
    ax.set_ylim(lo_y - 0.030, max(vals) + 0.024)
    titles(ax, "Ordered by bit width, with a step between 3-bit and 2-bit",
           sub="12M parameters, 80 tokens per parameter, quality lost against 16-bit",
           ylabel="Δ bits per byte  (worse than bf16)")
    ax.text(0, -0.185, "and the whole effect is smaller than my measured noise bar, "
            "which is why I am not claiming it",
            transform=ax.transAxes, color=WARN, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("lr-sensitivity")
def f_lr():
    import re
    pat = re.compile(r"^l-lrp-6m-([\d.]+|bf16)-x([\d.]+)$")
    pts = defaultdict(dict)
    for r in results():
        m = pat.match(r["run_id"])
        if m and r["status"] == "complete":
            pts[m.group(1)][float(m.group(2))] = r["bpb_val1"]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    style(ax)
    name = {"1.58": "ternary", "2": "2-bit", "3": "3-bit", "4": "4-bit", "bf16": "bf16"}
    for prec in ("bf16", "4", "3", "2", "1.58"):
        d = pts.get(prec)
        if not d:
            continue
        xs = sorted(d)
        ys = [d[x] for x in xs]
        wide = prec == "bf16"
        ax.plot(xs, ys, color=ARM[prec], lw=3.4 if wide else 2.2,
                marker="o", ms=11 if wide else 8, zorder=5 if wide else 4)
        ax.annotate(name[prec], (xs[-1], ys[-1]), xytext=(11, 0),
                    textcoords="offset points", color=ARM[prec],
                    fontsize=13 if wide else 11.5, fontweight="bold", va="center")
    ax.set_xscale("log")
    ticks = sorted({x for d in pts.values() for x in d})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}×" for t in ticks], fontsize=12.5)
    ax.minorticks_off()
    ax.set_xlim(min(ticks) * 0.85, max(ticks) * 1.55)
    titles(ax, "Low-bit training barely notices a bad learning rate",
           sub="6M parameters; across a 4× learning-rate range",
           xlabel="learning-rate multiplier", ylabel="validation bits per byte")
    ax.text(0, -0.19, "full precision swings 0.191 bits per byte across this range. "
            "ternary swings 0.034.", transform=ax.transAxes, color=INK2,
            fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(FIGS)
    for n in names:
        fig = FIGS[n]()
        if fig is None:
            print(f"  skip {n}: not enough data yet")
            continue
        p = OUT / f"{n}.png"
        fig.savefig(p, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
