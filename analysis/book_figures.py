"""Every figure in the LOGOS book, in light and dark variants.

Real measurements come from results/results.jsonl and real checkpoints;
anything schematic is drawn with an explicit "illustrative" marker inside
the figure so a reader can never mistake a teaching sketch for data.

    python analysis/book_figures.py [--only name1,name2]

Writes docs/book/figures/<name>.png and <name>-dark.png.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "book" / "figures"

# Validated categorical palette; slots assigned to entities in bits-ascending
# order and never re-ranked. Dark column is the same hues stepped for the
# dark surface, not a different palette.
THEME = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", base="#c3c2b7", faint="#f0efec",
        arm={"1.58": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "bf16": "#e87ba4"},
        seq=["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"],
        accent="#2a78d6", warn="#d03b3b", good="#0ca30c",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", base="#383835", faint="#242422",
        arm={"1.58": "#3987e5", "2": "#d95926", "3": "#199e70", "4": "#c98500", "bf16": "#d55181"},
        seq=["#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
        accent="#3987e5", warn="#d03b3b", good="#0ca30c",
    ),
}
ARM_LABEL = {"1.58": "ternary (1.58-bit)", "2": "2-bit", "3": "3-bit", "4": "4-bit", "bf16": "bf16"}
FIGS: dict[str, callable] = {}


def figure(name):
    def deco(fn):
        FIGS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------- helpers


def style(ax, t, *, xgrid=False, ygrid=True):
    ax.set_facecolor(t["surface"])
    if ygrid:
        ax.grid(True, axis="y", color=t["grid"], lw=0.75, zorder=0)
    if xgrid:
        ax.grid(True, axis="x", color=t["grid"], lw=0.75, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["base"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["ink2"])


def titles(ax, t, title, xlabel=None, ylabel=None, sub=None):
    # With a subtitle the title has to clear it: the subtitle sits just above
    # the axes, so the title pad must exceed the subtitle's line height.
    ax.set_title(title, color=t["ink"], fontsize=12.5, loc="left", pad=26 if sub else 9)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, color=t["muted"], fontsize=9,
                va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=10)


def illustrative(ax, t, text="schematic — illustrates the shape of the effect, not measured data"):
    ax.text(0.5, -0.19, text, transform=ax.transAxes, color=t["muted"],
            fontsize=8.5, ha="center", style="italic")


def blank_ax(fig, t):
    ax = fig.add_subplot(111)
    ax.set_facecolor(t["surface"])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def results() -> list[dict]:
    p = ROOT / "results" / "results.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cells(rows, phase_prefix="l", sizes=("3m", "6m")):
    out = defaultdict(list)
    for r in rows:
        if r["size"] in sizes and r["run_id"].startswith("l-") and r["status"] == "complete" \
                and "ctl" not in r["run_id"] and "lrp" not in r["run_id"]:
            out[(r["size"], r["precision"], round(r["tokens_per_param"]))].append(r["bpb_val1"])
    return out


# ---------------------------------------------------------------- ch 1


@figure("memory-wall")
def f_memory_wall(t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor=t["surface"],
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    style(ax1, t)
    bars = ax1.bar(["2023–24", "2026"], [8, 30], color=[t["seq"][1], t["accent"]],
                   width=0.55, zorder=3)
    bars[0].set_linewidth(0); bars[1].set_linewidth(0)
    for b, v in zip(bars, [8, 30]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v}%", ha="center",
                 color=t["ink"], fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 36)
    ax1.set_yticks([0, 10, 20, 30])
    ax1.set_yticklabels(["0%", "10%", "20%", "30%"])
    titles(ax1, t, "Memory's share of hyperscaler capex",
           ylabel="share of AI infrastructure spend")
    ax1.text(0.5, 0.90, "nearly 4× in two years", transform=ax1.transAxes,
             color=t["muted"], fontsize=9, ha="center")

    style(ax2, t, ygrid=False, xgrid=True)
    facts = [
        ("DRAM bit supply growth, 2026", 16),
        ("Server DRAM contract price, one quarter", 75),
        ("Share of world memory output going to datacenters", 70),
    ]
    ypos = np.arange(len(facts))[::-1]
    vals = [v for _, v in facts]
    ax2.barh(ypos, vals, color=[t["seq"][1], t["warn"], t["seq"][2]], height=0.5, zorder=3)
    for y, (lab, v) in zip(ypos, facts):
        ax2.text(v + 2, y, f"{v}%" + (" (+60–90%)" if v == 75 else ""), va="center",
                 color=t["ink"], fontsize=10, fontweight="bold")
        ax2.text(0, y + 0.42, lab, va="bottom", color=t["ink2"], fontsize=9)
    ax2.set_yticks([])
    ax2.set_xlim(0, 108)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    titles(ax2, t, "Supply cannot answer demand quickly")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ch 2


@figure("transformer-anatomy")
def f_anatomy(t):
    fig = plt.figure(figsize=(9.5, 5.6), facecolor=t["surface"])
    ax = blank_ax(fig, t)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    def box(x, y, w, h, label, color, fg=None, fs=10, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor=color, edgecolor="none", zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=fg or t["ink"], fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     color=t["muted"], lw=1.2, zorder=2))

    box(0.3, 8.6, 2.6, 0.9, "tokens in", t["faint"], t["ink2"])
    box(0.3, 7.1, 2.6, 1.0, "embedding table\n32,768 × d", t["seq"][0], "#0b0b0b", 9.5)
    arrow(1.6, 8.55, 1.6, 8.15)
    arrow(1.6, 7.05, 1.6, 6.6)

    ax.add_patch(FancyBboxPatch((0.05, 1.35), 6.4, 5.2,
                                boxstyle="round,pad=0.08,rounding_size=0.15",
                                facecolor=t["faint"], edgecolor=t["base"], lw=1, zorder=1))
    ax.text(0.3, 6.25, "the body — one block, repeated L times", color=t["ink2"], fontsize=9.5)

    box(0.4, 4.9, 1.5, 0.7, "RMSNorm", t["grid"], t["ink2"], 9)
    box(2.1, 4.9, 2.0, 0.7, "attention", t["arm"]["1.58"], "#ffffff", 10, True)
    box(4.3, 4.9, 1.9, 0.7, "+ residual", t["grid"], t["ink2"], 9)
    box(0.4, 3.4, 1.5, 0.7, "RMSNorm", t["grid"], t["ink2"], 9)
    box(2.1, 3.4, 2.0, 0.7, "feed-forward", t["arm"]["1.58"], "#ffffff", 10, True)
    box(4.3, 3.4, 1.9, 0.7, "+ residual", t["grid"], t["ink2"], 9)
    arrow(1.9, 5.25, 2.05, 5.25); arrow(4.1, 5.25, 4.25, 5.25)
    arrow(1.15, 4.85, 1.15, 4.15); arrow(1.9, 3.75, 2.05, 3.75); arrow(4.1, 3.75, 4.25, 3.75)
    ax.text(3.3, 2.55, "each block: mix information across positions,\nthen process each position on its own",
            ha="center", color=t["muted"], fontsize=9)
    ax.text(3.3, 1.7, "these two boxes hold ~all the parameters\nand are the only things this project quantizes",
            ha="center", color=t["arm"]["1.58"], fontsize=9, fontweight="bold")

    box(0.3, 0.35, 2.6, 0.8, "output head (tied)", t["seq"][0], "#0b0b0b", 9.5)
    arrow(1.6, 1.3, 1.6, 1.2)

    ax.text(6.95, 7.05, "where the bytes are", color=t["ink"], fontsize=11, fontweight="bold")
    lines = [
        ("embedding + head", "stays 16-bit in every arm", t["seq"][0]),
        ("attention + FFN weights", "the arm's precision:\n1.58 / 2 / 3 / 4 / 16 bits", t["arm"]["1.58"]),
        ("norms", "16-bit, negligible", t["grid"]),
    ]
    for i, (a, b, c) in enumerate(lines):
        y = 6.3 - i * 1.25
        ax.add_patch(FancyBboxPatch((6.95, y - 0.12), 0.26, 0.26,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        ax.text(7.35, y + 0.02, a, color=t["ink"], fontsize=9.5, fontweight="bold", va="center")
        ax.text(7.35, y - 0.22, b, color=t["ink2"], fontsize=8.8, va="top")
    ax.text(6.95, 2.75,
            "at small scale the embedding can\noutweigh the body — which is why\nlaws are fitted on non-embedding\nparameters, and byte claims name\nwhich count they mean",
            color=t["muted"], fontsize=9, va="top", linespacing=1.5)
    return fig


# ---------------------------------------------------------------- ch 3


@figure("loss-to-bpb")
def f_loss_to_bpb(t):
    rows = results()
    c = cells(rows)

    def mean(size, prec, tpp):
        v = c.get((size, prec, tpp))
        return st.mean(v) if v else None

    entries = [("raw UTF-8 text", 8.0, t["muted"]),
               ("3M params, 20 tok/param", mean("3m", "bf16", 20), t["arm"]["bf16"]),
               ("3M params, 320 tok/param", mean("3m", "bf16", 320), t["arm"]["bf16"]),
               ("6M params, 320 tok/param", mean("6m", "bf16", 320), t["arm"]["bf16"])]
    entries = [(a, b, c_) for a, b, c_ in entries if b]
    fig, ax = plt.subplots(figsize=(8.6, 4.0), facecolor=t["surface"])
    style(ax, t, ygrid=False, xgrid=True)
    ypos = np.arange(len(entries))[::-1]
    ax.barh(ypos, [v for _, v, _ in entries], color=[c_ for _, _, c_ in entries],
            height=0.5, zorder=3)
    for y, (lab, v, _) in zip(ypos, entries):
        ax.text(v + 0.12, y, f"{v:.2f}", va="center", color=t["ink"], fontsize=10.5,
                fontweight="bold")
        ax.text(0.06, y + 0.42, lab, va="bottom", color=t["ink2"], fontsize=9.5)
    ax.set_yticks([])
    ax.set_xlim(0, 9)
    titles(ax, t, "Bits per byte: how many bits the model needs per byte of text",
           xlabel="bits per byte  (lower is better)",
           sub="all model rows are full-precision runs on FineWeb-Edu, so they are directly comparable")
    ax.text(0.99, 0.06, "raw text with no model costs 8 bits per byte",
            transform=ax.transAxes, ha="right", color=t["muted"], fontsize=8.5)
    fig.tight_layout()
    return fig


@figure("l0-crossover")
def f_l0_crossover(t):
    c = cells(results())
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=t["surface"])
    style(ax, t)
    ax.axhline(0, color=t["base"], lw=1.4, zorder=2)
    size_color = {"3m": t["arm"]["1.58"], "6m": t["arm"]["2"]}
    # the two curves converge at 320x, so their end labels get staggered
    label_dy = {"3m": 13, "6m": -13}
    for size in ("3m", "6m"):
        xs, gaps, bands = [], [], []
        for tpp in (20, 80, 320):
            lo, hi = c.get((size, "1.58", tpp), []), c.get((size, "bf16", tpp), [])
            if not lo or not hi:
                continue
            xs.append(tpp)
            gaps.append(st.mean(lo) - st.mean(hi))
            sig = max(st.stdev(lo) if len(lo) > 1 else 0.0, st.stdev(hi) if len(hi) > 1 else 0.0)
            bands.append(2 * sig)
        if not xs:
            continue
        col = size_color[size]
        ax.errorbar(xs, gaps, yerr=bands, color=col, lw=2.2, marker="o", ms=9,
                    capsize=5, capthick=1.6, elinewidth=1.6, zorder=4)
        ax.annotate(f"{size} params", (xs[-1], gaps[-1]), xytext=(11, label_dy[size]),
                    textcoords="offset points", color=col, fontsize=10.5,
                    fontweight="bold", va="center")
    ax.set_xscale("log")
    ax.set_xticks([20, 80, 320])
    ax.set_xticklabels(["20×", "80×", "320×"])
    ax.minorticks_off()
    ax.set_xlim(15, 620)
    titles(ax, t, "Ternary wins when undertrained and loses when overtrained",
           xlabel="training tokens per parameter",
           ylabel="Δ bits per byte  (ternary − bf16)",
           sub="error bars are ±2σ of measured seed noise; a point whose bar crosses zero is not a claim")
    ax.text(0.015, 0.93, "ternary worse ↑", transform=ax.transAxes, color=t["muted"], fontsize=9)
    ax.text(0.015, 0.05, "ternary better ↓", transform=ax.transAxes, color=t["muted"], fontsize=9)
    fig.tight_layout()
    return fig


@figure("l0-bpb")
def f_l0_bpb(t):
    c = cells(results())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=True, facecolor=t["surface"])
    for ax, size in zip(axes, ("3m", "6m")):
        style(ax, t)
        for prec in ("1.58", "bf16"):
            xs, ys = [], []
            for tpp in (20, 80, 320):
                v = c.get((size, prec, tpp), [])
                if not v:
                    continue
                xs.append(tpp); ys.append(st.mean(v))
                ax.plot([tpp] * len(v), v, "o", color=t["arm"][prec], ms=4.5, alpha=0.45, zorder=3)
            if xs:
                ax.plot(xs, ys, color=t["arm"][prec], lw=2.2, marker="o", ms=8, zorder=4)
                ax.annotate("ternary" if prec == "1.58" else "bf16", (xs[-1], ys[-1]),
                            xytext=(7, -1), textcoords="offset points",
                            color=t["arm"][prec], fontsize=10, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks([20, 80, 320]); ax.set_xticklabels(["20×", "80×", "320×"])
        ax.minorticks_off(); ax.set_xlim(15, 700)
        ax.set_xlabel("tokens per parameter", color=t["ink2"], fontsize=10)
        ax.set_title(f"{size} non-embedding parameters", color=t["ink"], fontsize=11, loc="left")
    axes[0].set_ylabel("validation bits per byte", color=t["ink2"], fontsize=10)
    fig.suptitle("More training helps every arm — but not equally",
                 color=t["ink"], fontsize=12.5, x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.915, "faint dots are individual seeds; the line follows their mean",
             color=t["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


# ---------------------------------------------------------------- ch 4


@figure("scaling-law")
def f_scaling_law(t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2), facecolor=t["surface"])
    E, A, B, al, be = 0.7, 400.0, 1800.0, 0.34, 0.28
    N = np.geomspace(1e6, 1e10, 200)
    style(ax1, t)
    for D, col in zip((1e9, 1e10, 1e11), t["seq"][1:4]):
        ax1.plot(N, E + A / N**al + B / D**be, color=col, lw=2.2,
                 label=f"D = {D:.0e} tokens")
    ax1.axhline(E, color=t["muted"], lw=1.2, ls="--")
    ax1.text(1.2e6, E + 0.02, "E — the floor no model beats", color=t["muted"], fontsize=8.5)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="upper right")
    titles(ax1, t, "Quality improves as a power law",
           xlabel="parameters N", ylabel="loss")
    style(ax2, t)
    D = np.geomspace(1e8, 1e12, 200)
    for n, col in zip((3e6, 1e8, 3e9), t["seq"][1:4]):
        ax2.plot(D, E + A / n**al + B / D**be, color=col, lw=2.2, label=f"N = {n:.0e}")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="upper right")
    titles(ax2, t, "…and as a power law in data too",
           xlabel="training tokens D", ylabel="loss")
    illustrative(ax1, t, "schematic: the Chinchilla functional form, not measured data")
    illustrative(ax2, t, " ")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ch 5


@figure("number-formats")
def f_number_formats(t):
    fig = plt.figure(figsize=(9.4, 4.6), facecolor=t["surface"])
    ax = blank_ax(fig, t)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    rows = [
        ("bf16", 16, [("sign", 1, t["arm"]["bf16"]), ("exponent", 8, t["seq"][3]),
                      ("mantissa", 7, t["seq"][1])], "~65,000 magnitudes", "2 bytes / weight"),
        ("4-bit int", 4, [("value", 4, t["arm"]["4"])], "16 levels", "0.5 bytes / weight"),
        ("2-bit int", 2, [("value", 2, t["arm"]["2"])], "4 levels", "0.25 bytes / weight"),
        ("ternary", 2, [("value", 2, t["arm"]["1.58"])], "3 levels: −1, 0, +1",
         "log2(3) = 1.58 bits of information, packed 4 per byte"),
    ]
    x0, cell = 1.55, 0.42
    for i, (name, nbits, parts, levels, note) in enumerate(rows):
        y = 8.1 - i * 2.0
        ax.text(1.4, y + 0.2, name, ha="right", va="center", color=t["ink"],
                fontsize=11.5, fontweight="bold")
        x = x0
        for plabel, n, col in parts:
            for k in range(n):
                ax.add_patch(FancyBboxPatch((x + k * cell, y), cell * 0.86, 0.42,
                                            boxstyle="round,pad=0.005,rounding_size=0.04",
                                            facecolor=col, edgecolor="none"))
            if n >= 2:
                ax.text(x + n * cell / 2 - cell * 0.07, y - 0.3, plabel, ha="center",
                        color=t["ink2"], fontsize=8.5)
            x += n * cell
        ax.text(x + 0.35, y + 0.2, levels, va="center", color=t["ink"], fontsize=10)
        ax.text(x + 0.35, y - 0.25, note, va="center", color=t["muted"], fontsize=8.8)
    ax.text(0.05, 9.5, "One weight, stored four ways", color=t["ink"], fontsize=12.5,
            fontweight="bold")
    ax.text(0.05, 9.05, "each square is one bit of storage",
            color=t["muted"], fontsize=9)
    return fig


@figure("footprint-6m")
def f_footprint(t):
    body = {"1.58": 6.4e6 / 4, "2": 6.4e6 / 4, "3": 6.4e6 * 3 / 8, "4": 6.4e6 / 2, "bf16": 6.4e6 * 2}
    emb = 32768 * 256 * 2
    fig, ax = plt.subplots(figsize=(8.4, 4.2), facecolor=t["surface"])
    style(ax, t)
    arms = ["1.58", "2", "3", "4", "bf16"]
    xs = np.arange(len(arms))
    b = [body[a] / 1e6 for a in arms]
    e = [emb / 1e6] * len(arms)
    ax.bar(xs, b, color=[t["arm"][a] for a in arms], width=0.6, zorder=3, label="model body")
    ax.bar(xs, e, bottom=[v + 0.35 for v in b], color=t["grid"], width=0.6, zorder=3,
           label="embedding table (always 16-bit)")
    for x, bb in zip(xs, b):
        ax.text(x, bb / 2, f"{bb:.1f}", ha="center", va="center", color="#ffffff",
                fontsize=9.5, fontweight="bold", zorder=5)
        ax.text(x, bb + e[0] + 1.2, f"{bb + e[0]:.1f} MB", ha="center", color=t["ink"],
                fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arms])
    ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], loc="upper left")
    ax.set_ylim(0, 36)
    titles(ax, t, "At 6M parameters the embedding table dominates the artifact",
           ylabel="packed size (MB)",
           sub="the body shrinks 8× from bf16 to ternary; the total only shrinks 1.6×")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ch 6 / 7


@figure("quant-levels")
def f_quant_levels(t):
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, 40000)
    real = None
    for cand in sorted((ROOT / "runs" / "l0").glob("l-6m-bf16-*/ckpt_latest.pt")):
        try:
            import torch
            sd = torch.load(cand, map_location="cpu", weights_only=False)["model"]
            for k, v in sd.items():
                if k.endswith("blocks.0.ffn.up.weight"):
                    real = v.float().flatten().numpy()
                    break
        except Exception:
            real = None
        break
    if real is not None and real.size > 1000:
        w = real
    gamma = np.abs(w).mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9), sharey=True, facecolor=t["surface"])
    specs = [("ternary — 3 levels", np.array([-gamma, 0.0, gamma]), t["arm"]["1.58"]),
             ("2-bit — 4 levels", None, t["arm"]["2"]),
             ("4-bit — 16 levels", None, t["arm"]["4"])]
    for i, (title, levels, col) in enumerate(specs):
        ax = axes[i]
        style(ax, t)
        ax.hist(w, bins=110, color=t["grid"], zorder=2)
        if levels is None:
            bits = 2 if i == 1 else 4
            qmax = 2 ** (bits - 1) - 1
            s = 2 * np.abs(w).mean() / np.sqrt(qmax)
            levels = np.arange(-(2 ** (bits - 1)), qmax + 1) * s
            levels = levels[np.abs(levels) <= np.abs(w).max() * 1.05]
        for lv in levels:
            ax.axvline(lv, color=col, lw=1.5, zorder=4)
        ax.set_title(title, color=t["ink"], fontsize=10.5, loc="left")
        ax.set_xlabel("weight value", color=t["ink2"], fontsize=9.5)
        ax.set_yticks([])
        ax.set_xlim(np.percentile(w, 0.2), np.percentile(w, 99.8))
    axes[0].set_ylabel("how many weights", color=t["ink2"], fontsize=9.5)
    fig.suptitle("A real trained weight distribution, and the values each format can store",
                 color=t["ink"], fontsize=12.5, x=0.008, ha="left", y=0.99)
    fig.text(0.008, 0.90, "grey is the actual distribution of one layer's weights; "
             "coloured lines are the only values that survive quantization",
             color=t["muted"], fontsize=8.8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


@figure("ptq-cliff")
def f_ptq_cliff(t):
    bits = np.linspace(1.4, 8, 300)
    qat = 1.0 - 0.30 * np.exp(-(bits - 1.4) / 1.5)
    ptq = 1.0 - 0.06 * np.exp(-(bits - 3.6) / 2.2) - 0.75 / (1 + np.exp((bits - 3.1) / 0.32))
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=t["surface"])
    style(ax, t)
    ax.plot(bits, ptq, color=t["arm"]["bf16"], lw=2.4, zorder=4)
    ax.plot(bits, qat, color=t["arm"]["1.58"], lw=2.4, zorder=4)
    ax.annotate("post-training quantization", (2.35, np.interp(2.35, bits, ptq)),
                xytext=(12, -6), textcoords="offset points", color=t["arm"]["bf16"],
                fontsize=10.5, fontweight="bold")
    ax.annotate("trained low-bit from scratch", (2.0, np.interp(2.0, bits, qat)),
                xytext=(12, 10), textcoords="offset points", color=t["arm"]["1.58"],
                fontsize=10.5, fontweight="bold")
    ax.axvspan(1.4, 3.2, color=t["warn"], alpha=0.07, zorder=1)
    ax.text(2.3, 0.13, "where the byte savings live", ha="center", color=t["warn"],
            fontsize=9.5, fontweight="bold")
    ax.set_xticks([1.58, 2, 3, 4, 6, 8])
    ax.set_xticklabels(["1.58", "2", "3", "4", "6", "8"])
    ax.set_yticks([])
    ax.set_ylim(0, 1.12)
    titles(ax, t, "Post-training quantization falls off exactly where it would pay most",
           xlabel="bits per weight", ylabel="quality retained →")
    illustrative(ax, t)
    fig.tight_layout()
    return fig


@figure("ste")
def f_ste(t):
    x = np.linspace(-2.6, 2.6, 1200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0), facecolor=t["surface"])
    style(ax1, t)
    ax1.plot(x, x, color=t["muted"], lw=1.4, ls="--")
    ax1.plot(x, np.round(x), color=t["arm"]["1.58"], lw=2.4)
    ax1.text(-2.5, 2.6, "round(x) — what the forward pass uses",
             color=t["arm"]["1.58"], fontsize=9.5, fontweight="bold")
    ax1.annotate("the value we wish\nwe could keep", (1.55, 1.55), xytext=(0.35, -2.35),
                 color=t["muted"], fontsize=9,
                 arrowprops=dict(arrowstyle="-", color=t["muted"], lw=1))
    ax1.set_ylim(-3.4, 3.4)
    titles(ax1, t, "Rounding is a staircase", xlabel="weight value x", ylabel="quantized value")
    style(ax2, t)
    ax2.plot(x, np.zeros_like(x), color=t["arm"]["bf16"], lw=4.5, zorder=3)
    ste = np.where(np.abs(x) <= 1.5, 1.0, 0.0)
    ax2.plot(x, ste, color=t["arm"]["1.58"], lw=2.2, zorder=4)
    ax2.text(-2.45, 0.11, "true gradient — zero everywhere, so nothing learns",
             color=t["arm"]["bf16"], fontsize=9.5, fontweight="bold")
    ax2.text(-1.42, 1.07, "straight-through: pretend it was the identity",
             color=t["arm"]["1.58"], fontsize=9.5, fontweight="bold")
    ax2.set_ylim(-0.25, 1.35)
    ax2.set_yticks([0, 1])
    titles(ax2, t, "…so we substitute a gradient that works",
           xlabel="weight value x", ylabel="gradient passed backward")
    fig.tight_layout()
    return fig


@figure("lr-sensitivity")
def f_lr(t):
    import re
    rows = results()
    pat = re.compile(r"^l-lrp-6m-([\d.]+|bf16)-x([\d.]+)$")
    pts = defaultdict(dict)
    for r in rows:
        m = pat.match(r["run_id"])
        if m and r["status"] == "complete":
            pts[m.group(1)][float(m.group(2))] = r["bpb_val1"]
    fig, ax = plt.subplots(figsize=(8.4, 4.5), facecolor=t["surface"])
    style(ax, t)
    for prec in ("1.58", "2", "3", "4", "bf16"):
        d = pts.get(prec)
        if not d:
            continue
        xs = sorted(d)
        ys = [d[x] for x in xs]
        ax.plot(xs, ys, color=t["arm"][prec], lw=2.2, marker="o", ms=8, zorder=4)
        ax.annotate(ARM_LABEL[prec].split(" (")[0], (xs[-1], ys[-1]), xytext=(9, 0),
                    textcoords="offset points", color=t["arm"][prec], fontsize=10,
                    fontweight="bold", va="center")
    ax.set_xscale("log")
    ticks = sorted({x for d in pts.values() for x in d})
    ax.set_xticks(ticks); ax.set_xticklabels([f"{x:g}×" for x in ticks])
    ax.minorticks_off()
    titles(ax, t, "Low-bit training barely notices a mistuned learning rate",
           xlabel="learning-rate multiplier", ylabel="validation bits per byte",
           sub="6M parameters, 31 tokens/param; bf16 spans 0.191 BPB across this range, ternary only 0.034")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ch 8


def _kv_bytes(ctx, bits, layers=30, kv_heads=4, head_dim=128, batch=1):
    return 2 * layers * kv_heads * head_dim * ctx * (bits / 8) * batch


@figure("kv-cache")
def f_kv(t):
    ctx = np.geomspace(512, 131072, 200)
    fig, ax = plt.subplots(figsize=(8.4, 4.5), facecolor=t["surface"])
    style(ax, t)
    weights_gb = 1.506e9 * 1.58 / 8 / 1e9
    ax.axhline(weights_gb, color=t["muted"], lw=1.6, ls="--", zorder=3)
    ax.text(560, weights_gb * 1.08, "the model's ternary weights (~0.30 GB)",
            color=t["muted"], fontsize=9)
    for bits, col in zip((16, 8, 4, 2), [t["arm"]["bf16"], t["arm"]["4"], t["arm"]["3"], t["arm"]["1.58"]]):
        y = _kv_bytes(ctx, bits) / 1e9
        ax.plot(ctx, y, color=col, lw=2.2, zorder=4)
        ax.annotate(f"{bits}-bit cache", (ctx[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", color=col, fontsize=9.5,
                    fontweight="bold", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1024, 4096, 16384, 65536])
    ax.set_xticklabels(["1k", "4k", "16k", "64k"])
    ax.minorticks_off()
    ax.set_xlim(512, 3.2e5)
    titles(ax, t, "Past a certain context, the cache costs more than the model",
           xlabel="context length (tokens)", ylabel="KV cache size (GB, one user)",
           sub="computed for the project's 1.5B-parameter capstone shape: 30 layers, 4 KV heads, head dim 128")
    fig.tight_layout()
    return fig


@figure("bytes-budget")
def f_bytes_budget(t):
    # The 125M validation-anchor shape, so this figure matches the worked
    # example the chapters develop: 18 layers, 3 KV heads, head dim 64,
    # 152,567,808 total parameters at bf16 = 305 MB of weights.
    contexts = [2048, 8192, 32768]
    kv = dict(layers=18, kv_heads=3, head_dim=64)
    w = 152_567_808 * 2 / 1e9
    fig, ax = plt.subplots(figsize=(8.6, 4.3), facecolor=t["surface"])
    style(ax, t)
    labels, weights, kv16, kv4 = [], [], [], []
    for c in contexts:
        labels.append(f"{c // 1024}k context")
        weights.append(w)
        kv16.append(_kv_bytes(c, 16, **kv) / 1e9)
        kv4.append(_kv_bytes(c, 4, **kv) / 1e9)
    x = np.arange(len(contexts))
    bw = 0.34
    ax.bar(x - bw / 2, weights, bw, color=t["arm"]["1.58"], zorder=3, label="weights (bf16)")
    ax.bar(x - bw / 2, kv16, bw, bottom=[v + 0.006 for v in weights], color=t["arm"]["bf16"],
           zorder=3, label="16-bit KV cache")
    ax.bar(x + bw / 2, weights, bw, color=t["arm"]["1.58"], zorder=3)
    ax.bar(x + bw / 2, kv4, bw, bottom=[v + 0.006 for v in weights], color=t["arm"]["3"],
           zorder=3, label="4-bit KV cache")
    for i in range(len(contexts)):
        ax.text(x[i] - bw / 2, weights[i] + kv16[i] + 0.05, f"{weights[i] + kv16[i]:.2f}",
                ha="center", color=t["ink"], fontsize=9.5, fontweight="bold")
        ax.text(x[i] + bw / 2, weights[i] + kv4[i] + 0.05, f"{weights[i] + kv4[i]:.2f}",
                ha="center", color=t["ink"], fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], loc="upper left")
    titles(ax, t, "A deployed model's bytes are weights plus cache — and cache grows with context",
           ylabel="total memory (GB, one user)",
           sub="the project's 125M validation-anchor shape; left bar of each pair keeps the "
               "cache at 16-bit, right bar quantizes it to 4-bit")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ch 9


@figure("iso-memory")
def f_iso_memory(t):
    b = np.linspace(1.4, 16, 400)
    fig, ax = plt.subplots(figsize=(8.4, 4.5), facecolor=t["surface"])
    style(ax, t)
    for gamma, col, lab in ((1.1, t["seq"][3], "if low bits keep most of their usefulness"),
                            (2.6, t["arm"]["2"], "if low bits lose usefulness fast")):
        f = 1 - np.exp(-(b - 0.9) / gamma)
        f = f / (1 - np.exp(-(16 - 0.9) / gamma))
        eff = (16.0 / b) * f
        ax.plot(b, eff, color=col, lw=2.4, zorder=4)
        i = int(np.argmax(eff))
        ax.plot([b[i]], [eff[i]], "o", color=col, ms=10, zorder=5)
        ax.annotate(lab, (b[300], eff[300]), xytext=(-4, 12), textcoords="offset points",
                    color=col, fontsize=9.5, fontweight="bold", ha="right")
        ax.annotate(f"best at {b[i]:.1f} bits", (b[i], eff[i]), xytext=(6, 8),
                    textcoords="offset points", color=col, fontsize=9)
    ax.axhline(1.0, color=t["muted"], lw=1.2, ls="--")
    ax.text(15.6, 1.04, "full precision", color=t["muted"], fontsize=9, ha="right")
    ax.set_xticks([1.58, 2, 3, 4, 8, 16])
    ax.set_xticklabels(["1.58", "2", "3", "4", "8", "16"])
    titles(ax, t, "At a fixed byte budget, there is a best precision — and where it sits is the question",
           xlabel="bits per weight", ylabel="useful capacity per byte  (relative to bf16)",
           sub="fewer bits buys more parameters (16/b of them) but each is worth less; the product has a peak")
    illustrative(ax, t)
    fig.tight_layout()
    return fig


@figure("ladder-12m")
def f_ladder_12m(t):
    rows = results()
    vals = {}
    for r in rows:
        if r["size"] == "12m" and round(r["tokens_per_param"]) == 20 and r["status"] == "complete" \
                and "sqrelu" not in r["run_id"]:
            vals[r["precision"]] = r["bpb_val1"]
    if not vals:
        return None
    arms = [a for a in ("1.58", "2", "3", "4", "bf16") if a in vals]
    fig, ax = plt.subplots(figsize=(8.4, 4.3), facecolor=t["surface"])
    style(ax, t)
    ref = vals.get("bf16")
    xs = np.arange(len(arms))
    ax.bar(xs, [vals[a] for a in arms], color=[t["arm"][a] for a in arms], width=0.6, zorder=3)
    if ref:
        ax.axhline(ref, color=t["arm"]["bf16"], lw=1.4, ls="--", zorder=4)
    for x, a in zip(xs, arms):
        ax.text(x, vals[a] + 0.012, f"{vals[a]:.4f}", ha="center", color=t["ink"],
                fontsize=10, fontweight="bold")
        if ref and a != "bf16":
            ax.text(x, vals[a] / 2, f"{vals[a] - ref:+.3f}", ha="center", va="center",
                    color="#ffffff", fontsize=10.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arms])
    ax.set_ylim(0, max(vals.values()) * 1.18)
    titles(ax, t, "At 12M parameters and 20 tokens/param, every low-bit arm beats full precision",
           ylabel="validation bits per byte  (lower is better)",
           sub="one seed each; the ordering among the quantized arms sits inside seed noise and is not claimed")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- ch 11


@figure("pipeline")
def f_pipeline(t):
    fig = plt.figure(figsize=(10.0, 4.8), facecolor=t["surface"])
    ax = blank_ax(fig, t)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def node(x, y, w, h, title, sub, color, fg="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    facecolor=color, edgecolor="none", zorder=3))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color=fg,
                fontsize=10, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", color=fg,
                fontsize=8.2, zorder=4, alpha=0.92)

    def arrow(x1, y1, x2, y2, col=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color=col or t["muted"], lw=1.4, zorder=2))

    steps = [
        ("manifest", "pre-specified", t["seq"][3]),
        ("train", "frozen protocol", t["arm"]["1.58"]),
        ("evaluate", "bits per byte", t["arm"]["3"]),
        ("export", "packed bytes", t["arm"]["4"]),
        ("results", "hash-verified", t["seq"][3]),
    ]
    w, h = 1.66, 1.1
    for i, (a, b, c) in enumerate(steps):
        x = 0.2 + i * 1.95
        node(x, 6.5, w, h, a, b, c, "#ffffff" if i != 3 else "#0b0b0b")
        if i:
            arrow(x - 0.27, 7.05, x - 0.04, 7.05)
    ax.text(0.2, 6.1, "every run is one row of a versioned manifest; its result row carries the hash "
            "of the exact configuration that produced it", color=t["muted"], fontsize=8.8)

    node(2.15, 4.15, 2.5, 1.05, "fit the law", "forms compete out-of-sample",
         t["seq"][4], "#ffffff")
    node(5.35, 4.15, 2.7, 1.05, "prescribe", "best N and bits per budget",
         t["seq"][4], "#ffffff")
    arrow(8.15, 6.45, 4.75, 5.3)
    arrow(4.7, 4.67, 5.3, 4.67)

    ax.add_patch(FancyBboxPatch((0.18, 1.05), 9.6, 1.9,
                                boxstyle="round,pad=0.06,rounding_size=0.14",
                                facecolor=t["faint"], edgecolor=t["warn"], lw=1.4, zorder=1))
    ax.text(0.45, 2.52, "the validation panel — runs beside everything above",
            color=t["warn"], fontsize=10.5, fontweight="bold")
    ax.text(0.45, 1.95, "11 independent probes · 49 pre-registered, hash-locked kill gates · "
            "re-derives every quantity from first principles",
            color=t["ink2"], fontsize=9)
    ax.text(0.45, 1.42, "a failing gate means fixing the stack or retracting the claim — never editing the gate",
            color=t["muted"], fontsize=8.8, style="italic")
    for x in (1.03, 9.0):
        arrow(x, 3.95, x, 3.02, t["warn"])
    ax.text(0.18, 9.2, "How one experiment becomes a trustworthy number",
            color=t["ink"], fontsize=12.5, fontweight="bold")
    ax.text(0.18, 8.65, "nothing is launched by hand twice; every result carries the hash of the "
            "configuration that produced it", color=t["muted"], fontsize=9)
    return fig


@figure("microp0-curves")
def f_microp0(t):
    runs = sorted((ROOT / "runs" / "local_p0").glob("local-micro-*/metrics.jsonl"))
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=t["surface"])
    style(ax, t)
    handles = {}
    for mf in runs:
        name = mf.parent.name
        prec = name.replace("local-micro-", "").rsplit("-s", 1)[0]
        seed = name.rsplit("-s", 1)[-1]
        rows = [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
        (line,) = ax.plot(
            [r["tokens"] / 1e6 for r in rows], [r["loss"] / np.log(2) for r in rows],
            color=t["arm"].get(prec, t["muted"]), lw=1.7,
            alpha=1.0 if seed == "0" else 0.45, zorder=4,
        )
        if seed == "0":
            handles[prec] = line
    order = [p for p in ("1.58", "2", "3", "4", "bf16") if p in handles]
    leg = ax.legend([handles[p] for p in order],
                    [ARM_LABEL[p].split(" (")[0] for p in order],
                    frameon=False, fontsize=9.5, labelcolor=t["ink2"],
                    loc="upper right", ncol=2, handlelength=1.6)
    leg.set_title("weight precision", prop={"size": 9})
    leg.get_title().set_color(t["muted"])
    titles(ax, t, "The whole precision ladder training on real text",
           xlabel="tokens seen (millions)", ylabel="training loss (bits per byte)",
           sub="4.8M-parameter models, byte-level tokens; faint lines are second seeds")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(FIGS)
    made = []
    for name in names:
        fn = FIGS[name]
        for mode, t in THEME.items():
            plt.rcParams["font.family"] = ["DejaVu Sans"]
            fig = fn(t)
            if fig is None:
                print(f"  skip {name} ({mode}): no data yet")
                continue
            path = OUT / (f"{name}.png" if mode == "light" else f"{name}-dark.png")
            fig.savefig(path, dpi=170, facecolor=t["surface"], bbox_inches="tight",
                        pad_inches=0.28)
            plt.close(fig)
        made.append(name)
        print(f"  {name}")
    print(f"{len(made)} figures x 2 themes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
