"""Paper figures (Agg backend, save-to-file only): the RQ3 iso-memory
frontier (the headline figure), the b*(M, D) phase diagram, the RQ2
gap-vs-D/N curves, and the RQ4 extrapolation check (PLAN.md sections 2, 7-9).

Color follows the precision arm (fixed CVD-validated palette, never cycled);
markers are the secondary encoding. One axis per chart; grids recessive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from logos.fitting.fit import FitResult
from logos.fitting.prescribe import (
    GB,
    DEFAULT_BITS,
    WeightBytesFn,
    default_weight_bytes,
    phase_diagram,
)
from logos.fitting.select import _sigma_for_n

# Fixed precision -> (color, marker). Okabe-Ito subset in this order passes the
# palette validator (lightness band, chroma, CVD >= 8, normal-vision >= 15).
ARM_STYLE: dict[float, tuple[str, str, str]] = {
    1.58: ("#0072B2", "o", "1.58-bit"),
    2.0: ("#E69F00", "s", "2-bit"),
    3.0: ("#009E73", "^", "3-bit"),
    4.0: ("#56B4E9", "D", "4-bit"),
    16.0: ("#CC79A7", "v", "bf16"),
}
_INK, _MUTED, _GRID = "#333333", "#666666", "#dddddd"


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)


def _save(fig: plt.Figure, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


def plot_iso_memory_frontier(
    fit: FitResult,
    out_path: str | Path,
    *,
    D: float,
    bit_options: tuple[float, ...] = DEFAULT_BITS,
    M_range: tuple[float, float] = (0.125 * GB, 4 * GB),
    n_points: int = 64,
    measured_bytes: WeightBytesFn | None = None,
) -> str:
    """THE headline figure (RQ3): x = weight-memory budget (log GB), y =
    fitted L at the largest N each bit width affords, one line per width."""
    wbytes = measured_bytes or default_weight_bytes
    M = np.geomspace(M_range[0], M_range[1], n_points)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ends: list[tuple[float, str]] = []
    y_all: list[float] = []
    for b in bit_options:
        # invert bytes(N, b) <= M for N (monotone; exact for the default transform)
        N = np.array([M_i * 8.0 / b if measured_bytes is None else _invert(wbytes, M_i, b) for M_i in M])
        L = fit.predict(N, np.full_like(N, D), np.full_like(N, b))
        color, marker, label = ARM_STYLE[b]
        ax.plot(M / GB, L, color=color, linewidth=2, marker=marker, markersize=4,
                markevery=max(1, n_points // 8), label=label, zorder=3)
        ends.append((float(L[-1]), label))
        y_all += [float(L.min()), float(L.max())]
    # direct end-of-line labels, pushed apart so close lines stay readable
    min_gap = 0.03 * (max(y_all) - min(y_all) + 1e-12)
    ys = np.array([y for y, _ in sorted(ends)])
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for y_lab, (_, label) in zip(ys, sorted(ends)):
        ax.annotate(label, (M[-1] / GB, y_lab), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=_INK, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("weight-memory budget (GB)", color=_INK)
    ax.set_ylabel("fitted best val BPB", color=_INK)
    ax.set_title(f"Iso-memory frontier at D = {D:.2g} tokens (RQ3)", color=_INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK)
    _style(ax)
    return _save(fig, out_path)


def _invert(wbytes: WeightBytesFn, budget: float, bits: float) -> float:
    lo, hi = np.log(1e5), np.log(1e13)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if wbytes(np.exp(mid), bits) <= budget else (lo, mid)
    return float(np.exp(lo))


def plot_phase_diagram(
    fit: FitResult,
    out_path: str | Path,
    *,
    M_grid: np.ndarray,
    D_grid: np.ndarray,
    bits: tuple[float, ...] = DEFAULT_BITS,
    measured_bytes: WeightBytesFn | None = None,
    iso_flops: bool = True,
) -> str:
    """b*(M, D) heatmap (RQ3). b* is ordered, so a single-hue sequential ramp
    (light -> dark) with one discrete step per bit width.

    iso_flops (kimi3 review F4): overlays dashed iso-training-compute
    contours, C ~= 6*N*(M,b*)*D with N* = M*8/b*. At fixed bytes the
    big-N-low-bit corner costs multiples of the small-N-high-bit corner in
    training FLOPs; the prescription is deployment-optimal, and this overlay
    is the honest disclosure of what it costs to train."""
    bstar = phase_diagram(fit, M_grid, D_grid, bits, measured_bytes=measured_bytes)
    opts = sorted(bits)
    idx = np.searchsorted(opts, bstar)  # ordered category index
    ramp = ListedColormap([plt.cm.Blues(x) for x in np.linspace(0.25, 0.95, len(opts))])
    norm = BoundaryNorm(np.arange(len(opts) + 1) - 0.5, len(opts))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mesh = ax.pcolormesh(D_grid, np.asarray(M_grid) / GB, idx, cmap=ramp, norm=norm, shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training tokens D", color=_INK)
    ax.set_ylabel("weight-memory budget (GB)", color=_INK)
    ax.set_title("Optimal weight precision b*(M, D)", color=_INK)
    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(opts)))
    cbar.ax.set_yticklabels([ARM_STYLE.get(b, (None, None, f"{b:g}-bit"))[2] for b in opts],
                            color=_INK)
    cbar.outline.set_visible(False)
    if iso_flops:
        MM, DD = np.meshgrid(np.asarray(M_grid, dtype=float), np.asarray(D_grid, dtype=float),
                             indexing="ij")
        C = 6.0 * (MM * 8.0 / bstar) * DD  # training FLOPs at the prescribed arm
        levels = 10.0 ** np.arange(
            np.ceil(np.log10(C.min())), np.floor(np.log10(C.max())) + 1
        )
        cs = ax.contour(DD, MM / GB, C, levels=levels, colors=_MUTED,
                        linewidths=1.0, linestyles="--")
        ax.clabel(cs, fmt=lambda v: f"{v:.0e} FLOPs", fontsize=7, colors=_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)
    return _save(fig, out_path)


def plot_gap_curves(df: pd.DataFrame, out_path: str | Path) -> str:
    """RQ2 figure: delta BPB vs bf16 as a function of tokens/param, one line
    per quantized width, small multiples per ladder size (shared axes)."""
    sizes = list(df.groupby("size")["n_nonemb"].mean().sort_values().index)
    fig, axes = plt.subplots(1, len(sizes), figsize=(2.9 * len(sizes), 3.4),
                             sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        sub = df[df["size"] == size]
        cell = sub.groupby(["bits", "tokens"], as_index=False).agg(
            bpb=("bpb", "mean"), n_nonemb=("n_nonemb", "mean"))
        ref = cell[np.isclose(cell["bits"], 16.0)].set_index("tokens")["bpb"]
        for b in sorted(set(np.round(cell["bits"], 4))):
            if np.isclose(b, 16.0):
                continue
            arm = cell[np.isclose(cell["bits"], b)].sort_values("tokens")
            gap = arm["bpb"].to_numpy() - ref.loc[arm["tokens"]].to_numpy()
            dn = arm["tokens"].to_numpy() / arm["n_nonemb"].to_numpy()
            color, marker, label = ARM_STYLE[float(b)]
            ax.plot(dn, gap, color=color, marker=marker, markersize=5, linewidth=2, label=label)
        ax.axhline(0.0, color=_MUTED, linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(size, color=_INK, fontsize=10)
        ax.set_xlabel("tokens / param", color=_INK, fontsize=9)
        _style(ax)
    axes[0][0].set_ylabel("delta BPB vs bf16", color=_INK)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=_INK)
    fig.suptitle("Low-bit gap vs overtraining (RQ2)", color=_INK)
    return _save(fig, out_path)


def plot_extrapolation_check(
    fit: FitResult,
    df_holdout: pd.DataFrame,
    sigma: dict | float,
    out_path: str | Path,
) -> str:
    """RQ4 figure: predicted vs observed BPB on the held-out size, with the
    2x seed-sigma acceptance band around the diagonal."""
    cells = (
        df_holdout.groupby(["tokens", "precision"])
        .agg(n_nonemb=("n_nonemb", "mean"), bits=("bits", "mean"), bpb=("bpb", "mean"))
        .reset_index()
    )
    pred = np.asarray(fit.predict(cells["n_nonemb"], cells["tokens"], cells["bits"]))
    sig = _sigma_for_n(sigma, cells["n_nonemb"].to_numpy())
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(cells["bpb"].min(), pred.min())
    hi = max(cells["bpb"].max(), pred.max())
    pad = 0.06 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, 2)
    band = float(2 * sig.max())
    ax.fill_between(xs, xs - band, xs + band, color=_GRID, alpha=0.6, zorder=1,
                    label=f"2x seed-sigma band (+/-{band:.3f})")
    ax.plot(xs, xs, color=_MUTED, linewidth=1, zorder=2)
    for _, row in cells.iterrows():
        b = float(np.round(row["bits"], 4))
        color, marker, _ = ARM_STYLE[b]
        ax.scatter(row["bpb"], pred[row.name], s=48, color=color, marker=marker,
                   edgecolors="white", linewidths=1.2, zorder=3)
    handles = [plt.Line2D([], [], color=ARM_STYLE[b][0], marker=ARM_STYLE[b][1],
                          linestyle="", label=ARM_STYLE[b][2])
               for b in sorted(set(np.round(cells["bits"], 4)))]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], frameon=False,
              fontsize=8, labelcolor=_INK, loc="upper left")
    ax.set_xlabel("observed BPB (held-out size)", color=_INK)
    ax.set_ylabel("predicted BPB", color=_INK)
    ax.set_title(f"Extrapolation check, form {fit.form_name} (RQ4)", color=_INK)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(xs[0], xs[-1])
    ax.set_aspect("equal")
    _style(ax)
    return _save(fig, out_path)
