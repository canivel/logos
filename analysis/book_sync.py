"""Keep the book's numbers in step with the data, then rebuild the site.

The rule this enforces: **numbers are generated, interpretation is written.**
Any figure that a reader might check against `results/results.jsonl` — run
counts, gap tables, ladder rows — lives between AUTO markers in the Markdown
and is rewritten from the results file on every sync. The prose around those
blocks stays hand-written, because a script cannot decide what a result
means or whether it clears the noise bar.

    python analysis/book_sync.py            # sync, regenerate figures, rebuild
    python analysis/book_sync.py --check    # report drift, change nothing

Markers look like:

    <!-- AUTO:name -->
    ...generated...
    <!-- /AUTO:name -->

A marker with no generator, or a generator with no marker, is an error —
that way a block cannot silently stop being maintained.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
RESULTS = ROOT / "results" / "results.jsonl"

ARM_NAME = {"1.58": "ternary", "2": "2-bit", "3": "3-bit", "4": "4-bit", "bf16": "bf16"}
MINUS = "−"  # typographic minus, matching the book's hand-written prose


def signed(x: float, places: int = 4) -> str:
    return f"{x:+.{places}f}".replace("-", MINUS)
GRID_SIZES = ("3m", "6m", "12m", "25m", "60m")


def load() -> list[dict]:
    return [json.loads(x) for x in RESULTS.read_text().splitlines() if x.strip()]


def grid_cells(rows: list[dict]) -> dict[tuple[str, str, int], list[float]]:
    """(size, precision, tokens/param) -> BPB values, grid arms only.
    Excludes LR probes, LR controls and the FFN ablation, which are not
    grid points and would corrupt a per-cell mean."""
    cells: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        rid = r["run_id"]
        if not rid.startswith("l-") or r["status"] != "complete":
            continue
        if any(t in rid for t in ("lrp", "ctl", "sqrelu")):
            continue
        cells[(r["size"], r["precision"], round(r["tokens_per_param"]))].append(r["bpb_val1"])
    return cells


# --------------------------------------------------------------- generators


def gen_run_count(rows, cells) -> str:
    return str(len([r for r in rows if r["status"] == "complete"]))


def gen_l0_gap_table(rows, cells) -> str:
    """Ternary-vs-bf16 with the 2-sigma verdict, every size that has both."""
    out = [
        "| Model | Tokens/param | Ternary BPB | bf16 BPB | Gap | 2σ bar | Verdict |",
        "|-------|-------------:|------------:|---------:|----:|-------:|---------|",
    ]
    for size in GRID_SIZES:
        for tpp in (20, 80, 320):
            lo = cells.get((size, "1.58", tpp), [])
            hi = cells.get((size, "bf16", tpp), [])
            if not lo or not hi:
                continue
            gap = st.mean(lo) - st.mean(hi)
            n_seed = min(len(lo), len(hi))
            if n_seed > 1:
                bar = 2 * max(st.stdev(lo), st.stdev(hi))
                bar_s = f"{bar:.4f}"
                verdict = ("ternary better" if gap < -bar else
                           "bf16 better" if gap > bar else "inside the noise")
            else:
                bar_s = "one seed only"
                verdict = ("ternary better" if gap < 0 else "bf16 better") + " (unbarred)"
            out.append(
                f"| {size.upper()} | {tpp}× | {st.mean(lo):.4f} | {st.mean(hi):.4f} | "
                f"{signed(gap)} | {bar_s} | {verdict} |"
            )
    return "\n".join(out)


def _ladder_table(cells, size: str, tpp: int) -> str | None:
    cell = {p: st.mean(v) for (s, p, t), v in cells.items() if s == size and t == tpp}
    if len(cell) < 2:
        return None
    ref = cell.get("bf16")
    rows = ["| Arm | Validation BPB | vs bf16 |", "|-----|---------------:|--------:|"]
    for p, v in sorted(cell.items(), key=lambda kv: kv[1]):
        d = signed(v - ref, 3) if ref is not None and p != "bf16" else "—"
        rows.append(f"| {ARM_NAME.get(p, p)} | {v:.4f} | {d} |")
    return "\n".join(rows)


def gen_ladder_12m_20x(rows, cells) -> str:
    return _ladder_table(cells, "12m", 20) or "_(no runs recorded yet)_"


def gen_ladder_12m_80x(rows, cells) -> str:
    return _ladder_table(cells, "12m", 80) or "_(these runs are still training)_"


def gen_coverage(rows, cells) -> str:
    """Which grid cells exist, so a reader can see what is and is not measured."""
    out = ["| Size | 20× | 80× | 320× |", "|------|-----|-----|------|"]
    any_row = False
    for size in GRID_SIZES:
        marks = []
        for tpp in (20, 80, 320):
            arms = sorted({p for (s, p, t) in cells if s == size and t == tpp},
                          key=lambda p: (p == "bf16", p))
            marks.append(f"{len(arms)} arms" if arms else "—")
        if any(m != "—" for m in marks):
            any_row = True
            out.append(f"| {size.upper()} | " + " | ".join(marks) + " |")
    return "\n".join(out) if any_row else "_(no grid runs yet)_"


GENERATORS = {
    "run-count": gen_run_count,
    "l0-gap-table": gen_l0_gap_table,
    "ladder-12m-20x": gen_ladder_12m_20x,
    "ladder-12m-80x": gen_ladder_12m_80x,
    "coverage": gen_coverage,
}

MARKER_RE = re.compile(
    r"(<!-- AUTO:(?P<name>[a-z0-9-]+) -->\n)(?P<body>.*?)(<!-- /AUTO:(?P=name) -->)",
    re.S,
)


def sync(check_only: bool = False) -> tuple[int, list[str]]:
    rows = load()
    cells = grid_cells(rows)
    changed: list[str] = []
    seen: set[str] = set()

    for md in sorted(BOOK.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        if "<!-- AUTO:" not in text:
            continue

        def repl(m):
            name = m.group("name")
            seen.add(name)
            if name not in GENERATORS:
                raise SystemExit(f"{md.name}: no generator for AUTO block '{name}'")
            fresh = GENERATORS[name](rows, cells)
            if m.group("body").strip() != fresh.strip():
                changed.append(f"{md.name}:{name}")
            return f"{m.group(1)}{fresh}\n{m.group(4)}"

        new = MARKER_RE.sub(repl, text)
        if new != text and not check_only:
            md.write_text(new, encoding="utf-8")

    unused = set(GENERATORS) - seen
    if unused:
        raise SystemExit(f"generators with no marker in the book: {sorted(unused)}")
    return len(rows), changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report drift, change nothing")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    n_rows, changed = sync(check_only=args.check)
    print(f"{n_rows} result rows")
    if changed:
        print(("drift in " if args.check else "updated ") + f"{len(changed)} block(s):")
        for c in changed:
            print("   ", c)
    else:
        print("all AUTO blocks already current")
    if args.check:
        return 1 if changed else 0

    env = {"PYTHONPATH": str(ROOT / "src")}
    import os
    env = {**os.environ, **env}
    if not args.skip_figures:
        subprocess.run([sys.executable, str(ROOT / "analysis" / "book_figures.py")],
                       cwd=ROOT, env=env, check=True, capture_output=True)
        print("figures regenerated")
    subprocess.run([sys.executable, str(ROOT / "book" / "build.py")],
                   cwd=ROOT, env=env, check=True, capture_output=True)
    print("site rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
