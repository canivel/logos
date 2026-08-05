"""Boring fallback for the Tier 2 Eval & hygiene agent (PLAN.md s13/s15).

Find complete runs that lack a results row, then run the eval_and_archive
pipeline for each. Idempotent and cron-safe: a run already present in the
results store (or in the local ops/evaluated.txt ledger when the store is
not importable yet) is skipped.

crontab (5090 box, hourly):
    0 * * * * cd /path/to/logos && python ops/fallback/eval_on_complete.py /data/runs >> ops/eval_cron.log 2>&1

Windows Task Scheduler equivalent:
    schtasks /Create /SC HOURLY /TN "logos-eval" /TR "python F:\\Research\\logos\\ops\\fallback\\eval_on_complete.py F:\\Research\\logos\\runs"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:  # allow `python ops/fallback/eval_on_complete.py`
    sys.path.insert(0, str(_REPO))

from ops.kaos import tools  # noqa: E402

ACTOR = "fallback.eval_on_complete"


def evaluated_run_ids(ops_dir: str | Path | None = None) -> set[str]:
    """run_ids that already have a results row (guarded logos.results.store),
    unioned with the local ops/evaluated.txt ledger."""
    done: set[str] = set()
    try:
        from logos.results.store import load_results  # type: ignore

        rows = load_results()
        try:  # DataFrame-shaped
            done |= set(map(str, rows["run_id"]))  # type: ignore[index]
        except Exception:
            try:  # list-of-dicts shaped
                done |= {str(r.get("run_id")) for r in rows}  # type: ignore[union-attr]
            except Exception:
                pass
    except Exception:
        pass
    ledger = tools._resolve_ops_dir(ops_dir) / "evaluated.txt"
    if ledger.exists():
        done |= {ln.strip() for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()}
    return done


def _mark_evaluated(run_id: str, ops_dir: str | Path | None = None) -> None:
    ledger = tools._resolve_ops_dir(ops_dir) / "evaluated.txt"
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(run_id + "\n")


def tick(
    runs_dir: str | Path,
    *,
    dry_run: bool = False,
    hf_repo: str | None = None,
    ops_dir: str | Path | None = None,
) -> list[str]:
    """One pass: eval every complete-but-unevaluated run. Returns run_ids."""
    done = evaluated_run_ids(ops_dir)
    processed: list[str] = []
    for h in tools.scan_runs(runs_dir):
        if h.state != "complete" or h.run_id in done:
            continue
        tools.eval_and_archive(
            h.run_id, runs_dir,
            dry_run=dry_run, hf_repo=hf_repo,
            actor=ACTOR, tier=2, ops_dir=ops_dir,
        )
        if not dry_run:
            _mark_evaluated(h.run_id, ops_dir)
        processed.append(h.run_id)
    return processed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs_dir", help="directory holding <run_id>/ subdirs")
    p.add_argument("--dry-run", action="store_true",
                   help="record what would run; do not eval, archive, or mark")
    p.add_argument("--hf-repo", default=None,
                   help="HF repo id for checkpoint archival (HF_TOKEN)")
    p.add_argument("--ops-dir", default=None,
                   help="override ops/ dir for audit.jsonl and evaluated.txt")
    args = p.parse_args(argv)

    processed = tick(args.runs_dir, dry_run=args.dry_run,
                     hf_repo=args.hf_repo, ops_dir=args.ops_dir)
    print(f"[eval_on_complete] processed={processed or 'nothing to do'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
