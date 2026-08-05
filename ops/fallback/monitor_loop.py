"""Boring fallback for the Tier 1 Monitor agent (PLAN.md s13/s15).

scan_runs -> auto-resume preempted/stalled -> alert on nan/diverged.
No kaos, no LLM: plain Python calling ops.kaos.tools, safe to run from cron.

Run forever (control plane = the 5090 box):
    python ops/fallback/monitor_loop.py /data/runs --interval 600

One shot (for cron / Task Scheduler):
    python ops/fallback/monitor_loop.py /data/runs --once

crontab (Linux 5090 box, every 10 minutes):
    */10 * * * * cd /path/to/logos && python ops/fallback/monitor_loop.py /data/runs --once >> ops/monitor_cron.log 2>&1

Windows Task Scheduler (dev box) equivalent:
    schtasks /Create /SC MINUTE /MO 10 /TN "logos-monitor" /TR "python F:\\Research\\logos\\ops\\fallback\\monitor_loop.py F:\\Research\\logos\\runs --once"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:  # allow `python ops/fallback/monitor_loop.py`
    sys.path.insert(0, str(_REPO))

from ops.kaos import tools  # noqa: E402

ACTOR = "fallback.monitor_loop"


def tick(
    runs_dir: str | Path,
    *,
    manifest: str | Path | None = None,
    dry_run: bool = False,
    ops_dir: str | Path | None = None,
    now: float | None = None,
) -> list[tools.RunHealth]:
    """One monitor pass. Returns the health list for logging/tests."""
    health = tools.scan_runs(runs_dir, now=now)
    for h in health:
        if h.state in ("preempted", "stalled"):
            res = tools.resume_run(
                h.run_id, runs_dir, manifest,
                dry_run=dry_run, actor=ACTOR, tier=1, ops_dir=ops_dir,
            )
            tools.alert(
                f"auto-resume {h.run_id}: {h.state} ({h.detail}); "
                f"resumed={res.get('resumed')}",
                level="info", actor=ACTOR, tier=1, ops_dir=ops_dir,
            )
        elif h.state in ("nan", "diverged"):
            tools.alert(
                f"run {h.run_id} is {h.state}: {h.detail} "
                f"(last step={h.last_step}, loss={h.last_loss})",
                level="critical", actor=ACTOR, tier=1, ops_dir=ops_dir,
            )
    return health


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs_dir", help="directory holding <run_id>/ subdirs")
    p.add_argument("--manifest", default=None, help="manifest yaml for resume")
    p.add_argument("--interval", type=float, default=600.0,
                   help="seconds between passes (loop mode)")
    p.add_argument("--once", action="store_true", help="single pass, for cron")
    p.add_argument("--dry-run", action="store_true",
                   help="classify + alert but do not actually resume")
    p.add_argument("--ops-dir", default=None,
                   help="override ops/ dir for audit.jsonl and alerts.log")
    args = p.parse_args(argv)

    while True:
        health = tick(
            args.runs_dir, manifest=args.manifest,
            dry_run=args.dry_run, ops_dir=args.ops_dir,
        )
        counts: dict[str, int] = {}
        for h in health:
            counts[h.state] = counts.get(h.state, 0) + 1
        print(f"[monitor] {time.strftime('%H:%M:%S')} {counts or 'no runs'}")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
