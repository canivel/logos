"""Stop the training pipeline the moment the run in flight finishes.

The `STOP` file makes `run_queue.py` and `chain_queues.py` wind down between
runs, but a process that is already running has its loop in memory and will
never look at it. So when the stop is requested mid-run, this watches for the
in-flight run's row to appear in the results file — which happens after its
evaluation and export, i.e. when the run is genuinely complete and nothing is
lost — and only then kills the process tree.

    python scripts/stop_after_current.py --run-id l-12m-1.58-tp320-s0

Leaves the STOP file in place, so nothing restarts until it is deleted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.jsonl"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def has_result(run_id: str) -> bool:
    if not RESULTS.exists():
        return False
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line)["run_id"] == run_id:
            return True
    return False


def pipeline_pids() -> list[int]:
    """PIDs of the watchdog, chain and queue, parents first so killing the
    tree from the top catches the trainer holding the CUDA context."""
    out = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/format:csv"],
        capture_output=True, text=True,
    ).stdout
    found = []
    for name in ("watchdog.py", "chain_queues.py", "run_queue.py"):
        for line in out.splitlines():
            if name in line:
                pid = line.strip().rsplit(",", 1)[-1]
                if pid.isdigit():
                    found.append((name, int(pid)))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="the run currently in flight")
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--timeout-hours", type=float, default=36.0)
    args = ap.parse_args()

    (ROOT / "STOP").write_text(
        f"Requested {datetime.now():%Y-%m-%d %H:%M}. Finishing {args.run_id}, then stopping.\n"
        "Delete this file and relaunch scripts/watchdog.py to resume.\n",
        encoding="utf-8",
    )
    log(f"STOP file written; waiting for {args.run_id} to complete")

    deadline = time.time() + args.timeout_hours * 3600
    while time.time() < deadline:
        if has_result(args.run_id):
            log(f"{args.run_id} completed and evaluated")
            for name, pid in pipeline_pids():
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
                log(f"stopped {name} (pid {pid})")
            log("pipeline stopped. GPU is free. Delete STOP and relaunch the watchdog to resume.")
            return 0
        time.sleep(args.poll_seconds)

    log("timed out waiting; leaving everything running and the STOP file in place")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
