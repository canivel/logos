"""Run manifest queues back-to-back on one GPU, unattended.

The local program is a single-GPU pipeline: only one queue may train at a
time, but a phase should not sit idle waiting for a human to notice the
previous one finished. This waits for an already-running queue to print
"queue drained" into its log, then runs the listed queues in order.

Each queue is `scripts/run_queue.py`, which skips runs that already have a
results row, so re-running a chain is always safe.

    python scripts/chain_queues.py \
        --wait-for logs/l1_queue.log \
        --queue manifests/l1lrx.yaml:data/fineweb_edu_10bt:runs/l1lrx \
        --queue manifests/l2.yaml:data/fineweb_edu_10bt:runs/l2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONE_MARK = "queue drained"


def wait_for(log_path: Path, poll_s: int = 60, timeout_s: int = 14 * 24 * 3600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if log_path.exists() and DONE_MARK in log_path.read_text(errors="ignore"):
            return
        time.sleep(poll_s)
    raise TimeoutError(f"{log_path} never reported '{DONE_MARK}'")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait-for", default=None, help="log file of a queue already running")
    ap.add_argument(
        "--queue", action="append", required=True,
        help="manifest:data_dir:runs_dir (repeatable, run in order)",
    )
    ap.add_argument("--micro-batch-seqs", type=int, default=8)
    args = ap.parse_args()

    if args.wait_for:
        print(f"waiting for {args.wait_for} to drain...", flush=True)
        wait_for(Path(args.wait_for))
        print("previous queue drained", flush=True)

    for spec in args.queue:
        manifest, data_dir, runs_dir = spec.split(":", 2)
        name = Path(manifest).stem
        log = ROOT / "logs" / f"{name}_queue.log"
        err = ROOT / "logs" / f"{name}_queue.err"
        print(f"=== starting {name} ===", flush=True)
        with open(log, "a", encoding="utf-8") as fo, open(err, "a", encoding="utf-8") as fe:
            rc = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "run_queue.py"),
                    "--manifest", manifest, "--data-dir", data_dir, "--runs-dir", runs_dir,
                    "--micro-batch-seqs", str(args.micro_batch_seqs),
                ],
                stdout=fo, stderr=fe, cwd=ROOT,
            ).returncode
        print(f"=== {name} exited rc={rc} ===", flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
