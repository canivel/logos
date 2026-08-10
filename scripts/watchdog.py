"""Own the local training queue and restart it if the trainer wedges.

Why this exists: on 2026-08-09 the 12M ternary run at 80 tokens/param hung
with the GPU pinned at 100% utilisation, VRAM at 97%, and the process
holding a 10.5 GB host working set — the Windows WDDM signature of GPU
memory spilling into system RAM. No exception, no exit code, no progress:
metrics stopped for 45 minutes while everything looked healthy from the
outside. A crash the queue could have recovered from; a hang it could not.

So the watchdog does not inspect the trainer. It watches the one thing that
proves forward progress — the mtime of the newest metrics.jsonl — and if
that goes stale while the child is alive, it kills the child and relaunches.
The trainer checkpoints every 30 minutes and resumes exactly, and the queue
skips runs that already have a results row, so a restart costs at most the
work since the last checkpoint.

    python scripts/watchdog.py --runs-dir runs \
        --queue manifests/l1.yaml:data/fineweb_edu_10bt:runs/l1 \
        --queue manifests/l1lrx.yaml:data/fineweb_edu_10bt:runs/l1lrx \
        --micro-batch-seqs 4

Stall threshold defaults to 25 minutes, comfortably above the slowest
observed step time (5.4 s) times the 20-step log interval, and above a
checkpoint write.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def newest_progress(runs_dir: Path) -> tuple[float, Path | None]:
    """(mtime, path) of the most recent sign of forward progress.

    Not just metrics.jsonl. After the last training step the queue spends
    minutes in evaluation — bits-per-byte over two held-out sets, artifact
    export, parity check — and writes no metrics at all during it. Watching
    metrics alone would count that healthy stretch as a stall, kill the run,
    and then do it again on every restart, because a resumed finished run
    goes straight back into evaluation. So status.json (written when a run
    completes) and the results file (written when its evaluation lands) both
    count as progress.
    """
    best_t, best_p = 0.0, None
    candidates = list(runs_dir.rglob("metrics.jsonl"))
    candidates += list(runs_dir.rglob("status.json"))
    candidates.append(ROOT / "results" / "results.jsonl")
    for m in candidates:
        try:
            t = m.stat().st_mtime
        except OSError:
            continue
        if t > best_t:
            best_t, best_p = t, m
    return best_t, best_p


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def spawn(queues: list[str], micro: int) -> subprocess.Popen:
    cmd = [sys.executable, str(ROOT / "scripts" / "chain_queues.py"),
           "--micro-batch-seqs", str(micro)]
    for q in queues:
        cmd += ["--queue", q]
    log(f"starting queue chain: {' '.join(q.split(':')[0] for q in queues)}")
    return subprocess.Popen(
        cmd, cwd=ROOT,
        stdout=open(ROOT / "logs" / "chain.log", "a", encoding="utf-8"),
        stderr=open(ROOT / "logs" / "chain.err", "a", encoding="utf-8"),
    )


def kill_tree(proc: subprocess.Popen) -> None:
    """The chain spawns run_queue, which owns the CUDA context; killing only
    the parent would orphan the wedged trainer and keep the GPU busy."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        import signal
        try:
            proc.send_signal(signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--queue", action="append", required=True,
                    help="manifest:data_dir:runs_dir (repeatable, in order)")
    ap.add_argument("--micro-batch-seqs", type=int, default=4)
    # 45 min: comfortably longer than the slowest evaluation expected in the
    # grid (a 60M model scoring two ~24M-token held-out sets), while still
    # catching a hang well inside one checkpoint interval's worth of lost work.
    ap.add_argument("--stall-minutes", type=float, default=45.0)
    ap.add_argument("--poll-seconds", type=int, default=180)
    ap.add_argument("--max-restarts", type=int, default=12)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    (ROOT / "logs").mkdir(exist_ok=True)
    proc = spawn(args.queue, args.micro_batch_seqs)
    restarts = 0
    # Give the first run time to write its first metrics line.
    grace_until = time.time() + args.stall_minutes * 60

    while True:
        time.sleep(args.poll_seconds)
        rc = proc.poll()
        if rc is not None:
            log(f"queue chain exited rc={rc} — watchdog done")
            return rc

        mtime, path = newest_progress(runs_dir)
        now = time.time()
        if now < grace_until:
            continue
        stale_min = (now - mtime) / 60 if mtime else 999
        if stale_min < args.stall_minutes:
            continue

        if restarts >= args.max_restarts:
            log(f"STALL again ({stale_min:.0f} min) but hit --max-restarts "
                f"{args.max_restarts}; leaving it alone for a human")
            return 2
        restarts += 1
        log(f"STALL: no progress for {stale_min:.0f} min "
            f"(newest: {path}); killing and restarting "
            f"[restart {restarts}/{args.max_restarts}]")
        kill_tree(proc)
        time.sleep(20)  # let the driver release VRAM
        proc = spawn(args.queue, args.micro_batch_seqs)
        grace_until = time.time() + args.stall_minutes * 60


if __name__ == "__main__":
    raise SystemExit(main())
