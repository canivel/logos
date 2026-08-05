"""Queue-driven launcher (PLAN.md s7 execution notes, s13 Tier 3).

Reads a versioned manifest, launches pending runs cheapest-first through a
pluggable executor, and enforces the manifest kill criteria (divergence or
> max_wall_clock_mult x scheduled wall clock) with an audit trail. Dry-run
returns the decision record with zero side effects.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

from logos.config import LADDER, RunSpec
from logos.manifest.ledger import BudgetExceeded, BudgetLedger
from logos.manifest.schema import load_manifest

_SIZE_ORDER = list(LADDER)  # ladder is ordered small -> large

# 250m+ trains on 8x H100 SXM spot nodes; everything <=125m on 1x community
# H100 (PLAN.md s7 execution notes).
MULTI_NODE_SIZES = {"250m", "490m", "1b", "1.5b"}


def assumed_gpus(size: str) -> int:
    return 8 if size in MULTI_NODE_SIZES else 1


class Executor(Protocol):
    def launch(self, spec: RunSpec, manifest_path: Path, run_dir: Path) -> Any: ...


class LocalExecutor:
    """Runs the trainer as a subprocess; returns the return code."""

    def launch(self, spec: RunSpec, manifest_path: Path, run_dir: Path) -> int:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "logos.cli",
                "train",
                "--manifest",
                str(manifest_path),
                "--run-id",
                spec.run_id,
            ],
            capture_output=True,
            text=True,
        )
        return proc.returncode


class RunPodExecutor:
    """STUB: builds the exact RunPod API payload but never calls the API.
    Actual launches go through the KAOS ops layer (PLAN.md s13, Tier 3) so
    there is a single audited entry point."""

    IMAGE = "logos-train:latest"

    def build_payload(self, spec: RunSpec, manifest_path: Path) -> dict[str, Any]:
        if spec.size in MULTI_NODE_SIZES:
            pod = {
                "gpu_type": "H100 SXM",
                "gpu_count": 8,
                "cloud": "secure",
                "interruptible": True,  # spot with 30-min checkpointing
            }
        else:
            pod = {"gpu_type": "H100 PCIe", "gpu_count": 1, "cloud": "community", "interruptible": True}
        return {
            "name": spec.run_id,
            "image": self.IMAGE,
            **pod,
            "volume": "logos-network-volume",
            "docker_args": (
                f"python -m logos.cli train --manifest {manifest_path.name} "
                f"--run-id {spec.run_id}"
            ),
            "env": {"LOGOS_RUN_ID": spec.run_id, "LOGOS_PHASE": spec.phase},
        }

    def launch(self, spec: RunSpec, manifest_path: Path, run_dir: Path) -> dict[str, Any]:
        self.build_payload(spec, manifest_path)
        raise NotImplementedError("launch via ops layer")


class Launcher:
    """Queue driver over one manifest: next_pending() -> launch_next()."""

    def __init__(
        self,
        manifest_path: str | Path,
        runs_dir: str | Path,
        ledger: BudgetLedger | None = None,
        executor: Executor | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.runs_dir = Path(runs_dir)
        self.ledger = ledger
        self.executor = executor
        self.meta, self.specs = load_manifest(self.manifest_path)

    # ---- run state --------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def status(self, run_id: str) -> dict[str, Any]:
        p = self.run_dir(run_id) / "status.json"
        if not p.exists():
            return {"status": "pending"}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "pending"}

    def _write_status(self, run_id: str, payload: dict[str, Any]) -> None:
        d = self.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- queue ------------------------------------------------------------

    def _order_key(self, s: RunSpec) -> tuple:
        return (s.est_cost_usd, _SIZE_ORDER.index(s.size), s.tokens_per_param, s.seed, s.run_id)

    def next_pending(self) -> RunSpec | None:
        """Cheapest first (cost, then size, tokens/param, seed); skips runs
        whose status.json says complete/running/killed."""
        for spec in sorted(self.specs, key=self._order_key):
            if self.status(spec.run_id).get("status") in ("complete", "running", "killed"):
                continue
            return spec
        return None

    def launch_next(self, dry_run: bool = True) -> dict[str, Any]:
        """One scheduling decision. Dry-run: decision record, no side effects."""
        spec = self.next_pending()
        if spec is None:
            return {
                "run_id": None,
                "action": "idle",
                "reason": "no pending runs in manifest",
                "requires_human": False,
                "est_cost": 0.0,
            }
        check = (
            self.ledger.check(spec)
            if self.ledger is not None
            else {"ok": True, "reason": "no ledger attached", "requires_human": False}
        )
        record = {
            "run_id": spec.run_id,
            "action": "launch",
            "reason": check["reason"],
            "requires_human": check["requires_human"],
            "est_cost": spec.est_cost_usd,
        }
        if not check["ok"]:
            record["action"] = "blocked_budget"
            return record
        if check["requires_human"]:
            # Tier-3 gate: >$200 or 8x-node launch needs human approval.
            record["action"] = "hold_for_human"
            record["reason"] = ">$200 or 8x-node launch: human approval required (PLAN.md s13)"
            return record
        if dry_run:
            record["action"] = "would_launch"
            return record
        try:
            if self.ledger is not None:
                self.ledger.check_and_reserve(spec)
        except BudgetExceeded as e:
            record["action"] = "blocked_budget"
            record["reason"] = str(e)
            return record
        self._write_status(
            spec.run_id,
            {
                "status": "running",
                "started_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "manifest": self.manifest_path.name,
            },
        )
        if self.executor is not None:
            record["result"] = self.executor.launch(spec, self.manifest_path, self.run_dir(spec.run_id))
        record["action"] = "launched"
        return record

    # ---- kill criteria (Tier-2 enforcement, PLAN.md s13) -------------------

    def enforce_kill_criteria(self) -> list[dict[str, Any]]:
        """Scan running runs' metrics.jsonl for divergence or wall-clock blowout
        (> max_wall_clock_mult x scheduled, scheduled = est_gpu_hours * 3600 /
        assumed gpus). Writes <run_dir>/KILL + status killed with audit reason."""
        killed: list[dict[str, Any]] = []
        for spec in self.specs:
            if self.status(spec.run_id).get("status") != "running":
                continue
            reason = self._kill_reason(spec)
            if reason is None:
                continue
            ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            audit = {"run_id": spec.run_id, "action": "kill", "reason": reason, "ts": ts}
            (self.run_dir(spec.run_id) / "KILL").write_text(
                json.dumps(audit, indent=2), encoding="utf-8"
            )
            self._write_status(
                spec.run_id, {"status": "killed", "reason": reason, "killed_ts": ts}
            )
            killed.append(audit)
        return killed

    def _kill_reason(self, spec: RunSpec) -> str | None:
        metrics_path = self.run_dir(spec.run_id) / "metrics.jsonl"
        if not metrics_path.exists():
            return None
        wall_s = 0.0
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            loss = rec.get("loss")
            if rec.get("diverged") or (isinstance(loss, (int, float)) and not math.isfinite(loss)):
                return "divergence flag in metrics.jsonl"
            wall_s = max(wall_s, float(rec.get("wall_s", 0.0)))
        scheduled_s = spec.est_gpu_hours * 3600.0 / assumed_gpus(spec.size)
        if scheduled_s > 0 and wall_s > spec.max_wall_clock_mult * scheduled_s:
            return (
                f"wall clock {wall_s:.0f}s > {spec.max_wall_clock_mult:g}x "
                f"scheduled {scheduled_s:.0f}s"
            )
        return None
