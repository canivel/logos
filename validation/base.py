"""Falsifiable-probe base, after the KAOS eval harness (`kaos.eval.harness`):
kill gates are pre-registered and sha256-locked before results are examined;
an edited gate voids the run instead of silently passing.

Verdicts:
  ACCEPT — all gates passed.
  REJECT — a kill gate failed (the checked claim is falsified; fix the stack
           or the claim, never the gate).
  VOID   — the gate definitions no longer match their lock (tampering guard)
           or the probe could not run to completion.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

LOCK_DIR = Path(__file__).parent / "locks"


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    VOID = "VOID"


@dataclass
class GateOutcome:
    gate: str
    description: str
    passed: bool
    kill: bool = True
    detail: str = ""


@dataclass
class ProbeResult:
    probe: str
    verdict: Verdict
    gates: list[GateOutcome] = field(default_factory=list)
    error: str = ""
    wall_s: float = 0.0
    lock_sha: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


class Probe:
    """Subclass contract: set `name`, `description`, and `gate_specs` (the
    pre-registered list of (gate_id, description) pairs — this is what gets
    locked), then implement `collect() -> dict` (measurements only, no
    verdicts) and `gates(measurements) -> list[GateOutcome]` (pure judgment
    against the locked spec)."""

    name: str = "probe"
    description: str = ""
    gate_specs: list[tuple[str, str]] = []

    # ---- lock discipline ----

    def _spec_sha(self) -> str:
        payload = json.dumps(
            {"name": self.name, "gates": self.gate_specs}, sort_keys=True
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def lock_path(self) -> Path:
        return LOCK_DIR / f"{self.name}.lock.json"

    def ensure_lock(self) -> tuple[bool, str]:
        """Register the gate spec on first run; refuse if it changed since.
        Returns (ok, sha)."""
        sha = self._spec_sha()
        lp = self.lock_path()
        if not lp.exists():
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(
                json.dumps(
                    {"probe": self.name, "sha256": sha, "gates": self.gate_specs,
                     "registered": time.strftime("%Y-%m-%d %H:%M:%S")},
                    indent=2,
                )
            )
            return True, sha
        locked = json.loads(lp.read_text())["sha256"]
        return locked == sha, sha

    # ---- to implement ----

    def collect(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def gates(self, m: dict) -> list[GateOutcome]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- runner ----

    def run(self) -> ProbeResult:
        t0 = time.time()
        ok, sha = self.ensure_lock()
        if not ok:
            return ProbeResult(
                self.name, Verdict.VOID, error="gate spec differs from sha256 lock",
                wall_s=time.time() - t0, lock_sha=sha,
            )
        try:
            measurements = self.collect()
            outcomes = self.gates(measurements)
        except Exception:
            return ProbeResult(
                self.name, Verdict.VOID, error=traceback.format_exc(limit=8),
                wall_s=time.time() - t0, lock_sha=sha,
            )
        declared = {g for g, _ in self.gate_specs}
        emitted = {o.gate for o in outcomes}
        if emitted != declared:
            return ProbeResult(
                self.name, Verdict.VOID,
                error=f"emitted gates {sorted(emitted)} != locked spec {sorted(declared)}",
                gates=outcomes, wall_s=time.time() - t0, lock_sha=sha,
            )
        killed = any(o.kill and not o.passed for o in outcomes)
        verdict = Verdict.REJECT if killed else Verdict.ACCEPT
        return ProbeResult(self.name, verdict, outcomes, "", time.time() - t0, sha)
