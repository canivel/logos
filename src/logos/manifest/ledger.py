"""Budget ledger with HARD caps (PLAN.md s13: "caps hard-coded, not
prompt-enforced"). The launcher refuses beyond cap regardless of its reasoning;
changing CAPS requires a code commit, by design.

Tier-3 gate (PLAN.md s13): any launch over $200 or any 8x-node launch
(250m/490m/1b/1.5b) requires human approval -> requires_human=True.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from logos.config import RunSpec

# Hard caps in USD. Plan s12 upper bounds + headroom. Edit == code commit.
CAPS: dict[str, Any] = {
    "per_day_usd": 400.0,
    "per_phase_usd": {
        "p0": 400.0,
        "p1": 1300.0,
        "p2": 8000.0,
        "p2ext": 4200.0,
        "p3": 5500.0,
        "p4": 2000.0,
        "p5": 12000.0,
    },
    "total_usd": 30000.0,
}

# Tier-3 gate thresholds (PLAN.md s13).
TIER3_COST_USD = 200.0
TIER3_SIZES = {"250m", "490m", "1b", "1.5b"}  # 8x H100 SXM node launches


class BudgetExceeded(RuntimeError):
    """A reservation would break a hard cap. Not negotiable at runtime."""


def _today() -> str:
    return _dt.date.today().isoformat()


class BudgetLedger:
    """JSON-persisted ledger: {spent_usd_total, spent_by_phase, spent_by_day,
    entries}. Reservations book the estimate; record_actual reconciles."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "spent_usd_total": 0.0,
                "spent_by_phase": {},
                "spent_by_day": {},
                "entries": [],
            }
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    # ---- checks -----------------------------------------------------------

    def check(self, spec: RunSpec, day: str | None = None) -> dict[str, Any]:
        """Cap check without side effects -> {ok, reason, requires_human, ...}.
        The launcher's dry-run path calls this."""
        day = day or _today()
        est = float(spec.est_cost_usd)
        requires_human = est > TIER3_COST_USD or spec.size in TIER3_SIZES
        ok, reason = True, "within caps"
        phase_cap = CAPS["per_phase_usd"].get(spec.phase)
        day_spent = self.state["spent_by_day"].get(day, 0.0)
        phase_spent = self.state["spent_by_phase"].get(spec.phase, 0.0)
        if phase_cap is None:
            ok, reason = False, f"no cap defined for phase {spec.phase!r}"
        elif day_spent + est > CAPS["per_day_usd"]:
            ok = False
            reason = f"per-day cap: ${day_spent:,.0f} spent + ${est:,.0f} > ${CAPS['per_day_usd']:,.0f}"
        elif phase_spent + est > phase_cap:
            ok = False
            reason = f"per-phase cap ({spec.phase}): ${phase_spent:,.0f} spent + ${est:,.0f} > ${phase_cap:,.0f}"
        elif self.state["spent_usd_total"] + est > CAPS["total_usd"]:
            ok = False
            reason = f"total cap: ${self.state['spent_usd_total']:,.0f} spent + ${est:,.0f} > ${CAPS['total_usd']:,.0f}"
        return {
            "run_id": spec.run_id,
            "phase": spec.phase,
            "day": day,
            "est_cost_usd": est,
            "ok": ok,
            "reason": reason,
            "requires_human": requires_human,
        }

    def check_and_reserve(self, spec: RunSpec, day: str | None = None) -> dict[str, Any]:
        """Reserve est_cost against the caps or raise BudgetExceeded."""
        res = self.check(spec, day=day)
        if not res["ok"]:
            raise BudgetExceeded(f"{spec.run_id}: {res['reason']}")
        est, d = res["est_cost_usd"], res["day"]
        self.state["spent_usd_total"] += est
        self.state["spent_by_phase"][spec.phase] = (
            self.state["spent_by_phase"].get(spec.phase, 0.0) + est
        )
        self.state["spent_by_day"][d] = self.state["spent_by_day"].get(d, 0.0) + est
        self.state["entries"].append(
            {
                "run_id": spec.run_id,
                "phase": spec.phase,
                "day": d,
                "est_usd": est,
                "booked_usd": est,
                "actual_usd": None,
                "gpu_hours": None,
                "requires_human": res["requires_human"],
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
        )
        self._save()
        return res

    def record_actual(self, run_id: str, usd: float, gpu_hours: float) -> dict[str, Any]:
        """Reconcile a reservation with the actual spend (delta re-booked)."""
        entry = next(
            (e for e in reversed(self.state["entries"]) if e["run_id"] == run_id), None
        )
        if entry is None:
            raise KeyError(f"no reservation for run_id {run_id!r}")
        delta = float(usd) - entry["booked_usd"]
        self.state["spent_usd_total"] += delta
        self.state["spent_by_phase"][entry["phase"]] += delta
        self.state["spent_by_day"][entry["day"]] += delta
        entry["actual_usd"] = float(usd)
        entry["booked_usd"] = float(usd)
        entry["gpu_hours"] = float(gpu_hours)
        self._save()
        return entry
