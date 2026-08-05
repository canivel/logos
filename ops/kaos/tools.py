"""Concrete tool functions for the LOGOS ops tiers (PLAN.md section 13).

Pure Python, no kaos dependency: these are called both by the KAOS agents
(via ops.kaos.agents) and by the boring fallbacks (ops/fallback/*). Every
mutating tool appends a JSON line to ops/audit.jsonl -- the audit journal
that doubles as the KAOS demo artifact ("N runs, M preemption recoveries,
$X managed, zero manual restarts").

Interfaces to logos.manifest.* / logos.results.* / logos.eval.* are built
by other agents in parallel, so all of those imports are lazy and guarded.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- hard-coded autonomy caps (PLAN.md s13: "caps hard-coded, not
# prompt-enforced"). Any launch at/above either threshold requires a human
# approval file regardless of what a ledger or an agent says.
MAX_AUTONOMOUS_COST_USD = 200.0
MAX_AUTONOMOUS_GPUS = 8  # any 8x-node launch needs human approval

# --- health classification thresholds
STALL_MINUTES = 20.0  # no new metrics step in >20 min while "running"
HEARTBEAT_MINUTES = 10.0  # status.json heartbeat_ts stale => pod gone
DIVERGENCE_FACTOR = 2.0  # last loss > 2x rolling-min loss => diverged
METRICS_TAIL_LINES = 50

HEALTH_STATES = (
    "healthy",
    "nan",
    "diverged",
    "stalled",
    "preempted",
    "complete",
    "killed",
    "unknown",
)


def default_ops_dir() -> Path:
    """ops/ next to this package; override with LOGOS_OPS_DIR for tests."""
    env = os.environ.get("LOGOS_OPS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1]


def _resolve_ops_dir(ops_dir: str | Path | None) -> Path:
    d = Path(ops_dir) if ops_dir is not None else default_ops_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def audit(
    action: str,
    args: dict[str, Any],
    result: Any,
    *,
    actor: str = "ops",
    tier: int = 0,
    ops_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Append one JSON line to ops/audit.jsonl. Called by every mutating tool."""
    entry = {
        "ts": time.time(),
        "actor": actor,
        "tier": tier,
        "action": action,
        "args": args,
        "result": result,
    }
    path = _resolve_ops_dir(ops_dir) / "audit.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry


# ---------------------------------------------------------------------------
# Tier 1: monitor tools
# ---------------------------------------------------------------------------


@dataclass
class RunHealth:
    """One run's classified state from metrics.jsonl tail + status.json."""

    run_id: str
    state: str  # one of HEALTH_STATES
    last_step: int | None = None
    last_loss: float | None = None
    last_metric_ts: float | None = None
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_metrics_tail(path: Path, n: int = METRICS_TAIL_LINES) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def classify_run(
    run_dir: Path,
    *,
    now: float | None = None,
    stall_minutes: float = STALL_MINUTES,
    heartbeat_minutes: float = HEARTBEAT_MINUTES,
    divergence_factor: float = DIVERGENCE_FACTOR,
) -> RunHealth:
    """Classify a single <runs_dir>/<run_id> directory.

    Order of checks: terminal status.json states first, then NaN, then
    divergence, then preemption (heartbeat stale / pod gone), then stall
    (metrics stale but heartbeat fresh), else healthy.
    """
    now = time.time() if now is None else now
    run_id = run_dir.name
    status = _read_json(run_dir / "status.json") or {}
    tail = _read_metrics_tail(run_dir / "metrics.jsonl")

    last = tail[-1] if tail else {}
    last_step = last.get("step")
    last_loss = last.get("loss")
    last_ts = last.get("ts")
    if last_ts is None and tail:
        try:
            last_ts = (run_dir / "metrics.jsonl").stat().st_mtime
        except OSError:
            last_ts = None

    def health(state: str, detail: str) -> RunHealth:
        return RunHealth(
            run_id=run_id,
            state=state,
            last_step=last_step,
            last_loss=last_loss,
            last_metric_ts=last_ts,
            detail=detail,
            extra={"status": status.get("status")},
        )

    st = status.get("status")
    if st in ("complete", "diverged", "killed"):
        return health(st, f"status.json reports {st}")
    if st == "preempted":
        return health("preempted", "status.json reports preempted")
    if not tail and st is None:
        return health("unknown", "no metrics.jsonl and no status.json")

    # NaN / inf anywhere in the tail
    losses = [r["loss"] for r in tail if isinstance(r.get("loss"), (int, float))]
    if any(not math.isfinite(v) for v in losses):
        return health("nan", "non-finite loss in metrics tail")

    # divergence: recent loss blowing up vs. rolling min (kill criterion,
    # PLAN.md s7: "loss divergence or >3x scheduled wall clock")
    if len(losses) >= 5:
        lo = min(losses)
        if lo > 0 and losses[-1] > divergence_factor * lo:
            return health(
                "diverged",
                f"last loss {losses[-1]:.4g} > {divergence_factor}x min {lo:.4g}",
            )

    # preemption: pod gone / heartbeat stale
    hb = status.get("heartbeat_ts")
    if isinstance(hb, (int, float)) and now - hb > heartbeat_minutes * 60:
        return health("preempted", f"heartbeat stale by {(now - hb) / 60:.1f} min")

    # stall: no new step in >stall_minutes while nominally running
    if last_ts is not None and now - last_ts > stall_minutes * 60:
        return health("stalled", f"no metrics step in {(now - last_ts) / 60:.1f} min")

    return health("healthy", "recent metrics, finite loss")


def scan_runs(
    runs_dir: str | Path,
    *,
    now: float | None = None,
    stall_minutes: float = STALL_MINUTES,
    heartbeat_minutes: float = HEARTBEAT_MINUTES,
    divergence_factor: float = DIVERGENCE_FACTOR,
) -> list[RunHealth]:
    """Classify every run directory under runs_dir. Read-only (no audit)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    out: list[RunHealth] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        out.append(
            classify_run(
                run_dir,
                now=now,
                stall_minutes=stall_minutes,
                heartbeat_minutes=heartbeat_minutes,
                divergence_factor=divergence_factor,
            )
        )
    return out


def resume_run(
    run_id: str,
    runs_dir: str | Path,
    manifest: str | Path | None = None,
    *,
    executor: Any = None,
    dry_run: bool = False,
    actor: str = "ops",
    tier: int = 1,
    ops_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Re-issue the train command for a preempted/stalled run.

    Deterministic resume from ckpt_latest.pt via the Launcher's executor
    (PLAN.md s13 Tier 1 gate: "Resume is deterministic; worst case is a
    wasted restart"). If logos.manifest.launcher is not importable yet, we
    record the gap in the audit journal instead of crashing the loop.
    """
    runs_dir = Path(runs_dir)
    ckpt = runs_dir / run_id / "ckpt_latest.pt"
    spec = {
        "run_id": run_id,
        "resume_from": str(ckpt),
        "ckpt_exists": ckpt.exists(),
        "manifest": str(manifest) if manifest else None,
    }

    result: dict[str, Any]
    if dry_run:
        result = {"resumed": False, "reason": "dry_run", **spec}
    else:
        if executor is None:
            executor = _default_executor(manifest, runs_dir)
        if executor is None:
            result = {"resumed": False, "reason": "launcher_unavailable", **spec}
        else:
            ok, how = _dispatch_resume(executor, spec)
            result = {"resumed": ok, "via": how, **spec}

    audit("resume_run", {"run_id": run_id, "dry_run": dry_run}, result,
          actor=actor, tier=tier, ops_dir=ops_dir)
    return result


def _default_executor(manifest: str | Path | None, runs_dir: str | Path) -> Any:
    """Lazy, guarded lookup of logos.manifest.launcher's executor."""
    try:
        from logos.manifest.launcher import Launcher, LocalExecutor  # type: ignore
    except Exception:
        return None
    try:
        if manifest is not None:
            launcher = Launcher(manifest=manifest, runs_dir=runs_dir)  # type: ignore[call-arg]
            return getattr(launcher, "executor", launcher)
    except Exception:
        pass
    try:
        return LocalExecutor()  # type: ignore[call-arg]
    except Exception:
        return None


def _dispatch_resume(executor: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    """Duck-typed dispatch: the parallel-built executor API is not frozen."""
    for name in ("resume", "resume_run", "launch", "submit", "run"):
        fn = getattr(executor, name, None)
        if callable(fn):
            try:
                fn(spec)
                return True, name
            except TypeError:
                try:
                    fn(spec["run_id"])
                    return True, name
                except Exception:
                    continue
            except Exception:
                return False, f"{name}:error"
    return False, "no_executor_method"


def alert(
    msg: str,
    level: str = "info",
    *,
    actor: str = "ops",
    tier: int = 1,
    ops_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Webhook POST if LOGOS_ALERT_WEBHOOK is set, else append ops/alerts.log."""
    webhook = os.environ.get("LOGOS_ALERT_WEBHOOK")
    delivered_via = "log"
    ok = True
    if webhook:
        payload = json.dumps({"text": msg, "level": level}).encode("utf-8")
        req = urllib.request.Request(
            webhook, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                delivered_via = "webhook"
        except Exception as exc:  # network down: fall through to the log file
            delivered_via = f"webhook_failed:{type(exc).__name__}"
            ok = False
    if delivered_via != "webhook":
        log = _resolve_ops_dir(ops_dir) / "alerts.log"
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{level.upper()}] {msg}\n"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line)
        ok = True
    result = {"delivered_via": delivered_via, "ok": ok}
    audit("send_alert", {"msg": msg, "level": level}, result,
          actor=actor, tier=tier, ops_dir=ops_dir)
    return result


# ---------------------------------------------------------------------------
# Tier 2: eval & hygiene
# ---------------------------------------------------------------------------


def eval_and_archive(
    run_id: str,
    runs_dir: str | Path,
    *,
    dry_run: bool = False,
    hf_repo: str | None = None,
    results_path: str | Path | None = None,
    actor: str = "ops",
    tier: int = 2,
    ops_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Tier-2 pipeline: pull checkpoint -> BPB + lm-eval -> results row -> archive.

    All logos.eval / logos.results / huggingface_hub imports are lazy and
    guarded; with dry_run=True (or missing deps) each step is recorded as
    skipped rather than executed. Autonomy is bounded by the manifest's
    kill criteria, not by this function (PLAN.md s13 Tier 2 gate).
    """
    runs_dir = Path(runs_dir)
    ckpt = runs_dir / run_id / "ckpt_latest.pt"
    steps: dict[str, Any] = {"ckpt": str(ckpt), "ckpt_exists": ckpt.exists()}

    # 1) BPB
    bpb = None
    if not dry_run:
        try:
            from logos.eval import bpb as bpb_mod  # type: ignore

            for name in ("compute_bpb", "evaluate", "run", "main"):
                fn = getattr(bpb_mod, name, None)
                if callable(fn):
                    bpb = fn(str(ckpt))
                    break
            steps["bpb"] = bpb
        except Exception as exc:
            steps["bpb"] = f"unavailable:{type(exc).__name__}"
    else:
        steps["bpb"] = "skipped:dry_run"

    # 2) downstream (lm-eval suite)
    if not dry_run:
        try:
            from logos.eval import downstream  # type: ignore

            fn = getattr(downstream, "evaluate", None) or getattr(downstream, "run", None)
            steps["downstream"] = fn(str(ckpt)) if callable(fn) else "no_entrypoint"
        except Exception as exc:
            steps["downstream"] = f"unavailable:{type(exc).__name__}"
    else:
        steps["downstream"] = "skipped:dry_run"

    # 3) results row
    row = {"run_id": run_id, "bpb": bpb, "source": "ops.eval_and_archive"}
    if not dry_run:
        try:
            from logos.results.store import append_result  # type: ignore

            if results_path is not None:
                append_result(row, path=results_path)  # type: ignore[call-arg]
            else:
                append_result(row)
            steps["results_row"] = "appended"
        except Exception as exc:
            steps["results_row"] = f"unavailable:{type(exc).__name__}"
    else:
        steps["results_row"] = "skipped:dry_run"

    # 4) archive to HF (write-scoped HF_TOKEN) + network volume
    if not dry_run and hf_repo and ckpt.exists():
        try:
            from huggingface_hub import HfApi  # type: ignore

            HfApi().upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=f"{run_id}/ckpt_latest.pt",
                repo_id=hf_repo,
            )
            steps["hf_archive"] = f"uploaded:{hf_repo}"
        except Exception as exc:
            steps["hf_archive"] = f"unavailable:{type(exc).__name__}"
    else:
        steps["hf_archive"] = "skipped" if not dry_run else "skipped:dry_run"

    result = {"run_id": run_id, "dry_run": dry_run, **steps}
    audit("eval_and_archive", {"run_id": run_id, "dry_run": dry_run}, result,
          actor=actor, tier=tier, ops_dir=ops_dir)
    return result


# ---------------------------------------------------------------------------
# Tier 3: launcher (earned)
# ---------------------------------------------------------------------------


def _candidate_fields(candidate: Any) -> tuple[str, float, int]:
    """Extract (run_id, est_cost_usd, gpus) from a dict or RunSpec-like object."""

    def get(*names: str, default: Any = None) -> Any:
        for n in names:
            if isinstance(candidate, dict) and n in candidate:
                return candidate[n]
            v = getattr(candidate, n, None)
            if v is not None:
                return v
        return default

    run_id = str(get("run_id", "name", "id", default="unknown"))
    cost = float(get("est_cost", "est_cost_usd", "cost_usd", "cost", default=0.0))
    gpus = int(get("gpus", "num_gpus", "n_gpus", default=1))
    return run_id, cost, gpus


def _peek_next(launcher: Any) -> Any:
    for name in ("peek_next", "next_run", "next_pending", "peek"):
        fn = getattr(launcher, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                return None
    return None


def next_launch(
    dry_run: bool = False,
    *,
    launcher: Any = None,
    ledger: Any = None,
    manifest: str | Path | None = None,
    runs_dir: str | Path | None = None,
    actor: str = "ops",
    tier: int = 3,
    ops_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Tier-3 wrapper around Launcher.launch_next + BudgetLedger.

    Approval-file protocol: any launch requiring a human (reservation says
    requires_human, OR the hard-coded caps trip: cost > $200 or an 8x node)
    writes ops/pending_approvals/<run_id>.json and does NOT launch. A human
    approves by renaming the file to <run_id>.approved; the next call then
    proceeds. Caps are enforced here in code, never by prompt.
    """
    ops_path = _resolve_ops_dir(ops_dir)
    approvals = ops_path / "pending_approvals"
    approvals.mkdir(parents=True, exist_ok=True)

    def done(result: dict[str, Any]) -> dict[str, Any]:
        audit("next_launch", {"dry_run": dry_run}, result,
              actor=actor, tier=tier, ops_dir=ops_dir)
        return result

    if launcher is None:
        launcher = _default_launcher(manifest, runs_dir)
    if launcher is None:
        return done({"launched": False, "reason": "launcher_unavailable"})

    candidate = _peek_next(launcher)
    if candidate is None:
        return done({"launched": False, "reason": "no_pending_runs"})
    run_id, cost, gpus = _candidate_fields(candidate)

    # budget ledger: hard caps live in the ledger; BudgetExceeded refuses
    # the launch regardless of any agent's reasoning.
    if ledger is None:
        ledger = getattr(launcher, "ledger", None) or _default_ledger()
    reservation = None
    if ledger is not None:
        try:
            reservation = ledger.check_and_reserve(run_id=run_id, cost=cost, gpus=gpus)
        except TypeError:
            try:
                reservation = ledger.check_and_reserve(run_id, cost)
            except Exception as exc:
                if type(exc).__name__ == "BudgetExceeded":
                    return done({"launched": False, "reason": "budget_exceeded",
                                 "run_id": run_id, "cost": cost})
                raise
        except Exception as exc:
            if type(exc).__name__ == "BudgetExceeded":
                return done({"launched": False, "reason": "budget_exceeded",
                             "run_id": run_id, "cost": cost})
            raise

    requires_human = bool(getattr(reservation, "requires_human", False))
    requires_human = (
        requires_human or cost > MAX_AUTONOMOUS_COST_USD or gpus >= MAX_AUTONOMOUS_GPUS
    )

    approved = (approvals / f"{run_id}.approved").exists()
    if requires_human and not approved:
        pending = approvals / f"{run_id}.json"
        pending.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "est_cost_usd": cost,
                    "gpus": gpus,
                    "requested_ts": time.time(),
                    "approve_by": f"rename to {run_id}.approved",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _release(ledger, reservation)
        return done({
            "launched": False,
            "reason": "pending_approval",
            "run_id": run_id,
            "cost": cost,
            "gpus": gpus,
            "pending_file": str(pending),
        })

    ok, how = _dispatch_launch(launcher, candidate, dry_run)
    return done({
        "launched": ok,
        "run_id": run_id,
        "cost": cost,
        "gpus": gpus,
        "via": how,
        "dry_run": dry_run,
        "approved_by_human": approved,
    })


def _dispatch_launch(launcher: Any, candidate: Any, dry_run: bool) -> tuple[bool, str]:
    for name in ("launch_next", "launch"):
        fn = getattr(launcher, name, None)
        if not callable(fn):
            continue
        for call in (
            lambda: fn(dry_run=dry_run),
            lambda: fn(candidate, dry_run=dry_run),
            lambda: fn(candidate),
            lambda: fn(),
        ):
            try:
                call()
                return True, name
            except TypeError:
                continue
            except Exception:
                return False, f"{name}:error"
    return False, "no_launch_method"


def _release(ledger: Any, reservation: Any) -> None:
    """Best-effort release of an unused reservation."""
    for target, name in ((reservation, "release"), (ledger, "release")):
        fn = getattr(target, name, None) if target is not None else None
        if callable(fn):
            try:
                fn() if target is reservation else fn(reservation)
            except Exception:
                pass
            return


def _default_launcher(manifest: str | Path | None, runs_dir: str | Path | None) -> Any:
    try:
        from logos.manifest.launcher import Launcher  # type: ignore
    except Exception:
        return None
    try:
        kwargs: dict[str, Any] = {}
        if manifest is not None:
            kwargs["manifest"] = manifest
        if runs_dir is not None:
            kwargs["runs_dir"] = runs_dir
        return Launcher(**kwargs)
    except Exception:
        return None


def _default_ledger() -> Any:
    try:
        from logos.manifest.ledger import BudgetLedger  # type: ignore
    except Exception:
        return None
    try:
        return BudgetLedger()
    except Exception:
        return None
