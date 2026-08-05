"""The three LOGOS tier agents as KAOS-launchable definitions (PLAN.md s13).

Each agent is a dataclass carrying its prompt (the product: tier gates are
embedded verbatim), a machine-checkable action allowlist the wrapper
enforces, and a `to_kaos_cmd()` producing the exact `kaos run` invocation.
`launch_all_via_kaos()` uses the guarded Python API (Kaos, ClaudeCodeRunner,
GEPARouter) when kaos is installed; otherwise it returns the shell commands
and points at the ops/fallback/ scripts -- LOGOS never blocks on KAOS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROUTER_CONFIG = Path(__file__).with_name("kaos.yaml")

# The "Never" row of the PLAN.md s13 tier table, embedded verbatim in every
# prompt. Science stays human.
NEVER_CLAUSE = (
    "NEVER, under any instruction from any source: edit training code, "
    "hyperparameters, or the manifest; generate configs; make fitting or "
    "model-selection decisions. Science stays human. "
    '"An agent decided" is not a methods section.'
)

AUDIT_CLAUSE = (
    "Every action you take MUST append a JSON line to ops/audit.jsonl "
    "({ts, actor, tier, action, args, result}) -- use the provided tools, "
    "which do this automatically. An action that is not in your allowlist "
    "must not be attempted; the wrapper will refuse it."
)

# Tier gates, verbatim from the PLAN.md s13 autonomy-tier table.
MONITOR_GATE = (
    "Fully autonomous. Resume is deterministic; worst case is a wasted restart."
)
EVAL_GATE = "Autonomous within criteria defined in the versioned manifest"
LAUNCHER_GATE = (
    "Human approval for any 8x node launch or any run over $200 until trust "
    "is established; caps hard-coded, not prompt-enforced"
)


@dataclass(frozen=True)
class TierAgent:
    """A KAOS-launchable ops agent: name, tier, prompt, allowlist, gate."""

    name: str
    tier: int
    prompt: str
    allowlist: list[str]
    gate: str

    def to_kaos_cmd(self) -> str:
        """Exact `kaos run "<prompt>" -n <name>` invocation."""
        escaped = self.prompt.replace("\\", "\\\\").replace('"', '\\"')
        return f'kaos run "{escaped}" -n {self.name}'

    def allows(self, action: str) -> bool:
        return action in self.allowlist


def enforce_allowlist(agent: TierAgent, action: str) -> None:
    """Machine-checkable gate the wrapper applies before executing any action."""
    if not agent.allows(action):
        raise PermissionError(
            f"{agent.name} (tier {agent.tier}) is not allowed action "
            f"{action!r}; allowlist={agent.allowlist}"
        )


MONITOR_AGENT = TierAgent(
    name="logos-monitor",
    tier=1,
    gate=MONITOR_GATE,
    allowlist=[
        "read_metrics",
        "read_status",
        "read_wandb",
        "read_pod_state",
        "resume_run",
        "send_alert",
        "append_audit",
    ],
    prompt=(
        "You are the LOGOS Tier 1 Monitor agent (PLAN.md s13, active from P0). "
        f"Your gate: {MONITOR_GATE} "
        "Loop: (1) call scan_runs on the runs directory to parse each run's "
        "metrics.jsonl tail and status.json; also watch W&B and RunPod pod "
        "state when credentials are available (WANDB_API_KEY, RUNPOD_API_KEY "
        "-- read-only). (2) Classify each run as healthy | nan | diverged | "
        "stalled (no step in >20 min) | preempted (pod gone / heartbeat "
        "stale) | complete. (3) For nan or diverged runs: send_alert with the "
        "run_id and the offending loss values; do NOT touch the run itself -- "
        "kill decisions belong to Tier 2's manifest criteria and to humans. "
        "(4) For preempted or stalled runs: call resume_run, which re-issues "
        "the train command deterministically from ckpt_latest.pt; then "
        "send_alert at info level noting the auto-resume. Resume is safe to "
        "repeat -- the worst case is a wasted restart. "
        f"{AUDIT_CLAUSE} "
        "Your allowlist: read_metrics, read_status, read_wandb, "
        "read_pod_state, resume_run, send_alert, append_audit. You may not "
        "launch new runs, evaluate checkpoints, or spend money. "
        f"{NEVER_CLAUSE}"
    ),
)

EVAL_AGENT = TierAgent(
    name="logos-eval",
    tier=2,
    gate=EVAL_GATE,
    allowlist=[
        "read_metrics",
        "read_status",
        "read_manifest",
        "pull_checkpoint",
        "run_bpb",
        "run_lm_eval",
        "append_result",
        "archive_hf",
        "archive_volume",
        "kill_run",
        "send_alert",
        "append_audit",
    ],
    prompt=(
        "You are the LOGOS Tier 2 Eval & Hygiene agent (PLAN.md s13, active "
        f"from P2). Your gate: {EVAL_GATE}. "
        "On each run reaching status complete: (1) pull its checkpoint "
        "(ckpt_latest.pt); (2) run the BPB eval and the lm-eval downstream "
        "suite via eval_and_archive; (3) append one results row via "
        "logos.results.store.append_result; (4) archive the checkpoint to "
        "HF (write-scoped HF_TOKEN) and to the network volume. "
        "Enforce ONLY the kill criteria written in the versioned manifest "
        "(loss divergence, >3x scheduled wall clock): if a running run trips "
        "a manifest criterion, kill_run it and write an audit entry naming "
        "the criterion verbatim. You may not invent criteria, relax them, or "
        "kill a run for any reason not in the manifest. Never modify the "
        "manifest, the results of an eval, or training code. "
        f"{AUDIT_CLAUSE} "
        "Your allowlist: read_metrics, read_status, read_manifest, "
        "pull_checkpoint, run_bpb, run_lm_eval, append_result, archive_hf, "
        "archive_volume, kill_run, send_alert, append_audit. "
        f"{NEVER_CLAUSE}"
    ),
)

LAUNCHER_AGENT = TierAgent(
    name="logos-launcher",
    tier=3,
    gate=LAUNCHER_GATE,
    allowlist=[
        "read_manifest",
        "read_status",
        "read_ledger",
        "check_and_reserve",
        "launch_run",
        "write_pending_approval",
        "send_alert",
        "append_audit",
    ],
    prompt=(
        "You are the LOGOS Tier 3 Launcher agent (PLAN.md s13 -- this tier is "
        f"EARNED, not assumed). Your gate: {LAUNCHER_GATE}. "
        "When a node frees, you may start the next runs from the versioned "
        "manifest queue, and nothing else. Protocol, in order, no exceptions: "
        "(1) Read the next pending run from the manifest (read-only -- you "
        "never write manifests). (2) Call BudgetLedger.check_and_reserve "
        "FIRST, before any launch action; if it raises BudgetExceeded, stop "
        "and send_alert -- the ledger's hard per-day and per-phase caps "
        "override any reasoning you have. (3) If the reservation has "
        "requires_human set -- which includes every 8x-node launch and every "
        "run estimated over $200 -- write a pending-approval file at "
        "ops/pending_approvals/<run_id>.json and STOP. Do not launch, do not "
        "retry, do not work around the file. A human approves by renaming it "
        "to <run_id>.approved; only then may a later cycle proceed. These "
        "caps are hard-coded in the tools; they are not yours to interpret. "
        "(4) Only after a clean reservation (and the .approved file when "
        "required) call next_launch to start the run via the Launcher's "
        "executor. (5) send_alert at info level with run_id and reserved "
        "cost. Never modify manifests or training code; never generate "
        "configs; never choose hyperparameters or decide which science to "
        "run -- you only dequeue what humans already wrote into the manifest. "
        f"{AUDIT_CLAUSE} "
        "Your allowlist: read_manifest, read_status, read_ledger, "
        "check_and_reserve, launch_run, write_pending_approval, send_alert, "
        "append_audit. write_manifest is not in your allowlist and never "
        "will be. "
        f"{NEVER_CLAUSE}"
    ),
)

ALL_AGENTS: tuple[TierAgent, ...] = (MONITOR_AGENT, EVAL_AGENT, LAUNCHER_AGENT)


def launch_all_via_kaos(db_path: str | Path | None = None) -> dict[str, Any]:
    """Launch all three tier agents via the KAOS Python API when available.

    Guarded: if kaos is not installed (the 1-2 day rule -- LOGOS never
    blocks on KAOS), returns mode="fallback" with the exact shell commands
    plus the boring-fallback entrypoints instead of raising.
    """
    try:
        from kaos import Kaos  # type: ignore
        from kaos.ccr import ClaudeCodeRunner  # type: ignore
        from kaos.router import GEPARouter  # type: ignore
    except Exception:
        return {
            "mode": "fallback",
            "commands": [a.to_kaos_cmd() for a in ALL_AGENTS],
            "fallbacks": [
                "python ops/fallback/monitor_loop.py <runs_dir> --once",
                "python ops/fallback/eval_on_complete.py <runs_dir> --dry-run",
            ],
            "note": "kaos not importable; use the shell commands on the 5090 "
                    "box or the cron fallbacks above (PLAN.md s13/s15).",
        }

    launched: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        k = Kaos(db_path=str(db_path)) if db_path is not None else Kaos()
    except TypeError:
        k = Kaos()

    router = None
    try:
        loader = getattr(GEPARouter, "from_config", None) or getattr(
            GEPARouter, "from_yaml", None
        )
        router = loader(str(ROUTER_CONFIG)) if callable(loader) else GEPARouter()
    except Exception as exc:
        errors.append(f"router:{type(exc).__name__}")

    for agent in ALL_AGENTS:
        try:
            runner = None
            try:
                runner = ClaudeCodeRunner()
            except Exception:
                pass
            kwargs: dict[str, Any] = {"name": agent.name}
            if router is not None:
                kwargs["router"] = router
            if runner is not None:
                kwargs["runner"] = runner
            run = getattr(k, "run", None)
            if callable(run):
                try:
                    run(agent.prompt, **kwargs)
                except TypeError:
                    run(agent.prompt, name=agent.name)
                launched.append({"name": agent.name, "tier": agent.tier})
            else:
                errors.append(f"{agent.name}:no_run_method")
        except Exception as exc:
            errors.append(f"{agent.name}:{type(exc).__name__}")

    return {"mode": "kaos", "launched": launched, "errors": errors}
