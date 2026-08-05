"""Network-free tests for the LOGOS ops layer (PLAN.md s13).

Run: PYTHONPATH=src;. python -m pytest tests/test_ops.py -q
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops import arxiv_triage
from ops.fallback import eval_on_complete, monitor_loop
from ops.kaos import agents, tools

NOW = 1_800_000_000.0


# ---------------------------------------------------------------------------
# synthetic run dirs
# ---------------------------------------------------------------------------


def make_run(
    runs_dir: Path,
    run_id: str,
    losses: list[float],
    *,
    status: str | None = "running",
    metric_age_s: float = 60.0,
    heartbeat_age_s: float | None = 60.0,
) -> Path:
    d = runs_dir / run_id
    d.mkdir(parents=True)
    rows = []
    n = len(losses)
    for i, loss in enumerate(losses):
        rows.append(json.dumps({
            "step": (i + 1) * 100,
            "tokens": (i + 1) * 1_000_000,
            "loss": loss,
            "lr": 3e-4,
            "ts": NOW - metric_age_s - (n - 1 - i) * 30,
        }))
    (d / "metrics.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    st: dict = {}
    if status is not None:
        st["status"] = status
    if heartbeat_age_s is not None:
        st["heartbeat_ts"] = NOW - heartbeat_age_s
    (d / "status.json").write_text(json.dumps(st), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# (1) scan_runs classification
# ---------------------------------------------------------------------------


def test_scan_runs_classifies_states(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_run(runs, "r_healthy", [3.0, 2.8, 2.6, 2.5, 2.4])
    make_run(runs, "r_nan", [3.0, 2.8, float("nan")])
    make_run(runs, "r_diverged", [2.5, 2.2, 2.0, 2.1, 8.0])
    make_run(runs, "r_stalled", [3.0, 2.9, 2.8, 2.7, 2.6],
             metric_age_s=45 * 60, heartbeat_age_s=60.0)
    make_run(runs, "r_preempted", [3.0, 2.9, 2.8, 2.7, 2.6],
             metric_age_s=45 * 60, heartbeat_age_s=45 * 60)
    make_run(runs, "r_complete", [3.0, 2.5, 2.2, 2.1, 2.0], status="complete")

    by_id = {h.run_id: h for h in tools.scan_runs(runs, now=NOW)}
    assert by_id["r_healthy"].state == "healthy"
    assert by_id["r_nan"].state == "nan"
    assert by_id["r_diverged"].state == "diverged"
    assert by_id["r_stalled"].state == "stalled"
    assert by_id["r_preempted"].state == "preempted"
    assert by_id["r_complete"].state == "complete"
    assert by_id["r_healthy"].last_step == 500
    assert math.isclose(by_id["r_complete"].last_loss, 2.0)


def test_scan_runs_status_preempted_and_unknown(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_run(runs, "r_flag", [3.0, 2.9], status="preempted")
    (runs / "r_empty").mkdir()
    by_id = {h.run_id: h for h in tools.scan_runs(runs, now=NOW)}
    assert by_id["r_flag"].state == "preempted"
    assert by_id["r_empty"].state == "unknown"


# ---------------------------------------------------------------------------
# (2) audit journal: every mutating tool writes an entry
# ---------------------------------------------------------------------------


def _audit_entries(ops_dir: Path) -> list[dict]:
    path = ops_dir / "audit.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


def test_every_mutating_tool_appends_audit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LOGOS_ALERT_WEBHOOK", raising=False)
    ops_dir = tmp_path / "ops"
    runs = tmp_path / "runs"
    make_run(runs, "r1", [2.0], status="complete")

    tools.alert("hello", "info", ops_dir=ops_dir)
    tools.resume_run("r1", runs, dry_run=True, ops_dir=ops_dir)
    tools.eval_and_archive("r1", runs, dry_run=True, ops_dir=ops_dir)
    launcher = FakeLauncher({"run_id": "r1", "est_cost": 10.0, "gpus": 1})
    tools.next_launch(dry_run=True, launcher=launcher, ledger=FakeLedger(),
                      ops_dir=ops_dir)

    entries = _audit_entries(ops_dir)
    actions = [e["action"] for e in entries]
    assert {"send_alert", "resume_run", "eval_and_archive", "next_launch"} <= set(actions)
    for e in entries:
        assert set(e) == {"ts", "actor", "tier", "action", "args", "result"}
        assert e["ts"] <= time.time()

    # alert with no webhook lands in alerts.log too
    assert "hello" in (ops_dir / "alerts.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# (3) approval protocol
# ---------------------------------------------------------------------------


class FakeLauncher:
    def __init__(self, candidate: dict) -> None:
        self.candidate = candidate
        self.launched: list[tuple[str, bool]] = []

    def peek_next(self) -> dict:
        return self.candidate

    def launch_next(self, dry_run: bool = False) -> None:
        self.launched.append((self.candidate["run_id"], dry_run))


class FakeLedger:
    def __init__(self) -> None:
        self.reserved: list[dict] = []

    def check_and_reserve(self, run_id: str, cost: float, gpus: int):
        self.reserved.append({"run_id": run_id, "cost": cost, "gpus": gpus})
        return SimpleNamespace(requires_human=cost > 200.0 or gpus >= 8)


def test_next_launch_over_200_needs_approval_file(tmp_path: Path) -> None:
    ops_dir = tmp_path / "ops"
    launcher = FakeLauncher({"run_id": "p2_490m_320x", "est_cost": 350.0, "gpus": 8})
    ledger = FakeLedger()

    out = tools.next_launch(dry_run=True, launcher=launcher, ledger=ledger,
                            ops_dir=ops_dir)
    assert out["launched"] is False
    assert out["reason"] == "pending_approval"
    pending = ops_dir / "pending_approvals" / "p2_490m_320x.json"
    assert pending.exists()
    assert launcher.launched == []  # did NOT launch
    body = json.loads(pending.read_text(encoding="utf-8"))
    assert body["est_cost_usd"] == 350.0 and body["gpus"] == 8

    # human approves by renaming to .approved -> next cycle proceeds
    pending.rename(pending.with_suffix(".approved"))
    out2 = tools.next_launch(dry_run=True, launcher=launcher, ledger=ledger,
                             ops_dir=ops_dir)
    assert out2["launched"] is True
    assert out2["approved_by_human"] is True
    assert launcher.launched == [("p2_490m_320x", True)]  # dry-run executor


def test_next_launch_hard_caps_even_if_ledger_says_ok(tmp_path: Path) -> None:
    """Caps hard-coded, not prompt- (or ledger-) enforced."""
    ops_dir = tmp_path / "ops"

    class PermissiveLedger:
        def check_and_reserve(self, run_id, cost, gpus):
            return SimpleNamespace(requires_human=False)

    launcher = FakeLauncher({"run_id": "big8x", "est_cost": 50.0, "gpus": 8})
    out = tools.next_launch(launcher=launcher, ledger=PermissiveLedger(),
                            ops_dir=ops_dir)
    assert out["launched"] is False and out["reason"] == "pending_approval"
    assert launcher.launched == []


def test_next_launch_budget_exceeded(tmp_path: Path) -> None:
    class BudgetExceeded(Exception):
        pass

    class BrokeLedger:
        def check_and_reserve(self, run_id, cost, gpus):
            raise BudgetExceeded("daily cap")

    launcher = FakeLauncher({"run_id": "cheap", "est_cost": 5.0, "gpus": 1})
    out = tools.next_launch(launcher=launcher, ledger=BrokeLedger(),
                            ops_dir=tmp_path / "ops")
    assert out["launched"] is False and out["reason"] == "budget_exceeded"
    assert launcher.launched == []


def test_next_launch_without_launcher_is_safe(tmp_path: Path) -> None:
    out = tools.next_launch(ops_dir=tmp_path / "ops")
    assert out["launched"] is False
    assert out["reason"] == "launcher_unavailable"


# ---------------------------------------------------------------------------
# (4) arxiv triage --dry-run
# ---------------------------------------------------------------------------


def test_arxiv_triage_dry_run_digest(tmp_path: Path) -> None:
    rc = arxiv_triage.main(["--dry-run", "--out-dir", str(tmp_path / "digests")])
    assert rc == 0
    files = list((tmp_path / "digests").glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "SCOOP-WATCH" in text  # the precision+memory fixture paper
    assert "Precision-Aware Scaling Laws" in text
    assert "(score" in text
    # scored ordering: the scoop paper is ranked above the survey
    assert text.index("Precision-Aware Scaling Laws") < text.index(
        "A Survey of Instruction Tuning Datasets"
    )
    # the irrelevant survey is called LOW
    assert "[LOW]" in text


def test_arxiv_triage_scoring_calls() -> None:
    s, hits = arxiv_triage.score_paper(
        "Precision scaling laws with QAT under memory budgets",
        "kv cache included; overtraining axis; per-byte loss",
    )
    assert arxiv_triage.relevance_call(s) == "SCOOP-WATCH"
    assert s >= 10 and hits
    s_low, _ = arxiv_triage.score_paper("A dataset survey", "instruction data")
    assert arxiv_triage.relevance_call(s_low) == "LOW"


# ---------------------------------------------------------------------------
# (5) agent definitions
# ---------------------------------------------------------------------------


def test_to_kaos_cmd_well_formed() -> None:
    for agent in agents.ALL_AGENTS:
        cmd = agent.to_kaos_cmd()
        assert cmd.startswith('kaos run "')
        assert cmd.endswith(f'-n {agent.name}')
        # prompt is quoted intact: no unescaped double quotes inside
        inner = cmd[len('kaos run "'):cmd.rindex('"')]
        assert '"' not in inner.replace('\\"', "")


def test_prompts_contain_gates_verbatim() -> None:
    assert agents.MONITOR_AGENT.gate in agents.MONITOR_AGENT.prompt
    assert "worst case is a wasted restart" in agents.MONITOR_AGENT.prompt
    assert agents.EVAL_AGENT.gate in agents.EVAL_AGENT.prompt
    assert "versioned manifest" in agents.EVAL_AGENT.prompt
    lp = agents.LAUNCHER_AGENT.prompt
    assert agents.LAUNCHER_AGENT.gate in lp
    assert "check_and_reserve" in lp and "requires_human" in lp and "STOP" in lp
    assert "pending_approvals" in lp
    for agent in agents.ALL_AGENTS:
        assert agents.NEVER_CLAUSE in agent.prompt
        assert "audit" in agent.prompt.lower()


def test_allowlists_machine_checkable() -> None:
    for agent in agents.ALL_AGENTS:
        assert "write_manifest" not in agent.allowlist
        assert "append_audit" in agent.allowlist
    assert "resume_run" in agents.MONITOR_AGENT.allowlist
    assert "launch_run" not in agents.MONITOR_AGENT.allowlist
    with pytest.raises(PermissionError):
        agents.enforce_allowlist(agents.MONITOR_AGENT, "write_manifest")
    agents.enforce_allowlist(agents.MONITOR_AGENT, "resume_run")  # no raise


def test_launch_all_via_kaos_falls_back_without_kaos() -> None:
    out = agents.launch_all_via_kaos()
    assert out["mode"] in ("kaos", "fallback")
    if out["mode"] == "fallback":  # kaos not installed in this env
        assert len(out["commands"]) == 3
        assert any("logos-monitor" in c for c in out["commands"])


# ---------------------------------------------------------------------------
# fallback loops reuse the same tools
# ---------------------------------------------------------------------------


def test_monitor_loop_tick_resumes_and_alerts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LOGOS_ALERT_WEBHOOK", raising=False)
    ops_dir = tmp_path / "ops"
    runs = tmp_path / "runs"
    make_run(runs, "r_pre", [3.0, 2.9, 2.8, 2.7, 2.6],
             metric_age_s=45 * 60, heartbeat_age_s=45 * 60)
    make_run(runs, "r_nan", [3.0, float("nan")])

    monitor_loop.tick(runs, dry_run=True, ops_dir=ops_dir, now=NOW)
    actions = [e["action"] for e in _audit_entries(ops_dir)]
    assert "resume_run" in actions
    alerts = (ops_dir / "alerts.log").read_text(encoding="utf-8")
    assert "r_pre" in alerts and "r_nan" in alerts and "CRITICAL" in alerts


def test_eval_on_complete_tick_only_new_complete_runs(tmp_path: Path) -> None:
    ops_dir = tmp_path / "ops"
    runs = tmp_path / "runs"
    make_run(runs, "r_done", [2.0], status="complete")
    make_run(runs, "r_live", [2.0], status="running")

    first = eval_on_complete.tick(runs, dry_run=False, ops_dir=ops_dir)
    assert first == ["r_done"]
    # idempotent: second pass finds nothing new
    second = eval_on_complete.tick(runs, dry_run=False, ops_dir=ops_dir)
    assert second == []
    actions = [e["action"] for e in _audit_entries(ops_dir)]
    assert actions.count("eval_and_archive") == 1
