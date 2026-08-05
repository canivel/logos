# LOGOS ops runbook (KAOS layer + boring fallbacks)

PLAN.md section 13, operationalized. Doctrine: **agents run the toil, humans
run the experiment.** Every capability here has a cron/bash fallback because
LOGOS never blocks on KAOS (the 1-2 day rule, PLAN.md s13/s15).

## Autonomy tiers

| Tier | Agent | Actions | Gate |
|------|-------|---------|------|
| 1 (from P0) | `logos-monitor` | Watch W&B and pod state; detect NaN, divergence, stall, preemption; alert; auto-resume preempted runs from last checkpoint | Fully autonomous. Resume is deterministic; worst case is a wasted restart. |
| 2 (from P2) | `logos-eval` | On run completion: pull checkpoint, run BPB + lm-eval, log results, archive to HF and volume; enforce manifest kill criteria with an audit entry | Autonomous within criteria defined in the versioned manifest |
| 3 (earned) | `logos-launcher` | Start the next manifest runs when a node frees, within the budget ledger | Human approval for any 8x node launch or any run over $200; caps hard-coded (`ops/kaos/tools.py`: `MAX_AUTONOMOUS_COST_USD`, `MAX_AUTONOMOUS_GPUS`), not prompt-enforced |
| Never | — | Edit training code, hyperparameters, or the manifest; generate configs; make fitting or model-selection decisions | Science stays human. "An agent decided" is not a methods section. |

## Launching

### Via KAOS (control plane = the 5090 box)

```bash
# shell primitives (one sandboxed agent each; prompts embed the tier gates)
python -c "from ops.kaos.agents import ALL_AGENTS; [print(a.to_kaos_cmd()) for a in ALL_AGENTS]"
kaos run "<monitor prompt>" -n logos-monitor    # printed by the line above

# or all three via the Python API (guarded; falls back cleanly if kaos is absent)
python -c "from ops.kaos.agents import launch_all_via_kaos; print(launch_all_via_kaos('~/.kaos/logos.db'))"
```

Router config: `ops/kaos/kaos.yaml` — local model (ollama/vLLM on the 5090)
runs the always-on monitor classification; `anthropic` `claude-sonnet-5`
handles judgment calls (anomaly explanation, triage digests). Native SDKs
only; **litellm is banned**.

### Via the boring fallback (cron)

```bash
python ops/fallback/monitor_loop.py /data/runs --interval 600     # Tier 1 loop
python ops/fallback/monitor_loop.py /data/runs --once             # cron mode
python ops/fallback/eval_on_complete.py /data/runs [--dry-run]    # Tier 2
python ops/arxiv_triage.py [--dry-run] [--llm]                    # scoop watch
```

crontab on the 5090 box:

```cron
*/10 * * * * cd /path/to/logos && python ops/fallback/monitor_loop.py /data/runs --once >> ops/monitor_cron.log 2>&1
0 * * * *    cd /path/to/logos && python ops/fallback/eval_on_complete.py /data/runs >> ops/eval_cron.log 2>&1
0 7 * * *    cd /path/to/logos && python ops/arxiv_triage.py >> ops/triage_cron.log 2>&1
```

Windows (dev box) equivalents are in each script's docstring (`schtasks /Create ...`).

Tier 3 has **no autonomous cron fallback** by design: when KAOS is down,
launches go back to a human running `python -c "from ops.kaos.tools import
next_launch; print(next_launch())"` by hand.

## Scoped credentials (no agent holds more than its tier needs)

| Env var | Scope | Read by |
|---------|-------|---------|
| `RUNPOD_API_KEY` | **project-scoped** RunPod key (this project's pods only) | Tier 1 (read pod state), Tier 3 (launch) |
| `WANDB_API_KEY` | W&B **service account**, this project only | Tier 1 (read runs), trainer |
| `HF_TOKEN` | **write-scoped** HF token limited to the TRIT repos | Tier 2 (`eval_and_archive` upload) |
| `LOGOS_ALERT_WEBHOOK` | webhook URL for alerts (else `ops/alerts.log`) | `tools.alert` (all tiers) |
| `ANTHROPIC_API_KEY` | Anthropic API (judgment routes + `--llm` triage) | GEPA router cloud route, `arxiv_triage --llm` |
| `LOGOS_OPS_DIR` | override the ops/ state dir (tests) | all tools |

## Approval-file protocol (Tier 3)

1. `next_launch()` calls `BudgetLedger.check_and_reserve` **first**.
   `BudgetExceeded` ends the attempt, always.
2. If the reservation has `requires_human` — or the hard-coded caps trip
   (cost > $200, or gpus >= 8) — it writes
   `ops/pending_approvals/<run_id>.json` and does **not** launch.
3. A human reviews and approves by renaming the file to
   `<run_id>.approved`. Nothing else counts as approval.
4. The next `next_launch()` cycle sees the `.approved` file and proceeds
   (still through the ledger).

## Misbehavior rule

An agent tier that fails **twice on the same failure mode** drops to manual
until the fix ships (PLAN.md s13: "dogfooding is the point; yak-shaving is
the failure mode"). Concretely: stop the kaos agent (`kaos ... stop` or kill
the fallback cron line), leave the tier's fallback in `--dry-run`, file a
KAOS backlog entry, and only re-arm after the fix is tested against the
audit journal below.

## Audit journal

Every mutating tool appends `{ts, actor, tier, action, args, result}` to
`ops/audit.jsonl`. It is both the safety record and the KAOS demo artifact
("the LOGOS grid under autonomous ops: N runs, M preemption recoveries,
$X managed, zero manual restarts").

Query examples:

```bash
# via KAOS's journal (when agents ran under kaos)
kaos query "select action, count(*) from journal group by action"

# via the file (always works)
python - <<'PY'
import json, collections
c = collections.Counter()
spend = 0.0
for line in open("ops/audit.jsonl", encoding="utf-8"):
    e = json.loads(line)
    c[e["action"]] += 1
    if e["action"] == "next_launch" and e["result"].get("launched"):
        spend += e["result"].get("cost", 0.0)
print(dict(c))
print(f"preemption recoveries: {c['resume_run']}, $ managed: {spend:.0f}")
PY
```
