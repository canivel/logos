"""Probe manifest_integrity — PLAN.md phase tables vs the versioned manifests.

Run counts are re-encoded here from the PLAN text (s5-s11), never from the
generator. Results provenance is cross-checked by rebuilding each RunSpec
from its manifest row and recomputing config_hash; a tampered-manifest
fixture proves the check fires. Budget caps must be literal module constants
(PLAN.md s13: "caps hard-coded, not prompt-enforced").
"""

from __future__ import annotations

import ast
import inspect
import tempfile
import time
from pathlib import Path

import yaml

from logos.config import RunSpec
from logos.manifest import ledger as ledger_mod
from logos.manifest.ledger import BudgetExceeded, BudgetLedger
from logos.manifest.schema import ManifestError, load_manifest
from validation.base import GateOutcome, Probe

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "manifests"
RESULTS = REPO / "results" / "results.jsonl"

# PLAN.md phase tables, as written in the plan text:
#   p0 (s5):  "25M and 60M: ternary vs bf16, 20x and 320x, 3 seeds each (16 runs)"
#   p1 (s6):  25 LR probes + 4 FFN ablation (2 ffn x {20x, 80x}) + 15 gap study = 44
#   p2 (s7):  45 + 45 + 15 (reused from P1) + 15 + 10 = 130 runs
#   p2ext (s7): 490M @ 320x x {1.58, 4, bf16} = 3
#   p3 (s9):  1B x {1.58, 4, bf16} x {20x, 80x} = 6
#   p4 (s10): ~8 native KV-QAT runs + 1 GQA-ratio arm = 9
#   p5 (s11): 1 capstone (Tier A)
PLAN_COUNTS = {"p0": 16, "p1": 44, "p2": 130, "p2ext": 3, "p3": 6, "p4": 9, "p5": 1}
P2_REUSED_FROM_P1 = 15
PLAN_TOTAL_USD = (15_000.0, 30_000.0)  # s12 core-program window
PLAN_P2_GPU_H = 1800.0  # s7/s12 "~1,800 GPU-h", checked at +/-40%
RESULTS_POLL_S = 10.0


def _hash_table(manifests_dir: Path) -> dict[str, set[str]]:
    """run_id -> config hashes recomputed from RunSpecs rebuilt off manifest rows."""
    table: dict[str, set[str]] = {}
    for p in sorted(Path(manifests_dir).glob("*.yaml")):
        try:
            _, specs = load_manifest(p)
        except ManifestError:
            continue  # lr_rules.yaml etc.
        for s in specs:
            table.setdefault(s.run_id, set()).add(s.config_hash())
    return table


def _check_rows(rows: list[dict], table: dict[str, set[str]]) -> list[dict]:
    bad = []
    for r in rows:
        hs = table.get(r.get("run_id"))
        if hs is None:
            bad.append({"run_id": r.get("run_id"), "reason": "run_id not in any manifest"})
        elif r.get("config_hash") not in hs:
            bad.append({"run_id": r.get("run_id"), "reason": "config_hash mismatch"})
    return bad


def _numeric_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Dict):
        return all(_numeric_literal(v) for v in node.values)
    return False


class ManifestIntegrityProbe(Probe):
    name = "manifest_integrity"
    description = "manifests match PLAN.md tables; results provenance holds; budget caps refuse"
    gate_specs = [
        ("G1", "phase manifests hold exactly the PLAN.md run counts "
               "(p0:16 p1:44 p2:130 p2ext:3 p3:6 p4:9 p5:1); p2's 15 P1-reused rows "
               "are tagged and zero-cost"),
        ("G2", "run_ids unique across all manifests except the 15 documented p1/p2 reuses, "
               "which share run_id AND config_hash"),
        ("G3", "every results.jsonl row's config_hash matches a RunSpec rebuilt from a "
               "manifest row (recomputed hash); a tampered-manifest fixture makes the "
               "check fire and a clean fixture passes"),
        ("G4", "BudgetLedger: a >cap reservation raises BudgetExceeded; a >$200 run sets "
               "requires_human; CAPS/TIER3_COST_USD are literal numeric module constants "
               "(AST-verified, no env/config lookup)"),
        ("G5", "summed manifest est_cost_usd lands in the plan's $15k-30k window and "
               "p2 est_gpu_hours within +/-40% of ~1800"),
    ]

    def collect(self) -> dict:
        m: dict = {}

        # G1: counts + reuse tagging.
        specs_by_phase: dict[str, list[RunSpec]] = {}
        counts: dict[str, dict] = {}
        for ph in PLAN_COUNTS:
            _, specs = load_manifest(MANIFESTS / f"{ph}.yaml")
            specs_by_phase[ph] = specs
            counts[ph] = {
                "n": len(specs),
                "phases_ok": all(s.phase == ph for s in specs),
            }
        p1_ids = {s.run_id for s in specs_by_phase["p1"]}
        reused = [s for s in specs_by_phase["p2"] if s.run_id in p1_ids]
        m["g1"] = {
            "counts": counts,
            "reused_n": len(reused),
            "reused_tagged": all(any("reuse" in t for t in s.tags) for s in reused),
            "reused_zero_cost": all(
                s.est_cost_usd == 0.0 and s.est_gpu_hours == 0.0 for s in reused
            ),
        }

        # G2: cross-manifest uniqueness (all parseable run manifests, incl. p0local).
        owners: dict[str, list[tuple[str, str]]] = {}  # run_id -> [(file, hash)]
        for p in sorted(MANIFESTS.glob("*.yaml")):
            try:
                _, specs = load_manifest(p)
            except ManifestError:
                continue
            for s in specs:
                owners.setdefault(s.run_id, []).append((p.name, s.config_hash()))
        reused_ids = {s.run_id for s in reused}
        bad_dups, bad_reuse_hash = [], []
        for rid, occ in owners.items():
            if len(occ) == 1:
                continue
            files = {f for f, _ in occ}
            if rid in reused_ids and files == {"p1.yaml", "p2.yaml"}:
                if len({h for _, h in occ}) != 1:
                    bad_reuse_hash.append(rid)
            else:
                bad_dups.append((rid, sorted(files)))
        m["g2"] = {
            "n_run_ids": len(owners),
            "reused_ids_n": len(reused_ids),
            "bad_dups": bad_dups[:5],
            "n_bad_dups": len(bad_dups),
            "bad_reuse_hash": bad_reuse_hash[:5],
        }

        # G3: real rows (poll briefly), plus the tampered-copy fixture.
        deadline = time.time() + RESULTS_POLL_S
        while not RESULTS.exists() and time.time() < deadline:
            time.sleep(1.0)
        real: dict = {"present": RESULTS.exists()}
        if real["present"]:
            import json

            rows = [
                json.loads(line)
                for line in RESULTS.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            bad = _check_rows(rows, _hash_table(MANIFESTS))
            real.update(n_rows=len(rows), n_bad=len(bad), bad=bad[:5])
        m["g3_real"] = real

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            doc = yaml.safe_load((MANIFESTS / "p0.yaml").read_text(encoding="utf-8"))
            target = doc["runs"][0]
            _, p0_specs = load_manifest(MANIFESTS / "p0.yaml")
            row = {"run_id": p0_specs[0].run_id, "config_hash": p0_specs[0].config_hash()}
            clean_ok = not _check_rows([row], _hash_table(MANIFESTS))
            # Tamper a science field in a manifest copy; the row's recorded
            # hash must no longer match the recomputed one.
            target["tokens_per_param"] = float(target["tokens_per_param"]) + 1.0
            tdir = tmp / "manifests"
            tdir.mkdir()
            (tdir / "p0.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
            tampered_bad = _check_rows([row], _hash_table(tdir))
            m["g3_fixture"] = {
                "clean_ok": clean_ok,
                "tamper_detected": len(tampered_bad) == 1
                and tampered_bad[0]["reason"] == "config_hash mismatch",
            }

            # G4: ledger behavior in a temp dir.
            led = BudgetLedger(tmp / "ledger.json")

            def spec(rid: str, phase: str, cost: float, size: str = "60m") -> RunSpec:
                return RunSpec(run_id=rid, phase=phase, size=size, precision="bf16",
                               tokens_per_param=20.0, est_cost_usd=cost)

            over_raised = False
            try:
                led.check_and_reserve(spec("probe-over-cap", "p0", 1e6))
            except BudgetExceeded:
                over_raised = True
            m["g4_behavior"] = {
                "over_cap_raised": over_raised,
                "big_requires_human": led.check(spec("probe-big", "p2", 250.0))["requires_human"],
                "small_requires_human": led.check(spec("probe-small", "p2", 50.0))["requires_human"],
                "node8x_requires_human": led.check(
                    spec("probe-8x", "p2", 50.0, size="490m")
                )["requires_human"],
            }

        # G4: caps are literal constants in the source text.
        src = Path(inspect.getsourcefile(ledger_mod)).read_text(encoding="utf-8")
        tree = ast.parse(src)
        caps_literal = tier3_literal = False
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            for t in targets:
                if getattr(t, "id", None) == "CAPS":
                    caps_literal = _numeric_literal(node.value)
                elif getattr(t, "id", None) == "TIER3_COST_USD":
                    tier3_literal = _numeric_literal(node.value)
        m["g4_source"] = {
            "caps_literal": caps_literal,
            "tier3_literal": tier3_literal,
            "no_env": "environ" not in src and "getenv" not in src,
            "caps_runtime_numeric": all(
                isinstance(v, (int, float))
                for v in [ledger_mod.CAPS["per_day_usd"], ledger_mod.CAPS["total_usd"]]
                + list(ledger_mod.CAPS["per_phase_usd"].values())
            )
            and isinstance(ledger_mod.TIER3_COST_USD, (int, float)),
        }

        # G5: re-derived totals from raw manifest fields.
        totals: dict[str, dict] = {}
        for ph in PLAN_COUNTS:
            doc = yaml.safe_load((MANIFESTS / f"{ph}.yaml").read_text(encoding="utf-8"))
            totals[ph] = {
                "usd": sum(float(r["est_cost_usd"]) for r in doc["runs"]),
                "gpu_h": sum(float(r["est_gpu_hours"]) for r in doc["runs"]),
            }
        m["g5"] = {
            "per_phase": {k: round(v["usd"], 1) for k, v in totals.items()},
            "total_usd": sum(v["usd"] for v in totals.values()),
            "p2_gpu_h": totals["p2"]["gpu_h"],
        }
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)

        g1 = m["g1"]
        count_bad = {
            ph: (c["n"], PLAN_COUNTS[ph])
            for ph, c in g1["counts"].items()
            if c["n"] != PLAN_COUNTS[ph] or not c["phases_ok"]
        }
        g1_ok = (
            not count_bad
            and g1["reused_n"] == P2_REUSED_FROM_P1
            and g1["reused_tagged"]
            and g1["reused_zero_cost"]
        )
        out = [
            GateOutcome(
                "G1", specs["G1"], g1_ok,
                detail=(
                    f"counts {'match plan' if not count_bad else f'MISMATCH {count_bad}'}; "
                    f"p2 reused rows: {g1['reused_n']}/{P2_REUSED_FROM_P1}, "
                    f"tagged={g1['reused_tagged']}, zero_cost={g1['reused_zero_cost']}"
                ),
            )
        ]

        g2 = m["g2"]
        g2_ok = g2["n_bad_dups"] == 0 and not g2["bad_reuse_hash"]
        out.append(
            GateOutcome(
                "G2", specs["G2"], g2_ok,
                detail=(
                    f"{g2['n_run_ids']} run_ids, {g2['reused_ids_n']} documented reuses; "
                    + ("no undocumented duplicates, reuse hashes identical" if g2_ok
                       else f"bad_dups={g2['bad_dups']} reuse_hash_mismatch={g2['bad_reuse_hash']}")
                ),
            )
        )

        fx, real = m["g3_fixture"], m["g3_real"]
        real_ok = (not real["present"]) or real["n_bad"] == 0
        g3_ok = fx["clean_ok"] and fx["tamper_detected"] and real_ok
        real_note = (
            f"results.jsonl: {real['n_rows']} rows, {real['n_bad']} bad ({real.get('bad')})"
            if real["present"]
            else "results.jsonl absent after poll -> vacuous on real rows"
        )
        out.append(
            GateOutcome(
                "G3", specs["G3"], g3_ok,
                detail=f"fixture: clean_ok={fx['clean_ok']}, "
                       f"tamper_detected={fx['tamper_detected']}; {real_note}",
            )
        )

        gb, gs = m["g4_behavior"], m["g4_source"]
        g4_ok = (
            gb["over_cap_raised"]
            and gb["big_requires_human"]
            and not gb["small_requires_human"]
            and gb["node8x_requires_human"]
            and gs["caps_literal"]
            and gs["tier3_literal"]
            and gs["no_env"]
            and gs["caps_runtime_numeric"]
        )
        out.append(
            GateOutcome(
                "G4", specs["G4"], g4_ok,
                detail=f"behavior={gb}; source={gs}",
            )
        )

        g5 = m["g5"]
        lo, hi = PLAN_TOTAL_USD
        h_lo, h_hi = 0.6 * PLAN_P2_GPU_H, 1.4 * PLAN_P2_GPU_H
        g5_ok = lo <= g5["total_usd"] <= hi and h_lo <= g5["p2_gpu_h"] <= h_hi
        out.append(
            GateOutcome(
                "G5", specs["G5"], g5_ok,
                detail=(
                    f"total ${g5['total_usd']:,.0f} (window ${lo:,.0f}-${hi:,.0f}); "
                    f"p2 {g5['p2_gpu_h']:,.0f} GPU-h (band {h_lo:,.0f}-{h_hi:,.0f}); "
                    f"per-phase USD {g5['per_phase']}"
                ),
            )
        )
        return out
