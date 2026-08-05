"""Tests for the manifest / launcher / ledger / results subsystem.

Network-free, torch-free (ModelConfig math only). Run:
  PYTHONPATH=src python -m pytest tests/test_manifest.py -q
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from logos.config import Precision, RunSpec
from logos.manifest import generate
from logos.manifest.ledger import CAPS, BudgetExceeded, BudgetLedger
from logos.manifest.launcher import Launcher, RunPodExecutor
from logos.manifest.schema import ManifestError, load_manifest, save_manifest
from logos.results.store import (
    append_result,
    check_hash,
    gap_vs_sigma,
    load_results,
    seed_sigma,
)

# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def manifests_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("manifests")
    generate.main(["--out", str(out)])
    return out


def _phase_specs(manifests_dir, phase):
    _, specs = load_manifest(manifests_dir / f"{phase}.yaml")
    return specs


def _spec(run_id="t-25m", phase="p2", size="25m", precision="1.58", tp=20.0, **kw):
    s = RunSpec(
        run_id=run_id,
        phase=phase,
        size=size,
        precision=precision,
        tokens_per_param=tp,
        **kw,
    )
    if not s.est_cost_usd:
        try:
            s.est_gpu_hours, s.est_cost_usd = generate.estimate(s)
        except (KeyError, ValueError):
            pass  # invalid specs (validation tests) keep the 0.0 defaults
    return s


# --------------------------------------------------------------- 1. generate


def test_phase_run_counts(manifests_dir):
    counts = {
        "p0": 16,
        "p1": 44,  # 25 lr probes + 4 ffn ablation + 15 gap study
        "p2": 130,  # 45 + 45 + 15 reused + 15 + 10
        "p2ext": 3,
        "p3": 6,
        "p4": 9,  # 6 kv-qat + 2 ctx_ext + 1 gqa
        "p5": 1,
    }
    for phase, expected in counts.items():
        assert len(_phase_specs(manifests_dir, phase)) == expected, phase


def test_p1_breakdown_and_reuse(manifests_dir):
    p1 = _phase_specs(manifests_dir, "p1")
    assert sum("lr_probe" in s.tags for s in p1) == 25
    assert sum("ffn_ablation" in s.tags for s in p1) == 4
    gap = [s for s in p1 if "gap_study" in s.tags]
    assert len(gap) == 15
    p2 = _phase_specs(manifests_dir, "p2")
    reused = [s for s in p2 if "reused_from_p1" in s.tags]
    assert len(reused) == 15
    assert {s.run_id for s in reused} == {s.run_id for s in gap}
    assert all(s.est_cost_usd == 0.0 and s.est_gpu_hours == 0.0 for s in reused)
    # lr probes use the total_tokens override with tokens_per_param derived
    probe = next(s for s in p1 if "lr_probe" in s.tags)
    assert probe.total_tokens == 1_000_000_000
    assert probe.tokens_per_param > 0


def test_p0_seed_split_and_notes(manifests_dir):
    p0 = _phase_specs(manifests_dir, "p0")
    at20 = [s for s in p0 if s.tokens_per_param == 20]
    at320 = [s for s in p0 if s.tokens_per_param == 320]
    assert len(at20) == 4 and all(s.seed == 0 for s in at20)
    assert len(at320) == 12 and {s.seed for s in at320} == {0, 1, 2}
    assert all("seeds concentrated at 320x" in s.notes for s in p0)


def test_p3_p4_p5_details(manifests_dir):
    meta3, p3 = load_manifest(manifests_dir / "p3.yaml")
    assert meta3["prediction_protocol"].startswith("freeze fitted law")
    assert {(s.precision, s.tokens_per_param) for s in p3} == {
        (p, t) for p in ("1.58", "4", "bf16") for t in (20.0, 80.0)
    }
    meta4, p4 = load_manifest(manifests_dir / "p4.yaml")
    assert meta4["eval_jobs"]["kv_posthoc_sweep"]["kv_bits"] == [8, 4, 3, 2]
    assert sum("ctx_ext" in s.tags for s in p4) == 2
    assert sum(s.gqa_ratio == 8 for s in p4) == 1
    assert all(s.kv_qat_bits in (8, 4) for s in p4 if "kv_qat" in s.tags)
    _, p5 = load_manifest(manifests_dir / "p5.yaml")
    (cap,) = p5
    assert cap.total_tokens == 300_000_000_000
    assert "capstone" in cap.tags and cap.precision == "1.58"
    assert "bitnet2stage" in cap.notes and "DPO" in cap.notes
    # lr_rules.yaml exists with the frozen-protocol header
    text = (manifests_dir / "lr_rules.yaml").read_text(encoding="utf-8")
    assert "REPLACED by P1 probe results" in text
    assert (manifests_dir / "summary.md").exists()


# --------------------------------------------------------------- 2. round-trip


def test_manifest_round_trip(manifests_dir, tmp_path):
    for phase in ("p0", "p1", "p2", "p4", "p5"):
        meta, specs = load_manifest(manifests_dir / f"{phase}.yaml")
        p2 = tmp_path / f"{phase}-rt.yaml"
        save_manifest(specs, p2, meta)
        meta2, specs2 = load_manifest(p2)
        assert [s.config_hash() for s in specs] == [s.config_hash() for s in specs2]
        assert specs == specs2


def test_manifest_validation(tmp_path):
    good = _spec()
    with pytest.raises(ManifestError):
        save_manifest([good, good], tmp_path / "dup.yaml", {})  # duplicate run_id
    with pytest.raises(ManifestError):
        save_manifest([_spec(size="33m")], tmp_path / "size.yaml", {})
    with pytest.raises(ManifestError):
        save_manifest([_spec(precision="7")], tmp_path / "prec.yaml", {})
    with pytest.raises(ManifestError):
        save_manifest([_spec(tp=0.0)], tmp_path / "tp.yaml", {})


# --------------------------------------------------------------- 3. ledger


def test_ledger_caps_and_tier3(tmp_path):
    led = BudgetLedger(tmp_path / "ledger.json")
    small = _spec(run_id="a", est_gpu_hours=1.0, est_cost_usd=50.0)
    res = led.check_and_reserve(small, day="2026-08-05")
    assert not res["requires_human"]
    # >$200 flags requires_human (Tier-3 gate)
    big = _spec(run_id="b", est_gpu_hours=90.0, est_cost_usd=250.0)
    assert led.check(big, day="2026-08-06")["requires_human"]
    # any 8x-node size flags requires_human even when cheap
    node8 = _spec(run_id="c", size="250m", est_cost_usd=10.0)
    assert led.check(node8, day="2026-08-06")["requires_human"]
    # per-day accumulation: 50 + 250 booked, next 150 breaks the $400/day cap
    led.check_and_reserve(big, day="2026-08-05")
    assert led.state["spent_by_day"]["2026-08-05"] == pytest.approx(300.0)
    over_day = _spec(run_id="d", est_cost_usd=150.0)
    with pytest.raises(BudgetExceeded):
        led.check_and_reserve(over_day, day="2026-08-05")
    led.check_and_reserve(over_day, day="2026-08-06")  # fresh day is fine
    # per-phase cap: p0 capped at $400
    led2 = BudgetLedger(tmp_path / "ledger2.json")
    p0run = _spec(run_id="e", phase="p0", est_cost_usd=350.0)
    led2.check_and_reserve(p0run, day="2026-08-05")
    with pytest.raises(BudgetExceeded):
        led2.check_and_reserve(
            _spec(run_id="f", phase="p0", est_cost_usd=100.0), day="2026-08-06"
        )
    assert CAPS["per_phase_usd"]["p0"] == 400.0  # caps are code, not config


def test_ledger_reconcile_and_persist(tmp_path):
    path = tmp_path / "ledger.json"
    led = BudgetLedger(path)
    led.check_and_reserve(_spec(run_id="a", est_cost_usd=100.0), day="2026-08-05")
    led.record_actual("a", usd=80.0, gpu_hours=28.5)
    led = BudgetLedger(path)  # reload from disk
    assert led.state["spent_usd_total"] == pytest.approx(80.0)
    assert led.state["spent_by_day"]["2026-08-05"] == pytest.approx(80.0)
    assert led.state["entries"][0]["gpu_hours"] == 28.5


# --------------------------------------------------------------- 4. launcher


def _mini_manifest(tmp_path):
    specs = [
        _spec(run_id="exp-60m", size="60m", tp=320.0, seed=0),
        _spec(run_id="cheap-25m-s1", size="25m", tp=20.0, seed=1),
        _spec(run_id="cheap-25m-s0", size="25m", tp=20.0, seed=0),
        _spec(run_id="mid-60m", size="60m", tp=20.0, seed=0),
    ]
    path = tmp_path / "mini.yaml"
    save_manifest(specs, path, {"phase": "p2"})
    return path, specs


def test_launcher_ordering_and_skip(tmp_path):
    manifest, _ = _mini_manifest(tmp_path)
    runs = tmp_path / "runs"
    lch = Launcher(manifest, runs)
    # cheap first; equal cost broken by seed
    assert lch.next_pending().run_id == "cheap-25m-s0"
    (runs / "cheap-25m-s0").mkdir(parents=True)
    (runs / "cheap-25m-s0" / "status.json").write_text(json.dumps({"status": "complete"}))
    assert lch.next_pending().run_id == "cheap-25m-s1"
    rec = lch.launch_next(dry_run=True)
    assert rec["action"] == "would_launch" and rec["run_id"] == "cheap-25m-s1"
    assert not (runs / "cheap-25m-s1").exists()  # dry run: no side effects


def test_launcher_budget_and_tier3(tmp_path):
    specs = [
        _spec(run_id="big-490m", size="490m", tp=80.0),  # 8x node -> human gate
    ]
    manifest = tmp_path / "m.yaml"
    save_manifest(specs, manifest, {"phase": "p2"})
    led = BudgetLedger(tmp_path / "ledger.json")
    lch = Launcher(manifest, tmp_path / "runs", ledger=led)
    rec = lch.launch_next(dry_run=True)
    assert rec["action"] == "hold_for_human" and rec["requires_human"]


def test_launcher_kill_criteria(tmp_path):
    manifest, specs = _mini_manifest(tmp_path)
    runs = tmp_path / "runs"
    lch = Launcher(manifest, runs)
    # diverged run
    d = runs / "cheap-25m-s0"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"status": "running"}))
    with (d / "metrics.jsonl").open("w") as f:
        f.write(json.dumps({"step": 10, "loss": 3.1, "wall_s": 60}) + "\n")
        f.write(json.dumps({"step": 20, "loss": 9.9, "diverged": True, "wall_s": 120}) + "\n")
    # wall-clock blowout: scheduled = est_gpu_hours*3600/1 gpu; exceed 3x
    w = runs / "mid-60m"
    w.mkdir(parents=True)
    (w / "status.json").write_text(json.dumps({"status": "running"}))
    spec = next(s for s in specs if s.run_id == "mid-60m")
    blowout = spec.max_wall_clock_mult * spec.est_gpu_hours * 3600 * 1.5
    (w / "metrics.jsonl").write_text(json.dumps({"step": 5, "loss": 4.0, "wall_s": blowout}) + "\n")
    killed = lch.enforce_kill_criteria()
    assert {k["run_id"] for k in killed} == {"cheap-25m-s0", "mid-60m"}
    for rid in ("cheap-25m-s0", "mid-60m"):
        assert (runs / rid / "KILL").exists()
        assert json.loads((runs / rid / "status.json").read_text())["status"] == "killed"
    # killed runs no longer pending
    assert lch.next_pending().run_id == "cheap-25m-s1"


def test_runpod_executor_stub():
    ex = RunPodExecutor()
    from pathlib import Path

    small = ex.build_payload(_spec(run_id="s", size="60m"), Path("p2.yaml"))
    assert small["gpu_count"] == 1 and small["cloud"] == "community"
    big = ex.build_payload(_spec(run_id="b", size="490m"), Path("p2.yaml"))
    assert big["gpu_count"] == 8 and big["gpu_type"] == "H100 SXM"
    with pytest.raises(NotImplementedError, match="ops layer"):
        ex.launch(_spec(run_id="s", size="60m"), Path("p2.yaml"), Path("runs/s"))


# --------------------------------------------------------------- 5. results


def _fake_results(tmp_path, manifests_dir):
    """Synthetic results with known per-seed spread at 25m/20x."""
    path = tmp_path / "results.jsonl"
    _, p2 = load_manifest(manifests_dir / "p2.yaml")
    base = {"1.58": 1.10, "bf16": 1.00}
    offsets = {0: -0.01, 1: 0.0, 2: 0.01}  # std = 0.01 across 3 seeds
    for s in p2:
        if s.size != "25m" or s.tokens_per_param != 20 or s.precision not in base:
            continue
        bpb = base[s.precision] + offsets[s.seed]
        append_result(
            path,
            s,
            {
                "bpb_val1": bpb,
                "bpb_val2": bpb + 0.02,
                "downstream": {"arc_e": 0.31},
                "packed_bytes": 12_345_678,
                "gpu_hours": 0.1,
                "cost_usd": 0.28,
                "wall_s": 360.0,
            },
        )
    return path


def test_seed_sigma_and_gap(tmp_path, manifests_dir):
    path = _fake_results(tmp_path, manifests_dir)
    df = load_results(path)
    assert len(df) == 6
    sig = seed_sigma(df)
    row = sig[(sig["size"] == "25m") & (sig["precision"] == "1.58")].iloc[0]
    assert row["n_seeds"] == 3
    assert row["sigma_bpb"] == pytest.approx(0.01, rel=1e-6)
    gaps = gap_vs_sigma(df)
    g = gaps.iloc[0]
    assert g["gap_bpb"] == pytest.approx(0.10, abs=1e-9)
    assert g["exceeds_2sigma"] and g["verdict"] == "significant"  # 0.10 > 2*0.01


def test_gap_within_noise():
    rows = []
    for prec, base in (("1.58", 1.005), ("bf16", 1.000)):
        for seed, off in ((0, -0.02), (1, 0.0), (2, 0.02)):
            rows.append(
                {
                    "size": "60m",
                    "precision": prec,
                    "tokens_per_param": 20.0,
                    "seed": seed,
                    "bpb_val1": base + off,
                }
            )
    gaps = gap_vs_sigma(pd.DataFrame(rows))
    assert not gaps.iloc[0]["exceeds_2sigma"]
    assert gaps.iloc[0]["verdict"] == "within_noise"


def test_check_hash_catches_tamper(tmp_path, manifests_dir):
    path = _fake_results(tmp_path, manifests_dir)
    df = load_results(path)
    assert check_hash(df, manifests_dir).empty  # clean rows verify
    df.loc[0, "config_hash"] = "deadbeefdeadbeef"
    bad = check_hash(df, manifests_dir)
    assert len(bad) == 1
    assert bad.iloc[0]["reason"] == "config_hash mismatch"
    df.loc[1, "run_id"] = "not-a-run"
    assert len(check_hash(df, manifests_dir)) == 2
