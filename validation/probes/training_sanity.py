"""Probe training_sanity — PLAN.md s5: "a trainer we trust".

Targets the background micro-P0 runs in runs/local_p0 (curves recomputed
from metrics.jsonl, never from summaries), plus the probe's own CPU
kill-and-resume fixture mirroring the P0 spot-hardening requirement
("verified by killing a live run"). If no background artifacts appear, a
tiny 30-step CPU run stands in and the same curve gates apply (noted).
"""

from __future__ import annotations

import json
import math
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from logos.config import ModelConfig, Precision, RunSpec, TrainConfig
from logos.train.trainer import TrainerExtras, train
from validation.base import GateOutcome, Probe

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "runs" / "local_p0"
EXPECTED_RUNS = (
    "local-micro-1.58-s0", "local-micro-1.58-s1",
    "local-micro-bf16-s0", "local-micro-bf16-s1",
    "local-micro-2-s0", "local-micro-3-s0", "local-micro-4-s0",
)
POLL_S = 120.0
DECREASE_MIN = 0.05  # G2: last-quartile mean < first-quartile mean by >5%
ORDER_TOL = 0.15  # G5 adjacent-pair tolerance (loss units)
RESUME_TOL = 1e-4


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _fixture_data(tmp: Path) -> Path:
    """Low-entropy shards in prepare.py's on-disk schema: a 16-symbol
    alphabet inside a 512 vocab, so a 30-step CPU run visibly learns."""
    rng = np.random.default_rng(1234)
    dd = tmp / "data"
    dd.mkdir(parents=True)
    shards = []
    for i in range(2):
        toks = rng.integers(3, 19, size=16_384).astype(np.uint16)
        fname = f"shard_{i:05d}.bin"
        toks.tofile(dd / fname)
        shards.append({"file": fname, "tokens": 16_384, "utf8_bytes": 16_384, "docs": 1})
    val = {}
    for name in ("val1", "val2"):
        toks = rng.integers(3, 19, size=1024).astype(np.uint16)
        toks.tofile(dd / f"{name}.bin")
        val[name] = {"file": f"{name}.bin", "tokens": 1024, "utf8_bytes": 1024, "docs": 1}
    index = {
        "dataset": "synthetic-lowentropy", "dataset_config": None, "tokenizer": "synthetic",
        "vocab_size": 512, "eos_id": 2, "seed": 1234, "shards": shards,
        "total_tokens": 32_768, "val": val,
    }
    (dd / "index.json").write_text(json.dumps(index, indent=2))
    return dd


def _fixture_cfgs(n_steps: int) -> tuple[ModelConfig, TrainConfig, RunSpec]:
    mcfg = ModelConfig(
        d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=256,
        vocab_size=512, max_seq_len=64, precision=Precision.BF16,
    )
    tcfg = TrainConfig(
        lr=3e-3, total_tokens=n_steps * 1024, batch_tokens=1024, seq_len=64,
        checkpoint_interval_s=10**9,
    )
    spec = RunSpec(
        run_id=f"probe-fixture-{n_steps}", phase="p0", size="micro",
        precision="bf16", tokens_per_param=1.0, total_tokens=n_steps * 1024,
    )
    return mcfg, tcfg, spec


def _train_leg(spec, data_dir, run_dir, mcfg, tcfg, max_steps=None) -> dict:
    try:
        status = train(
            spec, data_dir, run_dir, device="cpu",
            extras=TrainerExtras(model_config=mcfg, train_config=tcfg, max_steps=max_steps),
        )
        return {"ok": True, "status": status}
    except (SystemExit, Exception):
        return {"ok": False, "error": traceback.format_exc(limit=6)}


# ---------------------------------------------------------------------------
# metrics analysis (recomputed from metrics.jsonl rows only)
# ---------------------------------------------------------------------------


def _read_rows(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _analyze(rows: list[dict]) -> dict:
    steps = [r["step"] for r in rows]
    replays = len(steps) - len(set(steps))
    by_step = {r["step"]: r for r in rows}  # last occurrence wins (resume re-log)
    ordered = [by_step[s] for s in sorted(by_step)]
    s = [r["step"] for r in ordered]
    losses = [r["loss"] for r in ordered]
    interval = s[0] if s else 0
    diffs = [b - a for a, b in zip(s, s[1:])]
    gapless = (
        bool(s)
        and interval > 0
        and all(d == interval for d in diffs[:-1])
        and (not diffs or 0 < diffs[-1] <= interval)
    )
    bt = ordered[0]["tokens"] // interval if ordered and interval else 0
    tokens_exact = (
        bool(ordered)
        and interval > 0
        and ordered[0]["tokens"] % interval == 0
        and all(r["tokens"] == r["step"] * bt for r in ordered)
    )
    q = max(1, len(losses) // 4)
    first_q, last_q = float(np.mean(losses[:q])), float(np.mean(losses[-q:]))
    return {
        "n_rows": len(rows),
        "replayed_steps": replays,
        "finite": all(math.isfinite(x) for x in losses),
        "gapless": gapless,
        "interval": interval,
        "batch_tokens": bt,
        "tokens_exact": tokens_exact,
        "first_q_mean": first_q,
        "last_q_mean": last_q,
        "decrease": (first_q - last_q) / first_q if first_q else 0.0,
        "last_step": s[-1] if s else None,
        "last_loss": losses[-1] if losses else None,
    }


class TrainingSanityProbe(Probe):
    name = "training_sanity"
    description = "loss curves sane, token accounting exact, resume == uninterrupted (PLAN.md s5)"
    gate_specs = [
        ("G1", "every available run: all metrics.jsonl losses finite, gapless log-interval "
               "step sequence, tokens == step * batch_tokens exactly"),
        ("G2", "every complete run learns: last-quartile mean loss < first-quartile mean "
               "by >5%"),
        ("G3", "every complete run's status.json says complete with final_loss (and step) "
               "exactly matching the last metrics row"),
        ("G4", "own CPU fixture: 20-step run interrupted at step 10 via max_steps, resumed "
               "from its checkpoint, matches the uninterrupted twin within 1e-4 "
               "(weights + final loss) — the P0 kill-and-resume requirement"),
        ("G5", "when >=5 s0 arms complete: end-of-training loss ordering satisfies "
               "bf16 <= {4,3} <= 2 <= 1.58 within 0.15 adjacent-pair tolerance "
               "(vacuous otherwise, ordering reported)"),
    ]

    def collect(self) -> dict:
        m: dict = {}

        # ---- background runs: poll for all 7, then take what exists ----
        def n_complete() -> int:
            k = 0
            for rid in EXPECTED_RUNS:
                p = RUNS_DIR / rid / "status.json"
                try:
                    if p.exists() and json.loads(p.read_text()).get("status") == "complete":
                        k += 1
                except (OSError, json.JSONDecodeError):
                    pass
            return k

        deadline = time.time() + POLL_S
        while n_complete() < len(EXPECTED_RUNS) and time.time() < deadline:
            time.sleep(5.0)

        runs: dict[str, dict] = {}
        for rid in EXPECTED_RUNS:
            rd = RUNS_DIR / rid
            rows = _read_rows(rd)
            if not rows:
                continue
            status = None
            sp = rd / "status.json"
            if sp.exists():
                try:
                    status = json.loads(sp.read_text())
                except json.JSONDecodeError:
                    status = None
            runs[rid] = {"analysis": _analyze(rows), "status": status}
        m["background"] = runs
        m["n_complete"] = n_complete()
        m["fallback_mode"] = not runs

        # ignore_cleanup_errors: trainer's TokenLoader memmaps the .bin files (Windows locks).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            tmp = Path(td)
            data_dir = _fixture_data(tmp)

            # ---- fallback fixture: 30-step run stands in for the curve gates ----
            if m["fallback_mode"]:
                mcfg, tcfg, spec = _fixture_cfgs(30)
                leg = _train_leg(spec, data_dir, tmp / "fixture30", mcfg, tcfg)
                if leg["ok"]:
                    m["fixture30"] = {
                        "analysis": _analyze(_read_rows(tmp / "fixture30")),
                        "status": json.loads((tmp / "fixture30" / "status.json").read_text()),
                    }
                else:
                    m["fixture30"] = {"error": leg["error"]}

            # ---- G4 resume fixture (always on our own runs) ----
            mcfg, tcfg, spec = _fixture_cfgs(20)
            a = _train_leg(spec, data_dir, tmp / "twin_a", mcfg, tcfg)  # uninterrupted
            b1 = _train_leg(spec, data_dir, tmp / "twin_b", mcfg, tcfg, max_steps=10)
            b2 = _train_leg(spec, data_dir, tmp / "twin_b", mcfg, tcfg)  # auto-resume
            if a["ok"] and b1["ok"] and b2["ok"]:
                sa = torch.load(tmp / "twin_a" / "ckpt_latest.pt",
                                map_location="cpu", weights_only=False)
                sb = torch.load(tmp / "twin_b" / "ckpt_latest.pt",
                                map_location="cpu", weights_only=False)
                w_diff = max(
                    float((sa["model"][k].float() - sb["model"][k].float()).abs().max())
                    for k in sa["model"]
                )
                m["resume"] = {
                    "ok": True,
                    "interrupted_at": b1["status"]["step"],
                    "steps": (sa["step"], sb["step"]),
                    "max_weight_diff": w_diff,
                    "final_loss_diff": abs(
                        a["status"]["final_loss"] - b2["status"]["final_loss"]
                    ),
                }
            else:
                m["resume"] = {
                    "ok": False,
                    "error": (a.get("error") or b1.get("error") or b2.get("error")),
                }
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)
        runs = dict(m["background"])
        note = f"{m['n_complete']}/7 background runs complete"
        if m["fallback_mode"]:
            note = "FALLBACK MODE: no background artifacts; 30-step CPU fixture stands in"
            fx = m.get("fixture30", {})
            if "analysis" in fx:
                runs["fixture-30step"] = fx
        complete = {
            rid: r for rid, r in runs.items()
            if r["status"] is not None and r["status"].get("status") == "complete"
        }

        # G1: all available runs.
        g1_bad = {
            rid: {k: a[k] for k in ("finite", "gapless", "tokens_exact", "replayed_steps")}
            for rid, r in runs.items()
            if not ((a := r["analysis"])["finite"] and a["gapless"] and a["tokens_exact"])
        }
        fixture_err = m.get("fixture30", {}).get("error") if m["fallback_mode"] else None
        g1_ok = bool(runs) and not g1_bad and fixture_err is None
        bts = {rid: r["analysis"]["batch_tokens"] for rid, r in runs.items()}
        out = [
            GateOutcome(
                "G1", specs["G1"], g1_ok,
                detail=f"{note}; {len(runs)} runs checked, batch_tokens={bts}"
                + (f"; FAILURES {g1_bad}" if g1_bad else "")
                + (f"; fixture error: {fixture_err}" if fixture_err else "")
                + ("; no runs at all" if not runs else ""),
            )
        ]

        # G2: complete runs learn.
        decreases = {rid: round(r["analysis"]["decrease"], 4) for rid, r in complete.items()}
        g2_bad = {rid: d for rid, d in decreases.items() if not d > DECREASE_MIN}
        skipped = sorted(set(runs) - set(complete))
        out.append(
            GateOutcome(
                "G2", specs["G2"], bool(complete) and not g2_bad,
                detail=f"loss decrease per complete run: {decreases}"
                + (f"; below 5%: {g2_bad}" if g2_bad else "")
                + (f"; skipped (not complete): {skipped}" if skipped else "")
                + ("; no complete runs" if not complete else ""),
            )
        )

        # G3: status.json vs last metrics row, exact.
        g3_bad = {}
        for rid, r in complete.items():
            a, st = r["analysis"], r["status"]
            if not (st.get("final_loss") == a["last_loss"] and st.get("step") == a["last_step"]):
                g3_bad[rid] = {
                    "status": (st.get("step"), st.get("final_loss")),
                    "metrics": (a["last_step"], a["last_loss"]),
                }
        out.append(
            GateOutcome(
                "G3", specs["G3"], bool(complete) and not g3_bad,
                detail=f"{len(complete)} complete runs, exact final_loss/step match"
                if complete and not g3_bad
                else f"mismatches: {g3_bad}" + ("; no complete runs" if not complete else ""),
            )
        )

        # G4: resume fixture.
        res = m["resume"]
        if res["ok"]:
            g4_ok = (
                res["max_weight_diff"] <= RESUME_TOL
                and res["final_loss_diff"] <= RESUME_TOL
                and res["steps"][0] == res["steps"][1]
            )
            detail = (
                f"interrupted at step {res['interrupted_at']}, both twins at step "
                f"{res['steps']}; max |dW|={res['max_weight_diff']:.3g}, "
                f"|dloss|={res['final_loss_diff']:.3g} (tol {RESUME_TOL})"
            )
        else:
            g4_ok, detail = False, f"fixture training failed: {res['error']}"
        out.append(GateOutcome("G4", specs["G4"], g4_ok, detail=detail))

        # G5: cross-arm ordering on the background s0 runs.
        arm_loss = {
            rid.split("-")[2]: r["analysis"]["last_q_mean"]
            for rid, r in complete.items()
            if rid.startswith("local-micro-") and rid.endswith("-s0")
        }
        if len(arm_loss) >= 5:
            pairs = [("bf16", "4"), ("bf16", "3"), ("4", "2"), ("3", "2"), ("2", "1.58")]
            viols = [
                f"{lo}({arm_loss[lo]:.4f}) > {hi}({arm_loss[hi]:.4f}) + {ORDER_TOL}"
                for lo, hi in pairs
                if not arm_loss[lo] <= arm_loss[hi] + ORDER_TOL
            ]
            ordering = ", ".join(f"{k}={v:.4f}" for k, v in sorted(arm_loss.items(), key=lambda kv: kv[1]))
            out.append(
                GateOutcome(
                    "G5", specs["G5"], not viols,
                    detail=f"end-of-training losses (asc): {ordering}"
                    + (f"; VIOLATIONS: {viols}" if viols else "; weak monotonicity holds"),
                )
            )
        else:
            out.append(
                GateOutcome(
                    "G5", specs["G5"], True,
                    detail=f"vacuous: only {len(arm_loss)} s0 arms complete (<5); "
                    f"losses so far: { {k: round(v, 4) for k, v in arm_loss.items()} }",
                )
            )
        return out
