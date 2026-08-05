"""Probe: statistical discipline of the results store (PLAN.md s3 principle 4).

Targets `results/results.jsonl` + `logos.results.store`. Every sigma/gap the
store reports is recomputed by hand in plain Python (no pandas) from the raw
JSONL rows; nothing may be flagged significant without a known sigma. G1-G3
run on a synthetic in-probe fixture ALWAYS (so the gates bite even before the
background micro-P0 lands) and additionally on the real results file.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
import warnings
from pathlib import Path

from logos.config import make_model
from logos.results.store import gap_vs_sigma, load_results, seed_sigma
from validation.base import GateOutcome, Probe

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "results.jsonl"
POLL_S, POLL_INTERVAL_S, MIN_MICRO_ROWS = 120.0, 5.0, 7
BPB_LO, BPB_HI = 0.5, 9.0  # sane byte-level micro-run range, bits/byte
LOW, HIGH = "1.58", "bf16"


def _fixture_rows() -> list[dict]:
    """Hand-built rows covering: 3-seed significant cell, 2-seed within-noise
    cell, mixed multi/single-seed cell, single-seed-both-arms cell with a big
    (bait) gap, and a second ladder size."""
    n = make_model("micro").n_nonemb
    cells = [
        # (size, precision, tpp, [per-seed bpb_val1])
        ("micro", "1.58", 20.0, [3.10, 3.14, 3.06]),
        ("micro", "bf16", 20.0, [2.90, 2.94, 2.86]),   # gap .20 > 2*.04 -> significant
        ("micro", "4", 20.0, [2.95]),                   # bystander arm
        ("micro", "1.58", 80.0, [2.80, 2.82]),
        ("micro", "bf16", 80.0, [2.78, 2.80]),          # gap .02 < 2*.0141 -> within noise
        ("micro", "1.58", 160.0, [2.75, 2.77]),
        ("micro", "bf16", 160.0, [2.60]),               # sigma from the multi-seed arm only
        ("micro", "1.58", 320.0, [2.70]),
        ("micro", "bf16", 320.0, [2.50]),               # n=1 both arms: NEVER significant
        ("25m", "1.58", 20.0, [3.00, 3.02]),
        ("25m", "bf16", 20.0, [2.70, 2.72]),
    ]
    rows = []
    for size, prec, tpp, vals in cells:
        nn = n if size == "micro" else make_model(size).n_nonemb
        for seed, v in enumerate(vals):
            rows.append({
                "run_id": f"fx-{size}-{prec}-{tpp:g}-s{seed}",
                "phase": "p0", "size": size, "n_nonemb": nn, "n_total": nn,
                "tokens": int(tpp * nn), "tokens_per_param": tpp,
                "precision": prec, "seed": seed, "ffn_type": "swiglu",
                "lr": 3e-3, "config_hash": "0" * 16,
                "bpb_val1": v, "bpb_val2": v + 0.01, "downstream": {},
                "packed_bytes": None, "gpu_hours": 0.0, "cost_usd": 0.0,
                "wall_s": 1.0, "status": "complete", "ts": 0.0,
            })
    return rows


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _hand_cells(rows: list[dict], col: str = "bpb_val1") -> dict[tuple, dict]:
    """Plain-Python per-(size, precision, tokens_per_param) mean / ddof=1 std,
    skipping non-finite values (mirrors pandas' skipna)."""
    vals: dict[tuple, list[float]] = {}
    for r in rows:
        v = r.get(col)
        if _finite(v):
            key = (r["size"], r["precision"], r["tokens_per_param"])
            vals.setdefault(key, []).append(float(v))
    out = {}
    for key, xs in vals.items():
        mean = sum(xs) / len(xs)
        sigma = (
            math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
            if len(xs) >= 2 else None
        )
        out[key] = {"mean": mean, "sigma": sigma, "n": len(xs)}
    return out


def _hand_gaps(cells: dict[tuple, dict]) -> dict[tuple, dict]:
    """Hand gap/sigma per (size, tpp): gap = mean(low) - mean(high); sigma =
    the larger of the two arms' seed sigmas (missing = single-seed arm
    ignored; both missing -> unknown); significant iff |gap| > 2 sigma."""
    out = {}
    for (size, prec, tpp), lo in cells.items():
        if prec != LOW:
            continue
        hi = cells.get((size, HIGH, tpp))
        if hi is None:
            continue
        sigmas = [c["sigma"] for c in (lo, hi) if c["sigma"] is not None]
        sigma = max(sigmas) if sigmas else None
        gap = lo["mean"] - hi["mean"]
        out[(size, tpp)] = {
            "gap": gap,
            "sigma": sigma,
            "exceeds": sigma is not None and abs(gap) > 2 * sigma,
        }
    return out


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _dataset(rows: list[dict]) -> dict:
    """Store outputs + hand recomputes for one row set."""
    import pandas as pd  # store's own dependency, used only to build its input

    df = pd.DataFrame(rows)
    sigma_rows = seed_sigma(df).to_dict("records")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN nanmax cells
        gap_rows = gap_vs_sigma(df).to_dict("records")
    return {
        "sigma_rows": sigma_rows,
        "gap_rows": gap_rows,
        "hand_cells": _hand_cells(rows),
        "hand_gaps": _hand_gaps(_hand_cells(rows)),
    }


class StatsDisciplineProbe(Probe):
    name = "stats_discipline"
    description = "seed-sigma math and 2-sigma significance discipline, recomputed by hand"
    gate_specs = [
        ("G1", "seed_sigma is a true ddof=1 std: matches the plain-Python "
               "recompute for every multi-seed cell; single-seed cells are NaN "
               "(fixture + real file when present)"),
        ("G2", "gap_vs_sigma's exceeds_2sigma flag is consistent with |gap| > "
               "2*max(arm sigmas) recomputed by hand (fixture + real file)"),
        ("G3", "2-sigma discipline: nothing is flagged significant where sigma "
               "is NaN/unknown; single-seed cells are never significant"),
        ("G4", "results-file audit (micro rows): finite positive sigma on "
               "multi-seed cells, all statuses complete, bpb in [0.5, 9], and "
               "bf16 mean <= ternary mean + 0.5 (vacuous if absent)"),
    ]

    def _poll_real_rows(self) -> tuple[list[dict], str]:
        """Wait up to POLL_S for the background micro-P0 to reach
        MIN_MICRO_ROWS micro rows; then take whatever is there."""
        t0 = time.time()
        note = ""
        while True:
            rows = []
            if RESULTS.exists():
                rows = [
                    json.loads(line)
                    for line in RESULTS.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            n_micro = sum(1 for r in rows if r.get("size") == "micro")
            if n_micro >= MIN_MICRO_ROWS:
                note = f"{len(rows)} rows ({n_micro} micro) after {time.time() - t0:.0f}s"
                return rows, note
            if time.time() - t0 >= POLL_S:
                note = (f"gave up after {POLL_S:.0f}s: "
                        + (f"{len(rows)} rows ({n_micro} micro)" if rows else "file absent"))
                return rows, note
            time.sleep(POLL_INTERVAL_S)

    def collect(self) -> dict:
        real_rows, poll_note = self._poll_real_rows()

        # Fixture round-trips through the store's own loader on a temp jsonl.
        tmpdir = Path(tempfile.mkdtemp(prefix="logos_stats_probe_"))
        try:
            fx_path = tmpdir / "results.jsonl"
            fx_path.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in _fixture_rows()) + "\n",
                encoding="utf-8",
            )
            fx_rows = [
                json.loads(line) for line in fx_path.read_text(encoding="utf-8").splitlines()
            ]
            assert len(load_results(fx_path)) == len(fx_rows)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        datasets = {"fixture": _dataset(fx_rows)}
        if real_rows:
            datasets["real"] = _dataset(real_rows)
        return {
            "datasets": datasets,
            "poll_note": poll_note,
            "micro_rows": [r for r in real_rows if r.get("size") == "micro"],
        }

    # ---- gate helpers (pure judgment) ----

    @staticmethod
    def _check_sigma(ds: dict) -> list[str]:
        errs = []
        by_key = {
            (r["size"], r["precision"], r["tokens_per_param"]): r for r in ds["sigma_rows"]
        }
        for key, hand in ds["hand_cells"].items():
            row = by_key.get(key)
            if row is None:
                errs.append(f"{key}: cell missing from seed_sigma")
                continue
            if int(row["n_seeds"]) != hand["n"]:
                errs.append(f"{key}: n_seeds {row['n_seeds']} != {hand['n']}")
            if not _close(float(row["mean_bpb"]), hand["mean"]):
                errs.append(f"{key}: mean {row['mean_bpb']} != {hand['mean']}")
            if hand["sigma"] is None:
                if not (isinstance(row["sigma_bpb"], float) and math.isnan(row["sigma_bpb"])):
                    errs.append(f"{key}: single-seed sigma should be NaN, got {row['sigma_bpb']}")
            elif not _close(float(row["sigma_bpb"]), hand["sigma"]):
                errs.append(f"{key}: sigma {row['sigma_bpb']} != {hand['sigma']}")
        return errs

    @staticmethod
    def _check_gaps(ds: dict) -> list[str]:
        errs = []
        by_key = {(r["size"], r["tokens_per_param"]): r for r in ds["gap_rows"]}
        for key, hand in ds["hand_gaps"].items():
            row = by_key.get(key)
            if row is None:
                errs.append(f"{key}: cell missing from gap_vs_sigma")
                continue
            if not _close(float(row["gap_bpb"]), hand["gap"]):
                errs.append(f"{key}: gap {row['gap_bpb']} != {hand['gap']}")
            if hand["sigma"] is not None and not _close(float(row["sigma_bpb"]), hand["sigma"]):
                errs.append(f"{key}: sigma {row['sigma_bpb']} != {hand['sigma']}")
            if bool(row["exceeds_2sigma"]) != hand["exceeds"]:
                errs.append(f"{key}: exceeds_2sigma {row['exceeds_2sigma']} != hand {hand['exceeds']}")
        return errs

    @staticmethod
    def _check_discipline(ds: dict) -> list[str]:
        errs = []
        for r in ds["gap_rows"]:
            key = (r["size"], r["tokens_per_param"])
            hand = ds["hand_gaps"].get(key, {})
            sigma_known = hand.get("sigma") is not None
            flagged = bool(r["exceeds_2sigma"]) or r["verdict"] == "significant"
            if flagged and not sigma_known:
                errs.append(f"{key}: flagged significant with unknown sigma (single-seed cells)")
            if (r["verdict"] == "significant") != bool(r["exceeds_2sigma"]):
                errs.append(f"{key}: verdict/flag inconsistent")
        return errs

    def gates(self, m: dict) -> list[GateOutcome]:
        out: list[GateOutcome] = []
        labels = sorted(m["datasets"])
        for gate, checker, spec in (
            ("G1", self._check_sigma, self.gate_specs[0][1]),
            ("G2", self._check_gaps, self.gate_specs[1][1]),
            ("G3", self._check_discipline, self.gate_specs[2][1]),
        ):
            errs = []
            for label in labels:
                errs += [f"[{label}] {e}" for e in checker(m["datasets"][label])]
            out.append(GateOutcome(
                gate, spec, passed=not errs,
                detail="; ".join(errs) if errs else f"datasets checked: {', '.join(labels)}",
            ))

        micro = m["micro_rows"]
        if not micro:
            out.append(GateOutcome(
                "G4", self.gate_specs[3][1], passed=True,
                detail=f"vacuous: no micro rows in results.jsonl ({m['poll_note']})",
            ))
            return out

        errs = []
        for r in micro:
            if r.get("status") != "complete":
                errs.append(f"{r['run_id']}: status {r.get('status')!r}")
            for col in ("bpb_val1", "bpb_val2"):
                v = r.get(col)
                if not _finite(v) or not (BPB_LO <= v <= BPB_HI):
                    errs.append(f"{r['run_id']}: {col}={v!r} outside [{BPB_LO}, {BPB_HI}]")
        cells = _hand_cells(micro)
        for key, c in cells.items():
            if c["n"] >= 2 and not (c["sigma"] is not None and math.isfinite(c["sigma"]) and c["sigma"] > 0):
                errs.append(f"{key}: multi-seed sigma not finite-positive ({c['sigma']})")
        tern = [c for (s, p, t), c in cells.items() if p == LOW]
        bf = [c for (s, p, t), c in cells.items() if p == HIGH]
        if tern and bf:
            tern_mean = sum(c["mean"] * c["n"] for c in tern) / sum(c["n"] for c in tern)
            bf_mean = sum(c["mean"] * c["n"] for c in bf) / sum(c["n"] for c in bf)
            if not bf_mean <= tern_mean + 0.5:
                errs.append(f"arm-swap suspicion: bf16 mean {bf_mean:.3f} > ternary {tern_mean:.3f} + 0.5")
            mono = f"bf16 mean {bf_mean:.3f} vs ternary {tern_mean:.3f}"
        else:
            mono = "monotone check skipped: an arm is absent"
        out.append(GateOutcome(
            "G4", self.gate_specs[3][1], passed=not errs,
            detail="; ".join(errs) if errs
            else f"{len(micro)} micro rows ({m['poll_note']}); {mono}",
        ))
        return out
