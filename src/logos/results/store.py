"""Append-only results store: results/results.jsonl.

One JSON line per completed run, keyed by run_id and config_hash so the
validation panel can cross-check every result row against the versioned
manifests (PLAN.md s12). Analysis helpers implement design principle 4
(PLAN.md s3): any reported gap must exceed 2 sigma for its size class.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from logos.config import RunSpec
from logos.manifest.schema import ManifestError, load_manifest


def append_result(path: str | Path, run_spec: RunSpec, metrics: dict[str, Any]) -> dict[str, Any]:
    """Append one result row. metrics supplies bpb_val1, bpb_val2, downstream,
    packed_bytes, gpu_hours, cost_usd, wall_s, status; everything else derives
    from the RunSpec."""
    m = run_spec.model_config()
    tokens = run_spec.total_tokens or int(run_spec.tokens_per_param * m.n_nonemb)
    row = {
        "run_id": run_spec.run_id,
        "phase": run_spec.phase,
        "size": run_spec.size,
        "n_nonemb": m.n_nonemb,
        "n_total": m.n_total,
        "tokens": tokens,
        "tokens_per_param": run_spec.tokens_per_param,
        "precision": run_spec.precision,
        "seed": run_spec.seed,
        "ffn_type": run_spec.ffn_type,
        "lr": run_spec.train_config().lr,
        "config_hash": run_spec.config_hash(),
        "bpb_val1": metrics.get("bpb_val1"),
        "bpb_val2": metrics.get("bpb_val2"),
        "downstream": metrics.get("downstream", {}),
        "packed_bytes": metrics.get("packed_bytes"),
        "gpu_hours": metrics.get("gpu_hours"),
        "cost_usd": metrics.get("cost_usd"),
        "wall_s": metrics.get("wall_s"),
        "status": metrics.get("status", "complete"),
        "ts": metrics.get("ts", time.time()),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def load_results(path: str | Path) -> pd.DataFrame:
    """Load results.jsonl into a DataFrame (empty frame if missing)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def seed_sigma(df: pd.DataFrame, bpb_col: str = "bpb_val1") -> pd.DataFrame:
    """Per-(size, precision, tokens_per_param) std of BPB across seeds: the
    measured noise floor (RQ1). ddof=1; sigma is NaN for single-seed cells."""
    g = df.groupby(["size", "precision", "tokens_per_param"])[bpb_col]
    out = g.agg(mean_bpb="mean", sigma_bpb=lambda x: x.std(ddof=1), n_seeds="count")
    return out.reset_index()


def gap_vs_sigma(
    df: pd.DataFrame,
    low: str = "1.58",
    high: str = "bf16",
    bpb_col: str = "bpb_val1",
) -> pd.DataFrame:
    """Ternary-vs-bf16 gaps with 2-sigma verdicts per (size, tokens_per_param).

    gap = mean BPB(low) - mean BPB(high); sigma = the larger of the two arms'
    seed sigmas for that cell (conservative). exceeds_2sigma implements
    PLAN.md s3 principle 4."""
    sig = seed_sigma(df, bpb_col=bpb_col)
    lo = sig[sig["precision"] == low].set_index(["size", "tokens_per_param"])
    hi = sig[sig["precision"] == high].set_index(["size", "tokens_per_param"])
    cells = lo.index.intersection(hi.index)
    rows = []
    for size, tp in cells:
        gap = lo.loc[(size, tp), "mean_bpb"] - hi.loc[(size, tp), "mean_bpb"]
        sigma = np.nanmax(
            [lo.loc[(size, tp), "sigma_bpb"], hi.loc[(size, tp), "sigma_bpb"]]
        )
        exceeds = bool(abs(gap) > 2 * sigma) if np.isfinite(sigma) else False
        rows.append(
            {
                "size": size,
                "tokens_per_param": tp,
                "gap_bpb": gap,
                "sigma_bpb": sigma,
                "two_sigma": 2 * sigma,
                "exceeds_2sigma": exceeds,
                "verdict": "significant" if exceeds else "within_noise",
            }
        )
    return pd.DataFrame(rows)


def check_hash(df: pd.DataFrame, manifests_dir: str | Path) -> pd.DataFrame:
    """Verify every result row's config_hash against its manifest RunSpec.

    Returns a DataFrame of mismatches (empty == all rows verified). A run_id
    may appear in more than one manifest (P1 gap-study rows re-emitted in P2);
    a row passes if it matches any manifest copy."""
    hashes: dict[str, set[str]] = {}
    for p in sorted(Path(manifests_dir).glob("*.yaml")):
        try:
            _, specs = load_manifest(p)
        except ManifestError:
            continue  # e.g. lr_rules.yaml is not a run manifest
        for s in specs:
            hashes.setdefault(s.run_id, set()).add(s.config_hash())
    bad = []
    for _, row in df.iterrows():
        rid, h = row["run_id"], row["config_hash"]
        if rid not in hashes:
            bad.append({"run_id": rid, "config_hash": h, "reason": "run_id not in any manifest"})
        elif h not in hashes[rid]:
            bad.append({"run_id": rid, "config_hash": h, "reason": "config_hash mismatch"})
    return pd.DataFrame(bad, columns=["run_id", "config_hash", "reason"])
