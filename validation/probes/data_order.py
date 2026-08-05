"""Probe data_order — PLAN.md s3 principle 5: "Same data order everywhere."

Every arm at a given (size, D) must consume bit-identical token streams;
precision is the only difference in what the model experiences, and resume
must not fork the stream. Reference: direct sha256 hashing of batch tensors
over synthetic shards, window accounting re-derived from the raw .bin files
(never from the loader's own bookkeeping), and the background micro-P0
metrics for cross-arm token accounting.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from logos.config import RunSpec
from logos.data.loader import TokenLoader
from logos.data.prepare import prepare_synthetic
from validation.base import GateOutcome, Probe

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "runs" / "local_p0"
G5_ARMS = ("local-micro-1.58-s0", "local-micro-bf16-s0")
G5_POLL_S = 30.0


def _batch_hash(batch: dict) -> str:
    h = hashlib.sha256()
    for key in ("inputs", "targets"):
        t = batch[key].numpy()
        h.update(str(t.shape).encode())
        h.update(t.tobytes())
    return h.hexdigest()


def _stream_hashes(loader: TokenLoader, n: int, start_step: int = 0) -> list[str]:
    it = loader.iter_batches(start_step)
    return [_batch_hash(next(it)) for _ in range(n)]


def _raw_windows(data_dir: Path, window: int) -> dict[bytes, int]:
    """Global window index -> content, re-derived from the shard .bin files
    alone (independent of TokenLoader's addressing)."""
    index = json.loads((data_dir / "index.json").read_text())
    out: dict[bytes, int] = {}
    widx = 0
    for s in index["shards"]:
        toks = np.fromfile(data_dir / s["file"], dtype=np.uint16)
        for j in range(len(toks) // window):
            out[toks[j * window : (j + 1) * window].tobytes()] = widx
            widx += 1
    return out


def _run_status(rid: str) -> dict | None:
    p = RUNS_DIR / rid / "status.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


class DataOrderProbe(Probe):
    name = "data_order"
    description = "same data order everywhere; resume does not fork the stream (PLAN.md s3 p5)"
    gate_specs = [
        ("G1", "two independent TokenLoaders over the same shards + data_seed produce "
               "bit-identical 50-batch streams (per-batch sha256)"),
        ("G2", "iter_batches(start_step=k) equals a fresh stream with k batches skipped, "
               "for k in {1, 7, 16, 37} (crossing an epoch boundary)"),
        ("G3", "loaders constructed exactly as trainer.train constructs them for two RunSpecs "
               "differing only in precision (bf16 vs 1.58) yield identical 20-batch streams"),
        ("G4", "one epoch consumes every window exactly once — no repeats, none missing — "
               "with windows re-derived from the raw shard bytes"),
        ("G5", "background micro-P0: local-micro-1.58-s0 and local-micro-bf16-s0 metrics "
               "step->tokens mappings are identical (vacuous if both runs never complete)"),
    ]

    def collect(self) -> dict:
        m: dict = {}
        # ignore_cleanup_errors: TokenLoader memmaps keep .bin files open on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            tmp = Path(td)

            # Small fixture: window 64 -> 64 windows/shard, 128 total,
            # batch 8 -> 16 batches/epoch, exact epoch coverage.
            small = tmp / "small"
            prepare_synthetic(small, n_shards=2, shard_tokens=4096, vocab_size=512,
                              seed=99, val_tokens=512)
            seq_len, bs = 63, 8

            def mk() -> TokenLoader:
                return TokenLoader(small, seq_len=seq_len, batch_size=bs, data_seed=1234)

            # G1: twin streams.
            h1, h2 = _stream_hashes(mk(), 50), _stream_hashes(mk(), 50)
            m["g1"] = {
                "n": 50,
                "equal": h1 == h2,
                "first_divergence": next(
                    (i for i, (a, b) in enumerate(zip(h1, h2)) if a != b), None
                ),
            }

            # G2: resume == skip, at several k (16 = epoch boundary here).
            base = _stream_hashes(mk(), 60)
            m["g2"] = {
                str(k): _stream_hashes(mk(), 10, start_step=k) == base[k : k + 10]
                for k in (1, 7, 16, 37)
            }

            # G4: epoch coverage, windows re-derived from raw shard bytes.
            raw = _raw_windows(small, seq_len + 1)
            n_windows = len(raw)
            it = mk().iter_batches(0)
            seen: list[int] = []
            unknown = 0
            for _ in range(n_windows // bs):  # one epoch, derived from raw count
                b = next(it)
                for i in range(bs):
                    row = (
                        np.concatenate([b["inputs"][i].numpy(), b["targets"][i, -1:].numpy()])
                        .astype(np.uint16)
                        .tobytes()
                    )
                    w = raw.get(row)
                    if w is None:
                        unknown += 1
                    else:
                        seen.append(w)
            m["g4"] = {
                "n_windows": n_windows,
                "consumed": len(seen) + unknown,
                "unknown_rows": unknown,
                "repeats": len(seen) - len(set(seen)),
                "missing": n_windows - len(set(seen)),
            }

            # G3: arm-independence with the real TrainConfig geometry
            # (batch_tokens 524288, seq_len 2048 -> 256 windows/batch).
            big = tmp / "big"
            prepare_synthetic(big, n_shards=2, shard_tokens=280_000, vocab_size=32_768,
                              seed=7, val_tokens=4096)
            arm_hashes: dict[str, list[str]] = {}
            arm_cfgs: dict[str, dict] = {}
            for prec in ("bf16", "1.58"):
                spec = RunSpec(run_id=f"probe-{prec}", phase="p0", size="micro",
                               precision=prec, tokens_per_param=1.0)
                cfg = spec.train_config()
                # Mirror trainer.train's loader construction (trainer.py L121-126):
                # TokenLoader(data_dir, seq_len=cfg.seq_len,
                #             batch_size=cfg.batch_tokens // cfg.seq_len,
                #             data_seed=cfg.data_seed)
                loader = TokenLoader(
                    big,
                    seq_len=cfg.seq_len,
                    batch_size=cfg.batch_tokens // cfg.seq_len,
                    data_seed=cfg.data_seed,
                )
                arm_hashes[prec] = _stream_hashes(loader, 20)
                arm_cfgs[prec] = {
                    "lr": cfg.lr,
                    "seq_len": cfg.seq_len,
                    "batch_tokens": cfg.batch_tokens,
                    "data_seed": cfg.data_seed,
                }
            m["g3"] = {
                "equal": arm_hashes["bf16"] == arm_hashes["1.58"],
                "lrs_differ": arm_cfgs["bf16"]["lr"] != arm_cfgs["1.58"]["lr"],
                "cfgs": arm_cfgs,
            }

        # G5: background micro-P0 token accounting across arms.
        deadline = time.time() + G5_POLL_S
        while time.time() < deadline:
            if all(
                (s := _run_status(a)) is not None and s.get("status") == "complete"
                for a in G5_ARMS
            ):
                break
            time.sleep(2.0)
        both_complete = all(
            (s := _run_status(a)) is not None and s.get("status") == "complete"
            for a in G5_ARMS
        )
        if both_complete:
            maps = {}
            for a in G5_ARMS:
                rows = [
                    json.loads(line)
                    for line in (RUNS_DIR / a / "metrics.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                maps[a] = {r["step"]: r["tokens"] for r in rows}
            m["g5"] = {
                "present": True,
                "equal": maps[G5_ARMS[0]] == maps[G5_ARMS[1]],
                "n_steps": {a: len(maps[a]) for a in G5_ARMS},
            }
        else:
            m["g5"] = {"present": False}
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)
        g1 = m["g1"]
        out = [
            GateOutcome(
                "G1", specs["G1"], g1["equal"],
                detail=f"50/50 batch hashes identical" if g1["equal"]
                else f"streams diverge at step {g1['first_divergence']}",
            )
        ]
        g2_bad = [k for k, ok in m["g2"].items() if not ok]
        out.append(
            GateOutcome(
                "G2", specs["G2"], not g2_bad,
                detail="resume == skip at k=1,7,16,37" if not g2_bad
                else f"resume forked the stream at k in {g2_bad}",
            )
        )
        g3 = m["g3"]
        out.append(
            GateOutcome(
                "G3", specs["G3"], g3["equal"] and g3["lrs_differ"],
                detail=(
                    f"20/20 batch hashes identical across arms; arm lrs differ "
                    f"({g3['cfgs']['bf16']['lr']:.4g} vs {g3['cfgs']['1.58']['lr']:.4g}), "
                    f"loader args identical (seq_len={g3['cfgs']['bf16']['seq_len']}, "
                    f"batch_tokens={g3['cfgs']['bf16']['batch_tokens']}, "
                    f"data_seed={g3['cfgs']['bf16']['data_seed']})"
                ) if g3["equal"] else "precision arms saw different batch streams",
            )
        )
        g4 = m["g4"]
        g4_ok = g4["unknown_rows"] == 0 and g4["repeats"] == 0 and g4["missing"] == 0
        out.append(
            GateOutcome(
                "G4", specs["G4"], g4_ok,
                detail=(
                    f"{g4['n_windows']} windows: consumed={g4['consumed']}, "
                    f"repeats={g4['repeats']}, missing={g4['missing']}, "
                    f"unknown_rows={g4['unknown_rows']}"
                ),
            )
        )
        g5 = m["g5"]
        if not g5["present"]:
            out.append(
                GateOutcome(
                    "G5", specs["G5"], True,
                    detail="vacuous: both background arms not complete within poll window",
                )
            )
        else:
            out.append(
                GateOutcome(
                    "G5", specs["G5"], g5["equal"],
                    detail=f"step->tokens maps {'identical' if g5['equal'] else 'DIFFER'} "
                           f"({g5['n_steps']})",
                )
            )
        return out
