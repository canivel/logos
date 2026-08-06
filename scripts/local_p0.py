"""Local micro-P0: the PLAN.md section-5 experiment at panel-verifiable scale.

Trains the full precision ladder {bf16, 4, 3, 2, 1.58} on REAL text (a
Project Gutenberg corpus, byte-level tokens so vocab=256 and BPB is exact by
construction) at the `micro` ladder size, 2 seeds on the headline pair
(ternary vs bf16), evaluates BPB on two held-out splits, exports + checks
logit parity, and appends rows to results/results.jsonl. Produces the
project's first figure: ternary-vs-bf16 loss curves.

This is NOT grid data (micro is excluded from law fitting); it exists so the
validation panel has real end-to-end artifacts to interrogate: loss curves,
seed sigma, gaps, BPB numbers, packed footprints, parity reports, and a
manifest whose hashes must match the results rows.

Usage: python scripts/local_p0.py [--corpus PATH] [--total-tokens N] [--device cuda]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from logos.config import DEFAULT_LR_MULT, BASE_LR, Precision, RunSpec, TrainConfig, make_model
from logos.data.prepare import _write_json  # canonical index writer
from logos.eval.bpb import bpb_from_val_dir
from logos.export.artifact import export_artifact, load_artifact
from logos.export.pack import measured_bytes
from logos.export.parity import export_parity, synthetic_prompts
from logos.manifest.schema import save_manifest
from logos.model.transformer import Transformer
from logos.results.store import append_result, gap_vs_sigma, load_results, seed_sigma
from logos.train.trainer import TrainerExtras, train

ROOT = Path(__file__).resolve().parents[1]
SEQ = 256
BATCH_TOKENS = 16_384  # 64 sequences/step
VOCAB = 256  # byte-level


def build_byte_shards(corpus: Path, out: Path) -> dict:
    """Bytes -> uint16 tokens; train/val1/val2 = 90/5/5 contiguous splits.
    Byte-level means utf8_bytes == tokens exactly (BPB sanity anchor)."""
    raw = corpus.read_bytes()
    toks = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    n = len(toks)
    cut1, cut2 = int(n * 0.90), int(n * 0.95)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": toks[:cut1], "val1": toks[cut1:cut2], "val2": toks[cut2:]}
    splits["train"].tofile(out / "shard_00000.bin")
    splits["val1"].tofile(out / "val1.bin")
    splits["val2"].tofile(out / "val2.bin")
    index = {
        "dataset": f"gutenberg-bytes:{corpus.name}",
        "dataset_config": None,
        "tokenizer": "byte-level",
        "vocab_size": VOCAB,
        "eos_id": 0,
        "seed": 1337,
        "shards": [
            {"file": "shard_00000.bin", "tokens": int(cut1), "utf8_bytes": int(cut1), "docs": 1}
        ],
        "total_tokens": int(cut1),
        "val": {
            "val1": {"file": "val1.bin", "tokens": int(cut2 - cut1), "utf8_bytes": int(cut2 - cut1), "docs": 1},
            "val2": {"file": "val2.bin", "tokens": int(n - cut2), "utf8_bytes": int(n - cut2), "docs": 1},
        },
    }
    _write_json(out / "index.json", index)
    return index


def arm_specs(total_tokens: int) -> list[RunSpec]:
    n_nonemb = make_model("micro").n_nonemb
    tp = round(total_tokens / n_nonemb, 4)
    specs = []
    for prec, seeds in [
        (Precision.W1_58, (0, 1)),
        (Precision.BF16, (0, 1)),
        (Precision.W2, (0,)),
        (Precision.W3, (0,)),
        (Precision.W4, (0,)),
    ]:
        for seed in seeds:
            specs.append(
                RunSpec(
                    run_id=f"local-micro-{prec.value}-s{seed}",
                    phase="p0",
                    size="micro",
                    precision=prec.value,
                    tokens_per_param=tp,
                    seed=seed,
                    lr=BASE_LR["micro"] * DEFAULT_LR_MULT[prec],
                    total_tokens=total_tokens,
                    seq_len=SEQ,
                    batch_tokens=BATCH_TOKENS,
                    tags=["local-smoke", "byte-level"],
                    notes="micro-P0 on real text; excluded from law fitting",
                )
            )
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        default=str(ROOT.parent / "The Project Gutenberg eBook of Frankenstein.txt"),
    )
    ap.add_argument("--total-tokens", type=int, default=2**22)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "local_p0"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "local_p0"))
    args = ap.parse_args()

    data_dir, runs_dir = Path(args.data_dir), Path(args.runs_dir)
    index = build_byte_shards(Path(args.corpus), data_dir)
    print(f"corpus: {index['dataset']}  train={index['total_tokens']:,} tokens")

    specs = arm_specs(args.total_tokens)
    save_manifest(
        specs,
        ROOT / "manifests" / "p0local.yaml",
        meta={"purpose": "local micro-P0 (panel artifacts, not grid data)", "corpus": index["dataset"]},
    )

    results_path = ROOT / "results" / "results.jsonl"
    done_ids = set()
    if results_path.exists():
        done_ids = {
            json.loads(x)["run_id"]
            for x in results_path.read_text().splitlines()
            if x.strip()
        }
    for spec in specs:
        run_dir = runs_dir / spec.run_id
        if spec.run_id in done_ids:
            print(f"skip {spec.run_id} (results row exists)")
            continue
        # train() auto-resumes; on an already-complete run it no-ops fast.
        mcfg = make_model("micro", Precision(spec.precision), vocab_size=VOCAB, max_seq_len=SEQ)
        tcfg = TrainConfig(
            lr=spec.lr,
            total_tokens=spec.total_tokens,
            batch_tokens=BATCH_TOKENS,
            seq_len=SEQ,
            seed=spec.seed,
            checkpoint_interval_s=600,
            dtype="bfloat16",
        )
        t0 = time.time()
        status = train(
            spec,
            data_dir=data_dir,
            run_dir=run_dir,
            device=args.device,
            log_interval=10,
            extras=TrainerExtras(model_config=mcfg, train_config=tcfg),
        )
        wall = time.time() - t0

        # Reload trained weights for eval/export.
        torch.manual_seed(spec.seed)
        model = Transformer(mcfg)
        state = torch.load(run_dir / "ckpt_latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        e1 = bpb_from_val_dir(model, data_dir, "val1", seq_len=SEQ, device=args.device)
        e2 = bpb_from_val_dir(model, data_dir, "val2", seq_len=SEQ, device=args.device)
        packed = measured_bytes(model)
        art_dir = run_dir / "artifact"
        export_artifact(model, art_dir)
        reloaded = load_artifact(art_dir)
        prompts = synthetic_prompts(vocab_size=VOCAB, n_prompts=8, seq_len=64, seed=7)
        par = export_parity(model, reloaded, prompts, device="cpu")
        (run_dir / "parity.json").write_text(json.dumps(par, indent=2, default=float))

        append_result(
            results_path,
            spec,
            dict(
                bpb_val1=e1["bpb"],
                bpb_val2=e2["bpb"],
                packed_bytes=packed["total_bytes"],
                gpu_hours=wall / 3600,
                cost_usd=0.0,
                wall_s=wall,
                status=status["status"],
                downstream={},
                extra={"final_loss": status["final_loss"], "parity_pass": bool(par["pass"])},
            ),
        )
        print(
            f"{spec.run_id}: loss={status['final_loss']:.4f} bpb1={e1['bpb']:.4f} "
            f"bpb2={e2['bpb']:.4f} packed={packed['total_bytes']:,}B "
            f"parity={'PASS' if par['pass'] else 'FAIL'} ({wall:.0f}s)"
        )

    # ---- summary: sigma, gaps, first figure ----
    df = load_results(results_path)
    df = df[df["size"] == "micro"]
    print("\nseed sigma:\n", seed_sigma(df).to_string(index=False))
    print("\nternary-vs-bf16 gap:\n", gap_vs_sigma(df).to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"bf16": "#404663", "1.58": "#B13214", "2": "#C87E29", "3": "#5B7C99", "4": "#3D7A5E"}
    for spec in specs:
        mf = runs_dir / spec.run_id / "metrics.jsonl"
        if not mf.exists():
            continue
        rows = [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
        ax.plot(
            [r["tokens"] / 1e6 for r in rows],
            [r["loss"] / np.log(2) for r in rows],
            color=colors[spec.precision],
            alpha=0.55 if spec.seed else 1.0,
            lw=1.2,
            label=f"{spec.precision}" if spec.seed == 0 else None,
        )
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel("train loss (bits/byte)")
    ax.set_title("micro-P0: precision ladder on real text (byte-level)")
    ax.legend(title="weight bits", fontsize=8)
    out_fig = ROOT / "analysis" / "out" / "local_p0_loss_curves.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    print(f"\nfigure: {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
