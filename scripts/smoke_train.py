#!/usr/bin/env python
"""End-to-end local smoke (PLAN.md s5 build tasks): synthetic shards -> tiny
model -> short train per precision arm {bf16, 1.58}. Proves the loop, the
loader, checkpointing and both quant paths execute before anything touches a
paid GPU. ~2-4 min for the default 300 steps on a 3080; use --steps to trim.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from logos.config import ModelConfig, Precision, RunSpec, TrainConfig  # noqa: E402
from logos.data.prepare import prepare_synthetic  # noqa: E402
from logos.train.trainer import TrainerExtras, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="runs dir (default: fresh temp dir)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8, help="sequences per step")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="logos_smoke_"))
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_tokens = args.batch_size * args.seq_len

    data_dir = out / "data"
    if not (data_dir / "index.json").exists():
        prepare_synthetic(
            data_dir, n_shards=2, shard_tokens=200_000, vocab_size=32_768, val_tokens=16_384
        )
    print(f"device={device} runs={out} steps={args.steps} batch_tokens={batch_tokens}")

    results = {}
    for prec in (Precision.BF16, Precision.W1_58):
        mcfg = ModelConfig(
            d_model=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=1,
            ffn_hidden=384,
            vocab_size=32_768,
            max_seq_len=args.seq_len,
            precision=prec,
        )
        tcfg = TrainConfig(
            lr=1e-3 * (2.0 if prec.is_quantized else 1.0),  # BitNet-style 2x for ternary
            total_tokens=args.steps * batch_tokens,
            batch_tokens=batch_tokens,
            seq_len=args.seq_len,
            warmup_tokens=max(batch_tokens, args.steps * batch_tokens // 20),
            dtype="bfloat16" if device.startswith("cuda") else "float32",
        )
        spec = RunSpec(
            run_id=f"smoke-{prec.value}",
            phase="p0",
            size="25m",
            precision=prec.value,
            tokens_per_param=1.0,
            tags=["smoke"],
        )
        print(f"\n== {prec.value}: nonemb={mcfg.n_nonemb / 1e6:.2f}M emb={mcfg.n_emb / 1e6:.2f}M ==")
        t0 = time.time()
        status = train(
            spec,
            data_dir=data_dir,
            run_dir=out / spec.run_id,
            device=device,
            log_interval=25,
            extras=TrainerExtras(model_config=mcfg, train_config=tcfg),
        )
        wall = time.time() - t0
        tps = status["tokens"] / wall
        results[prec.value] = {"final_loss": status["final_loss"], "tokens_per_s": tps}
        print(
            f"{prec.value}: final_loss={status['final_loss']:.4f} "
            f"tokens={status['tokens']} wall={wall:.1f}s tokens/s={tps:,.0f}"
        )

    print("\n" + json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
