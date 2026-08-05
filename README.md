# LOGOS

**Memory-optimal scaling laws for natively low-bit LLMs.** Given an inference
memory budget in bytes, how do you allocate parameters (N), weight precision
(b_w), KV precision (b_kv), and training tokens (D) to maximize quality?
LOGOS fits that law empirically for QAT-from-scratch (BitNet-style) models,
validates it by extrapolation, and cashes it in with a capstone model at a
real device budget. Model suite: **TRIT** (one trit = 1.58 bits).

Full research plan: [PLAN.md](PLAN.md). Sibling project:
[KAOS](https://github.com/canivel/kaos) runs the ops (Section 13 of the plan);
the science stays human.

## Layout

| Path | What |
|------|------|
| `src/logos/config.py` | The contract: `Precision`, `ModelConfig`, `TrainConfig`, `RunSpec`, the shape ladder |
| `src/logos/quant/` | The precision ladder: BitLinear (ternary absmean), ParetoQ-style 2/3/4-bit per-group int, WxA8 activations, STE |
| `src/logos/model/` | Llama-style decoder (RMSNorm, RoPE, GQA 4:1, tied 32k embeddings, subln on quantized arms, optional KV-QAT) |
| `src/logos/data/` | FineWeb-Edu → 32k-tokenized uint16 shards; deterministic fixed-order loader (same data order in every arm) |
| `src/logos/train/` | Trainer: cosine schedule, per-precision LR rule, 30-min + SIGTERM checkpointing, exact resume, divergence kill |
| `src/logos/eval/` | Bits-per-byte runner (the fitting target) + pinned lm-eval-harness wrapper |
| `src/logos/export/` | Packed formats (i2_s-style ternary, bit-packed 2/3/4), `.lpack` artifacts, GGUF hook, logit-parity gate |
| `src/logos/fitting/` | Forms A/B/C, Huber-on-log multi-start L-BFGS, LOSO selection, bootstrap CIs, b*(M,D) phase diagram, iso-memory frontier |
| `src/logos/manifest/` | Versioned run manifests for P0–P5, queue launcher, hard-capped budget ledger |
| `src/logos/results/` | Append-only results store, seed-sigma / 2-sigma gap discipline, manifest↔result hash checks |
| `manifests/` | Generated phase manifests (`python -m logos.manifest.generate`) |
| `ops/` | KAOS agent tier definitions (Monitor / Eval / Launcher), audit journal, cron fallbacks, arXiv scoop watch |
| `validation/` | **The validation panel**: falsifiable probes with pre-registered kill gates that verify benchmarks and results |
| `analysis/` | Fitting demo / notebook skeletons |
| `tests/` | Unit tests for everything above |

## Quickstart

```bash
uv venv --system-site-packages && uv pip install -e ".[dev]" --no-deps
python -m pytest -q                      # full unit suite
python scripts/smoke_train.py --out runs/smoke   # tiny end-to-end train, all arms
python -m logos.manifest.generate --out manifests/
logos-validate --all                     # run the validation panel
```

Real runs are defined only by manifests (`logos train --manifest manifests/p0.yaml --run-id ...`);
nothing is launched by hand twice.

## The rules that keep the law clean

1. One variable moves at a time — precision is the only difference between arms.
2. N means non-embedding parameters.
3. Loss (BPB) is the fitting target; benchmarks are the validator.
4. Any reported gap must exceed 2σ for its size class.
5. Same data order everywhere (`data_seed=1337`, project-wide).
6. Comparability beats per-arm maximum performance.
7. Fit forms compete; selection by held-out extrapolation, never in-sample fit.

See PLAN.md §3. The validation panel enforces these mechanically.
