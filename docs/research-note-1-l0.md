# Research Note 1 — L0: the ternary-vs-bf16 crossover, measured with a noise floor

*LOGOS-Local phase L0 · August 2026 · status: **PRELIMINARY — 320× tier still
training**; numbers below regenerate via `python analysis/l0_summary.py`*

## Claim under test (RQ1)

That this stack — quantizers, frozen-protocol trainer, deterministic data
order, BPB evaluator — reproduces the known qualitative dynamic on real data
at 3–6M parameters: **natively-trained ternary models match or beat bf16 when
undertrained, and fall behind as tokens/param grows** (Ouyang et al. 2024;
BitNet b1.58) — and that the effect exceeds the measured seed-noise floor.

## Setup

FineWeb-Edu (sample-10BT, 2.5B pretokenized tokens, TinyLlama 32k
tokenizer), sizes 3m/6m from the TRIT ladder, arms {1.58-bit, bf16},
tokens/param {20×, 80×, 320×}, 3 seeds at 3m / 2 at 6m (320×: 1 seed),
seq 1024, batch 2^18 tokens, identical data order everywhere. Single
RTX 3080, queue-driven, ~127k tok/s at 3m (MFU ≈ 0.15). Ternary arms carry
the BitNet-prior 2× LR multiplier until the L1 probes measure the rule —
the standing LR confound, resolved next phase.

## Result

![gap vs tokens/param](figures/l0_gap_vs_dn.png)

![bpb small multiples](figures/l0_bpb_vs_dn.png)

| Cell | gap (ternary−bf16) | 2σ | verdict |
|------|--------------------:|----:|---------|
| 3m @20× | −0.0687 | 0.0409 | **significant** |
| 3m @80× | +0.0178 | 0.1008 | within noise |
| 6m @20× | −0.0372 | 0.1311 | within noise |
| 6m @80× | +0.1675 | 0.1313 | **significant** |

Three observations:

1. **The crossover is real and sits between 20× and 80× at these sizes.**
   Ternary is significantly *better* at 20× (3m, beyond 2σ) and
   significantly *worse* by 80× (6m, beyond 2σ). Chinchilla-optimal is
   ~20×; every deployed model lives far to the right of it — which is why
   the D/N axis, not just the size axis, must be in the law (RQ2/RQ3).
2. **The gap grows with size at fixed D/N** (80×: +0.018 at 3m → +0.168 at
   6m). If this size–tokens interaction survives the grid, it is exactly
   the term Form A (effective capacity) and Form B (additive degradation)
   disagree about — the fit will have something real to bite on.
3. **The noise floor is workable but binding.** Seed σ ranges 0.012–0.066
   BPB; several cells land inside their own 2σ. The 3-seed rule at the
   small sizes is not bureaucracy — half this table would be unclaimable
   without it.

## Pending before this note is final

320× tier (both sizes; expect the ternary deficit to widen — the PTQ-like
monotone-growth vs saturation question is RQ2's core), the σ table refresh,
and the L1 LR probes to close the confound noted above.

## Provenance

Manifest `manifests/l0.yaml` (run ids `l-*`) · results in
`results/results.jsonl` (git-timestamped, hash-checked against the manifest
by the validation panel) · figures regenerate from `analysis/l0_summary.py`.
