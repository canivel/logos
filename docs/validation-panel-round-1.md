# Validation panel — round 1 report

*August 5, 2026 · overall verdict: **ACCEPT** (11/11 probes, 49/49 gates) ·
machine-readable: `validation/out/report.json`*

## What the panel is

A verification layer deliberately separate from the code it checks, borrowing
KAOS's falsifiable-eval discipline: each probe pre-registers its kill gates,
the gate spec is sha256-locked before results are examined, and a probe whose
gates change after locking is VOID. Probes must re-derive their expectations
independently — from paper equations, closed forms, hand-derived gradients,
format specs, or hidden synthetic ground truth — never from the function
under test. Three independent builder agents wrote the probes against the
locked contract; none had authored the subsystems they verify.

## Round-1 scorecard

| Probe | Verifies | Verdict |
|-------|----------|---------|
| quant_numerics | Ternary absmean + 2/3/4-bit group quant vs independent numpy re-derivation (≤4e-9); level sets; monotone quant error | ACCEPT 5/5 |
| ste_gradients | STE = identity on clip region; BitLinear/LSQ gradients vs hand-derived chain rules (≤7e-8 rel) | ACCEPT 4/4 |
| export_parity | Independent bit-stream decoder reproduces effective weights from raw artifacts; round-trip bitwise codes/scales; KL ≤ 1e-4 | ACCEPT 4/4 |
| footprint | Measured packed bytes == shape-derived arithmetic, all 5 arms; KV formula within 0.7% of plan's 500MB figure | ACCEPT 4/4 |
| bpb_definition | BPB vs closed form (6e-13) and vs an independent per-token loop; token conservation incl. partial windows | ACCEPT 4/4 |
| fit_recovery | Hidden-ground-truth law recovery (α/β ≤13%, f(b) ≤3%); LOSO picks the generating form; fitted vs true prescriptions agree on 100% of the (M,D) grid | ACCEPT 5/5 |
| stats_discipline | σ recomputed by hand; 2σ flags consistent; single-seed cells can never be "significant" | ACCEPT 4/4 |
| data_order | Bit-identical batch streams across arms and across kill/resume; true permutation coverage | ACCEPT 5/5 |
| manifest_integrity | Plan-table run counts re-encoded from the text; results↔manifest hash cross-check (fires on a tampered fixture); ledger caps are AST-verified literals | ACCEPT 5/5 |
| benchmark_pinning | lm-eval == 0.4.9 enforced; task suites match the plan; val splits verified disjoint from the binary data | ACCEPT 4/4 |
| training_sanity | 7/7 micro-P0 runs: finite gapless metrics, exact token accounting, bit-exact kill-and-resume, ordering audit | ACCEPT 5/5 |

## Defects the round caught (all fixed; no gate was weakened)

1. **CUDA resume crash** — `torch.load(map_location="cuda")` moved RNG
   ByteTensors to GPU; `set_rng_state` requires CPU. The CPU-only unit test
   could not see it; the first real GPU resume did.
2. **Status clobbering** — re-invoking a completed run rewrote `status.json`
   with `final_loss: NaN`. Completed runs are now immutable no-ops.
3. **BPB index schema mismatch** — the eval reader did not accept the schema
   the data pipeline actually writes; would have raised on first real eval.
4. **`phase` contaminated the science hash** — the same physical run re-used
   between phases (the documented P1→P2 reuse) hashed to two identities,
   breaking results↔manifest cross-checking. Caught by `manifest_integrity`'s
   REJECT; `phase` removed from the hashed fields.
5. **Parity-gate semantics** — max-abs logit difference is not a valid
   export-correctness criterion for deep quantized models: ulp-level scale
   recomputation amplifies through discontinuous activation quant (~1e-6 →
   ~1e-1 over 6 layers) while KL stays ~3e-4. Gate redefined as bitwise
   codes/scales + KL bound. Corroborated independently by the numerics agent
   on fresh shallow models.
6. **Forward/export ulp divergence (latent)** — the int-quant export divided
   by a scale that could differ from the forward path's grad-rescaled fp32
   value by 1 ulp; a boundary-sitting master could export a different code
   than training used. Export now replicates the forward arithmetic exactly.

## Why this matters for the research

Defects 1–3 would have burned GPU-hours or produced silently-wrong evals;
defect 4 would have quietly broken provenance for 15 grid runs; defects 5–6
would have produced spurious parity failures (or masked real ones) at the P0
exit gate. All were caught by construction — locked gates plus independent
re-derivation — before any citable result existed. The panel re-runs
(`logos-validate --all`) before every phase write-up; its verdict gates
publication of results from this repo.
