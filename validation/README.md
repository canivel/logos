# The validation panel

A **separate** verification layer for LOGOS benchmarks and results. Two design
commitments, both borrowed from KAOS's falsifiable-eval discipline:

1. **Independence.** Probes re-derive the quantity under test from first
   principles (paper definitions, hand-computable cases, synthetic ground
   truth, independent reimplementation) and compare against what `src/logos`
   produces. A probe that imports the function it is checking to compute its
   own expectation is worthless and gets rejected in review.
2. **Pre-registered kill gates.** Each probe declares its gates up front;
   the spec is sha256-locked in `validation/locks/` on first registration.
   Changing a gate after seeing results VOIDs the probe. Fix the stack or the
   claim — never the gate.

## Panel roster

| Probe | Verifies | Independent reference |
|-------|----------|----------------------|
| `quant_numerics` | Ternary absmean + 2/3/4-bit group quant produce exactly the claimed level sets, scales, and reconstruction error bounds | Hand-built numpy reimplementation from the BitNet b1.58 / ParetoQ equations |
| `ste_gradients` | STE backward = identity on clip region, zero outside | Finite differences on the smoothed surrogate + analytic expectation |
| `data_order` | Every precision arm at a given (size, D) consumes bit-identical token streams; resume does not fork the stream | Direct batch-stream hashing across arms and across a kill/resume |
| `bpb_definition` | BPB runner matches the definition (Σ NLL / ln2 / bytes) incl. partial windows | Closed-form uniform-logits case + independent per-token loop |
| `export_parity` | Packed artifact reloads to bit-identical effective weights and logits (the P0 gate) | Pack/unpack reimplemented from the format spec; logit diff on fresh prompts |
| `footprint` | Measured packed bytes match the format's exact bit math; ordering ternary<2<3<4<bf16; theoretical-vs-measured overhead is reported not assumed | Bit-level byte accounting from shapes alone |
| `fit_recovery` | The fitting pipeline recovers known ground-truth laws from synthetic noisy grids; LOSO selection picks the generating form; bootstrap CIs cover truth | Synthetic generator written inside the probe, params hidden from the fitter |
| `stats_discipline` | Seed-sigma math is a real std; every gap the results store marks "significant" exceeds 2σ for its size class; no claim rests on n=1 seeds where the plan requires 3 | Recomputed from raw results rows |
| `manifest_integrity` | Every result row's config hash matches a manifest RunSpec; run counts per phase match PLAN.md tables; budget ledger caps actually refuse | Plan tables re-encoded in the probe |
| `benchmark_pinning` | lm-eval version pin enforced; downstream task list matches the plan per phase; results carry harness+task versions | PLAN.md section lists |
| `training_sanity` | Smoke-run loss curves: monotone-ish decrease, no NaN, ternary-vs-bf16 ordering sane at tiny scale, resume curve == uninterrupted curve | Curve recomputation from metrics.jsonl |

## Running

```bash
logos-validate --list
logos-validate --all                 # full round; exit 1 on any REJECT
logos-validate --probe fit_recovery bpb_definition
```

Reports land in `validation/out/report.{json,md}`. The panel also runs as a
KAOS agent round (`kaos parallel` over probe groups) before any phase's
results are written up — see `ops/README.md`.

## Verdict semantics

- **ACCEPT** — every gate passed; the checked claims stand.
- **REJECT** — a kill gate failed. The associated benchmark/result is not
  citable until resolved. File the failure, fix the stack or retract the
  claim, re-run the panel.
- **VOID** — the probe could not run or its gates were altered post-lock.
  Treat as REJECT for release purposes.
