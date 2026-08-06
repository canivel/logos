# LOGOS methodology reference

Condensed from research plans v0.2 §3–§8 and v0.3; this is the day-to-day
reference for how results in this repo are produced and what they may claim.

## 1. Experimental discipline

| Rule | Mechanical enforcement |
|------|------------------------|
| One variable moves at a time: precision is the only difference between arms at a given (N, D) | `build_linear()` is the single dispatch point; arm-isolation unit tests assert the bf16 model contains zero quant modules and identical master init across arms |
| N = non-embedding parameters | `ModelConfig.n_nonemb` is the fitting covariate; both counts reported |
| Same data order everywhere | `data_seed=1337` frozen in `TrainConfig`; the panel's `data_order` probe hashes batch streams across arms and across kill/resume |
| Loss (BPB) fits, benchmarks validate | Two disjoint held-out sets; `bpb_definition` probe checks the metric against a closed form |
| No claim below 2σ | `results.store.gap_vs_sigma` computes verdicts; `stats_discipline` probe recomputes by hand and rejects significance flags on single-seed cells |
| Fit forms compete, selected out-of-sample | LOSO in `fitting.select`; `fit_recovery` probe verifies on hidden ground truth |
| Every run pre-specified | Versioned manifests with science-field config hashes; `manifest_integrity` probe cross-checks results ↔ manifests and re-encodes the plan's run counts |

## 2. The precision ladder

- **1.58-bit (ternary)** — BitLinear per BitNet b1.58: absmean scale
  γ = mean|W|, W̃ = RoundClip(W/γ, −1, 1)·γ, STE gradients, master weights
  in bf16/fp32, quantization computed in fp32. The architecture adds subln
  (RMSNorm before o_proj and down_proj inputs) on quantized arms.
- **2/3/4-bit** — per-group (128) symmetric integer quant with learnable
  per-group scales, LSQ-style gradient (1/√(numel·qmax) rescale), after the
  ParetoQ ladder. Same code path for all three widths.
- **Activations** — per-token absmax int8 (WxA8) on every quantized arm.
- **bf16 control** — plain `nn.Linear`; no quant modules anywhere.

Export packs ternary as i2_s-style 2-bit codes + per-tensor γ, and k-bit as
exact-width bitstreams + fp32 group scales. Footprints are **measured from
real packed bytes**, never derived from b/8 alone.

## 3. Model family (TRIT)

Llama-style pre-norm decoders: RMSNorm, RoPE, GQA 4:1, tied 32k embeddings,
SwiGLU (P1/L1 ablates squared-ReLU once, then frozen), head_dim 64–128,
FFN sized so non-emb params ≈ 12·L·d². Ladder: 3M, 6M, 12M, 25M, 60M
(local grid) / 125M, 250M (anchors) / 490M–1.5B (v0.2 reference design).

## 4. Training protocol

Single-stage cosine LR (warmup = min(1%, 250M tokens), floor 0.1×), AdamW
(0.9, 0.95), wd 0.1 on 2-D weights only (norms, embeddings, quant scales
exempt), grad clip 1.0, bf16 autocast, batch 2^18 tokens @ seq 1024
(local program; v0.2 used 2^19 @ 2048 — recorded per-run in manifests).
Per-precision LR = BASE_LR(size) × multiplier; multipliers start at BitNet
priors and are **replaced by the L1 probe measurements, then frozen**.
Gradient accumulation is the only hardware adaptation permitted — it never
changes `batch_tokens`. Checkpoints: atomic, every 30 min + on SIGTERM,
resume verified bit-exact. Divergence kill: NaN/inf immediately, or
best+2.0 nats sustained 100 steps.

## 5. Fitting

Candidate forms (L = validation BPB):
- **A (effective capacity):** L = E + A/(N·f(b))^α + B/D^β, f free per width
  (f(16)=1) or parametric 1−exp(−(b−b₀)/γ_f).
- **B (additive degradation):** L = E + A/N^α + B/D^β + C·g(b)·D^δ/N^η.
- **C:** both, L1-penalized.

Huber(δ=1e-3) on log-space residuals, 256-start L-BFGS-B, deterministic.
Selection: leave-one-size-out (and the upward-only variant for RQ4), never
in-sample. Uncertainty: stratified bootstrap over runs → percentile CIs on
every exponent, f(b), and derived crossover. Prescriptions minimize fitted L
subject to measured-bytes ≤ M, producing the iso-memory frontier and the
b*(M, D) phase diagram.

## 6. Blind extrapolation (the credibility test)

Before any anchor run launches: fit is frozen, predictions (BPB per anchor
arm, with the σ-implied band) are committed to the repo. Acceptance: anchors
within 2× the seed-σ band AND precision ordering correct. A miss is reported
as a bend in the law, not hidden.

## 7. Verification layer

Every load-bearing quantity has a falsifiable probe with pre-registered,
sha256-locked kill gates (`validation/`): independent numpy quantizer
re-derivations, hand-derived STE gradients, an independent bit-unpacker,
closed-form BPB anchors, hidden-ground-truth fit recovery, by-hand σ/2σ
recomputation, plan-table cross-checks, and training-artifact audits.
Changing a gate after registration voids the probe. `logos-validate --all`
must ACCEPT before any result is cited.

## 8. Ops

Manifest-driven queue launcher; budget ledger with caps as code constants
(L3 anchors: $100 hard cap); KAOS agent tiers (monitor → eval → launcher)
with allowlists and an append-only audit journal; cron fallbacks so the
science never blocks on the agent layer. Agents run the toil; humans run
the experiment — no agent edits configs, manifests, or fitting decisions.

## 9. Known limitations (standing disclosures)

1. Tied 32k bf16 embeddings dominate small artifacts (84% of ternary-25M);
   laws fit on non-emb N; L5 budgets are body-byte budgets; both reported.
2. Local grid tops out at 60M; extrapolation validated one size step up
   (125M; 250M stretch). Claims beyond that are labeled prediction.
3. Downstream benchmarks below ~60M are noise; they gate nothing.
4. Byte-level micro-P0 is a pipeline replication, not grid evidence.
5. Fake-quant (fp32 dequant matmuls) measures quality, not speed; kernel
   throughput claims come only from bitnet.cpp/llama.cpp exports.
