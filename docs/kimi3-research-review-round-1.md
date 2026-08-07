# kimi3 — LOGOS research-direction review, round 1

*August 2026 · external panel review by an AI reviewer (kimi3) · scope:
research direction, scientific soundness, and priorities — complementary to
the machine validation panel (`validation-panel-round-1.md`), which gates
correctness of implementation. This review gates the direction.*

**Materials reviewed:** repo @ `c3d6a43` (full science-bearing code read:
quantizers, transformer, trainer, data loader, BPB eval, export/packing,
results store, fitting, manifests, validation probes), plans v0.2/v0.3,
Research Notes 0–1, `results/results.jsonl` (28 rows), git history, prepared
corpus verified on disk, 2025–26 literature scan.

**Panel:** scaling-laws methodologist · quantization systems lead ·
statistics reviewer · program chair (novelty/impact) · devil's advocate

**Overall verdict: 7.5/10 — accept with revisions. The project is on the
right track; keep going. Four specific, cheap fixes below should land before
they become load-bearing.**

---

## 1. What the panel agrees is genuinely right

- **The question is real white space.** A memory-budget scaling law for
  *native* low-bit training, with the overtraining (D/N) axis mapped and
  blind-extrapolation validation, does not exist in the published literature
  at any scale. The closest 2026 entrant found — *Not All Bits Are Equal:
  Scale-Dependent Memory Optimization Strategies for Reasoning Models* (ICLR
  2026, "4-bit is broadly memory-optimal") — is PTQ-side on existing models,
  not a QAT-from-scratch law. The differentiators (native, overtraining axis,
  KV-in-budget, released suite) still hold. Keep the arXiv triage hot; the
  space is narrowing.
- **The hygiene is publication-grade, not decorative.** Config hashes,
  manifest-driven runs, append-only results store, sha256-locked kill gates,
  probes that re-derive expectations independently (`stats_discipline`
  recomputes σ in plain Python from raw rows; `fit_recovery` uses hidden
  ground truth). Round 1 caught 6 real defects pre-publication.
- **The quantizers are faithful.** BitLinear absmean (`bitlinear.py:32-35`),
  LSQ per-group with the 1/√(numel·qmax) rescale (`paretoq.py:56-61`),
  per-token absmax WxA8, STE exactly identity in the clip region. The
  ulp-level forward/export parity fix (`paretoq.py:67-81`) shows real care.
- **The science is already working.** L0 shows the crossover the project
  bets on: ternary significantly *better* at 3m/20× (−0.069 BPB, beyond 2σ),
  parity at 3m/80×, significantly *worse* at 6m/80× (+0.168, beyond 2σ). The
  gap grows with size at fixed D/N — the exact term Form A and Form B
  disagree about. The bet is paying out.
- **The v0.3 rescope is honest and smart.** Fit at 3–60M, blind-validate at
  125M, cap cloud at $100, desktop reproducibility as part of the
  contribution.

## 2. Findings, ranked by threat to the claims

### F1 — The LR confound is the #1 scientific risk; L1 as designed only half-closes it

Ternary runs at 2× bf16 LR (BitNet prior). In the undertrained regime, higher
LR alone wins — so "ternary beats bf16 at 20×" (the significant L0 result) is
currently indistinguishable from "2× LR beats 1× LR at 20×". Research Note 1
flags this honestly. But the L1 plan (probe a per-precision multiplier, then
freeze) has a deeper hole: **the optimal multiplier plausibly depends on D/N
itself** — the right LR at 20× is not the right LR at 320×. A frozen
per-precision rule can manufacture or erase a crossover.

**Fix:** add direct control arms — bf16 at 2× LR at 3m and 6m, 20× and 80×
(8 runs, ~10 GPU-h on the 3080). If bf16@2×LR closes the 20× gap, the
headline changes; better to know now. Cheap, decisive, and reviewers will ask
for exactly this.

### F2 — Data coverage: 2.5B tokens on disk vs 10BT assumed by the plan; the largest anchor needs ~10.2B

Verified on disk: `data/fineweb_edu_10bt/index.json` holds **2,500,106,180**
train tokens. L0 is safe (6m@320× = 2.05B, 82% of corpus). But **12m@320×
(3.84B) — in L1, the next phase — already repeats data**, 25m@320× (8B)
trains ~3 epochs, and 125m@80× (~10.2B) exceeds even the planned 10BT.
Because data order is identical across arms, repetition hurts all arms
equally, so *comparative* claims survive — but the β exponent of the fitted
law will be quietly distorted (per Muennighoff et al., repetition is nearly
free to ~4 epochs — but "nearly" is doing work in a law fitted to three
significant figures).

**Fix:** extend data prep to ~11–12BT **before** L1's 320× runs start. One
overnight job, one-time cost; also repairs the README↔plan drift (README's
reproduce command says 2.5B; v0.3 says 10BT).

### F3 — Embedding bytes undermine the deployment headline at local scale

The 32k bf16 tied embedding is 33–67MB at the L5 shapes — for a 64MB-body
ternary artifact, the embedding alone can exceed the body budget. The
disclosure is honest ("84% of ternary-25M", body-byte budgets), and since all
arms pay the same embedding tax the *comparisons* are fair. But "7.2×
smaller" becomes "~2–3× smaller total" the moment a reader prices the
embedding — and deployment bytes are the entire motivation.

**Fix:** (a) report total-bytes next to body-bytes in every headline figure;
(b) add a fixed, disclosed embedding treatment for L5 artifacts (int8
embedding export is one flag and known-safe), or an ALBERT-style factorized
embedding as a protocol constant. Do this before L5, not in the rebuttal.

### F4 — The prescription ignores training compute; reviewers will not

b*(M, D) answers "best quality at M bytes given D" — but at a fixed byte
budget, the big-N-low-bits arm can cost 5–10× the training FLOPs of the
small-N-high-bits arm. The framing (train once, deploy many; DRAM is the
scarce resource) is defensible — but only if the iso-compute slice is shown.

**Fix:** zero extra runs — add an iso-training-FLOPs contour to the phase
diagram in `fitting/prescribe.py` (C ≈ 6ND is computable per cell). One
figure, kills the most predictable referee objection.

## 3. Smaller findings (fix opportunistically)

- **F5 — W×A8 is bundled into "precision".** Every low-bit arm quantizes
  activations to int8 *and* adds subln; bf16 has neither. At 4-bit weights,
  A8 may dominate the degradation. One W4A16 control arm at 12m/80× (~5
  GPU-h) decomposes weight-vs-activation effects and pre-empts a referee.
- **F6 — Fit objective is unweighted by cell noise.** Measured seed σ varies
  ~4× across cells (0.012–0.066 BPB). Rows enter the Huber objective
  unweighted, so noisy cells pull as hard as clean ones. Inverse-variance
  weighting (1/σ² per cell, σ from the RQ1 table) is a ~10-line change in
  `fitting/fit.py` and makes bootstrap CIs more honest.
- **F7 — Bottom-rung shapes are atypical.** 3m = d256/L4 is very wide-shallow
  vs anything deployed. Fine for law-fitting, but the paper needs one
  sentence of shape-robustness evidence or disclosure.
- **F8 — RQ4's downstream half is weak at this scale, by the project's own
  admission.** Benchmarks <60M are noise. Reframe RQ4 locally as "loss
  ordering + 125M anchors"; move benchmark-ordering claims to future work
  rather than reporting noise.
- **F9 — RULER/NIAH at 12M params will be garbage.** For L4, keep it to "BPB
  degradation under KV quant" and long-context perplexity; don't cite task
  scores.

## 4. Priority order going forward

1. **Finish L0's 320× tier** (running now) — completes the monotone-crossover
   story.
2. **F2 data extension** — blocking for L1's 320× runs; nearly free.
3. **F1 bf16@2×LR controls + L1 LR probes** — before any gap result leaves
   the repo. This is the difference between "ternary wins undertrained"
   being a result or an artifact.
4. **L1 gap study as planned** — the gap-vs-D/N figure is the first
   genuinely novel result; everything else is infrastructure.
5. **F3/F4 (total-bytes headline, iso-FLOP contours)** — analysis-only; do
   while L2 trains.
6. **Then L2 → L3 exactly as planned.** The blind-anchor protocol is right;
   do not touch it.

**De-prioritize:** KAOS ops expansion beyond the toil tiers (per the
project's own protection rule), P2-extended tiers, downstream evals below
60M, infra polish. None of it is on the critical path to the preprint.

## 5. Bottom line

The track is right and the early data supports the thesis. The real risks
are not conceptual; they are four specific, cheap, fixable holes (LR
confound, data shortfall, embedding accounting, compute fairness), each of
which a good referee *will* find. Fix them now, while they are 10-line to
one-overnight-job fixes, and RQ1–RQ4 is a solid preprint.
