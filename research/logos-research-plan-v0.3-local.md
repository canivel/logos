# LOGOS-Local: Memory-Optimal Scaling Laws for Natively Low-Bit LLMs, on a Desk
### Research plan v0.3 (local edition), August 2026. Supersedes the compute plan of v0.2; the science, design principles, and stack are unchanged.

---

## 0. What changed from v0.2

v0.2 assumed ~$25k of RunPod credits. Reality: this is a solo, self-funded
project with **two local GPUs and ~$100 of cloud budget**:

| Resource | Available | Role |
|----------|-----------|------|
| RTX 3080 10GB (now) | unlimited hours, free | Engineering, L0 replication tier |
| RTX 5090 32GB (incoming) | unlimited hours, free | The main grid (L1-L2), capstone-lite (L5) |
| RunPod, 1x H100 | **~$100 total** | Blind extrapolation anchors ONLY (L3) |

The scientific bet gets *sharper*, not weaker: scaling-law papers have always
fitted small and extrapolated up. v0.2 fitted at 25M-500M and predicted 1B.
LOGOS-Local fits at **3M-60M** and predicts **125M and 250M** — the same
one-to-two-size-step extrapolation test, shifted down the ladder. Every
design principle from v0.2 §3 still binds (one variable at a time, non-emb N,
loss as target, 2σ discipline, fixed data order, frozen protocol, competing
fit forms). If the law's functional form is real, it must show up at this
scale; if it only appears above 100M, that is itself a finding the anchors
will expose.

**What we give up:** the 1B+ regime, the trained capstone at a phone-class
budget, and RQ5's full KV-QAT sweep (kept in reduced form). **What we keep:**
every research question, answered at a scale someone with a desktop can
reproduce end-to-end — which becomes part of the contribution: the entire
grid re-runs for under two GPU-months on one consumer card.

---

## 1. Thesis (unchanged)

Given an inference memory budget in bytes, how should you allocate parameter
count N, weight precision b_w, KV precision b_kv, and training tokens D to
maximize quality? We fit L(N, D, b_w) for natively low-bit (QAT-from-scratch,
BitNet-style) models, locate the crossovers where more-params-fewer-bits
beats fewer-params-more-bits, validate by blind extrapolation, and prescribe.

## 2. Research questions → local phases

| RQ | v0.2 phase | Local phase | Status of the answer |
|----|-----------|-------------|----------------------|
| RQ1 replication + noise floor | P0 | **L0** (3080, now) | Full answer at 3-6M; micro-P0 already reproduced the qualitative ternary-vs-bf16 undertrained dynamic |
| RQ2 gap vs tokens/param | P1 | **L1** (3080 → 5090) | Full answer at 12M, 20x-320x |
| RQ3 the fitted law + crossovers | P2 | **L2** (mostly 5090) | Full answer over 3M-60M x 5 precisions |
| RQ4 extrapolation + downstream ordering | P3 | **L3** (RunPod, ≤$100) | Blind anchors at 125M (250M stretch); downstream ordering read at 25M-125M (weaker than v0.2 — stated as such) |
| RQ5 KV in the budget | P4 | **L4** (local) | Post-hoc KV quant fully; native KV-QAT at 12M only |
| RQ6 the law's model wins per byte | P5 | **L5** (5090) | Capstone-lite at a 32-64MB *body* budget vs same-budget baselines |

## 3. Protocol deltas from v0.2 (all frozen project-wide)

1. **Sequence length 1024, batch 2^18 tokens** (v0.2: 2048 / 2^19), on every
   arm including the RunPod anchors — one protocol, no hardware confounds.
   Recorded per-run in the manifests (`seq_len`, `batch_tokens` are
   science-bearing RunSpec fields, included in the config hash).
2. **Tokenizer: TinyLlama_v1.1** (ungated Llama-2 32k SentencePiece; v0.2
   named Mistral's gated repo). Same "do not train a tokenizer" rule.
3. **Data: FineWeb-Edu sample-10BT**, pre-tokenized to uint16 shards once,
   `data_seed=1337` fixed; largest single-run D is 4.8B tokens (60m@80x), so
   a 10BT prep covers the whole program without repetition.
4. **Ladder extended downward**: 3m (d256 L4), 6m (d256 L8), 12m (d384 L7,
   hd96), joining 25m and 60m from the v0.2 ladder. GQA 4:1 and head_dim
   64-128 hold everywhere.
5. **Embedding accounting is a first-class limitation.** At 3M-60M the tied
   32k bf16 embedding is a large or dominant share of total bytes (84% of a
   ternary 25m artifact). The law is fitted on non-embedding N (v0.2
   principle 2) and byte budgets in L5 are **body-byte budgets**; both counts
   are reported everywhere. This is the honest cost of the local scale and is
   stated in every write-up.

## 4. Compute model and phase budgets

Measured 3080 throughput (this stack, micro runs): ~30-39k tok/s at tiny
scale. Planning numbers (conservative, seq 1024, grad accumulation):

| Size | 3080 tok/s | 5090 tok/s (est. ~3.5x) | H100 tok/s (est.) |
|------|-----------|--------------------------|--------------------|
| 3m | 35k | 120k | — |
| 6m | 30k | 105k | — |
| 12m | 22k | 75k | — |
| 25m | 15k | 52k | — |
| 60m | — (OOM-slow) | 28k | 90k |
| 125m | — | — | 55k |
| 250m | — | — | 30k |

**L0 — replication + noise floor (3080, start now).**
3m x {1.58, bf16} x {20x, 80x, 320x} x 3 seeds (18 runs) + 6m x {1.58, bf16}
x {20x, 80x} x 2 seeds (8 runs) + 6m x {1.58, bf16} x 320x x 1 seed (2 runs).
~28 runs, ~150 GPU-h ≈ **1-2 weeks of overnight 3080 time, $0.**
Exit: seed σ at 3m/6m below half the ternary-bf16 gap at 320x; MFU and
tok/s baselines recorded; queue-driven launcher running unattended.

**L1 — gap dynamics + protocol lock (3080 until the 5090 lands, then 5090).**
LR probes 6m x 5 precisions x {1x, 2x, 4x} multipliers @ ~30x (15 short
runs); FFN ablation 12m ternary: 2 new sq_relu runs (the swiglu side reuses
gap-study rows); gap study 12m x 5 precisions x {20x, 80x, 320x} with 320x
limited to {1.58, 4, bf16} (13 runs, reused in L2). 30 runs, ~140 GPU-h. **$0.**
Exit: LR rule and FFN choice frozen in `manifests/lr_rules.yaml`; the
gap-vs-D/N figure (first novel result).

**L2 — the grid (5090).**
{3m, 6m, 12m, 25m} x 5 precisions x {20x, 80x} (reusing every identical
L0/L1 row), {3m, 6m} x 5 x 320x, {12m, 25m} x {1.58, 4, bf16} x 320x,
60m x 5 x 20x, 60m x {1.58, 4, bf16} x 80x. 39 new runs (+24 reused),
~530 GPU-h ≈ **3-4 weeks of 5090 time, $0.** Fit as runs land (v0.2 §8 methodology
unchanged: Huber-on-log, multi-start L-BFGS, LOSO selection, bootstrap CIs).
Exit: fitted law with CIs; iso-memory frontier + b*(M,D) phase diagram over
8-256MB body budgets; workshop-paper draft.

**L3 — blind anchors (RunPod 1x H100, hard cap $100).**
Protocol identical to v0.2 §9: freeze the law, commit timestamped
predictions to the repo, THEN launch. 125m x {1.58, bf16} x {20x, 80x} +
125m x 4-bit x 20x: 5 runs, ~33 H100-h ≈ **$92 on-demand at $2.80/h; spot
(~$1.90/h) leaves ~40% margin.** The budget ledger's hard cap for L3 is
$100 — the launcher refuses beyond it. Stretch (only with spot savings or
found money): 250m x {1.58, bf16} x 20x (~$60-90 more).
Exit: prediction error vs the σ-implied band; precision ORDERING at 125m
correct or honestly reported broken. This closes the arXiv preprint.

**L4 — KV in the budget (local, mostly eval).**
Post-hoc KV quant {8, 4, 3, 2}-bit on all 12m/25m/60m checkpoints (eval
only, ~0 training cost) + native KV-QAT 12m x kv{8, 4} x {1.58, bf16} @ 80x
(4 runs, ~50 GPU-h). RULER/NIAH-lite at 1k-8k context. **$0.**

**L5 — capstone-lite (5090, 3-6 days).**
Train exactly what the fitted law prescribes at a **64MB body budget**
(expected: ternary at the largest N fitting 64MB ≈ ~215M params, ~40-80x —
but the law decides, not taste). Baselines at equal body bytes: best bf16
config (~32M params), a 4-bit PTQ of a ~128M bf16 model, and a from-scratch
4-bit QAT arm. Small-suite downstream + BPB + CPU tok/s via bitnet.cpp
export. **$0.** Ship as **trit-lite** with the full TRIT-Local checkpoint
ladder on HF.

**Program totals: ~$92 cash, ~830 local GPU-hours (≈ 7-9 weeks of part-time
wall clock), everything else is electricity. Generated per-run numbers:
`manifests/l0-l5.yaml` + `manifests/summary_local.md`.**

## 5. Timeline

| Window | Phase | Milestone |
|--------|-------|-----------|
| Weeks 1-3 | L0 on 3080; data prepped; ops running | σ table, replication figure, blog post 0 |
| Weeks 3-5 | L1 (5090 comes online in this window) | Gap figure, protocol frozen, blog post 1 |
| Weeks 5-10 | L2 grid on 5090 | Fitted law, headline figures, workshop draft |
| Weeks 10-12 | L3 anchors ($100) | Blind-extrapolation verdict, **arXiv preprint** |
| Weeks 12-15 | L4 + L5 | Total-footprint note, trit-lite + TRIT-Local release |
| ~Month 4 | Write-up | Full paper draft; everything public on HF + GitHub |

## 6. Risks specific to the local program

| Risk | Mitigation |
|------|------------|
| Seed noise swamps gaps at 3-12M | 3 seeds at 3m, 2 at 6m; 2σ rule enforced by the results store; contested cells get extra seeds (cheap here) |
| 320x tier monopolizes the 3080 | 320x runs queue overnight/weekends via the launcher; the 5090 absorbs the tail once online |
| Embedding bytes distort budget claims | Fit on non-emb N; body-byte budgets in L5; report both counts everywhere (§3.5) |
| $100 anchor budget overrun | Ledger hard cap; spot instances; 125m tier alone still answers RQ4 at one size step if 250m is cut |
| 5090 arrival slips | L0/L1 fully occupy the 3080 for weeks either way; L2's 3m-12m rows can start on the 3080 |
| Law only emerges above 100M | Then the anchors falsify the small-scale fit — a publishable negative result the blind protocol makes credible |
| Scooped | Same differentiators as v0.2 (native QAT, overtraining axis, memory framing) plus a new one: full desktop reproducibility |

## 7. Why this is still worth doing (the portfolio case)

1. **The law itself**: no published memory-budget scaling law for native QAT
   exists at ANY scale; 3M-250M with blind validation is a real first.
2. **The reproducibility story**: every figure regenerable on one consumer
   GPU; manifests + validation panel (11 falsifiable probes, sha256-locked
   gates) make the claims machine-checkable. That is rare and citable.
3. **The methodology artifact**: the frozen-protocol grid + KAOS-audited
   autonomous ops + adversarial validation panel is itself a demonstrable
   research-engineering contribution.
4. **The ladder**: TRIT-Local (~90 checkpoints, 5 precisions x 6 sizes x 3
   token regimes) is a public asset nobody else has published at this
   granularity, at any budget.
