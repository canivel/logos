# LOGOS: Memory-Optimal Scaling Laws for Natively Low-Bit LLMs
### Research plan v0.2, August 2026. Project LOGOS, sibling to KAOS. Model suite: TRIT (one trit = one base-3 digit = 1.58 bits, the thesis in the name).

---

## 1. Thesis

Chinchilla answered: given a compute budget, how do you split parameters vs tokens? Nobody has answered the deployment version: **given an inference memory budget in bytes, how do you allocate parameters (N), weight precision (b_w), KV precision (b_kv), and training tokens (D) to maximize quality?**

This project fits that law empirically for natively low-bit models (QAT from scratch, BitNet-style), validates it by extrapolation, and cashes it in by training a capstone model at a real device budget. The core claim we are chasing: at a fixed byte budget, more parameters at fewer bits beats fewer parameters at more bits, up to a crossover we will locate precisely.

**Why this is white space.** Precision scaling law work (Kumar et al. 2024; Ouyang et al. 2024, "low-bit favors undertrained LLMs") frames everything from the compute side. ParetoQ builds the 1-4 bit QAT ladder but optimizes for accuracy at a size, not quality per byte. Spectra trained a ternary suite but did not fit a memory-budget law or the overtraining axis. BitNet proved existence at 2B and stopped. No published work treats total inference footprint (weights + KV at a target context) as the binding constraint. That constraint is the actual one on every phone, laptop, and edge box, and the 2027 DRAM/HBM crunch makes it the constraint in datacenters too.

**Why it fits our compute.** Scaling laws are fit at 30M-1B parameters and extrapolated. The entire core program runs on single RunPod nodes (1x to 8x H100). No multi-node training until the optional stretch tier.

---

## 2. The answers we need to give (research questions)

| # | Question | Phase | Deliverable |
|---|----------|-------|-------------|
| RQ1 | Can our stack reproduce known ternary vs bf16 training dynamics, and what is run-to-run variance (seed sigma) per model size? | P0 | Loss-curve repro figure, sigma table |
| RQ2 | At fixed N and D, how does the low-bit quality gap evolve with tokens/param? Does native low-bit "favor undertrained" the way PTQ does, or does the gap saturate? | P1 | Gap-vs-D/N curves per bit width |
| RQ3 | At a fixed weight-memory budget, which precision wins, and where are the crossovers? What functional form fits L(N, D, b_w), and what is the effective capacity per bit f(b)? How close does f(b) get to the ~2 bits/param knowledge-capacity bound? | P2 | Fitted law, b*(M, D) phase diagram, iso-memory frontier plot (the headline figure) |
| RQ4 | Does the law extrapolate one size step up (fit on <=500M, predict 1B)? Does loss ordering predict downstream benchmark ordering? | P3 | Extrapolation error table, loss-vs-downstream correlation |
| RQ5 | When the KV cache enters the budget, at what context length does cache dominate, and how does the prescription shift across weight bits, KV bits, and GQA ratio? | P4 | Total-footprint law, context-dependent prescription chart |
| RQ6 | Does the model our law prescribes beat same-footprint baselines (a PTQ-quantized bf16 model, a smaller native bf16 model, BitNet b1.58 2B4T, Falcon-Edge 3B)? | P5 | Capstone model + head-to-head eval table |

Every phase closes one RQ. The paper is the six answers stitched together; the model release is the proof.

---

## 3. Experimental design principles

These rules exist so the fitted law is clean. Violating them silently is the main way this project fails.

1. **One variable moves at a time.** Architecture, tokenizer, data, data order, optimizer, and schedule are frozen across all arms. The only differences between arms at a given (N, D): the linear-layer precision (BitLinear substitution and its required norm) and the LR multiplier assigned to that precision by the P1 protocol.
2. **N means non-embedding parameters.** At 30M, a 32k vocab embedding rivals the model body. We use a fixed 32k tokenizer with tied embeddings, report both counts, and fit laws on non-embedding N (standard practice since Kaplan).
3. **Loss is the fitting target, benchmarks are the validator.** Downstream accuracy is too noisy below ~250M. We fit on smoothed final validation loss (bits-per-byte on two held-out sets), then check that loss ordering predicts benchmark ordering at 500M-1B (RQ4).
4. **Variance before claims.** 3 seeds at 30M and 60M for every cell, sigma extrapolated upward. Any reported gap must exceed 2 sigma for its size class.
5. **Same data order everywhere.** All arms at a given size see identical token sequences. Precision is the only difference in what the model experiences.
6. **Comparability beats per-arm maximum performance.** Single-stage cosine schedule for the whole grid, even though BitNet's two-stage schedule squeezes out more for ternary. The two-stage recipe is reserved for the capstone, where absolute quality matters. Document this bias explicitly in the paper.
7. **Fit forms compete.** At least two functional families, selected by held-out extrapolation, never by in-sample fit alone (Section 8).

---

## 4. Technical stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Trainer | torchtitan or Meta lingua (fork one, decide in P0 week 1) | FSDP2, torch.compile, FlashAttention, Llama-3-style reference. lingua is purpose-built for 1-7B research pretraining. Fallback for the tiniest probes: modded-nanoGPT. |
| Quantizers | ParetoQ reference implementations (1.58, 2, 3, 4-bit) + BitLinear per BitNet b1.58 2B4T report | Fake-quant with STE, absmean scaling for ternary, per-group int for 2-4 bit, 8-bit activations (WxA8) on all low-bit arms. Master weights bf16. |
| Architecture | Llama-style: RMSNorm pre-norm, RoPE, GQA 4:1, tied 32k embeddings, depth-leaning shapes | subln added where BitLinear requires it. One P1 ablation: SwiGLU vs squared-ReLU FFN at 125M ternary, then freeze the choice. |
| Data | FineWeb-Edu (350BT sample for the grid, full 1.3T for capstone) | Pre-tokenize to shards on a RunPod network volume. Fixed shuffle seed. |
| Tokenizer | Existing 32k BPE (Llama-2 or Mistral) | Do not train a custom tokenizer. Not the variable under study. |
| Eval | lm-eval-harness, version pinned in the repo | Plus a small internal BPB eval runner for the two val sets. |
| Long-context eval (P4) | RULER, NIAH, LongBench-e subset | |
| Export / inference | bitnet.cpp (ternary, i2_s packing), llama.cpp + torchao for 2-4 bit, GGUF | P0 includes a logit-parity check between the trainer and the exported artifact. |
| Tracking | W&B (all runs), configs and fitting notebooks in the repo, checkpoints to HF + network volume | |
| Ops / agents | KAOS agents for monitoring, evals, and gated launches (Section 13) | Boring fallback: cron + bash + webhooks. LOGOS never blocks on KAOS. |

**Model shape ladder** (non-embedding N ~= 12 x L x d^2, GQA 4:1, head_dim 64-128):

| N (non-emb) | d_model | Layers | Purpose |
|-------------|---------|--------|---------|
| ~25M | 512 | 8 | Variance + smoke tests |
| ~60M | 640 | 12 | Variance + LR probes |
| ~125M | 768 | 18 | P1 workhorse, reused in P2 |
| ~250M | 1024 | 20 | Grid |
| ~490M | 1280 | 25 | Grid |
| ~1.0B | 1664 | 30 | P3 extrapolation anchor |
| ~1.5B | 2048 | 30 | P5 capstone |

**KV footprint formula** (used from P4 on):
`KV_bytes = 2 x layers x kv_heads x head_dim x context_len x (b_kv / 8) x batch`
Example at capstone shape: bf16 KV at 8k context ~= 500MB. At 4-bit: ~125MB. This is why P4 exists.

---

## 5. Phase 0: Foundation and replication (RQ1)

**Goal.** A trainer we trust, a quantizer ladder we trust, and a measured noise floor. Nothing novel yet, by design.

**Build tasks**
- Fork trainer, integrate BitLinear + ParetoQ ladder behind a single `--precision` flag.
- Unit tests: STE gradient check, quant-dequant idempotence, per-precision weight histogram sanity, loss parity of `--precision bf16` against upstream trainer at 10M tokens.
- Data pipeline: download FineWeb-Edu sample, tokenize, shard to network volume.
- Spot-instance hardening: checkpoint every 30 min + on SIGTERM, auto-resume, verified by killing a live run.
- Export path: trainer checkpoint to GGUF to bitnet.cpp, logit parity within tolerance on 64 prompts.
- Eval harness wired: BPB runner + lm-eval smoke test.

**Experiments**
- 25M and 60M: ternary vs bf16, 20x and 320x tokens/param, 3 seeds each (16 runs, all tiny).
- Reproduce the qualitative Ouyang result: ternary matches or beats bf16 loss early in training, gap opens with overtraining.

**Benchmarks / metrics.** Training loss curves, final BPB on both val sets, seed sigma per (size, precision), MFU (target: >=35% bf16, >=28% low-bit at 125M+), tokens/s.

**Exit criteria (go/no-go).** Qualitative repro of the known ternary-vs-bf16 dynamic. Seed sigma at 60M below half of the ternary-bf16 gap at 320x. MFU targets hit. Export parity passes. If sigma is too high: fix data order or schedule before proceeding, do not scale noise.

**Compute.** ~60-80 GPU-h. **~$200-300.** Wall clock: 2-3 weeks part-time (most of it is engineering, not training).

---

## 6. Phase 1: Precision gap dynamics and protocol lock-in (RQ2)

**Goal.** Nail how the gap between each bit width and bf16 evolves with tokens/param, and lock the hyperparameter protocol so the P2 grid is clean. This phase produces the first novel figure: gap-vs-overtraining curves for *native* low-bit, the QAT analog of Ouyang's PTQ result.

**Experiments**
1. **LR transfer probes** at 60M: 5 LRs x 5 precisions, 1B tokens each (~25 short runs). Output: one LR rule per precision (expect ternary ~2x bf16 per BitNet; measure, don't assume). This rule is frozen for the rest of the project.
2. **FFN ablation** at 125M ternary: SwiGLU vs squared-ReLU, 20x and 80x. Freeze the winner across all arms.
3. **The gap study** at 125M: all 5 precisions {1.58, 2, 3, 4, bf16} x tokens/param {20x, 80x, 320x}, single seed (variance known from P0/P2 small sizes). 15 runs, designed with the exact P2 configs so they are reused in the grid.

**Benchmarks / metrics.** Delta-BPB vs bf16 as a function of D/N per precision; loss-curve crossover points; first downstream reads (ARC-e, HellaSwag, PIQA, SciQ, LAMBADA) at 320x to see if benchmarks separate at this scale.

**Answers to give.** Does the native-low-bit gap grow monotonically with D/N (PTQ-like), saturate, or shrink? Where does each bit width sit at 320x? Is 3-bit or 4-bit the local sweet spot at this size, consistent with ParetoQ's transition claim between "compensation" (<=2-bit) and "reconstruction" (>=3-bit) regimes?

**Exit criteria.** LR rule and FFN choice frozen and documented. Gap curves smooth and > 2 sigma. A short write-up (blog post 1) drafted: this figure alone is worth publishing.

**Compute.** ~220-260 GPU-h including probes and ablations. **~$700-1,000.** Wall clock: 3-4 weeks.

---

## 7. Phase 2: The memory-optimal sweep (RQ3, the core of the project)

**Goal.** The full grid, the fitted law, the headline figure.

**Run matrix**

| N | Tokens/param | Precisions | Seeds | Runs |
|---|--------------|------------|-------|------|
| 25M | 20x, 80x, 320x | all 5 | 3 | 45 (tiny) |
| 60M | 20x, 80x, 320x | all 5 | 3 | 45 (tiny) |
| 125M | 20x, 80x, 320x | all 5 | 1 | 15 (reused from P1) |
| 250M | 20x, 80x, 320x | all 5 | 1 | 15 |
| 490M | 20x, 80x | all 5 | 1 | 10 |
| 490M ext. | 320x | 1.58, 4, bf16 | 1 | 3 (extended tier) |

~130 runs total, ~118 of them under 10 GPU-h. The expensive tail: 250M@320x (~95 GPU-h each) and 490M runs.

**Execution notes.** Everything <=125M on single H100s (community cloud fine). 250M+ on 8x H100 SXM nodes, spot with checkpointing. Queue-driven: a simple launcher reads a run manifest, so the grid runs unattended while you work HRD/IFS hours. Kill criterion per run: loss divergence or >3x scheduled wall clock.

**Fitting.** See Section 8. Fit as runs land, not at the end; the fit tells you if a region of the grid needs densifying.

**Benchmarks / metrics.** Final BPB (fitting target); downstream suite at every 250M+ checkpoint: ARC-e/c, HellaSwag, PIQA, WinoGrande, BoolQ, SciQ, OBQA, LAMBADA (0-shot); packed-bytes footprint per checkpoint; bitnet.cpp CPU tokens/s for ternary arms.

**Answers to give.**
- The iso-memory frontier: at 0.25 / 0.5 / 1 / 2 GB weight budgets, which (N, b_w) wins, at 20x and at 320x?
- Crossover locations: does 1.58-bit x 4N beat 4-bit x 2N beat bf16 x N, and where does that ordering break?
- f(b): effective params per physical param at each bit width, with confidence intervals. Distance from the ~2 bits/param bound.
- Does the winner depend on D/N (i.e., does overtraining move the crossovers)? This interaction is the most novel result available here; nobody has published it for native QAT.

**Exit criteria.** Law fitted with bootstrap CIs; leave-one-size-out error acceptable (Section 8); headline figure drafted. Workshop paper submission target sits here (efficiency workshops: ES-FoMo, ENLSP, or equivalent 2026-2027 cycle).

**Compute.** Core ~1,500 GPU-h, +20% overhead => ~1,800 GPU-h, **~$5,000-6,000.** Extended tier (490M@320x anchors) adds ~1,150 GPU-h, **~$3,200.** Wall clock: 6-9 weeks, mostly unattended.

---

## 8. Fitting methodology (applies to P2-P4)

**Candidate forms** (L = validation BPB):
- **Form A, effective capacity:** L(N, D, b) = E + A / (N x f(b))^alpha + B / D^beta, with f(b) either free per bit width or parametric f(b) = 1 - exp(-(b - b0)/gamma) (Kumar-style). Captures "low precision = fewer effective params."
- **Form B, additive degradation:** L(N, D, b) = Chinchilla(N, D) + C x g(b) x D^delta / N^eta. Captures the Ouyang-style result that the gap grows with tokens and shrinks with size. A and B make different predictions in the overtrained regime; that is exactly the regime our 320x tier probes.
- Optionally Form C: both terms, penalized.

**Procedure.** Huber loss on log residuals, multi-start L-BFGS (Chinchilla-standard). Model selection by leave-one-size-out extrapolation error, never in-sample fit. Bootstrap over runs for CIs on every reported exponent and crossover. Acceptance bar for RQ4: predicted 1B loss within 2x the seed-sigma-implied band, and predicted precision *ordering* at 1B correct.

**Memory transform.** Weight bytes M_w = N x b_w / 8 (plus measured packing overhead per format, taken from real GGUF sizes, not theory). Optimal prescription: minimize fitted L subject to M_w <= M for each (M, D); output the b*(M, D) phase diagram.

---

## 9. Phase 3: Extrapolation and downstream validation (RQ4)

**Goal.** The credibility test. Predict 1B before training it, then train and check.

**Protocol.** Freeze the fitted law and publish the predictions internally (timestamped commit) before launching: predicted BPB for 1B x {1.58, 4, bf16} x {20x, 80x}. Then train those 6 anchors (option to trim to 5 by dropping 4-bit@20x). Compare.

**Benchmarks / metrics.** Prediction error vs band; full downstream suite plus MMLU 5-shot (finally meaningful at 1B); Spearman correlation between BPB ordering and mean-benchmark ordering across all 250M+ checkpoints; CPU inference numbers for the 1B ternary export (this is the first checkpoint people will actually download).

**Answers to give.** Does the law hold one step up? Which functional form won and what does that imply for 3-7B (stated as prediction, not claim)? Is loss a faithful proxy for downstream at these scales?

**Exit criteria.** Extrapolation within band, or an honest characterization of where it breaks (a law that bends at 1B is also a result). This closes the arXiv preprint: RQ1-RQ4 is a complete, self-contained paper.

**Compute.** ~1,050-1,450 GPU-h. **~$3,000-4,500.** Wall clock: 3-4 weeks (the 80x runs dominate).

---

## 10. Phase 4: Total-footprint laws with KV cache (RQ5)

**Goal.** Extend the budget from weight bytes to weights + KV at a target context. This is where the project separates from every pure-quantization paper.

**Experiments**
1. **Post-hoc KV sweep (cheap, eval-only):** quantize KV to {8, 4, 3, 2}-bit on existing 250M-1B checkpoints; measure BPB and long-context degradation. Establishes the free-lunch region.
2. **Native KV-QAT arms:** at 250M, train with KV fake-quant at {8, 4}-bit x weight precisions {1.58, 4, bf16}, 80x tokens, plus 32k mid-train context extension (~+10% tokens). ~8 runs.
3. **GQA-ratio arm:** 4:1 vs 8:1 at 250M ternary. GQA is architectural KV compression; the law should price it in the same currency as b_kv.

**Benchmarks / metrics.** RULER at 4k-32k, NIAH, LongBench-e subset, BPB; total measured footprint (packed weights + KV bytes at each context) per configuration.

**Answers to give.** At 8k / 32k context, what fraction of a 1-2GB budget should go to weights vs cache? Does native KV-QAT beat post-hoc KV quant the way native weight QAT beats PTQ? Where does the prescription flip from "spend bytes on params" to "spend bytes on cache precision"?

**Exit criteria.** Total-footprint prescription chart done; the capstone configuration is now fully specified by the law rather than by taste.

**Compute.** ~350-500 GPU-h. **~$1,000-1,500.** Wall clock: 3-5 weeks.

---

## 11. Phase 5: The capstone model (RQ6)

**Goal.** Train what the law prescribes at a real budget and beat everything else at that footprint. Target budget: **2GB device class** (weights + 8k-context KV + runtime headroom), i.e., recent mid-range phone / any laptop / Raspberry Pi 5 class.

**Plan of record (Tier A).** ~1.5B non-embedding params, ternary (assuming P2-P4 confirm ternary wins at this budget; otherwise train exactly what the law says), ~300B FineWeb-Edu + code/math mix tokens. Full-quality recipe now, not the grid recipe: BitNet-style two-stage LR and weight decay, data curriculum with a high-quality anneal in the last 10-15%, then SFT (~1-2B tokens of open instruction data) and light DPO. Expected artifact: ~0.4GB packed weights, ~0.55-0.7GB total at 8k context with 4-bit KV.

**Stretch (Tier B, only if credits allow):** ~3B on 500-600B tokens. Honest note: this needs multi-node or ~8 weeks on one 8x H100 node. Decide only after Tier A ships.

**Baselines to beat at equal measured footprint:**
- Best open bf16 model that fits 2GB natively (~0.9B class, e.g. current Qwen/Llama small).
- A ~4B open model PTQ-quantized to 4-bit (the "just quantize it" strawman that is actually the strongest competitor).
- BitNet b1.58 2B4T and Falcon-Edge 3B (the native-ternary incumbents).

**Benchmarks.** Full suite: ARC, HellaSwag, PIQA, WinoGrande, BoolQ, MMLU, GSM8K, IFEval (post-SFT), LAMBADA; RULER to 32k; CPU tokens/s and peak RSS via bitnet.cpp on x86 and Apple silicon; energy per token if measurable. One table, one model per column, footprint as the first row.

**Answers to give.** Did the law's prescription win at its budget? By how much, and on what? Where did it lose (be loud about this; it feeds the next iteration)?

**Release.** The capstone ships as **trit-1.5b**; the full ~140-checkpoint ladder ships as the **TRIT suite**, with configs and fitting notebooks, on HF under Canivel. The suite is the citable asset even for people who ignore the capstone.

**Compute.** Tier A: ~2,700-3,000 GPU-h (QAT throughput), ~14-16 days on one 8x H100 node. **~$8,000-9,500** including anneal, SFT, DPO, and eval. Tier B: ~$28,000-32,000 and multi-node complexity.

**Exit criteria.** Capstone matches or beats the PTQ strawman at equal footprint on the aggregate suite. If it does not, the paper honestly reports the crossover where PTQ still wins; the law is the contribution either way.

---

## 12. Compute and budget summary (RunPod)

Assumptions: H100 SXM at ~$2.80/GPU-h, 35-40% MFU bf16 stack, ~20% QAT throughput penalty, C ~= 6ND. All figures are estimates with ~+/-30% error bars; the run manifest tracks actuals.

| Phase | GPU-hours | Est. cost | Wall clock (part-time) |
|-------|-----------|-----------|------------------------|
| P0 Foundation | 60-80 | $200-300 | 2-3 wks |
| P1 Gap dynamics | 220-260 | $700-1,000 | 3-4 wks |
| P2 Core grid | ~1,800 | $5,000-6,000 | 6-9 wks |
| P2 Extended tier (optional) | ~1,150 | ~$3,200 | overlaps |
| P3 Extrapolation | 1,050-1,450 | $3,000-4,500 | 3-4 wks |
| P4 KV footprint | 350-500 | $1,000-1,500 | 3-5 wks |
| P5 Capstone Tier A | ~2,900 | $8,000-9,500 | 6-8 wks |
| Contingency (~15%) | | ~$3,000 | |
| **Core program total** | **~6,500-7,000** | **~$21,000-26,000** | **~7-9 months** |
| With P2 extended + Tier B stretch | | ~$50,000-60,000 | |

**RunPod credit ask: $25k covers the full core program through a released model. $10k gets you through the arXiv preprint (P0-P3).** Structure the ask in those two tranches.

Operational rules: spot instances everywhere with 30-min checkpointing to a network volume; single-node only through P5 Tier A; community cloud for <=125M runs; every run defined in a versioned manifest so nothing is launched by hand twice.

---

## 13. KAOS as the ops layer: agents on the project

Verdict: use it, scoped to operations and never to science. LOGOS becomes KAOS's first production workload; KAOS becomes the babysitter that protects your scarce hands-on hours. The rule that keeps both projects healthy: agents run the toil, humans run the experiment.

KAOS already has the right primitives for this job: per-agent sandboxes for anything an agent executes, audit journaling for every action taken against paid infrastructure, checkpoint/restore for agent state, MCP for wrapping RunPod, W&B, and HF as tools, and the GEPA router so cheap local models handle the always-on monitoring loop while cloud models handle judgment calls (triage, anomaly explanation). The control plane stays local on the 5090 box with RunPod driven purely through its API, which is consistent with local-first.

**Autonomy tiers**

| Tier | Agent | Actions | Gate |
|------|-------|---------|------|
| 1 (from P0) | Monitor | Watch W&B and pod state; detect NaN, divergence, stall, preemption; alert; auto-resume preempted runs from last checkpoint | Fully autonomous. Resume is deterministic; worst case is a wasted restart. |
| 2 (from P2) | Eval and hygiene | On run completion: pull checkpoint, run BPB + lm-eval, log results, archive to HF and volume; enforce manifest kill criteria (divergence, 3x wall clock) with an audit entry | Autonomous within criteria defined in the versioned manifest |
| 3 (earned) | Launcher | Start the next manifest runs when a node frees, within the budget ledger | Human approval for any 8x node launch or any run over $200 until trust is established; caps hard-coded, not prompt-enforced |
| Never | | Edit training code, hyperparameters, or the manifest; generate configs; make fitting or model-selection decisions | Science stays human. "An agent decided" is not a methods section. |

**Guardrails KAOS enforces (and gets battle-tested on)**
- Scoped credentials per agent: project-scoped RunPod key, write-scoped HF token, W&B service account. No agent holds more than its tier needs.
- A budget ledger with hard per-day and per-phase caps. The launcher refuses beyond cap regardless of its reasoning.
- Action allowlists plus the audit journal. The journal doubles as an artifact: "the LOGOS grid under autonomous ops: N runs, M preemption recoveries, $X managed, zero manual restarts" is the strongest KAOS demo you could publish, and a natural companion piece to the paper.

**Also agent-shaped, worth delegating:** daily arXiv alert triage against the scoop list in Section 14 (a digest with relevance calls you skim in five minutes), results bookkeeping into the fitting notebook's data files, and first-draft blog skeletons from W&B exports at each phase close.

**The protection rule.** LOGOS's critical path never blocks on KAOS. Any KAOS capability gap that costs more than 1-2 days gets a boring fallback (cron, bash, a webhook) and a KAOS backlog entry. If an agent tier misbehaves twice on the same failure mode, that tier drops to manual until the fix ships. Dogfooding is the point; yak-shaving is the failure mode.

---

## 14. Timeline and milestones

| Month | Focus | Milestone |
|-------|-------|-----------|
| 1 | P0 | Stack validated, sigma measured, blog post 0 (project announcement + thesis) |
| 2 | P1 | Gap-dynamics figure, protocol frozen, blog post 1 |
| 3-4 | P2 | Grid running unattended, first fits, workshop paper submitted |
| 5 | P3 | Predictions committed, 1B anchors trained, **arXiv preprint (RQ1-RQ4)** |
| 6 | P4 | Total-footprint law, capstone config locked |
| 7-8 | P5 | Capstone trained, SFT/DPO, head-to-head evals |
| 9 | Write-up | **Full paper to COLM/ICLR/NeurIPS cycle, trit-1.5b + the TRIT suite released on HF** |

Blog cadence: one post per phase under Canivel. Each phase is designed to end with a self-contained figure, so the story compounds publicly even before the paper lands. Set arXiv alerts now for: quantization-aware training, ternary LLM, precision scaling laws, KV cache compression (scoop monitoring, see risks).

---

## 15. Risks, kill criteria, pivots

| Risk | Signal | Mitigation / pivot |
|------|--------|--------------------|
| Ternary instability at 250M+ | Divergence despite BitNet recipe | QK-norm, tighter grad clip, LR backoff ladder. If still unstable: narrow the law to 2-4 bit. Still novel, smaller headline. |
| LR confound corrupts the grid | Per-precision LR rule doesn't transfer across sizes | Re-run P1 probes at 250M (cheap insurance, ~$150) before launching the 490M tier. |
| Seed noise swamps gaps at mid sizes | Effect < 2 sigma at 250M | Add seeds at 125-250M for the specific contested cells only; shrink grid elsewhere. |
| Fit non-identifiability | Forms A and B fit equally, predict differently | Densify the 320x tier (that's where they diverge). Report both with the disagreement as a finding. |
| Scooped on the core law | A precision+memory scaling paper appears | Differentiators to lean on: overtraining axis, native (not PTQ) setting, KV-in-the-budget, released suite. Ship blog posts early to timestamp. |
| No crossover found (bf16 wins per byte everywhere) | Frontier is monotone in b | Near-zero prior probability given BitNet/Spectra, but publishable negative result; pivot headline to "where and why low-bit fails." |
| RunPod spot churn burns time | Frequent preemptions on 8x nodes | Checkpoint discipline (P0 requirement), fall back to secure cloud for the long 490M/1B runs, accept ~15% cost premium. |
| Part-time bandwidth | Phases slip | The grid is queue-driven and unattended by design; your hands-on time concentrates in P0, P1, fitting, and writing. Protect 6-8 h/wk minimum. |
| KAOS becomes the project | Building runtime features instead of training models | The 1-2 day rule (Section 13): boring fallback ships, feature goes to the KAOS backlog. LOGOS's critical path never blocks on KAOS. |

---

## 16. Week 1 checklist

1. Create the repo (private for now): fork torchtitan or lingua, commit this plan as `PLAN.md`.
2. Implement `--precision {bf16,4,3,2,1.58}` with ParetoQ quantizers + BitLinear; land the unit tests.
3. RunPod: account, network volume, container image, W&B project.
4. Download FineWeb-Edu sample, tokenize with the 32k tokenizer, shard to the volume.
5. Launch the first 25M ternary-vs-bf16 pair. Kill it mid-run to verify resume works.
6. Draft the RunPod credits request using Section 12 ($10k tranche 1, $15k tranche 2).
7. Set the arXiv alerts from Section 14.
8. Stand up the KAOS Tier-1 monitor agent pointed at the first 25M pair (or the cron fallback if it costs more than a day).

The first figure you can show anyone (loss curves, ternary vs bf16 at 25M, 3 seeds) is achievable within ~10 days of part-time work. Everything after that is the same loop at larger N.
