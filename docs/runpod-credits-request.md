# RunPod Research Credits Request — Project LOGOS

**DRAFT for review — do not send as-is.** Numbers from PLAN.md §12; update
the contact/links block before submitting.

**Applicant:** Danilo Canivel (d.canivel@gmail.com) · HF: canivel · Repo: private (shared on request)
**Ask:** $25,000 in two tranches · **Hardware:** single-node 1×–8× H100 SXM, spot, community cloud where possible

---

## One-paragraph pitch

Chinchilla answered how to split a *compute* budget between parameters and
tokens. Nobody has answered the deployment version: given an **inference
memory budget in bytes**, how should you allocate parameter count, weight
precision, KV-cache precision, and training tokens? LOGOS fits that law
empirically for natively low-bit LLMs (QAT from scratch, BitNet-style),
validates it by extrapolation one size step up, and cashes it in by training
**trit-1.5b** — a ternary capstone model at a real 2GB device budget — plus a
~140-checkpoint open ladder (the TRIT suite) released on Hugging Face. Every
GPU-hour runs on RunPod.

## Why RunPod specifically

- The whole core program is **single-node** (1× H100 for ≤125M models,
  8× H100 SXM for 250M+), spot-priced, with 30-minute checkpointing to a
  network volume and verified auto-resume — built for spot economics.
- Queue-driven and unattended: every run is defined in a versioned manifest;
  an agent ops layer (KAOS) monitors, auto-resumes preemptions, evaluates,
  and archives, under a hard-capped budget ledger. Nothing is launched by
  hand twice.
- Public artifacts credit the compute provider: one blog post per phase, an
  arXiv preprint, a conference submission, and the model + suite release.

## The program (phases, GPU-hours, cost)

| Phase | What | GPU-h | Est. cost |
|-------|------|-------|-----------|
| P0 | Stack replication + noise floor (25–60M, ternary vs bf16) | 60–80 | $200–300 |
| P1 | Precision-gap dynamics vs overtraining; protocol lock | 220–260 | $700–1,000 |
| P2 | Core grid: ~130 runs, 5 precisions × 25M–490M × 20–320 tokens/param; the fitted law | ~1,800 | $5,000–6,000 |
| P3 | Blind extrapolation test at 1B (predictions committed before launch) | 1,050–1,450 | $3,000–4,500 |
| P4 | Total-footprint law: weights + KV cache at 4k–32k context | 350–500 | $1,000–1,500 |
| P5 | Capstone: trit-1.5b, ~300B tokens, two-stage recipe + SFT/DPO | ~2,900 | $8,000–9,500 |
| — | Contingency (~15%) | | ~$3,000 |
| **Total** | | **~6,500–7,000** | **~$21,000–26,000** |

## Tranche structure

- **Tranche 1 — $10,000:** carries P0 through P3, ending in a complete,
  self-contained arXiv preprint (the fitted memory-optimal law + the 1B
  extrapolation test) and the first public checkpoint people will download
  (1B ternary, CPU-deployable via bitnet.cpp).
- **Tranche 2 — $15,000:** P4 + P5, ending in the trit-1.5b release —
  a law-prescribed model at a 2GB footprint benchmarked head-to-head against
  same-footprint PTQ baselines, BitNet b1.58 2B4T, and Falcon-Edge — plus
  the full TRIT suite and fitting notebooks on Hugging Face.

## What exists already (de-risking)

The full training/eval/fitting stack is implemented and tested: quantizer
ladder (ternary absmean + 2–4-bit per-group QAT), frozen-protocol trainer
with verified kill-and-resume, BPB + pinned lm-eval harness, packed-export
parity gates, scaling-law fitting with leave-one-size-out selection and
bootstrap CIs, all 209 runs pre-specified in versioned manifests, and an
independent validation panel (11 falsifiable probes, 49 sha256-locked kill
gates, currently all green) that re-verifies every benchmark and result. A
micro-scale P0 replication already reproduces the known ternary-vs-bf16
undertrained-regime dynamic end-to-end.

## Deliverables & timeline (part-time, ~9 months)

Month 1–2: P0–P1 + blog posts 0–1 · Month 3–4: grid + workshop paper ·
Month 5: arXiv preprint (tranche-1 exit) · Month 6: total-footprint law ·
Month 7–8: capstone training · Month 9: full paper (COLM/ICLR/NeurIPS
cycle), trit-1.5b + TRIT suite public on HF, RunPod acknowledged in all of
the above.
