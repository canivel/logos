# Chapter 12 — Trusting the Numbers

*Software testing asks whether code does what its author meant. Research code has a nastier failure mode: the author meant the wrong thing, and the code obeys perfectly. This chapter is about the layer LOGOS built to catch that — eleven independent probes behind forty-nine kill gates, cryptographically locked before anyone looks at the results. You will see what "independent" means in practice, why locking a pass/fail rule in advance is the same idea as pre-registering a clinical trial, and what six real defects looked like before they were found.*

## The bugs that pass every test

Imagine your bits-per-byte evaluator divides by the wrong denominator — the token count instead of the UTF-8 byte count. Every unit test you would naturally write still passes. The function returns a float. It is positive. It falls as the model trains. It is stable across reruns. Every property a reasonable test asserts holds, and every number the pipeline produces is wrong by a constant factor that survives into a fitted exponent.

That is the characteristic shape of a research bug: no crash, no NaN, nothing suspicious, just plausible numbers. Ordinary tests are blind to it because they encode the same understanding that produced it. The author writes test and implementation from one mental model, and if the model is wrong, both are wrong together and agree with each other beautifully.

> **In plain terms:** a unit test checks the code against its author's intention. Nothing in a normal test suite checks the intention.

The only defence is a second, deliberately separate derivation of the same quantity, produced without looking at the first, plus a rule written in advance for what counts as agreement.

## The panel

LOGOS calls that layer the **validation panel**. It lives in `validation/`, entirely outside `src/logos/`, and runs with one command: `logos-validate --all`.

It holds eleven **probes**, each responsible for one load-bearing quantity, and each declaring a few **kill gates** — specific, falsifiable pass/fail conditions. Forty-nine gates in total. A probe returns **ACCEPT** when every gate passed, **REJECT** when one failed, or **VOID** when it could not run or its gates were altered after locking. VOID counts as REJECT for release purposes.

| Probe | What it verifies |
|-------|------------------|
| `quant_numerics` | Quantizers produce the claimed level sets, scales, and error bounds |
| `ste_gradients` | The straight-through estimator is identity in the clip region, zero outside |
| `data_order` | Arms consume bit-identical token streams; resume does not fork the stream |
| `bpb_definition` | The BPB runner matches the definition, including partial windows |
| `export_parity` | A packed artifact reloads to bit-identical weights |
| `footprint` | Measured packed bytes match the format's exact bit arithmetic |
| `fit_recovery` | The fitter recovers known laws; selection picks the true form |
| `stats_discipline` | Sigma is a real standard deviation; no single-seed cell is "significant" |
| `manifest_integrity` | Result hashes match manifest entries; ledger caps really refuse |
| `benchmark_pinning` | The harness version pin is enforced; validation splits are disjoint |
| `training_sanity` | Real run curves are finite, gapless, correctly accounted, resume-identical |

Round 1 ran on August 5, 2026 and returned ACCEPT on all eleven probes and all forty-nine gates — after fixing six real defects it found along the way. The roster with each probe's independent reference lives in [`validation/README.md`](../validation/README.md).

## Commitment one: independence

A probe must re-derive the quantity it checks from first principles. It may *call* the code under test to obtain the value being checked. It may never call that code to compute what the value should be.

This sounds pedantic until you picture the alternative. A probe that verifies the ternary quantizer by calling the ternary quantizer and comparing the result to itself will pass forever, including on the day the equation is wrong. It checks nothing except determinism. The panel's README puts it bluntly: such a probe "is worthless and gets rejected in review."

Independence takes five concrete shapes here. Reimplementing the mathematics in plain NumPy from the equations in the source papers. Deriving gradients by hand and checking them against finite differences. Writing a separate decoder for a binary format from its specification. Constructing a case whose answer is known in closed form. Generating synthetic data from parameters the code under test never sees, then asking whether it can find them again.

There is an organizational version of the same idea: the round-1 probes were written by three separate builder agents working against the locked contract, none of whom had authored the subsystems they were verifying. That is the software equivalent of not letting a lab grade its own samples.

## Commitment two: the gate is locked before the result is seen

Each probe declares its gates in advance, in words. The gate specification is then hashed with sha256 and the hash written into `validation/locks/`, timestamped, before any results are examined. Here is a real lock, for the BPB probe:

```json
{ "probe": "bpb_definition",
  "sha256": "bd8a5c58fd6f3ca953e17ea8b48e1d8178f1a6b4aa16f53102b8ccd9fc5fd564",
  "gates": [
    ["G1", "uniform-logits closed form: constant-zero-logit model gives
            BPB = T*log2(V)/B ... within 1e-9"],
    ["G2", "trained tiny model: bpb() matches the independent per-token
            NLL loop within 1e-6 bits/byte, incl. a partial final window"],
    ...],
  "registered": "2026-08-05 19:44:04" }
```

Change the gate text afterwards and its hash changes, the lock no longer matches, and the probe is VOID. Not failed — void, meaning it provides no evidence at all.

Why bother? Because of a very human failure. You write a tolerance of 1e-9, you run the probe, you get 3e-9, and every engineering instinct says *that's basically fine, float arithmetic, loosen it to 1e-8*. Sometimes that judgment is right. The problem is that you cannot tell, from inside the moment, whether you are exercising expertise or rationalizing a failure — and if the bar can move after you see the measurement, the bar has stopped carrying information.

This is the logic of pre-registration in clinical trials. A trial that declares its primary endpoint before enrolling patients makes a falsifiable prediction; a trial that picks its endpoint after unblinding will find *something* significant, because with enough outcomes measured, something always is. The gate lock is that commitment in miniature, and the same one [Chapter 10](10-designing-a-clean-experiment.md) described for the extrapolation predictions.

> **Why this matters:** the lock does not make the researcher more honest. It makes a specific kind of dishonesty impossible to perform accidentally, which is the kind that actually happens.

## Four probes up close

**`quant_numerics`** checks the quantizers, the mathematical heart of the project. Its reference is a hand-built NumPy reimplementation written from the equations in the BitNet b1.58 and ParetoQ papers, importing nothing from the LOGOS quantizer modules, then compared element by element. Its gates require ternary levels to be exactly {−gamma, 0, +gamma} with gamma equal to the mean absolute weight; require 2/3/4-bit reconstructions to match the NumPy per-group reference, with no more than 2^k distinct levels in any group; and require reconstruction error to fall monotonically from 2 to 3 to 4 bits on identical master weights. Round-1 agreement was about 4e-9, which is floating-point rounding and nothing else.

**`bpb_definition`** is the cleanest illustration of the closed-form idea. Take a model forced to output a uniform distribution — literally a module returning constant zero logits. Its loss is not a matter of opinion. Every token costs exactly log2(V) bits, so bits-per-byte must be tokens × log2(vocab) ÷ bytes, with no model, no training, no approximation. If the runner disagrees, the runner is wrong. Round 1 matched to 6e-13. A second gate compares the runner against an independently written per-token loop, on a case deliberately chosen so the final window is partial. A third requires every token but the first to be scored exactly once.

**`fit_recovery`** verifies machinery whose correct answer is normally unknowable, by manufacturing a situation where it is known. The probe hand-codes one candidate law form inside itself, draws hidden truth parameters, generates a noisy grid over sizes and token budgets, and hands only the grid to the fitter. Does it recover the hidden exponents and per-precision capacity factors within stated tolerances (round 1: exponents within 13%, capacity factors within 3%)? Does leave-one-size-out selection pick the form that actually generated the data? Do prescriptions from the fitted law agree with those from the true law across the memory-and-tokens grid (round 1: 100% of cells)? A fitter that cannot pass on data it is guaranteed to be able to explain has no business being pointed at real data.

**`data_order`** enforces the rule from [Chapter 10](10-designing-a-clean-experiment.md) that every arm sees identical tokens. It hashes each batch tensor with sha256 and compares fifty-batch streams from two independently constructed loaders. It builds loaders exactly the way the trainer does for two run specifications differing only in precision, and requires twenty batches of identical hashes. It checks that starting from step k matches running from zero and discarding k batches, including values of k that cross an epoch boundary. And it re-derives from the raw shard bytes, rather than the loader's bookkeeping, that one epoch consumes every window exactly once.

## Six defects, six lessons

None of the six would have announced itself.

**One: the CUDA resume crash.** Loading a checkpoint with `map_location="cuda"` moved the random-number-generator state onto the GPU, where PyTorch refuses it. The CPU-only unit test could not see it because on CPU the move was a no-op ([Chapter 11](11-building-the-machine.md) tells the full story). *Lesson: a test that cannot reach the failing configuration is not evidence about that configuration.*

**Two: status clobbering.** Re-invoking an already-completed run rewrote its status file with a final loss of NaN. Since the queue runner and the ops agents read that file to decide what is done, a stray re-invocation could have marked a good run as garbage. Completed runs are now immutable no-ops. *Lesson: idempotence is a correctness property, not a convenience.*

**Three: a schema mismatch.** The evaluation reader did not accept the index schema the data pipeline actually writes — two components of one project, each internally consistent, disagreeing at the seam. It would have raised on the first real evaluation, after the training hours were spent. *Lesson: test at the interfaces with real artifacts, not with fixtures both sides agree on.*

**Four: a non-scientific field leaking into the config hash.** The `phase` field was in the hashed set, so a run reused across two phases — a documented, intentional reuse — hashed to two identities, silently breaking the link between results and manifests for fifteen grid runs. The `manifest_integrity` probe caught it as a REJECT and `phase` was removed. *Lesson: provenance breaks quietly. It has to be checked, not assumed.*

**Five: the parity criterion itself was wrong.** The most interesting of the six, because the bug was in a check rather than in the thing checked. Exporting a model and reloading it should give back the same model, and the obvious test is the largest absolute difference between the two versions' output logits. That criterion is invalid for deep quantized models. On reload the ternary scale is recomputed as an fp32 mean, which can differ from the training-time value by one **ulp** — the smallest representable step at that magnitude. The discontinuous int8 activation quantization then amplifies that seed through depth, because a value sitting exactly on a rounding boundary can flip to the neighbouring integer. Measured: about 5e-6 after the first block, growing to roughly 0.15 by the sixth, while the output *distributions* stayed equivalent at a Kullback-Leibler divergence near 3e-4. A max-absolute-difference gate would have failed a perfectly correct export. The gate became bitwise comparison of packed codes and scales, with a divergence bound as secondary. *Lesson: an unfit metric produces false alarms, and false alarms train people to ignore alarms.*

**Six: a one-ulp mismatch between forward and export.** The integer-quantization export divided by a scale that could differ from the forward path's gradient-rescaled fp32 value by a single ulp, so a master weight sitting exactly on a boundary could export a different code than training had used. Nothing would have crashed; the deployed model would simply have differed from the trained one in a few weights. Export now replicates the forward arithmetic exactly. *Lesson: "the same formula" and "the same floating-point operations in the same order" are different claims.*

## The follow-up: tolerances do not transfer

The corrected parity gate then needed correcting again. The divergence bound had been calibrated on freshly initialized shallow models, which reload comfortably under 1e-3. Then the most-trained model in the grid — 6M parameters at 320 tokens per parameter, with the sharpest output distributions in the project — reloaded at 1.9e-3, over the bound, with **all fifty-six quantized layers bitwise identical**. The weights were provably correct. The bound was wrong.

The mechanism is that amplification grows with training. A well-trained model's activations sit closer to quantization boundaries and its output distribution is peakier, so the same ulp-level perturbation propagates further. The bound now sits at 5e-3 with the calibration evidence written into the source beside it, and real reconstruction errors land orders of magnitude above that, so the gate still bites.

> **Worth knowing:** a tolerance measured on a toy model is a statement about toy models. Every numerical threshold in a research codebase carries an unstated "under the conditions where I measured it", and those conditions change as the project gets better at its job.

This recalibration is legitimate precisely because it did not weaken a gate to let a claim through. It corrected a bound after establishing, by an independent route — bitwise identity of every packed layer — that the artifact was correct and the threshold miscalibrated.

## A second kind of review: the direction, not the code

The panel checks whether the implementation matches the intention. It cannot tell you whether the intention is a good scientific plan. For that, LOGOS commissioned a separate review: an AI reviewer panel, kimi3, playing five roles — scaling-laws methodologist, quantization systems lead, statistics reviewer, program chair, devil's advocate — reading the science-bearing code, the plans, the research notes, the raw results file, and the surrounding literature. The verdict was **7.5 out of 10, accept with revisions**, with four cheap fixes flagged before they became load-bearing.

| Finding | Status |
|---|---|
| F1 — the learning-rate confound: "ternary beats bf16 undertrained" was indistinguishable from "higher LR wins undertrained" | **Fixed.** Twelve control runs crossing precision against learning rate landed ([Chapter 10](10-designing-a-clean-experiment.md)) |
| F2 — data shortfall: 2.5B tokens on disk against later phases needing ~10B | **Fixed.** The corpus was extended to 11.37B tokens before the deeper runs started |
| F3 — embedding bytes undermine the headline: the 32k embedding can exceed the body budget at deployment scale | **Scheduled** before the capstone phase: total bytes beside body bytes, plus a fixed disclosed embedding treatment |
| F4 — the prescription ignores training compute, which a referee will not | **Fixed.** Iso-training-FLOPs contours added to the phase diagram; zero extra runs required |

A fifth item — the fitting objective weighting every cell equally despite seed noise varying about fourfold across cells — was implemented as inverse-variance weighting. Two further findings remain open by choice and are documented rather than quietly dropped: a control arm separating weight quantization from activation quantization, and the shape-robustness question about the very wide, very shallow models at the bottom of the ladder.

The review's most valuable output was not any single fix. It was the observation that the largest risk to the project was not conceptual. The question was sound and the early data supported the thesis; the threats were four specific, cheap, findable holes. That is usually where research risk actually lives.

## The rule that makes it mean something

An apparatus like this only works if it has authority, and the authority comes from one rule. The panel's verdict gates publication of results from the repository: `logos-validate --all` must return ACCEPT before any result is cited. When a gate fails there are exactly two permitted responses — fix the stack, or retract the claim. Weakening the gate is not on the list, because a gate changed after registration does not become more lenient, it becomes no gate at all.

That is a costly commitment, and it is meant to be. A verification layer you can negotiate with is a verification layer that tells you what you wanted to hear, on exactly the days when it matters most that it does not.

## What to remember

Unit tests check code against its author's intention, which leaves research code exposed to its worst failure mode: a wrong intention faithfully implemented, producing numbers that look entirely reasonable. LOGOS answers with eleven probes and forty-nine kill gates that re-derive every load-bearing quantity independently — NumPy reimplementations from the source papers, hand-derived gradients, a separately written bit-stream decoder, closed-form cases with known answers, and synthetic data generated from parameters the fitter never sees — each probe's criteria sha256-locked before any result is examined, exactly as a clinical trial registers its endpoint before unblinding. Round 1 caught six real defects, including two where the *check itself* was wrong rather than the code, and a follow-up showed that a tolerance measured on fresh toy models does not transfer to a well-trained one. An external review of the research direction scored the project 7.5 out of 10 and named four cheap holes, three of which are now closed. What holds the structure up is the rule that a failing gate is met by fixing the stack or retracting the claim, never by moving the gate.
