# Chapter 10 — Designing a Clean Experiment

*A scaling law is a curve fitted through a set of training runs, so it can never be more trustworthy than the runs beneath it. "Trustworthy" here is not a vague virtue; it is a set of specific, boring rules, each invented because somebody once got burned without it. By the end of this chapter you will know the seven rules LOGOS runs by, you will be able to name the failure each one prevents, and you will have watched the project catch a confound that could have inverted its own headline result.*

## The runs are the experiment

It is tempting to think of a scaling-law project as a fitting problem: gather losses, choose a form, turn a crank, out comes an equation. In practice the fitting takes an afternoon. The hard part is producing numbers that *deserve* to be fitted, because curve fitting does not know where its numbers came from. If one run got a slightly larger batch size, or saw its data in a different order, the fit absorbs that difference into its parameters and hands back a smooth, plausible, wrong curve. Nothing in the mathematics complains.

So the discipline lives upstream, in how runs are specified. LOGOS states its rules in [`docs/methodology.md`](../docs/methodology.md) and attaches a *mechanical enforcement* to each, because a rule that lives only in a researcher's head is a rule that gets broken at 2am on the twenty-ninth run.

> **In plain terms:** the law is a summary of the experiments. If the experiments are inconsistent with each other, the summary is fiction, and the fitting step will not tell you.

## Rule 1 — One variable moves at a time

An **arm** is one configuration in a comparison: the ternary arm, the 4-bit arm, the bf16 arm. LOGOS holds architecture, tokenizer, data, *data order*, optimizer, and schedule identical across every arm at a given size and token budget. The only permitted difference is the precision of the linear layers, plus the learning-rate multiplier that precision is assigned.

Picture the failure. Suppose the ternary run had also used a larger batch size, because ternary training was memory-cheap and it seemed a shame to waste the headroom. You measure ternary at 0.07 bits-per-byte better than bf16. What have you learned? Nothing statable. The difference could be precision, could be batch size, could be the two cancelling, and you cannot subtract one out because you only ran the corner where both moved. That is not weak evidence, it is *uninterpretable* evidence.

Mechanically: every linear layer is created by one function, `build_linear()`, returning either a quantized module or a plain `nn.Linear` ([Chapter 11](11-building-the-machine.md)). There is exactly one code path where arms diverge, and unit tests assert that the bf16 model contains zero quantization modules and that all arms start from identical master weights.

## Rule 2 — N means non-embedding parameters

Every scaling law has an N in it ([Chapter 4](04-scaling-laws.md)). LOGOS defines N as the **non-embedding parameter count**: the weights inside the transformer blocks, excluding the token embedding table. Both counts are reported everywhere.

Why be pedantic? Because at small scale the embedding dominates. LOGOS uses a fixed 32,768-entry vocabulary across the whole ladder, so at width 256 that table alone is about 8.4M parameters while the "3M" model's body is about 3.2M. Call that model "11M" and the 60M model "80M", and *parameters* now means two very different mixtures at the two ends of your ladder; the exponent you fit would partly measure the shrinking embedding fraction rather than the physics you care about. Hence the standing disclosure: the tied embedding is 84% of a ternary 25M export, and byte budgets at this scale are *body*-byte budgets.

> **Why this matters:** an ambiguous definition never announces itself. It quietly rescales your x-axis, and the fit comes out looking fine.

## Rule 3 — Loss fits, benchmarks validate

LOGOS fits its law to **bits-per-byte (BPB)**, held-out loss normalized by UTF-8 bytes rather than tokens so different tokenizers stay comparable ([Chapter 3](03-measuring-quality.md)). Downstream benchmarks are kept for validation only, on a *second, disjoint* held-out set, and never enter the fit.

This is a question of measurement resolution. Below roughly 60M parameters, benchmark scores sit near chance and jitter wildly between seeds. Fit a curve to that and you are mostly fitting noise, with no independent signal left to check the fit against. The cost is accepted openly: at these sizes, benchmarks gate nothing.

## Rule 4 — Measure the variance before you make a claim

This is the rule with teeth. Train one configuration twice with different weight-initialization seeds and you get two different losses. The spread is **seed noise**, its standard deviation written sigma. LOGOS measured sigma directly at 3M and 6M by running multiple seeds per cell: between 0.006 and 0.066 BPB depending on the cell, and 0.0519 BPB at 6M.

Now the arithmetic. Comparing two single-seed runs means looking at a *difference* of two noisy numbers, and a difference of two independent draws has standard deviation sigma times the square root of 2. LOGOS requires a gap to clear two of those. At 6M the bar is 2 × 0.0519 × 1.414, about 0.147 BPB. Anything smaller is reported as observed and explicitly not claimed.

Here is the real L0 table, ternary minus bf16, with each cell's own 2-sigma bar:

| Cell | gap (ternary − bf16), BPB | 2σ | verdict |
|------|-----:|-----:|---------|
| 3M @ 20× | −0.0687 | 0.0409 | ternary better, significant |
| 3M @ 80× | +0.0178 | 0.1008 | within noise |
| 3M @ 320× | +0.2413 | 0.0611 | bf16 better, significant |
| 6M @ 20× | −0.0372 | 0.1311 | within noise |
| 6M @ 80× | +0.1675 | 0.1313 | bf16 better, significant |
| 6M @ 320× | +0.2319 | (1 seed; 3M σ implies ~0.06) | bf16 better, beyond borrowed 2σ |

Two of the six cells fall inside their noise bar and are not claimed — including the −0.0372 at 6M/20×, which has the sign the project's thesis would like. Without measured sigma, both would have been reported as small real effects. That is exactly why the rule must be in place beforehand: inventing it afterwards is indistinguishable from rationalizing. Enforcement is mechanical, and a verification probe recomputes sigma by hand from raw result rows and rejects any significance flag on a single-seed cell.

## Rule 5 — Same data order everywhere

Every arm at a given size does not merely see the same *dataset*. It sees the same tokens, in the same order, in the same batches, at the same steps. The loader draws a fixed permutation seeded by `data_seed`, frozen at 1337 for the whole project, so a bf16 run and a ternary run at 3M and 20 tokens per parameter consume byte-identical streams.

The failure without this is subtle. Text is not uniform; a batch of dense technical prose moves the loss differently from a batch of forum chatter. If arms shuffle differently, their curves differ for reasons unrelated to precision, and at gaps of 0.02 to 0.07 BPB that is not negligible. Fixing the order removes the nuisance term instead of hoping it averages out. Because "byte-identical" is easy to assert and easy to get wrong, it is tested rather than asserted ([Chapter 12](12-trusting-the-numbers.md)).

## Rule 6 — Comparability beats per-arm maximum performance

A two-stage learning-rate schedule is known to squeeze extra quality out of ternary models. LOGOS does not use it in the grid. Every arm gets one single-stage cosine schedule.

That is a deliberate sacrifice: the ternary arms are almost certainly slightly worse than they could be. But a ternary arm on a two-stage schedule against a bf16 arm on a one-stage schedule is contaminated exactly as Rule 1 forbids, and the fitted precision term would be part quantization physics and part schedule. The better recipe is reserved for the final released model, where the goal is a good model rather than a clean contrast, and the bias is documented rather than hidden.

> **Worth knowing:** documenting a known bias is not weaker than eliminating it. It is what lets someone else quantify it later. A hidden bias cannot be corrected by anyone.

## Rule 7 — Fit forms compete, and the winner is chosen out-of-sample

LOGOS does not assume a functional form. It writes down two candidates encoding genuinely different physical stories — one where low precision effectively shrinks the model, one where it adds a separate penalty that grows with training ([Chapter 9](09-the-question.md)) — and makes them compete.

The competition is not decided by which fits existing data better, since any form with more freedom can hug points you already have. Selection is by **leave-one-size-out**: fit each form with one model size removed entirely, then score by how well it predicts that withheld size. A form that wins there has demonstrated the only thing a scaling law is for. [Chapter 14](14-fitting-the-law.md) covers the machinery; what matters here is that the rule was fixed before any grid data existed.

## The confound that nearly ate the headline

Ternary arms in LOGOS run at twice the bf16 learning rate. That comes from the BitNet line of work ([Chapter 7](07-training-in-low-precision.md)), where quantized training wants larger steps. But adopting it means two things move between the ternary and bf16 arms: precision, and learning rate. The protocol itself violates Rule 1.

Mostly this would be a technicality. At 20 tokens per parameter it was not, because a larger learning rate is *also* known to help when a model is undertrained. So the early headline — ternary significantly beats bf16 at 20 tokens per parameter — was indistinguishable from a far less interesting claim: a 2× learning rate beats a 1× rate when you stop early. An external reviewer named this the largest scientific risk to the project ([Chapter 12](12-trusting-the-numbers.md)).

The fix was twelve control runs on the same frozen corpus slice, fully crossing precision against learning rate at the contested cells:

| Cell | ternary @2× | ternary @1× | bf16 @1× | bf16 @2× |
|------|-----------:|-----------:|--------:|--------:|
| 3M @ 20× | **2.0530** | 2.1185 | 2.1217 | 2.1617 |
| 3M @ 80× | 1.6381 | 1.6532 | **1.6204** | 1.6460 |
| 6M @ 20× | **1.8718** | — | 1.9089 | 1.9297 |
| 6M @ 80× | 1.4885 | — | **1.3210** | 1.3462 |

Read the last column first. Giving bf16 the doubled learning rate made it *worse* in every cell tested: 2.1617 against 2.1217 at 3M/20×, 1.9297 against 1.9089 at 6M/20×, 1.6460 against 1.6204 at 3M/80×, 1.3462 against 1.3210 at 6M/80×. The alternative explanation does not merely lack support, it points backwards. The second column closes the other half: ternary at the bf16 rate loses its 20× advantage completely, 2.1185 against bf16's 2.1217, far inside the noise bar.

The conclusion follows. The crossover is a precision effect; ternary genuinely needs its larger steps, which converts the BitNet prior from an assumption into a measurement; and since the best per-arm rate is 1× for bf16 and 2× for ternary, the best-versus-best comparison a referee would demand reproduces the main table exactly.

## The second catch: an unexamined default is a claim

That could have been the end of it. The follow-up is the sharper lesson. To set multipliers for the other precisions, LOGOS ran fifteen probes at 6M parameters and 200M tokens, sweeping the multiplier over {1×, 2×, 4×} for each precision.

![Validation bits-per-byte against learning-rate multiplier for all five precision arms at 6M parameters. The bf16 curve is steep; the ternary and 2-bit curves are nearly flat across the same fourfold range.](figures/lr-sensitivity.png)

*Figure `lr-sensitivity`: validation BPB versus learning-rate multiplier, five precision arms, 6M parameters, 200M tokens, one seed each. Lower is better. The headline is the flatness, not the minima.*

| arm | 1× | 2× | 4× | spread | best |
|-----|----:|----:|----:|-------:|------|
| ternary | 1.7351 | **1.7010** | 1.7275 | 0.034 | 2× |
| 2-bit | **1.6460** | 1.6637 | 1.6799 | 0.034 | 1× (edge) |
| 3-bit | 1.6601 | **1.6125** | 1.6597 | 0.048 | 2× |
| 4-bit | 1.7134 | 1.6920 | **1.6216** | 0.092 | 4× (edge) |
| bf16 | **1.6792** | 1.7259 | 1.8705 | **0.191** | 1× (edge) |

The spread column holds a genuine finding, discussed in [Chapter 7](07-training-in-low-precision.md) and revisited in [Chapter 13](13-what-we-found.md): across a fourfold range of learning rates, full precision moves 0.191 BPB while ternary and 2-bit move 0.034.

But look at the "best" column. Three of five arms have their best value **at an edge of the probe grid**. To **bracket** a minimum means having measured points on both sides of it; these three are unbracketed, so what looks like an optimum is really the direction of an optimum you did not reach.

The bf16 case is dangerous in a specific direction. If full precision actually prefers something below 1×, then bf16 has been running handicapped in every comparison so far, and the ternary advantage at 20 tokens per parameter is inflated. Notice the control runs do not catch this. They tested bf16 at 2×, found it worse, and never looked below 1×. A careful fix for one confound left a neighbour standing.

So the grid is being extended before the rule freezes: twelve more runs adding 0.5× for every arm, 0.25× for bf16 and 2-bit, 8× for 4-bit, plus two extra seeds on the 4-bit comparison that is the largest difference still inside the noise bar. Until they land, the multiplier rule is written down as *provisional*, with an explicit freeze principle — deviate from the prior only on significant evidence — rather than read off unbracketed minima.

The general lesson deserves its own sentence. **An unexamined default silently becomes a claim.** The grid {1×, 2×, 4×} was chosen as a reasonable sweep, not as an assertion about where the optimum lives. The moment a number is read off it and frozen into the protocol, the grid's boundaries become part of every downstream result, invisibly.

## Pre-registration: writing the prediction down first

One rule remains, and it governs the project's central credibility test. LOGOS plans to fit its law on models from 3M to 60M and then predict the loss of a 125M model it has not trained. Before those validation runs launch, the fit is frozen and its predictions — a BPB number per anchor arm, with the sigma-implied band — are committed to the repository with a timestamp.

This is **pre-registration**, and its purpose is not ceremony. Fitting involves dozens of small judgment calls: which rows to include, how to weight them, whether an outlier is a bad run or real data. Each is defensible in isolation. Make them *after* seeing the anchor results and you will unconsciously make the ones that pull the prediction toward what you already know, experiencing the whole thing as ordinary diligence. The only reliable defence is to make the calls while the answer is genuinely unavailable, and leave a timestamped record proving it. The acceptance bar is stated in advance too: anchors must land within twice the seed-sigma band, and the predicted ordering of precisions must be correct. A miss gets reported as a bend in the law rather than quietly absorbed. No anchors have been run and no law has been fitted; what exists today is the protocol and the code that checks it.

## What to remember

Seven rules, one theme: a fitted law's credibility is built before any fitting happens, by ensuring every run differs from every other in exactly the ways you intend and no others. One variable moves at a time, N always means the same thing, loss fits while benchmarks validate, nothing is claimed inside its measured noise, data order is frozen project-wide, comparability outranks squeezing the last drop from any single arm, and the form is chosen by out-of-sample prediction. The learning-rate confound shows why the rules cannot be applied casually: the project's own protocol violated Rule 1, twelve control runs were needed to show the headline was a precision effect, and the follow-up found three of five probe optima on an unbracketed grid edge. Both catches came from looking deliberately, not from anything going visibly wrong. That is the whole craft, because a wrong number in this business does not look wrong. It looks like a result.
