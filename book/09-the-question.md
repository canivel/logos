# Chapter 9 — The Question

*After this chapter you will be able to state the question LOGOS exists to answer, precisely. You will understand the trade at its center: at a fixed byte budget, fewer bits per parameter buys more parameters, but each one is worth less, and somewhere between those two curves is an optimum. You will know what an iso-memory frontier and a phase diagram are, and you will see the real evidence collected so far — including the finding that makes the question genuinely open.*

## A budget you can hold in your head

Forget scaling laws for a moment. You have a device with 2 GB available for a language model: a midrange phone, a laptop, a slice of a shared GPU. The number is fixed and not negotiable, because the memory is physically not there.

Now spend it. You could load a 1-billion-parameter model at 16 bits per weight, which fills the budget exactly. Or 4 billion parameters at 4 bits. Or roughly 8 billion at 2 bits, or about 10 billion at the theoretical 1.58 bits of a ternary model. Same 2 GB every time, wildly different models.

And that is one choice of several. You also decide how long to train, which affects quality independently of size, and from [Chapter 8](08-the-kv-cache.md) you know you must decide how precisely to store the KV cache, because serving 8,192 tokens of context spends a sizeable slice of that 2 GB no matter what you do with the weights.

Four dials, one budget. Which setting gives the best model? Nobody has published the answer.

## The question, stated formally

Maximize model quality subject to weight_bytes + kv_bytes ≤ M, choosing the number of parameters N, the weight precision b_w, the cache precision b_kv, and the number of training tokens D.

Quality is measured as loss, specifically **bits-per-byte** (BPB): how many bits the model needs, on average, to encode one byte of held-out text. Lower is better, and bits-per-byte rather than per-token perplexity keeps the number comparable across models that tokenize differently.

Compare it to the law it is modeled on. Chinchilla asked how to split a **compute** budget between parameters and tokens, and its answer, that most models of the era were badly undertrained, reorganized the field. LOGOS asks the deployment version, with bytes as the scarce resource instead of FLOPs, and with two dials the compute framing does not have: the precision of the weights and of the cache. What makes it a research question rather than an engineering exercise is that these are *natively low-bit* models, trained from scratch at their target precision using the QAT machinery of [Chapter 7](07-training-in-low-precision.md) — not 16-bit models compressed afterward, the difference [Chapter 6](06-quantization.md) argued matters most below 4 bits.

## The trade at the center

**Fewer bits buy more parameters.** Weight bytes are N × b / 8, so at a fixed budget the affordable N is inversely proportional to b. Going from 16 bits to 1.58 gives 16 / 1.58 ≈ 10.1 times more parameters; going to 2 bits gives 8 times more. This part is exact and needs no experiment.

**But a low-bit parameter is worth less.** A ternary weight carries three possible values; a bf16 weight carries tens of thousands of distinguishable ones. If they were equally useful, low precision would win unconditionally at every budget and there would be nothing to study.

> **In plain terms:** you are choosing between a large team of clumsy workers and a small team of skilled ones, on a fixed payroll. The right answer is not obvious, and it is not the same for every job.

### Effective capacity

To make "worth less" precise, define the **effective capacity** f(b): the fraction of a full-precision parameter's usefulness that a b-bit parameter retains. By construction f(16) = 1, since bf16 is the reference; for any lower precision f(b) sits below 1, and how far below is empirical.

The definition earns its keep in what it lets you write. A model with N parameters at b bits behaves, to a first approximation, like a full-precision model with N × f(b) parameters. Parameters scale as 1/b, usefulness per parameter scales as f(b), and quality follows the product of the two, so the product has a maximum. In the code this is not a metaphor: `FormAFree` in `src/logos/fitting/forms.py` fits one free f per bit width and `FormAParam` fits a smooth curve through them, and the two compete on which extrapolates better.

### A worked example, illustrative only

Suppose the measured capacities came out as below. **These numbers are invented for the arithmetic. They are not measurements, and the real ones do not exist yet.**

| Precision | Params affordable in 2 GB | Illustrative f(b) | N × f(b) |
|---|---:|---:|---:|
| bf16 | 1.0B | 1.00 | 1.00B |
| 4-bit | 4.0B | 0.55 | 2.20B |
| 3-bit | 5.3B | 0.42 | 2.24B |
| 2-bit | 8.0B | 0.28 | 2.24B |
| ternary | 10.1B | 0.20 | 2.02B |

In this made-up world every low-bit arm beats full precision by roughly a factor of two in effective parameters, and the optimum is a broad plateau between 3-bit and 2-bit with ternary just past the turn. Notice the shape of the last column: it rises steeply, flattens, then dips, because the parameter gain from dropping bits is steady and multiplicative while the capacity loss accelerates as the levels run out. Change f(1.58) from 0.20 to 0.35 and ternary wins outright; change it to 0.08 and ternary falls below 4-bit. The entire prescription hinges on the shape of one function that has to be measured.

![Illustrative. At a fixed byte budget, fewer bits per parameter raises the affordable parameter count along the hyperbola N = 8M / b, while the effective-capacity curve f(b) falls in the same direction. Their product rises, peaks and falls, which is why an optimum exists rather than a blanket preference for the lowest precision available. Where the peak sits here is schematic; locating the real one is what the grid is being run for.](figures/iso-memory.png)

## How much can one parameter possibly hold?

There is a useful reference point for reading f(b). Work on knowledge capacity has found that a transformer parameter stores roughly 2 bits of factual knowledge, largely regardless of the precision it is stored in. That is striking, because it says most of a bf16 weight's storage is not carrying knowledge — a reason to expect low-bit models to lose less than their bit count suggests. It also gives a ceiling: a b-bit weight cannot store more than b raw bits, so the applicable bound is min(b, 2).

The project's `effective_capacity_summary` reports, per bit width, the fitted f, the implied knowledge bits, the bound, and the fraction of the bound achieved. That last column asks a sharper question than "how good is low precision" — it asks how much of the storage a low-bit weight *physically has* the training procedure manages to use. It is empty today, because the fit has not been run on real data.

## Two deliverables

The fitted law, whose machinery [Chapter 14](14-fitting-the-law.md) takes apart, is not the product. It is the input to two artifacts a practitioner can use.

### The iso-memory frontier

**Iso-memory** means "equal memory," the way an isobar connects points of equal pressure. The **iso-memory frontier** answers: at each byte budget, what is the best loss achievable at each bit width, using the largest model that fits? As a chart it is one curve per precision, budget on the horizontal axis and loss on the vertical, and where the curves cross is where the recommendation changes. The project plans these over body-byte budgets from 8 MB to 256 MB, the sizes its own artifacts occupy.

### The b*(M, D) phase diagram

A **phase diagram** is a picture borrowed from physics, where it shows which state of matter is stable at each combination of temperature and pressure: regions of solid, liquid, gas, with boundaries between them. No curve is plotted; the chart is a map, and its content is which region you are in.

The b*(M, D) diagram is the same idea with byte budget M on one axis and training tokens D on the other, and the color at each point is the *optimal weight precision* there. That is the deliverable a practitioner would use: your hardware fixes the budget, you know roughly how much data you can train on, you look up the point and read the color. `phase_diagram` in `prescribe.py` computes exactly this array by solving the constrained optimization at every grid point.

**No such diagram has been produced from real data in this project.** The versions in the repository are demonstrations on synthetic ground truth, used to verify that the fitting machinery recovers parameters it was given.

## The twist: the answer moves with training duration

If effective capacity were a fixed property of a bit width, the phase diagram would have vertical boundaries and D would not matter. The evidence says otherwise: the winning precision appears to depend on D/N, tokens per parameter, which is to say on how heavily a model has been trained relative to its size.

### What has actually been measured

Two results are in hand, told in full in [Chapter 13](13-what-we-found.md) and repeated here because of what they do to the question. Both are small, both are real, and both carry significance bars from measured seed-to-seed noise rather than from eyeballing.

> **Why this matters:** a gap smaller than the noise bar is not a small effect, it is no measured effect at all. Every claim below is stated against a bar computed from repeated runs that differ only in their random seed.

At **3M parameters**, the ternary-minus-bf16 gap in bits-per-byte across three training durations:

| Tokens/param | Gap (ternary − bf16) | 2σ bar | Verdict |
|---|---:|---:|---|
| 20× | −0.0687 | 0.0409 | ternary better, significant |
| 80× | +0.0178 | 0.1008 | inside noise, no claim |
| 320× | +0.2413 | 0.0611 | bf16 better, significant |

Negative means ternary wins. At 20 tokens per parameter ternary beats full precision by 0.0687 BPB, past the 0.0409 bar; at 80× the difference is inside the noise and nothing is claimed; at 320× ternary *loses* by 0.2413 BPB against a 0.0611 bar. The sign flips — same architecture, same data, same data order, same everything except how long it trained.

At **12M parameters** and 20 tokens per parameter, the comparison widens to the whole ladder:

| Arm | Val BPB | vs bf16 |
|---|---:|---:|
| 2-bit | 1.4945 | −0.104 |
| 4-bit | 1.4955 | −0.103 |
| 3-bit | 1.5153 | −0.083 |
| ternary | 1.5201 | −0.078 |
| bf16 | 1.5982 | — |

Every quantized arm beats full precision. Note what is *not* claimed: the ordering among the four quantized arms sits inside single-seed noise, so 2-bit heading the table is no evidence that 2-bit is best.

### What it implies

The answer to "which precision should I use?" is not a constant. It is a function of the training regime, and the direction of the effect is uncomfortable, because the regime where low precision loses is the regime everything real lives in.

Chinchilla-optimal is around 20 tokens per parameter, and deployed models are trained far past that, often ten or twenty times past it, because inference cost scales with model size while training cost is paid once. Every model you have used is heavily overtrained by that standard — exactly where the 3M evidence says ternary's advantage has not merely evaporated but inverted. A recommendation stated without reference to D is therefore incomplete.

## The white space

It is fair to ask why nobody has done this. Several groups have done adjacent things well, and the gap between them is narrow and specific.

Precision scaling laws exist from the **compute** side, relating precision to training cost and predicting degradation, but they treat memory as an output rather than the binding constraint. **ParetoQ** built the 1-to-4-bit ladder carefully and established where the regimes change, optimizing accuracy at a fixed model size rather than under a byte budget. **Spectra** trained and released a suite of ternary models without fitting a law for allocating a memory budget. **BitNet** proved that natively-trained low-bit models can match full-precision ones at some sizes, the necessary existence proof, and stopped there.

Each is a piece. Nobody has put total inference footprint, weights plus cache, on the left side of the constraint and asked what allocation maximizes quality for natively-trained low-bit models. That is the sentence LOGOS is trying to be able to write, and it cannot write it yet. The law is not fitted, and there is no released model, no completed cloud anchor, no measured f(b), no real phase diagram. What exists is a validated stack, a frozen protocol, a crossover measured with a proper noise floor at 3M and 6M, a first full-ladder cell at 12M, a learning-rate finding with a known gap in its probe grid, and a grid that is still running.

## What a good answer would be worth

It is easy to read this chapter as an interesting curiosity about bit widths. It is worth being explicit about why the answer would matter, and the cleanest way to see it is to notice that the obvious reasoning gives the wrong answer.

At a fixed byte budget, ternary weights buy roughly ten times more parameters than 16-bit ones. So on capacity arithmetic alone, low precision wins unless a ternary parameter retains less than about a tenth of a full-precision parameter's usefulness. That is a very low bar. By that logic low precision should win essentially always, at every budget, for every model.

It does not. The measurements in [Chapter 13](13-what-we-found.md) show ternary losing decisively once training runs long enough. So effective capacity by itself does not explain the behaviour, and whatever is missing lives in the interaction between precision and training duration. Pinning down that interaction is the actual scientific target of this project, and it is why the law has to have D in it rather than being a simple statement about bits.

> **In plain terms:** the arithmetic says low precision should always win, and experiment says it does not. The gap between those two is the thing worth measuring.

Suppose it works. What follows is a **design rule**: state a byte budget and an intended training duration, and the rule returns a parameter count and a bit width. That is the same shape of output Chinchilla produces for a compute budget, answering a question that binds far more often in practice.

Four things would change.

**What gets built.** Chinchilla reshaped the field within months, not because its mathematics was elegant but because it told practitioners what to train. There is currently no principled answer to "I have 2 GB, what is the best model that fits?" The standard procedure is to take an existing model, quantize it, and hope. A memory-optimal rule replaces hoping with arithmetic.

**What runs on hardware people own.** If more parameters at fewer bits genuinely wins at a phone-sized budget, then an on-device model is not a degraded copy of a real one. It is the correct design for that budget, and better than anything obtained by compressing something larger.

**The economics.** Memory is roughly 30% of hyperscaler infrastructure spending and rising ([Chapter 1](01-the-memory-wall.md)). A rule that delivers equal quality in materially fewer bytes is worth a great deal at that scale, and at the small end it is the difference between a deployment being possible and impossible.

**Who gets to participate.** The same byte scarcity that motivates the research also decides who can afford to do research at all. A method that needs fewer bytes for the same work is an answer to both problems at once.

> **Why this matters:** the negative result is valuable too, and this is not a consolation. If full precision wins per byte across the board, the honest conclusion is that native low-bit training does not pay for itself and the field should stop investing in it. That finding would save more effort than a positive one, and the design of this project — competing forms, blind extrapolation, a noise bar that refuses weak claims — is built to be equally capable of producing it.

## What to remember

The question is a constrained optimization you can state in one line: minimize loss subject to weight_bytes + kv_bytes ≤ M, choosing parameter count, weight precision, cache precision and training tokens. At its center is a trade between two opposing curves, since dropping from 16 bits to ternary buys about 10× more parameters while each retains only a fraction f(b) of a full-precision one's usefulness, and the product N × f(b) turns over somewhere. Answering it produces two usable artifacts: an iso-memory frontier showing the best achievable loss per bit width at each budget, and a b*(M, D) phase diagram whose color at each point is the precision you should build at. The genuinely novel axis is training duration, since at 3M parameters ternary beats bf16 by 0.0687 BPB at 20 tokens per parameter and loses by 0.2413 at 320×, both beyond their noise bars, so the winner depends on a regime every deployed model sits deep inside. None of this is settled here: the law is not fitted, and the honest status is a running grid and a well-posed question.
