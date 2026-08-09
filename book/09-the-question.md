# Chapter 9 — The Question

*After this chapter you will be able to state the question LOGOS exists to answer, in one sentence, precisely. You will understand the trade at its center: at a fixed byte budget, fewer bits per parameter buys more parameters, but each one is worth less, and somewhere between those two curves is an optimum. You will know what an iso-memory frontier and a phase diagram are and what each would tell a practitioner. And you will see the real evidence collected so far, including the finding that makes the question genuinely open — the best precision appears to depend on how long you train.*

## A budget you can hold in your head

Forget scaling laws for a moment and imagine a concrete situation.

You have a device with 2 GB of memory available for a language model. A midrange phone, a laptop running a model locally, a slice of a shared GPU. The number is fixed and it is not negotiable, because the memory is physically not there.

Now spend it. You could load a 1-billion-parameter model at 16 bits per weight, which fills the budget exactly. You could load a 4-billion-parameter model at 4 bits. You could go to roughly 8 billion parameters at 2 bits, or about 10 billion at the theoretical 1.58 bits of a ternary model. Same 2 GB every time. Wildly different models.

And that is only one of the choices. You also have to decide how long to train the thing, which affects quality independently of its size. From [Chapter 8](08-the-kv-cache.md) you know you must also decide how precisely to store the KV cache, because if you intend to serve 8,192 tokens of context then a sizeable slice of that 2 GB is going to the cache no matter what you do with the weights.

Four dials. One budget. Which setting gives the best model?

Nobody has published the answer. That is the whole of it.

## The question, stated formally

Here is the same thing in the language a paper would use.

Maximize model quality subject to weight_bytes + kv_bytes ≤ M, choosing the number of parameters N, the weight precision b_w, the cache precision b_kv, and the number of training tokens D.

Quality is measured as loss, specifically **bits-per-byte** (BPB): how many bits the model needs, on average, to encode one byte of held-out text. Lower is better, and using bits-per-byte rather than per-token perplexity means the number is comparable across models that might tokenize differently. Minimizing loss is the same thing as maximizing quality, so the objective flips to a minimization.

Compare it to the law it is modeled on. Chinchilla asked how to split a **compute** budget between parameters and tokens, and its answer, that most models of the era were badly undertrained, reorganized the field. LOGOS asks the deployment version of the same question, with bytes as the scarce resource instead of FLOPs, and with two extra dials that do not exist in the compute framing: the precision of the weights and the precision of the cache.

The distinction that makes it a research question rather than an engineering exercise is that these are *natively low-bit* models, trained from scratch at their target precision using the QAT machinery of [Chapter 7](07-training-in-low-precision.md). Not 16-bit models compressed afterward. That difference is exactly what [Chapter 6](06-quantization.md) argued matters most below 4 bits.

## The trade at the center

Everything turns on one tension, and it is simple enough to work out with arithmetic.

**Fewer bits buy more parameters.** Weight bytes are N × b / 8, so at a fixed budget the affordable N is inversely proportional to b. Going from 16 bits to 1.58 gives you 16 / 1.58 ≈ 10.1 times more parameters. Going to 2 bits gives you 8 times more. This part is exact and requires no experiment.

**But a low-bit parameter is worth less than a full-precision one.** A ternary weight carries three possible values. A bf16 weight carries tens of thousands of distinguishable ones. If they were equally useful, low precision would win unconditionally at every budget and there would be nothing to study.

### Effective capacity

The way to make that vague "worth less" precise is a single function. Define the **effective capacity** f(b): the fraction of a full-precision parameter's usefulness that a b-bit parameter retains. By construction f(16) = 1, since bf16 is the reference. For any lower precision f(b) is somewhere below 1, and how far below is an empirical question that no amount of theory settles.

The point of the definition is what it lets you write. A model with N parameters at b bits behaves, to a first approximation, like a full-precision model with N × f(b) parameters. That product is what the fitted law uses in place of N, and it is what decides which configuration wins:

- N scales as 1/b. Halve the bits, double the parameters.
- Usefulness per parameter scales as f(b), which falls as b falls.
- Quality follows the product, N × f(b).

Two curves pulling in opposite directions, and the product has a maximum. Where that maximum sits is what the project is measuring. In the code this is not a metaphor: `FormAFree` in `src/logos/fitting/forms.py` fits one free f value per bit width, and `FormAParam` fits a smooth parametric curve through them, and the two compete to see which extrapolates better.

### A worked example, illustrative only

Suppose the measured effective capacities came out as in this table. **These numbers are invented for the sake of the arithmetic. They are not measurements, and the real ones do not exist yet.**

| Precision | Params affordable in 2 GB | Illustrative f(b) | N × f(b) |
|---|---:|---:|---:|
| bf16 | 1.0B | 1.00 | 1.00B |
| 4-bit | 4.0B | 0.55 | 2.20B |
| 3-bit | 5.3B | 0.42 | 2.24B |
| 2-bit | 8.0B | 0.28 | 2.24B |
| ternary | 10.1B | 0.20 | 2.02B |

In this made-up world the low-bit arms all beat full precision by roughly a factor of two in effective parameters, and the optimum is a broad plateau between 3-bit and 2-bit, with ternary just past the turn. Notice how the effective column rises steeply from bf16, then flattens, then dips. That shape is the point. The parameter gain from dropping bits is steady and multiplicative, while the capacity loss accelerates as the levels run out, so the product turns over somewhere.

Change f(1.58) from 0.20 to 0.35 in that table and ternary wins outright at 3.5B effective. Change it to 0.08 and ternary drops below 4-bit and nearly below bf16. The entire prescription hinges on the shape of one function that has to be measured.

![Iso-memory trade between parameter count and bits per parameter](figures/iso-memory.png)

*Figure 9.1 — Illustrative. At a fixed byte budget, moving left along the horizontal axis reduces bits per parameter and raises the affordable parameter count along the hyperbola N = 8M / b. The effective-capacity curve f(b) falls in the same direction. Their product, plotted on top, rises, peaks, and falls, which is why an optimum exists rather than a monotone preference for the lowest precision available. The location of the peak in this figure is chosen for illustration; the real one is what the project's grid is being run to find.*

## How much can one parameter possibly hold?

There is a useful external reference point for reading f(b). A line of work on knowledge capacity has found that a transformer parameter stores roughly 2 bits of factual knowledge, regardless of whether that parameter is stored in 16 bits or fewer. The finding is striking because it says most of a bf16 weight's storage is not carrying knowledge, which is a theoretical reason to expect low-bit models to lose less than their bit count suggests.

It also gives a hard ceiling worth stating. A b-bit weight cannot store more than b raw bits, so for the low rungs of the ladder the real bound is min(b, 2): a 4-bit weight is capped at 2 bits by the knowledge result, while a ternary weight is capped at about 1.58 by physics.

The project's prescription code makes this comparison explicit. `effective_capacity_summary` in `src/logos/fitting/prescribe.py` reports, per bit width, the fitted f, the implied knowledge bits (2 × f, on the premise that a bf16 parameter holds about 2), the applicable bound, and the fraction of the bound achieved. That last column is the interesting one, because it asks a sharper question than "how good is low precision." It asks how much of the storage a low-bit weight *physically has* the training procedure manages to use. A ternary arm at 90% of its bound would say the quantizer is close to optimal and further gains must come from elsewhere. At 40% it would say there is a lot left on the table.

That column is empty today. The fit has not been run on real data.

## Two deliverables

The fitted law is not the product. It is the input to two artifacts a practitioner can actually use.

### The iso-memory frontier

**Iso-memory** means "equal memory," the same way an isobar on a weather map connects points of equal pressure. The **iso-memory frontier** answers: for each byte budget, what is the best loss achievable at each bit width, using the largest model that fits at that width?

Read as a chart, it is a family of curves, one per precision, with budget on the horizontal axis and loss on the vertical. Where the curves cross is where the recommendation changes. Below the crossing, one precision is the right answer; above it, another. The project plans these over body-byte budgets from 8 MB to 256 MB, which are the sizes its own artifacts actually occupy.

### The b*(M, D) phase diagram

A **phase diagram** is a picture borrowed from physics, where it shows which state of matter is stable at each combination of temperature and pressure: regions of solid, liquid, gas, with boundaries between them. No curve is plotted. The chart is a map, and its content is which region you are in.

The b*(M, D) diagram is the same idea with different axes. Byte budget M along one axis, training tokens D along the other, and the color at each point is the *optimal weight precision* there. Regions where ternary wins, regions where 4-bit wins, regions where full precision wins, and the boundaries between them.

That is the deliverable a practitioner would actually use. You know your memory budget, because your hardware fixes it. You know roughly how much data you can train on. You look up the point, read the color, and it tells you what precision to build at. `phase_diagram` in `prescribe.py` computes exactly this array, by solving the constrained optimization at every grid point.

**No such diagram has been produced from real data in this project.** The versions in the repository today are demonstrations on synthetic ground truth, used to verify that the fitting machinery recovers parameters it is known to have been given. The grid that would produce the real one is still running.

## The twist: the answer moves with training duration

Here is the axis that makes this a live question rather than a tidy measurement exercise.

If effective capacity were a fixed property of a bit width, the phase diagram would have vertical boundaries and D would not matter. The evidence says otherwise. The winning precision appears to depend on the ratio D/N, tokens per parameter, which is to say on how heavily the model has been trained relative to its size.

### What has actually been measured

Two results are in hand. Both are small, both are real, and both come with significance bars from measured seed-to-seed noise rather than from eyeballing.

At **3M parameters**, the ternary-minus-bf16 gap in bits-per-byte, across three training durations:

| Tokens/param | Gap (ternary − bf16) | 2σ bar | Verdict |
|---|---:|---:|---|
| 20× | −0.0687 | 0.0409 | ternary better, significant |
| 80× | +0.0178 | 0.1008 | inside noise, no claim |
| 320× | +0.2413 | 0.0611 | bf16 better, significant |

Negative means ternary wins. At 20 tokens per parameter, ternary beats full precision by 0.0687 BPB, comfortably past the 0.0409 two-sigma bar. At 80× the difference is inside the noise and nothing is claimed. At 320×, ternary *loses* by 0.2413 BPB against a 0.0611 bar, and the deficit shows no sign of saturating.

The sign flips. Same architecture, same data, same data order, same everything except how long it trained.

At **12M parameters** and 20 tokens per parameter, the comparison widens to the whole ladder:

| Arm | Val BPB | vs bf16 |
|---|---:|---:|
| 2-bit | 1.4945 | −0.104 |
| 4-bit | 1.4955 | −0.103 |
| 3-bit | 1.5153 | −0.083 |
| ternary | 1.5201 | −0.078 |
| bf16 | 1.5982 | — |

Every quantized arm beats full precision. Note carefully what is *not* being claimed: the ordering among the four quantized arms sits inside single-seed noise, so the fact that 2-bit heads the table is not evidence that 2-bit is best. Only the quantized-versus-bf16 separation is being read.

### What it implies

The answer to "which precision should I use?" is not a constant. It is a function of the training regime, and the direction of the effect is uncomfortable, because the regime where low precision loses is the regime everything real lives in.

Chinchilla-optimal is around 20 tokens per parameter. Deployed models are trained far past that, often ten or twenty times past it, because inference cost scales with model size while training cost is paid once. Every model you have used is heavily overtrained by the Chinchilla standard. That is precisely where the 3M evidence says ternary's advantage has not merely evaporated but inverted.

So a phase diagram with a D axis is not decoration. If the boundaries move with training duration, then any recommendation stated without reference to D is incomplete, and a result demonstrated at 20 tokens per parameter says little about a model trained at 320×. Mapping that dependence quantitatively, rather than showing it exists at one size, is the contribution.

## The white space

It is fair to ask why nobody has done this, given how many people work on quantization. The honest answer is that several groups have done adjacent things well, and the gap between them is narrow and specific.

Precision scaling laws exist from the **compute** side, relating precision to training cost and predicting degradation, but they treat memory as an output rather than the binding constraint. **ParetoQ** built the 1-to-4-bit ladder carefully and established where the regimes change, optimizing accuracy at a fixed model size rather than under a byte budget. **Spectra** trained and released a suite of ternary models, demonstrating that the artifacts work at scale, without fitting a law that says how to allocate a memory budget. **BitNet** proved that natively-trained low-bit models can match full-precision ones at some sizes, which was the necessary existence proof, and stopped there.

Each of those is a piece. Nobody has put total inference footprint, weights plus cache, on the left side of the constraint and asked what allocation maximizes quality for natively-trained low-bit models. That is the sentence LOGOS is trying to be able to write.

And it cannot write it yet. The law is not fitted. There is no released model, no completed cloud anchor, no measured f(b), no real phase diagram. What exists is a validated stack, a frozen protocol, a replication at micro scale, a crossover measured with a proper noise floor at 3M and 6M, a first full-ladder cell at 12M, a learning-rate sensitivity finding with a known gap in its probe grid, and a grid that is still running. The question in this chapter's title is a question. [Chapter 10](10-designing-a-clean-experiment.md) takes up what it would take to turn it into an answer, and how the project is arranged so that a wrong answer would be visible.

## What to remember

The question is a constrained optimization you can state in one line: minimize loss subject to weight_bytes + kv_bytes ≤ M, choosing parameter count, weight precision, cache precision, and training tokens. Its center is a trade with two opposing curves, since dropping from 16 bits to ternary buys about 10× more parameters while each parameter retains only a fraction f(b) of a full-precision one's usefulness, and the product N × f(b) turns over somewhere, which is why an optimum exists instead of a blanket preference for the fewest bits. Answering it produces two usable artifacts, an iso-memory frontier showing the best achievable loss per bit width at each budget, and a b*(M, D) phase diagram whose color at each point is the precision you should build at. The genuinely novel axis is training duration: at 3M parameters ternary beats bf16 by 0.0687 BPB at 20 tokens per parameter and loses by 0.2413 at 320×, both beyond their noise bars, so the winner depends on a regime that every deployed model sits deep inside. None of this is settled here, the law is not fitted, and the honest status of the project is a running grid and a well-posed question.
