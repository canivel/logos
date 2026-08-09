# Glossary

*Every term the book defines, in one place. Chapter references point to where the idea is developed properly.*

## Models and training

**Parameter** — one adjustable number inside a model. Training means changing these. A model's size is usually quoted as its parameter count. ([Chapter 2](02-what-a-language-model-is.md))

**Non-embedding parameters (N)** — the parameters in the model's body (attention and feed-forward layers), excluding the vocabulary lookup table. This project fits its laws on N because at small scale the embedding table can outweigh everything else, which would make "parameters" mean different things at different sizes. ([Chapter 2](02-what-a-language-model-is.md))

**Embedding table** — the lookup that turns each of the 32,768 vocabulary entries into a vector. Stays 16-bit in every arm of this project, and dominates the byte count of a small artifact. ([Chapter 5](05-numbers-in-a-computer.md))

**Token** — the unit a model reads: a word, word fragment, or character, mapped to an integer. ([Chapter 2](02-what-a-language-model-is.md))

**D** — the number of tokens a model is trained on.

**Tokens per parameter (D/N)** — the "how much training" axis. About 20× is compute-optimal; deployed models are usually trained far past it, because training cost is paid once and inference cost is paid forever. This project measures 20×, 80×, and 320×. ([Chapter 2](02-what-a-language-model-is.md))

**Overtrained** — trained well past the compute-optimal point. Not a criticism; it is what everyone does, and it is the regime where this project's results change sign. ([Chapter 4](04-scaling-laws.md))

**Transformer / decoder block** — the repeated unit of the architecture: a normalization, an attention layer that mixes information across positions, another normalization, and a feed-forward layer that processes each position independently, each wrapped in a residual connection. ([Chapter 2](02-what-a-language-model-is.md))

**Attention** — the mechanism letting each position draw on earlier positions, weighted by relevance. ([Chapter 2](02-what-a-language-model-is.md), [Chapter 8](08-the-kv-cache.md))

**Grouped-query attention (GQA)** — several query heads sharing one key/value head, cutting cache size. This project uses a 4:1 ratio. ([Chapter 8](08-the-kv-cache.md))

**KV cache** — the stored keys and values from earlier positions, kept so generation does not recompute them every step. Grows linearly with context length and with concurrent users, and can exceed the model's own weights. ([Chapter 8](08-the-kv-cache.md))

**Seed** — the number initializing a run's randomness. Same configuration, different seed, different result — which is why noise must be measured. ([Chapter 3](03-measuring-quality.md))

## Measuring quality

**Cross-entropy loss** — the model's surprise at the correct next token, in nats. Lower is better. ([Chapter 3](03-measuring-quality.md))

**Bits per byte (BPB)** — total loss in bits divided by the number of UTF-8 bytes of the original text. The project's primary metric, because it is tokenizer-independent and therefore comparable across models and papers. ([Chapter 3](03-measuring-quality.md))

**Perplexity** — the exponential of the loss. Intuitive, but tokenizer-dependent, so not used here as the fitting target. ([Chapter 3](03-measuring-quality.md))

**Seed sigma (σ)** — the standard deviation of a metric across runs differing only in random seed. Measured at 0.006 to 0.066 BPB at 3M–6M parameters. ([Chapter 3](03-measuring-quality.md))

**The 2σ rule** — no quality gap is claimed unless it exceeds twice the relevant seed sigma. For two single-seed runs the bar widens by a factor of √2. Roughly a third of measured cells fall inside their own noise and are reported as "no claim". ([Chapter 3](03-measuring-quality.md), [Chapter 10](10-designing-a-clean-experiment.md))

**Held-out set** — text kept out of training, used to measure quality. This project uses two disjoint ones. ([Chapter 3](03-measuring-quality.md))

## Numbers and precision

**Bit / byte** — one binary digit; eight of them. A model's footprint is parameters × bits ÷ 8, plus overheads. ([Chapter 5](05-numbers-in-a-computer.md))

**bf16** — 16-bit "brain float", the standard training format: fp32's exponent range with fewer mantissa bits. The full-precision baseline here. ([Chapter 5](05-numbers-in-a-computer.md))

**Scale factor** — the real number a stored integer is multiplied by to reconstruct an approximate weight. The core trick of all quantization. ([Chapter 5](05-numbers-in-a-computer.md))

**Ternary / 1.58-bit** — weights restricted to −1, 0, +1 with one shared scale. Three states carry log2(3) ≈ 1.58 bits of information; practical formats pack them at 2 bits each. The project's suite is named TRIT after this. ([Chapter 5](05-numbers-in-a-computer.md), [Chapter 7](07-training-in-low-precision.md))

**Per-group quantization** — one scale per group of weights (128 here) rather than one per tensor, so a few extreme values cannot waste the whole integer range. ([Chapter 6](06-quantization.md))

**Outliers** — the small number of weights or activation channels far larger than the rest. The main technical reason naive quantization fails. ([Chapter 6](06-quantization.md))

**W×A8** — k-bit weights with 8-bit activations, the configuration used on every low-bit arm here. ([Chapter 6](06-quantization.md), [Chapter 7](07-training-in-low-precision.md))

**Packed format** — the on-disk layout of quantized weights. This project measures real packed bytes rather than deriving them, since scales and 16-bit tensors are real storage. ([Chapter 5](05-numbers-in-a-computer.md))

## Quantization approaches

**Quantization** — representing weights or activations with fewer bits. ([Chapter 6](06-quantization.md))

**Post-training quantization (PTQ)** — train at full precision, then convert. Cheap, ubiquitous, holds up to about 4 bits, degrades sharply below. The practice this project argues is a patch rather than a solution. ([Chapter 6](06-quantization.md))

**Quantization-aware training (QAT)** — train with quantization in the loop so the model adapts to it. ([Chapter 7](07-training-in-low-precision.md))

**Native low-bit / from scratch** — QAT applied from the first step rather than as a fine-tune. What this project studies. ([Chapter 7](07-training-in-low-precision.md))

**Fake quantization** — during training, keep a full-precision master weight, quantize it on each forward pass, and update the master. The deployed model keeps only the quantized values. ([Chapter 7](07-training-in-low-precision.md))

**Master weight** — the full-precision copy that accumulates updates too small to survive rounding. ([Chapter 7](07-training-in-low-precision.md))

**Straight-through estimator (STE)** — pretending, for the backward pass only, that rounding was the identity function, so gradients flow. Without it, rounding's zero derivative would stop learning entirely. ([Chapter 7](07-training-in-low-precision.md))

**Absmean scaling** — BitNet's ternary rule: divide by the mean absolute weight, round and clip to {−1, 0, +1}, multiply back. ([Chapter 7](07-training-in-low-precision.md))

**Learned scale (LSQ)** — a quantization scale trained by gradient descent rather than computed from statistics. Used on the 2/3/4-bit arms. ([Chapter 7](07-training-in-low-precision.md))

**subln** — an extra normalization before the output projections, added on quantized arms for stability. ([Chapter 7](07-training-in-low-precision.md))

## Scaling laws

**Power law** — a relationship where each doubling of the input buys a fixed improvement; a straight line on log-log axes. ([Chapter 4](04-scaling-laws.md))

**Chinchilla** — the 2022 result establishing compute-optimal training as roughly 20 tokens per parameter. Answers how to spend a *training* budget, not a memory budget. ([Chapter 4](04-scaling-laws.md))

**Compute-optimal** — the split of a fixed training budget between model size and data that minimizes loss. ([Chapter 4](04-scaling-laws.md))

**Irreducible loss (E)** — the floor no model beats, set by the inherent unpredictability of the text. ([Chapter 4](04-scaling-laws.md))

**Effective capacity f(b)** — how much of a full-precision parameter's usefulness a b-bit parameter retains, with f(16) = 1. The central unknown of the fitted law. ([Chapter 9](09-the-question.md), [Chapter 14](14-fitting-the-law.md))

**Iso-memory frontier** — for each byte budget, the best achievable loss at each bit width. The project's headline figure; where the curves cross is the practical answer. ([Chapter 9](09-the-question.md))

**b\*(M, D) phase diagram** — a map whose colour at each (budget, training-duration) point is the winning bit width. Regions of one colour are regimes; the boundaries are crossovers. ([Chapter 9](09-the-question.md))

**Crossover** — the point where the ranking of two options reverses. The project's first real result is that a ternary-versus-full-precision crossover exists along the training-duration axis. ([Chapter 13](13-what-we-found.md))

## Method and verification

**Arm** — one precision configuration in an otherwise identical experiment. ([Chapter 10](10-designing-a-clean-experiment.md))

**Manifest** — a versioned file specifying every run before it launches, so nothing is started by hand twice. ([Chapter 11](11-building-the-machine.md))

**Config hash** — a fingerprint of a run's science-bearing settings, recorded with its result so any number traces back to the configuration that produced it. Cost estimates and labels are deliberately excluded. ([Chapter 11](11-building-the-machine.md))

**Confound** — a second thing that changed alongside the one you meant to change, making the result ambiguous. The learning-rate confound is this project's worked example. ([Chapter 10](10-designing-a-clean-experiment.md))

**Control arm** — a run added specifically to rule out an alternative explanation. Twelve of them established that the crossover is a precision effect, not a learning-rate effect. ([Chapter 13](13-what-we-found.md))

**Leave-one-size-out** — fitting on all model sizes but one and measuring the error on the one held out. How competing functional forms are chosen, since in-sample fit always favours the form with more free parameters. ([Chapter 14](14-fitting-the-law.md))

**Bootstrap** — resampling runs with replacement and refitting many times to get confidence intervals on every fitted quantity. ([Chapter 14](14-fitting-the-law.md))

**Huber loss** — a fitting objective that behaves like squared error for small residuals and linear for large ones, so a single bad run cannot dominate. ([Chapter 14](14-fitting-the-law.md))

**Pre-registration** — committing predictions, with a timestamp, before running the experiment that tests them. ([Chapter 14](14-fitting-the-law.md))

**Blind extrapolation** — the credibility test: freeze the law, publish predictions, then train the validation models and compare. ([Chapter 14](14-fitting-the-law.md))

**Validation panel** — 11 independent probes with 49 pre-registered, hash-locked kill gates that re-derive every load-bearing quantity from first principles. Run with `logos-validate --all`. ([Chapter 12](12-trusting-the-numbers.md))

**Kill gate** — a pass/fail criterion declared and hashed before results are examined. Editing one after the fact voids the probe rather than passing it. ([Chapter 12](12-trusting-the-numbers.md))
