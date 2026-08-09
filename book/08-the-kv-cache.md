# Chapter 8 — The KV Cache

*After this chapter you will know what the KV cache is, why it exists, and how to compute its size from a model's shape. You will be able to look at a deployment and say how many bytes go to weights and how many go to cache, and you will see why a memory budget that counts only weights is wrong the moment a real user types a long prompt.*

## What attention needs from the past

Only as much recap as this chapter needs. At each position a transformer runs **attention**, which lets that position look back at every earlier one and pull in information from the ones that seem relevant.

Three vectors are computed at every position by multiplying its hidden state by a learned matrix. The **query** says what this position is looking for, the **key** says what a position has to offer, and the **value** carries the content passed along if that position is selected. To attend, position 500 compares its query against the keys of positions 1 through 500, turns those comparisons into weights, and takes a weighted average of the corresponding values.

The structural fact that matters here: earlier positions contribute their keys and values and nothing else. Once a position has been processed, its key and value are the only things about it the future ever needs.

## Why we cache

Now think about generating text one token at a time. To produce token 501 the model needs keys and values for positions 1 through 500. To produce token 502 it needs the same set plus one.

Recomputing all 500 from scratch, then all 501, then all 502, is enormous waste, because those keys and values do not change. The key at position 7 was computed from the hidden state at position 7, fixed the moment token 7 was decided, and nothing later touches it.

So you keep them. The **KV cache** is exactly that: the stored keys and values for every position processed so far, held in memory for the whole generation so each new token costs one small computation instead of a full re-run of the history. It is the optimization that makes autoregressive generation practical, and it converts a compute problem into a memory problem.

> **In plain terms:** the KV cache is the model's working memory of the conversation so far. Not the text, but the internal representation of the text, pre-digested so it never has to be re-read.

## The formula, term by term

The project computes this in one line, `ModelConfig.kv_bytes` in `src/logos/config.py`:

    KV_bytes = 2 × layers × kv_heads × head_dim × context_length × (bits / 8) × batch

Each factor earns its place. The **2** is keys and values: two tensors per position, identical in shape, so everything doubles. **layers** is the number of transformer blocks, each of which attends independently with its own projections and therefore keeps its own cache. **kv_heads × head_dim** is the width of one position's key vector, since attention is split into **heads**, parallel copies that each attend over a slice of the representation.

**context_length** is how many positions are in the cache, and this factor is the one that hurts. It is exactly linear: doubling the context doubles the bytes, with no economies of scale anywhere. A model does not get more efficient at remembering as it remembers more.

**bits / 8** converts to bytes, and nothing in the formula forces the cache to have the same precision as the weights — which is the opening this chapter is about. **batch** is the number of sequences processed at once, which in a served deployment means concurrent users.

> **Why this matters:** weights are shared, so ten users hitting one model instance read the same weight bytes. Caches are not. Every user has their own conversation and therefore their own private cache, so cache memory scales with traffic while weight memory does not.

## A real shape, with real numbers

Take the largest shape in the LOGOS ladder, the 1.5B design in `config.py`: d_model 2048, 30 layers, 16 query heads, 4 key/value heads, head_dim 128. Worth stating plainly, this is a shape the project *prices*, not a model it trains — the local program fits its law between 3M and 60M parameters with paid anchors at 125M.

Start with one token of context, at 16-bit:

    2 × 30 layers × 4 kv_heads × 128 head_dim × 2 bytes = 61,440 bytes

About 61 KB per token. Fill an 8,192-token context, a modest prompt by 2026 standards:

    61,440 × 8,192 = 503,316,480 bytes ≈ 503 MB

Half a gigabyte, for one user, holding one conversation. Set that against the weights: at 1.5 billion non-embedding parameters packed at 2 bits each, the ternary body is about 377 MB, call it roughly 400 MB. **The cache is larger than the model.** Every gain from Chapters 6 and 7, the whole project of shrinking weights from 16 bits to under 2, has been more than eaten by a single 8k context nobody thought to count.

Cache precision is the obvious lever, and it works, because the formula is linear in bits:

| Cache precision | Bytes at 8,192 tokens | Relative to weights (~400 MB) |
|---|---:|---:|
| 16-bit | 503 MB | 1.26× |
| 8-bit | 252 MB | 0.63× |
| 4-bit | 126 MB | 0.31× |
| 3-bit | 94 MB | 0.24× |
| 2-bit | 63 MB | 0.16× |

Going from 16-bit to 4-bit turns the cache from the dominant term into a modest one. Whether the model still works well at 4-bit cache is a separate question, and precisely the one LOGOS intends to measure.

![KV cache size against context length for the project's 1.5B shape, drawn at 16, 8, 4, 3 and 2-bit cache precision. Every line is straight, because the cost is exactly linear in context. The 16-bit line crosses the model's own weight footprint before 8k tokens; the 4-bit line does not cross it until far beyond.](figures/kv-cache.png)

The same facts are worth seeing as shares of the whole bill:

![Where a deployed model's bytes go: the fixed weight footprint against the KV cache, which grows with context length and with concurrent users, for several weight precisions. At short contexts weights are nearly the whole bill; as context grows the cache share climbs past half and a weights-only budget stops being a reasonable approximation.](figures/bytes-budget.png)

## Grouped-query attention: compression built into the architecture

Look again at the formula and notice that it contains `kv_heads`, not `n_heads`. In the 1.5B shape those differ: 16 query heads, 4 key/value heads.

That is **grouped-query attention**, or **GQA**. Instead of every query head owning private key and value heads, several query heads share one. LOGOS uses a 4:1 ratio throughout its ladder, enforced by an assertion in `ModelConfig`, so each group of four query heads reads the same cached keys and values and attends to them with different queries.

The saving goes straight into the cache. Without GQA the 1.5B shape would need 16 key/value heads: 245,760 bytes per token instead of 61,440, and 2.01 GB instead of 503 MB at 8k context. GQA is not a storage trick. It is a decision about the model's architecture, made before training, that reduces cache bytes by construction and costs some expressiveness, because four query heads must now share one view of the past.

Which makes GQA and cache quantization two routes to the same destination. One changes what the model is, the other changes how its state is stored, and both spend the same currency. LOGOS plans to price them in that shared currency, with a GQA-ratio arm at 8:1 alongside the cache-precision arms, so the two ways of buying the same reduction can be compared directly.

## Why this changes the research question

A memory budget that counts only weights is not a conservative approximation. It is the wrong budget. The moment you commit to serving a particular context length, a fixed and often dominant slice of your bytes is spoken for before a single weight is loaded, and the slice grows with every concurrent user.

That changes the optimization. If you are choosing between a bigger model at lower weight precision and a smaller one at higher precision, the answer depends on how many bytes remain after the cache takes its cut. Push the context to 16k and the cut grows, leaving less for weights and shifting the optimum toward fewer parameters, or a cheaper cache, or both. The two decisions are coupled through a shared budget and cannot be optimized in isolation.

This is why the project's prescription code carries a `total_footprint_optimal` function that searches weight precision and cache precision jointly, subject to `weight_bytes + kv_bytes ≤ budget`. Its docstring is honest about the limitation: the fitted quality law does not yet depend on cache precision, so today the search just picks the cheapest cache that buys the most parameters. Changing that requires training runs that do not exist yet.

## What LOGOS will measure here, and what it will not

The KV phase has two halves and, as of this writing, no results. Being clear about that matters, because a plan described in the present tense reads exactly like a finding.

The cheap half is **post-hoc KV quantization**: take checkpoints that already exist at 12M, 25M and 60M parameters, round their caches to 8, 4, 3 and 2 bits at evaluation time, and measure what breaks. No training is required, which is why it covers the full sweep.

The expensive half is **native KV-QAT**, a small number of runs at 12M parameters where the cache is quantized *during* training, so the model learns with a coarse cache the way the weight arms learn with coarse weights. The machinery exists — `KVQuant` in `transformer.py` applies per-head absmax integer fake-quant to keys and values, using the same STE from [Chapter 7](07-training-in-low-precision.md) — and four arms are specified in the manifests at 8-bit and 4-bit cache crossed with ternary and bf16 weights. They have not been run.

The honest limitation is scale. Long-context retrieval benchmarks do not produce meaningful signal from a 12M-parameter model, which cannot reliably do those tasks even at full precision, so measuring cache-induced degradation on them would be measuring noise. This phase will report what it can measure: cache-induced increases in bits-per-byte, and perplexity at long context. Not task scores. Those need models an order of magnitude larger than this program trains.

## What to remember

The KV cache stores the keys and values of every processed position so that each new token costs one small computation instead of a re-run of the history, and its size is exactly 2 × layers × kv_heads × head_dim × context × bits/8 × batch. It is linear in context length, and unlike weights it is private to each concurrent user, so it grows with traffic. For the project's 1.5B shape at 8,192 tokens it comes to about 503 MB at 16-bit against roughly 400 MB of ternary weights, making the cache the larger term; dropping to 4-bit brings it to about 126 MB. Grouped-query attention at the project's 4:1 ratio already removes a factor of four by construction, which makes architecture and cache precision two ways of buying the same bytes. The practical consequence is that a weights-only budget is wrong for any real deployment, and the phase that will measure cache precision properly is specified but has produced no results yet.
