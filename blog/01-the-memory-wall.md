---
platform: Substack
status: READY TO PUBLISH
title: "Everyone is worried about GPUs. The real shortage is memory."
subtitle: "So I started training language models at 1.58 bits per weight on a gaming GPU. 73 runs in, here is what I found, and the one thing I can't claim yet."
tags: machine learning, LLMs, quantization, research
---

# Everyone is worried about GPUs. The real shortage is memory.

*So I started training language models at 1.58 bits per weight on a gaming GPU. 73 runs in, here is what I found, and the one thing I can't claim yet.*

---

![A cartoon graphics card buckling under an enormous sack of memory sticks strapped to its back, with more sticks spilling out and scattering across the floor behind it.](figures/illus-hero.png)

---

Everybody is talking about the GPU shortage. I think that is already the old story.

Look at where the money actually went in 2026. The big five are putting something like 600 to 630 billion into capex, and memory went from around 8% of that spend in 2023 and 2024 to about 30% now. Datacenters are absorbing roughly 70% of the world's memory output. Server DRAM contract prices moved 60 to 90% in a single quarter. Bit supply is growing about 16% against demand that would eat all of it.

When it normalizes depends who you ask. Intel says 2028. SK Hynix has said it could run past 2030.

![Memory as a share of hyperscaler infrastructure spending, 2023 to 2024 against 2026.](figures/memory-wall.png)

*Memory went from about 8% of hyperscaler infrastructure spend to about 30% in two years.*

So the constraint moved. It is not how many FLOPs you can buy anymore, it is how many bytes you can hold. And that changes what question you should be asking about your model.

## The standard answer is a patch

What does everyone do when a model doesn't fit? Train it in 16 bit, then quantize it after. Squeeze 16 bits down to 4, get the file 4x smaller, ship it.

It works. I use it. But I don't think it is the real answer, for three reasons.

First, it falls apart right where the savings get interesting. Post training quantization holds up fine at 8 bits, usually fine at 4, and then degrades fast below that. The regime where you actually get an 8x or 10x reduction is the regime it can't reach.

Second, it changes nothing about training. You still paid the full precision memory bill to train the thing. Compressing the artifact afterwards does not touch the economics of producing it.

Third, and this is the one I keep coming back to, it optimizes the wrong thing. Squeezing a model that was designed for 16 bits into 4 is asking "how little damage can we do here". The better question is "what is the best model that fits in this budget in the first place".

Those are not the same question and they do not have the same answer.

## So what is the actual question

[Chinchilla](https://arxiv.org/abs/2203.15556) told us how to split a compute budget between parameters and tokens. Roughly 20 tokens per parameter, and that reshaped how everyone trains.

Nobody published the deployment version of it. That one goes like this:

**Given a fixed number of bytes at inference time, how do you split them between parameter count, weight precision, KV cache precision, and training tokens, to get the best model?**

That is the question I am trying to answer. I wrote up the full framing in [chapter 9 of the book](https://canivel.github.io/logos/09-the-question.html) if you want the long version.

The intuition is simple enough. At a fixed byte budget, going from 16 bit to ternary lets you hold about 10x more parameters. If each of those parameters was as useful as a 16 bit one, low precision would always win. They are not as useful. So there is a peak somewhere, and where that peak sits is the whole thing.

---

![A balance scale weighing many small dots against a few large circles.](figures/illus-tradeoff.png)

---

## What I actually built

I don't have a cluster. I have an RTX 3080 with 10GB and about 100 dollars of cloud budget that I am saving for one specific experiment.

So the whole program is designed around that. 125 training runs, all pre specified in versioned manifests, draining on one card, chained so the GPU never sits idle between phases. Every result carries a hash of the exact config that produced it. It is all public.

- The code and every result: [github.com/canivel/logos](https://github.com/canivel/logos)
- The long explainer, 15 chapters, written for someone with basic deep learning knowledge: [canivel.github.io/logos](https://canivel.github.io/logos/)

I wrote the book part because I kept explaining the same concepts to people and figured I should just write it down properly. It starts from what a token is and goes to fitted scaling laws. If you want the machinery tour, [chapter 11](https://canivel.github.io/logos/11-building-the-machine.html) is the one.

## The finding

Here is the thing I did not expect to be so clean.

Train a small model for about 20 tokens per parameter, which is roughly the compute optimal point, and the ternary model beats the full precision one. Ternary meaning every weight is just -1, 0 or +1. 1.58 bits each.

Keep training the exact same models out to 320 tokens per parameter and the ordering flips hard.

At 3M parameters:

• 20 tokens/param: ternary is **0.069 bits per byte better** than bf16, past the 2 sigma bar of 0.041
• 80 tokens/param: +0.018, inside the noise, no claim
• 320 tokens/param: ternary is **0.241 worse**, four times the bar

So the answer to "which precision should I use" is not a constant. It is a crossover. And it matters because every model anyone actually deploys lives way out on the right side of that curve, in the overtrained regime.

I have now reproduced this at 3M, 6M and 12M parameters. At 12M and 20 tokens/param, all four quantized arms beat full precision.

![Quality gap between ternary and 16-bit weights against training tokens per parameter, at 3M, 6M and 12M parameters, with two-sigma error bars.](figures/crossover.png)

*Below zero means ternary is winning. The bars are two sigma of measured run to run noise, so anything crossing zero is not a claim.*

## The part I can't claim

This is the bit I find more interesting than the result.

Last week the 12M row at 80 tokens per parameter finished with all five precisions in it. Deficit against full precision:

• 4 bit: +0.039
• 3 bit: +0.048
• 2 bit: +0.107
• ternary: +0.148

Perfectly ordered by bit width. And look at the spacing. From 4 bit to 3 bit is 0.009. From 3 bit to 2 bit is 0.059. More than six times bigger. The arms group into two clusters instead of sliding smoothly.

![Quality lost against 16-bit for 4-bit, 3-bit, 2-bit and ternary weights at 12M parameters, with the whole effect and the measured noise bar drawn to the same scale.](figures/bit-regime.png)

*The two arrows on the right are the whole effect and my measured noise bar, at the same scale. That is the reason I am not claiming any of it.*

That shape has a name already. [ParetoQ](https://arxiv.org/abs/2502.02631) described a transition between a reconstruction regime at 3 bits and up, where the model can still approximate the weights full precision would have found, and a compensation regime at 2 bits and below where it has to find a genuinely different solution. They saw it quantizing trained models. This looks like the same boundary showing up in models trained low bit from step one.

Nice story. I am not claiming it.

Every one of those runs is a single seed. The whole spread from 4 bit to ternary is 0.109 bits per byte, and my measured noise bar at the size below is 0.131. So the 0.059 step that the entire story rests on is not even individually claimable. And a perfect ordering of four things comes up 1 time in 24 by chance, and I noticed it after looking, which is exactly the situation where people fool themselves.

I had three options. Claim it anyway. Spend about 32 GPU hours adding seeds to shore up one cell. Or write down in advance exactly what would have to be true and let runs that were already scheduled decide it.

I took the third one. [The prediction is committed with a timestamp](https://github.com/canivel/logos/blob/main/predictions/2026-08-11-bit-regime-step.md), three specific falsifiable calls for the 25M and 60M rows, written before either started training. Six binary outcomes. Five or six hits and it is worth spending seeds to nail down. Two or three and it is dead.

Only that option costs nothing and can actually be wrong.

## Why I keep going on about noise

If there is one thing I would want someone to take from this whole project it is this.

Train the same config twice with a different random seed and you get a different number. At the sizes I am working at, that spread is between 0.006 and 0.066 bits per byte depending on the cell. A lot of quantization results I read online are smaller than that.

So the rule in this project is that no quality gap gets claimed unless it clears two sigma for its size class. About a third of my measured comparisons come back as "inside the noise, no claim". That feels bad to write and it is the correct thing to write.

I also had to retract one of my own claims already. I had written that the low bit deficit grows with model size, which was true of the two sizes I had at the time. The third size came in and did not continue the trend. I rewrote the section and left a note explaining that the original claim was extrapolated from two points, which is one point too few to see a shape. [Chapter 3](https://canivel.github.io/logos/03-measuring-quality.html) is the one on measurement if you want the method.

## Also, low bit training is easier to tune than I expected

Quick one that surprised me. I ran 15 learning rate probes across all five precisions.

Over a 4x range of learning rates, full precision moved 0.191 bits per byte. Ternary and 2 bit moved 0.034. So quantized training is 5 to 6 times less sensitive to getting the learning rate wrong.

![Validation bits per byte against learning-rate multiplier for five weight precisions at 6M parameters.](figures/lr-sensitivity.png)

*bf16 is the steep one. The quantized arms are nearly flat across the same range.*

Makes sense once you think about what a quantizer is. It is a projection. A bigger optimizer step moves the master weight further, but unless it crosses a rounding boundary the effective weight the model actually uses does not change at all. The quantizer absorbs most of your mistake.

People describe native low bit training as delicate. On my numbers it is more forgiving than bf16. You don't have to find the learning rate, you have to avoid the cliff, and the cliff is further away. [Chapter 7](https://canivel.github.io/logos/07-training-in-low-precision.html) covers the mechanics of how any of this trains at all, since rounding has a zero gradient and by rights should stop learning dead.

---

![One lit desk in the foreground, distant datacenters on the horizon.](figures/illus-desk.png)

---

## What's next

The grid is still running as I write this. Next up is the extension of the learning rate probes, then the main grid at 25M and 60M, which is where the law actually gets fitted and where my prediction gets scored.

After that there is one cloud experiment. I freeze the fitted law, commit the predicted numbers publicly with a timestamp, and only then rent a GPU to train the 125M models and see if the prediction held. About 92 dollars. That is the whole cloud budget for the project.

If it misses, that is a result too and I will write it up the same way. A law that holds from 3M to 60M and bends at 125M tells you something real about where the small scale regime ends.

The roadmap is in [chapter 15](https://canivel.github.io/logos/15-the-road-ahead.html), including how to run any of it yourself. Everything works on one consumer card. That is not a limitation I am apologizing for, it is half the point. The same byte scarcity that makes this research worth doing is also deciding who gets to do research at all, and a method that needs fewer bytes to do the same work is an answer to both.

---

If you want the whole thing properly, start here: **[canivel.github.io/logos](https://canivel.github.io/logos/)**

Code, data, every result: **[github.com/canivel/logos](https://github.com/canivel/logos)**

I will post again when the law is fitted and the prediction gets scored, whichever way it goes.
