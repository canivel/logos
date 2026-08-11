# A memory budget, and the model that fits it

*This is a working notebook for a research project in progress. It explains, from the ground up, why the way we shrink language models is running out of road — and what a proper answer would look like. If you know roughly what a neural network is, you can read the whole thing.*

Every language model you can actually run is a compromise between two numbers: how good it is, and how many bytes it occupies. That second number decides whether a model runs on your phone, on your laptop, in a datacenter, or nowhere you can reach. And in 2026, bytes have become the expensive part.

The standard way the field handles this is to train a model at full precision and then squeeze it afterwards — **quantization**, applied as a finishing step. It works well enough down to about four bits per weight and then falls apart, which is unfortunate, because four bits is roughly where the savings stop being interesting.

This project takes the other road: train the model in low precision **from the beginning**, so it can adapt to the constraint while it learns. That raises a question nobody has answered, and it is the question this whole project exists to answer:

> **Given a fixed number of bytes at inference time, how should you spend them?** More parameters at fewer bits each, or fewer parameters at higher precision? How long should you train? How much of the budget should the attention cache take?

<div class="statrow">
<div class="stat"><b><!-- AUTO:run-count -->
73
<!-- /AUTO:run-count --></b><span>training runs completed so far</span></div>
<div class="stat"><b>1</b><span>consumer GPU doing the work</span></div>
<div class="stat"><b>≤$100</b><span>total cloud budget</span></div>
<div class="stat"><b>49</b><span>pre-registered verification gates</span></div>
</div>

## The first real finding

There is already an answer to part of the question, and it is not a constant. Whether low precision wins depends on **how long you train**.

![The gap between a ternary model and a full-precision one, as a function of how many tokens each parameter is trained on. Below zero means ternary is better. The bars are two standard deviations of measured run-to-run noise, so a point whose bar crosses zero is not a claim.](figures/l0-crossover.png)

Train a small model for about 20 tokens per parameter — roughly the compute-optimal point — and the **ternary** model, storing each weight as just −1, 0, or +1, is measurably *better* than the full-precision one at eight times fewer bytes. Keep training the same models to 320 tokens per parameter and the ordering flips decisively.

That matters because every model anyone actually deploys lives far out on the right of that chart. It also means the question "which precision should I use?" has no fixed answer — it has a *crossover*, and locating that crossover precisely, across model sizes and byte budgets, is what the rest of this project does.

## How to read this book

The book is built so you can start anywhere, but it is written to be read in order. Part I explains why the problem exists and gives you the vocabulary. Part II is the toolkit — scaling laws, how numbers are stored, what quantization actually does to a weight. Part III is the research itself: the question, the experimental design, the machinery, and how the numbers are kept honest. Part IV is what has been measured so far and what comes next.

<div class="cardgrid">
<div class="card"><h4><a href="01-the-memory-wall.html">Part I — Why this exists</a></h4><p>The memory crunch, what a language model is, and how we decide one model is better than another.</p></div>
<div class="card"><h4><a href="04-scaling-laws.html">Part II — The toolkit</a></h4><p>Scaling laws, bits and bytes, quantization, training in low precision, and the attention cache.</p></div>
<div class="card"><h4><a href="09-the-question.html">Part III — The research</a></h4><p>The unanswered question, how to design an experiment that can answer it, and how to trust the result.</p></div>
<div class="card"><h4><a href="13-what-we-found.html">Part IV — Results</a></h4><p>What the runs have shown so far, how the law gets fitted, and what remains.</p></div>
</div>

Every number in this book comes from a run recorded in the project's [results file](https://github.com/canivel/logos/blob/main/results/results.jsonl), and every figure regenerates from that file. Anything drawn to explain an idea rather than report a measurement is labelled *schematic* inside the figure itself. Where a result is still inside the noise, the book says so and makes no claim — that discipline is explained in [Chapter 3](03-measuring-quality.md) and is the reason to trust the rest.

> **Worth knowing:** This is a solo, self-funded project. The entire experimental grid runs on one desktop graphics card, with a hundred dollars of cloud compute reserved for a single validation step. That constraint is not an apology — it is part of the argument, and [Chapter 15](15-the-road-ahead.md) explains why.
