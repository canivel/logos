# Chapter 15 — The road ahead

*This chapter lays out what is left to do, what it costs, what would count as failure, and how to run any of it yourself. It also makes the case for why a project this size is worth doing on a desk rather than in a datacenter.*

## Where the project actually is

Two of six phases are done or nearly so. The replication tier established the noise floor and produced the crossover result. The protocol tier is mid-flight: the learning-rate probes are in, the grid extension that brackets them is queued behind the current runs, and the wider precision study at 12M parameters has begun to land.

| Phase | Question it answers | Status |
|-------|--------------------|--------|
| L0 | Does the stack reproduce known behaviour, and how noisy is a run? | Complete — 28 runs |
| L1 | How does the low-bit gap evolve with training duration? | Running |
| L2 | The law itself, across 3M–60M parameters and five precisions | Queued, auto-chained |
| L3 | Does the law predict a size it never saw? | Not started — needs L2 |
| L4 | When does the attention cache dominate the budget? | Not started |
| L5 | Does the law's prescription beat same-budget alternatives? | Not started |

Across all phases the manifests specify 125 local runs totalling about 884 GPU-hours, of which roughly 570 remain — weeks of wall-clock time on one card, running unattended. The queue drains cheapest-run-first, chains from one phase to the next without human intervention, and skips anything already recorded, so an interruption costs at most one run.

## The hundred-dollar experiment

Exactly one phase leaves the desk. L3 rents a single cloud GPU to train five models at 125M parameters — one size step above anything in the local grid — purely to test whether the fitted law predicted them correctly.

Five runs, about 33 GPU-hours, roughly $92 at on-demand pricing and less on interruptible instances. The budget ledger enforces a $100 cap as a code constant rather than a configuration value, which means exceeding it requires editing and committing source, not fat-fingering a number.

That is the entire cloud budget for the project. Everything else — every model, every ablation, every control run, the capstone — happens on hardware already sitting on a desk.

> **Why this matters:** the design forces a useful discipline. When cloud compute is effectively unlimited, the temptation is to answer every question by launching more runs. With five cloud runs total, each one has to be worth its place, which means the local grid has to be good enough that a single size step up is genuinely decisive.

## What would count as failure

A research plan that cannot fail is not a plan. Three outcomes would count, and each is worth stating in advance.

**The law does not extrapolate.** The 125M anchors come back outside the predicted band. This is the most likely failure and the least damaging — a law that holds from 3M to 60M and bends at 125M tells you something real about where the small-scale regime ends. It would be reported as the headline result rather than hidden in a limitations section.

**The two functional forms fit equally well and predict differently.** Then the honest output is both laws, the region where they disagree, and a statement that the current data cannot separate them. The fix is more runs in the overtrained tier, which is exactly where they diverge.

**Full precision wins everywhere per byte.** If the frontier turns out monotone in bit width, the entire premise is wrong. Given what BitNet, ParetoQ and Spectra have already shown — and given the crossover already measured here — this is unlikely, but it would be publishable as a negative result: a clear account of where and why low-bit training fails to pay for itself.

The one outcome that would *not* be acceptable is quietly reporting whichever framing made the numbers look best. The pre-registration protocol in [Chapter 14](14-fitting-the-law.md) and the verification panel in [Chapter 12](12-trusting-the-numbers.md) exist to make that difficult.

## The model at the end

If the law holds, the final phase spends it. The plan is to train whatever configuration the fitted law prescribes at a 64 MB body-byte budget — a size that runs comfortably on hardware people already own — and compare it against the alternatives at the same footprint: a full-precision model with proportionally fewer parameters, a larger model quantized after training to 4 bits, and a 4-bit model trained from scratch.

The 4-bit post-training baseline is the one that matters. It is the strongest thing in the field's standard toolkit and represents the "just quantize it" answer this project is arguing against. Beating it at equal bytes is the point; losing to it would be worth reporting just as clearly, with the crossover where it starts winning stated precisely.

Alongside it goes the checkpoint ladder — every model from the grid, with its configuration and results — since that ladder is arguably more useful to other researchers than any single model. It is the artifact that lets someone else fit their own law without repeating 800 GPU-hours.

## Run it yourself

The repository is designed to be re-run, not just read. Everything below works on a single consumer GPU.

```bash
git clone https://github.com/canivel/logos && cd logos
uv venv --system-site-packages && uv pip install -e ".[dev]" --no-deps

python -m pytest -q                  # the unit suite
python -m validation.panel --all     # the 49 verification gates
```

The panel is the interesting one to run first. It re-derives the quantizers, the loss metric, the packing formats, and the fitting machinery from first principles and checks the implementation against them. If it returns ACCEPT, the stack on your machine behaves the way the results in this book assume.

To reproduce a result end to end, prepare the corpus and drain a manifest:

```bash
python -m logos.cli data prepare --out data/fineweb_edu_10bt \
    --dataset-config sample-10BT --tokenizer TinyLlama/TinyLlama_v1.1 \
    --target-tokens 2500000000

python scripts/run_queue.py --manifest manifests/l0.yaml \
    --data-dir data/fineweb_edu_10bt --runs-dir runs/l0
```

That drains the 28-run replication tier — the one that produced the crossover figure — appending results to the same file format the analysis scripts read. A smaller taste, requiring no corpus download and about five minutes, is `python scripts/local_p0.py`, which trains the whole precision ladder on a public-domain novel and produces its own figure.

Every figure in this book regenerates from committed data:

```bash
python analysis/l0_summary.py        # the crossover table and figures
python analysis/lr_probe_summary.py  # the learning-rate analysis
python analysis/book_figures.py      # every figure here, both themes
```

## Why do this on a desk

There is a version of this project that runs at a real lab: models up to a billion parameters, cloud clusters, a released model people deploy. The plan for it exists in the repository as a reference design. It costs about $25,000 of compute.

This is not that. It is the same science, shifted two orders of magnitude down the ladder — fitted at 3M to 60M parameters, validated at 125M — because that is what one desktop card and a hundred dollars can pay for.

Something is genuinely lost in that shift, and the book has been explicit about it: at these sizes the embedding table distorts byte accounting, downstream benchmarks are pure noise, and extrapolating to a billion parameters would be a claim the data cannot support.

But something is gained that a well-funded version could not offer. Every result here is reproducible by anyone with a gaming GPU and a weekend. The grid is not an appendix to a paper describing runs nobody can repeat — it is the whole experiment, and it fits on a laptop's worth of disk. In a period when the compute needed to do frontier work is rationed by supply chains and capital, showing that a real scaling law can be fitted, validated against a blind test, and published from a desk is not a consolation prize. It is a second argument, running alongside the first.

The memory crunch that makes this research worth doing is the same one deciding who gets to do research at all. A method that needs fewer bytes to do the same work is an answer to both.

## What to remember

Two phases of six are done; the rest is roughly 700 hours of unattended local compute plus a single $100 cloud experiment whose only job is to test the fitted law against a size it never saw. Failure modes are named in advance — the law not extrapolating, the two forms staying indistinguishable, or full precision simply winning per byte — and each would be reported as a result rather than hidden. The final phase trains whatever the law prescribes at a 64 MB budget and puts it against the "just quantize it" baseline that represents current practice. Everything is runnable from the repository on one consumer GPU, which is not a limitation of this work so much as a second claim it is making.
