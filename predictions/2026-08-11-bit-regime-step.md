# Pre-registered prediction — a regime step between 2 and 3 bits

**Registered:** 2026-08-11, before any 25M or 60M grid run had started.
**Registered against:** results.jsonl at 73 rows; L2 not begun.
**Status:** OPEN

---

## What was observed

The 12M cell at 80 tokens per parameter completed with all five precisions.
Deficit against full precision (bf16 = 1.1766 BPB):

| Arm | Val BPB | Deficit | Step from the arm above |
|-----|--------:|--------:|------------------------:|
| 4-bit | 1.2157 | +0.0391 | — |
| 3-bit | 1.2242 | +0.0476 | +0.009 |
| 2-bit | 1.2832 | +0.1066 | **+0.059** |
| ternary | 1.3242 | +0.1476 | +0.041 |

Two features, neither claimable from single-seed runs:

1. The ordering is **perfectly monotone in bit width**.
2. The spacing is **not uniform**. The 4-to-3-bit step is 0.009; the
   3-to-2-bit step is 0.059, roughly six times larger. The arms fall into two
   groups — {4, 3} at a mean deficit of 0.043 and {2, 1.58} at 0.127 — with a
   gap of 0.084 between the groups that is larger than the spread within
   either.

This is the shape ParetoQ described as a transition between a *reconstruction*
regime at 3 bits and above, where a quantized model can still approximate the
weights a full-precision model would have learned, and a *compensation* regime
at 2 bits and below, where it must find a different solution entirely. ParetoQ
observed it for quantization applied to trained models. This would be the same
boundary appearing in models trained low-bit from the first step.

## Why this is not yet a result

Every arm here is a single seed. The whole 4-bit-to-ternary spread is 0.109
BPB, against a 2σ noise bar of 0.131 measured at 6M — so by the project's own
rule ([Chapter 3](../book/03-measuring-quality.md)) not one of these gaps is
individually claimable, including the 0.059 step the whole story rests on.

A perfectly monotone ordering of four arms has a 1-in-24 chance of arising from
noise alone, which is suggestive but was noticed after the fact, and post-hoc
patterns are exactly what pre-registration exists to discipline.

## The prediction

Adding seeds at 12M would cost roughly 32 GPU-hours and would only strengthen
one cell. The L2 grid already trains this same row at **25M and 60M**, so the
cheaper and much stronger test is whether the pattern repeats at sizes it was
not derived from.

**If the regime step is real, then at 25M and 60M at 80 tokens per parameter:**

- **P1.** The deficit ordering will be monotone in bit width:
  4-bit ≤ 3-bit ≤ 2-bit ≤ ternary.
- **P2.** The 3-bit-to-2-bit step will be the largest of the three adjacent
  steps at both sizes.
- **P3.** The 3-bit-to-2-bit step will exceed the 4-bit-to-3-bit step by at
  least a factor of two at both sizes.

**If the pattern is noise**, P1 fails at one or both sizes (any inversion), or
the largest step lands somewhere other than 3-to-2.

## How it gets scored

When the 25M and 60M rows at 80× complete, each of P1, P2 and P3 is marked
hit or miss per size, and the outcome is written into the Status line above
and into the book — whichever way it goes. Three predictions across two sizes
is six binary calls; five or six hits would make the regime step worth
claiming and worth spending seeds to nail down, two or three would kill it.

A miss is a real outcome and gets reported as one. The pattern is currently
just a shape in single-seed data that happens to match a published result,
which is precisely the kind of thing that feels convincing and often is not.
