# Blog drafts

One post per phase, per the plan's blog cadence (v0.2 §14). Drafts live here
so the claims in a post can be diffed against the results file that produced
them.

| Post | Phase | Status |
|------|-------|--------|
| `01-the-memory-wall.md` | L0 / early L1 | draft, ready to paste into Substack |

## House rules, same as the book

Every number traces to a row in `results/results.jsonl`. Nothing gets stated
as a finding that has not cleared the 2σ bar. Where something is suggestive
but inside the noise, the post says so rather than rounding it up into a
claim — that is the whole reason anyone should trust the next post.

## Figures

Two kinds, and the draft marks which is which:

- **`[CHART — upload ...]`** points at a real file in `blog/figures/`,
  generated from `results/results.jsonl` by `analysis/blog_figures.py`.
  Regenerate before posting so the chart matches the prose.
- **`[IMAGE n]`** is a decorative illustration with a generation prompt
  attached. These carry no data and are safe to swap for anything.

Blog charts are built separately from the book's because they are read at
about 700px, often on a phone, with no surrounding chapter to explain them,
and Substack renders on a light background whatever the reader's theme. So
they use larger type, fewer elements, the key number written on the chart,
and light mode only.

```bash
python analysis/blog_figures.py            # all
python analysis/blog_figures.py --only crossover,bit-regime
```

## Posting checklist

1. Re-run `python analysis/book_sync.py --check`. If numbers drifted since the
   draft was written, fix the post before publishing, not after.
2. Re-run `python analysis/blog_figures.py` so the charts match the numbers in
   the prose, then upload them where the `[CHART]` blocks sit.
3. Generate the decorative images from the prompts and insert them where the
   `[IMAGE n]` blocks sit. Delete the blocks.
4. Substack strips YAML front matter — copy from the `# heading` down. Put the
   subtitle line in Substack's own subtitle field rather than the body.
5. Check every link resolves, especially the chapter links, since the book
   filenames are generated.
