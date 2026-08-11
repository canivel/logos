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

## Posting checklist

1. Re-run `python analysis/book_sync.py --check`. If numbers drifted since the
   draft was written, fix the post before publishing, not after.
2. Generate the images from the prompts in the draft and insert them where the
   `[IMAGE n]` blocks sit. Delete the blocks.
3. Substack strips YAML front matter — copy from the `# heading` down. Put the
   subtitle line in Substack's own subtitle field rather than the body.
4. Check every link resolves, especially the chapter links, since the book
   filenames are generated.
