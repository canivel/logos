# The LOGOS book — source

*A memory budget, and the model that fits it* — the project's long-form
explainer. Published at **https://canivel.github.io/logos/**.

The Markdown files here are the source of truth and are readable directly on
GitHub. `build.py` renders them into the static site GitHub Pages serves from
[`docs/`](../docs/).

```bash
python analysis/book_figures.py   # regenerate every figure (light + dark)
python book/build.py              # markdown -> docs/*.html
```

## Layout

| File | What |
|------|------|
| `index.md` | Landing page: the argument in brief, the first result, how to read |
| `01`–`03` | Part I — why the project exists, what a language model is, how quality is measured |
| `04`–`08` | Part II — scaling laws, number formats, quantization, low-precision training, the KV cache |
| `09`–`12` | Part III — the research question, experimental design, the machinery, the verification layer |
| `13`–`15` | Part IV — results to date, fitting the law, the road ahead |
| `glossary.md` | Every bolded term, with the chapter that develops it |
| `build.py` | The site generator (mistune + jinja2 + pygments) |
| `book.css` | All styling; light and dark are both selected, not inverted |

## Conventions the builder depends on

- **Chapter registry** — `CHAPTERS` in `build.py` drives the nav, the ordering,
  and the prev/next pager. A new chapter needs an entry there and a matching
  `NN-slug.md`.
- **Figures** — `![caption](figures/name.png)` becomes a `<picture>` that swaps
  to `figures/name-dark.png` in dark mode. Both variants come from
  `analysis/book_figures.py`; never hand-edit files in `docs/figures/`.
- **Callouts** — a blockquote opening with `**In plain terms:**`,
  `**Why this matters:**`, `**Worth knowing:**`, or `**Heads up:**` renders as a
  styled aside.
- **Cross-references** — link chapters by their Markdown filename
  (`[Chapter 7](07-training-in-low-precision.md)`); the builder rewrites the
  extension so links work both on GitHub and on the site.
- **A trailing `*Figure N — ...*` paragraph** after an image is folded into that
  figure's caption rather than rendered twice.

## House rules for the prose

Every number must come from a run in `results/results.jsonl` or from code in
the repository. Anything drawn to explain an idea rather than report a
measurement is marked *schematic* inside the figure itself. Results that do not
clear the 2σ noise bar are reported as "no claim" — never as a finding. Nothing
claims a fitted law, a validated extrapolation, or a released model until those
exist.
