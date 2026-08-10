"""Build the LOGOS book: Markdown in book/ -> static site in docs/.

The Markdown files are the source of truth and stay readable on GitHub;
this renders them into a self-contained site GitHub Pages serves from
/docs (a .nojekyll file keeps Pages from reprocessing them).

    python book/build.py

What it adds on top of plain Markdown:
  * chapter nav (sidebar + prev/next), built from CHAPTERS below
  * theme-aware figures: ![cap](figures/x.png) becomes a <picture> that
    swaps to figures/x-dark.png in dark mode
  * styled callouts from "> **In plain terms:** ..." blockquotes
  * heading anchors, a per-chapter table of contents, reading time
  * syntax highlighting, and light/dark that follows the OS with a toggle
"""

from __future__ import annotations

import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import mistune
from jinja2 import Template
from mistune.directives import Admonition
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "book"
OUT = ROOT / "docs"
SITE_TITLE = "LOGOS"
SITE_SUB = "Memory-optimal scaling laws for natively low-bit language models"
REPO = "https://github.com/canivel/logos"


@dataclass
class Chapter:
    slug: str          # file stem, e.g. "01-the-memory-wall"
    title: str         # nav title
    part: str          # section grouping in the nav


CHAPTERS = [
    Chapter("index", "Start here", ""),
    Chapter("01-the-memory-wall", "1 · The memory wall", "Part I — Why this exists"),
    Chapter("02-what-a-language-model-is", "2 · What a language model is", "Part I — Why this exists"),
    Chapter("03-measuring-quality", "3 · Measuring quality", "Part I — Why this exists"),
    Chapter("04-scaling-laws", "4 · Scaling laws", "Part II — The toolkit"),
    Chapter("05-numbers-in-a-computer", "5 · Numbers in a computer", "Part II — The toolkit"),
    Chapter("06-quantization", "6 · Quantization", "Part II — The toolkit"),
    Chapter("07-training-in-low-precision", "7 · Training in low precision", "Part II — The toolkit"),
    Chapter("08-the-kv-cache", "8 · The KV cache", "Part II — The toolkit"),
    Chapter("09-the-question", "9 · The question", "Part III — The research"),
    Chapter("10-designing-a-clean-experiment", "10 · Designing a clean experiment", "Part III — The research"),
    Chapter("11-building-the-machine", "11 · Building the machine", "Part III — The research"),
    Chapter("12-trusting-the-numbers", "12 · Trusting the numbers", "Part III — The research"),
    Chapter("13-what-we-found", "13 · What we found so far", "Part IV — Results & what's next"),
    Chapter("14-fitting-the-law", "14 · Fitting the law", "Part IV — Results & what's next"),
    Chapter("15-the-road-ahead", "15 · The road ahead", "Part IV — Results & what's next"),
    Chapter("glossary", "Glossary", "Reference"),
]

CALLOUTS = {
    "In plain terms": "plain",
    "Why this matters": "matters",
    "Worth knowing": "aside",
    "Heads up": "warn",
}


class BookRenderer(mistune.HTMLRenderer):
    """Adds heading anchors, theme-aware figures, and highlighted code."""

    def __init__(self):
        super().__init__(escape=False)
        self.headings: list[tuple[int, str, str]] = []

    def heading(self, text, level, **attrs):
        slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")
        if level in (2, 3):
            self.headings.append((level, text, slug))
        anchor = f'<a class="anchor" href="#{slug}" aria-label="Link to this section">#</a>'
        return f'<h{level} id="{slug}">{text}{anchor}</h{level}>\n'

    def image(self, alt, url, title=None):
        if url.startswith("figures/") and url.endswith(".png"):
            dark = url[:-4] + "-dark.png"
            return (
                f'<figure class="fig">'
                f'<picture>'
                f'<source srcset="{dark}" media="(prefers-color-scheme: dark)" '
                f'class="dark-src">'
                f'<img src="{url}" alt="{html.escape(alt)}" loading="lazy" '
                f'data-light="{url}" data-dark="{dark}">'
                f'</picture>'
                f'<figcaption>{alt}</figcaption></figure>'
            )
        return super().image(alt, url, title)

    def block_code(self, code, info=None):
        lang = (info or "").split()[0] if info else ""
        if lang:
            try:
                lexer = get_lexer_by_name(lang, stripall=False)
                return highlight(code, lexer, HtmlFormatter(nowrap=False, cssclass="hl"))
            except Exception:
                pass
        return f"<pre><code>{html.escape(code)}</code></pre>\n"

    def link(self, text, url, title=None):
        # Rewrite intra-book markdown links to their built pages.
        m = re.match(r"^(\d\d-[a-z0-9-]+|glossary|index)\.md(#.*)?$", url)
        if m:
            url = ("index.html" if m.group(1) == "index" else f"{m.group(1)}.html") + (m.group(2) or "")
        return super().link(text, url, title)


CALLOUT_RE = re.compile(
    r"<blockquote>\s*<p><strong>(" + "|".join(CALLOUTS) + r"):</strong>(.*?)</p>\s*</blockquote>",
    re.S,
)


# A lone image on its own line becomes <p><figure>...</figure></p>, which is
# invalid HTML (browsers auto-close the <p> and orphan the figure). Unwrap it.
FIG_IN_P_RE = re.compile(r"<p>\s*(<figure class=\"fig\">.*?</figure>)\s*</p>", re.S)

# A chapter may follow an image with its own "*Figure 7.1 — ...*" paragraph.
# Fold that into the figure's caption instead of rendering two captions.
MANUAL_CAPTION_RE = re.compile(
    r"(<figure class=\"fig\">.*?<figcaption>)(.*?)(</figcaption></figure>)\s*"
    r"<p><em>(Figure\s.*?)</em></p>",
    re.S,
)
# "Figure `lr-sensitivity`: " / "Figure 7.1 — " prefixes are bookkeeping for
# the author, not something a reader needs in the caption.
CAPTION_PREFIX_RE = re.compile(r"^Figure\s*(<code>[^<]*</code>|[\d.]+)?\s*[:—-]\s*")


# analysis/book_sync.py delimits generated blocks with these; they are
# bookkeeping for the sync tool, not something to ship in the page source.
AUTO_MARKER_RE = re.compile(r"[ \t]*<!-- /?AUTO:[a-z0-9-]+ -->\n?")


def fold_manual_captions(html_text: str) -> str:
    html_text = AUTO_MARKER_RE.sub("", html_text)
    html_text = FIG_IN_P_RE.sub(r"\1", html_text)

    def sub(m):
        caption = CAPTION_PREFIX_RE.sub("", m.group(4).strip())
        caption = caption[:1].upper() + caption[1:]
        return f"{m.group(1)}{caption}{m.group(3)}"
    return MANUAL_CAPTION_RE.sub(sub, html_text)


def apply_callouts(html_text: str) -> str:
    def sub(m):
        kind = CALLOUTS[m.group(1)]
        return (
            f'<aside class="callout callout-{kind}">'
            f'<span class="callout-label">{m.group(1)}</span>'
            f'<div class="callout-body"><p>{m.group(2).strip()}</p></div></aside>'
        )
    return CALLOUT_RE.sub(sub, html_text)


PAGE = Template("""<!doctype html>
<html lang="en" data-theme-pref="system">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ page_title }}</title>
<meta name="description" content="{{ description }}">
<meta property="og:title" content="{{ page_title }}">
<meta property="og:description" content="{{ description }}">
<meta property="og:type" content="article">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>▚</text></svg>">
<link rel="stylesheet" href="assets/book.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <a class="brand" href="index.html"><span class="brand-mark">▚</span> {{ site_title }}</a>
  <div class="topbar-right">
    <a class="topbar-link" href="{{ repo }}">Repository</a>
    <button class="theme-toggle" type="button" aria-label="Switch colour theme">◐</button>
    <button class="nav-toggle" type="button" aria-label="Open chapter list">☰</button>
  </div>
</header>
<div class="shell">
  <nav class="sidebar" aria-label="Chapters">
    {% for part, items in nav %}
      {% if part %}<div class="nav-part">{{ part }}</div>{% endif %}
      <ul class="nav-list">
      {% for it in items %}
        <li><a href="{{ it.href }}" {% if it.current %}class="current" aria-current="page"{% endif %}>{{ it.title }}</a></li>
      {% endfor %}
      </ul>
    {% endfor %}
  </nav>
  <main id="main">
    {% if toc %}
    <details class="toc">
      <summary>On this page</summary>
      <ul>{% for level, text, slug in toc %}<li class="toc-l{{ level }}"><a href="#{{ slug }}">{{ text }}</a></li>{% endfor %}</ul>
    </details>
    {% endif %}
    <article class="prose">
      {{ content }}
    </article>
    <nav class="pager">
      {% if prev %}<a class="pager-prev" href="{{ prev.href }}"><span>Previous</span>{{ prev.title }}</a>{% else %}<span></span>{% endif %}
      {% if next %}<a class="pager-next" href="{{ next.href }}"><span>Next</span>{{ next.title }}</a>{% endif %}
    </nav>
    <footer class="foot">
      <p>{{ site_sub }} · <a href="{{ repo }}">source and data on GitHub</a> · Apache-2.0</p>
    </footer>
  </main>
</div>
<script>
(function () {
  var root = document.documentElement;
  var saved = localStorage.getItem('logos-theme');
  if (saved) root.setAttribute('data-theme', saved);
  function currentIsDark() {
    var attr = root.getAttribute('data-theme');
    if (attr) return attr === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function syncFigures() {
    var explicit = root.getAttribute('data-theme');
    var dark = currentIsDark();
    // A <source media="(prefers-color-scheme: dark)"> outranks the <img src>,
    // so an explicit choice has to retarget the source too, not just the img.
    document.querySelectorAll('figure.fig source.dark-src').forEach(function (s) {
      s.media = explicit ? (dark ? 'all' : 'not all') : '(prefers-color-scheme: dark)';
    });
    document.querySelectorAll('figure.fig img').forEach(function (img) {
      var want = dark ? img.dataset.dark : img.dataset.light;
      if (want && img.getAttribute('src') !== want) img.setAttribute('src', want);
    });
  }
  var btn = document.querySelector('.theme-toggle');
  if (btn) btn.addEventListener('click', function () {
    var next = currentIsDark() ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('logos-theme', next);
    syncFigures();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', syncFigures);
  syncFigures();
  var navBtn = document.querySelector('.nav-toggle');
  if (navBtn) navBtn.addEventListener('click', function () {
    document.querySelector('.sidebar').classList.toggle('open');
  });
})();
</script>
</body>
</html>
""")


def build_nav(current: str):
    groups: list[tuple[str, list[dict]]] = []
    for ch in CHAPTERS:
        item = {
            "title": ch.title,
            "href": "index.html" if ch.slug == "index" else f"{ch.slug}.html",
            "current": ch.slug == current,
        }
        if groups and groups[-1][0] == ch.part:
            groups[-1][1].append(item)
        else:
            groups.append((ch.part, [item]))
    return groups


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".nojekyll").write_text("")
    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    css = (SRC / "book.css").read_text(encoding="utf-8")
    css += "\n/* pygments */\n" + HtmlFormatter(style="friendly").get_style_defs(".hl")
    (assets / "book.css").write_text(css, encoding="utf-8")

    built = 0
    for i, ch in enumerate(CHAPTERS):
        md_path = SRC / f"{ch.slug}.md"
        if not md_path.exists():
            print(f"  ! missing {md_path.name}, skipping")
            continue
        renderer = BookRenderer()
        markdown = mistune.create_markdown(
            renderer=renderer, plugins=["table", "strikethrough", "footnotes", "url"]
        )
        raw = md_path.read_text(encoding="utf-8")
        body = apply_callouts(fold_manual_captions(markdown(raw)))
        title_m = re.search(r"^#\s+(.+)$", raw, re.M)
        title = title_m.group(1).strip() if title_m else ch.title
        first_para = ""
        for line in raw.splitlines():
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("!"):
                first_para = re.sub(r"[*_`\[\]]|\(.*?\)", "", s)[:180]
                break
        prev_ch = CHAPTERS[i - 1] if i > 0 else None
        next_ch = CHAPTERS[i + 1] if i + 1 < len(CHAPTERS) else None
        page = PAGE.render(
            page_title=f"{title} — {SITE_TITLE}" if ch.slug != "index" else f"{SITE_TITLE} — {SITE_SUB}",
            description=first_para or SITE_SUB,
            site_title=SITE_TITLE,
            site_sub=SITE_SUB,
            repo=REPO,
            nav=build_nav(ch.slug),
            content=body,
            toc=renderer.headings if ch.slug not in ("index",) else [],
            prev={"title": prev_ch.title, "href": "index.html" if prev_ch.slug == "index" else f"{prev_ch.slug}.html"} if prev_ch else None,
            next={"title": next_ch.title, "href": f"{next_ch.slug}.html"} if next_ch else None,
        )
        out_name = "index.html" if ch.slug == "index" else f"{ch.slug}.html"
        (OUT / out_name).write_text(page, encoding="utf-8")
        built += 1
        print(f"  {out_name}")

    figs = OUT / "book" / "figures"
    if figs.exists():
        dest = OUT / "figures"
        dest.mkdir(exist_ok=True)
        for f in figs.glob("*.png"):
            shutil.copy2(f, dest / f.name)
    print(f"{built} pages -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
