"""Daily arXiv scoop-watch for LOGOS (PLAN.md s14/s15).

Queries the arXiv API (urllib, no new deps) for the alert terms set in
PLAN.md s14, scores each paper against the s15 scoop list (differentiators:
overtraining axis, native QAT, KV-in-the-budget, memory-budget framing),
and emits a markdown digest at ops/digests/YYYY-MM-DD.md with a relevance
call per paper. Scoring is rule-based and local; an optional one-line LLM
take per SCOOP-WATCH paper is added via the guarded native `anthropic` SDK
when ANTHROPIC_API_KEY is set and --llm is passed. No litellm, ever.

Usage:
    python ops/arxiv_triage.py                 # query arXiv, write digest
    python ops/arxiv_triage.py --dry-run       # canned fixture, no network

crontab (5090 box, daily 07:00):
    0 7 * * * cd /path/to/logos && python ops/arxiv_triage.py >> ops/triage_cron.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # allow `python ops/arxiv_triage.py`
    sys.path.insert(0, str(_REPO))

# PLAN.md s14: "Set arXiv alerts now for: quantization-aware training,
# ternary LLM, precision scaling laws, KV cache compression".
ALERT_TERMS = [
    '"quantization-aware training"',
    '"ternary LLM"',
    '"precision scaling laws"',
    '"KV cache compression"',
    '"scaling law" AND "precision"',
]

# Rule-based scoring against the PLAN.md s15 scoop list. Weights: scoop
# differentiators score highest -- a paper hitting several of these is the
# "precision+memory scaling paper" we fear.
SCOOP_SIGNALS: dict[str, int] = {
    # core collision: precision x scaling law
    r"precision[- ]?(aware )?scaling law": 6,
    r"scaling law": 3,
    # s15 differentiators
    r"overtrain|tokens[- ]per[- ]param|chinchilla": 4,   # overtraining axis
    r"quantization[- ]aware training|\bQAT\b|natively? (low[- ]bit|quantized)|trained (in|at) low[- ]?precision": 4,  # native, not PTQ
    r"kv[- ]?cache": 3,                                   # KV-in-the-budget
    r"memory[- ](budget|optimal|constrained)|iso[- ]memory|per[- ]byte|bytes? of (weights|memory)": 4,  # memory framing
    # adjacent topics worth a skim
    r"ternary|1\.58[- ]?bit|bitnet": 3,
    r"\b(2|3|4)[- ]bit\b|low[- ]bit|sub[- ]4[- ]bit": 2,
    r"post[- ]training quantization|\bPTQ\b": 1,
    r"\bLLM\b|language model": 1,
}

# Relevance thresholds for the digest call.
CALLS = [(10, "SCOOP-WATCH"), (6, "HIGH"), (3, "MEDIUM"), (0, "LOW")]

ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"

# Canned fixture for --dry-run (network-free tests).
FIXTURE: list[dict[str, str]] = [
    {
        "id": "arXiv:2508.00001",
        "title": "Precision-Aware Scaling Laws for Quantization-Aware Training "
                 "under Memory Budgets",
        "summary": "We fit scaling laws for natively quantized LLMs trained "
                   "with QAT across bit widths and tokens-per-param ratios, "
                   "including the KV cache in an iso-memory comparison of "
                   "per-byte loss under overtraining.",
    },
    {
        "id": "arXiv:2508.00002",
        "title": "Faster KV Cache Compression via Learned Token Eviction",
        "summary": "A post-hoc method for compressing the KV cache of frozen "
                   "language models at inference time.",
    },
    {
        "id": "arXiv:2508.00003",
        "title": "A Survey of Instruction Tuning Datasets",
        "summary": "We catalogue datasets used for instruction tuning of "
                   "language models.",
    },
]


def score_paper(title: str, summary: str) -> tuple[int, list[str]]:
    """Rule-based relevance score + list of matched signal patterns."""
    text = f"{title} {summary}".lower()
    total = 0
    hits: list[str] = []
    for pattern, weight in SCOOP_SIGNALS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            total += weight
            hits.append(pattern)
    return total, hits


def relevance_call(score: int) -> str:
    for threshold, label in CALLS:
        if score >= threshold:
            return label
    return "LOW"


def fetch_arxiv(term: str, max_results: int = 25, days: int = 2) -> list[dict[str, str]]:
    """One arXiv API query, newest first. Returns [{id,title,summary}]."""
    query = urllib.parse.urlencode({
        "search_query": f"all:{term}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    })
    with urllib.request.urlopen(f"{ARXIV_API}?{query}", timeout=30) as resp:
        root = ET.fromstring(resp.read())
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    papers: list[dict[str, str]] = []
    for entry in root.iter(f"{_ATOM}entry"):
        published = entry.findtext(f"{_ATOM}published") or ""
        try:
            when = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
            if when < cutoff:
                continue
        except ValueError:
            pass
        papers.append({
            "id": (entry.findtext(f"{_ATOM}id") or "").strip(),
            "title": re.sub(r"\s+", " ", entry.findtext(f"{_ATOM}title") or "").strip(),
            "summary": re.sub(r"\s+", " ", entry.findtext(f"{_ATOM}summary") or "").strip(),
        })
    return papers


def _llm_one_liner(paper: dict[str, str]) -> str | None:
    """Optional one-line take via the guarded native anthropic SDK."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # native SDK -- litellm is banned
    except Exception:
        return None
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-5",  # judgment-call route (ops/kaos/kaos.yaml)
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "LOGOS studies memory-optimal scaling laws for natively "
                    "low-bit (QAT) LLMs: precision x model size x "
                    "tokens-per-param, with the KV cache inside the memory "
                    "budget. In ONE sentence: does this paper scoop any part "
                    "of that, and which?\n\n"
                    f"Title: {paper['title']}\nAbstract: {paper['summary']}"
                ),
            }],
        )
        if resp.stop_reason == "refusal":
            return None
        return next((b.text for b in resp.content if b.type == "text"), None)
    except Exception:
        return None


def gather(dry_run: bool) -> list[dict[str, Any]]:
    """Collect + dedupe papers from all alert terms (or the fixture)."""
    if dry_run:
        raw = list(FIXTURE)
    else:
        seen: dict[str, dict[str, str]] = {}
        for term in ALERT_TERMS:
            try:
                for paper in fetch_arxiv(term):
                    seen.setdefault(paper["id"] or paper["title"], paper)
            except Exception as exc:
                print(f"[triage] query failed for {term}: {exc}", file=sys.stderr)
        raw = list(seen.values())

    scored: list[dict[str, Any]] = []
    for paper in raw:
        s, hits = score_paper(paper["title"], paper["summary"])
        scored.append({**paper, "score": s, "hits": hits,
                       "call": relevance_call(s)})
    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored


def write_digest(
    papers: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    use_llm: bool = False,
    date: dt.date | None = None,
) -> Path:
    date = date or dt.date.today()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date.isoformat()}.md"

    lines = [
        f"# LOGOS arXiv triage -- {date.isoformat()}",
        "",
        f"Alert terms (PLAN.md s14): {', '.join(ALERT_TERMS)}",
        f"Papers scored: {len(papers)}. Calls: SCOOP-WATCH >= 10, HIGH >= 6, "
        "MEDIUM >= 3, LOW < 3.",
        "",
    ]
    if not papers:
        lines.append("Nothing new matched the alert terms. Back to training.")
    for p in papers:
        lines += [
            f"## [{p['call']}] (score {p['score']}) {p['title']}",
            "",
            f"- id: {p['id']}",
            f"- signals: {', '.join(p['hits']) if p['hits'] else 'none'}",
            f"- abstract: {p['summary'][:400]}",
        ]
        if use_llm and p["call"] == "SCOOP-WATCH":
            take = _llm_one_liner(p)
            if take:
                lines.append(f"- llm take: {take.strip()}")
        if p["call"] == "SCOOP-WATCH":
            lines.append(
                "- action: read today; check the s15 pivots (lean on "
                "overtraining axis, native QAT, KV-in-budget, released suite; "
                "ship blog post to timestamp)."
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true",
                   help="use the canned fixture instead of the network")
    p.add_argument("--out-dir", default=None,
                   help="digest directory (default ops/digests)")
    p.add_argument("--llm", action="store_true",
                   help="add a one-line LLM take per SCOOP-WATCH paper "
                        "(needs ANTHROPIC_API_KEY; guarded)")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "digests"
    papers = gather(dry_run=args.dry_run)
    path = write_digest(papers, out_dir, use_llm=args.llm)
    print(f"[triage] wrote {path} ({len(papers)} papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
