"""Append-only results store + variance/gap analysis (PLAN.md s3 principle 4:
any reported gap must exceed 2 sigma for its size class)."""

from logos.results.store import (  # noqa: F401
    append_result,
    check_hash,
    gap_vs_sigma,
    load_results,
    seed_sigma,
)
