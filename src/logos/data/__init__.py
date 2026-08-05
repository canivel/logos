"""Data pipeline: FineWeb-Edu tokenized shards and the deterministic loader
(PLAN.md sections 4-5). Same data order everywhere (principle 5)."""

from logos.data.loader import TokenLoader  # noqa: F401
from logos.data.prepare import prepare_fineweb, prepare_synthetic  # noqa: F401
