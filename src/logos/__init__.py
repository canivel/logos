"""LOGOS: memory-optimal scaling laws for natively low-bit LLMs.

Sibling project to KAOS (ops layer). Model suite: TRIT.
Research plan: research/logos-research-plan-v0.2.md (committed as PLAN.md).
"""

__version__ = "0.1.0"

from logos.config import (  # noqa: F401
    LADDER,
    ModelConfig,
    Precision,
    RunSpec,
    TrainConfig,
)
