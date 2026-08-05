"""Scaling-law fitting subsystem (PLAN.md sections 8-10): candidate forms,
Chinchilla-standard fitting, LOSO selection + bootstrap uncertainty, and the
memory-budget prescriptions the law exists to produce."""

from logos.fitting.fit import FitResult, fit_form, records_from_jsonl  # noqa: F401
from logos.fitting.forms import (  # noqa: F401
    Form,
    FormAFree,
    FormAParam,
    FormB,
    FormC,
    default_forms,
)
from logos.fitting.prescribe import (  # noqa: F401
    effective_capacity_summary,
    optimal_config,
    phase_diagram,
    total_footprint_optimal,
)
from logos.fitting.select import (  # noqa: F401
    bootstrap_ci,
    check_extrapolation,
    loso,
    predict_with_band,
    seed_sigma_by_size,
    select_by_loso,
)
