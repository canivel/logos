"""Run manifests, launcher, and budget ledger (PLAN.md s12-s13).

Operational rule: every run defined in a versioned manifest so nothing is
launched by hand twice.
"""

from logos.manifest.schema import (  # noqa: F401
    ManifestError,
    load_manifest,
    save_manifest,
    validate_specs,
)
