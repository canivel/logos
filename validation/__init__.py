"""LOGOS validation panel.

A separate verification layer, deliberately independent of src/logos authorship:
every check is a falsifiable probe with pre-registered, sha256-locked kill
gates (the KAOS eval discipline). Probes re-derive expected behavior from
first principles / reference definitions and compare against what the stack
produces — they never trust src/logos internals as ground truth for the
quantity under test.

Run: `logos-validate --all` or `python -m validation.panel --all`.
"""

from validation.base import GateOutcome, Probe, Verdict  # noqa: F401
