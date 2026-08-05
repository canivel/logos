"""KAOS-launchable tier agents and their concrete tools (PLAN.md section 13).

Nothing in this package hard-requires kaos: `tools` is pure Python and
`agents` guards every kaos import so the fallback loops in ops/fallback/
can reuse the same tool functions.
"""
