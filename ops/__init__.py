"""LOGOS ops layer: KAOS agents + boring fallbacks (PLAN.md section 13).

Doctrine: agents run the toil, humans run the experiment. Every component
here has a cron/bash fallback because LOGOS never blocks on KAOS
(the 1-2 day rule, PLAN.md sections 13 and 15).
"""
