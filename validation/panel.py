"""Panel orchestrator: discovers probes in validation/probes/, runs them,
aggregates a single verdict report (validation/out/report.json + report.md).

The panel is the machine half of the review round; the human half (or a KAOS
agent panel via `kaos parallel`) reads report.md and the REJECT details. A
REJECT anywhere means a claimed benchmark/result is not trustworthy as stated
— the stack or the claim gets fixed, never the gate (PLAN.md section 3:
violating design rules silently is the main way this project fails)."""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import time
from pathlib import Path

from validation.base import Probe, ProbeResult, Verdict

OUT_DIR = Path(__file__).parent / "out"


def discover(only: list[str] | None = None) -> list[Probe]:
    import validation.probes as probes_pkg

    probes: list[Probe] = []
    for mod_info in pkgutil.iter_modules(probes_pkg.__path__):
        mod = importlib.import_module(f"validation.probes.{mod_info.name}")
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, Probe)
                and obj is not Probe
                and getattr(obj, "name", "probe") != "probe"
            ):
                if only and obj.name not in only:
                    continue
                probes.append(obj())
    seen: set[str] = set()
    out = []
    for p in probes:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def render_md(results: list[ProbeResult]) -> str:
    icon = {Verdict.ACCEPT: "PASS", Verdict.REJECT: "**REJECT**", Verdict.VOID: "*VOID*"}
    lines = [
        "# LOGOS validation panel report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Probe | Verdict | Gates | Wall (s) |",
        "|-------|---------|-------|----------|",
    ]
    for r in results:
        n_pass = sum(1 for g in r.gates if g.passed)
        lines.append(
            f"| {r.probe} | {icon[r.verdict]} | {n_pass}/{len(r.gates)} | {r.wall_s:.1f} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r.probe} — {r.verdict.value}")
        for g in r.gates:
            mark = "x" if g.passed else " "
            kill = " [KILL]" if (g.kill and not g.passed) else ""
            lines.append(f"- [{mark}] `{g.gate}` {g.description}{kill}")
            if g.detail:
                lines.append(f"      {g.detail}")
        if r.error:
            lines.append(f"```\n{r.error}\n```")
        lines.append("")
    return "\n".join(lines)


def run_panel(only: list[str] | None = None) -> tuple[Verdict, list[ProbeResult]]:
    probes = discover(only)
    results = [p.run() for p in probes]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str)
    )
    (OUT_DIR / "report.md").write_text(render_md(results))
    if any(r.verdict is Verdict.REJECT for r in results):
        overall = Verdict.REJECT
    elif any(r.verdict is Verdict.VOID for r in results):
        overall = Verdict.VOID
    else:
        overall = Verdict.ACCEPT
    return overall, results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("logos-validate", description=__doc__)
    ap.add_argument("--all", action="store_true", help="run every probe")
    ap.add_argument("--probe", nargs="*", default=None, help="run specific probes by name")
    ap.add_argument("--list", action="store_true", help="list available probes")
    args = ap.parse_args(argv)
    if args.list:
        for p in discover():
            print(f"{p.name:24s} {p.description}")
        return 0
    only = None if args.all else args.probe
    if not args.all and not only:
        ap.error("pass --all, --probe <names>, or --list")
    overall, results = run_panel(only)
    for r in results:
        print(f"[{r.verdict.value:6s}] {r.probe} ({sum(g.passed for g in r.gates)}/{len(r.gates)} gates)")
        if r.error:
            print(f"         {r.error.splitlines()[-1] if r.error.splitlines() else r.error}")
    print(f"\nOverall: {overall.value}  (report: {OUT_DIR / 'report.md'})")
    return 0 if overall is Verdict.ACCEPT else 1


if __name__ == "__main__":
    raise SystemExit(main())
