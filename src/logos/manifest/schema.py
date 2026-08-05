"""YAML manifest IO for lists of RunSpec (PLAN.md s12: "every run defined in a
versioned manifest so nothing is launched by hand twice").

A manifest file is a mapping:
  {version, generated_by, git_sha, phase, meta, runs: [asdict(RunSpec), ...]}
Round-trip is stable: load(save(specs)) preserves every field and therefore
every RunSpec.config_hash() (the validation panel depends on this).
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from logos.config import LADDER, Precision, RunSpec

MANIFEST_VERSION = 1
GENERATED_BY = "logos.manifest.generate"

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Field coercions so YAML round-trips reproduce the exact Python types that
# went into config_hash() (e.g. tokens_per_param stays float, seed stays int).
_FLOAT_FIELDS = ("tokens_per_param", "est_gpu_hours", "est_cost_usd", "max_wall_clock_mult")
_INT_FIELDS = ("seed", "gqa_ratio")
_OPT_INT_FIELDS = ("total_tokens", "kv_qat_bits")
_OPT_FLOAT_FIELDS = ("lr", "lr_mult_override")
_STR_FIELDS = ("run_id", "phase", "size", "precision", "ffn_type", "notes")


class ManifestError(ValueError):
    """Raised on malformed or invalid manifests."""


def git_sha() -> str | None:
    """Best-effort HEAD sha for manifest provenance; null when unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def validate_specs(specs: list[RunSpec]) -> None:
    """Unique run_ids, known ladder sizes and precisions, tokens_per_param > 0."""
    seen: set[str] = set()
    for s in specs:
        if s.run_id in seen:
            raise ManifestError(f"duplicate run_id: {s.run_id}")
        seen.add(s.run_id)
        if s.size not in LADDER:
            raise ManifestError(f"{s.run_id}: unknown size {s.size!r} (not in LADDER)")
        try:
            Precision(s.precision)
        except ValueError as e:
            raise ManifestError(f"{s.run_id}: unknown precision {s.precision!r}") from e
        if not s.tokens_per_param > 0:
            raise ManifestError(f"{s.run_id}: tokens_per_param must be > 0")


def _coerce_run(raw: dict[str, Any]) -> RunSpec:
    known = {f.name for f in fields(RunSpec)}
    extra = set(raw) - known
    if extra:
        raise ManifestError(f"unknown RunSpec fields: {sorted(extra)}")
    d = dict(raw)
    for k in _STR_FIELDS:
        if k in d and d[k] is not None:
            d[k] = str(d[k])
    for k in _FLOAT_FIELDS:
        if k in d and d[k] is not None:
            d[k] = float(d[k])
    for k in _INT_FIELDS:
        if k in d and d[k] is not None:
            d[k] = int(d[k])
    for k in _OPT_INT_FIELDS:
        if d.get(k) is not None:
            d[k] = int(d[k])
    for k in _OPT_FLOAT_FIELDS:
        if d.get(k) is not None:
            d[k] = float(d[k])
    if d.get("tags") is None:
        d["tags"] = []
    d["tags"] = [str(t) for t in d["tags"]]
    try:
        return RunSpec(**d)
    except TypeError as e:
        raise ManifestError(f"bad run entry {raw.get('run_id')!r}: {e}") from e


def save_manifest(specs: list[RunSpec], path: str | Path, meta: dict[str, Any]) -> Path:
    """Write a versioned manifest. Phase comes from meta['phase'] or the specs."""
    validate_specs(specs)
    phase = meta.get("phase") or (specs[0].phase if specs else None)
    doc = {
        "version": MANIFEST_VERSION,
        "generated_by": GENERATED_BY,
        "git_sha": git_sha(),
        "phase": phase,
        "meta": meta,
        "runs": [asdict(s) for s in specs],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return path


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[RunSpec]]:
    """Load and validate a manifest -> (meta, specs)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "runs" not in doc:
        raise ManifestError(f"{path}: not a run manifest (no 'runs' key)")
    specs = [_coerce_run(r) for r in doc["runs"] or []]
    validate_specs(specs)
    meta = dict(doc.get("meta") or {})
    meta.setdefault("phase", doc.get("phase"))
    return meta, specs
