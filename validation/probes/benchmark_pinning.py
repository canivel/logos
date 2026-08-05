"""Probe benchmark_pinning — PLAN.md s4: "lm-eval-harness, version pinned";
task suites per s6/s7/s9/s11; BPB val sets held out (s5, principle 3).

The pin string and the task lists are re-encoded here from the plan text and
compared against logos.eval.downstream. Val-set disjointness is verified on
the binary data itself (byte-corpus split boundaries against the source
text, or content non-overlap), never from code comments.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

from validation.base import GateOutcome, Probe

REPO = Path(__file__).resolve().parents[2]
DOWNSTREAM_SRC = REPO / "src" / "logos" / "eval" / "downstream.py"
DATA_DIR = REPO / "data" / "local_p0"
RESULTS = REPO / "results" / "results.jsonl"
RESULTS_POLL_S = 10.0

# Re-encoded from the plan (PLAN.md s4 pin discipline; pin value fixed for the
# TRIT suite). Results from any other harness version are not comparable.
PIN = "0.4.9"

# Task suites re-encoded from PLAN.md text ("lambada" matches any lambada_* task):
#   s6  (P1): ARC-e, HellaSwag, PIQA, SciQ, LAMBADA (0-shot)
#   s7  (P2): + ARC-c, WinoGrande, BoolQ, OBQA
#   s9  (P3): + MMLU 5-shot
#   s11 (P5): + GSM8K, IFEval
PLAN_SMALL = {"arc_easy", "hellaswag", "piqa", "sciq", "lambada"}
PLAN_FULL = PLAN_SMALL | {"arc_challenge", "winogrande", "boolq", "openbookqa"}
PLAN_P3 = PLAN_FULL | {"mmlu"}
PLAN_CAPSTONE = PLAN_P3 | {"gsm8k", "ifeval"}
PLAN_MMLU_FEWSHOT = 5


def _norm(task: str) -> str:
    return "lambada" if task.startswith("lambada") else task


def _read_u16(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)


class BenchmarkPinningProbe(Probe):
    name = "benchmark_pinning"
    description = "lm-eval pin enforced; task suites match the plan; val sets truly held out"
    gate_specs = [
        ("G1", "downstream.py pins lm-eval == 0.4.9: pin constant and a version-mismatch "
               "raise path exist in source; the module imports cleanly without lm_eval; "
               "missing -> informative ImportError, wrong version -> RuntimeError"),
        ("G2", "task suites equal the PLAN.md lists: SMALL {arc_easy,hellaswag,piqa,sciq,"
               "lambada}, FULL +{arc_challenge,winogrande,boolq,openbookqa}, P3 +mmlu "
               "(5-shot), capstone +{gsm8k,ifeval}"),
        ("G3", "the two BPB val sets are disjoint from each other and from train, verified "
               "on the binary data (byte-corpus split boundaries, else synthetic content "
               "non-overlap)"),
        ("G4", "every results.jsonl row carries provenance: run_id, manifest-matching "
               "config_hash, and status complete for benchmark-bearing rows (vacuous if "
               "results.jsonl absent)"),
    ]

    # ------------------------------------------------------------------ G1

    def _collect_pin(self) -> dict:
        src = DOWNSTREAM_SRC.read_text(encoding="utf-8")
        source = {
            "pin_constant": bool(
                re.search(r'LM_EVAL_PIN\s*=\s*"' + re.escape(PIN) + r'"', src)
            ),
            "mismatch_raise": "raise RuntimeError" in src and "mismatch" in src,
            "missing_raise": "raise ImportError" in src,
        }
        saved = {
            k: sys.modules[k]
            for k in list(sys.modules)
            if k == "lm_eval" or k.startswith("lm_eval.") or k == "logos.eval.downstream"
        }
        for k in saved:
            del sys.modules[k]
        behav: dict = {}
        try:
            sys.modules["lm_eval"] = None  # forces ImportError on `import lm_eval`
            try:
                ds = importlib.import_module("logos.eval.downstream")
                behav["import_ok"] = True
            except Exception as e:  # module must import without lm_eval
                behav["import_ok"] = False
                behav["import_err"] = repr(e)
                return {"source": source, "behavior": behav}
            try:
                ds._require_lm_eval()
                behav["missing"] = "no error raised"
            except Exception as e:
                behav["missing"] = type(e).__name__
                behav["missing_informative"] = isinstance(e, ImportError) and PIN in str(e)
            fake = types.ModuleType("lm_eval")
            fake.__version__ = "0.0.0"
            sys.modules["lm_eval"] = fake
            try:
                ds._require_lm_eval()
                behav["mismatch"] = "no error raised"
            except Exception as e:
                behav["mismatch"] = type(e).__name__
                behav["mismatch_informative"] = (
                    isinstance(e, RuntimeError) and PIN in str(e) and "mismatch" in str(e)
                )
        finally:
            sys.modules.pop("lm_eval", None)
            sys.modules.pop("logos.eval.downstream", None)
            sys.modules.update(saved)
        return {"source": source, "behavior": behav}

    # ------------------------------------------------------------------ G3

    def _collect_valsets(self) -> dict:
        if (DATA_DIR / "index.json").exists():
            index = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))
            train = np.concatenate(
                [_read_u16(DATA_DIR / s["file"]) for s in index["shards"]]
            )
            val1 = _read_u16(DATA_DIR / index["val"]["val1"]["file"])
            val2 = _read_u16(DATA_DIR / index["val"]["val2"]["file"])
            out: dict = {"mode": "byte-corpus", "dataset": index.get("dataset")}
            src_name = str(index.get("dataset", "")).split(":", 1)[-1]
            candidates = [REPO / src_name, REPO.parent / src_name]
            src_path = next((p for p in candidates if p.exists()), None)
            nt, n1, n2 = len(train), len(val1), len(val2)
            if src_path is not None and index.get("tokenizer") == "byte-level":
                src = np.frombuffer(src_path.read_bytes(), dtype=np.uint8).astype(np.uint16)
                out["boundaries"] = {
                    "src_len": len(src),
                    "covers": len(src) >= nt + n1 + n2,
                    "train_is_prefix": np.array_equal(train, src[:nt]),
                    "val1_is_next": np.array_equal(val1, src[nt : nt + n1]),
                    "val2_is_next": np.array_equal(val2, src[nt + n1 : nt + n1 + n2]),
                }
                out["disjoint"] = all(out["boundaries"][k] for k in
                                      ("covers", "train_is_prefix", "val1_is_next", "val2_is_next"))
            else:
                tb = train.astype(np.uint8).tobytes()
                b1, b2 = val1.astype(np.uint8).tobytes(), val2.astype(np.uint8).tobytes()
                out["disjoint"] = (
                    b1 not in tb and b2 not in tb and b1 != b2
                    and b1 not in b2 and b2 not in b1
                )
                out["boundaries"] = "source text not found; content non-overlap check"
            return out
        # No local corpus: synthetic prepare run, doc/content disjointness from data.
        from logos.data.prepare import prepare_synthetic

        with tempfile.TemporaryDirectory() as td:
            index = prepare_synthetic(Path(td), n_shards=2, shard_tokens=16_384,
                                      vocab_size=512, seed=41, val_tokens=2048)
            tb = b"".join(
                _read_u16(Path(td) / s["file"]).tobytes() for s in index["shards"]
            )
            b1 = _read_u16(Path(td) / index["val"]["val1"]["file"]).tobytes()
            b2 = _read_u16(Path(td) / index["val"]["val2"]["file"]).tobytes()
            return {
                "mode": "synthetic",
                "disjoint": b1 not in tb and b2 not in tb and b1 != b2
                and b1 not in b2 and b2 not in b1,
            }

    # ------------------------------------------------------------------ G4

    def _collect_results(self) -> dict:
        deadline = time.time() + RESULTS_POLL_S
        while not RESULTS.exists() and time.time() < deadline:
            time.sleep(1.0)
        if not RESULTS.exists():
            return {"present": False}
        from logos.manifest.schema import ManifestError, load_manifest

        table: dict[str, set[str]] = {}
        for p in sorted((REPO / "manifests").glob("*.yaml")):
            try:
                _, specs = load_manifest(p)
            except ManifestError:
                continue
            for s in specs:
                table.setdefault(s.run_id, set()).add(s.config_hash())
        rows = [
            json.loads(line)
            for line in RESULTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        bad = []
        for r in rows:
            has_bench = bool(r.get("downstream")) or r.get("bpb_val1") is not None \
                or r.get("bpb_val2") is not None
            problems = []
            if not r.get("run_id"):
                problems.append("no run_id")
            if not r.get("config_hash"):
                problems.append("no config_hash")
            if "status" not in r:
                problems.append("no status")
            if has_bench:
                if r.get("status") != "complete":
                    problems.append(f"benchmark row with status={r.get('status')!r}")
                if r.get("config_hash") not in table.get(r.get("run_id"), set()):
                    problems.append("config_hash not in any manifest")
            if problems:
                bad.append({"run_id": r.get("run_id"), "problems": problems})
        return {"present": True, "n_rows": len(rows), "n_bad": len(bad), "bad": bad[:5]}

    # ------------------------------------------------------------------

    def collect(self) -> dict:
        m = {"g1": self._collect_pin()}
        ds = importlib.import_module("logos.eval.downstream")
        m["g2"] = {
            "small": {_norm(t) for t in ds.SMALL_SUITE} == PLAN_SMALL,
            "full": {_norm(t) for t in ds.FULL_SUITE} == PLAN_FULL,
            "p3": {_norm(t) for t in ds.P3_SUITE} == PLAN_P3,
            "capstone": {_norm(t) for t in ds.CAPSTONE_SUITE} == PLAN_CAPSTONE,
            "mmlu_fewshot": ds.FEWSHOT.get("mmlu") == PLAN_MMLU_FEWSHOT,
            "suites": {
                "small": sorted(ds.SMALL_SUITE),
                "full": sorted(ds.FULL_SUITE),
                "p3": sorted(ds.P3_SUITE),
                "capstone": sorted(ds.CAPSTONE_SUITE),
            },
        }
        m["g3"] = self._collect_valsets()
        m["g4"] = self._collect_results()
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)
        src, behav = m["g1"]["source"], m["g1"]["behavior"]
        g1_ok = (
            all(src.values())
            and behav.get("import_ok")
            and behav.get("missing_informative")
            and behav.get("mismatch_informative")
        )
        out = [
            GateOutcome(
                "G1", specs["G1"], g1_ok,
                detail=f"source={src}; behavior={behav}",
            )
        ]
        g2 = m["g2"]
        g2_ok = all(g2[k] for k in ("small", "full", "p3", "capstone", "mmlu_fewshot"))
        out.append(
            GateOutcome(
                "G2", specs["G2"], g2_ok,
                detail=(
                    "suites match plan lists; mmlu 5-shot"
                    if g2_ok
                    else f"mismatch flags: { {k: g2[k] for k in ('small','full','p3','capstone','mmlu_fewshot')} }; "
                         f"actual={g2['suites']}"
                ),
            )
        )
        g3 = m["g3"]
        out.append(
            GateOutcome(
                "G3", specs["G3"], bool(g3["disjoint"]),
                detail=f"mode={g3['mode']}; boundaries={g3.get('boundaries')}; "
                       f"disjoint={g3['disjoint']}",
            )
        )
        g4 = m["g4"]
        if not g4["present"]:
            out.append(
                GateOutcome(
                    "G4", specs["G4"], True,
                    detail="vacuous: results.jsonl absent after poll",
                )
            )
        else:
            out.append(
                GateOutcome(
                    "G4", specs["G4"], g4["n_bad"] == 0,
                    detail=f"{g4['n_rows']} rows, {g4['n_bad']} without provenance "
                           f"({g4['bad'] if g4['n_bad'] else 'none'})",
                )
            )
        return out
