"""Sequential queue runner for one manifest: train -> BPB eval -> packed
footprint -> parity -> results row, cheapest-first, skipping completed runs.

This is the local single-GPU counterpart of the launcher+Tier-2 ops pipeline:
everything it records lands in results/results.jsonl with manifest-matching
config hashes, so the fitting notebooks and the validation panel see local
runs exactly as they would see RunPod runs.

Stop anytime (Ctrl+C / kill): the trainer checkpoints on signals and the
identical command resumes. Typical use (LOGOS-Local L0):

    python scripts/run_queue.py --manifest manifests/l0.yaml \
        --data-dir data/fineweb_edu_10bt --runs-dir runs/l0 --wait-for-data
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
import yaml

from logos.config import Precision, make_model
from logos.eval.bpb import bpb_from_val_dir
from logos.export.artifact import export_artifact, load_artifact
from logos.export.pack import measured_bytes
from logos.export.parity import export_parity, synthetic_prompts
from logos.manifest.schema import load_manifest
from logos.model.transformer import Transformer
from logos.results.store import append_result, load_results
from logos.train.trainer import TrainerExtras, train

ROOT = Path(__file__).resolve().parents[1]


def wait_for_data(data_dir: Path, timeout_s: int = 7200) -> None:
    """Block until the data prep has written its val sets (end of prep)."""
    idx = data_dir / "index.json"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if idx.exists():
            doc = json.loads(idx.read_text())
            if isinstance(doc.get("val"), dict) and "val1" in doc["val"]:
                return
        time.sleep(30)
    raise TimeoutError(f"data prep did not finish within {timeout_s}s ({idx})")


def eval_and_record(spec, run_dir: Path, data_dir: Path, results_path: Path, device: str, wall: float, status: dict) -> None:
    mcfg = spec.model_config()
    if spec.seq_len:
        mcfg = dataclasses.replace(mcfg, max_seq_len=spec.seq_len)
    torch.manual_seed(spec.seed)
    model = Transformer(mcfg, kv_qat_bits=spec.kv_qat_bits)
    state = torch.load(run_dir / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    seq = spec.seq_len or 1024
    e1 = bpb_from_val_dir(model, data_dir, "val1", seq_len=seq, device=device)
    e2 = bpb_from_val_dir(model, data_dir, "val2", seq_len=seq, device=device)
    packed = measured_bytes(model)
    art = run_dir / "artifact"
    export_artifact(model, art)
    par = export_parity(
        model, load_artifact(art),
        synthetic_prompts(vocab_size=mcfg.vocab_size, n_prompts=8, seq_len=128, seed=7),
        device="cpu",
    )
    (run_dir / "parity.json").write_text(json.dumps(par, indent=2, default=float))
    append_result(
        results_path,
        spec,
        dict(
            bpb_val1=e1["bpb"], bpb_val2=e2["bpb"],
            packed_bytes=packed["total_bytes"],
            gpu_hours=wall / 3600, cost_usd=0.0, wall_s=wall,
            status=status["status"], downstream={},
            extra={"final_loss": status.get("final_loss"), "parity_pass": bool(par["pass"])},
        ),
    )
    print(
        f"  bpb1={e1['bpb']:.4f} bpb2={e2['bpb']:.4f} packed={packed['total_bytes']:,}B "
        f"parity={'PASS' if par['pass'] else 'FAIL'}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--results", default=str(ROOT / "results" / "results.jsonl"))
    ap.add_argument("--lr-rules", default=str(ROOT / "manifests" / "lr_rules.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--micro-batch-seqs", type=int, default=8,
                    help="grad-accum micro-batch (hardware lever only; 8 fits a 10GB 3080)")
    ap.add_argument("--max-runs", type=int, default=0, help="0 = drain the whole queue")
    ap.add_argument("--wait-for-data", action="store_true")
    args = ap.parse_args()

    data_dir, runs_dir = Path(args.data_dir), Path(args.runs_dir)
    if args.wait_for_data:
        print("waiting for data prep to finish...")
        wait_for_data(data_dir)
    lr_rules = yaml.safe_load(Path(args.lr_rules).read_text()) if Path(args.lr_rules).exists() else None
    _, specs = load_manifest(args.manifest)
    done_ids = set()
    if Path(args.results).exists():
        done_ids = set(load_results(args.results)["run_id"])

    queue = sorted(specs, key=lambda s: (s.est_cost_usd, s.est_gpu_hours))
    launched = 0
    for spec in queue:
        if spec.run_id in done_ids:
            continue
        if args.max_runs and launched >= args.max_runs:
            break
        print(f"[{launched + 1}] {spec.run_id}  (~{spec.est_gpu_hours:.1f} est GPU-h)")
        run_dir = runs_dir / spec.run_id
        cfg = dataclasses.replace(
            spec.train_config(lr_rules), micro_batch_seqs=args.micro_batch_seqs
        )
        t0 = time.time()
        status = train(
            spec, data_dir=data_dir, run_dir=run_dir, device=args.device,
            log_interval=20, extras=TrainerExtras(train_config=cfg),
        )
        eval_and_record(spec, run_dir, data_dir, Path(args.results), args.device, time.time() - t0, status)
        launched += 1
    print(f"queue drained: {launched} runs this session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
