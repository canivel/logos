"""Generate the exact phase manifests from PLAN.md sections 5-11.

GPU-hour estimate model (all figures land within the plan's ~+/-30-40% bars):
    FLOPs      = 6 * N_nonemb * D            (C ~= 6ND, PLAN.md s12)
    peak       = 989e12 FLOP/s               (H100 SXM bf16 dense peak)
    MFU        = 0.38 bf16 arms, 0.30 quantized arms (~20% QAT penalty)
    gpu_hours  = FLOPs / (MFU * peak) / 3600
    cost_usd   = gpu_hours * 2.80            (H100 SXM ~$2.80/GPU-h, PLAN.md s12)

Usage: python -m logos.manifest.generate --out manifests/
Writes p0..p5 manifests, lr_rules.yaml, and summary.md (estimates vs plan s12).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from logos.config import (
    ALL_PRECISIONS,
    DEFAULT_LR_MULT,
    Precision,
    RunSpec,
    make_model,
)
from logos.manifest.schema import save_manifest

H100_PEAK_FLOPS = 989e12
MFU_BF16 = 0.38
MFU_QAT = 0.30  # ~20% QAT throughput penalty (PLAN.md s12)
USD_PER_GPU_HOUR = 2.80

W158, W2, W3, W4, BF16 = (
    Precision.W1_58,
    Precision.W2,
    Precision.W3,
    Precision.W4,
    Precision.BF16,
)

# Plan s12 ranges: phase -> (gpu_h_lo, gpu_h_hi, usd_lo, usd_hi).
PLAN_S12 = {
    "p0": (60, 80, 200, 300),
    "p1": (220, 260, 700, 1000),
    "p2": (1800, 1800, 5000, 6000),
    "p2ext": (1150, 1150, 3200, 3200),
    "p3": (1050, 1450, 3000, 4500),
    "p4": (350, 500, 1000, 1500),
    "p5": (2900, 2900, 8000, 9500),
}


def estimate(spec: RunSpec) -> tuple[float, float]:
    """(gpu_hours, cost_usd) for one run under the module cost model."""
    m = spec.model_config()
    tokens = spec.total_tokens or int(spec.tokens_per_param * m.n_nonemb)
    mfu = MFU_QAT if Precision(spec.precision).is_quantized else MFU_BF16
    gpu_hours = 6.0 * m.n_nonemb * tokens / (mfu * H100_PEAK_FLOPS) / 3600.0
    return gpu_hours, gpu_hours * USD_PER_GPU_HOUR


def _mk(run_id: str, phase: str, size: str, precision: Precision, tp: float, **kw: Any) -> RunSpec:
    """Build a RunSpec and fill est_gpu_hours / est_cost_usd from the model."""
    spec = RunSpec(
        run_id=run_id,
        phase=phase,
        size=size,
        precision=precision.value,
        tokens_per_param=float(tp),
        **kw,
    )
    spec.est_gpu_hours, spec.est_cost_usd = estimate(spec)
    return spec


def _grid_id(size: str, precision: Precision, tp: float, seed: int) -> str:
    """Phase-free id for grid cells so P1 gap-study rows and P2 rows share ids."""
    return f"{size}-{precision.value}-tp{int(tp)}-s{seed}"


# ---------------------------------------------------------------- phases


def gen_p0() -> tuple[dict[str, Any], list[RunSpec]]:
    """P0 replication (PLAN.md s5): 25M/60M, ternary vs bf16, 20x/320x.

    The plan prints 16 runs; a full 2x2x2x3 crossing is 24, so seeds are
    concentrated at 320x (3 seeds) with 1 seed at 20x: 2*2*1 + 2*2*3 = 16.
    """
    specs: list[RunSpec] = []
    note = "seeds concentrated at 320x per P0 budget (plan prints 16 runs, not 24)"
    for size in ("25m", "60m"):
        for prec in (W158, BF16):
            for tp, seeds in ((20, (0,)), (320, (0, 1, 2))):
                for seed in seeds:
                    specs.append(
                        _mk(
                            f"p0-{_grid_id(size, prec, tp, seed)}",
                            "p0",
                            size,
                            prec,
                            tp,
                            seed=seed,
                            tags=["repro", "seeds_at_320x"],
                            notes=note,
                        )
                    )
    meta = {
        "phase": "p0",
        "goal": "RQ1: trainer/quantizer trust + measured seed noise floor",
        "seed_policy": note,
    }
    return meta, specs


def gen_p1() -> tuple[dict[str, Any], list[RunSpec]]:
    """P1 (PLAN.md s6): LR probes (25), FFN ablation (4), gap study (15)."""
    specs: list[RunSpec] = []
    # (a) LR transfer probes at 60M: 5 precisions x 5 LR multipliers, 1B tokens.
    for prec in ALL_PRECISIONS:
        for mult in (0.5, 1, 2, 4, 8):
            specs.append(
                _mk(
                    f"p1-lr-60m-{prec.value}-m{mult:g}",
                    "p1",
                    "60m",
                    prec,
                    1e9 / make_model("60m", prec).n_nonemb,  # from total_tokens override
                    total_tokens=1_000_000_000,
                    lr_mult_override=float(mult),
                    tags=["lr_probe"],
                    notes="output: one LR rule per precision, frozen in manifests/lr_rules.yaml",
                )
            )
    # (b) FFN ablation at 125M ternary: swiglu vs sq_relu at 20x/80x.
    for ffn in ("swiglu", "sq_relu"):
        for tp in (20, 80):
            specs.append(
                _mk(
                    f"p1-ffn-125m-{ffn}-tp{tp}",
                    "p1",
                    "125m",
                    W158,
                    tp,
                    ffn_type=ffn,
                    tags=["ffn_ablation"],
                    notes="freeze the winner across all arms (PLAN.md s6)",
                )
            )
    # (c) Gap study at 125M: 5 precisions x {20,80,320}, seed 0. These 15 ARE
    # the 125m rows of the P2 grid (same run_ids; P2 re-emits them at zero cost).
    for prec in ALL_PRECISIONS:
        for tp in (20, 80, 320):
            specs.append(
                _mk(
                    _grid_id("125m", prec, tp, 0),
                    "p1",
                    "125m",
                    prec,
                    tp,
                    tags=["gap_study", "reused_in_p2"],
                    notes="designed with the exact P2 configs; reused in the P2 grid",
                )
            )
    meta = {
        "phase": "p1",
        "goal": "RQ2: gap-vs-overtraining curves; freeze LR rule and FFN choice",
        "reuse": "gap_study run_ids are the 125m rows of p2.yaml (trained once, here)",
    }
    return meta, specs


def gen_p2() -> tuple[dict[str, Any], list[RunSpec]]:
    """P2 core grid, section-7 matrix verbatim (~130 runs, 15 reused from P1)."""
    specs: list[RunSpec] = []
    for size, seeds in (("25m", (0, 1, 2)), ("60m", (0, 1, 2))):
        for tp in (20, 80, 320):
            for prec in ALL_PRECISIONS:
                for seed in seeds:
                    specs.append(
                        _mk(
                            _grid_id(size, prec, tp, seed),
                            "p2",
                            size,
                            prec,
                            tp,
                            seed=seed,
                            tags=["grid"],
                        )
                    )
    # 125M rows: identical run_ids to the P1 gap study; already trained there.
    for tp in (20, 80, 320):
        for prec in ALL_PRECISIONS:
            s = _mk(
                _grid_id("125m", prec, tp, 0),
                "p2",
                "125m",
                prec,
                tp,
                tags=["grid", "reused_from_p1"],
                notes="trained as P1 gap_study; zero incremental cost in P2",
            )
            s.est_gpu_hours = 0.0
            s.est_cost_usd = 0.0
            specs.append(s)
    for tp in (20, 80, 320):
        for prec in ALL_PRECISIONS:
            specs.append(_mk(_grid_id("250m", prec, tp, 0), "p2", "250m", prec, tp, tags=["grid"]))
    for tp in (20, 80):
        for prec in ALL_PRECISIONS:
            specs.append(_mk(_grid_id("490m", prec, tp, 0), "p2", "490m", prec, tp, tags=["grid"]))
    meta = {
        "phase": "p2",
        "goal": "RQ3: the memory-optimal sweep; fitted law + iso-memory frontier",
        "reuse": "15 x 125m rows carry tag reused_from_p1 with zero cost",
    }
    return meta, specs


def gen_p2ext() -> tuple[dict[str, Any], list[RunSpec]]:
    """P2 extended tier: 490M @ 320x anchors, {1.58, 4, bf16}."""
    specs = [
        _mk(
            _grid_id("490m", prec, 320, 0),
            "p2ext",
            "490m",
            prec,
            320,
            tags=["grid", "extended_tier"],
        )
        for prec in (W158, W4, BF16)
    ]
    meta = {"phase": "p2ext", "goal": "490m@320x anchors for the overtrained tier"}
    return meta, specs


def gen_p3() -> tuple[dict[str, Any], list[RunSpec]]:
    """P3 extrapolation anchors: 1B x {1.58, 4, bf16} x {20x, 80x}."""
    specs = [
        _mk(
            _grid_id("1b", prec, tp, 0),
            "p3",
            "1b",
            prec,
            tp,
            tags=["extrapolation_anchor"],
        )
        for prec in (W158, W4, BF16)
        for tp in (20, 80)
    ]
    meta = {
        "phase": "p3",
        "goal": "RQ4: predict 1B before training it, then train and check",
        "prediction_protocol": "freeze fitted law + commit predictions before launch",
    }
    return meta, specs


def gen_p4() -> tuple[dict[str, Any], list[RunSpec]]:
    """P4 total-footprint arms (PLAN.md s10): native KV-QAT + GQA-ratio arm.

    The post-hoc KV sweep is EVAL-ONLY and lives in meta['eval_jobs'], not as
    RunSpecs (no training compute).
    """
    specs: list[RunSpec] = []
    for kv in (8, 4):
        for prec in (W158, W4, BF16):
            specs.append(
                _mk(
                    f"p4-250m-{prec.value}-kv{kv}-tp80",
                    "p4",
                    "250m",
                    prec,
                    80,
                    kv_qat_bits=kv,
                    tags=["kv_qat"],
                )
            )
    # 32k context-extension variants (the plan's "~8 runs"): 1.58/bf16 @ kv4.
    n125 = make_model("250m").n_nonemb
    for prec in (W158, BF16):
        specs.append(
            _mk(
                f"p4-250m-{prec.value}-kv4-tp80-ctx32k",
                "p4",
                "250m",
                prec,
                80,
                kv_qat_bits=4,
                total_tokens=int(80 * n125 * 1.10),
                tags=["kv_qat", "ctx_ext"],
                notes="+10% tokens mid-train context extension to 32k",
            )
        )
    specs.append(
        _mk(
            "p4-250m-1.58-gqa8-tp80",
            "p4",
            "250m",
            W158,
            80,
            gqa_ratio=8,
            tags=["gqa"],
            notes="8:1 GQA arm; the 4:1 baseline reuses the P2 grid run",
        )
    )
    meta = {
        "phase": "p4",
        "goal": "RQ5: weights + KV total-footprint law",
        "eval_jobs": {
            "kv_posthoc_sweep": {
                "description": "post-hoc KV-cache quantization on existing checkpoints; eval-only, no training runs",
                "checkpoint_globs": [
                    "runs/250m-*-tp*-s0/checkpoints/final*",
                    "runs/490m-*-tp*-s0/checkpoints/final*",
                    "runs/1b-*-tp*-s0/checkpoints/final*",
                ],
                "kv_bits": [8, 4, 3, 2],
                "evals": ["bpb", "ruler_4k_32k", "niah", "longbench_e"],
            }
        },
    }
    return meta, specs


def gen_p5() -> tuple[dict[str, Any], list[RunSpec]]:
    """P5 capstone Tier A: trit-1.5b, ternary, ~300B tokens (PLAN.md s11)."""
    n = make_model("1.5b", W158).n_nonemb
    total = 300_000_000_000
    spec = _mk(
        "p5-1.5b-capstone",
        "p5",
        "1.5b",
        W158,
        total / n,
        total_tokens=total,
        tags=["capstone", "tier_a"],
        notes=(
            "Full-quality recipe, not the grid recipe (PLAN.md s11): bitnet2stage "
            "two-stage LR + weight-decay schedule; high-quality data anneal in the "
            "last 10-15%; then SFT (~1-2B tokens open instruction data) and light "
            "DPO stages. Estimate covers pretrain FLOPs; anneal/SFT/DPO/eval ride "
            "the phase budget headroom."
        ),
    )
    meta = {
        "phase": "p5",
        "goal": "RQ6: beat same-footprint baselines at the 2GB device class",
        "schedule": "bitnet2stage + anneal + SFT/DPO (stages recorded in run notes)",
    }
    return meta, [spec]


PHASES = {
    "p0": gen_p0,
    "p1": gen_p1,
    "p2": gen_p2,
    "p2ext": gen_p2ext,
    "p3": gen_p3,
    "p4": gen_p4,
    "p5": gen_p5,
}


# ---------------------------------------------------------------- outputs

LR_RULES_HEADER = "# P0 defaults — REPLACED by P1 probe results; frozen thereafter (PLAN.md s6)\n"


def write_lr_rules(path: Path) -> None:
    lines = [LR_RULES_HEADER]
    for prec in ALL_PRECISIONS:
        lines.append(f'"{prec.value}": {DEFAULT_LR_MULT[prec]}\n')
    path.write_text("".join(lines), encoding="utf-8")


def summarize(all_specs: dict[str, list[RunSpec]]) -> str:
    """Markdown table: phase, runs, est GPU-h, est cost vs PLAN.md s12."""
    rows = []
    tot_runs = tot_h = tot_usd = 0.0
    for phase, specs in all_specs.items():
        h = sum(s.est_gpu_hours for s in specs)
        usd = sum(s.est_cost_usd for s in specs)
        lo_h, hi_h, lo_u, hi_u = PLAN_S12[phase]
        rows.append(
            f"| {phase} | {len(specs)} | {h:,.0f} | ${usd:,.0f} "
            f"| {lo_h}-{hi_h} | ${lo_u:,}-${hi_u:,} |"
        )
        tot_runs += len(specs)
        tot_h += h
        tot_usd += usd
    lines = [
        "# LOGOS run-manifest summary",
        "",
        "Cost model: FLOPs = 6*N_nonemb*D; H100 bf16 peak 989 TFLOP/s; "
        "MFU 0.38 bf16 / 0.30 quantized (~20% QAT penalty); $2.80/GPU-h. "
        "No overhead multiplier, so estimates sit at/below the plan's ranges "
        "(plan s12 carries ~+/-30% bars and +20% overhead on P2).",
        "",
        "| Phase | Runs | Est GPU-h | Est cost | Plan s12 GPU-h | Plan s12 cost |",
        "|-------|------|-----------|----------|----------------|---------------|",
        *rows,
        f"| **total** | **{tot_runs:.0f}** | **{tot_h:,.0f}** | **${tot_usd:,.0f}** "
        f"| ~6,500-7,000 core | ~$21,000-26,000 core |",
        "",
        "Reused runs (p2 125m rows, tag `reused_from_p1`) are emitted at zero cost.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("manifests"))
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    all_specs: dict[str, list[RunSpec]] = {}
    for phase, gen in PHASES.items():
        meta, specs = gen()
        save_manifest(specs, out / f"{phase}.yaml", meta)
        all_specs[phase] = specs
    write_lr_rules(out / "lr_rules.yaml")

    summary = summarize(all_specs)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
