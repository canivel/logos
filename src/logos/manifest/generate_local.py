"""LOGOS-Local manifest generator (research plan v0.3).

Run ids carry an `l-` prefix: the L-series shares ladder sizes with the
v0.2 reference manifests but under a different frozen protocol (seq 1024,
batch 2^18), so ids must never collide across the two programs.

Emits manifests/l0.yaml .. l5.yaml: the local program that fits the law at
3M-60M on the 3080/5090 and validates it with <= $100 of RunPod anchors.
Protocol deltas from the v0.2 grid are frozen here: seq_len=1024,
batch_tokens=2^18 on EVERY arm (including the H100 anchors), recorded as
science-bearing RunSpec fields.

Wall-clock estimates use the measured/planning tok/s model from plan v0.3 §4.
Local runs cost $0 cash; L3 anchor costs use $2.80/H100-h with a $100 hard
cap (enforced independently by the budget ledger).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from logos.config import ALL_PRECISIONS, RunSpec, make_model
from logos.manifest.schema import save_manifest

SEQ = 1024
BATCH = 262_144  # 2^18
H100_USD_H = 2.80

# tok/s planning model (plan v0.3 §4) per (size, gpu).
TOKPS = {
    ("3m", "3080"): 35_000, ("6m", "3080"): 30_000, ("12m", "3080"): 22_000,
    ("25m", "3080"): 15_000,
    ("3m", "5090"): 120_000, ("6m", "5090"): 105_000, ("12m", "5090"): 75_000,
    ("25m", "5090"): 52_000, ("60m", "5090"): 28_000,
    # H100: MFU-derived (~30% of 989 TFLOPs over 6*(N + d*V) FLOPs/token),
    # consistent with plan v0.2 s12's 250m@320x ~= 95 GPU-h figure.
    ("60m", "h100"): 500_000, ("125m", "h100"): 280_000, ("250m", "h100"): 150_000,
}

QUANT_PENALTY = 0.75  # quantized arms run ~25% slower (measured on the 3080)


def _spec(
    size: str, prec, tpp: float, seed: int, gpu: str, phase: str, tags: list[str],
    **kw,
) -> RunSpec:
    n = make_model(size).n_nonemb
    total = kw.pop("total_tokens", None) or int(tpp * n)
    tokps = TOKPS[(size, gpu)] * (QUANT_PENALTY if prec.value != "bf16" else 1.0)
    hours = total / tokps / 3600
    rid = kw.pop("run_id", None) or f"l-{size}-{prec.value}-tp{tpp:g}-s{seed}"
    return RunSpec(
        run_id=rid,
        phase=phase,
        size=size,
        precision=prec.value,
        tokens_per_param=round(total / n, 4),
        seed=seed,
        total_tokens=total,
        seq_len=SEQ,
        batch_tokens=BATCH,
        est_gpu_hours=round(hours, 2),
        est_cost_usd=round(hours * H100_USD_H, 2) if gpu == "h100" else 0.0,
        tags=[f"gpu-{gpu}", *tags],
        **kw,
    )


from logos.config import Precision  # noqa: E402

P158, P2, P3, P4, PBF = (
    Precision.W1_58, Precision.W2, Precision.W3, Precision.W4, Precision.BF16,
)
PAIR = [P158, PBF]
TRIO = [P158, P4, PBF]


def gen_l0() -> list[RunSpec]:
    """Replication + noise floor on the 3080 (28 runs)."""
    runs = []
    for prec in PAIR:
        for tpp in (20, 80, 320):
            for seed in (0, 1, 2):
                runs.append(_spec("3m", prec, tpp, seed, "3080", "l0", ["replication"]))
        for tpp in (20, 80):
            for seed in (0, 1):
                runs.append(_spec("6m", prec, tpp, seed, "3080", "l0", ["replication"]))
        runs.append(_spec("6m", prec, 320, 0, "3080", "l0", ["replication"]))
    return runs


def gen_l1() -> list[RunSpec]:
    """Gap dynamics + protocol lock (30 runs: 15 LR probes + 2 FFN + 13 gap)."""
    runs = []
    for prec in ALL_PRECISIONS:
        for mult in (1.0, 2.0, 4.0):
            runs.append(
                _spec(
                    "6m", prec, 31.1, 0, "3080", "l1", ["lr_probe"],
                    run_id=f"l-lrp-6m-{prec.value}-x{mult:g}",
                    total_tokens=200_000_000,
                    lr_mult_override=mult,
                )
            )
    # FFN ablation: sq_relu arms only; the swiglu side IS the gap study rows
    # 12m-1.58-tp{20,80}-s0 (documented reuse, no duplicate ids).
    for tpp in (20, 80):
        runs.append(
            _spec(
                "12m", P158, tpp, 0, "3080", "l1", ["ffn_ablation"],
                run_id=f"l-12m-1.58-sqrelu-tp{tpp}-s0",
                ffn_type="sq_relu",
            )
        )
    for prec in ALL_PRECISIONS:
        for tpp in (20, 80):
            runs.append(_spec("12m", prec, tpp, 0, "5090", "l1", ["gap_study"]))
    for prec in TRIO:
        runs.append(_spec("12m", prec, 320, 0, "5090", "l1", ["gap_study"]))
    return runs


def gen_l1ctl() -> list[RunSpec]:
    """LR-confound control arms (kimi3 review F1): fully cross precision x LR
    at the crossover cells. bf16 at the ternary 2x multiplier ({3m,6m} x
    {20x,80x} x 2 seeds = 8 runs) and ternary at 1x ({3m} x {20x,80x} x 2
    seeds = 4 runs). If bf16@2xLR closes the 20x gap, the headline changes.
    ~15 GPU-h on the 3080; runs before any gap result leaves the repo."""
    runs = []
    for size in ("3m", "6m"):
        for tpp in (20, 80):
            for seed in (0, 1):
                runs.append(
                    _spec(
                        size, PBF, tpp, seed, "3080", "l1ctl", ["lr_control"],
                        run_id=f"l-ctl-{size}-bf16-x2-tp{tpp}-s{seed}",
                        lr_mult_override=2.0,
                    )
                )
    for tpp in (20, 80):
        for seed in (0, 1):
            runs.append(
                _spec(
                    "3m", P158, tpp, seed, "3080", "l1ctl", ["lr_control"],
                    run_id=f"l-ctl-3m-1.58-x1-tp{tpp}-s{seed}",
                    lr_mult_override=1.0,
                )
            )
    return runs


def gen_l2() -> list[RunSpec]:
    """The grid on the 5090 — new rows only; reused L0/L1 rows are listed in
    meta (39 new runs)."""
    runs = []
    reused: list[str] = []
    for size in ("3m", "6m", "12m", "25m"):
        for prec in ALL_PRECISIONS:
            for tpp in (20, 80):
                rid = f"l-{size}-{prec.value}-tp{tpp}-s0"
                if size in ("3m", "6m") and prec in PAIR:
                    reused.append(rid)  # L0
                elif size == "12m":
                    reused.append(rid)  # L1 gap study
                else:
                    runs.append(_spec(size, prec, tpp, 0, "5090", "l2", ["grid"]))
    for size in ("3m", "6m"):
        for prec in ALL_PRECISIONS:
            rid = f"l-{size}-{prec.value}-tp320-s0"
            if prec in PAIR:
                reused.append(rid)  # L0
            else:
                runs.append(_spec(size, prec, 320, 0, "5090", "l2", ["grid"]))
    for prec in TRIO:
        reused.append(f"l-12m-{prec.value}-tp320-s0")  # L1
        runs.append(_spec("25m", prec, 320, 0, "5090", "l2", ["grid"]))
    for prec in ALL_PRECISIONS:
        runs.append(_spec("60m", prec, 20, 0, "5090", "l2", ["grid"]))
    for prec in TRIO:
        runs.append(_spec("60m", prec, 80, 0, "5090", "l2", ["grid"]))
    gen_l2.reused = reused  # type: ignore[attr-defined]
    return runs


def gen_l3() -> list[RunSpec]:
    """Blind extrapolation anchors on RunPod (5 runs, ~$92 on-demand, hard
    cap $100; spot pricing gives ~40% margin). Covers one size step up at
    BOTH D/N points for the headline pair plus 4-bit at 20x. The 250m pair
    is a stretch goal recorded in meta, launched only if spot savings or
    extra budget materialize. PROTOCOL: the fitted law's predictions are
    committed to the repo BEFORE any of these launch (v0.2 §9)."""
    runs = []
    for prec in PAIR:
        for tpp in (20, 80):
            runs.append(_spec("125m", prec, tpp, 0, "h100", "l3", ["anchor", "blind"]))
    runs.append(_spec("125m", P4, 20, 0, "h100", "l3", ["anchor", "blind"]))
    return runs


def gen_l4() -> list[RunSpec]:
    """Native KV-QAT at 12m (4 runs); the post-hoc KV sweep is eval-only."""
    runs = []
    for kv in (8, 4):
        for prec in PAIR:
            runs.append(
                _spec(
                    "12m", prec, 80, 0, "5090", "l4", ["kv_qat"],
                    run_id=f"l-12m-{prec.value}-kv{kv}-tp80-s0",
                    kv_qat_bits=kv,
                )
            )
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("manifests"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    phases = {
        "l0": (gen_l0(), {"purpose": "replication + noise floor (RQ1)", "gpu": "3080"}),
        "l1": (gen_l1(), {"purpose": "gap dynamics + protocol lock (RQ2)", "gpu": "3080/5090"}),
        "l1ctl": (
            gen_l1ctl(),
            {
                "purpose": "LR-confound controls (kimi3 F1): bf16@2xLR + ternary@1xLR "
                "at the L0 crossover cells",
                "gpu": "3080",
                "data_dir": "data/fineweb_edu_2p5b_view",
                "data_note": "MUST train on the frozen 2.5B view (hardlinked "
                "snapshot) so controls share the exact corpus + window "
                "permutation universe of the L0 cells they compare against; "
                "the live dir is being extended for L1+.",
            },
        ),
        "l2": (gen_l2(), {"purpose": "the grid (RQ3)", "gpu": "5090", "reused_runs": gen_l2.reused}),  # type: ignore[attr-defined]
        "l3": (
            gen_l3(),
            {
                "purpose": "blind extrapolation anchors (RQ4)",
                "gpu": "runpod-1xH100",
                "hard_cap_usd": 100,
                "prediction_protocol": "freeze fitted law + commit predictions before launch",
                "stretch_anchors": [
                    "250m-1.58-tp20-s0 (~19 H100-h)",
                    "250m-bf16-tp20-s0 (~14 H100-h)",
                    "launch only with spot savings or budget beyond $100",
                ],
            },
        ),
        "l4": (
            gen_l4(),
            {
                "purpose": "KV in the budget (RQ5, reduced)",
                "gpu": "5090",
                "eval_jobs": {
                    "posthoc_kv_sweep": {
                        "checkpoints": "runs/**/ckpt_latest.pt for sizes 12m,25m,60m",
                        "kv_bits": [8, 4, 3, 2],
                        "evals": ["bpb_val1", "bpb_val2", "ruler_lite_1k_8k"],
                    }
                },
            },
        ),
        "l5": (
            [],
            {
                "purpose": "capstone-lite (RQ6): the law-prescribed config at a "
                "64MB body budget; RunSpec generated by fitting.prescribe."
                "optimal_config after L2 lands — never by taste",
                "gpu": "5090",
                "baselines": [
                    "bf16 at equal body bytes",
                    "4-bit PTQ of a ~2x bf16 model",
                    "native 4-bit QAT arm",
                ],
            },
        ),
    }
    lines = ["| Phase | Runs | GPU-h (local) | H100-h | Cash |", "|---|---|---|---|---|"]
    for name, (runs, meta) in phases.items():
        save_manifest(runs, args.out / f"{name}.yaml", meta={"phase": name, **meta})
        local_h = sum(r.est_gpu_hours for r in runs if "gpu-h100" not in r.tags)
        h100_h = sum(r.est_gpu_hours for r in runs if "gpu-h100" in r.tags)
        cash = sum(r.est_cost_usd for r in runs)
        lines.append(f"| {name} | {len(runs)} | {local_h:.0f} | {h100_h:.1f} | ${cash:.0f} |")
        print(lines[-1])
    (args.out / "summary_local.md").write_text(
        "# LOGOS-Local run budget (generated)\n\n" + "\n".join(lines) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
