"""Trainer tests: schedule shape, exact resume, param groups, divergence.

All CPU/fp32 on synthetic data (network-free, PLAN.md s5 spot-hardening is
what the resume test verifies).
"""

from __future__ import annotations

import json

import pytest
import torch

from logos.config import ModelConfig, Precision, RunSpec, TrainConfig
from logos.data.prepare import prepare_synthetic
from logos.model.transformer import Transformer
from logos.train.schedule import lr_at, wd_at
from logos.train.trainer import TrainerExtras, build_param_groups, train

VOCAB = 256
SEQ = 32
BATCH_TOKENS = 4 * SEQ  # 4 sequences/step


def tiny_mcfg(precision: Precision = Precision.BF16) -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=64,
        vocab_size=VOCAB,
        max_seq_len=64,
        precision=precision,
    )


def tiny_tcfg(**kw) -> TrainConfig:
    base = dict(
        lr=1e-3,
        total_tokens=20 * BATCH_TOKENS,
        batch_tokens=BATCH_TOKENS,
        seq_len=SEQ,
        warmup_tokens=2 * BATCH_TOKENS,
        dtype="float32",
        checkpoint_interval_s=10**9,  # only end-of-run / interrupt saves
    )
    base.update(kw)
    return TrainConfig(**base)


def spec(run_id: str = "t") -> RunSpec:
    return RunSpec(run_id=run_id, phase="p0", size="25m", precision="bf16", tokens_per_param=1.0)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    prepare_synthetic(d, n_shards=1, shard_tokens=20_000, vocab_size=VOCAB, val_tokens=2_048)
    return d


# ---------------------------------------------------------------------------
# (1) cosine schedule shape
# ---------------------------------------------------------------------------


def test_cosine_schedule_shape():
    cfg = TrainConfig(lr=2e-3, total_tokens=1_000_000, warmup_tokens=100_000, min_lr_ratio=0.1)
    assert lr_at(cfg, 0) == 0.0
    assert lr_at(cfg, 50_000) == pytest.approx(1e-3)  # linear warmup
    assert lr_at(cfg, 100_000) == pytest.approx(cfg.lr)  # peak at end of warmup
    assert lr_at(cfg, 1_000_000) == pytest.approx(cfg.min_lr_ratio * cfg.lr)
    # monotone decreasing after warmup
    pts = [lr_at(cfg, t) for t in range(100_000, 1_000_001, 100_000)]
    assert all(a > b for a, b in zip(pts, pts[1:]))
    # halfway point of cosine = midpoint of peak and floor
    assert lr_at(cfg, 550_000) == pytest.approx(0.5 * (cfg.lr + cfg.min_lr_ratio * cfg.lr))
    # constant weight decay on the grid schedule
    assert wd_at(cfg, 0) == wd_at(cfg, 900_000) == cfg.weight_decay


def test_bitnet2stage_gated():
    cfg = TrainConfig(lr=1e-3, total_tokens=1_000_000, warmup_tokens=10_000, schedule="bitnet2stage")
    # weight decay switches off in stage 2 (capstone-only recipe, principle 6)
    assert wd_at(cfg, 100_000) == cfg.weight_decay
    assert wd_at(cfg, 900_000) == 0.0
    # abrupt LR drop across the stage boundary
    assert lr_at(cfg, 500_000) > lr_at(cfg, 500_001)
    assert lr_at(cfg, 1_000_000) == pytest.approx(cfg.min_lr_ratio * cfg.lr)


# ---------------------------------------------------------------------------
# (2) resume determinism
# ---------------------------------------------------------------------------


def test_resume_determinism(data_dir, tmp_path):
    kw = dict(data_dir=data_dir, device="cpu", log_interval=5)
    mk = lambda **e: TrainerExtras(model_config=tiny_mcfg(), train_config=tiny_tcfg(), **e)

    full = train(spec("full"), run_dir=tmp_path / "full", extras=mk(), **kw)
    assert full["status"] == "complete" and full["step"] == 20

    # interrupted run: stop at step 10 (checkpoint saved), then auto-resume
    part = train(spec("part"), run_dir=tmp_path / "part", extras=mk(max_steps=10), **kw)
    assert part["step"] == 10
    assert (tmp_path / "part" / "ckpt_latest.pt").exists()
    resumed = train(spec("part"), run_dir=tmp_path / "part", extras=mk(), **kw)
    assert resumed["step"] == 20 and resumed["tokens"] == full["tokens"]
    assert resumed["final_loss"] == pytest.approx(full["final_loss"], abs=1e-4)


# ---------------------------------------------------------------------------
# (3) param groups
# ---------------------------------------------------------------------------


def test_param_groups_no_decay_on_norms_embeddings_scales():
    torch.manual_seed(0)
    model = Transformer(tiny_mcfg(Precision.W2))  # W2 -> GroupIntLinear with .scale
    groups = build_param_groups(model, 0.1)
    assert len(groups[0]["params"]) > 0 and len(groups[1]["params"]) > 0
    wd_of = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        no_decay = (
            p.ndim < 2  # norms, subln, biases
            or name.endswith(".scale")  # quantizer scales
            or name in ("embed.weight", "head.weight")  # (tied) embeddings
        )
        assert wd_of[id(p)] == (0.0 if no_decay else 0.1), name
    # every param lands in exactly one group
    assert len(wd_of) == len(seen)


# ---------------------------------------------------------------------------
# (4) divergence detection
# ---------------------------------------------------------------------------


def test_grad_accumulation_is_science_neutral(data_dir, tmp_path):
    """micro_batch_seqs is a hardware lever only: same batch_tokens ->
    same trajectory (CPU fp32; tiny tolerance for summation order)."""
    kw = dict(data_dir=data_dir, device="cpu", log_interval=5)
    full = train(
        spec("acc-full"), run_dir=tmp_path / "af",
        extras=TrainerExtras(model_config=tiny_mcfg(), train_config=tiny_tcfg()), **kw,
    )
    accum = train(
        spec("acc-micro"), run_dir=tmp_path / "am",
        extras=TrainerExtras(
            model_config=tiny_mcfg(), train_config=tiny_tcfg(micro_batch_seqs=1)
        ), **kw,
    )
    assert accum["step"] == full["step"]
    assert accum["final_loss"] == pytest.approx(full["final_loss"], abs=5e-4)


def test_divergence_writes_status(data_dir, tmp_path):
    run_dir = tmp_path / "div"
    extras = TrainerExtras(
        model_config=tiny_mcfg(),
        train_config=tiny_tcfg(lr=1e5, total_tokens=100 * BATCH_TOKENS, warmup_tokens=0),
        divergence_margin=0.5,
        divergence_patience=2,
    )
    with pytest.raises(SystemExit) as ei:
        train(spec("div"), data_dir=data_dir, run_dir=run_dir, device="cpu", extras=extras)
    assert ei.value.code == 1  # non-zero exit
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "diverged"
    assert status["step"] < 100
