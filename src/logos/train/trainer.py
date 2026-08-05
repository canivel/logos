"""The training loop (PLAN.md s5: a trainer we trust).

Spot-instance hardening: atomic checkpoints every checkpoint_interval_s
seconds AND on SIGTERM/SIGINT (SIGBREAK where available; Windows may never
deliver SIGTERM), plus auto-resume from ckpt_latest.pt to the exact step with
the loader skipping deterministically. Divergence kill criterion per the run
manifest rules (PLAN.md s7). W&B is optional and never required.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from logos.config import ModelConfig, RunSpec, TrainConfig
from logos.data.loader import TokenLoader
from logos.model.transformer import Transformer
from logos.train.schedule import lr_at, wd_at

CKPT_NAME = "ckpt_latest.pt"

# bf16 peak FLOPs by device-name substring; anything else -> mfu null.
PEAK_FLOPS_BY_NAME = {"H100": 989e12, "3080": 59.5e12}


def detect_peak_flops(device: str) -> float | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(torch.device(device))
    for key, flops in PEAK_FLOPS_BY_NAME.items():
        if key in name:
            return flops
    return None


@dataclass
class TrainerExtras:
    """Knobs that are not part of the frozen TrainConfig contract."""

    eval_fn: Callable[[torch.nn.Module, int], Any] | None = None
    model_config: ModelConfig | None = None  # override RunSpec.model_config() (smoke/tests)
    train_config: TrainConfig | None = None  # override RunSpec.train_config() (smoke/tests)
    max_steps: int | None = None  # cap steps without touching the schedule
    peak_flops: float | None = None  # override MFU denominator
    # None -> use TrainConfig.divergence_margin / .divergence_patience (the
    # manifest-controlled values); set here only for tests/smoke overrides.
    divergence_margin: float | None = None
    divergence_patience: int | None = None


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """AdamW groups: decay on 2-D weight tensors only; norms (1-D),
    embeddings (incl. the tied head), and quantizer `.scale` params get 0.
    Tied tensors are deduplicated by identity."""
    seen: set[int] = set()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_scale = name.endswith(".scale")
        is_emb = name in ("embed.weight", "head.weight")
        (decay if p.ndim >= 2 and not is_scale and not is_emb else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "use_decay": True},
        {"params": no_decay, "weight_decay": 0.0, "use_decay": False},
    ]


def _write_status(run_dir: Path, **payload: Any) -> dict:
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, run_dir / "status.json")
    return payload


def train(
    run_spec: RunSpec,
    data_dir: str | Path,
    run_dir: str | Path,
    lr_rules: dict | None = None,
    device: str | None = None,
    batch_tokens_override: int | None = None,
    log_interval: int = 10,
    extras: TrainerExtras | None = None,
) -> dict:
    """Run one arm. Returns the final status dict; raises SystemExit(1) on
    divergence and SystemExit(130) on an interrupt-triggered checkpoint."""
    ex = extras or TrainerExtras()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = ex.train_config or run_spec.train_config(lr_rules)
    if batch_tokens_override:
        cfg = dataclasses.replace(cfg, batch_tokens=batch_tokens_override)
    assert cfg.batch_tokens % cfg.seq_len == 0, "batch_tokens must divide by seq_len"
    mcfg = ex.model_config or run_spec.model_config()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = str(device).startswith("cuda")
    use_autocast = on_cuda and cfg.dtype == "bfloat16"

    # Deterministic init from the weight seed (data order is data_seed's job).
    torch.manual_seed(cfg.seed)
    model = Transformer(mcfg, kv_qat_bits=run_spec.kv_qat_bits).to(device)
    opt = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )
    step_model = torch.compile(model) if cfg.compile else model

    loader = TokenLoader(
        data_dir,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_tokens // cfg.seq_len,
        data_seed=cfg.data_seed,
    )

    # ---- auto-resume (spot hardening) ----
    step, tokens, prev_wall = 0, 0, 0.0
    ckpt_path = run_dir / CKPT_NAME
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, tokens = state["step"], state["tokens"]
        prev_wall = state.get("wall_s", 0.0)
        torch.set_rng_state(state["rng"]["torch"])
        if on_cuda and state["rng"].get("cuda") is not None:
            try:
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
            except RuntimeError:
                pass  # resumed on a different GPU count; init RNG stands

    total_steps = math.ceil(cfg.total_tokens / cfg.batch_tokens)
    if ex.max_steps is not None:
        total_steps = min(total_steps, ex.max_steps)
    eval_every = cfg.eval_interval_tokens or max(1, cfg.total_tokens // 20)
    next_eval = (tokens // eval_every + 1) * eval_every
    peak = ex.peak_flops if ex.peak_flops is not None else detect_peak_flops(device)
    n_params = model.n_params()["total"]

    def save_ckpt() -> None:
        rng: dict[str, Any] = {"torch": torch.get_rng_state(), "cuda": None}
        if on_cuda:
            rng["cuda"] = torch.cuda.get_rng_state_all()
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "tokens": tokens,
            "wall_s": prev_wall + (time.time() - t_start),
            "rng": rng,
            "run_id": run_spec.run_id,
            "config_hash": run_spec.config_hash(),
        }
        tmp = run_dir / (CKPT_NAME + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)

    # ---- optional W&B (guarded; never required) ----
    wandb_run = None
    if os.environ.get("LOGOS_WANDB_PROJECT"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=os.environ["LOGOS_WANDB_PROJECT"],
                name=run_spec.run_id,
                config={**dataclasses.asdict(run_spec), **dataclasses.asdict(cfg)},
                resume="allow",
            )
        except Exception:
            wandb_run = None

    # ---- signals: checkpoint on SIGINT/SIGTERM (+SIGBREAK on Windows) ----
    stop = {"flag": False}
    installed: list[tuple[int, Any]] = []
    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            installed.append((sig, signal.signal(sig, lambda *_: stop.update(flag=True))))
        except (ValueError, OSError):
            pass  # not in the main thread / unsupported

    div_margin = ex.divergence_margin if ex.divergence_margin is not None else cfg.divergence_margin
    div_patience = (
        ex.divergence_patience if ex.divergence_patience is not None else cfg.divergence_patience
    )
    batch_seqs = cfg.batch_tokens // cfg.seq_len
    micro_seqs = cfg.micro_batch_seqs or batch_seqs
    n_micro = math.ceil(batch_seqs / micro_seqs)

    metrics_f = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    best, bad, loss_val = float("inf"), 0, float("nan")
    t_start = time.time()
    last_ckpt = time.time()
    batches = loader.iter_batches(step)

    def diverged() -> None:
        save_ckpt()
        _write_status(
            run_dir, status="diverged", final_loss=loss_val, step=step, tokens=tokens,
            wall_s=prev_wall + (time.time() - t_start),
        )
        raise SystemExit(1)

    try:
        while step < total_steps:
            t0 = time.time()
            batch = next(batches)
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            # Schedule from the token count at the END of this step.
            lr = lr_at(cfg, tokens + cfg.batch_tokens)
            wd = wd_at(cfg, tokens + cfg.batch_tokens)
            for g in opt.param_groups:
                g["lr"] = lr
                if g["use_decay"]:
                    g["weight_decay"] = wd
            # Gradient accumulation over micro-batches: identical effective
            # batch (batch_tokens) regardless of GPU memory.
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for i in range(n_micro):
                mi = inputs[i * micro_seqs : (i + 1) * micro_seqs]
                mt = targets[i * micro_seqs : (i + 1) * micro_seqs]
                if mi.numel() == 0:
                    continue
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
                ):
                    _, loss = step_model(mi, mt)
                (loss * (mi.shape[0] / batch_seqs)).backward()
                loss_sum += loss.item() * (mi.shape[0] / batch_seqs)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            tokens += cfg.batch_tokens
            loss_val = loss_sum
            dt = time.time() - t0

            # Divergence kill criterion (manifest rule, PLAN.md s7).
            if not math.isfinite(loss_val):
                diverged()
            if loss_val < best:
                best, bad = loss_val, 0
            elif loss_val > best + div_margin:
                bad += 1
                if bad >= div_patience:
                    diverged()
            else:
                bad = 0

            if step % log_interval == 0 or step == total_steps:
                mfu = (6 * n_params * cfg.batch_tokens) / (dt * peak) if peak and dt > 0 else None
                row = {
                    "step": step,
                    "tokens": tokens,
                    "loss": loss_val,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "dt": dt,
                    "mfu": mfu,
                }
                metrics_f.write(json.dumps(row) + "\n")
                metrics_f.flush()
                if wandb_run is not None:
                    wandb_run.log(row, step=step)

            if ex.eval_fn is not None and tokens >= next_eval:
                ex.eval_fn(model, step)
                next_eval += eval_every

            if stop["flag"] or (time.time() - last_ckpt) >= cfg.checkpoint_interval_s:
                save_ckpt()
                last_ckpt = time.time()
                if stop["flag"]:
                    _write_status(
                        run_dir, status="interrupted", final_loss=loss_val, step=step,
                        tokens=tokens, wall_s=prev_wall + (time.time() - t_start),
                    )
                    raise SystemExit(130)

        save_ckpt()
        return _write_status(
            run_dir, status="complete", final_loss=loss_val, step=step, tokens=tokens,
            wall_s=prev_wall + (time.time() - t_start),
        )
    finally:
        metrics_f.close()
        for sig, prev in installed:
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
