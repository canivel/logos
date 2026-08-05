"""Core configuration contract for LOGOS.

Everything else in the project (trainer, quantizers, manifests, fitting,
validation panel) builds against the types in this module. Design principles
(PLAN.md section 3): one variable moves at a time; N means non-embedding
parameters; same data order everywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Precision(str, Enum):
    """Weight precision arms. The only thing that varies between arms at a
    given (N, D) is the linear-layer precision (plus its subln norm and the
    LR multiplier the P1 protocol assigns to it)."""

    BF16 = "bf16"
    W4 = "4"
    W3 = "3"
    W2 = "2"
    W1_58 = "1.58"

    @property
    def bits(self) -> float:
        return {"bf16": 16.0, "4": 4.0, "3": 3.0, "2": 2.0, "1.58": 1.58}[self.value]

    @property
    def is_quantized(self) -> bool:
        return self is not Precision.BF16

    @property
    def int_levels(self) -> int:
        """Number of representable levels for integer arms (ternary = 3)."""
        return {"4": 16, "3": 8, "2": 4, "1.58": 3}[self.value] if self.is_quantized else 0


ALL_PRECISIONS = [Precision.W1_58, Precision.W2, Precision.W3, Precision.W4, Precision.BF16]


def _round_to(x: float, multiple: int) -> int:
    return int(round(x / multiple) * multiple)


@dataclass(frozen=True)
class ModelConfig:
    """Llama-style architecture (PLAN.md section 4): RMSNorm pre-norm, RoPE,
    GQA 4:1, tied 32k embeddings, depth-leaning shapes, SwiGLU FFN
    (P1 ablates vs squared-ReLU then freezes), subln where BitLinear needs it.
    """

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    # 32768 = Mistral-7B-v0.1's 32,000-token BPE padded to the next power of
    # two for GPU efficiency; the top 768 rows are unused. Report both counts
    # (paper: "fixed 32k tokenizer", PLAN.md principle 2).
    vocab_size: int = 32768
    head_dim: int = 0  # 0 -> d_model // n_heads
    ffn_type: str = "swiglu"  # "swiglu" | "sq_relu" (P1 ablation)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    max_seq_len: int = 2048
    precision: Precision = Precision.BF16
    act_bits: int = 8  # WxA8 on all low-bit arms; ignored for bf16
    weight_group_size: int = 128  # per-group int quant for 2-4 bit arms
    use_qk_norm: bool = False  # stability ladder rung for ternary at 250M+ (PLAN.md s15)

    def __post_init__(self):
        if self.head_dim == 0:
            object.__setattr__(self, "head_dim", self.d_model // self.n_heads)
        assert self.n_heads % self.n_kv_heads == 0, "GQA requires n_heads % n_kv_heads == 0"

    # ---- parameter accounting (fit laws on non-embedding N; report both) ----

    @property
    def attn_params_per_layer(self) -> int:
        qo = 2 * self.d_model * self.n_heads * self.head_dim
        kv = 2 * self.d_model * self.n_kv_heads * self.head_dim
        return qo + kv

    @property
    def ffn_params_per_layer(self) -> int:
        mult = 3 if self.ffn_type == "swiglu" else 2
        return mult * self.d_model * self.ffn_hidden

    @property
    def n_nonemb(self) -> int:
        return self.n_layers * (self.attn_params_per_layer + self.ffn_params_per_layer)

    @property
    def n_emb(self) -> int:
        n = self.vocab_size * self.d_model
        return n if self.tie_embeddings else 2 * n

    @property
    def n_total(self) -> int:
        return self.n_nonemb + self.n_emb

    def weight_bytes(self, precision: Precision | None = None) -> int:
        """Theoretical packed weight bytes M_w = N * b_w / 8 for the quantized
        body plus bf16 embeddings/norms. Real GGUF sizes (with packing
        overhead) are measured, not derived — see export/ and PLAN.md s8."""
        p = precision or self.precision
        body = int(self.n_nonemb * p.bits / 8)
        emb = self.n_emb * 2  # embeddings stay bf16 in every arm
        return body + emb

    def kv_bytes(self, context_len: int, kv_bits: float = 16.0, batch: int = 1) -> int:
        """KV_bytes = 2 x layers x kv_heads x head_dim x context x (b_kv/8) x batch."""
        return int(
            2 * self.n_layers * self.n_kv_heads * self.head_dim * context_len * (kv_bits / 8) * batch
        )


def make_model(
    size: str,
    precision: Precision = Precision.BF16,
    ffn_type: str = "swiglu",
    **overrides: Any,
) -> ModelConfig:
    """Build a ladder model. FFN hidden is sized so total non-emb params land
    on ~12 * L * d^2 (the ladder targets), i.e. 3*d*h ~= 9.5*d^2 for SwiGLU
    after GQA-4:1 attention takes 2.5*d^2."""
    shape = LADDER[size]
    d, layers, heads, kv = shape["d_model"], shape["n_layers"], shape["n_heads"], shape["n_kv_heads"]
    head_dim = shape.get("head_dim", d // heads)
    attn = 2 * d * heads * head_dim + 2 * d * kv * head_dim
    target_per_layer = 12 * d * d
    mult = 3 if ffn_type == "swiglu" else 2
    ffn_hidden = _round_to((target_per_layer - attn) / (mult * d), 64)
    cfg = dict(
        d_model=d,
        n_layers=layers,
        n_heads=heads,
        n_kv_heads=kv,
        head_dim=head_dim,
        ffn_hidden=ffn_hidden,
        precision=precision,
        ffn_type=ffn_type,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


# Model shape ladder (PLAN.md section 4). head_dim 64-128, GQA 4:1 throughout.
# "micro" (~4.7M) is not part of the law-fitting grid: it exists for local
# smoke experiments and the validation panel's end-to-end probes.
LADDER: dict[str, dict[str, int]] = {
    "micro": dict(d_model=256, n_layers=6, n_heads=4, n_kv_heads=1),
    "25m": dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=2),
    "60m": dict(d_model=640, n_layers=12, n_heads=8, n_kv_heads=2, head_dim=80),
    "125m": dict(d_model=768, n_layers=18, n_heads=12, n_kv_heads=3),
    "250m": dict(d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4),
    "490m": dict(d_model=1280, n_layers=25, n_heads=20, n_kv_heads=5),
    "1b": dict(d_model=1664, n_layers=30, n_heads=16, n_kv_heads=4, head_dim=104),
    "1.5b": dict(d_model=2048, n_layers=30, n_heads=16, n_kv_heads=4, head_dim=128),
}


@dataclass(frozen=True)
class TrainConfig:
    """Frozen across all arms at a given size except lr (per-precision rule)."""

    lr: float
    total_tokens: int
    batch_tokens: int = 524288  # 2^19 tokens/step at small sizes
    seq_len: int = 2048
    warmup_tokens: int = 0  # 0 -> min(1% of total, 250M tokens)
    schedule: str = "cosine"  # single-stage cosine for the whole grid (principle 6)
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    seed: int = 0  # weight-init seed; data order uses data_seed (same everywhere)
    data_seed: int = 1337  # FIXED for the whole project: same data order in every arm
    checkpoint_interval_s: int = 1800  # 30 min + on SIGTERM (spot hardening)
    eval_interval_tokens: int = 0  # 0 -> 20 evals per run
    dtype: str = "bfloat16"
    compile: bool = False
    # Hardware-fitting lever that does NOT change the science: sequences per
    # micro-batch for gradient accumulation. 0 -> whole batch in one forward.
    # batch_tokens stays frozen across arms (principle 6); only this varies
    # with GPU memory.
    micro_batch_seqs: int = 0
    # Divergence kill criterion (PLAN.md s7 manifest rules, Tier-2 enforced).
    divergence_margin: float = 2.0  # nats over best before a step counts as bad
    divergence_patience: int = 100  # consecutive bad steps before the kill

    @property
    def warmup(self) -> int:
        return self.warmup_tokens or min(int(0.01 * self.total_tokens), 250_000_000)


# Per-precision LR multipliers. P0 defaults follow BitNet ("expect ternary ~2x
# bf16; measure, don't assume"). The P1 LR-transfer probes at 60M REPLACE these
# and the result is frozen in manifests/lr_rules.yaml for the rest of the project.
DEFAULT_LR_MULT: dict[Precision, float] = {
    Precision.BF16: 1.0,
    Precision.W4: 1.25,
    Precision.W3: 1.5,
    Precision.W2: 2.0,
    Precision.W1_58: 2.0,
}

# Base LR by ladder size (bf16 arm), ~mup-informed sqrt-width scaling from 3e-3@512.
BASE_LR: dict[str, float] = {
    "micro": 3.0e-3,
    "25m": 3.0e-3,
    "60m": 2.7e-3,
    "125m": 2.4e-3,
    "250m": 2.1e-3,
    "490m": 1.9e-3,
    "1b": 1.6e-3,
    "1.5b": 1.5e-3,
}


@dataclass
class RunSpec:
    """One row of a versioned run manifest. Nothing is launched by hand twice."""

    run_id: str
    phase: str  # p0 | p1 | p2 | p2ext | p3 | p4 | p5
    size: str  # ladder key
    precision: str  # Precision value
    tokens_per_param: float
    seed: int = 0
    ffn_type: str = "swiglu"
    lr: float | None = None  # None -> BASE_LR[size] * lr_mult(precision)
    lr_mult_override: float | None = None
    total_tokens: int | None = None  # None -> tokens_per_param * n_nonemb
    kv_qat_bits: int | None = None  # P4 native KV-QAT arms
    gqa_ratio: int = 4  # P4 GQA-ratio arm uses 8
    tags: list[str] = field(default_factory=list)
    est_gpu_hours: float = 0.0
    est_cost_usd: float = 0.0
    max_wall_clock_mult: float = 3.0  # kill criterion: >3x scheduled wall clock
    notes: str = ""

    def model_config(self) -> ModelConfig:
        over: dict[str, Any] = {}
        if self.gqa_ratio != 4:
            shape = LADDER[self.size]
            over["n_kv_heads"] = max(1, shape["n_heads"] // self.gqa_ratio)
        return make_model(self.size, Precision(self.precision), self.ffn_type, **over)

    def train_config(self, lr_rules: dict[str, float] | None = None) -> TrainConfig:
        m = self.model_config()
        total = self.total_tokens or int(self.tokens_per_param * m.n_nonemb)
        if self.lr is not None:
            lr = self.lr
        else:
            mult = (
                self.lr_mult_override
                if self.lr_mult_override is not None
                else (lr_rules or {p.value: DEFAULT_LR_MULT[p] for p in ALL_PRECISIONS})[
                    self.precision
                ]
            )
            lr = BASE_LR[self.size] * mult
        return TrainConfig(lr=lr, total_tokens=total, seed=self.seed)

    # Fields that affect the science of a run. Cost estimates, tags, notes,
    # ops knobs AND phase are excluded: re-estimating budgets never changes
    # the hash, and a P1 gap-study run re-emitted in the P2 grid (the
    # documented reuse, PLAN.md s6) keeps one identity.
    _SCIENCE_FIELDS = (
        "run_id",
        "size",
        "precision",
        "tokens_per_param",
        "seed",
        "ffn_type",
        "lr",
        "lr_mult_override",
        "total_tokens",
        "kv_qat_bits",
        "gqa_ratio",
    )

    def config_hash(self) -> str:
        """Content hash of the science-bearing fields, logged with every
        result row; the validation panel cross-checks results against
        manifests through this."""
        d = asdict(self)
        payload = json.dumps(
            {k: d[k] for k in self._SCIENCE_FIELDS}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
