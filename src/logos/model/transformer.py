"""Llama-style decoder for the TRIT suite.

RMSNorm pre-norm, RoPE, GQA, tied 32k embeddings, SwiGLU (or squared-ReLU,
the P1 ablation) FFN. When the precision arm is quantized, every linear in
the model body is a fake-quant linear from logos.quant and the architecture
adds subln: an extra RMSNorm on the inputs of o_proj and down_proj, per the
BitNet b1.58 recipe. Embeddings, norms and the (tied) head stay bf16 in every
arm. Optional per-head KV fake-quant supports the P4 native KV-QAT arms.

Architecture, init, and everything else here is frozen across arms
(PLAN.md principle 1); `precision` is the only lever.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from logos.config import ModelConfig, Precision
from logos.quant.factory import build_linear
from logos.quant.ste import ste_round


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 [T, hd/2]


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, hd]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs[None, :, None, :]).flatten(3)
    return out.to(x.dtype)


class KVQuant(nn.Module):
    """Per-head absmax integer fake-quant on K and V (P4 native KV-QAT)."""

    def __init__(self, bits: int):
        super().__init__()
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        s = self.qmax / x32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        q = torch.clamp(ste_round(x32 * s), -self.qmax - 1, self.qmax) / s
        return q.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, kv_qat_bits: int | None = None):
        super().__init__()
        self.cfg = cfg
        hd, H, KV = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        q = cfg.precision
        self.wq = build_linear(cfg.d_model, H * hd, q, cfg.weight_group_size, cfg.act_bits)
        self.wk = build_linear(cfg.d_model, KV * hd, q, cfg.weight_group_size, cfg.act_bits)
        self.wv = build_linear(cfg.d_model, KV * hd, q, cfg.weight_group_size, cfg.act_bits)
        self.wo = build_linear(H * hd, cfg.d_model, q, cfg.weight_group_size, cfg.act_bits)
        self.subln = RMSNorm(H * hd, cfg.norm_eps) if q.is_quantized else None
        self.qk_norm = (
            nn.ModuleList([RMSNorm(hd, cfg.norm_eps), RMSNorm(hd, cfg.norm_eps)])
            if cfg.use_qk_norm
            else None
        )
        self.kv_quant = KVQuant(kv_qat_bits) if kv_qat_bits else None

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        cfg = self.cfg
        q = self.wq(x).view(B, T, cfg.n_heads, cfg.head_dim)
        k = self.wk(x).view(B, T, cfg.n_kv_heads, cfg.head_dim)
        v = self.wv(x).view(B, T, cfg.n_kv_heads, cfg.head_dim)
        if self.qk_norm is not None:
            q, k = self.qk_norm[0](q), self.qk_norm[1](k)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        if self.kv_quant is not None:
            k, v = self.kv_quant(k), self.kv_quant(v)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, hd]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).reshape(B, T, -1)
        if self.subln is not None:
            out = self.subln(out)
        return self.wo(out)


class FFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        q = cfg.precision
        self.ffn_type = cfg.ffn_type
        self.up = build_linear(cfg.d_model, cfg.ffn_hidden, q, cfg.weight_group_size, cfg.act_bits)
        self.down = build_linear(
            cfg.ffn_hidden, cfg.d_model, q, cfg.weight_group_size, cfg.act_bits
        )
        self.gate = (
            build_linear(cfg.d_model, cfg.ffn_hidden, q, cfg.weight_group_size, cfg.act_bits)
            if cfg.ffn_type == "swiglu"
            else None
        )
        self.subln = RMSNorm(cfg.ffn_hidden, cfg.norm_eps) if q.is_quantized else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            h = F.silu(self.gate(x)) * self.up(x)
        else:  # squared ReLU
            h = F.relu(self.up(x)).pow(2)
        if self.subln is not None:
            h = self.subln(h)
        return self.down(h)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kv_qat_bits: int | None = None):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, kv_qat_bits)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = FFN(cfg)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, kv_qat_bits: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, kv_qat_bits) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        freqs = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_freqs", freqs, persistent=False)
        self.apply(self._init)
        # Scaled residual-branch init (GPT-2/Llama practice): out projections
        # down-scaled by sqrt(2 * n_layers).
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = tokens.shape
        freqs = self.rope_freqs[:T]
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x, freqs)
        x = self.final_norm(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss

    # ---- accounting helpers used by results logging and the validation panel ----

    def n_params(self) -> dict[str, int]:
        emb = self.embed.weight.numel()
        scales = sum(p.numel() for n, p in self.named_parameters() if n.endswith(".scale"))
        norms = sum(
            p.numel() for n, p in self.named_parameters() if "norm" in n or n.endswith("subln.weight")
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "emb": emb,
            "nonemb": total - emb - scales - norms,
            "norms": norms,
            "quant_scales": scales,
            "total": total,
        }
