"""Reference packed weight formats + measured footprint (PLAN.md s4, s8).

Ternary: i2_s-style 2-bit packing — codes {-1,0,1} stored as unsigned {0,1,2},
4 codes/byte little-endian (code i lives at bits 2i..2i+1 of byte i//4), one
fp32 gamma per tensor. Int group (2/3/4-bit): codes bit-packed little-endian
at exact width (2b: 4/byte, 3b: 8 codes in 3 bytes, 4b: 2/byte) plus fp32
per-group scales. pack/unpack are pure numpy with exact roundtrip.

`measured_bytes` computes the REAL packed footprint of a checkpoint — this
number, not N*b/8 theory, feeds the memory transform in fitting (PLAN.md s8:
"measured packing overhead per format, taken from real sizes, not theory").
"""

from __future__ import annotations

import numpy as np
import torch

from logos.config import Precision
from logos.quant.bitlinear import BitLinear
from logos.quant.paretoq import GroupIntLinear

FMT_BY_PRECISION = {
    "1.58": "i2_s",
    "2": "gint2",
    "3": "gint3",
    "4": "gint4",
    "bf16": "bf16",
}


def _as_numpy(codes) -> np.ndarray:
    if hasattr(codes, "detach"):  # torch tensor without importing torch here
        codes = codes.detach().cpu().numpy()
    return np.ascontiguousarray(codes)


def _pack_bits(vals: np.ndarray, bits: int) -> bytes:
    """Little-endian bitstream: value i occupies bits [i*bits, (i+1)*bits)."""
    vals = vals.astype(np.uint8).ravel()
    bit_arr = ((vals[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(bit_arr.ravel(), bitorder="little").tobytes()


def _unpack_bits(raw: bytes, bits: int, n: int) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    bit_arr = np.unpackbits(buf, bitorder="little")[: n * bits].reshape(n, bits)
    return (bit_arr << np.arange(bits, dtype=np.uint8)).sum(axis=1).astype(np.uint8)


# ---- ternary (i2_s-style) ----


def pack_ternary(codes) -> bytes:
    """codes {-1,0,1} -> 2-bit stream storing {0,1,2}, 4 codes/byte."""
    c = _as_numpy(codes).astype(np.int8)
    assert np.abs(c).max(initial=0) <= 1, "ternary codes must be in {-1,0,1}"
    return _pack_bits((c + 1).astype(np.uint8), 2)


def unpack_ternary(raw: bytes, shape) -> np.ndarray:
    """Inverse of pack_ternary -> int8 codes in {-1,0,1} with given shape."""
    n = int(np.prod(shape)) if len(shape) else 1
    vals = _unpack_bits(raw, 2, n)
    return (vals.astype(np.int8) - 1).reshape(shape)


# ---- per-group int (2/3/4-bit) ----


def pack_int(codes, bits: int) -> bytes:
    """Signed codes in [-(2^(b-1)), 2^(b-1)-1] -> exact-width bitstream
    (offset-binary: stored value = code + 2^(b-1)). Scales travel separately."""
    assert bits in (2, 3, 4)
    off = 1 << (bits - 1)
    c = _as_numpy(codes).astype(np.int16)
    assert c.min(initial=0) >= -off and c.max(initial=0) <= off - 1, f"codes out of {bits}-bit range"
    return _pack_bits((c + off).astype(np.uint8), bits)


def unpack_int(raw: bytes, bits: int, shape) -> np.ndarray:
    """Inverse of pack_int -> int8 codes with given shape."""
    assert bits in (2, 3, 4)
    off = 1 << (bits - 1)
    n = int(np.prod(shape)) if len(shape) else 1
    vals = _unpack_bits(raw, bits, n)
    return (vals.astype(np.int16) - off).astype(np.int8).reshape(shape)


# ---- measured footprint ----


@torch.no_grad()
def measured_bytes(model) -> dict:
    """Real packed footprint of a Transformer per its precision arm.

    body_bytes: packed quantized-linear codes (+ bf16 norms and, on the bf16
    arm, bf16 linears at 2 B/param); scale_bytes: fp32 gammas/group scales;
    emb_bytes: (tied) embedding at 2 B/param. bf16 tensors count 2 bytes per
    param everywhere. Feeds the memory transform in fitting (PLAN.md s8).
    """
    body = emb = scale = 0
    handled: set[int] = set()
    for _, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            codes, _ = mod.packed_weight()
            body += len(pack_ternary(codes))
            scale += 4  # per-tensor fp32 gamma
            handled.add(id(mod.weight))
        elif isinstance(mod, GroupIntLinear):
            codes, scales = mod.packed_weight()
            body += len(pack_int(codes, mod.bits))
            scale += scales.numel() * 4
            handled.add(id(mod.weight))
            handled.add(id(mod.scale))
    seen: set[int] = set()
    for name, p in model.named_parameters():  # tied head deduped by torch
        if id(p) in handled or id(p) in seen:
            continue
        seen.add(id(p))
        if "embed" in name or name == "head.weight":
            emb += p.numel() * 2  # embeddings stay bf16 in every arm
        else:
            body += p.numel() * 2  # norms (+ bf16-arm linears) at 2 B/param
    precision: Precision = model.cfg.precision
    return {
        "body_bytes": body,
        "emb_bytes": emb,
        "scale_bytes": scale,
        "total_bytes": body + emb + scale,
        "fmt": FMT_BY_PRECISION[precision.value],
    }
