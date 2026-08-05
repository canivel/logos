"""Tests for packed formats, measured footprint and the P0 export-parity
gate (PLAN.md s5: export parity is an exit criterion)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from logos.config import ModelConfig, Precision, make_model
from logos.export.artifact import export_artifact, load_artifact
from logos.export.pack import (
    measured_bytes,
    pack_int,
    pack_ternary,
    unpack_int,
    unpack_ternary,
)
from logos.export.parity import logit_parity, synthetic_prompts
from logos.model.transformer import Transformer
from logos.quant.bitlinear import BitLinear
from logos.quant.paretoq import GroupIntLinear

EDGE_SHAPES = [(1,), (3,), (4,), (5,), (7, 5), (1, 1), (8,), (24,), (128, 64), (2, 3, 4)]


def _tiny_cfg(precision: Precision) -> ModelConfig:
    return ModelConfig(
        d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, ffn_hidden=96,
        vocab_size=256, max_seq_len=64, precision=precision, weight_group_size=32,
    )


# ---- (1) exact pack/unpack roundtrip ----


def test_pack_ternary_roundtrip():
    rng = np.random.default_rng(0)
    for shape in EDGE_SHAPES:
        codes = rng.integers(-1, 2, size=shape).astype(np.int8)
        raw = pack_ternary(codes)
        assert len(raw) == math.ceil(codes.size * 2 / 8)
        out = unpack_ternary(raw, shape)
        assert out.dtype == np.int8
        np.testing.assert_array_equal(out, codes)
    # torch tensors accepted too
    t = torch.tensor([[-1, 0, 1], [1, 1, -1]], dtype=torch.int8)
    np.testing.assert_array_equal(unpack_ternary(pack_ternary(t), (2, 3)), t.numpy())


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_pack_int_roundtrip(bits):
    rng = np.random.default_rng(bits)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    for shape in EDGE_SHAPES:
        codes = rng.integers(qmin, qmax + 1, size=shape).astype(np.int8)
        raw = pack_int(codes, bits)
        assert len(raw) == math.ceil(codes.size * bits / 8)
        np.testing.assert_array_equal(unpack_int(raw, bits, shape), codes)
    # exact-width check: 8 3-bit codes fit in exactly 3 bytes
    full = np.arange(qmin, qmax + 1, dtype=np.int8)
    if bits == 3:
        assert len(pack_int(full, 3)) == 3
    np.testing.assert_array_equal(unpack_int(pack_int(full, bits), bits, full.shape), full)


# ---- (2) measured footprint on the 25m ladder entry ----


def test_measured_bytes_25m():
    totals = {}
    for p in [Precision.W1_58, Precision.W2, Precision.W3, Precision.W4, Precision.BF16]:
        model = Transformer(make_model("25m", p))
        mb = measured_bytes(model)
        totals[p] = mb["total_bytes"]
        nonemb = model.n_params()["nonemb"]
        if p is Precision.W1_58:
            # ternary body ~= nonemb / 4 bytes (2 bits/param) + norm overhead
            assert abs(mb["body_bytes"] - nonemb / 4) / (nonemb / 4) < 0.03
            assert mb["fmt"] == "i2_s"
            assert mb["scale_bytes"] < 1024  # one fp32 gamma per linear
        if p is Precision.BF16:
            assert mb["scale_bytes"] == 0
            assert abs(mb["body_bytes"] - 2 * nonemb) / (2 * nonemb) < 0.03
        # embeddings are bf16 in every arm: 2 bytes/param, tied -> counted once
        assert mb["emb_bytes"] == model.n_params()["emb"] * 2
        assert mb["total_bytes"] == mb["body_bytes"] + mb["emb_bytes"] + mb["scale_bytes"]
        del model
    # measured footprint ordering across arms at fixed N (PLAN.md s7)
    t = [totals[p] for p in (Precision.W1_58, Precision.W2, Precision.W3,
                             Precision.W4, Precision.BF16)]
    assert t == sorted(t) and len(set(t)) == 5


# ---- (3) full circle: export -> load -> logit parity (the P0 gate) ----


@pytest.mark.parametrize(
    "precision", [Precision.W1_58, Precision.W2, Precision.W3, Precision.W4, Precision.BF16]
)
def test_export_load_logit_parity(precision, tmp_path):
    torch.manual_seed(7)
    model = Transformer(_tiny_cfg(precision)).eval()
    out_dir = export_artifact(model, tmp_path / precision.name)
    reloaded = load_artifact(out_dir)
    prompts = synthetic_prompts(vocab_size=256, n_prompts=8, seq_len=32, seed=1)
    res = logit_parity(model, reloaded, prompts, device="cpu", atol=1e-2, rtol=0.0)
    assert res["pass"], res
    assert res["max_abs_diff"] < 1e-2  # fp32 CPU: observed ~1e-6
    assert res["kl_div_max"] < 1e-4


@pytest.mark.parametrize("precision", [Precision.W1_58, Precision.W2, Precision.W3, Precision.W4])
def test_reloaded_masters_requantize_identically(precision, tmp_path):
    """Fake-quant idempotence on reloaded masters: quantize_weight() of the
    reloaded model matches the original per layer, and the packed codes are
    bitwise identical (the exact-parity argument in export/artifact.py)."""
    torch.manual_seed(11)
    model = Transformer(_tiny_cfg(precision)).eval()
    reloaded = load_artifact(export_artifact(model, tmp_path / "art"))
    pairs = [
        (m1, m2)
        for (_, m1), (_, m2) in zip(model.named_modules(), reloaded.named_modules())
        if isinstance(m1, (BitLinear, GroupIntLinear))
    ]
    assert pairs
    with torch.no_grad():
        for m1, m2 in pairs:
            c1, s1 = m1.packed_weight()
            c2, s2 = m2.packed_weight()
            assert torch.equal(c1, c2)  # codes reproduce bitwise
            wq1, wq2 = m1.quantize_weight(), m2.quantize_weight()
            if isinstance(m1, GroupIntLinear):
                assert torch.equal(s1, s2)  # scale Parameter restored bitwise
                assert torch.equal(wq1, wq2)
            else:
                # gamma is recomputed as mean|W|; equal up to summation-order ulps
                assert torch.allclose(s1, s2, rtol=1e-6, atol=0)
                assert torch.allclose(wq1, wq2, rtol=1e-6, atol=1e-12)
