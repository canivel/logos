"""Tests for the BPB runner (the fitting target, PLAN.md s3) and the
downstream wrapper's install guard."""

from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest
import torch

from logos.config import ModelConfig, Precision
from logos.eval.bpb import bpb, bpb_from_val_dir
from logos.model.transformer import Transformer

VOCAB = 32768
HAVE_LM_EVAL = importlib.util.find_spec("lm_eval") is not None


@pytest.fixture(scope="module")
def uniform_model():
    """Tiny vocab-32768 model with all parameters zeroed: logits are exactly
    zero everywhere -> uniform distribution, NLL = ln(32768) per token."""
    cfg = ModelConfig(
        d_model=64, n_layers=1, n_heads=2, n_kv_heads=1, ffn_hidden=128,
        vocab_size=VOCAB, max_seq_len=256, precision=Precision.BF16,
    )
    m = Transformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    return m.eval()


def _tokens(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, VOCAB, size=n).astype(np.uint16)


def test_bpb_uniform_hand_computable(uniform_model):
    """Uniform logits on k scored tokens and b bytes -> BPB = k*log2(V)/b."""
    toks = _tokens(300)
    n_bytes = 4321
    res = bpb(uniform_model, toks, n_bytes, seq_len=64, device="cpu", batch_size=4)
    k = res["n_tokens"]
    assert k == len(toks) - 1  # every token but the context-less first
    assert res["bpb"] == pytest.approx(k * math.log2(VOCAB) / n_bytes, rel=1e-6)
    assert res["nll_sum_nats"] == pytest.approx(k * math.log(VOCAB), rel=1e-6)
    assert res["n_bytes"] == n_bytes


def test_bpb_stride_and_partial_window(uniform_model):
    """Token count conservation: partial final windows, overlapping strides
    and different batch sizes all score exactly len(stream)-1 tokens."""
    toks = _tokens(3 * 64 + 7, seed=1)  # not a multiple of seq_len
    n_bytes = 1000
    base = bpb(uniform_model, toks, n_bytes, seq_len=64, device="cpu", batch_size=4)
    overlapped = bpb(
        uniform_model, toks, n_bytes, seq_len=64, device="cpu", batch_size=4, stride=32
    )
    rebatched = bpb(uniform_model, toks, n_bytes, seq_len=64, device="cpu", batch_size=1)
    for res in (base, overlapped, rebatched):
        assert res["n_tokens"] == len(toks) - 1
    # uniform model: NLL is context-free, so all schedules agree exactly
    assert overlapped["bpb"] == pytest.approx(base["bpb"], rel=1e-6)
    assert rebatched["bpb"] == pytest.approx(base["bpb"], rel=1e-6)


def test_bpb_from_val_dir_both_schemas(uniform_model, tmp_path):
    """index.json read is defensive: nested {'val1': {...}} and flat
    {'val2_bin': ...} schemas both work (another agent writes this file)."""
    toks1, toks2 = _tokens(200, seed=2), _tokens(150, seed=3)
    toks1.tofile(tmp_path / "val1.bin")
    toks2.tofile(tmp_path / "val2.bin")
    index = {
        "val1": {"bin": "val1.bin", "n_tokens": len(toks1), "n_bytes": 900},
        "val2_bin": "val2.bin",
        "val2_n_tokens": len(toks2),
        "val2_n_bytes": 700,
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    r1 = bpb_from_val_dir(uniform_model, tmp_path, "val1", seq_len=64, batch_size=4)
    r2 = bpb_from_val_dir(uniform_model, tmp_path, "val2", seq_len=64, batch_size=4)
    d1 = bpb(uniform_model, toks1, 900, seq_len=64, device="cpu", batch_size=4)
    d2 = bpb(uniform_model, toks2, 700, seq_len=64, device="cpu", batch_size=4)
    assert r1["bpb"] == pytest.approx(d1["bpb"], rel=1e-9)
    assert r2["bpb"] == pytest.approx(d2["bpb"], rel=1e-9)
    assert r1["n_tokens"] == len(toks1) - 1
    assert r2["n_tokens"] == len(toks2) - 1


def test_downstream_suites_defined():
    from logos.eval import downstream

    assert set(downstream.SMALL_SUITE) == {
        "arc_easy", "hellaswag", "piqa", "sciq", "lambada_openai",
    }
    assert set(downstream.SMALL_SUITE) < set(downstream.FULL_SUITE)
    assert "mmlu" in downstream.P3_SUITE
    assert {"gsm8k", "ifeval"} < set(downstream.CAPSTONE_SUITE)
    assert downstream.LM_EVAL_PIN == "0.4.9"


@pytest.mark.skipif(HAVE_LM_EVAL, reason="lm_eval installed; missing-dep path unreachable")
def test_downstream_missing_lm_eval_raises_clear_error():
    from logos.eval import downstream

    with pytest.raises(ImportError, match=r"uv pip install lm-eval==0\.4\.9"):
        downstream.run_downstream(None, None, downstream.SMALL_SUITE, device="cpu")
