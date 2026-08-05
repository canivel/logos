"""Data pipeline tests: loader determinism (principle 5: same data order
everywhere), exact resume-skip, window coverage, and the synthetic prepare's
on-disk schema (byte counts feed the bits-per-byte eval, principle 3)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from logos.data.loader import TokenLoader
from logos.data.prepare import prepare_synthetic

VOCAB = 512
SEQ = 32


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    prepare_synthetic(d, n_shards=2, shard_tokens=4_096, vocab_size=VOCAB, val_tokens=1_024)
    return d


def make_loader(data_dir, **kw) -> TokenLoader:
    base = dict(seq_len=SEQ, batch_size=4, data_seed=1337)
    base.update(kw)
    return TokenLoader(data_dir, **base)


def test_loader_determinism(data_dir):
    a, b = make_loader(data_dir), make_loader(data_dir)
    ita, itb = a.iter_batches(0), b.iter_batches(0)
    for _ in range(12):
        ba, bb = next(ita), next(itb)
        assert torch.equal(ba["inputs"], bb["inputs"])
        assert torch.equal(ba["targets"], bb["targets"])
        assert ba["inputs"].dtype == torch.int64
        assert ba["inputs"].shape == (4, SEQ)
    # inputs/targets are shifted views of the same window
    assert torch.equal(ba["inputs"][:, 1:], ba["targets"][:, :-1])


def test_loader_different_seed_differs(data_dir):
    a = make_loader(data_dir).get_batch(0)
    b = make_loader(data_dir, data_seed=7).get_batch(0)
    assert not torch.equal(a["inputs"], b["inputs"])


def test_resume_skip_equals_skipping(data_dir):
    k = 5
    ref = make_loader(data_dir)
    it0 = ref.iter_batches(0)
    for _ in range(k):
        next(it0)
    expected = next(it0)  # batch index k
    fresh = make_loader(data_dir)
    got = next(fresh.iter_batches(k))
    assert torch.equal(got["inputs"], expected["inputs"])
    assert torch.equal(got["targets"], expected["targets"])
    # and across an epoch boundary
    step = ref.batches_per_epoch + 3
    assert torch.equal(
        next(make_loader(data_dir).iter_batches(step))["inputs"], ref.get_batch(step)["inputs"]
    )


def test_window_coverage_no_token_reused(data_dir):
    loader = make_loader(data_dir)
    window = SEQ + 1
    used: set[tuple[int, int]] = set()
    for w in range(loader.n_windows):
        s, a, b = loader.window_span(w)
        assert b - a == window
        assert b <= len(loader.shards[s])
        for pos in range(a, b):
            key = (s, pos)
            assert key not in used, "token used twice within an epoch"
            used.add(key)
    assert len(used) == loader.n_windows * window
    # the epoch shuffle is a true permutation of all windows
    perm = loader.epoch_permutation(0)
    assert np.array_equal(np.sort(perm), np.arange(loader.n_windows))
    # one epoch of batches covers each window at most once
    seen = set()
    for step in range(loader.batches_per_epoch):
        e, b = divmod(step, loader.batches_per_epoch)
        idxs = loader.epoch_permutation(e)[b * 4 : (b + 1) * 4]
        for w in idxs:
            assert int(w) not in seen
            seen.add(int(w))


def test_synthetic_prepare_schema_and_determinism(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    idx1 = prepare_synthetic(d1, n_shards=2, shard_tokens=4_096, vocab_size=VOCAB, val_tokens=1_024)
    idx2 = prepare_synthetic(d2, n_shards=2, shard_tokens=4_096, vocab_size=VOCAB, val_tokens=1_024)
    assert idx1 == idx2  # deterministic given the seed
    assert idx1 == json.loads((d1 / "index.json").read_text())

    assert idx1["total_tokens"] == sum(s["tokens"] for s in idx1["shards"]) == 2 * 4_096
    for s in idx1["shards"]:
        assert s["utf8_bytes"] > 0 and s["docs"] > 0
        f = d1 / s["file"]
        assert f.stat().st_size == s["tokens"] * 2  # uint16
        toks = np.fromfile(f, dtype=np.uint16)
        assert toks.max() < VOCAB
        assert (d1 / s["file"]).read_bytes() == (d2 / s["file"]).read_bytes()

    # two disjoint held-out sets with byte counts (bits-per-byte eval)
    for name in ("val1", "val2"):
        v = idx1["val"][name]
        assert v["tokens"] == 1_024 and v["utf8_bytes"] > 0
        assert (d1 / v["file"]).stat().st_size == v["tokens"] * 2
    assert (d1 / "val1.bin").read_bytes() != (d1 / "val2.bin").read_bytes()

    # loader can open the val splits too
    val_loader = TokenLoader(d1, seq_len=SEQ, batch_size=2, split="val1")
    assert val_loader.n_windows == 1_024 // (SEQ + 1)
