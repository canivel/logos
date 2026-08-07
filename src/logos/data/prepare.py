"""FineWeb-Edu -> tokenized uint16 shards (PLAN.md s4-5).

Streams HuggingFaceFW/fineweb-edu (sample-350BT), tokenizes with the fixed
32k tokenizer (Mistral-7B-v0.1 by default; "do not train a custom tokenizer,
not the variable under study"), appends EOS per document, and packs into
~100M-token uint16 .bin shards. index.json records, per shard, the token
count AND the total UTF-8 bytes of source text — required later for the
bits-per-byte eval on the two held-out val sets (principle 3).

Held-out sets: after the train shards are cut, the next `val_docs` documents
of the stream become val1 and the following `val_docs` become val2 (disjoint
by construction, never seen in training).

Deterministic given the same seed (the stream order is fixed; the seed is
recorded in index.json). Resumable-ish: prepare_state.json checkpoints the
consumed-doc count at every shard flush, so a crash costs at most one shard
of work.

`prepare_synthetic` generates deterministic pseudo-text shards with the same
on-disk schema for network-free tests and smoke runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

INDEX_NAME = "index.json"
STATE_NAME = "prepare_state.json"


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# synthetic mode (tests / smoke, no network)
# ---------------------------------------------------------------------------


def _synth_stream(
    seed_seq: list[int], n_tokens: int, vocab_size: int, eos_id: int
) -> tuple[np.ndarray, int, int]:
    """Deterministic pseudo-text: EOS-terminated docs of random ids in
    [3, vocab), with a pseudo UTF-8 byte count of 3-5 bytes/token per doc."""
    rng = np.random.default_rng(seed_seq)
    out = np.empty(n_tokens, dtype=np.uint16)
    filled, nbytes, ndocs = 0, 0, 0
    while filled < n_tokens:
        length = int(rng.integers(16, 192))
        doc = np.empty(length + 1, dtype=np.uint16)
        doc[:-1] = rng.integers(3, vocab_size, size=length)
        doc[-1] = eos_id
        take = min(doc.size, n_tokens - filled)
        out[filled : filled + take] = doc[:take]
        nbytes += int(rng.integers(3, 6)) * take
        filled += take
        ndocs += 1
    return out, nbytes, ndocs


def prepare_synthetic(
    out_dir: str | Path,
    n_shards: int = 2,
    shard_tokens: int = 65_536,
    vocab_size: int = 32_768,
    seed: int = 1337,
    val_tokens: int = 16_384,
    eos_id: int = 2,
) -> dict:
    """Write deterministic pseudo-text shards + val sets + index.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(n_shards):
        toks, nbytes, ndocs = _synth_stream([seed, i], shard_tokens, vocab_size, eos_id)
        fname = f"shard_{i:05d}.bin"
        toks.tofile(out / fname)
        shards.append({"file": fname, "tokens": int(toks.size), "utf8_bytes": nbytes, "docs": ndocs})
    val = {}
    for k, name in enumerate(("val1", "val2")):
        toks, nbytes, ndocs = _synth_stream([seed, 1_000_000 + k], val_tokens, vocab_size, eos_id)
        fname = f"{name}.bin"
        toks.tofile(out / fname)
        val[name] = {"file": fname, "tokens": int(toks.size), "utf8_bytes": nbytes, "docs": ndocs}
    index = {
        "dataset": "synthetic",
        "dataset_config": None,
        "tokenizer": "synthetic",
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "seed": seed,
        "shards": shards,
        "total_tokens": sum(s["tokens"] for s in shards),
        "val": val,
    }
    _write_json(out / INDEX_NAME, index)
    return index


# ---------------------------------------------------------------------------
# real mode (FineWeb-Edu streaming)
# ---------------------------------------------------------------------------


def _encode_batch(tok, texts: list[str], eos_id: int) -> tuple[list[np.ndarray], int]:
    ids_batch = tok(texts, add_special_tokens=False)["input_ids"]
    arrs, nbytes = [], 0
    for text, ids in zip(texts, ids_batch):
        ids.append(eos_id)
        arrs.append(np.asarray(ids, dtype=np.uint16))
        nbytes += len(text.encode("utf-8"))
    return arrs, nbytes


def prepare_fineweb(
    out_dir: str | Path,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-350BT",
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    shard_tokens: int = 100_000_000,
    target_tokens: int | None = None,
    val_docs: int = 20_000,
    seed: int = 1337,
    batch_docs: int = 256,
) -> dict:
    """Stream, tokenize, shard. Stops the train split at `target_tokens`
    (None = exhaust the stream), then cuts val1/val2 from the next docs."""
    from datasets import load_dataset  # local import: keep test imports network-free
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    tok.model_max_length = int(1e9)  # silence length warnings; we pack, not truncate
    eos_id = tok.eos_token_id
    assert eos_id is not None and tok.vocab_size <= 65_536, "shards are uint16"

    state_path = out / STATE_NAME
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"docs_consumed": 0, "shards": []}
    shards: list[dict] = state["shards"]

    ds = load_dataset(dataset, name=dataset_config, split="train", streaming=True)
    it = iter(ds)
    for _ in range(state["docs_consumed"]):  # resume: deterministic skip
        next(it)

    def snapshot(val: dict | None = None) -> dict:
        index = {
            "dataset": dataset,
            "dataset_config": dataset_config,
            "tokenizer": tokenizer_name,
            "vocab_size": tok.vocab_size,
            "eos_id": eos_id,
            "seed": seed,
            "shards": shards,
            "total_tokens": sum(s["tokens"] for s in shards),
            "val": val or {},
        }
        _write_json(out / INDEX_NAME, index)
        return index

    buf: list[np.ndarray] = []
    buf_tokens = buf_bytes = buf_docs = 0
    trained = sum(s["tokens"] for s in shards)

    def flush() -> None:
        nonlocal buf, buf_tokens, buf_bytes, buf_docs, trained
        arr = np.concatenate(buf)
        fname = f"shard_{len(shards):05d}.bin"
        arr.tofile(out / fname)
        shards.append(
            {"file": fname, "tokens": int(arr.size), "utf8_bytes": buf_bytes, "docs": buf_docs}
        )
        trained += int(arr.size)
        state["docs_consumed"] += buf_docs
        buf, buf_tokens, buf_bytes, buf_docs = [], 0, 0, 0
        snapshot()
        _write_json(state_path, state)  # checkpoint the doc count every shard

    done = target_tokens is not None and trained >= target_tokens
    while not done:
        texts = []
        for _ in range(batch_docs):
            try:
                texts.append(next(it)["text"])
            except StopIteration:
                done = True
                break
        if texts:
            arrs, nbytes = _encode_batch(tok, texts, eos_id)
            buf.extend(arrs)
            buf_bytes += nbytes
            buf_docs += len(texts)
            buf_tokens += sum(a.size for a in arrs)
            if buf_tokens >= shard_tokens:
                flush()
        if target_tokens is not None and trained + buf_tokens >= target_tokens:
            done = True
    if buf:
        flush()

    # Held-out sets: two disjoint slices immediately after the train cut.
    val: dict[str, dict] = {}
    for name in ("val1", "val2"):
        v_arrs: list[np.ndarray] = []
        v_bytes = v_docs = 0
        while v_docs < val_docs:
            texts = []
            for _ in range(min(batch_docs, val_docs - v_docs)):
                try:
                    texts.append(next(it)["text"])
                except StopIteration:
                    break
            if not texts:
                break
            arrs, nbytes = _encode_batch(tok, texts, eos_id)
            v_arrs.extend(arrs)
            v_bytes += nbytes
            v_docs += len(texts)
        arr = np.concatenate(v_arrs) if v_arrs else np.empty(0, dtype=np.uint16)
        fname = f"{name}.bin"
        arr.tofile(out / fname)
        val[name] = {"file": fname, "tokens": int(arr.size), "utf8_bytes": v_bytes, "docs": v_docs}
    # Record the val block's stream position so extend_fineweb can skip it.
    state["val_docs"] = sum(v["docs"] for v in val.values())
    state["train_docs_pre_val"] = state["docs_consumed"]
    _write_json(state_path, state)
    return snapshot(val)


def extend_fineweb(
    out_dir: str | Path,
    target_tokens: int,
    batch_docs: int = 256,
) -> dict:
    """Grow the TRAIN split of an existing prepared dir to `target_tokens`
    while keeping val1/val2 byte-identical.

    Leak guard: the val sets were cut from the stream immediately after the
    original train cut, so a naive re-run of prepare_fineweb would consume
    those exact documents into training. This resumes the stream at
    train_docs_pre_val + val_docs + train_docs_post_val — i.e. skipping the
    frozen val block — and appends new train shards after it. Existing
    shards, val bins, and their byte counts are untouched; only new shards
    and total_tokens change in index.json.

    Protocol note (kimi3 review F2): every (size, D/N) cell must be
    internally consistent — never mix corpus versions within a cell. Extend
    only BETWEEN phases, before any run of the next phase starts.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(out_dir)
    index = json.loads((out / INDEX_NAME).read_text())
    state = json.loads((out / STATE_NAME).read_text())
    if "val_docs" not in state:  # dirs prepared before this function existed
        state["val_docs"] = sum(v["docs"] for v in index["val"].values())
        state["train_docs_pre_val"] = state["docs_consumed"] - state.get(
            "train_docs_post_val", 0
        )
    post = state.get("train_docs_post_val", 0)
    skip = state["train_docs_pre_val"] + state["val_docs"] + post

    tok = AutoTokenizer.from_pretrained(index["tokenizer"])
    tok.model_max_length = int(1e9)
    eos_id = index["eos_id"]
    shards: list[dict] = index["shards"]
    trained = sum(s["tokens"] for s in shards)
    if trained >= target_tokens:
        return index

    ds = load_dataset(index["dataset"], name=index["dataset_config"], split="train", streaming=True)
    it = iter(ds)
    for _ in range(skip):
        next(it)

    buf: list[np.ndarray] = []
    buf_tokens = buf_bytes = buf_docs = 0

    def flush() -> None:
        nonlocal buf, buf_tokens, buf_bytes, buf_docs, trained
        arr = np.concatenate(buf)
        fname = f"shard_{len(shards):05d}.bin"
        arr.tofile(out / fname)
        shards.append(
            {"file": fname, "tokens": int(arr.size), "utf8_bytes": buf_bytes, "docs": buf_docs}
        )
        trained += int(arr.size)
        state["train_docs_post_val"] = state.get("train_docs_post_val", 0) + buf_docs
        state["docs_consumed"] += buf_docs
        buf, buf_tokens, buf_bytes, buf_docs = [], 0, 0, 0
        index["total_tokens"] = trained
        _write_json(out / INDEX_NAME, index)
        _write_json(out / STATE_NAME, state)

    done = False
    while not done:
        texts = []
        for _ in range(batch_docs):
            try:
                texts.append(next(it)["text"])
            except StopIteration:
                done = True
                break
        if texts:
            arrs, nbytes = _encode_batch(tok, texts, eos_id)
            buf.extend(arrs)
            buf_bytes += nbytes
            buf_docs += len(texts)
            buf_tokens += sum(a.size for a in arrs)
            if buf_tokens >= 100_000_000:
                flush()
        if trained + buf_tokens >= target_tokens:
            done = True
    if buf:
        flush()
    return index
