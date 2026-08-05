"""Bits-per-byte runner: the project's fitting target (PLAN.md s3, principle 3).

BPB = (sum of token NLL in nats over the full token stream / ln 2) / total
UTF-8 bytes of the underlying text. The byte count comes from the data
pipeline's index.json (source bytes per val set) and is accepted as an
argument — never recomputed from text here.

Evaluation uses non-overlapping windows (stride = seq_len) by default over a
uint16 token .bin. Every token in the stream except the very first (which has
no context) is scored exactly once, including the final partial window; with
stride < seq_len, overlapping windows score only their fresh targets so the
token count is conserved regardless of stride.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _load_tokens(token_shard) -> torch.Tensor:
    """Accept a path to a uint16 .bin, a numpy array, or a torch tensor."""
    if isinstance(token_shard, (str, Path)):
        arr = np.fromfile(str(token_shard), dtype=np.uint16)
        return torch.from_numpy(arr.astype(np.int64))
    if isinstance(token_shard, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(token_shard).astype(np.int64))
    if isinstance(token_shard, torch.Tensor):
        return token_shard.long().flatten()
    raise TypeError(f"token_shard must be a path, ndarray or tensor, got {type(token_shard)}")


def _windows(
    tokens: torch.Tensor, seq_len: int, stride: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, target_ids) windows; already-scored targets are
    masked to -100 so each stream position is scored exactly once."""
    n = tokens.numel()
    scored_upto = 0  # highest stream position already scored as a target
    start = 0
    while start < n - 1:
        inp = tokens[start : start + seq_len]
        tgt = tokens[start + 1 : start + 1 + inp.numel()]
        if tgt.numel() < inp.numel():  # final partial window
            inp = inp[: tgt.numel()]
        tgt = tgt.clone()
        n_masked = max(0, scored_upto - start)  # target j is stream pos start+1+j
        if n_masked:
            tgt[:n_masked] = -100
        scored_upto = start + inp.numel()
        yield inp, tgt
        start += stride


@torch.no_grad()
def bpb(
    model: nn.Module,
    token_shard,
    n_bytes: int,
    seq_len: int = 2048,
    device: str = "cpu",
    batch_size: int = 8,
    stride: int | None = None,
) -> dict:
    """BPB over a token stream. Returns dict(bpb, nll_sum_nats, n_tokens, n_bytes).

    `n_bytes` is the UTF-8 byte count of the source text (from index.json).
    Default stride=seq_len gives non-overlapping windows; stride < seq_len
    gives sliding-window eval with longer contexts per token.
    """
    stride = stride or seq_len
    assert 1 <= stride <= seq_len, "stride must be in [1, seq_len]"
    assert n_bytes > 0, "n_bytes must be positive"
    tokens = _load_tokens(token_shard)
    model = model.to(device).eval()

    nll_sum = 0.0
    n_tok = 0
    batch_inp: list[torch.Tensor] = []
    batch_tgt: list[torch.Tensor] = []

    def flush():
        nonlocal nll_sum, n_tok
        if not batch_inp:
            return
        maxlen = max(t.numel() for t in batch_inp)
        inp = torch.stack(
            [F.pad(t, (0, maxlen - t.numel()), value=0) for t in batch_inp]
        ).to(device)
        tgt = torch.stack(
            [F.pad(t, (0, maxlen - t.numel()), value=-100) for t in batch_tgt]
        ).to(device)
        out = model(inp)
        logits = out[0] if isinstance(out, tuple) else out
        nll_sum += F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            tgt.view(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        n_tok += int((tgt != -100).sum().item())
        batch_inp.clear()
        batch_tgt.clear()

    for inp, tgt in _windows(tokens, seq_len, stride):
        batch_inp.append(inp)
        batch_tgt.append(tgt)
        if len(batch_inp) == batch_size:
            flush()
    flush()

    return {
        "bpb": (nll_sum / math.log(2)) / n_bytes,
        "nll_sum_nats": nll_sum,
        "n_tokens": n_tok,
        "n_bytes": int(n_bytes),
    }


def _read_index(data_dir: Path, which: str) -> tuple[str, int | None, int]:
    """Defensive index.json read. Preferred schema:
    {"val1": {"bin": "val1.bin", "n_tokens": int, "n_bytes": int}, ...};
    flat fallback: {"val1_bin": ..., "val1_n_tokens": ..., "val1_n_bytes": ...}.
    """
    with open(data_dir / "index.json", encoding="utf-8") as f:
        idx = json.load(f)
    entry = idx.get(which)
    if isinstance(entry, dict):
        bin_name = entry.get("bin") or entry.get("path") or entry.get("file") or f"{which}.bin"
        n_tokens = entry.get("n_tokens")
        n_bytes = entry.get("n_bytes") or entry.get("bytes") or entry.get("source_bytes")
    else:
        bin_name = idx.get(f"{which}_bin") or idx.get(f"{which}_path") or f"{which}.bin"
        n_tokens = idx.get(f"{which}_n_tokens")
        n_bytes = idx.get(f"{which}_n_bytes") or idx.get(f"{which}_bytes")
    if n_bytes is None:
        raise KeyError(
            f"index.json in {data_dir} has no byte count for {which!r}; "
            "BPB needs the source UTF-8 byte count (PLAN.md s3)"
        )
    return str(bin_name), n_tokens, int(n_bytes)


def bpb_from_val_dir(
    model: nn.Module,
    data_dir,
    which: str = "val1",
    seq_len: int = 2048,
    device: str = "cpu",
    batch_size: int = 8,
    stride: int | None = None,
) -> dict:
    """BPB on a held-out set described by data_dir/index.json (which='val1'|'val2')."""
    data_dir = Path(data_dir)
    bin_name, n_tokens, n_bytes = _read_index(data_dir, which)
    tokens = np.fromfile(str(data_dir / bin_name), dtype=np.uint16)
    if n_tokens is not None:
        tokens = tokens[: int(n_tokens)]
    return bpb(model, tokens, n_bytes, seq_len, device, batch_size, stride)
