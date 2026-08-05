"""Deterministic sharded token loader.

Map-style over fixed-length (seq_len+1) windows across shards. Window order
is a fixed permutation seeded by TrainConfig.data_seed (1337 project-wide),
so every arm at a given size sees IDENTICAL token sequences — precision is
the only difference in what the model experiences (PLAN.md principle 5).

Windows are non-overlapping within each shard (partial tails are dropped),
so no token is used twice within an epoch. Resume is pure index arithmetic:
`iter_batches(start_step)` computes the (epoch, permutation slice) for that
step and never touches skipped data. Past one epoch the permutation reseeds
per epoch from (data_seed, epoch), still identical across arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TokenLoader:
    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        batch_size: int,
        data_seed: int = 1337,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.index = json.loads((self.data_dir / "index.json").read_text())
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.window = seq_len + 1
        infos = self.index["shards"] if split == "train" else [self.index["val"][split]]
        self.shards = [
            np.memmap(self.data_dir / s["file"], dtype=np.uint16, mode="r") for s in infos
        ]
        self.shard_windows = [len(s) // self.window for s in self.shards]
        self.cum = np.concatenate([[0], np.cumsum(self.shard_windows)]).astype(np.int64)
        self.n_windows = int(self.cum[-1])
        assert self.n_windows >= batch_size, "not enough windows for a single batch"
        self.batches_per_epoch = self.n_windows // batch_size
        self._perm_cache: tuple[int, np.ndarray] | None = None

    # ---- window addressing (exposed for the coverage checks in tests) ----

    def window_span(self, widx: int) -> tuple[int, int, int]:
        """Global window index -> (shard_idx, start_token, end_token)."""
        s = int(np.searchsorted(self.cum, widx, side="right") - 1)
        j = widx - int(self.cum[s])
        return s, j * self.window, (j + 1) * self.window

    def _window_tokens(self, widx: int) -> np.ndarray:
        s, a, b = self.window_span(widx)
        return np.asarray(self.shards[s][a:b], dtype=np.int64)

    def epoch_permutation(self, epoch: int) -> np.ndarray:
        """Fixed shuffle: seeded by (data_seed, epoch), identical everywhere."""
        if self._perm_cache is not None and self._perm_cache[0] == epoch:
            return self._perm_cache[1]
        perm = np.random.default_rng([self.data_seed, epoch]).permutation(self.n_windows)
        self._perm_cache = (epoch, perm)
        return perm

    # ---- batching ----

    def get_batch(self, step: int) -> dict[str, torch.Tensor]:
        """Batch for a global step: dict(inputs, targets), int64 [B, seq_len]."""
        epoch, b = divmod(step, self.batches_per_epoch)
        perm = self.epoch_permutation(epoch)
        idxs = perm[b * self.batch_size : (b + 1) * self.batch_size]
        toks = torch.from_numpy(np.stack([self._window_tokens(int(w)) for w in idxs]))
        return {"inputs": toks[:, :-1].contiguous(), "targets": toks[:, 1:].contiguous()}

    def iter_batches(self, start_step: int = 0):
        """Infinite deterministic stream from a global step (exact resume)."""
        step = start_step
        while True:
            yield self.get_batch(step)
            step += 1
