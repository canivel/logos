"""Probe: the BPB runner matches the paper definition (PLAN.md s3 principle 3).

Independent references — never `logos.eval.bpb` itself: (1) a constant-zero-
logit model whose BPB has the closed form T * log2(V) / B; (2) a hand-written
per-token NLL loop (one window per forward, log_softmax gather, no batching or
masking tricks) over the same non-overlapping-window definition; (3) the
byte-level local_p0 anchor where tokens == utf8 bytes by construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from logos.config import ModelConfig, make_model
from logos.eval.bpb import bpb, bpb_from_val_dir
from logos.model.transformer import Transformer
from validation.base import GateOutcome, Probe

ROOT = Path(__file__).resolve().parents[2]
LOCAL_P0 = ROOT / "data" / "local_p0"

#: G1 byte count: large enough that float32 logit rounding sits below the
#: 1e-9 gate, while one mis-scored token still shifts bpb by log2(V)/B ~ 8e-8.
G1_VOCAB, G1_N, G1_SEQ, G1_BYTES = 313, 1000, 128, 10**8

#: (n_tokens, seq_len) cases for G2/G3; the first has n % seq_len != 0 so the
#: final partial window must be scored.
CASES = ((1000, 96), (512, 128))


class _ZeroLogits(nn.Module):
    """Constant zero logits over the vocab: forced-uniform model."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab
        self._sink = nn.Parameter(torch.zeros(1))  # gives .to(device) a target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*x.shape, self.vocab)


def _independent_nll(model: nn.Module, tokens: torch.Tensor, seq_len: int) -> tuple[float, list[int]]:
    """Per-token NLL from first principles: non-overlapping windows at
    stride=seq_len, one plain forward each, log_softmax gather. Returns
    (nll_sum_nats, scored stream positions in scoring order)."""
    n = tokens.numel()
    nll, pos = 0.0, []
    start = 0
    while start < n - 1:
        inp = tokens[start : start + seq_len]
        tgt = tokens[start + 1 : start + 1 + inp.numel()]
        if tgt.numel() < inp.numel():  # final partial window
            inp = inp[: tgt.numel()]
        out = model(inp.unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        logp = torch.log_softmax(logits.float().squeeze(0), dim=-1)
        nll += float(-logp[torch.arange(tgt.numel()), tgt].sum())
        pos.extend(range(start + 1, start + 1 + inp.numel()))
        start += seq_len
    return nll, pos


def _tiny_trained_model() -> Transformer:
    """A few gradient steps on random data, then float64 so bpb()'s batched
    forward and the single-window loop see bit-identical float32 logits."""
    torch.manual_seed(7)
    cfg = ModelConfig(
        d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128,
        vocab_size=256, max_seq_len=256,
    )
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.randint(0, 256, (4, 129))
    for _ in range(5):
        _, loss = model(data[:, :-1], data[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.double().eval()


class BpbDefinitionProbe(Probe):
    name = "bpb_definition"
    description = "BPB runner vs closed form, independent NLL loop, and byte-level anchor"
    gate_specs = [
        ("G1", "uniform-logits closed form: constant-zero-logit model gives "
               "BPB = T*log2(V)/B (T scored tokens, B bytes) within 1e-9"),
        ("G2", "trained tiny model: bpb() matches the independent per-token NLL "
               "loop within 1e-6 bits/byte, incl. a partial final window"),
        ("G3", "token conservation: reported n_tokens == scored positions == n-1; "
               "every position except 0 scored exactly once at stride=seq_len"),
        ("G4", "byte-level anchor (data/local_p0): bpb == nll_sum_nats/ln2/n_bytes "
               "exactly; n_bytes == index tokens == utf8_bytes; n_tokens == tokens-1 "
               "(vacuous if the index is absent)"),
    ]

    def collect(self) -> dict:
        m: dict = {}

        # G1: forced-uniform closed form.
        torch.manual_seed(0)
        tokens = torch.randint(0, G1_VOCAB, (G1_N,))
        r = bpb(_ZeroLogits(G1_VOCAB), tokens, n_bytes=G1_BYTES, seq_len=G1_SEQ)
        m["g1"] = {
            "bpb": r["bpb"],
            "closed": (G1_N - 1) * math.log2(G1_VOCAB) / G1_BYTES,
            "n_tokens": r["n_tokens"],
            "expect_tokens": G1_N - 1,
        }

        # G2/G3: independent loop on a trained-ish model, two stream lengths.
        model = _tiny_trained_model()
        m["cases"] = []
        for n, seq_len in CASES:
            torch.manual_seed(100 + n)
            toks = torch.randint(0, 256, (n,))
            with torch.no_grad():
                nll_ind, pos = _independent_nll(model, toks, seq_len)
            r = bpb(model, toks, n_bytes=n, seq_len=seq_len)  # byte-level fiction: 1 byte/token
            m["cases"].append({
                "n": n,
                "seq_len": seq_len,
                "bpb": r["bpb"],
                "bpb_independent": nll_ind / math.log(2) / n,
                "n_tokens": r["n_tokens"],
                "positions": pos,
            })

        # G4: byte-level anchor, if the background experiment has produced it.
        index_path = LOCAL_P0 / "index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                val1 = json.load(f)["val"]["val1"]
            torch.manual_seed(20260805)
            model = Transformer(make_model("micro", vocab_size=256)).eval()
            r = bpb_from_val_dir(model, LOCAL_P0, "val1", seq_len=512)
            m["g4"] = {
                "result": r,
                "idx_tokens": int(val1["tokens"]),
                "idx_bytes": int(val1["utf8_bytes"]),
            }
        else:
            m["g4"] = None
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        out: list[GateOutcome] = []

        g1 = m["g1"]
        diff = abs(g1["bpb"] - g1["closed"])
        out.append(GateOutcome(
            "G1", self.gate_specs[0][1],
            passed=diff <= 1e-9 and g1["n_tokens"] == g1["expect_tokens"],
            detail=f"|bpb - closed| = {diff:.3e}; n_tokens = {g1['n_tokens']}",
        ))

        diffs = [abs(c["bpb"] - c["bpb_independent"]) for c in m["cases"]]
        out.append(GateOutcome(
            "G2", self.gate_specs[1][1],
            passed=all(d <= 1e-6 for d in diffs),
            detail="; ".join(
                f"n={c['n']} S={c['seq_len']}: |diff| = {d:.3e}"
                for c, d in zip(m["cases"], diffs)
            ),
        ))

        conserved, details = True, []
        for c in m["cases"]:
            pos = c["positions"]
            ok = (
                sorted(pos) == list(range(1, c["n"]))
                and len(pos) == len(set(pos))
                and c["n_tokens"] == len(pos) == c["n"] - 1
            )
            conserved &= ok
            details.append(f"n={c['n']}: n_tokens={c['n_tokens']} scored={len(pos)} ok={ok}")
        out.append(GateOutcome("G3", self.gate_specs[2][1], passed=conserved,
                               detail="; ".join(details)))

        if m["g4"] is None:
            out.append(GateOutcome(
                "G4", self.gate_specs[3][1], passed=True,
                detail="vacuous: data/local_p0/index.json absent (background P0 not landed)",
            ))
        else:
            g4, r = m["g4"], m["g4"]["result"]
            identity = r["bpb"] == r["nll_sum_nats"] / math.log(2) / r["n_bytes"]
            bytes_ok = r["n_bytes"] == g4["idx_bytes"] == g4["idx_tokens"]
            tok_ok = r["n_tokens"] == g4["idx_tokens"] - 1
            out.append(GateOutcome(
                "G4", self.gate_specs[3][1],
                passed=identity and bytes_ok and tok_ok,
                detail=(
                    f"bpb={r['bpb']:.6f} identity={identity}; n_bytes={r['n_bytes']} "
                    f"idx tokens={g4['idx_tokens']} bytes={g4['idx_bytes']}; "
                    f"n_tokens={r['n_tokens']} (tokens-1: every token but the first scored)"
                ),
            ))
        return out
