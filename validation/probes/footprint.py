"""footprint: measured packed bytes vs byte accounting re-derived from shapes
alone, and the KV formula vs the plan's arithmetic (PLAN.md s4, s8).

Byte model (independent of src/logos/export): ternary = 2 bits/code
(ceil(n/4) bytes) + 4 bytes gamma per tensor; k-bit = k bits/code
(ceil(n*k/8) bytes) + 4 bytes per (out_channel, group) scale, group size 128
along the input dim; everything else (embeddings, norms, subln) 2 bytes/param
bf16. KV_bytes = 2 * layers * kv_heads * head_dim * context * (b_kv/8) * batch.
"""

from __future__ import annotations

import math

from validation.base import GateOutcome, Probe

TINY = dict(
    d_model=256, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=512,
    vocab_size=1024, max_seq_len=64, weight_group_size=128,
)

# The plan's 1.5b capstone shape (PLAN.md s4 table + KV formula example).
CAP = dict(n_layers=30, n_kv_heads=4, head_dim=128)


def _tiny_arith(prec_value: str) -> dict:
    """Expected byte buckets from the TINY shape numbers alone."""
    d, L = TINY["d_model"], TINY["n_layers"]
    hd = d // TINY["n_heads"]
    qdim = TINY["n_heads"] * hd
    kvdim = TINY["n_kv_heads"] * hd
    ffn = TINY["ffn_hidden"]
    gs = TINY["weight_group_size"]
    # (out, in) of every body linear per layer: wq, wk, wv, wo, gate, up, down
    linears = [(qdim, d), (kvdim, d), (kvdim, d), (d, qdim), (ffn, d), (ffn, d), (d, ffn)]
    quantized = prec_value != "bf16"
    body = scale = 0
    for _ in range(L):
        for out_f, in_f in linears:
            n = out_f * in_f
            if not quantized:
                body += n * 2
            elif prec_value == "1.58":
                body += math.ceil(n / 4)
                scale += 4
            else:
                k = int(prec_value)
                body += math.ceil(n * k / 8)
                scale += out_f * (in_f // gs) * 4
    norms = L * 2 * d + d  # attn_norm + ffn_norm per layer, final_norm
    if quantized:
        norms += L * (qdim + ffn)  # subln on o_proj and down_proj inputs
    body += norms * 2
    emb = TINY["vocab_size"] * d * 2  # tied embedding, bf16
    return {"body_bytes": body, "emb_bytes": emb, "scale_bytes": scale,
            "total_bytes": body + emb + scale, "norm_bytes": norms * 2}


def _cap_kv(context: int, kv_bits: float = 16.0, batch: int = 1) -> int:
    return int(2 * CAP["n_layers"] * CAP["n_kv_heads"] * CAP["head_dim"]
               * context * (kv_bits / 8) * batch)


class FootprintProbe(Probe):
    name = "footprint"
    description = (
        "measured_bytes vs shape-derived bit-level byte accounting; "
        "precision ordering; theory-vs-measured overhead; KV formula"
    )
    gate_specs = [
        ("G1", "measured_bytes() equals the shape-derived arithmetic exactly "
               "(body/emb/scale/total) for all 5 arms on the tiny model"),
        ("G2", "total packed bytes strictly ordered ternary < 2b < 3b < 4b < bf16 "
               "on the tiny model"),
        ("G3", "ModelConfig.weight_bytes underestimates measured for 2/3/4-bit and "
               "the gap equals exactly the predicted scale bytes plus the bf16 "
               "norm/subln bytes the theory omits"),
        ("G4", "ModelConfig.kv_bytes at the 1.5b shape, 8k context, bf16 is within "
               "2% of the plan's ~500MB and exactly linear in context, kv_bits, "
               "and batch (matching the plan formula)"),
    ]

    def collect(self) -> dict:
        import torch

        from logos.config import ModelConfig, Precision, make_model
        from logos.export.pack import measured_bytes
        from logos.model.transformer import Transformer

        m: dict = {"arms": {}}
        for prec in (Precision.W1_58, Precision.W2, Precision.W3, Precision.W4,
                     Precision.BF16):
            cfg = ModelConfig(precision=prec, **TINY)
            torch.manual_seed(17)
            model = Transformer(cfg).eval()
            meas = measured_bytes(model)
            exp = _tiny_arith(prec.value)
            m["arms"][prec.value] = {
                "measured": {k: meas[k] for k in
                             ("body_bytes", "emb_bytes", "scale_bytes", "total_bytes")},
                "expected": {k: exp[k] for k in
                             ("body_bytes", "emb_bytes", "scale_bytes", "total_bytes")},
                "theory": cfg.weight_bytes(),
                "norm_bytes": exp["norm_bytes"],
            }

        cfg15 = make_model("1.5b")
        m["kv_8k"] = cfg15.kv_bytes(8192)
        m["kv_plan_8k"] = _cap_kv(8192)
        m["kv_16k_2x"] = cfg15.kv_bytes(16384) == 2 * m["kv_8k"] == _cap_kv(16384)
        m["kv_8bit_half"] = 2 * cfg15.kv_bytes(8192, kv_bits=8.0) == m["kv_8k"]
        m["kv_4bit"] = cfg15.kv_bytes(8192, kv_bits=4.0)
        m["kv_4bit_quarter"] = 4 * m["kv_4bit"] == m["kv_8k"] == _cap_kv(8192)
        m["kv_batch4"] = cfg15.kv_bytes(8192, batch=4) == 4 * m["kv_8k"] == _cap_kv(8192, batch=4)
        return m

    def gates(self, m: dict) -> list[GateOutcome]:
        specs = dict(self.gate_specs)
        arms = m["arms"]

        g1_bad = [
            f"{p}: measured={a['measured']} expected={a['expected']}"
            for p, a in arms.items() if a["measured"] != a["expected"]
        ]
        totals = {p: a["measured"]["total_bytes"] for p, a in arms.items()}
        order = [totals["1.58"], totals["2"], totals["3"], totals["4"], totals["bf16"]]
        g2_ok = all(a < b for a, b in zip(order, order[1:]))

        g3_bad = []
        for p in ("2", "3", "4"):
            a = arms[p]
            gap = a["measured"]["total_bytes"] - a["theory"]
            pred = a["expected"]["scale_bytes"] + a["norm_bytes"]
            if not (gap > 0 and gap == pred):
                g3_bad.append(f"{p}b: gap={gap} predicted={pred}")

        kv_rel = abs(m["kv_8k"] / 500e6 - 1.0)
        g4_ok = (kv_rel <= 0.02 and m["kv_8k"] == m["kv_plan_8k"] and m["kv_16k_2x"]
                 and m["kv_8bit_half"] and m["kv_4bit_quarter"] and m["kv_batch4"])
        return [
            GateOutcome("G1", specs["G1"], not g1_bad,
                        detail="; ".join(g1_bad) or
                        f"exact match, totals={totals}"),
            GateOutcome("G2", specs["G2"], g2_ok,
                        detail=f"totals 1.58<2<3<4<bf16: {order}"),
            GateOutcome("G3", specs["G3"], not g3_bad,
                        detail="; ".join(g3_bad) or "gap == scale + norm bytes for 2/3/4-bit"),
            GateOutcome("G4", specs["G4"], g4_ok,
                        detail=(f"kv@8k={m['kv_8k']/1e6:.1f}MB (plan ~500MB, "
                                f"rel dev {kv_rel:.4f}); 4-bit={m['kv_4bit']/1e6:.1f}MB; "
                                f"linear in ctx/bits/batch={m['kv_16k_2x'] and m['kv_8bit_half'] and m['kv_4bit_quarter'] and m['kv_batch4']}")),
        ]
