"""Single switch point for `--precision`: every linear in the model body is
built here, so an arm differs from bf16 in exactly one code path."""

from __future__ import annotations

from torch import nn

from logos.config import Precision
from logos.quant.bitlinear import BitLinear
from logos.quant.paretoq import GroupIntLinear


def build_linear(
    in_features: int,
    out_features: int,
    precision: Precision,
    group_size: int = 128,
    act_bits: int = 8,
) -> nn.Linear:
    if precision is Precision.BF16:
        return nn.Linear(in_features, out_features, bias=False)
    if precision is Precision.W1_58:
        return BitLinear(in_features, out_features, bias=False, act_bits=act_bits)
    bits = int(precision.value)
    gs = group_size
    while in_features % gs != 0:  # small ladder dims (e.g. 640-dim kv proj)
        gs //= 2
    return GroupIntLinear(
        in_features, out_features, bits=bits, bias=False, group_size=gs, act_bits=act_bits
    )
