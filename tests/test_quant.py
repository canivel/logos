"""Unit tests for the quantizer ladder (PLAN.md section 5 build tasks):
STE gradient check, quant-dequant idempotence, per-precision weight histogram
sanity."""

import torch

from logos.config import Precision
from logos.quant import ActQuant, BitLinear, GroupIntLinear, build_linear
from logos.quant.ste import ste_round, ste_round_clip

torch.manual_seed(0)


class TestSTE:
    def test_round_identity_gradient(self):
        x = torch.randn(64, requires_grad=True)
        ste_round(x * 3).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, 3.0))

    def test_round_clip_gradient_mask(self):
        x = torch.tensor([-2.0, -0.5, 0.3, 1.7], requires_grad=True)
        ste_round_clip(x, -1.0, 1.0).sum().backward()
        assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_forward_is_rounded(self):
        x = torch.tensor([0.4, 0.6, -1.2])
        assert torch.equal(ste_round(x), torch.tensor([0.0, 1.0, -1.0]))


class TestBitLinear:
    def test_weight_is_ternary(self):
        layer = BitLinear(128, 64)
        wq = layer.quantize_weight()
        gamma = layer.weight.float().abs().mean()
        levels = torch.unique(wq)
        assert len(levels) <= 3
        assert torch.allclose(levels.abs().max(), gamma, atol=1e-6)

    def test_codes_idempotent(self):
        layer = BitLinear(128, 64)
        codes, gamma = layer.packed_weight()
        # Restore masters via the parity rescale: codes * gamma / mean|codes|.
        rho = codes.float().abs().mean()
        with torch.no_grad():
            layer.weight.copy_(codes.float() * gamma / rho)
        codes2, gamma2 = layer.packed_weight()
        assert torch.equal(codes, codes2)
        assert torch.allclose(gamma2, gamma, rtol=1e-5)
        wq = layer.quantize_weight()
        assert torch.allclose(wq, codes.float() * gamma, rtol=1e-5, atol=1e-7)

    def test_gradient_flows_to_master(self):
        layer = BitLinear(32, 16)
        out = layer(torch.randn(4, 32))
        out.sum().backward()
        g = layer.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


class TestGroupInt:
    def test_level_count_per_group(self):
        for bits in (2, 3, 4):
            layer = GroupIntLinear(128, 32, bits=bits, group_size=64)
            wq = layer.quantize_weight().view(32, 2, 64)
            for oc in range(0, 32, 7):
                for g in range(2):
                    n_levels = len(torch.unique(wq[oc, g]))
                    assert n_levels <= 2**bits, f"{bits}-bit group has {n_levels} levels"

    def test_value_idempotence(self):
        layer = GroupIntLinear(128, 32, bits=4, group_size=64)
        wq1 = layer.quantize_weight()
        with torch.no_grad():
            layer.weight.copy_(wq1)  # scale param unchanged -> exact idempotence
        wq2 = layer.quantize_weight()
        assert torch.allclose(wq1, wq2, atol=1e-7)

    def test_codes_within_range(self):
        for bits in (2, 3, 4):
            layer = GroupIntLinear(64, 16, bits=bits, group_size=64)
            codes, _ = layer.packed_weight()
            assert codes.min() >= -(2 ** (bits - 1)) and codes.max() <= 2 ** (bits - 1) - 1

    def test_scale_gradient(self):
        layer = GroupIntLinear(64, 16, bits=2, group_size=64)
        layer(torch.randn(4, 64)).sum().backward()
        assert layer.scale.grad is not None and torch.isfinite(layer.scale.grad).all()


class TestActQuant:
    def test_int8_levels_per_token(self):
        aq = ActQuant(bits=8)
        x = torch.randn(4, 256)
        q = aq(x)
        scale = 127.0 / x.abs().amax(dim=-1, keepdim=True)
        codes = q * scale
        assert torch.allclose(codes, codes.round(), atol=1e-3)
        assert codes.abs().max() <= 128.0 + 1e-3

    def test_identity_gradient(self):
        x = torch.randn(2, 16, requires_grad=True)
        ActQuant()(x).sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-6)


class TestFactory:
    def test_dispatch(self):
        assert type(build_linear(64, 64, Precision.BF16)) is torch.nn.Linear
        assert isinstance(build_linear(64, 64, Precision.W1_58), BitLinear)
        for p, bits in [(Precision.W2, 2), (Precision.W3, 3), (Precision.W4, 4)]:
            layer = build_linear(64, 64, p)
            assert isinstance(layer, GroupIntLinear) and layer.bits == bits

    def test_group_size_fallback(self):
        # 80-dim input (60m ladder head_dim) is not divisible by 128.
        layer = build_linear(160, 64, Precision.W4, group_size=128)
        assert 160 % layer.group_size == 0
