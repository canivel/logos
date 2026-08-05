"""Architecture invariants (PLAN.md principles 1 and 2): precision is the only
difference between arms; parameter accounting matches the config contract."""

import pytest
import torch

from logos.config import ALL_PRECISIONS, Precision, make_model
from logos.model import Transformer
from logos.quant import ActQuant, BitLinear, GroupIntLinear

TINY = dict(max_seq_len=128)


def tiny(precision=Precision.BF16, **kw):
    torch.manual_seed(0)
    return Transformer(make_model("25m", precision, **TINY, **kw), **kw.pop("extra", {}))


class TestAccounting:
    def test_nonemb_matches_config(self):
        for p in (Precision.BF16, Precision.W1_58):
            cfg = make_model("25m", p, **TINY)
            net = Transformer(cfg)
            assert net.n_params()["nonemb"] == cfg.n_nonemb

    def test_ladder_targets(self):
        targets = {"25m": 25e6, "60m": 60e6, "125m": 125e6, "250m": 250e6, "490m": 490e6}
        for size, t in targets.items():
            n = make_model(size).n_nonemb
            assert abs(n - t) / t < 0.05, f"{size}: {n:,} vs target {t:,.0f}"

    def test_tied_embeddings(self):
        net = tiny()
        assert net.head.weight is net.embed.weight


class TestArmIsolation:
    def test_bf16_has_no_quant_modules(self):
        net = tiny(Precision.BF16)
        mods = list(net.modules())
        assert not any(isinstance(m, (BitLinear, GroupIntLinear, ActQuant)) for m in mods)
        assert all(b.attn.subln is None and b.ffn.subln is None for b in net.blocks)

    def test_quant_arms_have_subln(self):
        for p in (Precision.W1_58, Precision.W4):
            net = tiny(p)
            assert all(b.attn.subln is not None and b.ffn.subln is not None for b in net.blocks)

    def test_body_linears_are_quantized(self):
        net = tiny(Precision.W1_58)
        for blk in net.blocks:
            for lin in (blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.wo, blk.ffn.up, blk.ffn.down, blk.ffn.gate):
                assert isinstance(lin, BitLinear)
        assert type(net.head) is torch.nn.Linear  # head/embeddings stay bf16

    def test_same_master_init_across_arms(self):
        # Same seed -> identical master weights regardless of precision arm.
        a, b = tiny(Precision.BF16), tiny(Precision.W1_58)
        assert torch.equal(a.blocks[0].attn.wq.weight, b.blocks[0].attn.wq.weight)


class TestForward:
    @pytest.mark.parametrize("p", ALL_PRECISIONS, ids=[p.value for p in ALL_PRECISIONS])
    def test_forward_backward_finite(self, p):
        net = tiny(p)
        x = torch.randint(0, 32768, (2, 64))
        _, loss = net(x, x)
        loss.backward()
        assert torch.isfinite(loss)
        grads = [q.grad for q in net.parameters() if q.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)

    def test_causality(self):
        net = tiny()
        net.eval()
        x = torch.randint(0, 32768, (1, 32))
        with torch.no_grad():
            l1, _ = net(x)
            x2 = x.clone()
            x2[0, -1] = (x2[0, -1] + 1) % 32768
            l2, _ = net(x2)
        assert torch.allclose(l1[0, :-1], l2[0, :-1], atol=1e-5)
        assert not torch.allclose(l1[0, -1], l2[0, -1], atol=1e-5)

    def test_ffn_ablation_arm(self):
        net = Transformer(make_model("25m", Precision.W1_58, ffn_type="sq_relu", **TINY))
        assert net.blocks[0].ffn.gate is None
        x = torch.randint(0, 32768, (1, 32))
        _, loss = net(x, x)
        assert torch.isfinite(loss)

    def test_kv_qat_arm(self):
        cfg = make_model("25m", Precision.W1_58, **TINY)
        net = Transformer(cfg, kv_qat_bits=4)
        assert all(b.attn.kv_quant is not None for b in net.blocks)
        x = torch.randint(0, 32768, (1, 32))
        _, loss = net(x, x)
        loss.backward()
        assert torch.isfinite(loss)

    def test_qk_norm_arm(self):
        net = Transformer(make_model("25m", Precision.W1_58, use_qk_norm=True, **TINY))
        x = torch.randint(0, 32768, (1, 32))
        _, loss = net(x, x)
        assert torch.isfinite(loss)


class TestHistograms:
    """Per-precision weight histogram sanity (P0 build task)."""

    def test_effective_weight_level_counts(self):
        for p, max_levels in [(Precision.W1_58, 3), (Precision.W2, 4), (Precision.W3, 8)]:
            net = tiny(p)
            lin = net.blocks[0].attn.wq
            wq = lin.quantize_weight()
            if isinstance(lin, BitLinear):
                assert len(torch.unique(wq)) <= max_levels
            else:
                gs = lin.group_size
                w = wq.view(wq.shape[0], -1, gs)
                assert len(torch.unique(w[0, 0])) <= max_levels
