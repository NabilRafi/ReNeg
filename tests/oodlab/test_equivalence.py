"""Our methods vs the vendored official postprocessors on the same stream.

fp32 inputs: agreement to float precision (the only difference is operation order).
fp16 inputs + emulate_fp16: bit-identical scores.
"""
import pytest
import torch

from oodlab.methods import AdaNegConfig, NegLabelConfig, TANLConfig
from oodlab.official.equivalence import compare_adaneg, compare_neglabel, compare_tanl

SMALL = dict(num_neg=100, queue_len=30)


def h(t):
    return t.half()


@pytest.mark.parametrize("name,cfg", [
    ("paper", TANLConfig.paper().with_(**SMALL)),
    ("official_sh", TANLConfig.official_sh().with_(**SMALL)),
    ("step2", TANLConfig.paper().with_(step=2, **SMALL)),
    ("step0_neglabel_score", TANLConfig.paper().with_(step=0, **SMALL)),
])
def test_tanl_fp32_matches_official(bank, stream_batches, name, cfg):
    r = compare_tanl(bank.id_text, bank.corpus, bank.noise, stream_batches, bank.n_neglabel, cfg)
    assert r["max_abs_diff"] < 1e-4, (name, r)
    assert r["pred_agreement"] == 1.0, (name, r)


@pytest.mark.parametrize("step", [1, 2, 0])
def test_tanl_fp16_bit_exact(bank, stream_batches, step):
    cfg = TANLConfig.paper().with_(emulate_fp16=True, exact_fp16_final=True, step=step, **SMALL)
    r = compare_tanl(h(bank.id_text), h(bank.corpus), h(bank.noise), [h(b) for b in stream_batches], bank.n_neglabel, cfg)
    assert r["max_abs_diff"] == 0.0, r
    assert r["pred_agreement"] == 1.0, r


def test_tanl_fixed_threshold_extension(bank, stream_batches):
    """auto_threshold=False is our ablation switch (the official code always uses the auto threshold)."""
    from oodlab.methods import TANL

    t = TANL(bank.id_text, bank.corpus, bank.noise, TANLConfig.paper().with_(auto_threshold=False, record=True, **SMALL))
    assert all(t.step(b).info["thr"] == 0.5 for b in stream_batches[:3])


def test_tanl_fp16_fast_final_close(bank, stream_batches):
    """Without the slow exact loop the final score differs by at most ~1 fp16 ulp."""
    cfg = TANLConfig.paper().with_(emulate_fp16=True, exact_fp16_final=False, **SMALL)
    r = compare_tanl(h(bank.id_text), h(bank.corpus), h(bank.noise), [h(b) for b in stream_batches], bank.n_neglabel, cfg)
    assert r["max_abs_diff"] < 2e-3, r


def test_tanl_fast_threshold_close(bank, stream_batches):
    cfg = TANLConfig.paper().with_(threshold_impl="fast", **SMALL)
    r = compare_tanl(bank.id_text, bank.corpus, bank.noise, stream_batches, bank.n_neglabel, cfg)
    assert r["max_abs_diff"] < 1e-4, r


def test_tanl_logit_scale_is_passed_through(bank, stream_batches):
    cfg = TANLConfig.paper().with_(logit_scale=100.003, **SMALL)
    r = compare_tanl(bank.id_text, bank.corpus, bank.noise, stream_batches, bank.n_neglabel, cfg)
    assert r["max_abs_diff"] < 1e-4, r


@pytest.mark.parametrize("g", [5, 100])
def test_neglabel(bank, stream_batches, g):
    neg = bank.corpus[: bank.n_neglabel]
    r = compare_neglabel(bank.id_text, neg, stream_batches, NegLabelConfig(group_num=g))
    assert r["max_abs_diff"] < 1e-4 and r["pred_agreement"] == 1.0, r
    r16 = compare_neglabel(h(bank.id_text), h(neg), [h(b) for b in stream_batches],
                           NegLabelConfig(group_num=g, emulate_fp16=True))
    assert r16["max_abs_diff"] == 0.0 and r16["pred_agreement"] == 1.0, r16


@pytest.mark.parametrize("in_score", ["combine", "adaonly", "vanillaonly"])
def test_adaneg_fp32(bank, stream_batches, in_score):
    neg = bank.corpus[: bank.n_neglabel]
    r = compare_adaneg(bank.id_text, neg, stream_batches, AdaNegConfig(in_score=in_score))
    assert r["max_abs_diff"] < 1e-4 and r["pred_agreement"] == 1.0, r


def test_adaneg_fp16_bit_exact(bank, stream_batches):
    neg = bank.corpus[: bank.n_neglabel]
    r = compare_adaneg(h(bank.id_text), h(neg), [h(b) for b in stream_batches], AdaNegConfig(emulate_fp16=True))
    assert r["max_abs_diff"] == 0.0 and r["pred_agreement"] == 1.0, r


def test_adaneg_fp16_entropy_is_nan():
    """Documented finding: the official fp16 entropy is NaN, so memory is never replaced."""
    p = torch.softmax(torch.randn(4, 1000).half() * 30, dim=1)
    e = -(p * torch.log(p + 1e-8)).sum(dim=-1)
    assert torch.isnan(e).all()


def test_variant_lcb_reduces_to_tanl(bank, stream_batches):
    """A modification must reproduce TANL exactly at its neutral setting (kappa = 0)."""
    from oodlab.methods import TANL
    from oodlab.methods.variants import TANL_LCB

    cfg = TANLConfig.paper().with_(**SMALL)
    a = TANL(bank.id_text, bank.corpus, bank.noise, cfg)
    b = TANL_LCB(bank.id_text, bank.corpus, bank.noise, cfg, kappa=0.0)
    c = TANL_LCB(bank.id_text, bank.corpus, bank.noise, cfg, kappa=2.0)
    sa = torch.cat([a.step(x).score for x in stream_batches])
    sb = torch.cat([b.step(x).score for x in stream_batches])
    sc = torch.cat([c.step(x).score for x in stream_batches])
    assert torch.equal(sa, sb)
    assert not torch.equal(sa, sc)
