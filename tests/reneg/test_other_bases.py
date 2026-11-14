"""ReNeg's heads on other base detectors (NegLabel, AdaNeg, DDE): the DDE re-implementation equals the official
recomputation, and HeadsOnBase reads only the base's score.

Needs the real oodlab (NegLabel, AdaNeg); skipped with the test stub."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

tanl_mod = pytest.importorskip("oodlab.methods.tanl")
if not hasattr(tanl_mod.TANL, "selected_T"):
    pytest.skip("oodlab is the test stub, not the real package", allow_module_level=True)

from oodlab.methods.adaneg import AdaNeg, AdaNegConfig  # noqa: E402
from oodlab.methods.neglabel import NegLabel, NegLabelConfig  # noqa: E402
from reneg.core import ReNegConfig  # noqa: E402
from reneg.dde import DDEConfig, DDEFeat, selfcheck  # noqa: E402
from reneg.multi import HeadsOnBase, ReNegHead, logit64  # noqa: E402
from reneg.transport import l2n  # noqa: E402

BASE = dict(n_min=1, pool_frac=0.0, pool_in_text=False, id_admit="softmax+base")
MODES = {"balanced": ReNegConfig(**BASE, image_mode="kg_clip", vote=True), "max": ReNegConfig(**BASE),
         "safe": ReNegConfig(**BASE, image_mode="kg_clip_caught", vote=True)}


def _world(seed=0, C=20, M=200, D=32):
    g = torch.Generator().manual_seed(seed)
    id_text = l2n(torch.randn(C, D, generator=g))
    neg = l2n(torch.randn(M, D, generator=g))
    noise = l2n(torch.randn(15, D, generator=g))
    X = l2n(torch.cat([id_text[torch.randint(0, C, (300,), generator=g)] + 0.6 * torch.randn(300, D, generator=g),
                       neg[torch.randint(0, M, (200,), generator=g)] + 0.6 * torch.randn(200, D, generator=g)]))
    pool = l2n(neg[:40] + 0.3 * torch.randn(40, D, generator=g))
    return id_text, neg, noise, X[torch.randperm(500, generator=g)], pool


@pytest.mark.parametrize("duplicate", [True, False])
def test_dde_running_sums_equal_the_official_recomputation(duplicate):
    worst, n_id, n_ood = selfcheck(duplicate=duplicate)
    assert n_id > 64 and n_ood > 64                      # both caches overflowed, so eviction was exercised
    assert worst < 1e-5


def test_dde_score_is_a_probability_and_reset_restores_the_start():
    id_text, neg, noise, X, _ = _world(2)
    m = DDEFeat(id_text, neg, noise, DDEConfig(hat_M=30, cache_size=100, logit_scale=20.0))
    a = torch.cat([m.step(X[s:s + 64]).score for s in range(0, len(X), 64)])
    assert float(a.min()) >= 0 and float(a.max()) <= 1
    m.reset()
    b = torch.cat([m.step(X[s:s + 64]).score for s in range(0, len(X), 64)])
    assert torch.equal(a, b)


def _bases(id_text, neg, noise):
    ada = AdaNegConfig.official().with_(logit_scale=20.0)
    return {"neglabel": (NegLabel(id_text, neg, NegLabelConfig(group_num=10, logit_scale=20.0)), 1.0),
            "adaneg": (AdaNeg(id_text, neg, ada), 1.0 + ada.lambda_),
            "dde": (DDEFeat(id_text, neg, noise, DDEConfig(hat_M=30, cache_size=100, logit_scale=20.0)), 1.0)}


@pytest.mark.parametrize("name", ["neglabel", "adaneg", "dde"])
def test_heads_on_base_equals_base_then_head(name):
    id_text, neg, noise, X, pool = _world(3)
    cl = np.arange(40) % 7
    base, smax = _bases(id_text, neg, noise)[name]
    ref_base, _ = _bases(id_text, neg, noise)[name]
    m = HeadsOnBase(base, name, id_text, pool, cl, MODES, score_max=smax)
    heads = {k: ReNegHead(id_text, pool, cl, rc) for k, rc in MODES.items()}
    assert m.score_names == [name] + [f"{k} on {name}" for k in MODES]
    for s in range(0, len(X), 64):
        out = m.step(X[s:s + 64])
        o = ref_base.step(X[s:s + 64])
        assert torch.allclose(out.info["scores"][name], o.score.float())
        lt = logit64(o.score / smax)
        assert torch.isfinite(lt).all()
        for k, h in heads.items():
            ref = h.step(l2n(X[s:s + 64].float()), lt).float()
            assert torch.allclose(out.info["scores"][f"{k} on {name}"], ref, atol=1e-6), (name, k, s)
        assert torch.equal(out.score.cpu(), out.info["scores"][f"balanced on {name}"])


def test_head_with_zero_weight_keeps_the_base_ranking():
    id_text, neg, noise, X, pool = _world(4)
    base, smax = _bases(id_text, neg, noise)["neglabel"]
    m = HeadsOnBase(base, "neglabel", id_text, pool, np.arange(40) % 7,
                    {"balanced": MODES["balanced"].with_(omega=0.0)}, score_max=smax)
    s_base, s_head = [], []
    for s in range(0, len(X), 64):
        out = m.step(X[s:s + 64])
        s_base.append(out.info["scores"]["neglabel"])
        s_head.append(out.info["scores"]["balanced on neglabel"])
    a, b = torch.cat(s_base).double(), torch.cat(s_head).double()
    keep = (a > 1e-6) & (a < 1 - 1e-6)                    # logit clamps the extremes
    assert torch.equal(a[keep].argsort(), b[keep].argsort())


def test_dde_paper_and_release_settings():
    p, r = DDEConfig.paper(), DDEConfig.release()
    assert (p.ood_thres, p.cache_size, p.hat_M) == (0.25, 1000, 500)        # Sec. 4.1 of the paper
    assert (r.ood_thres, r.cache_size, r.hat_M) == (0.10, 5000, 500)        # the released postprocessor
    assert p.thres + p.gap * (1 - p.thres) == r.thres + r.gap * (1 - r.thres) == 0.75
