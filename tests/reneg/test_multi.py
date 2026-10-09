"""ReNeg heads on a shared base: on TANL a head gives ReNegTANL's scores; MultiBase runs TINS + TANL + heads.

Needs the real oodlab (TANL; the vendored CLIP for TINS); skipped with the test stub."""
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

from reneg.core import ReNegConfig  # noqa: E402
from reneg.multi import HeadSpec, MultiBase, ReNegHead, logit64  # noqa: E402
from reneg.reneg_tanl import ReNegTANL  # noqa: E402
from reneg.transport import l2n  # noqa: E402

BASE = dict(n_min=1, pool_frac=0.0, pool_in_text=False, id_admit="softmax+base")
MODES = {"balanced": ReNegConfig(**BASE, image_mode="kg_clip", vote=True), "max": ReNegConfig(**BASE),
         "safe": ReNegConfig(**BASE, image_mode="kg_clip_caught", vote=True)}


def _world(seed=0, C=20, N=600, D=32):
    g = torch.Generator().manual_seed(seed)
    id_text = l2n(torch.randn(C, D, generator=g))
    corpus = l2n(torch.randn(N, D, generator=g))
    noise = l2n(torch.randn(15, D, generator=g))
    X = l2n(torch.cat([id_text[torch.randint(0, C, (300,), generator=g)] + 0.6 * torch.randn(300, D, generator=g),
                       torch.randn(200, D, generator=g)]))
    pool = l2n(torch.randn(40, D, generator=g))
    Xp = l2n(id_text.repeat(2, 1) + 0.3 * torch.randn(2 * C, D, generator=g))
    yp = np.r_[np.arange(C), np.arange(C)]
    return id_text, corpus, noise, X[torch.randperm(500, generator=g)], pool, (Xp, yp)


@pytest.mark.parametrize("mode", ["balanced", "max", "safe"])
def test_head_on_tanl_equals_reneg_tanl(mode):
    id_text, corpus, noise, X, pool, prior = _world(1)
    cl = np.arange(40) % 7
    cfg = tanl_mod.TANLConfig().with_(num_neg=100, queue_len=50)
    t = tanl_mod.TANL(id_text, corpus, noise, cfg.with_(record=True))
    r = ReNegTANL(id_text, corpus, noise, pool_text=pool, pool_cluster=cl, cfg=cfg, rcfg=MODES[mode], id_prior=prior)
    h = ReNegHead(id_text, pool, cl, MODES[mode], id_prior=prior)
    t.reset(), r.reset(), h.reset()
    for s in range(0, len(X), 64):
        out = t.step(X[s:s + 64])
        a = h.step(X[s:s + 64], logit64(out.score), extra_mask=out.info["id_mask"].numpy(), use_vote_mask=False)
        b = r.step(X[s:s + 64]).score
        assert torch.allclose(a.double(), b.double(), atol=1e-4), (mode, s, float((a.double() - b.double()).abs().max()))


class _FakeTINS:
    """Stands in for TINSFeat (same interface): a fixed text score, no inversion."""
    def __init__(self, id_text, neg):
        self.T, self.N = id_text, neg

    def reset(self):
        pass

    def step(self, feats):
        from oodlab.methods.base import StepOutput
        v = l2n(feats.float())
        a = torch.logsumexp(100 * v @ self.T.T, 1)
        b = torch.logsumexp(100 * v @ self.N.T, 1)
        return StepOutput(pred=(v @ self.T.T).argmax(1), score=torch.sigmoid(a - b), info={})


def test_multibase_scores_every_head_and_resets():
    id_text, corpus, noise, X, pool, prior = _world(2)
    cl = np.arange(40) % 7
    cfg = tanl_mod.TANLConfig().with_(num_neg=100, queue_len=50, record=True)
    pools = {"blind": (pool[:30], cl[:30]), "full": (pool, cl)}
    heads = [HeadSpec("balanced on tins", "tins", MODES["balanced"]),
             HeadSpec("max on tins, no prior", "tins", MODES["max"], prior=False),
             HeadSpec("balanced on tanl+tins", "tanl+tins", MODES["balanced"], pool="full"),
             HeadSpec("balanced on tanl", "tanl", MODES["balanced"])]
    m = MultiBase(_FakeTINS(id_text, corpus[:100]), tanl_mod.TANL(id_text, corpus, noise, cfg), id_text, pools, heads,
                  primary="balanced on tins", id_prior=prior)
    m.cfg = m.cfg.with_(init_seed=3)
    m.reset()
    assert m.tanl.cfg.init_seed == 3
    outs = [m.step(X[s:s + 64]) for s in range(0, len(X), 64)]
    for k in ("tins", "tanl", "tanl+tins", "balanced on tins", "max on tins, no prior", "balanced on tanl+tins",
              "balanced on tanl"):
        v = torch.cat([o.info["scores"][k] for o in outs])
        assert v.shape == (500,) and torch.isfinite(v).all(), k
    assert torch.equal(torch.cat([o.score for o in outs]), torch.cat([o.info["scores"]["balanced on tins"] for o in outs]))
    # the head on TANL inside MultiBase equals a standalone ReNegTANL with the same seed
    r = ReNegTANL(id_text, corpus, noise, pool_text=pool[:30], pool_cluster=cl[:30], cfg=cfg.with_(init_seed=3),
                  rcfg=MODES["balanced"], id_prior=prior)
    r.reset()
    ref = torch.cat([r.step(X[s:s + 64]).score for s in range(0, len(X), 64)]).double()
    got = torch.cat([o.info["scores"]["balanced on tanl"] for o in outs]).double()
    assert torch.allclose(got, ref, atol=1e-4)
    m.reset()
    assert all(h.idm.counts.sum() == (40 if spec.prior else 0) for spec, h in zip(m.specs, m.heads.values()))
