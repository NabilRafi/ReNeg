"""ReNeg on top of TANL: with the pool and the image score off it ranks images exactly like TANL.

Needs the real oodlab (Colab after C0, or a local copy on sys.path); skipped with the test stub.
"""
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
from reneg.reneg_tanl import ReNegTANL, logit  # noqa: E402
from reneg.transport import l2n  # noqa: E402


def _world(seed=0, C=20, N=600, D=32):
    g = torch.Generator().manual_seed(seed)
    id_text = l2n(torch.randn(C, D, generator=g))
    corpus = l2n(torch.randn(N, D, generator=g))
    noise = l2n(torch.randn(15, D, generator=g))
    X = l2n(torch.cat([id_text[torch.randint(0, C, (300,), generator=g)] + 0.6 * torch.randn(300, D, generator=g),
                       torch.randn(200, D, generator=g)]))
    return id_text, corpus, noise, X[torch.randperm(500, generator=g)]


def test_reneg_tanl_reduces_to_tanl():
    id_text, corpus, noise, X = _world()
    cfg = tanl_mod.TANLConfig().with_(num_neg=100, queue_len=50)
    t = tanl_mod.TANL(id_text, corpus, noise, cfg)
    r = ReNegTANL(id_text, corpus, noise, pool_text=None, cfg=cfg, rcfg=ReNegConfig(use_pool=False, use_image=False))
    t.reset()
    r.reset()
    for s in range(0, len(X), 64):
        a = t.step(X[s : s + 64]).score.double()
        b = r.step(X[s : s + 64]).score.double()
        assert torch.allclose(logit(a), b, atol=1e-4)           # ReNeg reports TANL's score as log-odds


def test_reneg_tanl_runs_with_pool_and_image():
    id_text, corpus, noise, X = _world(1)
    g = torch.Generator().manual_seed(2)
    pool = l2n(torch.randn(40, id_text.shape[1], generator=g))
    cfg = tanl_mod.TANLConfig().with_(num_neg=100, queue_len=50)
    r = ReNegTANL(id_text, corpus, noise, pool_text=pool, pool_cluster=np.arange(40) % 7, cfg=cfg,
                  rcfg=ReNegConfig(n_min=1))
    r.reset()
    out = [r.step(X[s : s + 64]) for s in range(0, len(X), 64)]
    s = torch.cat([o.score for o in out])
    assert s.shape == (500,) and torch.isfinite(s).all()
    assert r.gate.summary()["caught"] > 0 and r.idm.ready()


def test_reneg_tanl_agreed_admission_and_id_prior():
    id_text, corpus, noise, X = _world(3)
    g = torch.Generator().manual_seed(4)
    pool = l2n(torch.randn(40, id_text.shape[1], generator=g))
    cfg = tanl_mod.TANLConfig().with_(num_neg=100, queue_len=50)
    Xp = l2n(id_text.repeat(2, 1) + 0.3 * torch.randn(40, id_text.shape[1], generator=g))
    yp = np.r_[np.arange(20), np.arange(20)]
    r = ReNegTANL(id_text, corpus, noise, pool_text=pool, pool_cluster=np.arange(40) % 7, cfg=cfg,
                  rcfg=ReNegConfig(n_min=1, image_mode="kg_clip", vote=True, id_admit="softmax+base"),
                  id_prior=(Xp, yp))
    r.reset()
    assert r.idm.ready() and r.idm.counts.sum() == 40          # the prior is in before the first batch
    out = [r.step(X[s : s + 64]) for s in range(0, len(X), 64)]
    assert torch.isfinite(torch.cat([o.score for o in out])).all()
    r.reset()
    assert r.idm.counts.sum() == 40                            # and again after every reset
