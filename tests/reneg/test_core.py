"""ReNeg core: KG gate, online ID model, NegLabel and TANL bases."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reneg.core import KGGate, OnlineIdModel, ReNegConfig, ReNegNegLabel, admit_mask  # noqa: E402
from reneg.reneg_tanl import tanl_score_with_pool  # noqa: E402
from reneg.transport import l2n  # noqa: E402


def _brute_tanl(lg, n_id, step, P=None):
    """The official loop: mean over prefixes of the ID softmax mass in [ID | first m negatives (+ pool)]."""
    n_neg = lg.shape[1] - n_id
    out = []
    for i in range(n_id, n_id + n_neg, step):
        sub = lg[:, : i + step].double()
        a = torch.logsumexp(sub[:, :n_id], 1)
        b = torch.logsumexp(sub[:, n_id:], 1)
        if P is not None:
            b = torch.logaddexp(b, P.double())
        out.append(torch.sigmoid(a - b))
    return torch.stack(out, 1).mean(1)


def test_tanl_score_matches_the_official_loop_with_and_without_pool():
    g = torch.Generator().manual_seed(0)
    lg = 100 * 0.03 * torch.randn(16, 50 + 40, generator=g)
    P = 100 * 0.03 * torch.randn(16, generator=g) + 2.0
    for step in (1, 3):
        assert float((tanl_score_with_pool(lg, 50, step, None).double() - _brute_tanl(lg, 50, step)).abs().max()) < 1e-5
        assert float((tanl_score_with_pool(lg, 50, step, P).double() - _brute_tanl(lg, 50, step, P)).abs().max()) < 1e-5
    # an empty pool (-inf mass) is TANL's score exactly
    none = tanl_score_with_pool(lg, 50, 1, None)
    empty = tanl_score_with_pool(lg, 50, 1, torch.full((16,), -float("inf")))
    assert torch.equal(none, empty)


def _toy(D=32, C=10, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    T_id = l2n(torch.randn(C, D, generator=g))
    pool = l2n(torch.randn(K, D, generator=g))
    return T_id, pool, g


def test_gate_admits_names_with_large_margins_and_keeps_the_prior_before_evidence():
    T_id, pool, g = _toy()
    cfg = ReNegConfig(prior=0.1, n_ev=2, delta=1.0, beta=0.5, temp=0.5)
    gate = KGGate(pool, np.array([0, 0, 1, 1, 2, 2]), cfg)
    assert np.allclose(gate.pi, 0.1)
    # images right on pool name 0 (large margin) and images between ID class 0 and pool name 2 (tiny margin)
    near0 = l2n(pool[0][None] + 0.05 * torch.randn(20, 32, generator=g))
    tie = l2n(0.5 * T_id[0][None] + 0.505 * pool[2][None] + 0.01 * torch.randn(20, 32, generator=g))
    for _ in range(3):
        gate.update(torch.cat([near0, tie]), T_id)
    assert gate.n[0] > 0 and gate.pi[0] > 0.8                       # large margins: admitted
    assert gate.n[2] > 0 and gate.pi[2] < 0.3                       # caught images only tie an ID name: kept out
    assert np.allclose(gate.pi[[3, 5]], 0.1)                          # no evidence: prior


def test_online_id_model_builds_prototypes_after_n_min():
    T_id, _, g = _toy()
    cfg = ReNegConfig(n_min=2, n_bg=50)
    m = OnlineIdModel(10, 32, cfg)
    assert not m.ready()
    X = l2n(T_id[[0, 0, 1]] + 0.1 * torch.randn(3, 32, generator=g))
    m.add(X, torch.tensor([0, 0, 1]), np.array([True, True, True]))
    assert m.ready() and m.prototypes().shape == (1, 32)             # class 0 has 2, class 1 only 1
    assert m.background().shape == (50, 32)
    lab, mask = admit_mask(X, T_id, cfg.with_(id_admit="all"))
    assert mask.all() and lab.shape == (3,)


def test_reneg_neglabel_reduces_to_neglabel_without_pool_and_image():
    T_id, pool, g = _toy()
    neg = l2n(torch.randn(40, 32, generator=g))
    X = l2n(torch.randn(300, 32, generator=g))
    cfg = ReNegConfig(use_pool=False, use_image=False, tau=100.0)
    m = ReNegNegLabel(T_id, neg, pool, np.zeros(6, int), cfg)
    s = m.run(X, batch=64)
    ref = (torch.logsumexp(100 * X @ T_id.T, 1) - torch.logsumexp(100 * X @ neg.T, 1)).numpy()
    assert np.abs(s - ref).max() < 1e-3


def test_reneg_neglabel_scores_before_it_updates():
    T_id, pool, g = _toy()
    neg = l2n(torch.randn(40, 32, generator=g))
    X = l2n(torch.randn(128, 32, generator=g))
    cfg = ReNegConfig(prior=0.2)
    m = ReNegNegLabel(T_id, neg, pool, np.arange(6), cfg)
    s_first = m.run(X[:64], batch=64)                                  # first batch: prior-weighted pool, no image side
    logw = torch.cat([torch.zeros(40), torch.log(torch.full((6,), 0.2))])
    negs = torch.cat([neg, pool])
    ref = (torch.logsumexp(100 * X[:64] @ T_id.T, 1) - torch.logsumexp(100 * X[:64] @ negs.T + logw, 1)).numpy()
    assert np.abs(s_first - ref).max() < 1e-3


def _device_run(device, mode):
    T_id, pool, g = _toy()
    neg = l2n(torch.randn(40, 32, generator=g))
    X = l2n(torch.randn(640, 32, generator=g) + 0.5 * T_id[torch.randint(0, 10, (640,), generator=g)])
    cfg = ReNegConfig(n_min=1, image_mode=mode, vote=True, n_bg=200, k_vis=1, n_ev=1)
    m = ReNegNegLabel(T_id, neg, pool, np.array([0, 0, 1, 1, 2, 2]), cfg, device=device)
    return m.run(X, batch=64), m


def test_device_argument_runs_and_keeps_state_on_that_device():
    for mode in ("plain", "kg_clip", "kg_clip_caught"):
        s, m = _device_run("cpu", mode)
        assert np.isfinite(s).all()
        assert m.gate.T.device.type == "cpu" and m.idm.sums.device.type == "cpu"
        if torch.cuda.is_available():                                   # Colab: the GPU run agrees with the CPU run
            s_gpu, m_gpu = _device_run("cuda", mode)
            assert m_gpu.gate.fsum.device.type == "cuda" and m_gpu.idm.noise.device.type == "cuda"
            assert np.abs(s_gpu - s).max() < 1e-2, (mode, np.abs(s_gpu - s).max())


def test_admit_mask_rules():
    T_id, _, g = _toy()
    X = l2n(torch.randn(50, 32, generator=g) + 2.0 * T_id[torch.randint(0, 10, (50,), generator=g)])
    base = np.zeros(50, bool); base[::2] = True
    cfg = ReNegConfig(p_min=0.5)
    lab, soft = admit_mask(X, T_id, cfg)
    _, b = admit_mask(X, T_id, cfg.with_(id_admit="base"), base_mask=base)
    _, both = admit_mask(X, T_id, cfg.with_(id_admit="softmax+base"), base_mask=base)
    assert np.array_equal(b, base) and np.array_equal(both, soft & base)
    try:
        admit_mask(X, T_id, cfg.with_(id_admit="softmax+base"))
        assert False, "needs the base mask"
    except ValueError:
        pass
