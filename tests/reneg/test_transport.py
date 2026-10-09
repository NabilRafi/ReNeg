"""Closed forms of the transport ladder, checked against independent computations."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reneg.transport import (ClassStats, candidate_grid, change_profile, fit_transport, l2n, pseudo_label_stats,
                             zero_shot)


def _toy(C=40, D=16, seed=0, n=12):
    g = torch.Generator().manual_seed(seed)
    T = l2n(torch.randn(C, D, generator=g, dtype=torch.float64))
    A = torch.eye(D, dtype=torch.float64) + 0.3 * torch.randn(D, D, generator=g, dtype=torch.float64) / D ** 0.5
    gap = 0.7 * l2n(torch.randn(D, generator=g, dtype=torch.float64))
    U = l2n(T @ A.t() + gap)
    feats, labels = [], []
    for c in range(C):
        feats.append(l2n(U[c] + 0.2 * torch.randn(n, D, generator=g, dtype=torch.float64) / D ** 0.5))
        labels += [c] * n
    st = ClassStats.empty(C, D).add(torch.cat(feats), np.array(labels))
    return T, st, A, gap


def _weighted_ridge_direct(T, st, lam_abs, n_cap=20):
    """Solve min sum w||y - W x||^2 + lam||W - I||^2 row by row with lstsq on stacked data."""
    V = st.directions()
    w = st.counts.clamp(max=n_cap)
    mT = (w[:, None] * T).sum(0) / w.sum()
    mV = (w[:, None] * V).sum(0) / w.sum()
    X, Y = T - mT, V - mV
    D = X.shape[1]
    sw = w.sqrt()[:, None]
    # stack sqrt(w) X ; sqrt(lam) I  ->  targets sqrt(w) Y ; sqrt(lam) I   (row convention: Y ~ X W^T)
    Xs = torch.cat([sw * X, lam_abs ** 0.5 * torch.eye(D, dtype=torch.float64)])
    Ys = torch.cat([sw * Y, lam_abs ** 0.5 * torch.eye(D, dtype=torch.float64)])
    Wt = torch.linalg.lstsq(Xs, Ys).solution
    return Wt.t(), mT, mV


def test_r2_matches_direct_ridge():
    T, st, _, _ = _toy()
    for lam in (1e-3, 0.1, 5.0):
        tr = fit_transport("R2", T, st, n_min=1, lam=lam, lam_is_relative=False)
        W, mT, mV = _weighted_ridge_direct(T, st, lam)
        assert torch.allclose(tr.W, W, atol=1e-8), lam
        assert torch.allclose(tr.m_T, mT) and torch.allclose(tr.m_V, mV)


def test_large_lambda_gives_gap_shift():
    T, st, _, _ = _toy()
    r1 = fit_transport("R1", T, st, n_min=1)
    r2 = fit_transport("R2", T, st, n_min=1, lam=1e9)
    q = l2n(torch.randn(25, T.shape[1], dtype=torch.float64))
    assert torch.allclose(r1.apply(q), r2.apply(q), atol=1e-5)


def test_kernel_view_equals_r2():
    """B2 with the centred linear kernel is exactly R2 (the 'plain words' formula in the spec)."""
    T, st, _, _ = _toy()
    for lam in (0.01, 0.3):
        r2 = fit_transport("R2", T, st, n_min=1, lam=lam, lam_is_relative=False)
        b2 = fit_transport("B2", T, st, n_min=1, lam=lam, kernel="linear", lam_is_relative=False)
        q = l2n(torch.randn(30, T.shape[1], dtype=torch.float64))
        assert torch.allclose(r2.apply(q), b2.apply(q), atol=1e-5)


def test_procrustes_is_rotation_and_anchored():
    T, st, _, _ = _toy()
    b1 = fit_transport("B1", T, st, n_min=1, lam=0.1)
    I = torch.eye(T.shape[1], dtype=torch.float64)
    assert torch.allclose(b1.W @ b1.W.t(), I, atol=1e-8)
    b1_big = fit_transport("B1", T, st, n_min=1, lam=1e9)
    assert torch.allclose(b1_big.W, I, atol=1e-6)


def test_procrustes_recovers_a_rotation():
    g = torch.Generator().manual_seed(3)
    C, D = 60, 12
    T = l2n(torch.randn(C, D, generator=g, dtype=torch.float64))
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    V = T @ Q.t()
    st = ClassStats.empty(C, D).add(V, np.arange(C))
    b1 = fit_transport("B1", T, st, n_min=1, lam=1e-9, lam_is_relative=False)
    q = l2n(torch.randn(10, D, generator=g, dtype=torch.float64))
    target = l2n(b1.m_V + (q - b1.m_T) @ Q.t())
    assert torch.allclose(b1.apply(q).double(), target, atol=1e-4)


def test_transport_beats_raw_text_on_known_map():
    T, st, A, gap = _toy(C=80, D=16, n=30)
    g = torch.Generator().manual_seed(9)
    q = l2n(torch.randn(50, T.shape[1], generator=g, dtype=torch.float64))
    truth = l2n(q @ A.t() + gap)
    cos = lambda P: float((P.double() * truth).sum(1).mean())
    r0 = cos(fit_transport("R0", T, st).apply(q))
    r1 = cos(fit_transport("R1", T, st, n_min=1).apply(q))
    r2 = cos(fit_transport("R2", T, st, n_min=1, lam=0.03).apply(q))
    assert r1 > r0 and r2 > r1, (r0, r1, r2)


def test_eigen_bound_and_profile():
    T, st, _, _ = _toy()
    tr = fit_transport("R2", T, st, n_min=1, lam=0.3)
    prof = change_profile(tr, T, st, n_min=1)
    assert prof["bound_holds"]


def test_fallback_to_r0_without_classes():
    T, st, _, _ = _toy()
    empty = ClassStats.empty(st.n_classes, st.dim)
    tr = fit_transport("R2", T, empty)
    assert tr.kind == "R0"


def test_zero_shot_and_pseudo_labels():
    T, st, A, gap = _toy()
    feats = l2n(T.float() + 0.01 * torch.randn(T.shape))
    pred, prob = zero_shot(feats, T.float(), 100.0)
    assert (pred == np.arange(len(T))).mean() > 0.9 and prob.min() > 0
    s, info = pseudo_label_stats(feats, T.float(), 100.0, p_min=0.5, true_labels=np.arange(len(T)))
    assert info["n_kept"] > 0 and info["pseudo_label_acc_kept"] > 0.9


def test_candidate_grid_size():
    assert len(candidate_grid()) == 2 + 8 + 8 + 24
