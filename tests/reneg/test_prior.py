"""Plan B2 helpers: prior sampling round trip, the image-to-image map, alignment, centroids."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reneg.prior import (align_by_keys, apply_image_map, concept_centroids, fake_prior, fit_image_map,  # noqa: E402
                         load_prior_embeddings, run_prior)
from reneg.transport import l2n  # noqa: E402


def test_run_prior_round_trip_and_resume(tmp_path):
    prior = fake_prior(48)
    names = [f"name {i}" for i in range(11)]
    path = str(tmp_path / "p.npz")
    run_prior(prior, names[:5] + names[5:], path, n_per=3, batch=4, log=lambda *_: None)
    P, got, n_per = load_prior_embeddings(path)
    assert got == names and n_per == 3 and P.shape == (11, 48)
    ref = l2n(prior(["a photo of a name 0."], n_per=3, seed=0).mean(0, keepdim=True))
    assert float((P[0] * ref[0]).sum()) > 0.99                       # int8 storage keeps the prototype
    calls = []
    run_prior(lambda *a, **k: calls.append(1) or prior(*a, **k), names, path, n_per=3, batch=4, log=lambda *_: None)
    assert not calls                                                  # everything was already done: resumed


def test_image_map_recovers_a_linear_relation_and_generalises():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(40, 24, generator=g) / 6
    X = l2n(torch.randn(3000, 40, generator=g) + 2.0)                 # a cone, like CLIP image embeddings
    Y = l2n(X @ A + 0.3)
    W = fit_image_map(X[:2500], Y[:2500], lam=0.01)
    cos = (apply_image_map(W, X[2500:]) * Y[2500:]).sum(1)
    assert float(cos.mean()) > 0.99


def test_align_and_centroids():
    ia, ib = align_by_keys(["a", "b", "c", "d"], ["d", "x", "b"])
    assert ia.tolist() == [1, 3] and ib.tolist() == [2, 0]
    X = l2n(torch.randn(10, 8))
    groups = np.array([0, 0, 1, 1, -1, 2, 2, 2, 0, 1])
    C = concept_centroids(X, groups, 3)
    assert C.shape == (3, 8) and torch.allclose(C[2], l2n(X[[5, 6, 7]].sum(0, keepdim=True))[0], atol=1e-6)
    A, B = concept_centroids(X, groups, 3, split=np.arange(10) % 2 == 0)
    assert A.shape == B.shape == (3, 8)
