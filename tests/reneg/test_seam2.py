"""Seam 2 on the synthetic world: the gate admits names that catch OOD images, and stops names that steal ID."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from synthetic_world import make_world  # noqa: E402

from reneg.packs import load_kg_hood, load_kg_pool  # noqa: E402
from reneg.seam2 import Seam2, Seam2Config, id_model_from_stats  # noqa: E402
from reneg.transport import l2n, oracle_stats  # noqa: E402


def _setup(seed=0):
    w = make_world(C=40, D=48, seed=seed, imgs_per_class=30, n_ssb=30, spread=0.6, noise=0.8)
    X_id = torch.tensor(w.sets["imagenet_test"]["feats"]).double()
    y_id = w.sets["imagenet_test"]["labels"]
    st = oracle_stats(X_id[::2], y_id[::2], 40)
    idm = id_model_from_stats(st, 3, X_id[1::3])
    X_ood = torch.tensor(w.sets["ssb_hard"]["feats"]).double()
    return w, idm, X_id[1::2], X_ood


def test_soft_gate_runs_and_limits_stealing():
    w, idm, X_id, X_ood = _setup()
    names = [n for n in w.name_to_text if n not in w.id_names][:200]
    pool = torch.tensor(np.stack([w.name_to_text[n] for n in names])).double()
    # add strong stealers: near copies of ID names
    steal = l2n(torch.tensor(w.id_text[:20]).double() + 0.05 * torch.randn(20, w.dim, dtype=torch.float64))
    pool = torch.cat([pool, steal])
    cluster = np.arange(len(pool)) % 25
    bg = l2n(torch.randn(500, w.dim, dtype=torch.float64))
    X = torch.cat([X_id, X_ood])
    perm = np.random.default_rng(0).permutation(len(X))
    for cfg in (Seam2Config(gate=False), Seam2Config(prior=0.05, delta_in=1.0, beta=0.5, n_ev=2)):
        m = Seam2(w.id_text, w.neg_text, pool, cluster, idm, bg, cfg)
        s = m.run(X[torch.as_tensor(perm)])
        assert np.isfinite(s).all() and len(s) == len(X)
        if cfg.gate:
            pi_steal, pi_rest = m.pi[-20:].mean(), m.pi[:-20].mean()
            assert pi_steal <= pi_rest + 0.05, (pi_steal, pi_rest)


def test_kg_resources():
    pool = load_kg_pool()
    assert pool.get("version") == 2 and {"rel", "lcs", "cluster"} <= set(pool["pool"][0])
    hood = load_kg_hood()
    assert len(hood["hood"]) == 1000 and hood["relations"][0] == "sibling"
