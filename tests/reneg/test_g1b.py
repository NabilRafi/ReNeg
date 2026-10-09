"""G1b: the corrected transport diagnostics, end to end on the synthetic world."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from synthetic_world import FakeCache, fake_bank, make_world  # noqa: E402

from reneg.g1b import G1b, G1bConfig, cfg_name, lse_taus  # noqa: E402
from reneg.transport import Transport, l2n  # noqa: E402
from reneg.wordnet import get_wordnet  # noqa: E402


def _small_cfg(**kw):
    base = dict(p_min=0.3, n_min=3, n_folds=4, r2_lams=(0.1, 1.0), b1_lams=(1.0,), b2_hs=(0.2,), b2_lams=(1.0,),
                bprime_negatives=200, top_maps=2, cold_ns=(64, 256), concept_min_images=6)
    base.update(kw)
    return G1bConfig(**base)


def test_r1_centred_is_raw_text_centred():
    """In the centred geometry the gap shift is raw text again: R1(t) - m_V = t - m_T."""
    rng = np.random.default_rng(0)
    t = l2n(torch.tensor(rng.normal(size=(30, 16))))
    m_T, m_V = t.mean(0), l2n(torch.tensor(rng.normal(size=16)))
    r1 = Transport("R1", m_T=m_T, m_V=m_V)
    assert torch.allclose(r1.raw(t) - m_V, t - m_T, atol=1e-12)


def test_shared_shift_cancels_in_the_score():
    """A shift added to every prototype's similarity for one image cancels in LSE(pos) - LSE(neg)."""
    rng = np.random.default_rng(1)
    X = l2n(torch.tensor(rng.normal(size=(50, 8)), dtype=torch.float32))
    P = l2n(torch.tensor(rng.normal(size=(10, 8)), dtype=torch.float32))
    N = l2n(torch.tensor(rng.normal(size=(40, 8)), dtype=torch.float32))
    a, b = lse_taus(X, P, N, [20.0])
    shift = 3.0 * torch.rand(50, 1)                 # per-image constant
    a2 = torch.logsumexp(20.0 * (X @ P.t() + shift), 1).numpy()
    b2 = torch.logsumexp(20.0 * (X @ N.t() + shift), 1).numpy()
    assert np.allclose(a[:, 0] - b[:, 0], a2 - b2, atol=1e-4)


def test_g1b_end_to_end(tmp_path):
    w = make_world(C=60, D=48, seed=3, imgs_per_class=20)
    cache, bank = FakeCache(w), fake_bank(w)
    g = G1b(cache, bank, _small_cfg())
    prep = g.prepare()
    assert prep["half_A"] + prep["half_B"] == len(w.sets["imagenet_test"]["labels"])
    wn = get_wordnet(download=False)
    b = g.part_b(wn)
    bt = g.tables["bprime"]
    assert {cfg_name("R0", "raw"), cfg_name("R0", "cent"), cfg_name("R1", "raw")} <= set(bt.config)
    assert bt.pood_auroc.between(0, 100).all() and bt.catch.between(0, 100).all()
    h = g.part_h()
    assert h["r0_raw_top1"] > 50                    # the synthetic world is easy for raw text
    d = g.part_d()
    dt = g.tables["dprime"]
    assert (dt.role == "NegLabel form").sum() == 1 and "near_fpr95" in dt.columns
    assert d["T_star"].split("|")[0] not in ("R0",)
    enc = lambda names: torch.nn.functional.normalize(w.encode(names), dim=1)
    o = g.part_o(enc, wn)
    ot = g.tables["oracle"]
    assert list(ot.variant) == ["text", "text + names", "map", "map + names", "map + true centroids"]
    assert ot.id_stolen.dropna().between(0, 100).all()
    dec = g.decide()
    assert dec["verdict"] in ("KEEP-TRANSPORT", "CENTER-ONLY", "NO-STATIC-GAIN")
    e = g.part_e()
    assert ("ran" in e) and (e["ran"] == (dec["verdict"] == "KEEP-TRANSPORT"))
    path = g.save(str(tmp_path))
    txt = open(path).read()
    assert "VERDICT" in txt and "PLAN B" in txt
    res = json.load(open(os.path.join(tmp_path, "g1b_results.json")))
    assert res["decision"]["verdict"] == dec["verdict"]
    packs = g.export_pack(str(tmp_path / "pack"))
    z1, z2 = np.load(packs[0]), np.load(packs[1])
    assert z1["id_text"].shape == (60, 48) and z1["val_feats"].dtype == np.float16
    assert "test_B_feats" in z2.files and len(z2["test_B_labels"]) == len(z2["test_B_feats"])
    print(txt[:4000])


def test_perfect_transport_is_the_ceiling():
    """With the real centroids of the oracle concepts, near-OOD cannot get worse than with their mapped names
    by much: the ceiling row must be at least as good as the mapped names, up to noise."""
    w = make_world(C=60, D=48, seed=4, imgs_per_class=20, distortion=0.8)
    g = G1b(FakeCache(w), fake_bank(w), _small_cfg())
    g.prepare()
    g.part_b(get_wordnet(download=False))
    g.part_h()
    g.part_d()
    g.part_o(lambda names: torch.nn.functional.normalize(w.encode(names), dim=1), get_wordnet(download=False))
    near = g.results["O"]["near_fpr95"]
    assert near["map + true centroids"] <= near["map + names"] + 5.0, near
