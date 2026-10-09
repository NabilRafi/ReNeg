"""Self-check, concept test, scores, cold start and the decision, end to end on a synthetic world."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from synthetic_world import FakeCache, fake_bank, make_world  # noqa: E402

from reneg.concepts import build_concepts, folder_of, readable  # noqa: E402
from reneg.g1 import G1, G1Config  # noqa: E402
from reneg.scores import neg_score, pair_metrics  # noqa: E402
from reneg.selfcheck import choose_ladder, leave_group_out, make_folds  # noqa: E402
from reneg.transport import candidate_grid, oracle_stats  # noqa: E402
from reneg.wordnet import get_wordnet, subtree_groups, text_cluster_groups  # noqa: E402


def test_make_folds_partition():
    groups = np.repeat(np.arange(9), 7)
    classes = np.arange(63)
    folds = make_folds(groups, classes, n_folds=4)
    allc = np.concatenate(folds)
    assert sorted(allc.tolist()) == classes.tolist()
    for f in folds:                              # whole groups only
        for g in np.unique(groups[f]):
            assert set(np.where(groups == g)[0]) <= set(f.tolist())


def test_selfcheck_prefers_transport_on_synthetic_world():
    w = make_world(C=60, D=48, seed=1, spread=0.3, distortion=0.8)   # fine-grained siblings: retrieval is hard
    bank = fake_bank(w)
    s = w.sets["imagenet_test"]
    st = oracle_stats(torch.tensor(s["feats"]), s["labels"], len(w.id_names))
    groups, _ = text_cluster_groups(bank.id_text, n_groups=8)
    df = leave_group_out(bank.id_text, st, groups, candidate_grid(lams=(0.01, 0.1, 1.0), hs=(0.1,)), n_min=3, n_folds=4)
    ch = choose_ladder(df, eps=0.01)
    r0 = float(df[df.candidate == "R0"].top1.iloc[0])
    assert ch["chosen"] != "R0", df.head()
    assert ch["gain_top1_over_R0"] > 0.02 or ch["gain_align_over_R0"] > 0.02, (r0, ch)


def test_ladder_uses_alignment_when_retrieval_saturates():
    w = make_world(C=60, D=48, seed=1)                                # easy world: every map retrieves perfectly
    bank = fake_bank(w)
    s = w.sets["imagenet_test"]
    st = oracle_stats(torch.tensor(s["feats"]), s["labels"], len(w.id_names))
    groups, _ = text_cluster_groups(bank.id_text, n_groups=8)
    df = leave_group_out(bank.id_text, st, groups, candidate_grid(lams=(0.01, 0.1, 1.0), hs=(0.1,)), n_min=3, n_folds=4)
    ch = choose_ladder(df)
    assert ch["chosen"] != "R0" and ch["gain_align_over_R0"] > 0.1, ch


def test_concept_parsing():
    assert folder_of("ssb_hard/n04542943/x.JPEG") == "n04542943"
    assert folder_of("texture/banded/banded_0002.jpg") == "banded"
    assert readable("amphiuma_means") == "amphiuma means"
    wn = get_wordnet(download=False)
    if wn is not None:
        assert readable("n02099601", wn) == "golden retriever"


def test_wordnet_groups_split_big_subtrees():
    wn = get_wordnet(download=False)
    if wn is None:
        return
    w = make_world(C=60, D=16, seed=2)
    g, names = subtree_groups(w.id_wnids, wn, max_size=10)
    assert np.bincount(g).max() <= 10 or len(names) > 1


def test_scores_and_metrics():
    rng = np.random.default_rng(0)
    pos = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
    neg = torch.nn.functional.normalize(torch.randn(30, 8), dim=1)
    idv = torch.nn.functional.normalize(pos[rng.integers(0, 10, 200)] + 0.1 * torch.randn(200, 8), dim=1)
    oodv = torch.nn.functional.normalize(neg[rng.integers(0, 30, 200)] + 0.1 * torch.randn(200, 8), dim=1)
    a, b = neg_score(idv, pos, neg, 50.0), neg_score(oodv, pos, neg, 50.0)
    assert (a >= 0).all() and (a <= 1).all()
    m = pair_metrics(a, b)
    assert m["auroc"] > 95 and m["fpr95"] < 20


def test_g1_end_to_end(tmp_path):
    w = make_world(C=60, D=48, seed=3)
    cache, bank = FakeCache(w), fake_bank(w)
    cfg = G1Config(p_min=0.3, n_min=3, lams=(0.01, 0.1, 1.0), hs=(0.1,), n_folds=4, cold_ns=(64, 256, 1024),
                   cold_min_classes=10)
    g = G1(cache, bank, cfg)
    a = g.part_a()
    assert a["n_classes_eligible"] > 30
    wn = get_wordnet(download=False)
    b = g.part_b(wn)
    assert b["chosen"] in g.fits
    c = g.part_c(lambda names: torch.nn.functional.normalize(w.encode(names), dim=1), wn)
    assert c["encoder_ok"]
    d = g.part_d()
    assert "text_near_fpr95" in d
    e = g.part_e()
    dec = g.decide()
    assert dec["verdict"] in ("PASS", "PARTIAL", "R1-ONLY", "FAIL")
    path = g.save(str(tmp_path))
    assert os.path.isfile(path)
    res = json.load(open(os.path.join(tmp_path, "g1_results.json")))
    assert res["decision"]["verdict"] == dec["verdict"]
    print(open(path).read()[:3000])


def test_validation_judge_removes_pseudo_label_bias():
    """Pseudo-label centroids favour raw text on top-1; judging on labelled validation centroids does not."""
    from reneg.selfcheck import choose_online
    from reneg.transport import pseudo_label_stats

    w = make_world(C=300, D=96, n_neg=500, imgs_per_class=12, seed=0, spread=0.5, distortion=0.6, noise=1.0)
    bank = fake_bank(w)
    s, v = w.sets["imagenet_test"], w.sets["imagenet_val"]
    st, _ = pseudo_label_stats(torch.tensor(s["feats"]), bank.id_text, 100.0, 0.5)
    val = oracle_stats(torch.tensor(v["feats"]), v["labels"], len(w.id_names))
    groups, _ = text_cluster_groups(bank.id_text, n_groups=12)
    cands = candidate_grid(lams=(0.1, 1.0), hs=(0.2,))
    judged = leave_group_out(bank.id_text, st, groups, cands, n_folds=4, eval_stats=val)
    assert choose_ladder(judged)["chosen"] != "R0", judged.head()
    online = leave_group_out(bank.id_text, st, groups, cands, n_folds=4)
    assert choose_online(online)["chosen"] != "R0", online.head()
