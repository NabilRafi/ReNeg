"""End-to-end on the fake Kaggle tree with a random tiny CLIP: text bank -> cache -> runner -> churn."""
import os

import numpy as np
import pytest
import torch

import oodlab
from oodlab.cache import FeatureCache, encode_manifest
from oodlab.data import DEFAULT_BUILD, build_manifest
from oodlab.data.openood import obtain_imglist_dir
from oodlab.methods import MCM, AdaNeg, AdaNegConfig, MCMConfig, NegLabel, NegLabelConfig, TANL, TANLConfig
from oodlab.runner import PROTOCOLS, Runner, acceptance_table, compare_with_targets, summarize
from oodlab.stream import pair_stream, temporal_stream


@pytest.fixture(scope="module")
def built(fake_kaggle, tiny_clip):
    from oodlab.textbank import build_textbank

    cache = FeatureCache(os.path.join(fake_kaggle["work"], "cache")).make()
    bank = build_textbank(tiny_clip, "ViT-B/16", oodlab.imagenet_classnames(), oodlab.CORPUS_DIR,
                          total_neg=500, corpus_limit=2000)
    cache.save_textbank(bank)
    imgl = obtain_imglist_dir(fake_kaggle["scratch"], [fake_kaggle["inputs"]], allow_download=False)
    icache = {}
    for name in DEFAULT_BUILD:
        m = build_manifest(name, [fake_kaggle["inputs"]], fake_kaggle["scratch"], imgl, allow_download=False,
                           strict_counts=False, _imagenet_cache=icache)
        encode_manifest(tiny_clip, m, cache, batch_size=32, num_workers=0, shard_size=150, log_every=10**6)
    return cache


def test_textbank_layout(built):
    b = built.load_textbank()
    assert len(set(b.words)) == len(b.words) == 2000
    assert b.n_selected == 500 and 0 < b.n_sel_adj < 500
    assert b.is_adj[: b.n_sel_adj].all() and not b.is_adj[b.n_sel_adj: b.n_selected].any()
    ms = b.mining_score
    assert (ms[: b.n_sel_adj].diff() >= 0).all()
    assert torch.allclose(b.corpus_text.float().norm(dim=1), torch.ones(len(b)), atol=1e-3)
    assert b.noise_feats.shape[0] == 15 and b.logit_scale > 0


def test_cache_contents(built):
    assert set(DEFAULT_BUILD) <= set(built.available())
    fs = built.load_set("imagenet_val_all")
    assert fs.feats.dtype == torch.float16 and len(fs.keys) == len(fs)
    info = oodlab.utils.read_json(built.info_path)
    assert info["textbank"] and "ninco" in info["sets"]


def test_encoding_resumes(built, fake_kaggle, tiny_clip, tmp_path):
    """A finished shard of the *same* image list is reused; shards of another list are discarded."""
    import json

    from oodlab.cache import shard_fingerprint
    from oodlab.data.manifest import Manifest

    m = Manifest.load(built.manifests_dir, "sun")
    sentinel = np.full((20, 64), 0.25, dtype=np.float16)    # fake "already encoded" shard (norm 2, not 1)

    def plant(root, fp):
        c = FeatureCache(str(root)).make()
        part = os.path.join(c.images_dir, "sun.partial")
        os.makedirs(part)
        np.savez(os.path.join(part, "shard_00000.npz"), feats=sentinel, ok=np.ones(20, bool))
        json.dump({"fp": fp}, open(os.path.join(part, "fingerprint.json"), "w"))
        return c

    c1 = plant(tmp_path / "same", shard_fingerprint(m, 20, "ViT-B/16"))
    with pytest.raises(RuntimeError, match="unit-norm"):       # reused -> the planted shard is detected
        encode_manifest(tiny_clip, m, c1, batch_size=16, num_workers=0, shard_size=20, log_every=10**6)
    c2 = plant(tmp_path / "other", "not-this-list")
    fs = encode_manifest(tiny_clip, m, c2, batch_size=16, num_workers=0, shard_size=20, log_every=10**6)
    assert len(fs) == len(m)                                   # discarded and re-encoded


def test_live_vs_cache_fp16_model(built, tiny_clip):
    """With an fp16 CLIP (as on GPU) the live official run and the cache replay agree exactly."""
    import copy

    from oodlab.clipwrap import import_clip

    convert_weights = import_clip().model.convert_weights

    from oodlab.cache import encode_manifest as enc
    from oodlab.clipwrap import model_dtype
    from oodlab.data.manifest import Manifest
    from oodlab.official.live import live_vs_cache
    from oodlab.textbank import build_textbank

    m16 = copy.deepcopy(tiny_clip)
    convert_weights(m16)
    assert model_dtype(m16) == torch.float16 and next(m16.parameters()).dtype == torch.float32  # the trap
    c = FeatureCache(os.path.join(os.path.dirname(built.root), "cache_fp16")).make()
    b = build_textbank(m16, "ViT-B/16", oodlab.imagenet_classnames(), oodlab.CORPUS_DIR, total_neg=300, corpus_limit=800)
    idm, nm = Manifest.load(built.manifests_dir, "imagenet_val_all"), Manifest.load(built.manifests_dir, "ninco")
    ids, ood = enc(m16, idm, c, batch_size=32, num_workers=0, log_every=10**6), enc(m16, nm, c, batch_size=32, num_workers=0, log_every=10**6)
    rep = live_vs_cache(m16, b, idm, ids, nm, ood, n_id=120, device="cpu", batch_size=32, num_workers=0,
                        cfg=TANLConfig.paper().with_(num_neg=60))
    assert rep["max_abs_score_diff"] == 0.0 and rep["pass"], rep


def test_streams(built):
    a, o = built.load_set("imagenet_val_all"), built.load_set("inaturalist")
    s0, s1 = pair_stream(a, o, seed=0, batch_size=32), pair_stream(a, o, seed=1, batch_size=32)
    assert sorted(s0.rows.tolist()) == list(range(len(a) + len(o)))
    assert not np.array_equal(s0.rows, s1.rows)
    assert (s0.labels == -1).sum() == len(o)
    idf = pair_stream(a, o, seed=0, order="id_first")
    assert (idf.labels[: len(a)] >= 0).all() and (idf.labels[len(a):] == -1).all()
    t = temporal_stream(a, [built.load_set(n) for n in ["sun", "places"]], seed=0, batch_size=32)
    assert [s.reset for s in t.segments] == [True, False]
    assert len(t) == 2 * len(a) + len(built.load_set("sun")) + len(built.load_set("places"))
    ts = temporal_stream(a, [built.load_set(n) for n in ["sun", "places"]], seed=0, id_mode="split")
    assert len(ts) == len(a) + len(built.load_set("sun")) + len(built.load_set("places"))
    batches = list(t.batches())
    assert all(e - s <= 32 for _, s, e, _ in batches)
    assert sum(r for *_, r in batches) == 1


def test_runner_protocols(built):
    b = built.load_textbank()
    r = Runner(built, batch_size=32, window=64)
    tanl = TANL(b.id_text, b.corpus_text, b.noise_feats, TANLConfig.paper().with_(num_neg=100, logit_scale=b.logit_scale))
    dfs = [
        r.evaluate(tanl, "four_ood", seeds=[0, 1], config="paper"),
        r.evaluate(tanl, "openood_v15", seeds=[0], config="paper"),
        r.evaluate(AdaNeg(b.id_text, b.neg_text, AdaNegConfig(emulate_fp16=True)), "near_50k", seeds=[0], config="fp16"),
        r.evaluate(NegLabel(b.id_text, b.neg_text, NegLabelConfig()), "four_ood", seeds=[0, 1, 2], config="g100"),
        r.evaluate(MCM(b.id_text_simple, MCMConfig.openood_vlm()), "openood_val", config="vlm"),
    ]
    import pandas as pd

    df = pd.concat(dfs, ignore_index=True)
    assert np.isfinite(df["fpr95"]).all() and np.isfinite(df["auroc"]).all()
    assert (df[df.method == "neglabel"]["seed"] == 0).all()   # stateless -> one seed
    s = summarize(df)
    assert {"mean:four", "mean:near", "mean:far"} <= set(s["key"])
    cmp = compare_with_targets(s, "tanl", "four_ood")
    assert len(cmp) == 5
    acc = acceptance_table(s)
    assert set(acc["status"]) <= {"PASS", "CHECK"}
    t = r.evaluate_temporal(tanl, orders={"S-P": ["sun", "places"]}, seeds=[0], config="paper")
    assert list(t["position"]) == [0, 1] and "post_shift_fpr95" in t.columns


def test_churn_tracker_is_passive(built):
    from oodlab.diagnostics import ChurnTracker, assert_passive, oracle_selection
    from oodlab.runner import run_stream

    b = built.load_textbank()
    cfg = TANLConfig.paper().with_(num_neg=100)
    s = pair_stream(built.load_set("imagenet_val_all"), built.load_set("ssb_hard"), seed=0, batch_size=32)
    rep = assert_passive(TANL(b.id_text, b.corpus_text, b.noise_feats, cfg),
                         TANL(b.id_text, b.corpus_text, b.noise_feats, cfg.with_(record=True)), s)
    assert rep["identical"]
    m = TANL(b.id_text, b.corpus_text, b.noise_feats, cfg.with_(record=True))
    tr = ChurnTracker({0: oracle_selection(m, s, 0)})
    run_stream(m, s, hooks=[tr])
    summ = tr.summary()
    for k in ["final_surv_first", "final_surv_init", "churn_first10", "unique_labels", "dwell_median", "decision"]:
        assert k in summ, k
    df = tr.frame()
    assert len(df) == s.n_batches()
    assert ((df["churn"] >= 0) & (df["churn"] <= 1)).all()
    assert df["surv_first"].iloc[0] == 1.0


def test_live_vs_cache_runs(built, tiny_clip):
    from oodlab.data.manifest import Manifest
    from oodlab.official.live import live_vs_cache

    b = built.load_textbank()
    rep = live_vs_cache(tiny_clip, b, Manifest.load(built.manifests_dir, "imagenet_val_all"),
                        built.load_set("imagenet_val_all"), Manifest.load(built.manifests_dir, "ninco"),
                        built.load_set("ninco"), n_id=100, device="cpu", batch_size=32, num_workers=0,
                        cfg=TANLConfig.paper().with_(num_neg=100))
    assert rep["pred_agreement"] == 1.0
    assert abs(rep["live_fpr95"] - rep["cache_fpr95"]) < 5   # CPU/fp32 model: cache holds fp16 copies


def test_fi_imagenet_derived(built, fake_kaggle):
    from oodlab.data.fi_imagenet import fi_imagenet_sets, find_lists, read_val_ids

    ood_p, amb_p = find_lists([fake_kaggle["inputs"]])
    full = built.load_set("imagenet_val_all")
    id_clean, ood = fi_imagenet_sets(full, read_val_ids(ood_p), read_val_ids(amb_p))
    assert len(ood) == 20 and (ood.labels == -1).all()
    assert len(id_clean) == len(full) - 35
    r = Runner(built, batch_size=32)
    r.extra_sets.update({id_clean.name: id_clean, ood.name: ood})
    b = built.load_textbank()
    df = r.evaluate(NegLabel(b.id_text, b.neg_text), "fi_imagenet_1k")
    assert len(df) == 1 and df["n_id"].iloc[0] == len(id_clean)
