"""Compact exports: int8 round trip, file splitting, the candidate pool."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from synthetic_world import FakeCache, fake_bank, make_world  # noqa: E402

from reneg.packs import dq8, export_pool, export_streams, load_kg_pool, q8  # noqa: E402
from reneg.transport import l2n  # noqa: E402


def test_int8_round_trip_is_accurate():
    X = l2n(torch.randn(500, 512))
    P = l2n(torch.randn(50, 512))
    q, s = q8(X)
    Y = dq8(q, s)
    assert q.dtype == np.int8 and s.dtype == np.float16
    assert float((Y @ P.T - X @ P.T).abs().max()) < 3e-3


def test_export_streams_splits_large_sets(tmp_path):
    w = make_world(C=40, D=64, seed=5, imgs_per_class=30)
    files = export_streams(FakeCache(w), str(tmp_path), ["imagenet_test", "ssb_hard", "ninco", "missing_set"],
                           max_mb=0.05, log=lambda *_: None)
    keys = set()
    for f in files:
        keys |= set(np.load(f).files)
    parts = sorted(k for k in keys if k.startswith("imagenet_test__part") and k.endswith("__q"))
    assert len(parts) >= 2                                        # 1,200 x 64 bytes does not fit in 50 kB
    n = sum(np.load(f)[k].shape[0] for f in files for k in np.load(f).files if k.startswith("imagenet_test") and k.endswith("__q"))
    assert n == len(w.sets["imagenet_test"]["labels"])
    assert any(k.startswith("ssb_hard") and k.endswith("__folder_names") for k in keys)


def test_pool_file_and_export(tmp_path):
    pool = load_kg_pool()
    assert len(pool["pool"]) > 10000
    r = pool["pool"][0]
    assert {"name", "wnid", "wup", "nearest_id", "in_ssb_hard", "in_ninco"} <= set(r)
    small = {"pool": pool["pool"][:50]}
    w = make_world(C=10, D=32, seed=1)
    bank = fake_bank(w)
    enc = lambda names: l2n(torch.randn(len(names), 32))
    path = export_pool(bank, enc, str(tmp_path), pool=small, log=lambda *_: None)
    z = np.load(path)
    assert z["pool_text"].shape == (50, 32) and len(z["pool_names"]) == 50


def test_export_textbank_round_trip(tmp_path):
    from types import SimpleNamespace
    from reneg.packs import export_textbank, load_textbank_export
    N, D = 3000, 64
    bank = SimpleNamespace(corpus_text=l2n(torch.randn(N, D)).half(), words=[f"w{i}" for i in range(N)],
                           is_adj=torch.zeros(N, dtype=torch.bool), n_selected=1000, noise_feats=torch.randn(15, D),
                           id_text=l2n(torch.randn(10, D)), id_names=[f"c{i}" for i in range(10)], logit_scale=100.0)
    files = export_textbank(bank, str(tmp_path), max_mb=0.05, log=lambda *_: None)
    assert len(files) >= 3                                        # 3,000 x 66 bytes does not fit in 50 kB
    d = load_textbank_export(files[::-1])                         # order of the paths does not matter
    assert d["corpus_text"].shape == (N, D)
    assert float((d["corpus_text"] - l2n(bank.corpus_text.float())).abs().max()) < 1e-2
    assert list(d["words"][:3]) == ["w0", "w1", "w2"] and int(d["n_selected"]) == 1000
    try:
        load_textbank_export(files[:-1])
        assert False, "a missing part must be reported"
    except ValueError:
        pass
