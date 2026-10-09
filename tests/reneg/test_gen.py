"""Generation probe plumbing: batching, saving, resuming (fake generator and encoder)."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from synthetic_world import FakeCache, fake_bank, make_world  # noqa: E402

from reneg.gen import fake_generator, probe_names, run_probe  # noqa: E402
from reneg.transport import l2n  # noqa: E402


def _enc(images):
    x = torch.tensor(np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images]))
    return l2n(torch.cat([x, torch.ones(len(images), 5)], 1))


def test_probe_saves_and_resumes(tmp_path):
    names, tags, folders = [f"thing {i}" for i in range(5)], ["ninco"] * 5, [f"f{i}" for i in range(5)]
    out = str(tmp_path / "probe.npz")
    calls = {"n": 0}
    base = fake_generator(16)

    def flaky(prompts, seed, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("session died")
        return base(prompts, seed)

    try:
        run_probe(flaky, _enc, names, tags, folders, out, n_per=2, batch=3, save_every=1, log=lambda *_: None)
    except RuntimeError:
        pass
    z = np.load(out)
    assert 0 < len(z["name_idx"]) < 10
    run_probe(base, _enc, names, tags, folders, out, n_per=2, batch=3, save_every=1, log=lambda *_: None)
    z = np.load(out)
    pairs = set(zip(z["name_idx"].tolist(), z["rep"].tolist()))
    assert len(pairs) == 10 and z["emb"].shape == (10, 8) and z["emb"].dtype == np.float16


def test_probe_names_from_cache():
    w = make_world(C=30, D=16, seed=2)
    names, tags, folders = probe_names(FakeCache(w), fake_bank(w), wn=None, ssb_max=5, n_imagenet=4)
    assert tags.count("imagenet") == 4 and tags.count("ninco") >= 2
    assert len(names) == len(tags) == len(folders)


def test_generate_safely_halves_on_oom():
    from reneg.gen import generate_safely

    base = fake_generator(8)
    calls = []

    def gen(prompts, seed, **kw):
        calls.append(len(prompts))
        if len(prompts) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return base(prompts, seed)

    out = generate_safely(gen, [f"p{i}" for i in range(8)], seed=1, log=lambda *_: None)
    assert len(out) == 8 and max(c for c in calls if c <= 2) == 2
