"""TINS re-implementation: runs end to end with a tiny random CLIP (real tokenizer and text encoder architecture).
Needs the real oodlab (vendored CLIP); skipped with the test stub."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:                                                   # the vendored CLIP imports torchvision.transforms at import time;
    import torchvision  # noqa: F401                   # outside Colab it may be missing: a stub is enough (no images here)
except ImportError:
    import types
    tv = types.ModuleType("torchvision")
    tr = types.ModuleType("torchvision.transforms")
    for _n in ("Compose", "Resize", "CenterCrop", "ToTensor", "Normalize"):
        setattr(tr, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))
    tr.InterpolationMode = type("InterpolationMode", (), {"BICUBIC": 3})
    tv.transforms = tr
    sys.modules["torchvision"], sys.modules["torchvision.transforms"] = tv, tr

clipwrap = pytest.importorskip("oodlab.clipwrap")
if not hasattr(clipwrap, "tiny_random_clip"):
    pytest.skip("oodlab is the test stub, not the real package", allow_module_level=True)

from reneg.tins import (TINSConfig, TINSFeat, build_init_candidates, build_static_negatives,  # noqa: E402
                        collect_negative_words, encode_texts, encode_with_pseudo_tokens, grouped_score, tokenize)
from reneg.transport import l2n  # noqa: E402


def _setup(C=8, D=64):
    model = clipwrap.tiny_random_clip(embed_dim=D).float()
    for p in model.parameters():
        p.requires_grad_(False)
    names = [f"class{i}" for i in range(C)]
    pos = encode_texts(model, [f"The nice {n}." for n in names])
    g = torch.Generator().manual_seed(0)
    proto = l2n(pos + 0.5 * torch.randn(C, D, generator=g))
    return model, names, pos, proto


def test_pseudo_token_encoder_matches_encode_text_when_token_unchanged():
    model, _, _, _ = _setup()
    tok = tokenize(["a photo of a dog."])
    pos = torch.tensor([(tokenize(["a photo of a dog"]) != 0).sum() - 2])
    z = model.token_embedding(tok)[0, pos[0]][None]
    a = encode_with_pseudo_tokens(model, tok, z, pos)
    b = model.encode_text(tok)
    assert torch.allclose(a, b, atol=1e-5)


def test_trimmed_encoder_equals_the_full_length_one_with_gradients():
    model, _, _, _ = _setup()
    words = ["dog", "abdominal aorta", "x", "light brown"]
    tok = tokenize([f"a photo of a {w}." for w in words])
    pos = (tokenize([f"a photo of a {w}" for w in words]) != 0).sum(1) - 2
    g = torch.Generator().manual_seed(3)
    z0 = model.token_embedding(tok)[torch.arange(len(words)), pos] + 0.1 * torch.randn(len(words), 64, generator=g)
    za, zb = z0.clone().requires_grad_(True), z0.clone().requires_grad_(True)
    a = encode_with_pseudo_tokens(model, tok, za, pos, trim=True)
    b = encode_with_pseudo_tokens(model, tok, zb, pos, trim=False)
    assert torch.allclose(a, b, atol=1e-5)
    w = torch.randn(a.shape, generator=g)
    (a * w).sum().backward()
    (b * w).sum().backward()
    assert torch.allclose(za.grad, zb.grad, atol=1e-5)


def test_grouped_score_with_one_group_equals_the_formula():
    g = torch.Generator().manual_seed(1)
    v, pos, neg = [l2n(torch.randn(n, 16, generator=g)) for n in (5, 4, 12)]
    s = grouped_score(v, pos, neg, 100.0, 1, False)
    ep = torch.exp(100 * v @ pos.T).sum(1)
    en = 4 * torch.exp(100 * v @ neg.T).mean(1)
    assert torch.allclose(s, ep / (ep + en), atol=1e-5)


def test_tins_runs_inverts_candidates_and_fills_the_bank():
    model, names, pos, proto = _setup()
    nouns, adjs = collect_negative_words(positive_labels=names)
    assert len(nouns) > 1000 and len(adjs) > 100
    neg, words = build_static_negatives(model, pos, proto, n=200, positive_labels=names, log=lambda *a: None)
    assert neg.shape[0] == len(words) == 200
    cands = build_init_candidates(model, words, proto)
    m = TINSFeat(model, pos, proto, neg, cands, TINSConfig(inversion_steps=3, ood_threshold=1.01, extra_text_length=50))
    g = torch.Generator().manual_seed(2)
    X = l2n(torch.randn(70, 64, generator=g))
    out = [m.step(X[s:s + 32]) for s in range(0, 70, 32)]
    sc = torch.cat([o.score for o in out])
    assert sc.shape == (70,) and torch.isfinite(sc).all() and ((sc >= 0) & (sc <= 1)).all()
    assert all(o.info["cand"].all() for o in out)            # threshold above 1: every image is a candidate
    assert m.bank.shape[0] <= 50
    m.reset()
    assert m.bank.shape[0] == 0
