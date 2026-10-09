"""FAKE CLIP: text 'encoding' is a lookup in resources/fake_text.npz (names -> vectors)."""
import os

import numpy as np
import torch

from . import RESOURCES


def backbone_tag(name):
    return name.lower().replace("/", "-").replace("@", "-")


class _FakeModel:
    def __init__(self):
        z = np.load(os.path.join(RESOURCES, "fake_text.npz"), allow_pickle=False)
        self.table = {n: v for n, v in zip(z["names"].tolist(), z["vecs"])}
        self.dtype = torch.float32


def load_clip(name="ViT-B/16", device="cpu", input_roots=None, download_root=None, random_init=False):
    if download_root:
        os.makedirs(download_root, exist_ok=True)
    return _FakeModel()


def label_text_features(model, labels, templates, batch_size=1000, logger=None, progress=None):
    assert len(templates) == 1 and templates[0] == "The nice {}."
    dim = len(next(iter(model.table.values())))

    def vec(l):                      # names outside the fake world get a fixed pseudo-random vector
        if l in model.table:
            return model.table[l]
        rng = np.random.default_rng(abs(hash(l)) % (2 ** 32))
        return rng.normal(size=dim)

    v = torch.tensor(np.stack([vec(l) for l in labels]), dtype=torch.float32)
    return v / v.norm(dim=1, keepdim=True)
