import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .clipwrap import backbone_tag


@dataclass
class FeatureSet:
    name: str
    feats: torch.Tensor
    labels: np.ndarray
    keys: list
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.feats.shape[0])

    @property
    def dim(self):
        return int(self.feats.shape[1])

    def subset(self, idx, name=None):
        idx = np.asarray(idx, dtype=np.int64)
        return FeatureSet(name or self.name, self.feats[torch.as_tensor(idx)], self.labels[idx],
                          [self.keys[i] for i in idx.tolist()], dict(self.meta))


class FeatureCache:
    def __init__(self, root, backbone="ViT-B/16"):
        self.root = root
        self.backbone = backbone
        self.dir = os.path.join(root, backbone_tag(backbone))
        self.images_dir = os.path.join(self.dir, "images")
        self.manifests_dir = os.path.join(self.dir, "manifests")
        self.textbank_path = os.path.join(self.dir, "textbank.pt")

    def set_path(self, name):
        return os.path.join(self.images_dir, f"{name}.npz")

    def has(self, name):
        return os.path.isfile(self.set_path(name))

    def has_textbank(self):
        return os.path.isfile(self.textbank_path)

    def available(self):
        if not os.path.isdir(self.images_dir):
            return []
        return sorted(f[:-4] for f in os.listdir(self.images_dir) if f.endswith(".npz"))

    def load_set(self, name):
        with np.load(self.set_path(name), allow_pickle=False) as z:
            return FeatureSet(name, torch.from_numpy(z["feats"].copy()), z["labels"].astype(np.int64), z["keys"].tolist())

    def load_textbank(self):
        from .textbank import TextBank
        return TextBank.load(self.textbank_path)
