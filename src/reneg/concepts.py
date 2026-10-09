"""Class names and image centroids of OOD sets, used only to *evaluate* the transport (G1).

oodlab stores, for every cached image, a key: the relative path from the OpenOOD image list
(for example ``ssb_hard/n04542943/n04542943_1234.JPEG`` or ``texture/banded/banded_0002.jpg``).
The folder that contains the file names the class. For SSB-hard the folder is a WordNet id,
which WordNet turns into a name; for NINCO and Textures the folder name is the class name.
Sets whose images sit in one flat folder (iNaturalist, SUN, Places) have no class names.

These labels never reach the method or its tuning; G1 uses them only to check whether the
transport puts a concept's name next to that concept's real images.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .transport import l2n
from .wordnet import is_wnid, wnid_name

GENERIC_FOLDERS = {"", ".", "images", "image", "img", "imgs", "val", "test", "train", "data", "ood"}


def folder_of(key: str) -> str:
    parts = str(key).replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else ""


def readable(folder: str, wn=None) -> Optional[str]:
    if is_wnid(folder):
        return wnid_name(wn, folder) if wn is not None else None
    name = folder.replace("_", " ").replace("-", " ").strip()
    return " ".join(name.split()) or None


@dataclass
class ConceptSet:
    dataset: str
    folders: List[str]          # one per concept
    names: List[str]            # readable names used in the prompt
    counts: np.ndarray          # images per concept
    centroids: torch.Tensor     # (K, D) unit vectors
    image_concept: np.ndarray   # concept index per image, -1 if dropped
    note: str = ""

    def __len__(self) -> int:
        return len(self.folders)

    def preview(self, k: int = 8) -> str:
        pairs = [f"{n} ({c})" for n, c in zip(self.names[:k], self.counts[:k].tolist())]
        return f"{self.dataset}: {len(self)} concepts, e.g. " + ", ".join(pairs)


def build_concepts(dataset: str, feats: torch.Tensor, keys: Sequence[str], wn=None, min_images: int = 5) -> ConceptSet:
    """Group a set's images by class folder and compute each class's centroid direction."""
    folders = [folder_of(k) for k in keys]
    count = Counter(folders)
    usable = [f for f, n in count.items() if n >= min_images and f.lower() not in GENERIC_FOLDERS]
    if len(usable) < 2:
        return ConceptSet(dataset, [], [], np.zeros(0, dtype=np.int64), torch.zeros(0, feats.shape[1]),
                          np.full(len(keys), -1), note=f"no class folders found (e.g. key {keys[0] if len(keys) else '-'})")
    names, kept = [], []
    dropped_names = 0
    for f in sorted(usable):
        n = readable(f, wn)
        if n is None:
            dropped_names += 1
            continue
        kept.append(f)
        names.append(n)
    index = {f: i for i, f in enumerate(kept)}
    image_concept = np.array([index.get(f, -1) for f in folders], dtype=np.int64)
    X = l2n(torch.as_tensor(feats).float())
    K = len(kept)
    sums = torch.zeros(K, X.shape[1])
    m = torch.as_tensor(image_concept >= 0)
    sums.index_add_(0, torch.as_tensor(image_concept[image_concept >= 0]), X[m])
    counts = np.bincount(image_concept[image_concept >= 0], minlength=K)
    note = f"{dropped_names} concepts without a name were dropped" if dropped_names else ""
    return ConceptSet(dataset, kept, names, counts, l2n(sums), image_concept, note=note)
