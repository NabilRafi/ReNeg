"""WordNet helpers (NLTK): names for wnids and subtree groups of ImageNet classes.

On Colab ``get_wordnet()`` downloads the WordNet corpus once (about 10 MB). If that is not
possible, callers fall back to text-embedding clusters (see ``text_cluster_groups``).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

WNID_RE = re.compile(r"^n\d{8}$")


def get_wordnet(download: bool = True):
    """The NLTK WordNet reader, or None if NLTK or its data is unavailable."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None
    try:
        wn.ensure_loaded()
        return wn
    except LookupError:
        if not download:
            return None
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as wn2
            wn2.ensure_loaded()
            return wn2
        except Exception:
            return None


def is_wnid(s: str) -> bool:
    return bool(WNID_RE.match(s or ""))


def synset_of(wn, wnid: str):
    return wn.synset_from_pos_and_offset("n", int(wnid[1:]))


def wnid_name(wn, wnid: str) -> Optional[str]:
    try:
        return synset_of(wn, wnid).lemma_names()[0].replace("_", " ")
    except Exception:
        return None


def subtree_groups(wnids: Sequence[str], wn, max_size: Optional[int] = None, start_depth: int = 3,
                   min_groups: int = 10) -> Tuple[np.ndarray, List[str]]:
    """Group classes by a common WordNet ancestor.

    Each class starts at its ancestor ``start_depth`` steps below the root (on its longest
    hypernym path). Any group larger than ``max_size`` is split one level deeper, and while
    there are fewer than ``min_groups`` groups the largest splittable group is split too.
    Returns (group index per class, group names).
    """
    n = len(wnids)
    if max_size is None:
        max_size = min(120, max(8, n // 8))
    paths = []
    for w in wnids:
        try:
            p = synset_of(wn, w).hypernym_paths()
            paths.append(max(p, key=len))
        except Exception:
            paths.append([])
    depth = [min(start_depth, max(len(p) - 1, 0)) for p in paths]

    def key(i):
        return paths[i][depth[i]].name() if paths[i] else f"unknown:{wnids[i]}"

    def grouped():
        groups = defaultdict(list)
        for i in range(n):
            groups[key(i)].append(i)
        return groups

    def split(members) -> bool:
        moved = False
        for i in members:
            if paths[i] and depth[i] < len(paths[i]) - 1:
                depth[i] += 1
                moved = True
        return moved

    for _ in range(64):                                   # bounded: WordNet paths are < 20 deep
        groups = grouped()
        big = [m for m in groups.values() if len(m) > max_size]
        if big:
            if not any(split(m) for m in big):
                break
            continue
        if len(groups) < min_groups:
            order = sorted(groups.values(), key=len, reverse=True)
            if not any(split(m) for m in order if len(m) > 1):
                break
            continue
        break
    names = sorted({key(i) for i in range(n)})
    index = {nm: j for j, nm in enumerate(names)}
    return np.array([index[key(i)] for i in range(n)], dtype=np.int64), names


@torch.no_grad()
def text_cluster_groups(id_text: torch.Tensor, n_groups: int = 25, iters: int = 50, seed: int = 0) -> Tuple[np.ndarray, List[str]]:
    """Fallback when WordNet is unavailable: spherical k-means on the class-name embeddings."""
    X = torch.nn.functional.normalize(torch.as_tensor(id_text).float(), dim=1)
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:n_groups]].clone()
    for _ in range(iters):
        a = (X @ C.t()).argmax(1)
        for k in range(n_groups):
            m = a == k
            if m.any():
                C[k] = torch.nn.functional.normalize(X[m].mean(0), dim=0)
    a = (X @ C.t()).argmax(1).numpy().astype(np.int64)
    uniq = np.unique(a)
    remap = {u: j for j, u in enumerate(uniq)}
    return np.array([remap[x] for x in a], dtype=np.int64), [f"text-cluster-{j}" for j in range(len(uniq))]


def describe_groups(groups: np.ndarray, names: List[str], top: int = 12) -> str:
    c = Counter(groups.tolist())
    parts = [f"{names[g]} ({n})" for g, n in c.most_common(top)]
    return f"{len(c)} groups; largest: " + ", ".join(parts)
