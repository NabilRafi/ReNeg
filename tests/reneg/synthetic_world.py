"""A small synthetic CLIP-like world with a known text -> image map, for tests and smoke runs.

Every concept has a text vector t. Its images sit around u = n(A t + gap), where A is the
identity plus a small class-dependent distortion and ``gap`` is a shared modality-gap vector.
So a good transport should beat raw text, and R2 / B-maps should beat the pure gap shift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


def _n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@dataclass
class World:
    dim: int
    id_names: List[str]
    id_text: np.ndarray                      # (C, D)
    id_wnids: List[str]
    neg_names: List[str]
    neg_text: np.ndarray                     # (M, D)
    name_to_text: Dict[str, np.ndarray]
    sets: Dict[str, dict] = field(default_factory=dict)   # name -> {feats, labels, keys}
    A: Optional[np.ndarray] = None
    gap: Optional[np.ndarray] = None

    def encode(self, names):
        return torch.tensor(np.stack([self.name_to_text[n] for n in names]), dtype=torch.float32)


def _wordnet_leaves(n: int, seed: int = 0):
    """Real noun synsets (wnid, name, hypernym path) so WordNet grouping can be exercised."""
    try:
        from nltk.corpus import wordnet as wn

        wn.ensure_loaded()
    except Exception:
        return None
    roots = ["dog.n.01", "bird.n.01", "fish.n.01", "vehicle.n.01", "fruit.n.01", "tool.n.01", "musical_instrument.n.01",
             "furniture.n.01", "insect.n.01", "building.n.01"]
    rng = np.random.default_rng(seed)
    per = int(np.ceil(n / len(roots)))
    out = []
    for r in roots:
        leaves = [s for s in wn.synset(r).closure(lambda s: s.hyponyms()) if not s.hyponyms()]
        leaves = sorted(leaves, key=lambda s: s.name())
        pick = rng.choice(len(leaves), size=min(per * 3, len(leaves)), replace=False)
        for i in pick:
            s = leaves[i]
            out.append((f"n{s.offset():08d}", s.lemma_names()[0].replace("_", " "), r))
    return out


def make_world(C: int = 60, D: int = 64, n_neg: int = 400, imgs_per_class: int = 20, distortion: float = 0.35,
               gap_scale: float = 0.8, noise: float = 0.8, seed: int = 0, n_ninco: int = 20, n_tex: int = 12,
               n_ssb: int = 30, spread: float = 0.7) -> World:
    rng = np.random.default_rng(seed)
    leaves = _wordnet_leaves(C + n_neg + n_ssb + 200, seed)
    groups = 10
    gdirs = _n(rng.normal(size=(groups, D)))

    def text_for(group):
        return _n(gdirs[group] + spread * _n(rng.normal(size=D)))

    names, wnids, texts, gidx = [], [], [], []
    if leaves:
        rng.shuffle(leaves)
        by_root = {}
        for w, nm, r in leaves:
            by_root.setdefault(r, []).append((w, nm))
        roots = sorted(by_root)
        k = 0
        while len(names) < C:
            r = roots[k % len(roots)]
            if by_root[r]:
                w, nm = by_root[r].pop()
                if nm not in names:
                    names.append(nm)
                    wnids.append(w)
                    gidx.append(roots.index(r) % groups)
            k += 1
        spare = [(w, nm, roots.index(r) % groups) for r in roots for (w, nm) in by_root[r]]
    else:
        names = [f"id class {i}" for i in range(C)]
        wnids = [f"n{10000000 + i:08d}" for i in range(C)]
        gidx = [i % groups for i in range(C)]
        spare = [(f"n{20000000 + i:08d}", f"spare concept {i}", i % groups) for i in range(n_neg + n_ssb + 50)]
    id_text = np.stack([text_for(g) for g in gidx])
    A = np.eye(D) + distortion * rng.normal(size=(D, D)) / np.sqrt(D)
    gap = gap_scale * _n(rng.normal(size=D))

    def image_dir(t):
        return _n(t @ A.T + gap)

    def images(t, n):
        u = image_dir(t)
        return _n(u[None, :] + noise * rng.normal(size=(n, D)) / np.sqrt(D))

    name_to_text = {n: t for n, t in zip(names, id_text)}
    # OOD concepts: SSB-hard-like (wnid folders), NINCO-like and Textures-like (name folders)
    rng.shuffle(spare)
    ssb = spare[:n_ssb]
    rest = spare[n_ssb:]
    ninco_names = [f"ninco thing {i}" for i in range(n_ninco)]
    tex_names = [f"texture {c}" for c in "abcdefghijklmnopqrstuvwxyz"[:n_tex]]
    for nm in ninco_names + tex_names:
        name_to_text[nm] = text_for(int(rng.integers(groups)))
    for w, nm, g in ssb:
        name_to_text[nm] = text_for(g)
    neg_names = [nm for (_, nm, _) in rest[:n_neg]]
    for nm in neg_names:
        name_to_text.setdefault(nm, text_for(int(rng.integers(groups))))
    # the OOD concepts' names are also in the negative corpus, as in NegLabel's WordNet corpus
    neg_names = list(dict.fromkeys(neg_names + ninco_names[: n_ninco // 2] + [nm for (_, nm, _) in ssb[: n_ssb // 2]]))
    neg_text = np.stack([name_to_text[n] for n in neg_names])
    world = World(D, names, id_text, wnids, neg_names, neg_text, name_to_text, A=A, gap=gap)

    def id_set(n_per, key_offset):
        feats, labels, keys = [], [], []
        for c in range(C):
            f = images(id_text[c], n_per)
            feats.append(f)
            labels += [c] * n_per
            keys += [f"ILSVRC2012_val_{key_offset + c * n_per + j:08d}" for j in range(n_per)]
        return {"feats": np.concatenate(feats), "labels": np.array(labels), "keys": keys}

    def ood_set(prefix, concepts, per):
        feats, keys = [], []
        for folder, nm in concepts:
            f = images(name_to_text[nm], per)
            feats.append(f)
            keys += [f"{prefix}/{folder}/{folder}_{j:04d}.jpg" for j in range(per)]
        feats = np.concatenate(feats)
        return {"feats": feats, "labels": -np.ones(len(feats), dtype=np.int64), "keys": keys}

    def flat_set(prefix, n):
        pool = [name_to_text[nm] for nm in neg_names[: max(10, len(neg_names) // 4)]]
        feats = np.stack([images(pool[i % len(pool)], 1)[0] for i in range(n)])
        return {"feats": feats, "labels": -np.ones(n, dtype=np.int64), "keys": [f"{prefix}/images/{i:05d}.jpg" for i in range(n)]}

    world.sets["imagenet_test"] = id_set(imgs_per_class, 0)
    world.sets["imagenet_val"] = id_set(max(3, imgs_per_class // 4), 900000)
    world.sets["imagenet_val_all"] = world.sets["imagenet_test"]
    ninco_c = [(nm.replace(" ", "_"), nm) for nm in ninco_names]
    tex_c = [(nm.split(" ")[1], nm) for nm in tex_names]
    for nm in tex_names:                         # textures are named by their folder ("a", "b", ...)
        name_to_text[nm.split(" ")[1]] = name_to_text[nm]
    tex_c = [(f, f) for f, _ in tex_c]
    world.sets["ninco"] = ood_set("ninco", ninco_c, 12)
    world.sets["textures_all"] = ood_set("dtd/images", tex_c, 10)
    world.sets["textures"] = ood_set("texture", tex_c, 8)
    world.sets["ssb_hard"] = ood_set("ssb_hard", [(w, nm) for (w, nm, _) in ssb], 8)
    world.sets["inaturalist"] = flat_set("inaturalist", 300)
    world.sets["sun"] = flat_set("sun", 300)
    world.sets["places"] = flat_set("places", 300)
    world.sets["openimage_o"] = flat_set("openimage_o", 300)
    world.sets["openimage_o_val"] = flat_set("openimage_o", 120)
    return world


# ---------------------------------------------------------------------------
# minimal stand-ins for oodlab's cache objects (same attribute names)
# ---------------------------------------------------------------------------


@dataclass
class FakeFS:
    name: str
    feats: torch.Tensor
    labels: np.ndarray
    keys: List[str]
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.feats.shape[0])

    def subset(self, idx, name=None):
        idx = np.asarray(idx, dtype=np.int64)
        return FakeFS(name or self.name, self.feats[torch.as_tensor(idx)], self.labels[idx], [self.keys[i] for i in idx])


class FakeCache:
    def __init__(self, world: World):
        self.world = world

    def has(self, name):
        return name in self.world.sets

    def available(self):
        return sorted(self.world.sets)

    def load_set(self, name):
        s = self.world.sets[name]
        return FakeFS(name, torch.tensor(s["feats"], dtype=torch.float16), np.asarray(s["labels"]), list(s["keys"]))


@dataclass
class FakeBank:
    id_names: List[str]
    id_text: torch.Tensor
    corpus_text: torch.Tensor
    n_selected: int
    id_wnids: List[str]
    logit_scale: float = 100.0

    @property
    def neg_text(self):
        return self.corpus_text[: self.n_selected]


def fake_bank(world: World) -> FakeBank:
    return FakeBank(world.id_names, torch.tensor(world.id_text, dtype=torch.float32),
                    torch.tensor(world.neg_text, dtype=torch.float32), len(world.neg_names), world.id_wnids)
