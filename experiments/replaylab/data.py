"""The exported features (scripts/export_features.py): test streams, text, the KG pool, metrics and stream order.

Everything loads lazily on first use, so ``from replaylab.data import id_text, pool_text`` reads only the files
those names need. Image features are 8-bit codes, restored to unit float32 rows (``dq8``).

  id_text, neg_text, id_names     ImageNet class text (1,000) and NegLabel's 10,000 negatives (TANL's corpus head)
  pool_text, pool_cluster, ...    the 14,526-name WordNet pool; BLIND masks out the names that are SSB-hard or NINCO
                                  classes (the "blind" pool used for every headline result)
  load_set(name)                  {"X", "labels", "folder", "folder_names"} of one exported set
  pair_stream(X_id, X_ood, seed)  the official stream order: ConcatDataset([OOD, ID]) shuffled by torch.randperm
  metrics(id_scores, ood_scores)  FPR95 in both conventions and AUROC (percent)
  shots()                         the 5k labelled ID images (OpenOOD's ImageNet ID-val split) and their mask in
                                  Four-OOD's 50k ID set
  stats, m_img, val_X, val_y, val_ood, pack1()   G1b's analysis pack (only the Seam 1 analyses need it)
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from reneg.packs import RESOURCES, dq8, load_textbank_export
from reneg.scores import _metrics_fallback
from reneg.transport import ClassStats, l2n

from .paths import EXPORTS, out

NEAR, FAR = ("ssb_hard", "ninco"), ("inaturalist", "textures", "openimage_o")
OOD_SETS = NEAR + FAR
FOUR = ("inaturalist", "sun", "places", "textures_all")
DT = torch.float32
_CACHE: Dict = {}


def T(a) -> torch.Tensor:
    return l2n(torch.as_tensor(np.asarray(a, dtype=np.float32)))


def _need(pattern: str) -> list:
    paths = sorted(glob.glob(os.path.join(EXPORTS, pattern)))
    if not paths:
        raise FileNotFoundError(f"no {pattern} in {EXPORTS}: run scripts/export_features.py and set RENEG_EXPORTS")
    return paths


# ------------------------------------------------------------------------------------------------ text side
def textbank() -> Dict:
    """TANL's corpus (69,554 unit rows, official order) with the words, NegLabel's split, ID text, noise images."""
    if "tb" not in _CACHE:
        _CACHE["tb"] = load_textbank_export(_need("reneg_textbank_*.npz"))
    return _CACHE["tb"]


def _text():
    if "text" not in _CACHE:
        tb = textbank()
        n = int(tb["n_selected"])
        _CACHE["text"] = {"id_text": T(tb["id_text"]), "neg_text": l2n(tb["corpus_text"][:n].float()),
                          "id_names": [str(s) for s in tb["id_names"]]}
    return _CACHE["text"]


def _pool():
    if "pool" not in _CACHE:
        z = np.load(_need("reneg_pool.npz")[0])
        with open(os.path.join(RESOURCES, "kg_pool.json")) as f:
            kg = json.load(f)["pool"]
        names = [str(s) for s in z["pool_names"]]
        if [e["name"] for e in kg] != names:
            raise ValueError("reneg_pool.npz and src/reneg/resources/kg_pool.json list different names")
        in_ssb, in_ninco = z["pool_in_ssb_hard"].astype(bool), z["pool_in_ninco"].astype(bool)
        _CACHE["pool"] = {
            "pool_text": T(z["pool_text"]), "pool_names": names, "pool_wnid": [str(s) for s in z["pool_wnid"]],
            "pool_wup": z["pool_wup"].astype(float), "pool_nearest": [str(s) for s in z["pool_nearest_id"]],
            "pool_in_ssb": in_ssb, "pool_in_ninco": in_ninco, "BLIND": ~(in_ssb | in_ninco),
            "pool_cluster": np.array([e["cluster"] for e in kg]), "pool_rel": np.array([e["rel"] for e in kg]),
            "pool_synset": [e["synset"] for e in kg]}
    return _CACHE["pool"]


def pool_subset(pool: str = "blind") -> Tuple[torch.Tensor, np.ndarray]:
    """(text, cluster) of the full or blind KG pool; ``+neg`` appends NegLabel's 10,000 negatives (one cluster each)."""
    p = _pool()
    idx = np.arange(len(p["BLIND"])) if pool.startswith("full") else np.nonzero(p["BLIND"])[0]
    P, cl = p["pool_text"][torch.as_tensor(idx)], p["pool_cluster"][idx]
    if pool.endswith("+neg"):
        neg = _text()["neg_text"]
        P = torch.cat([P, neg])
        cl = np.r_[cl, cl.max() + 1 + np.arange(neg.shape[0])]
    return P, cl


# --------------------------------------------------------------------------------------------- image sets
def _streams() -> Dict[str, Dict[str, np.ndarray]]:
    if "streams" not in _CACHE:
        st = {}
        for path in _need("reneg_stream*.npz"):
            z = np.load(path)
            for base in sorted(k[:-3] for k in z.files if k.endswith("__q")):
                st[base] = {k[len(base) + 2:]: z[k] for k in z.files if k.startswith(base + "__")}
        _CACHE["streams"] = st
    return _CACHE["streams"]


def has_set(name: str) -> bool:
    st = _streams()
    return name in st or f"{name}__part0" in st


def load_set(name: str) -> Dict:
    """dict(X (N, D) unit float32, labels (N,), folder (N,) or None, folder_names or None); parts are joined."""
    st = _streams()
    parts = [st[name]] if name in st else [st[f"{name}__part{k}"] for k in range(1000) if f"{name}__part{k}" in st]
    if not parts:
        raise KeyError(f"{name} is not in the exports at {EXPORTS} (scripts/export_features.py)")
    X = torch.cat([dq8(p["q"], p["s"]) for p in parts])
    labels = np.concatenate([p["labels"] for p in parts]).astype(int)
    folder, names = None, None
    if all("folder_names" in p for p in parts):
        names = sorted(set().union(*[set(map(str, p["folder_names"])) for p in parts]))
        idx = {f: i for i, f in enumerate(names)}
        folder = np.concatenate([np.array([idx[str(p["folder_names"][j])] for j in p["folder"]], dtype=int)
                                 for p in parts])
    return {"X": X, "labels": labels, "folder": folder, "folder_names": names}


def shots() -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """The few-shot setting's labelled images: OpenOOD's ImageNet ID-val split (5,000 images, about 5 per class,
    disjoint from OpenOOD v1.5's 45k ID test images). Returns (features, labels, mask over Four-OOD's 50k ID images).

    As the GPU pipeline (reneg.pipeline.shot_mask): a Four-OOD ID image is a shot if it matches an ID-val image
    (cosine > 0.999), 5,002 rows (4 near-duplicates). Without ``imagenet_val`` in the exports: the rows that match
    none of the 45k test images (cosine > 0.995), 4,998 rows, as the original local analysis did."""
    if "shots" in _CACHE:
        return _CACHE["shots"]
    va = load_set("imagenet_val_all")
    cache = out("shot_mask.npy")
    if has_set("imagenet_val"):
        v = load_set("imagenet_val")
        Xv, yv = v["X"], v["labels"]
        if os.path.isfile(cache):
            mask = np.load(cache)
        else:
            best = torch.full((len(va["X"]),), -1.0)
            for s in range(0, len(va["X"]), 4096):
                best[s:s + 4096] = (va["X"][s:s + 4096] @ Xv.T).max(1).values
            mask = (best > 0.999).numpy()
            np.save(cache, mask)
    else:
        if os.path.isfile(cache):
            mask = np.load(cache)
        else:
            Xt = load_set("imagenet_test")["X"]
            best = torch.full((len(va["X"]),), -1.0)
            for s in range(0, len(Xt), 4096):
                best = torch.maximum(best, (Xt[s:s + 4096] @ va["X"].T).max(0).values)
            mask = ~(best > 0.995).numpy()
            np.save(cache, mask)
        Xv, yv = va["X"][torch.as_tensor(np.nonzero(mask)[0])], va["labels"][mask]
    _CACHE["shots"] = (Xv, yv, mask)
    return _CACHE["shots"]


# -------------------------------------------------------------------------------- G1b analysis pack (Seam 1)
def pack1() -> Dict[str, np.ndarray]:
    """reneg_pack_1.npz (scripts/export_features.py --what pack): class statistics of the confident half of the
    ImageNet test images, the mean image, ImageNet val, OpenImage-O val and the SSB-hard / NINCO concept names."""
    if "pack1" not in _CACHE:
        cands = [os.path.join(EXPORTS, n) for n in ("reneg_pack_1.npz", "pack1.npz")]
        path = next((p for p in cands if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(f"no reneg_pack_1.npz in {EXPORTS}: scripts/export_features.py --what pack")
        _CACHE["pack1"] = np.load(path)
    return _CACHE["pack1"]


def _val():
    if "val" not in _CACHE:
        if has_set("imagenet_val") and has_set("openimage_o_val"):
            v = load_set("imagenet_val")
            _CACHE["val"] = (v["X"], v["labels"], load_set("openimage_o_val")["X"])
        else:
            p = pack1()
            _CACHE["val"] = (T(p["val_feats"]), p["val_labels"].astype(int), T(p["val_ood_feats"]))
    return _CACHE["val"]


def __getattr__(name):                     # PEP 562: module attributes that load on first use
    if name in ("id_text", "neg_text", "id_names"):
        return _text()[name]
    if name in ("pool_text", "pool_names", "pool_wnid", "pool_wup", "pool_nearest", "pool_in_ssb", "pool_in_ninco",
                "BLIND", "pool_cluster", "pool_rel", "pool_synset"):
        return _pool()[name]
    if name in ("val_X", "val_y", "val_ood"):
        return _val()[("val_X", "val_y", "val_ood").index(name)]
    if name == "stats":
        p = pack1()
        return ClassStats(torch.tensor(p["stats_sums"]).double(), torch.tensor(p["stats_counts"]).double())
    if name == "m_img":
        return torch.tensor(pack1()["m_img"]).float()
    if name == "id_wnids":
        return [str(s) for s in pack1()["id_wnids"]]
    raise AttributeError(name)


# ------------------------------------------------------------------------------------ scores and streams
def metrics(id_s, ood_s) -> Dict[str, float]:
    """fpr95 (OpenOOD: OOD positive), fpr95_idpos (MCM / NegLabel / TINS: ID positive), auroc; percent."""
    conf = np.concatenate([np.asarray(id_s, float), np.asarray(ood_s, float)])
    lab = np.concatenate([np.zeros(len(id_s), int), -np.ones(len(ood_s), int)])
    return _metrics_fallback(conf, lab)


def lse(S, tau=100.0):
    return torch.logsumexp(tau * S, 1)


def neglabel_form(X, chunk=4096, tau=100.0, negs=None, ids=None):
    """NegLabel's score: LSE over ID text minus LSE over the negatives."""
    negs = _text()["neg_text"] if negs is None else negs
    ids = _text()["id_text"] if ids is None else ids
    res = []
    for s in range(0, len(X), chunk):
        v = X[s:s + chunk]
        res.append((lse(v @ ids.T, tau) - lse(v @ negs.T, tau)).numpy())
    return np.concatenate(res)


def perm(n, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randperm(n, generator=g).numpy()


def pair_stream(X_id, X_ood, seed=0):
    """Official order: ConcatDataset([OOD, ID]) shuffled with torch.randperm(seed). Returns X, is_ood, idx."""
    X = torch.cat([X_ood, X_id])
    is_ood = np.r_[np.ones(len(X_ood), bool), np.zeros(len(X_id), bool)]
    p = perm(len(X), seed)
    return X[torch.as_tensor(p)], is_ood[p], p


def eval_mask_45k(p: np.ndarray, n_ood: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Stream positions scored on Four-OOD's 45k protocol: every OOD image and the ID images that are not shots."""
    mask = shots()[2] if mask is None else mask
    id_row = np.where(p >= n_ood, p - n_ood, -1)
    ev = np.ones(len(p), bool)
    ev[id_row >= 0] = ~mask[id_row[id_row >= 0]]
    return ev


class Timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.msg}] {time.time() - self.t:.1f}s", flush=True)
