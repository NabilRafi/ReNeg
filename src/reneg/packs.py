"""Compact exports of the feature cache for analysis outside Colab, and the WordNet near-OOD candidate pool.

Image features are stored as int8 with one fp16 scale per image (x = q * scale / 127): half the size of fp16,
and a dot product with a unit prototype changes by about 3e-4 (0.03 logits at temperature 100).

The candidate pool (``resources/kg_pool.json``) was built from WordNet 3.0 around the 1,000 ImageNet classes:
every noun up to two levels above a class and down to its cousins, minus the ImageNet classes themselves,
all their ancestors and descendants, and any name equal to an ImageNet lemma. 14,526 unique names. Each entry
records its Wu-Palmer similarity to the nearest ImageNet class and whether the name is an SSB-hard or NINCO
class (for the blind variant, which removes benchmark classes from the pool).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .concepts import folder_of
from .transport import l2n

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def load_kg_pool(path: Optional[str] = None) -> Dict:
    with open(path or os.path.join(RESOURCES, "kg_pool.json")) as f:
        return json.load(f)


def q8(X: torch.Tensor):
    """int8 codes and fp16 scales of L2-normalised rows."""
    X = l2n(torch.as_tensor(X).float())
    s = X.abs().max(1).values.clamp_min(1e-8)
    q = torch.round(X / s[:, None] * 127).clamp(-127, 127).to(torch.int8)
    return q.numpy(), s.half().numpy()


def dq8(q: np.ndarray, s: np.ndarray) -> torch.Tensor:
    return l2n(torch.as_tensor(q, dtype=torch.float32) * torch.as_tensor(s, dtype=torch.float32)[:, None] / 127.0)


def _set_arrays(fs, name: str) -> Dict[str, np.ndarray]:
    q, s = q8(fs.feats)
    out = {f"{name}__q": q, f"{name}__s": s, f"{name}__labels": np.asarray(fs.labels).astype(np.int16)}
    folders = [folder_of(k) for k in fs.keys]
    uniq = sorted(set(folders))
    if 1 < len(uniq) < 5000:                          # class folders (SSB-hard wnids, NINCO / Textures names)
        idx = {f: i for i, f in enumerate(uniq)}
        out[f"{name}__folder_names"] = np.asarray(uniq, dtype=str)
        out[f"{name}__folder"] = np.asarray([idx[f] for f in folders], dtype=np.int16)
    return out


def export_streams(cache, out_dir: str, sets: Sequence[str], max_mb: float = 24.0, prefix: str = "reneg_stream",
                   log=print) -> List[str]:
    """Write the named cache sets into as few .npz files as fit under ``max_mb`` each (a set too large for one
    file is split into parts named <set>__partK)."""
    os.makedirs(out_dir, exist_ok=True)
    budget = max_mb * 1e6
    files, cur, cur_bytes = [], {}, 0.0

    def flush():
        nonlocal cur, cur_bytes
        if cur:
            path = os.path.join(out_dir, f"{prefix}_{len(files) + 1}.npz")
            np.savez_compressed(path, **cur)
            files.append(path)
            log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB): {sorted({k.split('__')[0] for k in cur})}")
        cur, cur_bytes = {}, 0.0

    for name in sets:
        if not cache.has(name):
            log(f"skipped {name}: not in the cache")
            continue
        fs = cache.load_set(name)
        per_img = fs.feats.shape[1] + 8
        n_parts = int(np.ceil(len(fs) * per_img / budget))
        bounds = np.linspace(0, len(fs), n_parts + 1).astype(int)
        for p in range(n_parts):
            sub = fs.subset(np.arange(bounds[p], bounds[p + 1]))
            key = name if n_parts == 1 else f"{name}__part{p}"
            arrs = _set_arrays(sub, key)
            size = sum(a.nbytes for a in arrs.values())
            if cur_bytes + size > budget:
                flush()
            cur.update(arrs)
            cur_bytes += size
    flush()
    return files


def export_pool(bank, text_encoder, out_dir: str, pool: Optional[Dict] = None, prompt_note: str = "",
                log=print) -> str:
    """CLIP text embeddings (fp16) of the candidate pool, plus the words behind NegLabel's 10,000 negatives."""
    pool = pool or load_kg_pool()
    rows = pool["pool"]
    names = [r["name"] for r in rows]
    embs = []
    for s in range(0, len(names), 2000):
        embs.append(text_encoder(names[s : s + 2000]).float())
    E = l2n(torch.cat(embs))
    words = list(getattr(bank, "words", [])[: int(getattr(bank, "n_selected", 0))])
    out = {"pool_text": E.half().numpy(), "pool_names": np.asarray(names, dtype=str),
           "pool_wnid": np.asarray([r["wnid"] for r in rows], dtype=str),
           "pool_wup": np.asarray([r["wup"] for r in rows], dtype=np.float32),
           "pool_nearest_id": np.asarray([r["nearest_id"] for r in rows], dtype=str),
           "pool_in_ssb_hard": np.asarray([r["in_ssb_hard"] for r in rows], dtype=bool),
           "pool_in_ninco": np.asarray([r["in_ninco"] for r in rows], dtype=bool),
           "neg_words": np.asarray(words, dtype=str), "id_names": np.asarray(list(bank.id_names), dtype=str),
           "prompt": np.asarray(prompt_note)}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "reneg_pool.npz")
    np.savez_compressed(path, **out)
    log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB): {len(names)} pool names, {len(words)} NegLabel words")
    return path


def export_textbank(bank, out_dir: str, max_mb: float = 24.0, prefix: str = "reneg_textbank", log=print) -> List[str]:
    """TANL's whole word corpus (69,554 words in the official order) as 8-bit codes, with the words, NegLabel's
    split, the ID text and the 15 noise-image features: everything TANL needs to run on the exported streams."""
    os.makedirs(out_dir, exist_ok=True)
    q, s = q8(bank.corpus_text)
    n, per = q.shape[0], q.shape[1] + 2
    n_parts = max(1, int(np.ceil(n * per / (max_mb * 1e6))))
    bounds = np.linspace(0, n, n_parts + 1).astype(int)
    files = []
    for p in range(n_parts):
        a, b = int(bounds[p]), int(bounds[p + 1])
        arr = {"corpus_q": q[a:b], "corpus_s": s[a:b], "start": np.asarray(a), "part": np.asarray(p),
               "n_parts": np.asarray(n_parts), "n_corpus": np.asarray(n)}
        if p == 0:
            arr.update(words=np.asarray(list(bank.words), dtype=str),
                       is_adj=np.asarray(torch.as_tensor(bank.is_adj).cpu().numpy(), dtype=bool),
                       n_selected=np.asarray(int(bank.n_selected)),
                       noise_feats=torch.as_tensor(bank.noise_feats).float().cpu().numpy(),
                       id_text=torch.as_tensor(bank.id_text).float().cpu().numpy(),
                       id_names=np.asarray(list(bank.id_names), dtype=str),
                       logit_scale=np.asarray(float(getattr(bank, "logit_scale", 100.0))))
        path = os.path.join(out_dir, f"{prefix}_{p + 1}.npz")
        np.savez_compressed(path, **arr)
        files.append(path)
        log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB): corpus rows {a}-{b}")
    return files


def load_textbank_export(paths: Sequence[str]) -> Dict:
    """Inverse of export_textbank: corpus (N, D) float32 unit rows plus the metadata."""
    parts = sorted((np.load(p, allow_pickle=False) for p in paths), key=lambda z: int(z["start"]))
    out = {k: parts[0][k] for k in parts[0].files if not k.startswith("corpus_")}
    out["corpus_text"] = torch.cat([dq8(z["corpus_q"], z["corpus_s"]) for z in parts])
    if out["corpus_text"].shape[0] != int(out["n_corpus"]):
        raise ValueError(f"missing parts: {out['corpus_text'].shape[0]} of {int(out['n_corpus'])} corpus rows")
    return out


def load_kg_hood(path: Optional[str] = None) -> Dict:
    """Per ImageNet class: [pool index, relation index, Wu-Palmer] for each pool name in its WordNet neighbourhood
    (siblings, parent's siblings, siblings' children, cousins, cousins' children)."""
    with open(path or os.path.join(RESOURCES, "kg_hood.json")) as f:
        return json.load(f)
