"""Feature cache: every image is encoded once; all experiments replay streams from here.

Layout (one folder per backbone)::

    <root>/<backbone-tag>/
        CACHE_INFO.json            what is inside + versions (written after every change)
        textbank.pt                ID / corpus / noise text-side features  (oodlab.textbank)
        images/<set>.npz           feats (N, D) fp16 | labels (N,) int64 | keys (N,) str
        images/<set>.json          manifest meta, timing, dropped (unreadable) images
        manifests/<set>.csv        the exact image list that was encoded

Encoding is resumable: it writes ``images/<set>.partial/shard_XXXXX.npz`` files of
``shard_size`` images and skips finished shards after a restart.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .clipwrap import backbone_tag, build_transform, encode_image_batch, input_resolution, load_image, model_dtype
from .data.manifest import Manifest
from .utils import atomic_np_savez, read_json, write_json

CACHE_FORMAT = "oodlab.cache.v1"


@dataclass
class FeatureSet:
    name: str
    feats: torch.Tensor        # (N, D) fp16, L2-normalised
    labels: np.ndarray         # (N,) int64; -1 for OOD
    keys: List[str]
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    @property
    def dim(self) -> int:
        return int(self.feats.shape[1])

    def subset(self, idx: Sequence[int], name: Optional[str] = None) -> "FeatureSet":
        idx_t = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        return FeatureSet(name or self.name, self.feats[idx_t], self.labels[np.asarray(idx, dtype=np.int64)],
                          [self.keys[i] for i in np.asarray(idx, dtype=np.int64).tolist()], dict(self.meta))

    def exclude_keys(self, keys: Sequence[str], name: Optional[str] = None) -> "FeatureSet":
        drop = set(keys)
        keep = [i for i, k in enumerate(self.keys) if k not in drop]
        return self.subset(keep, name=name)

    def select_keys(self, keys: Sequence[str], name: Optional[str] = None, label: Optional[int] = None) -> "FeatureSet":
        pos = {k: i for i, k in enumerate(self.keys)}
        missing = [k for k in keys if k not in pos]
        if missing:
            raise KeyError(f"{len(missing)} keys not in {self.name}, e.g. {missing[:3]}")
        fs = self.subset([pos[k] for k in keys], name=name)
        if label is not None:
            fs.labels = np.full(len(fs), label, dtype=np.int64)
        return fs


class FeatureCache:
    def __init__(self, root: str, backbone: str = "ViT-B/16"):
        self.root = root
        self.backbone = backbone
        self.dir = os.path.join(root, backbone_tag(backbone))
        self.images_dir = os.path.join(self.dir, "images")
        self.manifests_dir = os.path.join(self.dir, "manifests")
        self.textbank_path = os.path.join(self.dir, "textbank.pt")
        self.info_path = os.path.join(self.dir, "CACHE_INFO.json")

    def make(self) -> "FeatureCache":
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.manifests_dir, exist_ok=True)
        return self

    # ------------------------------------------------------------------ queries
    def set_path(self, name: str) -> str:
        return os.path.join(self.images_dir, f"{name}.npz")

    def has(self, name: str) -> bool:
        return os.path.isfile(self.set_path(name))

    def has_textbank(self) -> bool:
        return os.path.isfile(self.textbank_path)

    def available(self) -> List[str]:
        if not os.path.isdir(self.images_dir):
            return []
        return sorted(f[:-4] for f in os.listdir(self.images_dir) if f.endswith(".npz"))

    # ------------------------------------------------------------------ io
    def save_set(self, fs: FeatureSet) -> str:
        self.make()
        path = self.set_path(fs.name)
        atomic_np_savez(path, feats=fs.feats.to(torch.float16).cpu().numpy(), labels=fs.labels.astype(np.int64),
                        keys=np.asarray(fs.keys, dtype=str))
        write_json({"name": fs.name, "n": len(fs), "dim": fs.dim, **fs.meta}, os.path.join(self.images_dir, f"{fs.name}.json"))
        self.write_info()
        return path

    def load_set(self, name: str) -> FeatureSet:
        path = self.set_path(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} is not in the cache ({self.dir}); available: {self.available()}")
        with np.load(path, allow_pickle=False) as z:
            feats = torch.from_numpy(z["feats"].copy())
            labels = z["labels"].astype(np.int64)
            keys = z["keys"].tolist()
        meta_path = os.path.join(self.images_dir, f"{name}.json")
        meta = read_json(meta_path) if os.path.exists(meta_path) else {}
        return FeatureSet(name, feats, labels, keys, meta)

    def load_textbank(self):
        from .textbank import TextBank

        if not self.has_textbank():
            raise FileNotFoundError(f"no textbank.pt in {self.dir} - run notebook 01 first")
        return TextBank.load(self.textbank_path)

    def save_textbank(self, bank) -> None:
        self.make()
        bank.save(self.textbank_path)
        self.write_info()

    def write_info(self) -> None:
        sets = {}
        for n in self.available():
            mp = os.path.join(self.images_dir, f"{n}.json")
            m = read_json(mp) if os.path.exists(mp) else {}
            sets[n] = {"n": m.get("n"), "dim": m.get("dim"), "dropped": len(m.get("dropped_keys", []))}
        info = {"format": CACHE_FORMAT, "backbone": self.backbone, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "textbank": self.has_textbank(), "sets": sets}
        write_json(info, self.info_path)

    def summary(self) -> List[Dict]:
        rows = []
        for n in self.available():
            mp = os.path.join(self.images_dir, f"{n}.json")
            m = read_json(mp) if os.path.exists(mp) else {}
            rows.append({"set": n, "images": m.get("n"), "dim": m.get("dim"), "dropped": len(m.get("dropped_keys", [])),
                         "expected": m.get("expected"), "img_per_s": m.get("img_per_s")})
        return rows


def find_existing_caches(input_roots: Sequence[str], max_depth: int = 5) -> List[str]:
    """Cache roots inside attached inputs (folders holding ``*/CACHE_INFO.json``)."""
    from .paths import SKIP_DIR_NAMES

    found = []
    frontier = [(r, 0) for r in input_roots if os.path.isdir(r)]
    while frontier:
        d, depth = frontier.pop(0)
        if os.path.isfile(os.path.join(d, "CACHE_INFO.json")):
            found.append(os.path.dirname(d))
            continue
        if depth >= max_depth:
            continue
        try:
            with os.scandir(d) as it:
                subs = []
                for i, e in enumerate(it):
                    if i >= 2000:
                        subs = []
                        break
                    if e.is_dir(follow_symlinks=False) and e.name not in SKIP_DIR_NAMES:
                        subs.append(e.path)
        except (PermissionError, FileNotFoundError):
            continue
        frontier.extend((s, depth + 1) for s in sorted(subs))
    return sorted(set(found))


def import_from(src_root: str, dst: FeatureCache, names: Optional[Sequence[str]] = None, logger=None) -> List[str]:
    """Copy finished sets (and the text bank) from an attached cache into ``dst``."""
    src = FeatureCache(src_root, dst.backbone)
    if not os.path.isdir(src.dir):
        return []
    dst.make()
    copied = []
    if src.has_textbank() and not dst.has_textbank():
        shutil.copy2(src.textbank_path, dst.textbank_path)
        copied.append("textbank")
    # unfinished encodings: copy their finished shards so encoding resumes where it stopped
    for d in sorted(os.listdir(src.images_dir)) if os.path.isdir(src.images_dir) else []:
        if d.endswith(".partial") and not dst.has(d[: -len(".partial")]):
            target = os.path.join(dst.images_dir, d)
            if not os.path.exists(target):
                shutil.copytree(os.path.join(src.images_dir, d), target)
                copied.append(d)
    for n in src.available():
        if names is not None and n not in names:
            continue
        if dst.has(n):
            continue
        for ext in (".npz", ".json"):
            p = os.path.join(src.images_dir, n + ext)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(dst.images_dir, n + ext))
        mp = os.path.join(src.manifests_dir, n + ".csv")
        if os.path.exists(mp):
            shutil.copy2(mp, os.path.join(dst.manifests_dir, n + ".csv"))
        copied.append(n)
    if copied:
        dst.write_info()
        if logger:
            logger.info(f"[cache] imported from {src_root}: {copied}")
    return copied


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[str], transform, size: int):
        self.paths = list(paths)
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            x = self.transform(load_image(self.paths[i]))
            ok = True
        except Exception:  # unreadable/corrupt file: recorded and dropped later
            x = torch.zeros(3, self.size, self.size)
            ok = False
        return x, i, ok


def shard_fingerprint(manifest: Manifest, shard_size: int, backbone: str) -> str:
    import hashlib

    return hashlib.sha1(("\n".join(manifest.keys) + f"|{shard_size}|{backbone}").encode()).hexdigest()


def _default_workers() -> int:
    n = os.cpu_count() or 2
    return max(0, min(8, n))


@torch.no_grad()
def encode_manifest(
    model,
    manifest: Manifest,
    cache: FeatureCache,
    batch_size: int = 256,
    num_workers: Optional[int] = None,
    shard_size: int = 10000,
    max_bad_frac: float = 0.001,
    logger=None,
    log_every: int = 20,
) -> FeatureSet:
    """Encode every image of ``manifest`` and store the result in ``cache`` (resumable)."""
    cache.make()
    name = manifest.name
    log = logger.info if logger else print
    size = input_resolution(model)
    transform = build_transform(image_size=size, pre_size=round(size * 256 / 224))
    workers = _default_workers() if num_workers is None else num_workers
    part_dir = os.path.join(cache.images_dir, f"{name}.partial")
    # shards are only reusable for the very same image list, shard size and backbone
    fp = shard_fingerprint(manifest, shard_size, cache.backbone)
    fp_path = os.path.join(part_dir, "fingerprint.json")
    if os.path.isdir(part_dir):
        old = json.load(open(fp_path)).get("fp") if os.path.exists(fp_path) else None
        if old != fp:
            log(f"[encode] {name}: discarding partial shards of a different image list / shard size")
            shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir, exist_ok=True)
    with open(fp_path, "w") as f:
        json.dump({"fp": fp, "n": len(manifest), "shard_size": shard_size}, f)
    manifest.save(cache.manifests_dir)

    n = len(manifest)
    n_shards = (n + shard_size - 1) // shard_size
    t_start = time.time()
    done_imgs = 0
    for s in range(n_shards):
        shard_path = os.path.join(part_dir, f"shard_{s:05d}.npz")
        lo, hi = s * shard_size, min(n, (s + 1) * shard_size)
        if os.path.isfile(shard_path):
            log(f"[encode] {name}: shard {s + 1}/{n_shards} already done")
            continue
        ds = _ImageDataset(manifest.paths[lo:hi], transform, size)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=False,
            prefetch_factor=4 if workers > 0 else None,
        )
        feats = None
        ok_all = np.zeros(hi - lo, dtype=bool)
        t0 = time.time()
        for b, (x, idx, ok) in enumerate(dl):
            f = encode_image_batch(model, x).to(torch.float16).cpu()
            if feats is None:
                feats = torch.empty(hi - lo, f.shape[1], dtype=torch.float16)
            feats[idx] = f
            ok_all[idx.numpy()] = ok.numpy().astype(bool)
            if (b + 1) % log_every == 0:
                k = min((b + 1) * batch_size, hi - lo)
                rate = k / max(time.time() - t0, 1e-6)
                left = (n - lo - k) / max(rate, 1e-6)
                log(f"[encode] {name}: {lo + k}/{n} images  {rate:.0f} img/s  ~{left / 60:.1f} min left")
        atomic_np_savez(shard_path, feats=feats.numpy(), ok=ok_all)
        done_imgs += hi - lo
        log(f"[encode] {name}: shard {s + 1}/{n_shards} saved ({hi - lo} images, {time.time() - t0:.0f}s)")

    # ---- assemble ---------------------------------------------------------------
    parts, oks = [], []
    for s in range(n_shards):
        with np.load(os.path.join(part_dir, f"shard_{s:05d}.npz")) as z:
            parts.append(z["feats"])
            oks.append(z["ok"])
    feats = torch.from_numpy(np.concatenate(parts, axis=0))
    ok = np.concatenate(oks)
    bad = np.flatnonzero(~ok)
    if len(bad) > max_bad_frac * n:
        raise RuntimeError(f"{name}: {len(bad)} of {n} images could not be read, e.g. {[manifest.paths[i] for i in bad[:3]]}")
    keep = np.flatnonzero(ok)
    dropped = [manifest.keys[i] for i in bad.tolist()]
    if dropped:
        log(f"[encode] WARNING {name}: dropped {len(dropped)} unreadable images, e.g. {dropped[:3]}")
    norms = feats[torch.as_tensor(keep)].float().norm(dim=1)
    if not torch.isfinite(norms).all() or (norms - 1).abs().max() > 1e-2:
        raise RuntimeError(f"{name}: features are not unit-norm (min {norms.min():.4f}, max {norms.max():.4f})")
    elapsed = time.time() - t_start
    meta = dict(manifest.meta)
    meta.update({
        "backbone": cache.backbone,
        "dropped_keys": dropped,
        "encode_seconds": round(elapsed, 1),
        "img_per_s": round(done_imgs / elapsed, 1) if done_imgs and elapsed > 0 else None,
        "model_dtype": str(model_dtype(model)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    fs = FeatureSet(name, feats[torch.as_tensor(keep)], manifest.labels[keep], [manifest.keys[i] for i in keep.tolist()], meta)
    cache.save_set(fs)
    shutil.rmtree(part_dir, ignore_errors=True)
    log(f"[encode] {name}: {len(fs)} features saved -> {cache.set_path(name)}")
    return fs
