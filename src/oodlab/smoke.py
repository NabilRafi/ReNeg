"""A miniature fake Kaggle input tree for testing the notebooks without data.

    make_fake_kaggle("/tmp/x/input")

creates
    imagenet-object-localization-challenge/   (LOC_synset_mapping.txt, LOC_val_solution.csv, flat val/)
    openood-raw-zips/                          (benchmark_imglist.zip + one zip per OOD dataset)
    fi-lists/                                  (fi_imagenet_1k_ood.txt, fi_imagenet_1k_ambiguous.txt)

The zips deliberately use two different internal layouts so that the list-path
resolver is exercised (NINCO sits under an extra ``images_largescale/`` folder).
"""
from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, List

import numpy as np

SMOKE_SIZES = {"val": 400, "inaturalist": 60, "sun": 50, "places": 50, "texture": 40, "ssb_hard": 80,
               "ninco": 40, "openimage_o": 60}


def _jpeg(rng: np.random.Generator, w: int = 64, h: int = 48, tint: int = 0) -> bytes:
    from PIL import Image

    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    arr[..., tint % 3] = np.clip(arr[..., tint % 3].astype(int) + 60, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _write_zip(path: str, members: Dict[str, bytes]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def make_fake_kaggle(inputs: str, seed: int = 0, sizes: Dict[str, int] = SMOKE_SIZES) -> Dict[str, str]:
    from . import imagenet_classnames
    from .data.imagenet import official_wnids

    marker = os.path.join(inputs, ".fake_kaggle_ok")
    base = os.path.join(inputs, "imagenet-object-localization-challenge")
    zips = os.path.join(inputs, "openood-raw-zips")
    fi = os.path.join(inputs, "fi-lists")
    out = {"imagenet": base, "zips": zips, "fi": fi}
    if os.path.exists(marker):
        return out
    rng = np.random.default_rng(seed)
    wnids, names = official_wnids(), imagenet_classnames()

    # ---- ImageNet (Kaggle competition layout) ------------------------------------------
    val_dir = os.path.join(base, "ILSVRC", "Data", "CLS-LOC", "val")
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(base, "ILSVRC", "Data", "CLS-LOC", "train", wnids[0]), exist_ok=True)
    with open(os.path.join(base, "LOC_synset_mapping.txt"), "w") as f:
        for w, n in zip(wnids, names):
            f.write(f"{w} {n}\n")
    n_val = sizes["val"]
    labels = [(i * 7) % 20 for i in range(n_val)]
    ids = [f"ILSVRC2012_val_{i + 1:08d}" for i in range(n_val)]
    with open(os.path.join(base, "LOC_val_solution.csv"), "w") as f:
        f.write("ImageId,PredictionString\n")
        for i, lab in zip(ids, labels):
            f.write(f"{i},{wnids[lab]} 1 2 30 40 {wnids[lab]} 3 4 20 20\n")
    for i, lab in zip(ids, labels):
        with open(os.path.join(val_dir, f"{i}.JPEG"), "wb") as f:
            f.write(_jpeg(rng, tint=lab))

    # ---- OpenOOD image lists -----------------------------------------------------------
    lists: Dict[str, List[str]] = {}
    n_test = int(n_val * 0.9)
    lists["test_imagenet.txt"] = [f"imagenet_1k/val/{wnids[l]}/{i}.JPEG {l}" for i, l in zip(ids[:n_test], labels[:n_test])]
    lists["val_imagenet.txt"] = [f"imagenet_1k/val/{wnids[l]}/{i}.JPEG {l}" for i, l in zip(ids[n_test:], labels[n_test:])]

    members: Dict[str, Dict[str, bytes]] = {}

    def add(key: str, rel: str, zip_prefix: str = "") -> None:
        members.setdefault(key, {})[zip_prefix + rel] = _jpeg(rng, tint=len(members.get(key, {})))

    inat = [f"inaturalist/images/{k:05d}.jpg" for k in range(sizes["inaturalist"])]
    for r in inat:
        add("inaturalist", r)
    lists["test_inaturalist.txt"] = [f"{r} -1" for r in inat]
    for key in ("sun", "places"):
        for k in range(sizes[key]):
            add(key, f"{key}/images/{key}_{k:05d}.jpg")
    cats = ["banded", "dotted", "striped", "zigzagged", "woven"]
    tex = [f"texture/images/{c}/{c}_{k:04d}.jpg" for c in cats for k in range(sizes["texture"] // len(cats))]
    for r in tex:
        add("texture", r)
    lists["test_textures.txt"] = [f"{r} -1" for r in tex[::2]]   # OpenOOD's Textures list is a subset
    ssb = [f"ssb_hard/n{10000000 + k // 8:08d}/img_{k:05d}.JPEG" for k in range(sizes["ssb_hard"])]
    for r in ssb:
        add("ssb_hard", r)
    lists["test_ssb_hard.txt"] = [f"{r} -1" for r in ssb]
    ninco = [f"ninco/NINCO_OOD_classes/class_{k // 10}/ninco_{k:05d}.jpg" for k in range(sizes["ninco"])]
    for r in ninco:
        add("ninco", r, zip_prefix="images_largescale/")        # different layout on purpose
    lists["test_ninco.txt"] = [f"{r} -1" for r in ninco]
    oi = [f"openimage_o/images/{k:016x}.jpg" for k in range(sizes["openimage_o"])]
    for r in oi:
        add("openimage_o", r)
    n_oi_val = max(1, len(oi) // 6)
    lists["val_openimage_o.txt"] = [f"{r} -1" for r in oi[:n_oi_val]]
    lists["test_openimage_o.txt"] = [f"{r} -1" for r in oi[n_oi_val:]]

    _write_zip(os.path.join(zips, "benchmark_imglist.zip"),
               {f"benchmark_imglist/imagenet/{k}": ("\n".join(v) + "\n").encode() for k, v in lists.items()})
    for key, m in members.items():
        _write_zip(os.path.join(zips, f"{key}.zip"), m)

    # ---- Fi-ImageNet-1k style lists -------------------------------------------------------
    os.makedirs(fi, exist_ok=True)
    with open(os.path.join(fi, "fi_imagenet_1k_ood.txt"), "w") as f:
        f.write("\n".join(f"{i}.JPEG" for i in ids[5:25]) + "\n")
    with open(os.path.join(fi, "fi_imagenet_1k_ambiguous.txt"), "w") as f:
        f.write("\n".join(ids[30:45]) + "\n")
    open(marker, "w").close()
    return out
