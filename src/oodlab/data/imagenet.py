"""ImageNet-1K validation set on Kaggle (competition "imagenet-object-localization-challenge").

Kaggle layout (flat validation folder, labels in a CSV):

    <base>/LOC_synset_mapping.txt           n01440764 tench, Tinca tinca   (line i -> class i)
    <base>/LOC_val_solution.csv             ImageId,PredictionString
    <base>/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG ... (50,000 files)

OpenOOD's lists (``test_imagenet.txt`` 45k / ``val_imagenet.txt`` 5k) use class
sub-folders instead; we align them to Kaggle files by the ``ILSVRC2012_val_XXXXXXXX`` id.
We never list or walk the (1.3M-file) training directory.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..paths import find_file
from .manifest import Manifest

KAGGLE_SLUGS = ["imagenet-object-localization-challenge", "imagenet-object-localization-challenge-2012"]
VAL_ID_RE = re.compile(r"(ILSVRC2012_val_\d{8})")


@dataclass
class ImageNetLocation:
    base: str
    synset_mapping: str
    val_solution: Optional[str]
    val_dir: str
    layout: str  # "flat" (Kaggle) or "classdirs"


def locate_imagenet(input_roots: Sequence[str]) -> ImageNetLocation:
    """Find the Kaggle ImageNet competition data (bounded search, no recursion into train/)."""
    candidates = []
    for root in input_roots:
        for slug in KAGGLE_SLUGS:
            candidates.append(os.path.join(root, slug))
            candidates.append(os.path.join(root, "competitions", slug))
    base = next((c for c in candidates if os.path.isfile(os.path.join(c, "LOC_synset_mapping.txt"))), None)
    if base is None:
        found = find_file("LOC_synset_mapping.txt", input_roots, max_depth=3)
        if found is None:
            raise FileNotFoundError(
                "Could not find LOC_synset_mapping.txt under the attached inputs. In Kaggle: "
                "Add Input -> Competitions -> 'ImageNet Object Localization Challenge' (accept the rules first)."
            )
        base = os.path.dirname(found)
    synsets = os.path.join(base, "LOC_synset_mapping.txt")
    sol = os.path.join(base, "LOC_val_solution.csv")
    val_dir = os.path.join(base, "ILSVRC", "Data", "CLS-LOC", "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Expected validation images at {val_dir}")
    probe = os.path.join(val_dir, "ILSVRC2012_val_00000001.JPEG")
    layout = "flat" if os.path.isfile(probe) else "classdirs"
    return ImageNetLocation(base, synsets, sol if os.path.isfile(sol) else None, val_dir, layout)


def read_synsets(path: str) -> Tuple[List[str], List[str]]:
    wnids, names = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, _, name = line.partition(" ")
            wnids.append(wnid)
            names.append(name)
    if len(wnids) != 1000:
        raise ValueError(f"{path}: expected 1000 synsets, got {len(wnids)}")
    if wnids != sorted(wnids):
        raise ValueError("synset mapping is not in sorted wnid order; class indices would be wrong")
    official = official_wnids()
    if wnids != official:
        bad = sum(a != b for a, b in zip(wnids, official))
        raise ValueError(f"{path}: {bad} wnids differ from OpenOOD-VLM's class order")
    return wnids, names


def official_wnids() -> List[str]:
    """Class order used by OpenOOD-VLM (``all_wnids``); index i <-> class name i."""
    import json

    from .. import RESOURCES

    with open(os.path.join(RESOURCES, "imagenet_wnids.json")) as f:
        return json.load(f)


def read_val_solution(path: str) -> Dict[str, str]:
    """ImageId -> wnid (first token of PredictionString)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["ImageId"]] = row["PredictionString"].split()[0]
    return out


def imagenet_val_manifest(loc: ImageNetLocation, name: str = "imagenet_val_all", expected: Optional[int] = 50000) -> Manifest:
    wnids, _ = read_synsets(loc.synset_mapping)
    idx_of = {w: i for i, w in enumerate(wnids)}
    if loc.layout == "flat":
        if loc.val_solution is None:
            raise FileNotFoundError("Flat val folder but LOC_val_solution.csv is missing")
        sol = read_val_solution(loc.val_solution)
        ids = sorted(sol)
        paths = [os.path.join(loc.val_dir, f"{i}.JPEG") for i in ids]
        labels = [idx_of[sol[i]] for i in ids]
    else:  # val/<wnid>/*.JPEG
        ids, paths, labels = [], [], []
        for w in wnids:
            d = os.path.join(loc.val_dir, w)
            for fn in sorted(os.listdir(d)):
                m = VAL_ID_RE.search(fn)
                if m:
                    ids.append(m.group(1))
                    paths.append(os.path.join(d, fn))
                    labels.append(idx_of[w])
    if expected is not None and len(ids) != expected:
        raise ValueError(f"expected {expected:,} validation images, found {len(ids)}")
    return Manifest(name=name, paths=paths, labels=np.asarray(labels), keys=ids,
                    meta={"source": "kaggle-imagenet", "layout": loc.layout})


def restrict_to_openood_list(full: Manifest, entries: List[Tuple[str, int]], name: str) -> Manifest:
    """Subset of the 50k manifest listed in an OpenOOD image list, checking labels agree."""
    pos = {k: i for i, k in enumerate(full.keys)}
    idx, mismatched = [], 0
    for rel, lab in entries:
        m = VAL_ID_RE.search(rel)
        if not m or m.group(1) not in pos:
            raise KeyError(f"{name}: cannot map list entry {rel!r} to a Kaggle validation image")
        i = pos[m.group(1)]
        if lab >= 0 and int(full.labels[i]) != lab:
            mismatched += 1
        idx.append(i)
    if mismatched:
        raise ValueError(f"{name}: {mismatched} labels in the OpenOOD list disagree with Kaggle's solution file")
    sub = full.subset(idx, name=name)
    sub.meta.update({"source": "kaggle-imagenet + OpenOOD list", "list_entries": len(entries)})
    return sub
