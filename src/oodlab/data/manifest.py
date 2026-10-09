"""A manifest = the exact list of images (and labels) that make up one evaluation set."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils import read_json, write_json


@dataclass
class Manifest:
    name: str
    paths: List[str]                 # absolute paths used for encoding
    labels: np.ndarray               # int64; ImageNet class index for ID, -1 for OOD
    keys: List[str]                  # stable identifiers (e.g. relative path) for alignment
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if not (len(self.paths) == len(self.labels) == len(self.keys)):
            raise ValueError(f"{self.name}: paths/labels/keys lengths differ")

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def is_ood(self) -> bool:
        return bool(len(self.labels)) and bool((self.labels == -1).all())

    def subset(self, idx: Sequence[int], name: Optional[str] = None) -> "Manifest":
        idx = list(idx)
        return Manifest(
            name=name or self.name,
            paths=[self.paths[i] for i in idx],
            labels=self.labels[idx],
            keys=[self.keys[i] for i in idx],
            meta=dict(self.meta),
        )

    def missing_files(self, sample: Optional[int] = 500, seed: int = 0) -> List[str]:
        """Paths that do not exist (checks a random sample unless ``sample=None``)."""
        n = len(self.paths)
        if sample is None or sample >= n:
            idx = range(n)
        else:
            idx = np.random.default_rng(seed).choice(n, size=sample, replace=False)
        return [self.paths[i] for i in idx if not os.path.isfile(self.paths[i])]

    # ------------------------------------------------------------------ io
    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        csv_path = os.path.join(directory, f"{self.name}.csv")
        tmp = csv_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "label", "path"])
            for k, l, p in zip(self.keys, self.labels.tolist(), self.paths):
                w.writerow([k, l, p])
        os.replace(tmp, csv_path)
        write_json({"name": self.name, "n": len(self), **self.meta}, os.path.join(directory, f"{self.name}.json"))
        return csv_path

    @classmethod
    def load(cls, directory: str, name: str) -> "Manifest":
        csv_path = os.path.join(directory, f"{name}.csv")
        keys, labels, paths = [], [], []
        with open(csv_path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                keys.append(row["key"])
                labels.append(int(row["label"]))
                paths.append(row["path"])
        meta_path = os.path.join(directory, f"{name}.json")
        meta = read_json(meta_path) if os.path.exists(meta_path) else {}
        meta.pop("name", None)
        meta.pop("n", None)
        return cls(name=name, paths=paths, labels=np.asarray(labels), keys=keys, meta=meta)
