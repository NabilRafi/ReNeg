"""Test streams built from cached features.

The official evaluator (``OODEvaluatorClipTTA._eval_ood``) does, for every OOD set::

    postprocessor.reset_memory()
    combined = ConcatDataset([ood_dataset, id_dataset])
    loader = DataLoader(combined, batch_size=256, shuffle=True)   # unseeded shuffle

``pair_stream`` reproduces this with a *seeded* permutation (``torch.randperm`` with a
``torch.Generator``, which is what ``DataLoader(shuffle=True)`` uses internally).

``temporal_stream`` implements TANL's temporal-shift protocol (Tab. A15): OOD sets
arrive one after another and the memory is **not** reset between them. Each segment
is a shuffled (OOD_k + ID) set, batched on its own, exactly as if the official loop
ran without ``reset_memory()``. (The official configs named ``temshift`` still reset
per dataset, so they do not implement this; see GUIDE.md.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .cache import FeatureSet

ORDERS = ("shuffled", "id_first", "ood_first")


@dataclass
class Segment:
    name: str        # OOD set name of this segment
    start: int       # position in the stream (inclusive)
    end: int         # (exclusive)
    reset: bool      # reset the method's memory at the start of this segment


@dataclass
class Stream:
    """A sequence of samples (rows of ``bank``) split into segments and batches."""

    bank: torch.Tensor                 # (R, D) all features the stream can refer to
    rows: np.ndarray                   # (T,) row of ``bank`` for each stream position
    labels: np.ndarray                 # (T,) ImageNet class (ID) or -1 (OOD)
    segments: List[Segment]
    batch_size: int = 256
    sources: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name -> (offset, n) in bank
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.rows))

    def batches(self) -> Iterator[Tuple[int, int, int, bool]]:
        """Yields (segment index, start, end, reset_before). Batches never cross segments."""
        for si, seg in enumerate(self.segments):
            first = True
            for s in range(seg.start, seg.end, self.batch_size):
                yield si, s, min(s + self.batch_size, seg.end), (seg.reset and first)
                first = False

    def n_batches(self) -> int:
        return sum((seg.end - seg.start + self.batch_size - 1) // self.batch_size for seg in self.segments)

    def feats(self, s: int, e: int, device="cpu") -> torch.Tensor:
        idx = torch.as_tensor(self.rows[s:e], device=self.bank.device)
        return self.bank.index_select(0, idx).to(device)

    def to(self, device) -> "Stream":
        self.bank = self.bank.to(device)
        return self

    def source_of(self, positions: np.ndarray) -> np.ndarray:
        """Name of the source set of each stream position."""
        out = np.empty(len(positions), dtype=object)
        r = self.rows[positions]
        for name, (off, n) in self.sources.items():
            m = (r >= off) & (r < off + n)
            out[m] = name
        return out


def _perm(n: int, seed: int) -> np.ndarray:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randperm(n, generator=g).numpy()


def _bank(sets: Sequence[FeatureSet]) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]], List[np.ndarray]]:
    feats, sources, labels = [], {}, []
    off = 0
    for fs in sets:
        if fs.name in sources:
            continue
        feats.append(fs.feats)
        labels.append(fs.labels)
        sources[fs.name] = (off, len(fs))
        off += len(fs)
    return torch.cat(feats, dim=0), sources, labels


def _segment_order(n_ood: int, n_id: int, seed: int, order: str) -> np.ndarray:
    """Indices into ConcatDataset([OOD, ID]) (OOD first, as in the official code)."""
    n = n_ood + n_id
    if order == "shuffled":
        return _perm(n, seed)
    ood = _perm(n_ood, seed)
    idp = n_ood + _perm(n_id, seed + 1)
    if order == "id_first":
        return np.concatenate([idp, ood])
    if order == "ood_first":
        return np.concatenate([ood, idp])
    raise ValueError(f"order must be one of {ORDERS}")


def pair_stream(id_set: FeatureSet, ood_set: FeatureSet, seed: int = 0, batch_size: int = 256,
                order: str = "shuffled") -> Stream:
    """One ID/OOD pair, memory reset at the start (the official protocol)."""
    bank, sources, _ = _bank([ood_set, id_set])
    o_off, n_ood = sources[ood_set.name]
    i_off, n_id = sources[id_set.name]
    concat_rows = np.concatenate([np.arange(o_off, o_off + n_ood), np.arange(i_off, i_off + n_id)])
    concat_labels = np.concatenate([np.full(n_ood, -1, dtype=np.int64), id_set.labels.astype(np.int64)])
    idx = _segment_order(n_ood, n_id, seed, order)
    return Stream(
        bank=bank, rows=concat_rows[idx], labels=concat_labels[idx],
        segments=[Segment(ood_set.name, 0, len(idx), reset=True)], batch_size=batch_size, sources=sources,
        meta={"kind": "pair", "id": id_set.name, "ood": [ood_set.name], "seed": seed, "order": order},
    )


def temporal_stream(id_set: FeatureSet, ood_sets: Sequence[FeatureSet], seed: int = 0, batch_size: int = 256,
                    id_mode: str = "full", reset_between: bool = False) -> Stream:
    """OOD sets one after another without memory reset (TANL Tab. A15).

    id_mode="full":  every segment is shuffle(OOD_k + all ID)   (matches the paper's per-dataset numbers)
    id_mode="split": ID is split into disjoint chunks, one per segment (no image is seen twice)
    reset_between=True turns this into the standard protocol (useful as a control).
    """
    bank, sources, _ = _bank([id_set, *ood_sets])
    i_off, n_id = sources[id_set.name]
    if id_mode == "split":
        chunks = np.array_split(_perm(n_id, seed + 7), len(ood_sets))
    elif id_mode == "full":
        chunks = [np.arange(n_id)] * len(ood_sets)
    else:
        raise ValueError("id_mode must be 'full' or 'split'")
    rows, labels, segments = [], [], []
    pos = 0
    for k, (ood, chunk) in enumerate(zip(ood_sets, chunks)):
        o_off, n_ood = sources[ood.name]
        seg_rows = np.concatenate([np.arange(o_off, o_off + n_ood), i_off + np.sort(chunk)])
        seg_labels = np.concatenate([np.full(n_ood, -1, dtype=np.int64), id_set.labels[np.sort(chunk)].astype(np.int64)])
        idx = _perm(len(seg_rows), seed * 1000 + k)
        rows.append(seg_rows[idx])
        labels.append(seg_labels[idx])
        segments.append(Segment(ood.name, pos, pos + len(idx), reset=(k == 0) or reset_between))
        pos += len(idx)
    return Stream(
        bank=bank, rows=np.concatenate(rows), labels=np.concatenate(labels), segments=segments,
        batch_size=batch_size, sources=sources,
        meta={"kind": "temporal", "id": id_set.name, "ood": [o.name for o in ood_sets], "seed": seed,
              "id_mode": id_mode, "reset_between": reset_between},
    )


TEMPORAL_ORDERS = {  # TANL Tab. A15 (I = iNaturalist, S = SUN, P = Places, T = Textures)
    "I-S-P-T": ["inaturalist", "sun", "places", "textures_all"],
    "S-P-T-I": ["sun", "places", "textures_all", "inaturalist"],
    "P-T-I-S": ["places", "textures_all", "inaturalist", "sun"],
    "T-I-S-P": ["textures_all", "inaturalist", "sun", "places"],
}
