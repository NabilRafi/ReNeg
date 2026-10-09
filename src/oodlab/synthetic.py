"""Synthetic CLIP-like features for unit tests and smoke runs (no GPU, no downloads).

ID images sit near one of C "ID label" text vectors; OOD images sit near a hidden
subset of "corpus" text vectors, so negative-label methods have something real to
find. Everything is L2-normalised like CLIP features.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


@dataclass
class SyntheticBank:
    id_text: torch.Tensor      # (C, D)
    corpus: torch.Tensor       # (N, D) - first n_neglabel play the NegLabel negatives
    noise: torch.Tensor        # (15, D)
    ood_concepts: torch.Tensor # indices into corpus that generate OOD images
    n_neglabel: int


def make_bank(C: int = 50, N: int = 3000, D: int = 256, n_neglabel: int = 500, n_ood_concepts: int = 60, seed: int = 0) -> SyntheticBank:
    g = torch.Generator().manual_seed(seed)
    gap = _unit(torch.randn(D, generator=g))  # shared "modality gap" direction for text
    id_text = _unit(_unit(torch.randn(C, D, generator=g)) + 0.3 * gap)
    corpus = _unit(_unit(torch.randn(N, D, generator=g)) + 0.3 * gap)
    noise = _unit(torch.randn(15, D, generator=g))
    ood_concepts = torch.randperm(N, generator=g)[:n_ood_concepts]
    return SyntheticBank(id_text, corpus, noise, ood_concepts, n_neglabel)


def make_images(bank: SyntheticBank, n: int, kind: str = "id", signal: float = 0.35, seed: int = 1):
    """Returns (features (n, D), labels (n,)); OOD labels are -1."""
    g = torch.Generator().manual_seed(seed)
    D = bank.id_text.shape[1]
    if kind == "id":
        lab = torch.randint(0, bank.id_text.shape[0], (n,), generator=g)
        base = bank.id_text[lab]
    else:
        pick = bank.ood_concepts[torch.randint(0, len(bank.ood_concepts), (n,), generator=g)]
        base = bank.corpus[pick]
        lab = torch.full((n,), -1, dtype=torch.long)
    f = _unit(signal * base + _unit(torch.randn(n, D, generator=g)))
    return f, lab
