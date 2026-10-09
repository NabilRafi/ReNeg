"""Fi-ImageNet-1k (Rozumnyi et al., 2026): near-OOD images *inside* the ImageNet-1K val set.

The data is not public yet (release on acceptance). Once you have the image lists,
no new encoding is needed: the images are ImageNet validation images, which are
already in the cache as ``imagenet_val_all``. Put two text files anywhere in an
attached dataset / the working directory:

    fi_imagenet_1k_ood.txt         the 655 OOD images
    fi_imagenet_1k_ambiguous.txt   the 980 ambiguous images (removed from ID, as the authors do)

Any line containing ``ILSVRC2012_val_XXXXXXXX`` is understood (file names, paths or CSV rows).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..cache import FeatureSet
from .imagenet import VAL_ID_RE

OOD_LIST = "fi_imagenet_1k_ood.txt"
AMBIGUOUS_LIST = "fi_imagenet_1k_ambiguous.txt"
EXPECTED_OOD = 655
EXPECTED_AMBIGUOUS = 980


def read_val_ids(path: str) -> List[str]:
    out = []
    with open(path) as f:
        for line in f:
            m = VAL_ID_RE.search(line)
            if m:
                out.append(m.group(1))
    return list(dict.fromkeys(out))  # de-duplicate, keep order


def find_lists(roots: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    from ..paths import find_file

    return find_file(OOD_LIST, roots, max_depth=4), find_file(AMBIGUOUS_LIST, roots, max_depth=4)


def fi_imagenet_sets(imagenet_all: FeatureSet, ood_ids: Sequence[str], ambiguous_ids: Sequence[str] = (),
                     logger=None) -> Tuple[FeatureSet, FeatureSet]:
    """(ID set without Fi-ImageNet-1k's OOD + ambiguous images, Fi-ImageNet-1k OOD set)."""
    ood = imagenet_all.select_keys(list(ood_ids), name="fi_imagenet_1k", label=-1)
    id_clean = imagenet_all.exclude_keys(list(ood_ids) + list(ambiguous_ids), name="imagenet_val_all_minus_fi")
    msg = (f"[fi-imagenet-1k] OOD {len(ood)} (expected {EXPECTED_OOD}), ambiguous removed {len(ambiguous_ids)} "
           f"(expected {EXPECTED_AMBIGUOUS}), ID left {len(id_clean)}")
    (logger.info if logger else print)(msg)
    return id_clean, ood
