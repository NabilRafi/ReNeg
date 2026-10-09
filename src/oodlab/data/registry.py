"""Which evaluation sets exist, where their images come from, and how many to expect."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .imagenet import imagenet_val_manifest, locate_imagenet, restrict_to_openood_list
from .manifest import Manifest
from .openood import obtain_dataset_dir, ood_manifest_from_folder, ood_manifest_from_list, read_imglist


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    role: str                     # "id" or "ood"
    drive_key: Optional[str]      # OpenOOD Google-Drive archive holding the images (None = Kaggle ImageNet)
    imglist: Optional[str]        # OpenOOD list file (benchmark_imglist/imagenet/<file>); None = all images
    expected: Optional[int]       # expected number of images (a warning, not an error, if different)
    note: str = ""


SPECS: Dict[str, DatasetSpec] = {s.name: s for s in [
    # ---- in-distribution: ImageNet-1K validation (Kaggle competition data) ----
    DatasetSpec("imagenet_val_all", "id", None, None, 50000, "all 50k; ID set of the Four-OOD protocol (test_imagenet_all)"),
    DatasetSpec("imagenet_test", "id", None, "test_imagenet.txt", 45000, "OpenOOD v1.5 ID test split"),
    DatasetSpec("imagenet_val", "id", None, "val_imagenet.txt", 5000, "OpenOOD v1.5 ID validation split (tuning only)"),
    # ---- Four-OOD (MOS subsets) ----
    DatasetSpec("inaturalist", "ood", "inaturalist", "test_inaturalist.txt", 10000, "Four-OOD and OpenOOD far-OOD"),
    DatasetSpec("sun", "ood", "sun", None, 10000, "Four-OOD; OpenOOD ships no list -> all images"),
    DatasetSpec("places", "ood", "places", None, 10000, "Four-OOD; OpenOOD ships no list -> all images"),
    DatasetSpec("textures_all", "ood", "texture", None, 5640, "Four-OOD 'dtd': all DTD images"),
    # ---- OpenOOD v1.5 ----
    DatasetSpec("textures", "ood", "texture", "test_textures.txt", None, "OpenOOD v1.5 far-OOD Textures"),
    DatasetSpec("openimage_o", "ood", "openimage_o", "test_openimage_o.txt", 15869, "OpenOOD v1.5 far-OOD test split"),
    DatasetSpec("openimage_o_val", "ood", "openimage_o", "val_openimage_o.txt", 1763, "OpenOOD v1.5 OOD validation (tuning only)"),
    DatasetSpec("ssb_hard", "ood", "ssb_hard", "test_ssb_hard.txt", 49000, "OpenOOD v1.5 near-OOD"),
    DatasetSpec("ninco", "ood", "ninco", "test_ninco.txt", 5879, "OpenOOD v1.5 near-OOD"),
]}

# Default build order: small first, so problems surface early.
DEFAULT_BUILD = [
    "imagenet_val_all", "imagenet_test", "imagenet_val",
    "ninco", "textures_all", "textures", "inaturalist", "sun", "places",
    "openimage_o", "openimage_o_val", "ssb_hard",
]


def build_manifest(
    name: str,
    input_roots: Sequence[str],
    scratch: str,
    imglist_dir: Optional[str],
    logger=None,
    allow_download: bool = True,
    strict_counts: bool = True,
    _imagenet_cache: Dict[str, Manifest] = {},
) -> Manifest:
    """Create the manifest for ``name`` (downloading/extracting its images if needed).

    ``strict_counts=False`` is only for smoke tests with a miniature fake ImageNet.
    """
    spec = SPECS[name]
    if spec.role == "id":
        if "full" not in _imagenet_cache:
            loc = locate_imagenet(input_roots)
            if logger:
                logger.info(f"[imagenet] {loc.base} ({loc.layout} layout)")
            _imagenet_cache["full"] = imagenet_val_manifest(loc, expected=50000 if strict_counts else None)
        full = _imagenet_cache["full"]
        if spec.imglist is None:
            m = full.subset(range(len(full)), name=name)
        else:
            if imglist_dir is None:
                raise ValueError(f"{name} needs the OpenOOD image lists")
            m = restrict_to_openood_list(full, read_imglist(os.path.join(imglist_dir, spec.imglist)), name)
    else:
        list_path = os.path.join(imglist_dir, spec.imglist) if (spec.imglist and imglist_dir) else None
        if spec.imglist and not (list_path and os.path.isfile(list_path)):
            # never silently fall back to "all images": e.g. Textures and OpenImage-O are list subsets
            raise FileNotFoundError(f"{name} needs the OpenOOD list {spec.imglist} (image lists not available)")
        ddir = obtain_dataset_dir(spec.drive_key, scratch, input_roots, logger=logger, allow_download=allow_download)
        if list_path:
            m = ood_manifest_from_list(name, read_imglist(list_path), ddir, list_path)
        else:
            m = ood_manifest_from_folder(name, ddir)
    m.meta.update({"spec_note": spec.note, "expected": spec.expected})
    if spec.expected is not None and len(m) != spec.expected and logger:
        logger.warning(f"[{name}] {len(m)} images (expected {spec.expected}) - check the source before trusting results")
    return m


def drive_keys_for(names: List[str]) -> List[str]:
    return sorted({SPECS[n].drive_key for n in names if SPECS[n].drive_key})
