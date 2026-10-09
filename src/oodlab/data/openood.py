"""OpenOOD v1.5 benchmark data: Google-Drive downloads, image lists and path resolution.

Fixes relative to ``scripts/download/download.py`` in OpenOOD-VLM:
* we choose the output filename ourselves (the official script assumed Drive would
  name the file ``<dataset>.zip`` and crashed when it did not);
* downloads are retried with back-off and validated as real zip files (Drive's
  "quota exceeded" page is HTML, which used to be unzipped and fail later);
* we can also take zips or extracted folders from an attached Kaggle dataset, so
  Drive is needed only once.
"""
from __future__ import annotations

import os
import shutil
import time
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..paths import find_dir_containing, list_images
from .manifest import Manifest

# Google Drive ids copied from OpenOOD-VLM/scripts/download/download.py (same as OpenOOD v1.5).
DRIVE_IDS: Dict[str, str] = {
    "benchmark_imglist": "1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP",
    "inaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "sun": "1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp",
    "places": "1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3",
    "texture": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
    "ssb_hard": "1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE",
    "ninco": "1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6",
    "openimage_o": "1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE",
    "imagenet_v2": "1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho",
    "imagenet_r": "1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU",
    "imagenet_c": "1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt",
}


class DownloadError(RuntimeError):
    pass


def drive_url(key: str) -> str:
    return f"https://drive.google.com/uc?id={DRIVE_IDS[key]}"


def gdown_zip(key: str, out_dir: str, retries: int = 4, wait_s: int = 30, logger=None) -> str:
    """Download ``<key>.zip`` from Google Drive into ``out_dir`` and check it is a zip."""
    import gdown

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{key}.zip")
    if os.path.isfile(out) and zipfile.is_zipfile(out):
        return out
    last = None
    for attempt in range(1, retries + 1):
        try:
            if logger:
                logger.info(f"[download] {key}: attempt {attempt}/{retries} from {drive_url(key)}")
            gdown.download(id=DRIVE_IDS[key], output=out, quiet=False)
            if os.path.isfile(out) and zipfile.is_zipfile(out):
                return out
            last = "downloaded file is not a zip (Drive probably served a quota/virus-scan page)"
        except Exception as e:  # gdown raises several exception types
            last = f"{type(e).__name__}: {e}"
        if os.path.exists(out) and not zipfile.is_zipfile(out):
            os.remove(out)
        if attempt < retries:
            time.sleep(wait_s * attempt)
    raise DownloadError(
        f"Could not download {key} from Google Drive ({last}).\n"
        f"Fix: open {drive_url(key)} in your browser, download {key}.zip, upload it to a private "
        f"Kaggle dataset (e.g. 'openood-raw-zips'), attach it, and re-run - the code picks zips up from inputs."
    )


def extract_zip(zip_path: str, dest: str, logger=None) -> str:
    os.makedirs(dest, exist_ok=True)
    marker = os.path.join(dest, ".extracted_ok")
    if os.path.exists(marker):
        return dest
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if logger:
            logger.info(f"[extract] {os.path.basename(zip_path)}: {len(members)} entries -> {dest}")
        zf.extractall(dest)
    open(marker, "w").close()
    return dest


def find_zip_in_inputs(key: str, input_roots: Sequence[str], max_depth: int = 3) -> Optional[str]:
    """A user-uploaded ``<key>.zip`` in an attached dataset (bounded search)."""
    from ..paths import find_file

    return find_file(f"{key}.zip", input_roots, max_depth=max_depth)


def obtain_dataset_dir(key: str, scratch: str, input_roots: Sequence[str] = (), logger=None, allow_download: bool = True) -> str:
    """Directory containing the extracted images for OpenOOD dataset ``key``.

    Order: (1) already extracted in scratch, (2) extracted folder in an attached input
    (a folder named ``key``), (3) ``key.zip`` in an attached input, (4) Google Drive.
    """
    dest = os.path.join(scratch, "extracted", key)
    if os.path.exists(os.path.join(dest, ".extracted_ok")):
        return dest
    for root in input_roots:
        d = find_dir_containing(key, [root], max_depth=2)
        if d is not None and os.path.isdir(os.path.join(d, key)) and list_images_quick(os.path.join(d, key)):
            if logger:
                logger.info(f"[data] using pre-extracted {key} from {os.path.join(d, key)}")
            return os.path.join(d, key)
    z = find_zip_in_inputs(key, input_roots)
    if z is None:
        if not allow_download:
            raise DownloadError(f"{key}: no zip in inputs and downloads are disabled")
        z = gdown_zip(key, os.path.join(scratch, "zips"), logger=logger)
    return extract_zip(z, dest, logger=logger)


def list_images_quick(d: str, limit: int = 3) -> bool:
    n = 0
    for _, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                n += 1
                if n >= limit:
                    return True
    return n > 0


# ---------------------------------------------------------------------------
# image lists
# ---------------------------------------------------------------------------


def obtain_imglist_dir(scratch: str, input_roots: Sequence[str] = (), logger=None, allow_download: bool = True) -> str:
    """Folder that contains OpenOOD's ``benchmark_imglist/imagenet/*.txt``."""
    for root in list(input_roots) + [os.path.join(scratch, "extracted")]:
        d = find_dir_containing(os.path.join("imagenet", "test_imagenet.txt"), [root], max_depth=3)
        if d:
            return os.path.join(d, "imagenet")
    z = find_zip_in_inputs("benchmark_imglist", input_roots)
    if z is None:
        if not allow_download:
            raise DownloadError("benchmark_imglist: not found in inputs and downloads are disabled")
        z = gdown_zip("benchmark_imglist", os.path.join(scratch, "zips"), logger=logger)
    dest = extract_zip(z, os.path.join(scratch, "extracted", "benchmark_imglist"), logger=logger)
    d = find_dir_containing(os.path.join("imagenet", "test_imagenet.txt"), [dest], max_depth=3)
    if d is None:
        raise FileNotFoundError("benchmark_imglist.zip has no imagenet/test_imagenet.txt")
    return os.path.join(d, "imagenet")


def read_imglist(path: str) -> List[Tuple[str, int]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel, _, extra = line.partition(" ")
            try:
                lab = int(extra.strip()) if extra.strip() else -1
            except ValueError:
                lab = -1  # dict-style extras are not used for ImageNet OOD lists
            out.append((rel, lab))
    return out


def resolve_list_paths(entries: List[Tuple[str, int]], dataset_dir: str) -> Tuple[List[str], List[str]]:
    """Map list entries (``ssb_hard/n.../x.JPEG``) to files under ``dataset_dir``.

    Tries the natural joins first, then falls back to a basename index of
    ``dataset_dir`` (disambiguated by the longest matching path suffix).
    Returns (paths, missing_entries).
    """
    parent = os.path.dirname(dataset_dir.rstrip("/"))
    index: Optional[Dict[str, List[str]]] = None
    paths, missing = [], []
    for rel, _ in entries:
        parts = rel.split("/")
        cands = [os.path.join(dataset_dir, rel), os.path.join(parent, rel)]
        if len(parts) > 1:
            cands.append(os.path.join(dataset_dir, *parts[1:]))
        hit = next((c for c in cands if os.path.isfile(c)), None)
        if hit is None:
            if index is None:
                index = defaultdict(list)
                for p in list_images(dataset_dir):
                    index[os.path.basename(p)].append(p)
            opts = index.get(parts[-1], [])
            if len(opts) == 1:
                hit = opts[0]
            elif len(opts) > 1:
                def suffix_len(p):
                    a, b = p.split("/")[::-1], parts[::-1]
                    n = 0
                    while n < min(len(a), len(b)) and a[n] == b[n]:
                        n += 1
                    return n
                hit = max(opts, key=suffix_len)
        if hit is None:
            missing.append(rel)
        else:
            paths.append(hit)
    return paths, missing


def ood_manifest_from_list(name: str, entries: List[Tuple[str, int]], dataset_dir: str, list_file: str) -> Manifest:
    paths, missing = resolve_list_paths(entries, dataset_dir)
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} of {len(entries)} listed images not found, e.g. {missing[:3]}")
    keys = [rel for rel, _ in entries]
    return Manifest(name=name, paths=paths, labels=np.full(len(paths), -1), keys=keys,
                    meta={"source": "OpenOOD list", "list_file": os.path.basename(list_file)})


def ood_manifest_from_folder(name: str, dataset_dir: str) -> Manifest:
    paths = list_images(dataset_dir)
    keys = [os.path.relpath(p, dataset_dir) for p in paths]
    return Manifest(name=name, paths=paths, labels=np.full(len(paths), -1), keys=keys,
                    meta={"source": "all images in folder"})


def cleanup(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        os.remove(path)
