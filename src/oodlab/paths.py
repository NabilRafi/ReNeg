"""Where things live, on Kaggle or locally, and a *bounded* file search.

The previous notebook hung for up to an hour because a recursive glob followed
symlinks into ImageNet's 1.3M training files. ``find_file`` below never does
that: it has a depth limit, never follows symlinked directories, skips known
huge folders, and stops descending into any directory with too many entries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_WORKING = "/kaggle/working"

# Directory names we never descend into while searching.
SKIP_DIR_NAMES = {"train", "test", "Annotations", "__MACOSX", ".git", "ImageSets"}


def on_kaggle() -> bool:
    return os.path.isdir(KAGGLE_INPUT) and os.path.isdir(KAGGLE_WORKING)


@dataclass
class Dirs:
    work: str       # persistent outputs (Kaggle: /kaggle/working, 20 GB cap)
    scratch: str    # big temporary files (downloads, extracted images); not persisted
    inputs: str     # attached datasets / notebook outputs (read-only on Kaggle)

    @property
    def cache(self) -> str:
        return os.path.join(self.work, "cache")

    @property
    def logs(self) -> str:
        return os.path.join(self.work, "logs")

    @property
    def results(self) -> str:
        return os.path.join(self.work, "results")

    def make(self) -> "Dirs":
        for d in [self.work, self.scratch, self.cache, self.logs, self.results]:
            os.makedirs(d, exist_ok=True)
        return self


def default_dirs(work: Optional[str] = None, scratch: Optional[str] = None, inputs: Optional[str] = None) -> Dirs:
    if on_kaggle():
        work = work or KAGGLE_WORKING
        # Kaggle's documented temp folder is /kaggle/temp; /tmp always exists. Neither is saved.
        scratch = scratch or next((os.path.join(d, "oodlab_scratch") for d in ("/kaggle/temp", "/kaggle/tmp")
                                   if os.path.isdir(d)), "/tmp/oodlab_scratch")
        inputs = inputs or KAGGLE_INPUT
    else:
        base = os.environ.get("OODLAB_HOME", os.path.abspath("oodlab_work"))
        work = work or os.path.join(base, "working")
        scratch = scratch or os.path.join(base, "scratch")
        inputs = inputs or os.path.join(base, "input")
    return Dirs(work=work, scratch=scratch, inputs=inputs)


def find_file(
    name: str,
    roots: Sequence[str],
    max_depth: int = 4,
    max_entries: int = 5000,
    skip_names: Iterable[str] = SKIP_DIR_NAMES,
) -> Optional[str]:
    """Breadth-first search for an exact filename under ``roots``.

    Directories are not followed if they are symlinks, are named in
    ``skip_names``, or contain more than ``max_entries`` entries.
    """
    skip = set(skip_names)
    frontier: List[tuple] = [(r, 0) for r in roots if os.path.isdir(r)]
    while frontier:
        d, depth = frontier.pop(0)
        try:
            with os.scandir(d) as it:
                entries = []
                for i, e in enumerate(it):
                    if i >= max_entries:
                        entries = None
                        break
                    entries.append(e)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        if entries is None:
            continue  # too big to be a place where a metadata file lives
        for e in entries:
            if e.name == name and e.is_file():
                return e.path
        if depth < max_depth:
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False) and e.name not in skip:
                        frontier.append((e.path, depth + 1))
                except OSError:
                    continue
    return None


def find_dir_containing(marker_rel: str, roots: Sequence[str], max_depth: int = 4) -> Optional[str]:
    """Find a directory ``D`` such that ``D/marker_rel`` exists (bounded search)."""
    frontier = [(r, 0) for r in roots if os.path.isdir(r)]
    while frontier:
        d, depth = frontier.pop(0)
        if os.path.exists(os.path.join(d, marker_rel)):
            return d
        if depth >= max_depth:
            continue
        try:
            with os.scandir(d) as it:
                subs = []
                for i, e in enumerate(it):
                    if i >= 5000:
                        subs = []
                        break
                    if e.is_dir(follow_symlinks=False) and e.name not in SKIP_DIR_NAMES:
                        subs.append(e.path)
        except (PermissionError, FileNotFoundError):
            continue
        frontier.extend((s, depth + 1) for s in subs)
    return None


def list_images(folder: str, exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")) -> List[str]:
    """All image files under ``folder`` (does not follow symlinks), sorted for determinism.

    Only call this on a *dataset* folder you extracted yourself (thousands of files),
    never on an ImageNet train directory.
    """
    exts = tuple(e.lower() for e in exts)
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in {"__MACOSX"} and not d.startswith("."))
        for fn in filenames:
            if fn.startswith("."):
                continue
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def rel(path: str, root: str) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path
