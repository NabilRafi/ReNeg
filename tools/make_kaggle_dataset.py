#!/usr/bin/env python
"""Pack oodlab for Kaggle (and for Colab notebook C0): dist/oodlab_code.zip.

    python tools/make_kaggle_dataset.py

Upload the zip as a private Kaggle dataset named ``oodlab-code`` and attach it to notebooks 00-04
(docs/KAGGLE_GUIDE.md); the notebooks find ``oodlab/__init__.py`` inside it. Layout::

    oodlab_code/oodlab/       the package (src/oodlab)
    oodlab_code/notebooks/    Kaggle notebooks 00-04
    oodlab_code/tests/        its unit tests (tests/oodlab)
    oodlab_code/licenses/     OpenOOD-VLM (MIT), OpenAI CLIP (MIT), NegLabel word lists (Apache-2.0)
    oodlab_code/GUIDE.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_colab_bundle import add_tree  # noqa: E402


def build_dataset(out: str, prefix: str = "oodlab_code") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, os.path.join(ROOT, "src", "oodlab"), f"{prefix}/oodlab")
        add_tree(z, os.path.join(ROOT, "tests", "oodlab"), f"{prefix}/tests")
        add_tree(z, os.path.join(ROOT, "licenses"), f"{prefix}/licenses")
        for p in sorted(glob.glob(os.path.join(ROOT, "notebooks", "0_baselines_kaggle", "*.ipynb"))):
            z.write(p, f"{prefix}/notebooks/{os.path.basename(p)}")
        z.write(os.path.join(ROOT, "docs", "KAGGLE_GUIDE.md"), f"{prefix}/GUIDE.md")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "oodlab_code.zip"))
    a = ap.parse_args(argv)
    path = build_dataset(a.out)
    with zipfile.ZipFile(path) as z:
        n = len(z.namelist())
    print(f"wrote {path} ({n} files, {os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
