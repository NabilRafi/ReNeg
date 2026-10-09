#!/usr/bin/env python
"""Pack the code and notebooks for Google Colab: dist/ReNeg_Colab_bundle.zip (docs/COLAB_GUIDE.md, step 2).

    python tools/make_colab_bundle.py

Layout expected by the set-up cell of every Colab notebook (src/reneg/colab.py)::

    ReNeg_Colab_bundle/reneg_code/reneg/     the reneg package (src/reneg)
    ReNeg_Colab_bundle/reneg_code/tests/     its unit tests (tests/reneg; notebook C1 runs them)
    ReNeg_Colab_bundle/notebooks/*.ipynb     C0, C1, G1, G1b, G2a, G2b, P1, P2, G3, G3b, P4, D1, G4
    ReNeg_Colab_bundle/GUIDE_COLAB.md

Upload it to MyDrive/ReNeg/uploads/. The oodlab code goes to Drive separately (tools/make_kaggle_dataset.py).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def add_tree(z: zipfile.ZipFile, src: str, arc: str) -> int:
    n = 0
    for dp, dn, fn in os.walk(src):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            if f.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(dp, f)
            z.write(p, os.path.join(arc, os.path.relpath(p, src)))
            n += 1
    return n


def colab_notebooks():
    return sorted(p for p in glob.glob(os.path.join(ROOT, "notebooks", "*", "*.ipynb"))
                  if not os.path.basename(os.path.dirname(p)).startswith("0_"))


def build_bundle(out: str, prefix: str = "ReNeg_Colab_bundle") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        n = add_tree(z, os.path.join(ROOT, "src", "reneg"), f"{prefix}/reneg_code/reneg")
        n += add_tree(z, os.path.join(ROOT, "tests", "reneg"), f"{prefix}/reneg_code/tests")
        for p in colab_notebooks():
            z.write(p, f"{prefix}/notebooks/{os.path.basename(p)}")
            n += 1
        z.write(os.path.join(ROOT, "docs", "COLAB_GUIDE.md"), f"{prefix}/GUIDE_COLAB.md")
        n += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "ReNeg_Colab_bundle.zip"))
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import reneg

    path = build_bundle(a.out)
    with zipfile.ZipFile(path) as z:
        n = len(z.namelist())
    print(f"wrote {path} (reneg {reneg.__version__}, {n} files, {os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
