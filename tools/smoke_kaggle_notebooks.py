#!/usr/bin/env python
"""Execute the Kaggle notebooks 00-04 end to end in smoke mode (fake data, random tiny CLIP, CPU, ~5 minutes).

    python tools/smoke_kaggle_notebooks.py [work_dir]

This is how the notebooks are tested without Kaggle. Executed copies are written to <work_dir>/executed/ so
failures can be inspected. Needs nbclient and ipykernel (pip install -e ".[test]").
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB = os.path.join(ROOT, "notebooks", "0_baselines_kaggle")


def main(work: str) -> int:
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "executed"))
    os.environ.update({"OODLAB_SMOKE": "1", "OODLAB_HOME": os.path.join(work, "home"),
                       "OODLAB_CODE": os.path.join(ROOT, "src"), "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    failures = 0
    for fn in sorted(f for f in os.listdir(NB) if f.endswith(".ipynb")):
        nb = nbformat.read(os.path.join(NB, fn), as_version=4)
        t0 = time.time()
        client = NotebookClient(nb, kernel_name="python3", timeout=1800, resources={"metadata": {"path": work}})
        try:
            client.execute()
            status = "OK"
        except CellExecutionError as e:
            status = "FAILED"
            failures += 1
            print(str(e)[-3000:])
        nbformat.write(nb, os.path.join(work, "executed", fn))
        print(f"{fn:34s} {status:7s} {time.time() - t0:6.1f}s", flush=True)
    return failures


if __name__ == "__main__":
    work = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "smoke_kaggle")
    sys.exit(1 if main(work) else 0)
