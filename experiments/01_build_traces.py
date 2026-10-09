#!/usr/bin/env python
"""Run TANL once on every benchmark stream and store its trace (the input of every replay experiment).

    python experiments/01_build_traces.py                      # OpenOOD v1.5 + Four-OOD, stream orders 0, 1, 2
    python experiments/01_build_traces.py --seeds 0 --sets ninco textures

27 streams (5 OpenOOD v1.5 pairs on ImageNet test 45k, 4 Four-OOD pairs on ImageNet val 50k, 3 orders): 1-9 minutes
each on two CPU cores, about 120 MB each (fp16), ~3.5 GB in total. Existing traces are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import torch  # noqa: E402

from replaylab.data import FOUR, OOD_SETS  # noqa: E402
from replaylab.paths import TRACES  # noqa: E402
from replaylab.traces import build_traces  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="stream orders")
    ap.add_argument("--protocols", nargs="*", default=["openood_v15", "four_ood"], choices=["openood_v15", "four_ood"])
    ap.add_argument("--sets", nargs="*", help="only these OOD sets")
    ap.add_argument("--extend", choices=["full", "blind"], help="ablation: append the KG pool to TANL's corpus")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-images", type=int, help="quick mode: the first N images of each stream (tests only)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    print(f"traces -> {TRACES}")
    for seed in a.seeds:
        for prot in a.protocols:
            id_name, sets_ = ("imagenet_test", OOD_SETS) if prot == "openood_v15" else ("imagenet_val_all", FOUR)
            sets_ = [s for s in sets_ if not a.sets or s in a.sets]
            if sets_:
                build_traces(sets_, seed=seed, id_name=id_name, extend=a.extend, max_images=a.max_images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
