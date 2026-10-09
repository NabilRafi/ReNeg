#!/usr/bin/env python
"""Seam 1, probe P1: do generated images (SDXL-Turbo) give near-OOD names a usable image prototype?

    python experiments/08_seam1_generation.py                # quality screen + stream test -> planb_results.json
    python experiments/08_seam1_generation.py --gap          # plus two generation-gap corrections (planb2)

Pre-registered rule (Oct 1): generated prototypes must find their own real concept (top-1 among the dataset's
concepts) at least as often as the text embedding, and cut NINCO FPR95 by >= 3 points against text-seeded
evidence in the stream test. Result: FAIL (generated 59% / 44% top-1 on NINCO / SSB-hard vs text 78% / 60%; no
stream gain), so ReNeg grounds names in the test stream instead. Needs the G1b pack (export_features --what pack)
and results/seam1/reneg_gen_probe.npz (notebook P1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from reneg.transport import l2n  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.paths import out  # noqa: E402
from replaylab.seam1 import Seam1  # noqa: E402


def gap_corrections(S1: Seam1, log=print):
    """Shift every generated prototype by the mean generated-to-real gap (no OOD data, no test labels): 'mean' uses
    the mean of all generated images vs the mean image; 'pair' the paired shift on the ImageNet probe classes."""
    mean_gen = l2n(S1.g_emb.mean(0, keepdim=True))[0]
    stats = D.stats
    dirs = l2n(stats.directions().float())
    p_in = np.array([i for i in range(S1.NP) if S1.g_tags[i] == "imagenet"])
    cls = np.array([int(S1.g_folders[i]) for i in p_in])
    ok = stats.counts.numpy()[cls] >= 3
    delta_pair = (dirs[torch.as_tensor(cls[ok])] - S1.g_proto[torch.as_tensor(p_in[ok])]).mean(0)
    delta_mean = l2n(D.m_img)[None][0] - mean_gen
    log(f"classes used for the paired shift {int(ok.sum())} | norm delta_pair {float(delta_pair.norm()):.3f} "
        f"delta_mean {float(delta_mean.norm()):.3f}")
    rows = []
    for kind, d in (("mean", delta_mean), ("pair", delta_pair)):
        gp_all = l2n(S1.g_proto + d)
        for ds, ws, seeds in (("ninco", (1, 4, 16), (0, 1, 2)), ("ssb_hard", (4,), (0,))):
            p, c = S1.probe_rows(ds)
            allB = S1.CT[ds]["cB"]
            gp = gp_all[torch.as_tensor(p)]
            top1 = 100 * float(((gp @ allB.T).argmax(1).numpy() == c).mean())
            X_ood = S1.ood_half_b(ds, c)
            pool = S1.CT[ds]["text"][torch.as_tensor(c)]
            for w in ws:
                r = [S1.online_init(X_ood, pool, init_proto=gp, w=w, seed=s) for s in seeds]
                rows.append({"shift": kind, "set": ds, "w": w, "top1": round(top1, 1),
                             "FPR95": round(float(np.mean([a for a, _ in r])), 2),
                             "AUROC": round(float(np.mean([b for _, b in r])), 2)})
                log(rows[-1])
            f, au = S1.online_init(X_ood, pool, init_proto=gp, w=4, update=False)
            rows.append({"shift": kind, "set": ds, "w": "4, static", "top1": round(top1, 1), "FPR95": f, "AUROC": au})
            log(rows[-1])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gap", action="store_true", help="also test the generation-gap corrections")
    ap.add_argument("--gen-probe", default=None, help="P1's reneg_gen_probe.npz")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    S1 = Seam1(a.gen_probe)
    q = S1.quality()
    print(json.dumps(q, indent=1))
    rows = S1.stream_test("ninco") + S1.stream_test("ssb_hard", ws=(4,), seeds=(0,))
    res = {"quality": q, "stream": rows}
    if a.gap:
        res["gap_corrections"] = gap_corrections(S1)
    path = out("planb_results.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
