#!/usr/bin/env python
"""How much would a SOLVED Seam 1 add on top of ReNeg-on-TANL? (an upper-bound analysis, not a method)

    python experiments/11_solved_seam1_ceiling.py               # NINCO and SSB-hard, order 0 -> oracle_prior.jsonl

The blind KG pool plus the benchmark's own concept names (NINCO 64, SSB-hard 980) appended, each concept name given
an image-space prior = mean of the REAL images of half A of that concept (a perfect text-to-image translation), as
a pseudo-count w on its visual prototype. Scored on half B of the probed concepts (images the prior never saw)
against all ID images, on the full stream (TANL trace of order 0). Arms: blind pool | + concept names, no prior |
+ oracle real prior (w) | + text-embedding prior | + generated prior (P1 SDXL-Turbo, probed concepts only).
Finding (Oct 2): even a perfect bridge adds little over the stream-grounded prototypes ReNeg already builds.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import reneg.core as RC  # noqa: E402
from reneg.transport import l2n  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.paths import out  # noqa: E402
from replaylab.replay import MODES, append  # noqa: E402
from replaylab.seam1 import Seam1, replay_prior  # noqa: E402
from replaylab.traces import load_trace, stream_for  # noqa: E402


def run(S1: Seam1, ds, path, mode="balanced", seed=0, ws=(4.0,), log=print):
    t, d = S1.CT[ds], S1.S[ds]
    tr = load_trace(ds, seed, "imagenet_test")
    X, is_ood, p = stream_for(ds, seed, "imagenet_test", tr)
    n_ood = len(d["X"])
    ood_row = np.where(p < n_ood, p, -1)                   # stream position -> OOD row (or -1)
    pr, pc = S1.probe_rows(ds)                             # probed concepts (generated images exist for these)
    probed = np.zeros(len(t["folders"]), bool)
    probed[pc] = True
    evalmask = np.ones(len(X), bool)
    o = ood_row >= 0
    evalmask[o] = (~t["A"][ood_row[o]]) & probed[d["folder"][ood_row[o]]]
    cfg = RC.ReNegConfig(**MODES[mode])
    P0, cl0 = D.pool_subset("blind")
    nC = len(t["folders"])
    Pc = l2n(t["text"].float())
    P1 = torch.cat([P0, Pc])
    cl1 = np.r_[cl0, cl0.max() + 1 + np.arange(nC)]
    pidx = len(cl0) + np.arange(nC)
    gen = torch.zeros(nC, S1.dim)
    gen[torch.as_tensor(pc)] = S1.g_proto[torch.as_tensor(pr)].float()
    arms = [("blind pool", P0, cl0, None, None), ("+ concept names, no prior", P1, cl1, None, None)]
    for w in ws:
        arms += [(f"+ oracle real prior (half A), w={w:g}", P1, cl1, pidx, t["cA"].float()),
                 (f"+ text-embedding prior, w={w:g}", P1, cl1, pidx, Pc),
                 (f"+ generated prior (probed only), w={w:g}", P1, cl1, pidx[pc], gen[torch.as_tensor(pc)])]
    res = []
    for name, P, cl, pi_, pv in arms:
        w = float(name.split("w=")[1]) if "w=" in name else 0.0
        s = replay_prior(tr, X, P, cl, cfg, pi_, pv, w=w, id_text=S1.id_text)
        m = D.metrics(s[evalmask & ~is_ood], s[evalmask & is_ood])
        res.append({"set": ds, "mode": mode, "arm": name, "fpr95": round(m["fpr95"], 2), "auroc": round(m["auroc"], 2)})
        append(path, res[-1])
        log(res[-1])
    tm = D.metrics(tr["tanl"][evalmask & ~is_ood], tr["tanl"][evalmask & is_ood])
    res.append({"set": ds, "mode": mode, "arm": "TANL", "fpr95": round(tm["fpr95"], 2), "auroc": round(tm["auroc"], 2)})
    append(path, res[-1])
    log(res[-1])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="*", default=["ninco", "ssb_hard"])
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    S1 = Seam1()
    path = out("oracle_prior.jsonl")
    for ds in a.sets:
        run(S1, ds, path, "balanced", ws=(4.0, 16.0))
        run(S1, ds, path, "max", ws=(4.0,))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
