#!/usr/bin/env python
"""Seam 1, probe P2: does a diffusion prior (text -> image embedding, ViT-L/14 space) beat the text embedding?

    python experiments/09_seam1_prior.py --p2-dir results/seam1/p2

Pre-registered (Oct 2, before any prior output was seen):
  Screen 1 (ViT-L/14 space): the prior prototype finds its own real concept (top-1 among the dataset's concepts,
           real centroid from the other half of the images) at least 10 points more often than the ViT-L/14 text
           embedding, on NINCO (64) AND on the 300 probed SSB-hard concepts.
  Screen 2 (map): the ViT-L/14 -> ViT-B/16 image map keeps mean cosine >= 0.9 on NINCO and SSB-hard images.
  Stream test (ViT-B/16, P1 protocol, 3 orders, w = 4): mapped prior prototypes cut NINCO FPR95 by >= 3 points
           against text-seeded evidence.
Result: FAIL. Inputs: notebook P2's reneg_l14_centroids.npz, reneg_l14_text.npz, reneg_l14_to_b16_map.npz and
reneg_prior_probe.npz, plus what experiments/08 needs.
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

from reneg.packs import dq8  # noqa: E402
from reneg.prior import apply_image_map, load_prior_embeddings  # noqa: E402
from replaylab.data import T  # noqa: E402
from replaylab.paths import RESULTS, out  # noqa: E402
from replaylab.seam1 import Seam1  # noqa: E402


def load_p2(S1: Seam1, d):
    c = np.load(os.path.join(d, "reneg_l14_centroids.npz"))
    t = np.load(os.path.join(d, "reneg_l14_text.npz"))
    mp = np.load(os.path.join(d, "reneg_l14_to_b16_map.npz"))
    W = torch.as_tensor(mp["W"]).double()
    checks = json.loads(str(mp["checks"]))
    probe, names, _ = load_prior_embeddings(os.path.join(d, "reneg_prior_probe.npz"))
    if list(names) != S1.g_names:
        raise ValueError("P2 used different probe names than P1")
    text = {k: dq8(t[f"{k}__q"], t[f"{k}__s"]) for k in ("id", "probe", "pool")}
    cents = {k: (list(c[f"{k}__folders"]), T(c[f"{k}__A"]), T(c[f"{k}__B"])) for k in ("ninco", "ssb_hard", "imagenet")}
    return cents, text, probe, W, checks


def screen1(S1: Seam1, cents, text, probe):
    res = {}
    for ds in ("ninco", "ssb_hard", "imagenet"):
        folders, A, B = cents[ds]
        fidx = {f: i for i, f in enumerate(folders)}
        p = np.array([i for i in range(S1.NP) if S1.g_tags[i] == ds])
        c = np.array([fidx[S1.g_folders[i]] for i in p])

        def top1(Q):
            return round(100 * float(((Q @ B.T).argmax(1).numpy() == c).mean()), 1)
        res[ds] = {"prior": top1(probe[torch.as_tensor(p)]), "text": top1(text["probe"][torch.as_tensor(p)]),
                   "realA": top1(A[torch.as_tensor(c)]),
                   "cos(prior,real)": round(float((probe[torch.as_tensor(p)] * B[torch.as_tensor(c)]).sum(1).mean()), 3)}
    res["pass"] = bool(all(res[d]["prior"] - res[d]["text"] >= 10 for d in ("ninco", "ssb_hard")))
    return res


def stream(S1: Seam1, W, probe, ws=(1, 4, 16), seeds=(0, 1, 2), log=print):
    mapped = apply_image_map(W, probe)
    rows = []
    for ds, sd in (("ninco", seeds), ("ssb_hard", (0,))):
        p, c = S1.probe_rows(ds)
        X_ood = S1.ood_half_b(ds, c)
        pool = S1.CT[ds]["text"][torch.as_tensor(c)]
        pr = mapped[torch.as_tensor(p)]
        base = [S1.online_init(X_ood, pool, seed=s) for s in sd]
        rows.append({"set": ds, "arm": "text-seeded evidence", "FPR95": round(float(np.mean([a for a, _ in base])), 2)})
        for w in (ws if ds == "ninco" else (4,)):
            r = [S1.online_init(X_ood, pool, init_proto=pr, w=w, seed=s) for s in sd]
            rows.append({"set": ds, "arm": f"mapped prior prototype (w={w})",
                         "FPR95": round(float(np.mean([a for a, _ in r])), 2)})
        allB = S1.CT[ds]["cB"]
        rows.append({"set": ds, "arm": "top-1 mapped prior -> real B/16 concept (%)",
                     "FPR95": round(100 * float(((pr @ allB.T).argmax(1).numpy() == c).mean()), 1)})
        log(rows[-3:])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p2-dir", default=os.path.join(RESULTS, "seam1", "p2"))
    ap.add_argument("--gen-probe", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    S1 = Seam1(a.gen_probe)
    cents, text, probe, W, checks = load_p2(S1, a.p2_dir)
    s1 = screen1(S1, cents, text, probe)
    print("screen 1 (ViT-L/14 space):", json.dumps(s1, indent=1))
    s2 = {"checks": checks, "pass": bool(all(checks[d] >= 0.9 for d in ("ninco", "ssb_hard")))}
    print("screen 2 (image map):", s2)
    rows = stream(S1, W, probe)
    base = [r for r in rows if r["set"] == "ninco" and r["arm"] == "text-seeded evidence"][0]["FPR95"]
    w4 = [r for r in rows if r["set"] == "ninco" and r["arm"] == "mapped prior prototype (w=4)"][0]["FPR95"]
    verdict = {"screen1": s1["pass"], "screen2": s2["pass"], "stream_gain_w4": round(float(base - w4), 2),
               "stream_pass": bool(base - w4 >= 3)}
    print("verdict:", verdict)
    path = out("prior_eval.json")
    json.dump({"screen1": s1, "screen2": s2, "stream": rows, "verdict": verdict}, open(path, "w"), indent=1)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
