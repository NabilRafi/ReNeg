#!/usr/bin/env python
"""Seam 1 re-analysis: six more training-free bridges between a near-OOD name and its images (none beats text).

    python experiments/10_seam1_bridges.py            # retrieval screen -> seam1_rethink.json

Screen (as P1 / P2): a prototype for name j should find j's REAL concept (centroid of half B of the dataset's
images) among all concepts of the dataset. Reference (Oct 3): text NINCO 78%, SSB-hard 60%; generated 59% / 44%;
real half A 100% / 94%. Bridges (none uses the test labels):
  gen rc       domain re-centring: generated - mean(all generated) + mean(real stream images)
  gen filt     CLIP-consistency filter: keep generated images whose nearest text over [ID ; pool ; probe names] is j
  text+gen rc  text plus the re-centred generated prototype (both unit vectors)
  rel          relative representation: cosines to 1,000 anchors (ID text for text, ID image centroids for images)
  sig          TIP-X-style signature: softmax over [ID ; pool] text, compared by KL
  text gap     text shifted by the mean text-to-image gap (G1's R1, here only for the retrieval screen)
"""
from __future__ import annotations

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


def zrow(M):
    return (M - M.mean(1, keepdim=True)) / M.std(1, keepdim=True)


class Bridges:
    def __init__(self, S1: Seam1):
        self.S1 = S1
        id_text = S1.id_text.float()
        self.mu_gen = l2n(S1.g_emb.float().mean(0, keepdim=True))[0]
        # real image mean: the stream the method would see (ImageNet test + both OOD sets)
        self.mu_real = l2n(torch.cat([S1.Xid, S1.S["ssb_hard"]["X"], S1.S["ninco"]["X"]]).float().mean(0, keepdim=True))[0]
        self.mu_text = l2n(id_text.mean(0, keepdim=True))[0]
        yA = S1.yid[0::2]                    # ID image anchors: class centroids of half A (what the ID model builds)
        self.anch_img = l2n(torch.stack([S1.XidA[torch.as_tensor(np.nonzero(yA == k)[0])].mean(0)
                                         for k in range(id_text.shape[0])]))
        self.anch_txt = l2n(id_text)
        self.ALLT = l2n(torch.cat([id_text, D.pool_text.float()]))

    def protos(self, p, ds):
        S1 = self.S1
        t = S1.CT[ds]
        fidx = {f: i for i, f in enumerate(t["folders"])}
        c = np.array([fidx[S1.g_folders[i]] for i in p])
        text = t["text"][torch.as_tensor(c)].float()
        gen = S1.g_proto[torch.as_tensor(p)].float()
        rc = l2n(gen - self.mu_gen + self.mu_real)
        names_T = l2n(torch.cat([self.ALLT, t["text"].float()]))
        filt = []
        for k, i in enumerate(p):
            E = S1.g_emb[torch.as_tensor(np.nonzero(S1.g_idx == i)[0])].float()
            keep = (E @ names_T.T).argmax(1).numpy() == (self.ALLT.shape[0] + c[k])
            filt.append(l2n(E[torch.as_tensor(np.nonzero(keep)[0])].mean(0, keepdim=True))[0] if keep.any() else gen[k])
        filt = torch.stack(filt)
        filt_rc = l2n(filt - self.mu_gen + self.mu_real)
        gap = l2n(text - self.mu_text + self.mu_real)
        return c, {"text": text, "gen": gen, "gen rc": rc, "gen filt": filt, "gen filt rc": filt_rc,
                   "text+gen rc": l2n(l2n(text) + rc), "text+gen filt rc": l2n(l2n(text) + filt_rc), "text gap": gap}

    def screen(self, log=print):
        S1 = self.S1
        res_all = {}
        for ds in ("ninco", "ssb_hard"):
            p, _ = S1.probe_rows(ds)
            c, P = self.protos(p, ds)
            allB = S1.CT[ds]["cB"].float()
            res = {k: round(float(((Q @ allB.T).argmax(1).numpy() == c).mean() * 100), 1) for k, Q in P.items()}
            rel_t, rel_B = zrow(P["text"] @ self.anch_txt.T), zrow(allB @ self.anch_img.T)
            res["text rel"] = round(float(((l2n(rel_t) @ l2n(rel_B).T).argmax(1).numpy() == c).mean() * 100), 1)
            rel_g = zrow(P["gen"] @ self.anch_img.T)
            res["gen rel"] = round(float(((l2n(rel_g) @ l2n(rel_B).T).argmax(1).numpy() == c).mean() * 100), 1)
            S_dir, S_rel = zrow(P["text"] @ allB.T), zrow(l2n(rel_t) @ l2n(rel_B).T)
            res["text dir+rel"] = round(float(((S_dir + S_rel).argmax(1).numpy() == c).mean() * 100), 1)

            def sig(V):
                return torch.log_softmax(100.0 * V @ self.ALLT.T, 1)
            sB = sig(allB)
            for k in ("gen", "gen rc"):
                sg = sig(P[k])
                pg = sg.exp()
                kl = (pg * sg).sum(1, keepdim=True) - pg @ sB.T          # KL(g || B) for every concept
                res[f"{k} sig"] = round(float((kl.argmin(1).numpy() == c).mean() * 100), 1)
            res["real A"] = round(float(((S1.CT[ds]["cA"][torch.as_tensor(c)].float() @ allB.T).argmax(1).numpy()
                                         == c).mean() * 100), 1)
            res_all[ds] = res
            log(ds, json.dumps(res))
        return res_all


def main(argv=None) -> int:
    torch.set_num_threads(2)
    res = Bridges(Seam1()).screen()
    path = out("seam1_rethink.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
