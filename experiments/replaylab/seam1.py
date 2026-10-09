"""Shared code of the Seam 1 analyses: can a text-to-image bridge give unseen near-OOD names an image prototype?

Inputs: G1b's analysis pack (``data.pack1``: concept names and text of SSB-hard and NINCO, the class statistics of
the confident half of ImageNet test, the mean image) and P1's generated-image probe (SDXL-Turbo, 4 images per
name, encoded with ViT-B/16; results/seam1/reneg_gen_probe.npz).

Concept tables split each benchmark concept's images by stream parity: half A builds the real centroid (what a
perfect bridge would give), half B is scored. The online stream test (``online_init``) is P1's protocol: ImageNet
test half B against the probed concepts' half-B images, the concept names as extra negatives, their visual
prototypes seeded with a given prototype (generated, mapped, real) at pseudo-count w and updated online.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

import reneg.core as RC
from reneg.reneg_tanl import logit
from reneg.transport import l2n

from . import data as D
from .data import T, lse, metrics, pair_stream
from .paths import RESULTS
from .replay import DEFAULT, GateX, image_score_x
from .traces import text_from_trace

GEN_PROBE = os.environ.get("RENEG_GEN_PROBE", os.path.join(RESULTS, "seam1", "reneg_gen_probe.npz"))


class Seam1:
    def __init__(self, gen_probe: Optional[str] = None):
        G = np.load(gen_probe or GEN_PROBE)
        self.g_emb = T(G["emb"])
        self.g_idx = G["name_idx"].astype(int)
        self.g_names, self.g_tags, self.g_folders = list(G["names"]), list(G["tags"]), list(G["folders"])
        self.NP = len(self.g_names)
        self.g_proto = l2n(torch.stack([self.g_emb[torch.as_tensor(np.nonzero(self.g_idx == i)[0])].mean(0)
                                        for i in range(self.NP)]))
        self.S = {n: D.load_set(n) for n in ("imagenet_test", "ssb_hard", "ninco")}
        self.Xid, self.yid = self.S["imagenet_test"]["X"], self.S["imagenet_test"]["labels"]
        self.XidA, self.XidB = self.Xid[0::2], self.Xid[1::2]
        self.dim = self.Xid.shape[1]
        self.P1 = D.pack1()
        # the pack's own fp16 text side (as in the original analyses); the 8-bit text bank export otherwise
        self.id_text = T(self.P1["id_text"]) if "id_text" in self.P1.files else D.id_text
        self.neg_text = T(self.P1["neg_text"]) if "neg_text" in self.P1.files else D.neg_text
        self.CT = {ds: self.concept_table(ds) for ds in ("ssb_hard", "ninco")}
        stats = D.stats
        self.Cel = l2n(stats.directions()[stats.eligible(3)].float())
        g = torch.Generator().manual_seed(0)
        self.BG = l2n(l2n(D.m_img)[None, :] + 0.25 * l2n(torch.randn(10000, self.dim, generator=g)))

    # ------------------------------------------------------------------------- concept tables (A/B halves)
    def concept_table(self, ds):
        d = self.S[ds]
        n = len(d["folder_names"])
        A = np.arange(len(d["X"])) % 2 == 0
        cA, cB = torch.zeros(n, self.dim), torch.zeros(n, self.dim)
        for i in range(n):
            m = d["folder"] == i
            cA[i] = d["X"][torch.as_tensor(np.nonzero(m & A)[0])].mean(0)
            cB[i] = d["X"][torch.as_tensor(np.nonzero(m & ~A)[0])].mean(0)
        return {"names": list(self.P1[f"concept_names__{ds}"]), "text": T(self.P1[f"concept_text__{ds}"]),
                "cA": l2n(cA), "cB": l2n(cB), "folders": d["folder_names"], "A": A}

    def probe_rows(self, ds):
        """Probe indices of dataset ds and their concept indices."""
        fidx = {f: i for i, f in enumerate(self.CT[ds]["folders"])}
        p = [i for i in range(self.NP) if self.g_tags[i] == ds]
        return np.array(p), np.array([fidx[self.g_folders[i]] for i in p])

    # ----------------------------------------------------------------------------- prototype quality screen
    def quality(self):
        res = {}
        for ds in ("ninco", "ssb_hard"):
            p, c = self.probe_rows(ds)
            t = self.CT[ds]
            ti = torch.as_tensor
            gp, tx, cA, cB = self.g_proto[ti(p)], t["text"][ti(c)], t["cA"][ti(c)], t["cB"][ti(c)]
            allB = t["cB"]                                   # retrieve among ALL concepts of the dataset (64 / 980)

            def r1(Q):
                return float(((Q @ allB.T).argmax(1).numpy() == c).mean() * 100)
            res[ds] = {"n": len(p), "cos(gen,real)": float((gp * cB).sum(1).mean()),
                       "cos(realA,realB)": float((cA * cB).sum(1).mean()), "cos(text,real)": float((tx * cB).sum(1).mean()),
                       "top1 gen->real": r1(gp), "top1 text->real": r1(tx), "top1 realA->realB": r1(cA)}
            simB = gp @ allB.T
            own = simB[torch.arange(len(c)), ti(c)]
            simB[torch.arange(len(c)), ti(c)] = -1
            res[ds]["gen: own - best other"] = float((own - simB.max(1).values).mean())
        p = np.array([i for i in range(self.NP) if self.g_tags[i] == "imagenet"])
        if len(p):
            id_text = self.id_text
            cls = np.array([int(self.g_folders[i]) for i in p])
            rows = np.isin(self.g_idx, p)
            lab = np.array([int(self.g_folders[i]) for i in self.g_idx[rows]])
            pred = (self.g_emb[torch.as_tensor(np.nonzero(rows)[0])] @ id_text.T).argmax(1).numpy()
            yB = self.yid[1::2]
            real = l2n(torch.stack([self.XidB[torch.as_tensor(np.nonzero(yB == k)[0])].mean(0)
                                    for k in range(id_text.shape[0])]))
            gp = self.g_proto[torch.as_tensor(p)]
            res["imagenet"] = {
                "n": len(p), "zero-shot acc of generated images": float((pred == lab).mean() * 100),
                "zero-shot acc of real test images": float(((self.Xid @ id_text.T).argmax(1).numpy() == self.yid).mean() * 100),
                "cos(gen,real)": float((gp * real[torch.as_tensor(cls)]).sum(1).mean()),
                "cos(text,real)": float((id_text[torch.as_tensor(cls)] * real[torch.as_tensor(cls)]).sum(1).mean()),
                "top1 gen->real (1000 classes)": float(((gp @ real.T).argmax(1).numpy() == cls).mean() * 100),
                "top1 text->real (1000 classes)": float(((id_text[torch.as_tensor(cls)] @ real.T).argmax(1).numpy() == cls).mean() * 100)}
        return res

    # ------------------------------------------------------------------------------- online stream test
    def ell(self, X, Pid, Pneg, tau=100.0):
        return (lse(X @ Pid.T, tau) - lse(X @ Pneg.T, tau)).numpy()

    def online_init(self, X_ood, pool, init_proto=None, w=0.0, update=True, k=2, omega=0.25, tau=100.0, batch=256,
                    seed=0, X_id=None):
        id_text, neg_text = self.id_text, self.neg_text
        X_id = self.XidB if X_id is None else X_id
        X, is_ood, _ = pair_stream(X_id, X_ood, seed)
        negs_t = torch.cat([neg_text, pool])
        nP = pool.shape[0]
        sums, cnt = torch.zeros(nP, self.dim), np.zeros(nP)
        if init_proto is not None and w > 0:
            sums += w * init_proto
            cnt += w
        Pall = torch.cat([id_text, negs_t])
        scores = np.zeros(len(X))
        kk = min(k, w) if w > 0 else k
        for s in range(0, len(X), batch):
            v = X[s:s + batch]
            lt = self.ell(v, id_text, negs_t, tau)
            adm = cnt >= kk
            E = l2n(sums[torch.as_tensor(adm)]) if adm.any() else None
            li = self.ell(v, self.Cel, self.BG if E is None else torch.cat([self.BG, E]), tau)
            scores[s:s + batch] = (1 - omega) * lt + omega * li
            if update:
                top = (v @ Pall.T).argmax(1).numpy() - id_text.shape[0] - neg_text.shape[0]
                h = np.nonzero(top >= 0)[0]
                if len(h):
                    sums.index_add_(0, torch.as_tensor(top[h]), v[torch.as_tensor(h)])
                    np.add.at(cnt, top[h], 1)
        m = metrics(scores[~is_ood], scores[is_ood])
        return round(m["fpr95"], 2), round(m["auroc"], 2)

    def ood_half_b(self, ds, c):
        t, d = self.CT[ds], self.S[ds]
        keepB = (~t["A"]) & np.isin(d["folder"], c)                      # OOD half B of the probed concepts
        return d["X"][torch.as_tensor(np.nonzero(keepB)[0])]

    def stream_test(self, ds, ws=(1, 4, 16), seeds=(0, 1, 2), protos=None, log=print):
        """P1's stream test. protos: optional {arm name: (n_probe, D) prototypes} replacing the generated ones."""
        p, c = self.probe_rows(ds)
        t = self.CT[ds]
        X_ood = self.ood_half_b(ds, c)
        pool = t["text"][torch.as_tensor(c)]
        real = t["cA"][torch.as_tensor(c)]
        gen = self.g_proto[torch.as_tensor(p)]
        rows = []

        def arm(name, **kw):
            r = [self.online_init(X_ood, pool, seed=s, **kw) for s in seeds]
            rows.append({"set": ds, "arm": name, "FPR95": round(float(np.mean([a for a, _ in r])), 2),
                         "AUROC": round(float(np.mean([b for _, b in r])), 2)})
            log(rows[-1])

        m = metrics(D.neglabel_form(self.XidB, negs=self.neg_text, ids=self.id_text),
                    D.neglabel_form(X_ood, negs=self.neg_text, ids=self.id_text))
        rows.append({"set": ds, "arm": "NegLabel form (reference)", "FPR95": round(m["fpr95"], 2),
                     "AUROC": round(m["auroc"], 2)})
        log(rows[-1])
        arm("text only: pool names, no image side", omega=0.0)
        arm("text-seeded evidence (stream only)")
        for w in ws:
            arm(f"perfect map (real centroid, w={w})", init_proto=real, w=w)
            arm(f"generated prototype (w={w})", init_proto=gen, w=w)
            for name, P_ in (protos or {}).items():
                arm(f"{name} (w={w})", init_proto=P_, w=w)
        arm("perfect map, static (no stream update)", init_proto=real, w=4, update=False)
        arm("generated, static (no stream update)", init_proto=gen, w=4, update=False)
        return rows


@torch.no_grad()
def replay_prior(trace, X, P, cl, rcfg, prior_idx=None, prior_vec=None, w=4.0, opts=None, batch=256, id_text=None):
    """ReNeg on a TANL trace (agreed admission off, as in the Oct 2 run) whose KG gate starts with image-space
    prototypes ``prior_vec`` for the names ``prior_idx`` at pseudo-count w (a solved Seam 1)."""
    opts = {**DEFAULT, **(opts or {})}
    dt = rcfg.torch_dtype
    T_id = l2n(torch.as_tensor(D.id_text if id_text is None else id_text).to(dt))
    gate = GateX(P, cl, rcfg)
    if prior_idx is not None:
        gate.fsum[torch.as_tensor(prior_idx)] = w * prior_vec.to(dt)
        gate.n_vis[prior_idx] = w
    idm = RC.OnlineIdModel(T_id.shape[0], T_id.shape[1], rcfg)
    votes = RC.VoteHistory()
    res = np.zeros(len(X))
    for s in range(0, len(X), batch):
        Xr = l2n(X[s:s + batch].to(dt))
        a, cum = trace["a"][s:s + batch], trace["cum"][s:s + batch]
        lt = logit(text_from_trace(a, cum, trace["step"], None, 0.0)).to(dt)
        li = image_score_x(Xr, idm, gate, rcfg, opts, T_id)
        res[s:s + batch] = RC.fuse(lt, li, rcfg).double().numpy()
        tanl_idm = np.asarray(trace["id_mask"][s:s + batch], bool)
        lab, adm = RC.admit_mask(Xr, T_id, rcfg, base_mask=tanl_idm)
        if opts["idadm"] == "softmax+tanl":
            adm = adm & tanl_idm
        vote = votes.vote(logit(torch.as_tensor(trace["tanl"][s:s + batch])).numpy())
        pm = vote if opts["proto"] == "vote" else np.ones(len(Xr), bool)
        caught = gate.update_x(Xr, T_id, vote if rcfg.vote else None, pm)
        adm = adm & ~gate.exclusion(caught, rcfg.id_exclude)
        idm.add(Xr, lab, adm)
    return res
