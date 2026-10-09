"""TANL traces: run oodlab's TANL once per stream, keep what ReNeg needs, then replay ReNeg on top cheaply.

TANL's dynamics do not depend on ReNeg (ReNeg only reads TANL's selection and score), so for every batch we store

  a     (B,)    LSE over the ID logits
  cum   (B, M)  logcumsumexp over TANL's M selected negatives, in TANL's order
  id_mask (B,)  TANL's ID-admission mask,  pred (B,),  tanl (B,) TANL's own score

and ``replay(trace, X, ...)`` gives exactly ``ReNegTANL.step``'s scores (experiments/00_check_equivalence.py).
A trace of one stream takes 1-9 minutes on two CPU cores and about 120 MB (fp16).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

import reneg.core as RC
from oodlab.methods.tanl import TANL, TANLConfig
from reneg.reneg_tanl import logit
from reneg.transport import l2n

from . import data as D
from .data import load_set, metrics, pair_stream, textbank
from .paths import TRACES

ID_OF = {"openood_v15": "imagenet_test", "four_ood": "imagenet_val_all"}


@torch.no_grad()
def make_trace(tanl: TANL, X: torch.Tensor, batch: int = 256, store_dtype=torch.float32):
    tanl.reset()
    A, CUM, IDM, PRED, CONF = [], [], [], [], []
    t0 = time.time()
    for s in range(0, len(X), batch):
        v = X[s:s + batch]
        o = tanl.step(v)
        sel = o.info["sel"].long()
        lg = tanl.logits(v, tanl.selected_T(sel)).float()
        C = tanl.C
        A.append(torch.logsumexp(lg[:, :C], 1))
        CUM.append(torch.logcumsumexp(lg[:, C:], 1).to(store_dtype))
        IDM.append(o.info["id_mask"].cpu())
        PRED.append(o.pred.cpu())
        CONF.append(o.score.float().cpu())
    return {"a": torch.cat(A), "cum": torch.cat(CUM), "id_mask": torch.cat(IDM).numpy(),
            "pred": torch.cat(PRED), "tanl": torch.cat(CONF).numpy(), "sec": time.time() - t0,
            "step": int(tanl.cfg.step)}


def text_from_trace(a, cum, step, P=None, frac=1.0):
    """TANL's activation-aware score with an optional pool mass, from a trace slice (closed form).
    frac: the share of TANL's averaged terms (the ones over the most negatives) that also count the pool mass."""
    n_neg = cum.shape[1]
    ms = [min(j + step, n_neg) for j in range(0, n_neg, step)] if step > 0 else [n_neg]
    sel = cum[:, [m - 1 for m in ms]].float()
    if P is not None and frac > 0:
        k0 = int(round((1 - frac) * sel.shape[1]))
        sel = torch.cat([sel[:, :k0], torch.logaddexp(sel[:, k0:], P[:, None])], 1)
    return torch.sigmoid(a[:, None] - sel).mean(1)


@torch.no_grad()
def replay(trace, X, id_text, pool_text_, pool_cluster, rcfg: RC.ReNegConfig, batch: int = 256, pool_prior=None):
    """ReNeg on top of a TANL trace (the method as packaged in reneg.core, softmax admission within TANL's mask).
    Returns the fused scores, the text log-odds, the gate and the ID model."""
    dt = rcfg.torch_dtype
    T_id = l2n(torch.as_tensor(id_text).to(dt))
    gate = RC.KGGate(pool_text_, pool_cluster, rcfg, pool_prior) if (pool_text_ is not None and rcfg.use_pool) else None
    idm = RC.OnlineIdModel(T_id.shape[0], T_id.shape[1], rcfg)
    votes = RC.VoteHistory()
    res = np.zeros(len(X))
    lt_all = np.zeros(len(X))
    for s in range(0, len(X), batch):
        Xr = l2n(X[s:s + batch].to(dt))
        a, cum = trace["a"][s:s + batch], trace["cum"][s:s + batch]
        P = None
        if gate is not None and rcfg.pool_in_text:
            Tp, lw = gate.text_terms()
            if Tp.shape[0]:
                P = torch.logsumexp(rcfg.tau * Xr @ Tp.T + lw[None, :], 1).float()
        conf = text_from_trace(a, cum, trace["step"], P, getattr(rcfg, "pool_frac", 1.0))
        lt = logit(conf).to(dt)
        li = RC.image_score(Xr, idm, gate, rcfg) if rcfg.use_image else None
        s_ = RC.fuse(lt, li, rcfg)
        res[s:s + batch] = s_.double().numpy()
        lt_all[s:s + batch] = lt.double().numpy()
        lab, adm = RC.admit_mask(Xr, T_id, rcfg, base_mask=trace["id_mask"][s:s + batch])
        if gate is not None:
            vote = None
            if rcfg.vote:                                   # TANL's own score, as ReNegTANL uses it
                vote = votes.vote(logit(torch.as_tensor(trace["tanl"][s:s + batch])).numpy())
            caught = gate.update(Xr, T_id, ood_vote=vote)
            adm = adm & ~gate.exclusion(caught, rcfg.id_exclude)
        idm.add(Xr, lab, adm)
    return res, lt_all, gate, idm


# ----------------------------------------------------------------------------------------- trace files
def trace_path(ds, seed, id_name="imagenet_test", tag=""):
    return os.path.join(TRACES, f"{id_name}__{ds}__s{seed}{('__' + tag) if tag else ''}.pt")


def load_trace(ds, seed, id_name="imagenet_test", tag=""):
    p = trace_path(ds, seed, id_name, tag)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no TANL trace {p}: run experiments/01_build_traces.py first")
    return torch.load(p, weights_only=False)


def tanl_inputs():
    """(ID text, corpus, noise images, n_neglabel) from the exported text bank."""
    tb = textbank()
    return (l2n(torch.as_tensor(tb["id_text"]).float()), tb["corpus_text"],
            l2n(torch.as_tensor(tb["noise_feats"]).float()), int(tb["n_selected"]))


def build_traces(sets_, seed=0, id_name="imagenet_test", cfg=None, extend=None, max_images=None, log=print):
    """TANL (paper hyper-parameters) on each (ID, OOD set) stream in the official order, saved to RENEG_TRACES.
    extend: None (TANL's corpus) or "full" / "blind" (the KG pool appended to TANL's corpus, an ablation).
    max_images: quick mode (tests only): the first N images of each stream."""
    os.makedirs(TRACES, exist_ok=True)
    tid, corpus, noise, n_sel = tanl_inputs()
    cfg = (cfg or TANLConfig()).with_(record=True, init_seed=seed)
    if extend:
        corpus = torch.cat([corpus, D.pool_subset(extend)[0]])
        log(f"corpus extended with the {extend} KG pool: {tuple(corpus.shape)}")
    X_id = load_set(id_name)["X"]
    for ds in sets_:
        p = trace_path(ds, seed, id_name, tag=f"ext_{extend}" if extend else "")
        if os.path.isfile(p):
            log(f"exists: {p}")
            continue
        X, is_ood, _ = pair_stream(X_id, load_set(ds)["X"], seed)
        if max_images:
            X, is_ood = X[:max_images], is_ood[:max_images]
        tanl = TANL(tid, corpus, noise, cfg, n_neglabel=n_sel)
        tr = make_trace(tanl, X, store_dtype=torch.float16)
        tr["is_ood"] = is_ood
        r = metrics(tr["tanl"][~is_ood], tr["tanl"][is_ood])
        tr["tanl_metrics"] = r
        torch.save(tr, p)
        log(f"{id_name} / {ds} seed {seed}: TANL FPR95 {r['fpr95']:.2f} AUROC {r['auroc']:.2f} "
            f"({tr['sec'] / 60:.1f} min)")


def stream_for(ds, seed, id_name="imagenet_test", trace=None):
    """(X, is_ood, p) of a stream in the official order, checked against (and cut to the length of) its trace."""
    X, is_ood, p = pair_stream(load_set(id_name)["X"], load_set(ds)["X"], seed)
    if trace is not None:
        n = len(trace["tanl"])
        X, is_ood, p = X[:n], is_ood[:n], p[:n]
        if not (np.asarray(trace["is_ood"]) == is_ood).all():
            raise ValueError(f"trace and stream order disagree for {id_name} / {ds} seed {seed}")
    return X, is_ood, p
