"""The experimental replay engine: ReNeg on a TANL trace (or any base score) with the options the design tests
used. With the defaults it is ReNeg exactly as packaged in reneg.core (``ReNegTANL``); the options are the variants
that were tested and either kept (agreed ID admission, few-shot prior) or rejected (see docs/NEGATIVE_RESULTS.md).

opts (all default to the method as packaged):
  proto      "all"   every text-caught image enters its KG name's visual prototype
             "vote"  only caught images that the base score puts below its running Otsu threshold
  idadm      "softmax"        ID admission: CLIP softmax >= p_min within the base's ID mask
             "softmax+tanl"   ... and only where the base agrees (the "agreed" admission of every headline result)
             "softmax+tanl+imgagree"  ... and the text-predicted class is the nearest ID image prototype
  center     image score computed after removing the running stream mean
  softcatch  KG penalty weighted by the text posterior of the best KG name against the best ID name
  classcal   kappa > 0: class-wise calibration of the penalty on earlier confident ID images
  k_vis      caught images before a name gets a visual prototype
  excl       "pen" | "nearcaught": keep penalised / nearly caught images out of the ID model
  lag        admit a batch L batches later, minus images whose KG name the gate trusts by then (pi > lag_pi)
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

import reneg.core as RC
from reneg.reneg_tanl import logit
from reneg.transport import l2n

from . import data as D
from .traces import text_from_trace

DEFAULT = dict(proto="all", idadm="softmax", center=False, softcatch=False, classcal=0.0, k_vis=None, excl=None,
               excl_delta=0.0, lag=0, lag_pi=0.5)

# ReNeg's three operating points on the TANL base (the KG pool adds no mass to TANL's text score)
B0 = dict(n_min=1, pool_frac=0.0, pool_in_text=False)
MODES = {"balanced": dict(B0, image_mode="kg_clip", vote=True), "max": dict(B0),
         "safe": dict(B0, image_mode="kg_clip_caught", vote=True)}

JOBS4 = [("imagenet_val_all", d) for d in D.FOUR]
JOBSO = [("imagenet_test", d) for d in D.OOD_SETS]
SCREEN = [("imagenet_val_all", "places"), ("imagenet_val_all", "textures_all"), ("imagenet_val_all", "sun"),
          ("imagenet_test", "ninco"), ("imagenet_test", "textures"), ("imagenet_test", "openimage_o")]
JOBS = {"four": JOBS4, "openood": JOBSO, "all": JOBS4 + JOBSO, "screen": SCREEN,
        "screen7": SCREEN[:4] + [("imagenet_test", "ssb_hard")] + SCREEN[4:]}


class GateX(RC.KGGate):
    def update_x(self, X, T_id, ood_vote, proto_mask):
        """As KGGate.update, but only images in proto_mask enter fsum / n_vis (the margin evidence uses all)."""
        self._T_id = T_id
        S_id = X @ T_id.T
        S_pool = X @ self.T.T
        best_id = S_id.max(1).values
        bp, j = S_pool.max(1)
        hit_t = bp > best_id
        hit = hit_t.numpy()
        jn = j.numpy()
        caught_by = np.where(hit, jn, -1)
        if not hit.any():
            return caught_by
        jh = jn[hit]
        x = (bp[hit_t] - best_id[hit_t]).double().numpy()
        np.add.at(self.n, jh, 1)
        np.add.at(self.xsum, jh, x)
        np.add.at(self.x2sum, jh, x * x)
        pm = hit & proto_mask
        if pm.any():
            self.fsum.index_add_(0, torch.as_tensor(jn[pm]), X[torch.as_tensor(pm)])
            np.add.at(self.n_vis, jn[pm], 1)
        if ood_vote is not None:
            np.add.at(self.vsum, jh, np.asarray(ood_vote, float)[hit])
        if self.cfg.gate:
            self._gate()
        return caught_by


def image_score_x(X, idm, gate, cfg, opts, T_id):
    if not idm.ready():
        return None
    V, bg = idm.prototypes(), idm.background()
    Vp, lw = gate.visual_terms()
    if opts["center"]:
        mu = l2n(idm.img_sum[None])[0]
        Xc, Vc = l2n(X - mu), l2n(V - mu)
        Vpc = l2n(Vp - mu) if Vp is not None else None
        bgc = l2n(bg - mu)
    else:
        Xc, Vc, Vpc, bgc = X, V, Vp, bg
    tau = cfg.tau
    if cfg.image_mode in ("kg_clip", "kg_clip_caught"):
        if Vpc is None:
            return torch.zeros(X.shape[0], dtype=X.dtype)
        pen = torch.clamp(RC.lse_w(Xc @ Vc.T, tau) - RC.lse_w(Xc @ Vpc.T, tau, lw) + cfg.clip_margin, max=0.0)
        if cfg.image_mode == "kg_clip_caught":
            pen = pen * gate.catch_weight(X)
        elif opts["softcatch"]:
            tid = (X @ T_id.T).max(1).values
            bp, j = (X @ gate.T.T).max(1)
            w = torch.as_tensor(gate.pi, dtype=X.dtype)[j]
            pen = pen * w * torch.sigmoid(tau * (bp - tid))
        return pen
    l_bg = RC.lse_w(Xc @ bgc.T, tau)
    li_id = RC.lse_w(Xc @ Vc.T, tau) - l_bg
    li_ood = torch.zeros_like(li_id)
    if Vpc is not None:
        neg = torch.cat([bgc, Vpc])
        logw = torch.cat([torch.zeros(bgc.shape[0], dtype=lw.dtype), lw])
        li_ood = l_bg - RC.lse_w(Xc @ neg.T, tau, logw)
    return cfg.id_weight * li_id + cfg.ood_weight * li_ood


def _prior(prior):
    """True -> the 5k labelled ID images (data.shots()); (X, y) -> those; False / None -> no prior."""
    if prior is True:
        Xv, yv, _ = D.shots()
        return Xv, yv
    return prior if prior else None


@torch.no_grad()
def replay_x(trace, X, id_text, pool_text, pool_cluster, rcfg, opts=None, batch=256, prior=None, oracle_y=None):
    """ReNeg on a TANL trace with the experimental options above.

    prior: labelled ID images that enter the ID image model before the stream (the few-shot setting).
    oracle_y: true labels (-1 = OOD) for the oracle ID models of experiments/05 (opts["oracle_mode"]):
      full      true labels, true ID images only (upper bound)     purity    true ID only, CLIP's label
      decontam  realistic admission minus its OOD images            coverage  realistic admission plus every ID image
      label     realistic admission, true label for ID images
    """
    opts = {**DEFAULT, **(opts or {})}
    dt = rcfg.torch_dtype
    id_text = D.id_text if id_text is None else id_text
    T_id = l2n(torch.as_tensor(id_text).to(dt))
    gate = GateX(pool_text, pool_cluster, rcfg)
    idm = RC.OnlineIdModel(T_id.shape[0], T_id.shape[1], rcfg)
    pr = _prior(prior)
    if pr is not None:
        idm.add(l2n(pr[0].to(dt)), torch.as_tensor(pr[1]), np.ones(len(pr[0]), bool))
    votes = RC.VoteHistory()
    res = np.zeros(len(X))
    C = T_id.shape[0]
    cal_s, cal_n = np.zeros(C), np.zeros(C)
    pending = []
    for s in range(0, len(X), batch):
        Xr = l2n(X[s:s + batch].to(dt))
        a, cum = trace["a"][s:s + batch], trace["cum"][s:s + batch]
        lt = logit(text_from_trace(a, cum, trace["step"], None, 0.0)).to(dt)
        li = image_score_x(Xr, idm, gate, rcfg, opts, T_id)
        excl_m = None
        if opts["excl"] == "pen":                    # keep images the KG penalty pushes down out of the ID model
            pen_kg = li if rcfg.image_mode in ("kg_clip", "kg_clip_caught") else \
                image_score_x(Xr, idm, gate, rcfg.with_(image_mode="kg_clip"), opts, T_id)
            excl_m = (pen_kg.double().numpy() < -0.05) if pen_kg is not None else None
        elif opts["excl"] == "nearcaught":           # ... or whose best KG name is within delta of the best ID name
            excl_m = ((Xr @ gate.T.T).max(1).values > (Xr @ T_id.T).max(1).values - opts["excl_delta"]).numpy()
        tanl_idm = np.asarray(trace["id_mask"][s:s + batch], bool)
        if opts["classcal"] and li is not None:
            tp = (Xr @ T_id.T).argmax(1).numpy()
            raw = li.double().numpy().copy()
            m_c = cal_s[tp] / (cal_n[tp] + opts["classcal"])
            li = torch.as_tensor(np.minimum(0.0, raw - m_c), dtype=dt)
            np.add.at(cal_s, tp[tanl_idm], raw[tanl_idm])
            np.add.at(cal_n, tp[tanl_idm], 1.0)
        res[s:s + batch] = RC.fuse(lt, li, rcfg).double().numpy()
        lab, adm = RC.admit_mask(Xr, T_id, rcfg, base_mask=tanl_idm)
        if opts["idadm"].startswith("softmax+tanl"):
            adm = adm & tanl_idm
        if excl_m is not None:
            adm = adm & ~excl_m
        if opts["idadm"].endswith("+imgagree") and idm.ready():
            # cross-modal agreement: the text-predicted class is also the nearest ID image prototype
            el = np.nonzero(idm.counts >= rcfg.n_min)[0]
            near = torch.as_tensor(el)[(Xr @ idm.prototypes().T).argmax(1)].numpy()
            labn = torch.as_tensor(lab).numpy()
            adm = adm & ((near == labn) | (idm.counts[labn] < rcfg.n_min))
        if oracle_y is not None:
            yb = oracle_y[s:s + batch]
            om = opts.get("oracle_mode", "full")
            if om == "full":
                lab, adm = torch.as_tensor(np.maximum(yb, 0)), yb >= 0
            elif om == "purity":
                adm = yb >= 0
            elif om == "decontam":
                adm = adm & (yb >= 0)
            elif om == "coverage":
                adm = adm | (yb >= 0)
            elif om == "label":
                lab = torch.where(torch.as_tensor(yb >= 0), torch.as_tensor(np.maximum(yb, 0)), torch.as_tensor(lab))
        vote = votes.vote(logit(torch.as_tensor(trace["tanl"][s:s + batch])).numpy())
        pm = vote if opts["proto"] == "vote" else np.ones(len(Xr), bool)
        caught = gate.update_x(Xr, T_id, vote if rcfg.vote else None, pm)
        if oracle_y is None or opts.get("oracle_mode") in ("label", "decontam", "coverage"):
            adm = adm & ~gate.exclusion(caught, rcfg.id_exclude)
        if opts["lag"]:
            # lagged admission: a batch joins the ID model L batches later, minus the images whose catching KG name
            # the gate trusts by then (pi > lag_pi); scoring still uses only earlier batches' state
            pending.append((Xr, lab, adm, caught))
            if len(pending) > opts["lag"]:
                Xo, lo, ao, co = pending.pop(0)
                hit = co >= 0
                bad = np.zeros(len(co), bool)
                bad[hit] = gate.pi[co[hit]] > opts["lag_pi"]
                idm.add(Xo, lo, ao & ~bad)
        else:
            idm.add(Xr, lab, adm)
    return res


@torch.no_grad()
def replay_base(lt_base, vote_lt, base_mask, X, P, cl, rcfg, opts=None, prior=None, batch=256, id_text=None):
    """ReNeg on an arbitrary base (TINS, TANL+TINS). lt_base, vote_lt: (N,) log-odds (higher = more ID); base_mask:
    (N,) bool base ID mask, or None for 'not voted OOD'. Same order of operations as ReNegTANL.step."""
    opts = {**DEFAULT, **(opts or {})}
    dt = rcfg.torch_dtype
    id_text = D.id_text if id_text is None else id_text
    T_id = l2n(torch.as_tensor(id_text).to(dt))
    gate = GateX(P, cl, rcfg)
    idm = RC.OnlineIdModel(T_id.shape[0], T_id.shape[1], rcfg)
    pr = _prior(prior)
    if pr is not None:
        idm.add(l2n(pr[0].to(dt)), torch.as_tensor(pr[1]), np.ones(len(pr[0]), bool))
    votes = RC.VoteHistory()
    res = np.zeros(len(X))
    for s in range(0, len(X), batch):
        Xr = l2n(X[s:s + batch].to(dt))
        lt = torch.as_tensor(lt_base[s:s + batch]).to(dt)
        li = image_score_x(Xr, idm, gate, rcfg, opts, T_id)
        res[s:s + batch] = RC.fuse(lt, li, rcfg).double().numpy()
        vote = votes.vote(np.asarray(vote_lt[s:s + batch], float))
        bm = ~vote if base_mask is None else np.asarray(base_mask[s:s + batch], bool)
        lab, adm = RC.admit_mask(Xr, T_id, rcfg, base_mask=bm)
        if opts["idadm"] == "softmax+tanl":
            adm = adm & bm
        pm = vote if opts["proto"] == "vote" else np.ones(len(Xr), bool)
        caught = gate.update_x(Xr, T_id, vote if rcfg.vote else None, pm)
        adm = adm & ~gate.exclusion(caught, rcfg.id_exclude)
        idm.add(Xr, lab, adm)
    return res


def mode_config(mode: str, opts: Optional[dict] = None) -> RC.ReNegConfig:
    """An operating point with the config-level options of a plan item (k_vis, id_admit, cfg)."""
    cfg = RC.ReNegConfig(**MODES[mode])
    opts = opts or {}
    if opts.get("k_vis"):
        cfg = cfg.with_(k_vis=int(opts["k_vis"]))
    if opts.get("id_admit"):
        cfg = cfg.with_(id_admit=opts["id_admit"])
    if opts.get("cfg"):
        cfg = cfg.with_(**opts["cfg"])
    return cfg


# ------------------------------------------------------------------------------------------- result rows
def append(path: str, rec: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def done(path: str, keys) -> set:
    if not os.path.isfile(path):
        return set()
    with open(path) as fh:
        return {tuple(r.get(k) for k in keys) for r in map(json.loads, fh)}
