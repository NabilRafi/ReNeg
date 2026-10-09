"""ReNeg on any base score, and one pass that runs TINS and TANL once and several ReNeg heads on top of them.

ReNeg never changes its base: per batch it reads the base's log-odds lt, votes with it (running Otsu threshold),
and admits images to its ID image prototypes only where CLIP's softmax and the base agree. So one pass of the
expensive base (TINS: 30 gradient steps per OOD-looking image) can feed many ReNeg heads, each with its own state,
and every head gives exactly the score it would give alone.

    ReNegHead.step(X, lt, extra_mask=None, use_vote_mask=True)
        s = (1 - omega) * lt + omega * li                        (score with the state of earlier batches)
        vote = lt below the running Otsu threshold of lt
        base ID mask = (not voted OOD if use_vote_mask) and extra_mask
        updates: KG gate (catching, LCB, vote), ID image prototypes (softmax >= p_min and base ID mask)

Bases used by MultiBase (notebook G4):
    tins       lt = logit(S_TINS); base ID mask = not voted OOD
    tanl       lt = logit(S_TANL); base ID mask = TANL's own confident-ID mask (as ReNegTANL)
    tanl+tins  lt = (logit(S_TANL) + logit(S_TINS)) / 2; base ID mask = TANL's ID mask and not voted OOD
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .core import KGGate, OnlineIdModel, ReNegConfig, VoteHistory, admit_mask, fuse, image_score
from .transport import l2n

try:                                                            # the real oodlab (Colab / Kaggle / local copy)
    from oodlab.methods.base import StepOutput, StreamMethod
except Exception:                                               # noqa: BLE001
    StepOutput, StreamMethod = None, object


def logit64(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.double().clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


class ReNegHead:
    """ReNeg's own state (KG gate, online ID image model, vote history) on top of a score computed elsewhere."""

    def __init__(self, id_text: torch.Tensor, pool_text: torch.Tensor, pool_cluster, rcfg: ReNegConfig,
                 device="cpu", id_prior: Optional[Tuple] = None):
        self.rcfg = rcfg
        self.device = torch.device(device)
        dt = rcfg.torch_dtype
        self.T_id = l2n(torch.as_tensor(id_text).to(device=self.device, dtype=dt))
        self.gate = KGGate(pool_text, pool_cluster, rcfg, device=self.device) if rcfg.use_pool and len(pool_text) else None
        self.idm = OnlineIdModel(self.T_id.shape[0], self.T_id.shape[1], rcfg, device=self.device)
        self.id_prior = None
        if id_prior is not None:
            Xp, yp = id_prior
            self.id_prior = (l2n(torch.as_tensor(Xp).to(device=self.device, dtype=dt)),
                             torch.as_tensor(np.asarray(yp)).long().to(self.device))
        self.reset()

    def reset(self) -> None:
        if self.gate is not None:
            self.gate.reset()
        self.idm.reset()
        if self.id_prior is not None:
            self.idm.add(self.id_prior[0], self.id_prior[1], np.ones(len(self.id_prior[0]), bool))
        self.votes = VoteHistory()

    @torch.no_grad()
    def step(self, X: torch.Tensor, lt: torch.Tensor, extra_mask: Optional[np.ndarray] = None,
             use_vote_mask: bool = True) -> torch.Tensor:
        rc = self.rcfg
        Xr = l2n(X.to(device=self.device, dtype=rc.torch_dtype))
        lt_r = lt.to(device=self.device, dtype=rc.torch_dtype)
        li = image_score(Xr, self.idm, self.gate, rc) if rc.use_image else None
        s = fuse(lt_r, li, rc)
        vote = self.votes.vote(lt.double().cpu().numpy())          # the base's own score, running Otsu threshold
        bm = ~vote if use_vote_mask else np.ones(len(Xr), bool)
        if extra_mask is not None:
            bm = bm & np.asarray(extra_mask, bool)
        lab, adm = admit_mask(Xr, self.T_id, rc, base_mask=bm)
        if self.gate is not None:
            caught_by = self.gate.update(Xr, self.T_id, ood_vote=vote if rc.vote else None)
            adm = adm & ~self.gate.exclusion(caught_by, rc.id_exclude)
        self.idm.add(Xr, lab, adm)
        return s


@dataclass
class HeadSpec:
    name: str
    base: str                  # "tins" | "tanl" | "tanl+tins"
    rcfg: ReNegConfig
    pool: str = "blind"        # key of the pools dict
    prior: bool = True         # the few-shot ID images enter the ID image model at every reset


@dataclass
class MultiConfig:
    init_seed: int = 0         # set by oodlab's runner (seed_method): TANL's queue permutation follows the stream seed

    def with_(self, **kw) -> "MultiConfig":
        return replace(self, **kw)


class MultiBase(StreamMethod):
    """TINS and TANL once per batch, plus ReNeg heads on top of TINS, TANL and TANL+TINS. The returned score is
    the head named ``primary``; every score goes into ``info["scores"]`` (for a hook to save)."""
    name = "multi"

    def __init__(self, tins, tanl, id_text: torch.Tensor, pools: Dict[str, Tuple[torch.Tensor, np.ndarray]],
                 heads: Sequence[HeadSpec], primary: str, device="cpu", id_prior: Optional[Tuple] = None):
        self.tins, self.tanl = tins, tanl
        self.cfg = MultiConfig()
        self.specs = list(heads)
        self.primary = primary
        self.heads = {h.name: ReNegHead(id_text, pools[h.pool][0], pools[h.pool][1], h.rcfg, device=device,
                                        id_prior=id_prior if h.prior else None) for h in self.specs}
        if primary not in self.heads and primary not in ("tins", "tanl", "tanl+tins"):
            raise ValueError(f"unknown primary score {primary!r}")
        self.reset()

    def reset(self) -> None:
        if self.tins is not None:
            self.tins.reset()
        if self.tanl is not None:
            if hasattr(self.tanl, "cfg") and hasattr(self.tanl.cfg, "init_seed"):
                self.tanl.cfg = self.tanl.cfg.with_(init_seed=int(self.cfg.init_seed))
            self.tanl.reset()
        for h in getattr(self, "heads", {}).values():
            h.reset()

    @torch.no_grad()
    def step(self, feats: torch.Tensor):
        out_s = self.tins.step(feats)
        out_a = self.tanl.step(feats) if self.tanl is not None else None
        X = l2n(feats.float())
        lt = {"tins": logit64(out_s.score)}
        scores = {"tins": out_s.score.float().cpu()}
        id_mask = None
        if out_a is not None:
            lt["tanl"] = logit64(out_a.score)
            lt["tanl+tins"] = 0.5 * (lt["tanl"] + lt["tins"])
            scores["tanl"] = out_a.score.float().cpu()
            scores["tanl+tins"] = lt["tanl+tins"].float().cpu()
            id_mask = out_a.info["id_mask"].cpu().numpy().astype(bool)
        for spec in self.specs:
            if spec.base == "tins":
                s = self.heads[spec.name].step(X, lt["tins"])
            elif spec.base == "tanl":
                s = self.heads[spec.name].step(X, lt["tanl"], extra_mask=id_mask, use_vote_mask=False)
            elif spec.base == "tanl+tins":
                s = self.heads[spec.name].step(X, lt["tanl+tins"], extra_mask=id_mask)
            else:
                raise ValueError(spec.base)
            scores[spec.name] = s.float().cpu()
        main = scores[self.primary].to(out_s.score.device)
        info = {"scores": scores, "tins": out_s.info}
        return StepOutput(pred=out_s.pred, score=main, info=info)
