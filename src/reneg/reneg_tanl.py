"""ReNeg on top of TANL (Zhang et al., CVPR 2026).

TANL is left exactly as it is: its queues, its automatic threshold and its per-batch selection of M activated
negatives from the 69,554-word corpus. ReNeg adds, per batch:

1. KG near-OOD names in the text score. TANL's activation-aware score averages, over the M selected negatives
   sorted by activation, sigmoid(LSE(ID) - LSE(first m negatives)). ReNeg adds the gated pool mass
   P = LSE(tau v.T_pool + log pi) to every term::

       conf = mean_m sigmoid( LSE(ID) - logaddexp(LSE(first m negatives), P) )

   With no pool (or every pi = 0) this is TANL's score exactly. lt = logit(conf).
2. The image score li (ID image prototypes from the stream vs background + visual prototypes of the gated KG
   names), fused in log-odds: s = (1 - omega) lt + omega li.

oodlab must be importable (it is on Colab and Kaggle; locally add the oodlab code folder to sys.path).

Devices: ReNeg's tensors live on TANL's device unless ``reneg_device`` says otherwise (e.g. "cpu" as a fallback
while TANL runs on the GPU).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .core import KGGate, OnlineIdModel, ReNegConfig, VoteHistory, admit_mask, fuse, image_score
from .transport import l2n

try:                                                            # the real oodlab (Colab / Kaggle / local copy)
    from oodlab.methods.tanl import TANL, TANLConfig
    from oodlab.methods.base import StepOutput
except Exception:                                               # noqa: BLE001  (lets the module import for docs)
    TANL, TANLConfig, StepOutput = object, None, None


def tanl_score_with_pool(lg: torch.Tensor, n_id: int, step: int, pool_mass: Optional[torch.Tensor],
                         frac: float = 1.0) -> torch.Tensor:
    """TANL's activation-aware score (closed form, as oodlab.methods.scoring.activation_aware_score) with an extra
    negative mass ``pool_mass`` (B,) added to every term. lg: (B, C+M) scaled logits, negatives sorted."""
    n_neg = lg.shape[1] - n_id
    a = torch.logsumexp(lg[:, :n_id], dim=1, keepdim=True)
    if n_neg <= 0:
        neg = pool_mass[:, None] if pool_mass is not None else None
        return (torch.ones(lg.shape[0], dtype=lg.dtype, device=lg.device) if neg is None
                else torch.sigmoid(a - neg).squeeze(1))
    if step == 0:
        s = torch.logsumexp(lg[:, n_id:], dim=1, keepdim=True)
        if pool_mass is not None:
            s = torch.logaddexp(s, pool_mass[:, None])
        return torch.sigmoid(a - s).squeeze(1)
    cum = torch.logcumsumexp(lg[:, n_id:], dim=1)
    ms = [min(j + step, n_neg) for j in range(0, n_neg, step)]
    sel = cum.index_select(1, torch.tensor([m - 1 for m in ms], device=lg.device))
    if pool_mass is not None and frac > 0:
        k0 = int(round((1 - frac) * sel.shape[1]))         # only the terms over the most negatives count the pool
        sel = torch.cat([sel[:, :k0], torch.logaddexp(sel[:, k0:], pool_mass[:, None])], 1)
    return torch.sigmoid(a - sel).mean(dim=1)


def logit(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.double().clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


class ReNegTANL(TANL):
    name = "reneg_tanl"

    def __init__(self, id_text: torch.Tensor, corpus_text: torch.Tensor, noise_feats: torch.Tensor,
                 pool_text: Optional[torch.Tensor] = None, pool_cluster=None, cfg=None,
                 rcfg: Optional[ReNegConfig] = None, device: str = "cpu", n_neglabel: Optional[int] = None,
                 pool_prior=None, reneg_device: Optional[str] = None, id_prior=None):
        """id_prior: optional (feats, labels) of a few labelled ID images (e.g. OpenOOD's ImageNet ID-val split,
        about 5 per class); they enter the online ID image model at every reset (the few-shot setting of TINS)."""
        if TANLConfig is None:
            raise ImportError("oodlab is not importable: add the oodlab code folder to sys.path")
        cfg = (cfg or TANLConfig()).with_(record=True)
        self.rcfg = rcfg or ReNegConfig()
        super().__init__(id_text, corpus_text, noise_feats, cfg=cfg, device=device, n_neglabel=n_neglabel)
        dt = self.rcfg.torch_dtype
        self.rdev = torch.device(reneg_device if reneg_device is not None else device)
        self.T_id_r = l2n(torch.as_tensor(id_text).to(device=self.rdev, dtype=dt))
        self.gate = None
        if pool_text is not None and self.rcfg.use_pool and len(pool_text):
            self.gate = KGGate(pool_text, pool_cluster, self.rcfg, pool_prior, device=self.rdev)
        self.idm = OnlineIdModel(self.T_id_r.shape[0], self.T_id_r.shape[1], self.rcfg, device=self.rdev)
        self.id_prior = None
        if id_prior is not None:
            Xp, yp = id_prior
            self.id_prior = (l2n(torch.as_tensor(Xp).to(device=self.rdev, dtype=dt)),
                             torch.as_tensor(np.asarray(yp)).long().to(self.rdev))
        self.votes = VoteHistory()
        self.reset()

    def reset(self) -> None:
        super().reset()
        if getattr(self, "gate", None) is not None:
            self.gate.reset()
        if getattr(self, "idm", None) is not None:
            self.idm.reset()
            if getattr(self, "id_prior", None) is not None:
                Xp, yp = self.id_prior
                self.idm.add(Xp, yp, np.ones(len(Xp), bool))
        self.votes = VoteHistory()

    def config_dict(self):
        from dataclasses import asdict
        return {"tanl": asdict(self.cfg), "reneg": asdict(self.rcfg)}

    @torch.no_grad()
    def step(self, feats: torch.Tensor):
        rc = self.rcfg
        out = super().step(feats)                                   # TANL's own dynamics, unchanged
        sel = out.info["sel"].to(self.all_T.device).long()
        X = l2n(feats.to(self.all_T.device).float())
        lg = self.logits(X, self.selected_T(sel)).float().to(self.rdev)   # (B, C+M) as TANL scores them
        Xr = X.to(device=self.rdev, dtype=rc.torch_dtype)
        P = None
        if self.gate is not None and rc.pool_in_text:
            Tp, lw = self.gate.text_terms()
            if Tp.shape[0]:
                P = torch.logsumexp(rc.tau * Xr @ Tp.T + lw[None, :], 1).float()
        conf = tanl_score_with_pool(lg, self.C, int(self.cfg.step), P, rc.pool_frac)
        lt = logit(conf).to(rc.torch_dtype)
        li = image_score(Xr, self.idm, self.gate, rc) if rc.use_image else None
        s = fuse(lt, li, rc)
        # updates (score-then-update)
        lab, adm = admit_mask(Xr, self.T_id_r, rc, base_mask=out.info["id_mask"].cpu().numpy())
        if self.gate is not None:
            vote = self.votes.vote(logit(out.score).cpu().numpy()) if rc.vote else None   # TANL's own score
            caught_by = self.gate.update(Xr, self.T_id_r, ood_vote=vote)
            adm = adm & ~self.gate.exclusion(caught_by, rc.id_exclude)
        self.idm.add(Xr, lab, adm)
        info = {"tanl_score": out.score.cpu(), "thr": out.info.get("thr")}
        return StepOutput(pred=out.pred, score=s.float().to(out.score.device), info=info)
