"""DDE (Zhu et al., ECCV 2026) OOD score on cached CLIP features: a base detector for ReNeg's head.

Re-implemented op for op from the official postprocessor (github.com/PolyU-VCLab/OpenOOD-VLM, commit 48b0233,
``openood/postprocessors/dde_postprocessor.py`` with ``scripts/ood/dde/imagenet.sh``; MIT licence), in float32:

* caches: the ID cache starts from the ID text features, the OOD cache from the 15 noise-image features; both are
  first-in-first-out (5,000 entries in the release, 1,000 in the paper);
* per batch: the relevance of a negative label = its mean softmax mass over the OOD cache minus over the ID cache;
  the ``hat_M`` most relevant labels give a provisional score; confident images (score > 0.75, or < 0.10 in the
  release and < 0.25 in the paper) refine
  the relevance (0.95 cache + 0.05 batch); the ``hat_M`` most relevant labels then give the final score, the ID
  mass in [ID | selected negatives].

DDE's Gaussian branch (PFDE) only changes the predicted class, not the OOD score, and is left out.

Which configuration reproduces the paper (FPR95, OOD positive, stream order 0, SSB-hard / NINCO; published
61.02 / 56.26):

* the release as it is (``DDEConfig.release()`` with the 10,000 NegLabel negatives as ``neg_text`` and
  ``duplicate_negatives=True``: the released network class sets ``text_features_unselected = text_features_neg``,
  so the selection bank is the 10,000 negatives twice): NINCO 66.20;
* the whole 69,554-word corpus as the selection bank with the release's thresholds and caches: 65.68 / 56.74;
* the whole corpus with the settings stated in the paper (Sec. 4.1: lambda_pos 0.75, lambda_neg 0.25, queue
  length 1,000, hat_M 500), ``DDEConfig.paper()``: 62.15 / 55.88. This is the base used for ReNeg's head;
* the 10,000-negative bank with the paper's settings: 63.51 / 59.42 (duplicated), 65.94 / 58.53 (not duplicated).

Cache means are kept as running sums: each cached feature's softmax row is fixed, so this equals recomputing the
means over the caches at every batch as the official code does (``selfcheck``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

try:                                                            # the real oodlab (Colab / Kaggle / local copy)
    from oodlab.methods.base import StepOutput, StreamMethod
except Exception:                                               # noqa: BLE001
    StepOutput, StreamMethod = None, object


@dataclass
class DDEConfig:
    hat_M: int = 500
    thres: float = 0.5
    gap: float = 0.5                    # ID activation: score > thres + gap * (1 - thres) = 0.75
    ood_thres: float = 0.10             # hard-coded in the official postprocess()
    cache_size: int = 5000              # official: getattr(self, "id_cache_size", 5000)
    logit_scale: float = 100.0
    duplicate_negatives: bool = False   # True: the released file's bank [ID | negatives | negatives]

    @staticmethod
    def release() -> "DDEConfig":
        """The values in the released postprocessor and ``scripts/ood/dde/imagenet.sh`` (the field defaults)."""
        return DDEConfig()

    @staticmethod
    def paper() -> "DDEConfig":
        """The values stated in the paper (Sec. 4.1): lambda_neg 0.25 and queue length 1,000."""
        return DDEConfig(ood_thres=0.25, cache_size=1000)

    def with_(self, **kw) -> "DDEConfig":
        return replace(self, **kw)


class DDEFeat(StreamMethod):
    name = "dde"

    def __init__(self, id_text: torch.Tensor, neg_text: torch.Tensor, noise_feats: torch.Tensor,
                 cfg: Optional[DDEConfig] = None, device: str = "cpu"):
        self.cfg = cfg or DDEConfig()
        self.device = device
        self.T_id = torch.as_tensor(id_text).float().to(device)             # (C, D)
        self.noise = torch.as_tensor(noise_feats).float().to(device)
        self.C = self.T_id.shape[0]
        self.T = torch.cat([self.T_id, torch.as_tensor(neg_text).float().to(device)]).t().contiguous()   # (D, C + M)
        self.reset()

    def _all_logits(self, f: torch.Tensor) -> torch.Tensor:
        lg = self.cfg.logit_scale * (f @ self.T)
        if self.cfg.duplicate_negatives:
            lg = torch.cat([lg, lg[:, self.C:]], 1)
        return lg

    def _neg_rows(self, f: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._all_logits(f), 1)[:, self.C:]

    @staticmethod
    def _sum64(rows: torch.Tensor) -> torch.Tensor:
        return rows.sum(0, dtype=torch.float64)

    def reset(self) -> None:
        self.id_cache = self.T_id.clone()
        self.ood_cache = self.noise.clone()
        self.id_sum = self._sum64(self._neg_rows(self.id_cache))
        self.ood_sum = self._sum64(self._neg_rows(self.ood_cache))

    def _push(self, which: str, feats: torch.Tensor, rows: torch.Tensor) -> None:
        cache = torch.cat([getattr(self, which + "_cache"), feats])
        s = getattr(self, which + "_sum") + self._sum64(rows)
        over = cache.shape[0] - self.cfg.cache_size
        if over > 0:
            s = s - self._sum64(self._neg_rows(cache[:over]))
            cache = cache[over:]
        setattr(self, which + "_cache", cache)
        setattr(self, which + "_sum", s)

    def _score(self, lg: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        idx = relevance.sort(descending=True)[1][: self.cfg.hat_M]
        sel = torch.cat([lg[:, : self.C], lg[:, self.C:][:, idx]], 1)
        return torch.softmax(sel, 1)[:, : self.C].sum(-1)

    @torch.no_grad()
    def step(self, feats: torch.Tensor):
        c = self.cfg
        f = feats.to(self.device).float()
        lg = self._all_logits(f)
        act = torch.softmax(lg, 1)[:, self.C:]
        m_id = self.id_sum / self.id_cache.shape[0]
        m_ood = self.ood_sum / self.ood_cache.shape[0]
        conf0 = self._score(lg, m_ood - m_id)
        a_id = conf0 > (c.thres + c.gap * (1 - c.thres))
        a_ood = conf0 < c.ood_thres
        r_id, r_ood = act[a_id], act[a_ood]
        pos = 0.95 * m_id + 0.05 * self._sum64(r_id) / len(r_id) if len(r_id) else m_id
        neg = 0.95 * m_ood + 0.05 * self._sum64(r_ood) / len(r_ood) if len(r_ood) else m_ood
        conf = self._score(lg, neg - pos)
        if len(r_id):
            self._push("id", f[a_id], r_id)
        if len(r_ood):
            self._push("ood", f[a_ood], r_ood)
        return StepOutput(pred=lg[:, : self.C].argmax(1), score=conf,
                          info={"n_id_admit": int(a_id.sum()), "n_ood_admit": int(a_ood.sum())})


def selfcheck(n: int = 1200, seed: int = 0, duplicate: bool = True):
    """Max |score difference| between DDEFeat (running sums) and the official recomputation over the caches, on
    random features with a small cache so that entries are evicted. Returns (difference, ID admitted, OOD admitted)."""
    g = torch.Generator().manual_seed(seed)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True)                                      # noqa: E731
    C = 30
    idt, neg, noise = (l2(torch.randn(k, 16, generator=g)) for k in (C, 200, 15))
    X = l2(torch.randn(n, 16, generator=g) + 2 * idt[torch.randint(0, C, (n,), generator=g)])
    cfg = DDEConfig(hat_M=20, cache_size=64, logit_scale=20.0, thres=0.5, gap=0.2, ood_thres=0.45,
                    duplicate_negatives=duplicate)
    m = DDEFeat(idt, neg, noise, cfg)
    T_all = torch.cat([idt, neg, neg] if duplicate else [idt, neg]).t()
    idc, oodc = idt.clone(), noise.clone()
    worst, n_id, n_ood = 0.0, 0, 0

    def score(lg, rel):
        sel = torch.cat([lg[:, :C], lg[:, C:][:, rel.sort(descending=True)[1][: cfg.hat_M]]], 1)
        return torch.softmax(sel, 1)[:, :C].sum(-1)

    for s in range(0, n, 32):
        f = X[s:s + 32]
        out = m.step(f)
        lg = cfg.logit_scale * f @ T_all
        act = torch.softmax(lg, 1)
        sid = torch.softmax(cfg.logit_scale * idc @ T_all, 1)[:, C:]
        sood = torch.softmax(cfg.logit_scale * oodc @ T_all, 1)[:, C:]
        c0 = score(lg, sood.mean(0) - sid.mean(0))
        ai, ao = c0 > cfg.thres + cfg.gap * (1 - cfg.thres), c0 < cfg.ood_thres
        p = 0.95 * sid.mean(0) + 0.05 * act[ai][:, C:].mean(0) if ai.any() else sid.mean(0)
        q = 0.95 * sood.mean(0) + 0.05 * act[ao][:, C:].mean(0) if ao.any() else sood.mean(0)
        ref = score(lg, q - p)
        if ai.any():
            idc = torch.cat([idc, f[ai]])[-cfg.cache_size:]
        if ao.any():
            oodc = torch.cat([oodc, f[ao]])[-cfg.cache_size:]
        worst = max(worst, float((ref - out.score).abs().max()))
        n_id += int(ai.sum())
        n_ood += int(ao.sum())
    return worst, n_id, n_ood
