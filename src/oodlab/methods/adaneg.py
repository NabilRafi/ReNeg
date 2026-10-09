"""AdaNeg (Zhang & Zhang, NeurIPS 2024) on cached features.

Faithful to ``TTAPromptPostprocessor_noadagap`` (postprocessor ``ttapromptnoadagap``),
which ``scripts/ood/adaneg/imagenet.sh`` runs with:
memleng=10, thres=0.5, gap=0.5, lambdaval=0.1, group_num=5, random_permute=True,
in_score=combine, samada=False.

A note that matters for reproduction
-----------------------------------
The official code computes the per-sample entropy that decides memory replacement
as ``-(p * log(p + 1e-8)).sum()`` on **fp16** probabilities. In fp16, most of the
1,000 (or 10,000) probabilities underflow to 0, ``1e-8`` also rounds to 0, and
``0 * log(0)`` is NaN - so every entropy is NaN and the "replace the highest-entropy
slot" rule never fires. The official ImageNet numbers therefore come from a memory
that keeps the *first* confident features of each class.

* ``emulate_fp16=True``  runs the official fp16 arithmetic op-for-op (NaN entropies included).
* ``emulate_fp16=False`` runs in float32, i.e. the entropy rule described in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from .base import StepOutput, StreamMethod
from .scoring import (
    entropy,
    neglabel_group_permutation,
    neglabel_grouped_score,
    neglabel_grouped_score_fp16,
    scaled_logits,
    scaled_logits_half,
)


@dataclass
class AdaNegConfig:
    memory_len: int = 10      # official "memleng"
    thres: float = 0.5
    gap: float = 0.5
    lambda_: float = 0.1      # official "lambdaval"
    group_num: int = 5
    random_permute: bool = True
    perm_seed: int = 0
    in_score: str = "combine"  # combine | adaonly | vanillaonly | multiply
    logit_scale: float = 100.0
    emulate_fp16: bool = False
    proxy_chunk: Optional[int] = None  # fp16 path: None = one product like the official code (~3 GB on GPU
                                       # for 11k labels x batch 256); set e.g. 2048 to save memory

    @staticmethod
    def official() -> "AdaNegConfig":
        return AdaNegConfig()

    def with_(self, **kw) -> "AdaNegConfig":
        return replace(self, **kw)


class AdaNeg(StreamMethod):
    name = "adaneg"

    def __init__(self, id_text: torch.Tensor, neg_text: torch.Tensor, cfg: Optional[AdaNegConfig] = None, device: str = "cpu"):
        self.cfg = cfg or AdaNegConfig()
        self.device = device
        self.dt = torch.float16 if self.cfg.emulate_fp16 else torch.float32
        text = torch.cat([id_text, neg_text], dim=0).to(device=device, dtype=self.dt)      # (K, D)
        self.text_T = text.t().contiguous()   # official layout (D, K): logit_scale * image_features @ text_features
        self.C = id_text.shape[0]
        self.K = text.shape[0]
        self.perm = neglabel_group_permutation(self.K - self.C, self.cfg.group_num, self.cfg.random_permute, self.cfg.perm_seed)
        self.reset()

    # ------------------------------------------------------------------ memory
    def reset(self) -> None:
        L = self.cfg.memory_len
        D, K = self.text_T.shape
        self.mem = torch.zeros(K, 1 + L, D, device=self.device, dtype=self.dt)
        self.mem[:, 0] = self.text_T.t()     # slot 0 holds the label's text feature
        self.fill = [1] * K  # official "indice_memory" (starts at 1: slot 0 holds the text feature)
        self.ent = torch.zeros(K, 1 + L, dtype=self.dt)  # official "entropy_memory" (kept on CPU)

    def _insert(self, c: int, feat: torch.Tensor, e: torch.Tensor) -> None:
        L = self.cfg.memory_len
        if self.fill[c] == L:
            if (e < self.ent[c]).sum() == 0:
                return  # also taken when e is NaN (fp16) -> memory is never replaced
            _, order = torch.sort(self.ent[c])
            j = int(order[-1])
            self.mem[c, j] = feat
            self.ent[c, j] = e
        else:
            j = self.fill[c]
            self.mem[c, j] = feat
            self.ent[c, j] = e
            self.fill[c] += 1

    def _update_memory(self, f, out_v, conf_v):
        cfg, C = self.cfg, self.C
        if cfg.emulate_fp16:
            ent_fn = lambda lg: -(torch.softmax(lg, dim=1) * torch.log(torch.softmax(lg, dim=1) + 1e-8)).sum(dim=-1)
        else:
            ent_fn = lambda lg: entropy(torch.softmax(lg, dim=1))
        id_mask = (conf_v > (cfg.thres + cfg.gap * (1 - cfg.thres))).cpu()
        pred_id = out_v[:, :C].argmax(dim=1).cpu()
        ent_id = ent_fn(out_v[:, :C]).cpu()
        for i in torch.nonzero(id_mask).flatten().tolist():
            self._insert(int(pred_id[i]), f[i], ent_id[i])
        ood_mask = (conf_v < (cfg.thres - cfg.gap * cfg.thres)).cpu()
        pred_neg = (out_v[:, C:].argmax(dim=1) + C).cpu()
        ent_neg = ent_fn(out_v[:, C:]).cpu()
        for i in torch.nonzero(ood_mask).flatten().tolist():
            self._insert(int(pred_neg[i]), f[i], ent_neg[i])
        return int(id_mask.sum()), int(ood_mask.sum())

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, feats: torch.Tensor) -> StepOutput:
        return self._step_fp16(feats) if self.cfg.emulate_fp16 else self._step_fp32(feats)

    def _combine(self, conf_in, conf_v):
        cfg = self.cfg
        if cfg.in_score == "combine":
            return conf_in + conf_v * cfg.lambda_
        if cfg.in_score == "adaonly":
            return conf_in
        if cfg.in_score == "vanillaonly":
            return conf_v
        if cfg.in_score == "multiply":
            return conf_in * conf_v
        raise ValueError(cfg.in_score)

    def _step_fp32(self, feats):
        cfg, C = self.cfg, self.C
        f = feats.to(device=self.device, dtype=torch.float32)
        out_v = scaled_logits(f, self.text_T, cfg.logit_scale)
        conf_v = neglabel_grouped_score(out_v, C, cfg.group_num, self.perm)
        n_id, n_ood = self._update_memory(f, out_v, conf_v)
        proxies = self.mem.mean(dim=1)
        proxies = proxies / proxies.norm(dim=-1, keepdim=True)
        out = scaled_logits(f, proxies.t(), cfg.logit_scale)   # = official elementwise product-sum, as a matmul
        prob_all = torch.softmax(out_v, dim=1) + cfg.lambda_ * torch.softmax(out, dim=1)
        pred = prob_all[:, :C].argmax(dim=1)
        conf = self._combine(neglabel_grouped_score(out, C, cfg.group_num, self.perm), conf_v)
        return StepOutput(pred=pred, score=conf, info={"n_id_admit": n_id, "n_ood_admit": n_ood})

    def _step_fp16(self, feats):
        """Op-for-op replica of the official fp16 computation."""
        cfg, C = self.cfg, self.C
        f = feats.to(device=self.device, dtype=torch.float16)
        out_v = scaled_logits_half(f, self.text_T, cfg.logit_scale)          # fp16
        prob_v = torch.softmax(out_v, dim=1)
        conf_v = neglabel_grouped_score_fp16(out_v, C, cfg.group_num, self.perm)
        n_id, n_ood = self._update_memory(f, out_v, conf_v)
        proxies = self.mem.mean(1)
        proxies /= proxies.norm(dim=-1, keepdim=True)
        # official: logit_scale * (image_features.unsqueeze(1) * sa_text_features).sum(-1), chunked over K
        s = torch.tensor(cfg.logit_scale, dtype=torch.float32, device=f.device)
        if cfg.proxy_chunk is None:
            out = s * (f.unsqueeze(1) * proxies).sum(-1)                        # fp16, exactly as official
        else:
            out = torch.cat([s * (f.unsqueeze(1) * proxies[k0 : k0 + cfg.proxy_chunk]).sum(-1)
                             for k0 in range(0, self.K, cfg.proxy_chunk)], dim=1)
        prob_tta = torch.softmax(out, dim=1)
        prob_all = prob_v + prob_tta * cfg.lambda_
        pred = prob_all[:, :C].argmax(dim=1)
        conf_in = neglabel_grouped_score_fp16(out, C, cfg.group_num, self.perm)
        conf = self._combine(conf_in, conf_v)
        return StepOutput(pred=pred, score=conf.float(), info={"n_id_admit": n_id, "n_ood_admit": n_ood})
