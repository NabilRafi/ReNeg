"""NegLabel (Jiang et al., ICLR 2024) and MCM (Ming et al., NeurIPS 2022) on cached features."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from .base import StepOutput, StreamMethod
from .scoring import (
    neglabel_group_permutation,
    neglabel_grouped_score,
    neglabel_grouped_score_fp16,
    scaled_logits,
    scaled_logits_half,
)


@dataclass
class NegLabelConfig:
    group_num: int = 100        # NegLabel's default grouping; the repo also runs 10 and 1
    random_permute: bool = True
    perm_seed: int = 0
    logit_scale: float = 100.0
    emulate_fp16: bool = False

    def with_(self, **kw) -> "NegLabelConfig":
        return replace(self, **kw)


class NegLabel(StreamMethod):
    """Static negative labels (the 10,000 mined by NegMining) + grouped score."""

    name = "neglabel"
    stateless = True

    def __init__(self, id_text: torch.Tensor, neg_text: torch.Tensor, cfg: Optional[NegLabelConfig] = None, device: str = "cpu"):
        self.cfg = cfg or NegLabelConfig()
        self.device = device
        dt = torch.float16 if self.cfg.emulate_fp16 else torch.float32
        # official layout (D, K): logits = logit_scale * image_features @ text_features
        self.text_T = torch.cat([id_text, neg_text], dim=0).to(device=device, dtype=dt).t().contiguous()
        self.C = id_text.shape[0]
        self.perm = neglabel_group_permutation(self.text_T.shape[1] - self.C, self.cfg.group_num, self.cfg.random_permute, self.cfg.perm_seed)

    @torch.no_grad()
    def step(self, feats: torch.Tensor) -> StepOutput:
        if self.cfg.emulate_fp16:  # op-for-op official fp16 path
            lg = scaled_logits_half(feats.to(self.device), self.text_T, self.cfg.logit_scale)
            conf = neglabel_grouped_score_fp16(lg, self.C, self.cfg.group_num, self.perm).float()
        else:
            lg = scaled_logits(feats.to(self.device), self.text_T, self.cfg.logit_scale)
            conf = neglabel_grouped_score(lg, self.C, self.cfg.group_num, self.perm)
        pred = lg[:, : self.C].argmax(dim=1)
        return StepOutput(pred=pred, score=conf)


@dataclass
class MCMConfig:
    temperature: float = 1.0       # MCM paper: softmax(cosine / T) with T = 1
    use_logit_scale: bool = False  # True -> softmax(logit_scale * cosine / T), as OpenOOD-VLM's "mcm"
    logit_scale: float = 100.0
    append_neg_mean: bool = False  # the "fixedclip_oodprompt" net appends a (-mean ID text) column;
                                   # the MCM script uses "fixedclip", which does not

    @staticmethod
    def paper() -> "MCMConfig":
        return MCMConfig()

    @staticmethod
    def openood_vlm(tau: float = 1.0, logit_scale: float = 100.0) -> "MCMConfig":
        """scripts/ood/mcm/official.sh: softmax(logit_scale * cos / tau) over the 1,000 ID labels.
        The script also turns on APS (tau searched on the OpenOOD validation split);
        see ``tune_mcm_tau`` in the runner notebook."""
        return MCMConfig(temperature=tau, use_logit_scale=True, logit_scale=logit_scale)

    def with_(self, **kw) -> "MCMConfig":
        return replace(self, **kw)


class MCM(StreamMethod):
    """Maximum Concept Matching: max softmax over ID labels only."""

    name = "mcm"
    stateless = True

    def __init__(self, id_text: torch.Tensor, cfg: Optional[MCMConfig] = None, device: str = "cpu"):
        self.cfg = cfg or MCMConfig()
        self.device = device
        text = id_text.to(device=device, dtype=torch.float32)
        if self.cfg.append_neg_mean:
            m = -text.mean(0)
            text = torch.cat([text, (m / m.norm()).unsqueeze(0)], dim=0)
        self.text_T = text.t().contiguous()
        self.C = id_text.shape[0]

    @torch.no_grad()
    def step(self, feats: torch.Tensor) -> StepOutput:
        cos = feats.to(self.device).float() @ self.text_T
        if self.cfg.use_logit_scale:
            cos = cos * self.cfg.logit_scale
        p = torch.softmax(cos / self.cfg.temperature, dim=1)
        conf, pred = p.max(dim=1)
        return StepOutput(pred=pred, score=conf)
