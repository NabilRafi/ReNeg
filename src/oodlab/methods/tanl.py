"""TANL - Test-time Activated Negative Labels (Zhang et al., CVPR 2026), on cached features.

This is a faithful re-implementation of ``ActivatedNegPostprocessor.postprocess``
from the official repository (YBZh/OpenOOD-VLM), restructured so each step is a
method you can override:

    reset()                  -> initial queues from ID label text + noise images
    activation(feats)        -> softmax over [ID | corpus], corpus columns (Eq. 5)
    select(combined)         -> which corpus labels become negatives (top-M, Eq. 6)
    score(feats, selected)   -> activation-aware score (Eq. 15)
    threshold()              -> automatic (Otsu-style) threshold over recent scores
    admit(conf, thr)         -> which samples enter the ID / OOD queues (Eq. 9)

``tests/test_equivalence.py`` runs the vendored official class and this class on the
same stream and checks the scores agree (1e-6 in fp32, bit-identical in fp16).

Text features are kept in the official *column* layout (D x K, like ``net.text_features``)
and every product is written exactly like the official expression, so the same BLAS
kernels run in the fp16 emulation on GPU too.

Official hyper-parameter names -> ours:
    beta -> num_neg (M) | memleng -> queue_len (L) | gap -> gap (g)
    alpha -> alpha      | gamma -> step (0 = plain NegLabel score, 1 = Eq. 15)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from .base import StepOutput, StreamMethod
from .scoring import (
    activation_aware_score,
    activation_aware_score_fp16_loop,
    find_best_threshold,
    find_best_threshold_fast,
    scaled_logits,
    scaled_logits_half,
)


@dataclass
class TANLConfig:
    num_neg: int = 1000          # M: negatives used for scoring (official "beta")
    queue_len: int = 300         # L: FIFO length of the ID / OOD activation queues ("memleng")
    gap: float = 0.2             # g: confidence gap for admitting samples to the queues
    alpha: float = 0.95          # weight of queue history vs the current batch (Eq. 13-14)
    step: int = 1                # official "gamma": 1 = activation-aware score, 0 = plain NegLabel score
    score_queue_len: int = 20000 # recent scores used by the automatic threshold
    auto_threshold: bool = True  # Otsu-style threshold (as in the paper/code); False = fixed_threshold (ablation)
    fixed_threshold: float = 0.5
    logit_scale: float = 100.0   # use TextBank.logit_scale (= clip_model.logit_scale.exp())
    emulate_fp16: bool = False   # imitate the official fp16 arithmetic (logits + final score)
    exact_fp16_final: bool = False  # with emulate_fp16: run the official 1000-softmax loop for the
                                    # fp16 scores (bit-exact but slower; the dynamics are exact either way)
    init_seed: int = 0           # seeds the two row permutations made at reset
    threshold_impl: str = "official"  # "official" (verbatim loop) or "fast" (vectorised)
    record: bool = False         # put selected indices / masks into StepOutput.info

    @staticmethod
    def paper() -> "TANLConfig":
        """Hyper-parameters stated in the paper (Sec. 4.1): M=1000, g=0.2, L=300, alpha=0.95."""
        return TANLConfig()

    @staticmethod
    def official_sh() -> "TANLConfig":
        """What scripts/ood/TANL/official.sh runs at commit c6fef2f (alpha=0.0, gap=0.5)."""
        return TANLConfig(alpha=0.0, gap=0.5)

    def with_(self, **kw) -> "TANLConfig":
        return replace(self, **kw)


class TANL(StreamMethod):
    name = "tanl"

    def __init__(
        self,
        id_text: torch.Tensor,
        corpus_text: torch.Tensor,
        noise_feats: torch.Tensor,
        cfg: Optional[TANLConfig] = None,
        device: str = "cpu",
        n_neglabel: Optional[int] = None,
    ):
        """``id_text`` (C, D), ``corpus_text`` (N, D) in the official corpus order, ``noise_feats`` (15, D).

        ``n_neglabel``: if given (10,000 for the real text bank), the ID prediction is taken from
        the [ID | first n_neglabel corpus rows] product exactly like the official code; otherwise
        from the [ID | corpus] product (identical except for rare fp16 ties).
        """
        self.cfg = cfg or TANLConfig()
        self.device = device
        store = torch.float16 if self.cfg.emulate_fp16 else torch.float32
        id_text = id_text.to(device=device, dtype=store)
        corpus = corpus_text.to(device=device, dtype=store)
        self.noise = noise_feats.to(device=device, dtype=store)
        self.C = int(id_text.shape[0])
        self.N = int(corpus.shape[0])
        if self.cfg.num_neg > self.N:
            raise ValueError(f"num_neg={self.cfg.num_neg} > corpus size {self.N}")
        # official layout: net.text_features_all is (D, C+N)
        self.all_T = torch.cat([id_text, corpus], dim=0).t().contiguous()
        self.pred_T = (torch.cat([id_text, corpus[:n_neglabel]], dim=0).t().contiguous()
                       if n_neglabel else None)
        del id_text, corpus
        self.reset()

    # ------------------------------------------------------------------ helpers
    @property
    def id_rows(self) -> torch.Tensor:
        """(C, D) view of the ID text features (official: ``text_features_all[:, :C].t()``)."""
        return self.all_T[:, : self.C].t()

    def logits(self, feats: torch.Tensor, text_T: torch.Tensor) -> torch.Tensor:
        return scaled_logits(feats, text_T, self.cfg.logit_scale, self.cfg.emulate_fp16)

    def selected_T(self, selected: torch.Tensor) -> torch.Tensor:
        """(D, C+M) text matrix [ID | selected negatives], built like the official code."""
        return torch.cat([self.all_T[:, : self.C], self.all_T[:, self.C :].t()[selected].t()], dim=-1)

    def activation(self, feats: torch.Tensor, chunk: int = 512, fp16_softmax: bool = False) -> torch.Tensor:
        """Per-sample activation of every corpus label: softmax over [ID | corpus] (Eq. 5).

        ``fp16_softmax`` reproduces the official *initialisation*, where the softmax is
        taken on fp16 logits (the per-batch activations use a float32 softmax).
        """
        outs = []
        for s in range(0, feats.shape[0], chunk):
            lg = self.logits(feats[s : s + chunk], self.all_T)
            p = torch.softmax(lg.to(torch.float16), dim=1).float() if fp16_softmax else torch.softmax(lg, dim=1)
            outs.append(p[:, self.C :])
        return torch.cat(outs, dim=0)

    def select(self, combined: torch.Tensor) -> torch.Tensor:
        """Indices (into the corpus) of the negatives to use, strongest first.

        Official: ``combined.sort(descending=True)[1][:beta]``. Override this to change
        the selection rule (e.g. an LCB rule, see variants.py) without touching anything else.
        """
        return combined.sort(descending=True)[1][: self.cfg.num_neg]

    def score(self, feats: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """Activation-aware score in float32 (closed form of the official loop)."""
        return activation_aware_score(self.logits(feats, self.selected_T(selected)), self.C, self.cfg.step)

    def score_fp16(self, feats: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """The official fp16 score: fp16 logits, then either the exact 1000-softmax loop or
        the closed form rounded to fp16. Returns float32 values of fp16 numbers."""
        text_T = self.selected_T(selected)
        if self.cfg.exact_fp16_final:
            lg = scaled_logits_half(feats, text_T, self.cfg.logit_scale)
            return activation_aware_score_fp16_loop(lg, self.C, self.cfg.step).float()
        return activation_aware_score(self.logits(feats, text_T), self.C, self.cfg.step).half().float()

    def threshold(self) -> float:
        if not self.cfg.auto_threshold:
            return float(self.cfg.fixed_threshold)
        if self.cfg.threshold_impl == "fast":
            return find_best_threshold_fast(self.score_q)
        return find_best_threshold(self.score_q)

    def admit(self, conf: torch.Tensor, thr: float):
        g = self.cfg.gap
        id_mask = conf > (thr + g * (1 - thr))
        ood_mask = conf < (thr - g * thr)
        return id_mask, ood_mask

    @staticmethod
    def _push(queue: torch.Tensor, rows: torch.Tensor, max_len: int) -> torch.Tensor:
        q = torch.cat([queue, rows], dim=0)
        return q[-max_len:] if q.shape[0] > max_len else q

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        cfg = self.cfg
        gen = torch.Generator().manual_seed(cfg.init_seed)
        # ID queue <- activations of the ID *label text* features (official init). One product for all
        # C rows, like the official code (chunking could change GPU kernels and hence fp16 rounding).
        pos = self.activation(self.id_rows, chunk=max(self.C, 1), fp16_softmax=cfg.emulate_fp16)
        pos = pos[torch.randperm(pos.shape[0], generator=gen).to(pos.device)]
        self.pos_q = pos[-cfg.queue_len :] if pos.shape[0] > cfg.queue_len else pos
        # OOD queue <- activations of 15 noise images.
        neg = self.activation(self.noise, chunk=max(int(self.noise.shape[0]), 1), fp16_softmax=cfg.emulate_fp16)
        neg = neg[torch.randperm(neg.shape[0], generator=gen).to(neg.device)]
        self.neg_q = neg[-cfg.queue_len :] if neg.shape[0] > cfg.queue_len else neg
        # Initial selection and the score queue used by the automatic threshold
        # (official: computed on fp16 logits without .float() -> fp16 scores).
        combined = self.neg_q.mean(0) - self.pos_q.mean(0)
        sel = self.select(combined)
        if cfg.emulate_fp16:
            conf_id, conf_noise = self.score_fp16(self.id_rows, sel), self.score_fp16(self.noise, sel)
        else:
            conf_id, conf_noise = self.score(self.id_rows, sel), self.score(self.noise, sel)
        self.score_q = torch.cat([conf_id, conf_noise], dim=0).float()
        self.init_selection = sel.detach().clone()
        self.n_steps = 0

    @torch.no_grad()
    def step(self, feats: torch.Tensor) -> StepOutput:
        cfg = self.cfg
        feats = feats.to(self.device)
        lg_all = self.logits(feats, self.all_T)                    # official output_all (B, C+N)
        if self.pred_T is not None:                                # official output (B, C+10000)
            pred = self.logits(feats, self.pred_T)[:, : self.C].argmax(dim=1)
        else:
            pred = lg_all[:, : self.C].argmax(dim=1)
        act = torch.softmax(lg_all, dim=1)[:, self.C :]             # official: softmax(output_all.float())
        del lg_all

        base_pos = self.pos_q.mean(0)
        base_neg = self.neg_q.mean(0)
        # 1) score with the selection implied by the queues only -> threshold + admission
        #    (official: output.float() then the loop in float32 -> our closed form)
        sel0 = self.select(base_neg - base_pos)
        conf0 = self.score(feats, sel0).float()
        self.score_q = torch.cat([self.score_q, conf0], dim=0)
        if self.score_q.shape[0] > cfg.score_queue_len:
            self.score_q = self.score_q[-cfg.score_queue_len :]
        thr = self.threshold()
        id_mask, ood_mask = self.admit(conf0, thr)

        # 2) batch-adaptive activations (Eq. 12-14) -> final selection and score
        pos_ref = cfg.alpha * base_pos + (1 - cfg.alpha) * act[id_mask].mean(0) if id_mask.any() else base_pos
        neg_ref = cfg.alpha * base_neg + (1 - cfg.alpha) * act[ood_mask].mean(0) if ood_mask.any() else base_neg
        sel = self.select(neg_ref - pos_ref)
        conf = self.score_fp16(feats, sel) if cfg.emulate_fp16 else self.score(feats, sel)

        # 3) update the FIFO queues with this batch's confident samples
        if id_mask.any():
            self.pos_q = self._push(self.pos_q, act[id_mask], cfg.queue_len)
        if ood_mask.any():
            self.neg_q = self._push(self.neg_q, act[ood_mask], cfg.queue_len)
        self.n_steps += 1

        info = {}
        if cfg.record:
            info = {
                "sel": sel.to("cpu", torch.int32),
                "sel0": sel0.to("cpu", torch.int32),
                "thr": float(thr),
                "id_mask": id_mask.cpu(),
                "ood_mask": ood_mask.cpu(),
                "conf0": conf0.cpu(),
            }
        return StepOutput(pred=pred, score=conf, info=info)
