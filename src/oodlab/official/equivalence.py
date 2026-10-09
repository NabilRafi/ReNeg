"""Run the vendored *official* postprocessors and our re-implementations on the same stream.

Use this to prove that a change to our code did not change behaviour, and (on
Kaggle) that our cached-feature pipeline reproduces the official algorithm on
real CLIP features.

    from oodlab.official.equivalence import compare_tanl
    report = compare_tanl(id_text, corpus, noise, stream_batches, n_neglabel=10000)
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import torch

from ..methods import AdaNeg, AdaNegConfig, NegLabel, NegLabelConfig, TANL, TANLConfig


class MockNet:
    """Just enough of OpenOOD-VLM's ``FixedCLIP_NegOODPrompt`` for the postprocessors.

    Images are passed as their (already L2-normalised) CLIP features, so
    ``net(features, return_feat=True)`` returns them unchanged.
    Official tensors store classes in *columns* (D x K); we mirror that.
    """

    def __init__(self, id_text, corpus, noise, n_neglabel: int, logit_scale: float = 100.0):
        self.n_cls = id_text.shape[0]
        self.text_features = torch.cat([id_text, corpus[:n_neglabel]], 0).t().contiguous()
        self.text_features_unselected = corpus[n_neglabel:].t().contiguous()
        self.text_features_all = torch.cat([id_text, corpus], 0).t().contiguous()
        self.noise_image_features = noise
        self.logit_scale = torch.tensor(logit_scale, dtype=torch.float32, device=id_text.device)
        self.n_output = self.text_features.shape[1]

    def eval(self):
        return self

    def __call__(self, data, return_feat: bool = False):
        if return_feat:
            return data, self.text_features.t(), self.logit_scale
        return self.logit_scale * data @ self.text_features


def _cfg(**args):
    return SimpleNamespace(postprocessor=SimpleNamespace(postprocessor_args=SimpleNamespace(**args), postprocessor_sweep={}))


def official_tanl(cfg: TANLConfig):
    from .tanl_official import ActivatedNegPostprocessor

    return ActivatedNegPostprocessor(
        _cfg(tau=1.0, beta=cfg.num_neg, in_score="sum", alpha=cfg.alpha, gamma=cfg.step, group_num=5,
             group_size=1000, random_permute=False, thres=0.5, samada=False, gap=cfg.gap, cluster_num=0,
             cossim=False, memleng=cfg.queue_len)
    )


def official_adaneg(cfg: AdaNegConfig):
    from .adaneg_official import TTAPromptPostprocessor_noadagap

    return TTAPromptPostprocessor_noadagap(
        _cfg(tau=1.0, beta=5.5, in_score=cfg.in_score, memleng=cfg.memory_len, lambdaval=cfg.lambda_,
             thres=cfg.thres, samada=False, gap=cfg.gap, group_num=cfg.group_num, random_permute=cfg.random_permute)
    )


def official_neglabel(cfg: NegLabelConfig):
    from .neglabel_official import OneOodPromptDevelopPostprocessor

    return OneOodPromptDevelopPostprocessor(
        _cfg(tau=1.0, beta=1.0, in_score="sum", group_num=cfg.group_num, random_permute=cfg.random_permute)
    )


def _run_official(post, net, batches: Iterable[torch.Tensor], seed: Optional[int]) -> List:
    if seed is not None:
        torch.manual_seed(seed)  # the official reset draws two permutations from the global RNG
    out = []
    with contextlib.redirect_stdout(io.StringIO()):  # the official code prints shapes
        for b in batches:
            pred, conf = post.postprocess(net, b)
            out.append((pred.cpu(), conf.float().cpu()))
    return out


def _run_ours(method, batches) -> List:
    method.reset()
    out = []
    for b in batches:
        o = method.step(b)
        out.append((o.pred.cpu(), o.score.float().cpu()))
    return out


def _compare(a: List, b: List) -> Dict[str, float]:
    ca = torch.cat([x[1] for x in a])
    cb = torch.cat([x[1] for x in b])
    pa = torch.cat([x[0] for x in a])
    pb = torch.cat([x[0] for x in b])
    d = (ca - cb).abs()
    return {
        "n": int(ca.numel()),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "frac_diff_gt_1e-4": float((d > 1e-4).float().mean()),
        "pred_agreement": float((pa == pb).float().mean()),
    }


def compare_tanl(id_text, corpus, noise, batches: List[torch.Tensor], n_neglabel: int = 10000,
                 cfg: Optional[TANLConfig] = None, device: str = "cpu") -> Dict[str, float]:
    """Official ActivatedNegPostprocessor vs oodlab.methods.TANL on the same batches.

    Feed float32 features for an exact comparison (differences ~1e-6). Feed float16
    features and ``cfg.emulate_fp16=True`` to compare against the fp16 official run.
    """
    cfg = cfg or TANLConfig()
    net = MockNet(id_text.to(device), corpus.to(device), noise.to(device), n_neglabel, logit_scale=cfg.logit_scale)
    batches = [b.to(device) for b in batches]
    off = _run_official(official_tanl(cfg), net, batches, seed=cfg.init_seed)
    ours = _run_ours(TANL(id_text, corpus, noise, cfg, device=device, n_neglabel=n_neglabel), batches)
    return _compare(off, ours)


def compare_adaneg(id_text, neg_text, batches: List[torch.Tensor], cfg: Optional[AdaNegConfig] = None,
                   device: str = "cpu") -> Dict[str, float]:
    cfg = cfg or AdaNegConfig()
    empty = torch.zeros(0, id_text.shape[1], dtype=id_text.dtype)
    net = MockNet(id_text.to(device), neg_text.to(device), empty.to(device), n_neglabel=neg_text.shape[0],
                  logit_scale=cfg.logit_scale)
    batches = [b.to(device) for b in batches]
    off = _run_official(official_adaneg(cfg), net, batches, seed=None)
    ours = _run_ours(AdaNeg(id_text, neg_text, cfg, device=device), batches)
    return _compare(off, ours)


def compare_neglabel(id_text, neg_text, batches: List[torch.Tensor], cfg: Optional[NegLabelConfig] = None,
                     device: str = "cpu") -> Dict[str, float]:
    cfg = cfg or NegLabelConfig()
    empty = torch.zeros(0, id_text.shape[1], dtype=id_text.dtype)
    net = MockNet(id_text.to(device), neg_text.to(device), empty.to(device), n_neglabel=neg_text.shape[0],
                  logit_scale=cfg.logit_scale)
    batches = [b.to(device) for b in batches]
    off = _run_official(official_neglabel(cfg), net, batches, seed=None)
    ours = _run_ours(NegLabel(id_text, neg_text, cfg, device=device), batches)
    return _compare(off, ours)
