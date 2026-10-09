"""Live-vs-cache acceptance check.

Runs the vendored **official** TANL postprocessor on images decoded and encoded on
the fly (like ``main.py`` would), and our TANL on the **cached** features, over the
same stream order. If the cache is faithful, FPR95/AUROC agree to two decimals
(scores agree to fp16 precision; tiny differences can come from GPU kernels picking
a different accumulation order for a different batch composition).
"""
from __future__ import annotations

import contextlib
import io
import time
from typing import Dict, Optional

import numpy as np
import torch

from ..cache import FeatureSet
from ..clipwrap import build_transform, encode_image_batch, input_resolution, load_image, model_dtype
from ..data.manifest import Manifest
from ..methods import TANL, TANLConfig
from ..metrics import openood_metrics
from .equivalence import MockNet, official_tanl


class LiveNet(MockNet):
    """``FixedCLIP_NegOODPrompt`` stand-in whose forward pass really encodes images."""

    def __init__(self, model, id_text, corpus, noise, n_neglabel: int, logit_scale: float = 100.0):
        super().__init__(id_text, corpus, noise, n_neglabel, logit_scale=logit_scale)
        self.model = model

    def __call__(self, data, return_feat: bool = False):
        f = encode_image_batch(self.model, data)
        if return_feat:
            return f, self.text_features.t(), self.logit_scale
        return self.logit_scale * f @ self.text_features


class _Paths(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_image(self.paths[i]))


@torch.no_grad()
def live_vs_cache(model, bank, id_manifest: Manifest, id_fs: FeatureSet, ood_manifest: Manifest, ood_fs: FeatureSet,
                  n_id: int = 5000, n_ood: Optional[int] = None, seed: int = 0, cfg: Optional[TANLConfig] = None,
                  device: str = "cuda", batch_size: int = 256, num_workers: int = 4, logger=None) -> Dict:
    log = logger.info if logger else print
    half = model_dtype(model) == torch.float16   # True on GPU (official set-up). NB: CLIP's first
                                                 # parameter (positional_embedding) stays fp32, use model.dtype
    cfg = (cfg or TANLConfig.paper()).with_(emulate_fp16=half, exact_fp16_final=half,
                                            logit_scale=float(model.logit_scale.detach().float().exp().item()))
    g = torch.Generator().manual_seed(seed)
    id_keys = [id_fs.keys[i] for i in torch.randperm(len(id_fs), generator=g)[:n_id].tolist()]
    n_ood = len(ood_fs) if n_ood is None else min(n_ood, len(ood_fs))
    ood_keys = [ood_fs.keys[i] for i in torch.randperm(len(ood_fs), generator=g)[:n_ood].tolist()]

    # stream = shuffle(OOD + ID), same order for both runs
    order = torch.randperm(n_ood + n_id, generator=torch.Generator().manual_seed(seed + 1)).numpy()
    m_pos_id = {k: i for i, k in enumerate(id_manifest.keys)}
    m_pos_ood = {k: i for i, k in enumerate(ood_manifest.keys)}
    paths = [ood_manifest.paths[m_pos_ood[k]] for k in ood_keys] + [id_manifest.paths[m_pos_id[k]] for k in id_keys]
    labels = np.concatenate([np.full(n_ood, -1), id_fs.select_keys(id_keys).labels])
    feats = torch.cat([ood_fs.select_keys(ood_keys).feats, id_fs.select_keys(id_keys).feats], 0)
    paths = [paths[i] for i in order]
    labels = labels[order]
    feats = feats[torch.as_tensor(order)]

    # ---- live: official postprocessor, images -> CLIP -> scores
    dt = model_dtype(model)
    net = LiveNet(model, bank.id_text.to(device, dt), bank.corpus_text.to(device, dt), bank.noise_feats.to(device, dt),
                  n_neglabel=bank.n_selected, logit_scale=float(model.logit_scale.detach().float().exp().item()))
    post = official_tanl(cfg)
    size = input_resolution(model)
    dl = torch.utils.data.DataLoader(_Paths(paths, build_transform(size, round(size * 256 / 224))),
                                     batch_size=batch_size, shuffle=False, num_workers=num_workers)
    batches = iter(dl)  # creating the iterator draws a seed from the global RNG, so seed *after* it
    torch.manual_seed(cfg.init_seed)  # the official reset draws its permutations from the global RNG
    live_scores, live_pred = [], []
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        for x in batches:
            pred, conf = post.postprocess(net, x.to(device))
            live_scores.append(conf.float().cpu())
            live_pred.append(pred.cpu())
    t_live = time.time() - t0
    live_scores = torch.cat(live_scores).numpy().astype(np.float64)
    live_pred = torch.cat(live_pred).numpy()

    # ---- replay: our TANL on cached features
    ours = TANL(bank.id_text, bank.corpus_text, bank.noise_feats, cfg, device=device, n_neglabel=bank.n_selected)
    # (on a CPU/fp32 model the cache holds fp16-rounded copies of fp32 features, so expect tiny differences)
    rs, rp = [], []
    for s in range(0, len(order), batch_size):
        o = ours.step(feats[s : s + batch_size].to(device))
        rs.append(o.score.float().cpu())
        rp.append(o.pred.cpu())
    cache_scores = torch.cat(rs).numpy().astype(np.float64)
    cache_pred = torch.cat(rp).numpy()

    m_live = openood_metrics(live_scores, labels, live_pred)
    m_cache = openood_metrics(cache_scores, labels, cache_pred)
    d = np.abs(live_scores - cache_scores)
    rep = {
        "n_id": n_id, "n_ood": n_ood, "ood_set": ood_fs.name,
        "live_fpr95": round(m_live["fpr95"], 2), "cache_fpr95": round(m_cache["fpr95"], 2),
        "live_auroc": round(m_live["auroc"], 2), "cache_auroc": round(m_cache["auroc"], 2),
        "live_acc": round(m_live["acc"], 2), "cache_acc": round(m_cache["acc"], 2),
        "max_abs_score_diff": float(d.max()), "frac_scores_identical": float((d == 0).mean()),
        "pred_agreement": float((live_pred == cache_pred).mean()), "live_seconds": round(t_live, 1),
    }
    rep["pass"] = bool(rep["live_fpr95"] == rep["cache_fpr95"] and rep["live_auroc"] == rep["cache_auroc"])
    log(f"[live-vs-cache] {rep}")
    return rep
