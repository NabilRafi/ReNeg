"""NegLabel-form scores in text space or image space, their fusion, and OpenOOD metrics.

For an image v, ID prototypes P (C, D) and negative prototypes N (M, D)::

    S(v) = sum_c exp(s v.p_c) / (sum_c exp(s v.p_c) + sum_j exp(s v.n_j))
         = sigmoid(LSE_c(s v.p_c) - LSE_j(s v.n_j))

With P = ID name embeddings and N = negative-label embeddings this is NegLabel's score
(without its grouping trick). With transported or image-centroid prototypes it is the
image-space score S_vis of the spec. Higher = more ID-like.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .transport import l2n


@torch.no_grad()
def lse_parts(feats: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, scale: float = 100.0, chunk: int = 4096,
              device: str = "cpu"):
    """(LSE over ID prototypes, LSE over negative prototypes) for every image, float64 numpy."""
    P = l2n(torch.as_tensor(pos).float()).to(device)
    Nn = l2n(torch.as_tensor(neg).float()).to(device)
    a, b = [], []
    for s in range(0, feats.shape[0], chunk):
        v = l2n(feats[s : s + chunk].to(device, torch.float32))
        a.append(torch.logsumexp(scale * v @ P.t(), dim=1).double().cpu())
        b.append(torch.logsumexp(scale * v @ Nn.t(), dim=1).double().cpu())
    return torch.cat(a).numpy(), torch.cat(b).numpy()


def score_from_parts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(b - a, -700, 700)))


def neg_score(feats, pos, neg, scale: float = 100.0, device: str = "cpu") -> np.ndarray:
    a, b = lse_parts(feats, pos, neg, scale=scale, device=device)
    return score_from_parts(a, b)


def fuse(s_text: np.ndarray, s_vis: np.ndarray, omega: float) -> np.ndarray:
    return (1.0 - omega) * s_text + omega * s_vis


# ---------------------------------------------------------------------------
# metrics: oodlab's when available (identical to OpenOOD), otherwise the same algorithm here
# ---------------------------------------------------------------------------


def _metrics_fallback(conf: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    from sklearn import metrics as skm

    conf = np.asarray(conf, dtype=np.float64)
    label = np.asarray(label).astype(int)
    ood = (label == -1).astype(int)
    fpr_l, tpr_l, _ = skm.roc_curve(ood, -conf)                    # OpenOOD: OOD is the positive class
    fpr = float(fpr_l[np.argmax(tpr_l >= 0.95)])
    auroc = float(skm.auc(fpr_l, tpr_l))
    fpr_i, tpr_i, _ = skm.roc_curve(1 - ood, conf)                 # MCM / NegLabel papers: ID positive
    fpr_idpos = float(fpr_i[np.argmax(tpr_i >= 0.95)])
    return {"fpr95": 100 * fpr, "auroc": 100 * auroc, "fpr95_idpos": 100 * fpr_idpos}


def pair_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    conf = np.concatenate([np.asarray(id_scores, dtype=np.float64), np.asarray(ood_scores, dtype=np.float64)])
    label = np.concatenate([np.zeros(len(id_scores), dtype=int), -np.ones(len(ood_scores), dtype=int)])
    try:
        from oodlab.metrics import openood_metrics

        m = openood_metrics(conf, label)
        return {k: float(m[k]) for k in ("fpr95", "auroc", "fpr95_idpos") if k in m}
    except Exception:
        return _metrics_fallback(conf, label)
