"""OOD metrics.

``openood_metrics`` is a line-for-line port of ``openood/evaluators/metrics.py``
from OpenOOD-VLM, which is what produced TANL's 9.81 and AdaNeg's 17.95.

IMPORTANT - two FPR@95 conventions exist:

* OpenOOD (used by TANL/AdaNeg/this repo): OOD is the *positive* class.
  FPR@95 = fraction of **ID** samples flagged as OOD at the threshold that
  catches 95% of OOD samples.  -> ``fpr95`` below.
* MCM / NegLabel papers: ID is the positive class.
  FPR@95 = fraction of **OOD** samples accepted as ID at the threshold that
  keeps 95% of ID samples.  -> ``fpr95_idpos`` below.

They are different numbers. Always compare like with like; we report both.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn import metrics as skm


def auc_and_fpr_recall(conf: np.ndarray, label: np.ndarray, tpr_th: float = 0.95):
    """Verbatim OpenOOD logic: OOD (label == -1) is positive; larger conf = more ID."""
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    fpr_list, tpr_list, _ = skm.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]
    precision_in, recall_in, _ = skm.precision_recall_curve(1 - ood_indicator, conf)
    precision_out, recall_out, _ = skm.precision_recall_curve(ood_indicator, -conf)
    auroc = skm.auc(fpr_list, tpr_list)
    aupr_in = skm.auc(recall_in, precision_in)
    aupr_out = skm.auc(recall_out, precision_out)
    return auroc, aupr_in, aupr_out, fpr


def fpr_at_tpr_id_positive(conf: np.ndarray, label: np.ndarray, tpr_th: float = 0.95) -> float:
    """MCM/NegLabel convention: ID positive; FPR measured on OOD samples."""
    id_indicator = (label != -1).astype(int)
    fpr_list, tpr_list, _ = skm.roc_curve(id_indicator, conf)
    return float(fpr_list[np.argmax(tpr_list >= tpr_th)])


def id_accuracy(pred: Optional[np.ndarray], label: np.ndarray) -> float:
    if pred is None:
        return float("nan")
    m = label != -1
    if m.sum() == 0:
        return float("nan")
    return float((pred[m] == label[m]).mean())


def openood_metrics(conf, label, pred=None) -> Dict[str, float]:
    """All metrics for one ID/OOD pair. Values are percentages (x100) like the papers."""
    # float16 scores (official fp16 emulation) convert to float64 exactly, so ties are kept.
    conf = np.asarray(conf).astype(np.float64)
    label = np.asarray(label).astype(int)
    pred = None if pred is None else np.asarray(pred).astype(int)
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, 0.95)
    return {
        "fpr95": 100.0 * float(fpr),
        "auroc": 100.0 * float(auroc),
        "aupr_in": 100.0 * float(aupr_in),
        "aupr_out": 100.0 * float(aupr_out),
        "fpr95_idpos": 100.0 * fpr_at_tpr_id_positive(conf, label, 0.95),
        "acc": 100.0 * id_accuracy(pred, label),
        "n_id": int((label != -1).sum()),
        "n_ood": int((label == -1).sum()),
    }
