import numpy as np
from sklearn import metrics as skm


def openood_metrics(conf, label, pred=None):
    conf = np.asarray(conf, dtype=np.float64)
    label = np.asarray(label).astype(int)
    ood = (label == -1).astype(int)
    f, t, _ = skm.roc_curve(ood, -conf)
    fpr = float(f[np.argmax(t >= 0.95)])
    fi, ti, _ = skm.roc_curve(1 - ood, conf)
    return {"fpr95": 100 * fpr, "auroc": 100 * float(skm.auc(f, t)), "fpr95_idpos": 100 * float(fi[np.argmax(ti >= 0.95)])}
