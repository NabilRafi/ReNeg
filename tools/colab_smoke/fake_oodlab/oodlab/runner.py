from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .metrics import openood_metrics

PROTOCOLS = {"four_ood": ("imagenet_val_all", ["inaturalist", "sun", "places", "textures_all"]),
             "openood_val": ("imagenet_val", ["openimage_o_val"]),
             "openood_v15": ("imagenet_test", ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"])}


@dataclass(frozen=True)
class Protocol:
    name: str
    id_set: str
    groups: Dict[str, List[str]]
    note: str = ""


GROUPS = {"ssb_hard": "near", "ninco": "near", "inaturalist": "far", "textures": "far", "openimage_o": "far"}


def _score(m, feats):
    v = feats.float() / feats.float().norm(dim=1, keepdim=True)
    a = torch.logsumexp(m.cfg.logit_scale * v @ m.id_text.float().t(), 1)
    b = torch.logsumexp(m.cfg.logit_scale * v @ m.neg.float().t(), 1)
    return torch.sigmoid(a - b).numpy()


class Runner:
    def __init__(self, cache, device="cpu", logger=None, keep_orders=False, **kw):
        self.cache = cache

    def missing(self, protocol):
        id_set, oods = PROTOCOLS[protocol]
        have = set(self.cache.available())
        return [n for n in [id_set, *oods] if n not in have]

    def get(self, name):
        return self.cache.load_set(name)

    def evaluate(self, method, protocol, seeds=(0,), config="default", **kw):
        id_set, oods = PROTOCOLS[protocol]
        sid = _score(method, self.cache.load_set(id_set).feats)
        rows = []
        for seed in seeds:
            for ds in oods:
                so = _score(method, self.cache.load_set(ds).feats)
                conf = np.concatenate([sid, so])
                lab = np.concatenate([np.zeros(len(sid)), -np.ones(len(so))])
                m = openood_metrics(conf, lab)
                group = "four" if protocol == "four_ood" else GROUPS.get(ds, "val")
                rows.append({"method": "tanl", "config": config, "protocol": protocol, "group": group, "dataset": ds,
                             "seed": seed, "fpr95": m["fpr95"], "auroc": m["auroc"]})
        return pd.DataFrame(rows)


def summarize(df):
    return df.groupby(["method", "config", "protocol"]).agg(fpr95=("fpr95", "mean"), auroc=("auroc", "mean")).reset_index()
