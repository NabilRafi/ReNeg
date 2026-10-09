#!/usr/bin/env python
"""Pseudo near-OOD tuning on the TANL base: choose ReNeg's settings without touching any benchmark OOD image.

    python experiments/06_pseudo_ood_tuning.py traces       # TANL once per pseudo task (4 tasks, ~10 min)
    python experiments/06_pseudo_ood_tuning.py modes        # penalty mode x fusion weight x pool mass in the text
    python experiments/06_pseudo_ood_tuning.py admission    # which stream images may build the ID prototypes

Tasks: ImageNet test, even stream positions (22,500 images, labels known). Near: 3 random splits that hold out 100
ImageNet classes; their images are the pseudo near-OOD, the other 900 classes are ID, and the candidate pool is
the KG pool plus the held-out names (the "full" analogue). Far: the same images (all 1,000 classes) against
OpenOOD's OpenImage-O validation split (1,763 images). Objective: mean pseudo-near FPR95 + far FPR95.
Rows: <RENEG_EXP_OUT>/pseudo_tanl.jsonl and pseudo_admit.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import reneg.core as RC  # noqa: E402
from oodlab.methods.tanl import TANL, TANLConfig  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.paths import TRACES, out  # noqa: E402
from replaylab.replay import B0, append  # noqa: E402
from replaylab.traces import make_trace, replay, tanl_inputs  # noqa: E402


def source_data(source):
    if source == "val":                     # OpenOOD's ImageNet val split (5,000 images)
        return D.val_X, D.val_y
    X = D.load_set("imagenet_test")
    return X["X"][0::2], X["labels"][0::2]


def tasks(source="halfA", seeds=(0, 1, 2), n_hold=100, blind=False, far=True, stream_seed=0):
    X, y = source_data(source)
    id_text = D.id_text
    pool_text, pool_cluster = D.pool_subset("full")
    C = id_text.shape[0]
    res = []
    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        H = np.sort(rng.choice(C, n_hold, replace=False))
        keep = np.setdiff1d(np.arange(C), H)
        inH = np.isin(y, H)
        Xs, is_ood, _ = D.pair_stream(X[torch.as_tensor(np.nonzero(~inH)[0])], X[torch.as_tensor(np.nonzero(inH)[0])],
                                      stream_seed)
        if blind:
            P, cl = pool_text, pool_cluster
        else:
            P = torch.cat([pool_text, id_text[torch.as_tensor(H)]])
            cl = np.r_[pool_cluster, pool_cluster.max() + 1 + np.arange(len(H))]
        res.append({"name": f"near{sd}", "X": Xs, "is_ood": is_ood, "T_id": id_text[torch.as_tensor(keep)], "P": P,
                    "cl": cl})
    if far:
        Xs, is_ood, _ = D.pair_stream(X, D.val_ood, stream_seed)
        res.append({"name": "far", "X": Xs, "is_ood": is_ood, "T_id": id_text, "P": pool_text, "cl": pool_cluster})
    return res


def evaluate(score_fn, task_list):
    """score_fn(task) -> scores. Per-task FPR95/AUROC and the objective (mean near FPR + far FPR)."""
    res = {}
    for t in task_list:
        s = score_fn(t)
        m = D.metrics(s[~t["is_ood"]], s[t["is_ood"]])
        res[t["name"]] = (round(m["fpr95"], 2), round(m["auroc"], 2))
    near = [v for k, v in res.items() if k.startswith("near")]
    summ = {"pnear": round(float(np.mean([a for a, _ in near])), 2) if near else float("nan"),
            "pnear_auc": round(float(np.mean([b for _, b in near])), 2) if near else float("nan")}
    if "far" in res:
        summ.update({"vfar": res["far"][0], "vfar_auc": res["far"][1]})
    summ["objective"] = round(summ["pnear"] + summ.get("vfar", 0.0), 2)
    return summ, res


def task_traces(task_list, src="halfA", log=print):
    _, corpus, noise, n_sel = tanl_inputs()
    os.makedirs(TRACES, exist_ok=True)
    for t in task_list:
        p = os.path.join(TRACES, f"pseudo_{src}_{t['name']}.pt")
        if os.path.isfile(p):
            t["trace"] = torch.load(p, weights_only=False)
            continue
        tanl = TANL(t["T_id"], corpus, noise, TANLConfig().with_(record=True), n_neglabel=n_sel)
        tr = make_trace(tanl, t["X"], store_dtype=torch.float16)
        torch.save(tr, p)
        t["trace"] = tr
        r = D.metrics(tr["tanl"][~t["is_ood"]], tr["tanl"][t["is_ood"]])
        log(f"{t['name']}: TANL FPR95 {r['fpr95']:.2f} AUROC {r['auroc']:.2f} ({tr['sec'] / 60:.1f} min)")
    return task_list


def reneg_scores(rcfg):
    def fn(t):
        return replay(t["trace"], t["X"], t["T_id"], t["P"] if rcfg.use_pool else None, t["cl"], rcfg)[0]
    return fn


def tanl_scores(t):
    return t["trace"]["tanl"]


def modes(T, path, log=print):
    base = RC.ReNegConfig(n_min=1, vote=True)
    res0, per0 = evaluate(tanl_scores, T)
    log(f"TANL: {res0}")
    append(path, {"name": "TANL", "kw": {}, **res0, "per": per0})
    for mode in ("kg_clip", "kg_clip_caught"):
        for om in (0.25, 0.4, 0.6):
            for frac in (0.0, 0.1):
                name = f"{mode} + vote, omega {om}, frac {frac}"
                kw = {"image_mode": mode, "omega": om, "pool_frac": frac, "pool_in_text": frac > 0}
                res, per = evaluate(reneg_scores(base.with_(**kw)), T)
                log(f"{name}: {res}")
                append(path, {"name": name, "kw": kw, **res, "per": per})


def admission(T, path, log=print):
    modes_ = {"max": dict(B0), "balanced": dict(B0, image_mode="kg_clip", vote=True)}
    rules = {"softmax 0.5": {}, "softmax 0.7": {"p_min": 0.7}, "softmax 0.5, no KG-admitted": {"id_exclude": "admitted"},
             "TANL ID mask": {"id_admit": "base"}, "TANL ID mask, no KG-admitted": {"id_admit": "base",
                                                                                  "id_exclude": "admitted"},
             "softmax 0.5 and TANL ID mask (agreed)": {"id_admit": "softmax+base"}}
    res0, _ = evaluate(tanl_scores, T)
    log(f"TANL: {res0}")
    for mname, mkw in modes_.items():
        for rname, rkw in rules.items():
            res, per = evaluate(reneg_scores(RC.ReNegConfig(**mkw, **rkw)), T)
            log(f"{mname} | {rname}: {res}")
            append(path, {"mode": mname, "rule": rname, **res, "per": per})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["traces", "modes", "admission"])
    ap.add_argument("--source", default="halfA", choices=["halfA", "val"])
    ap.add_argument("--n-hold", type=int, default=100)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    T = task_traces(tasks(a.source, n_hold=a.n_hold), a.source)
    if a.cmd == "modes":
        modes(T, out("pseudo_tanl.jsonl"))
    elif a.cmd == "admission":
        admission(T, out("pseudo_admit.jsonl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
