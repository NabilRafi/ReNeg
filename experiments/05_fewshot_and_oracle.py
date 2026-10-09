#!/usr/bin/env python
"""Few-shot ID prior and oracle ID models: how much does the purity of ReNeg's ID image prototypes matter?

    python experiments/05_fewshot_and_oracle.py run                 # every item of plans/fewshot_oracle.json
    python experiments/05_fewshot_and_oracle.py report

Few-shot setting (as TINS and InterNeg, which use 16 ImageNet train images per class): OpenOOD's 5k ImageNet ID-val
images (about 5 per class, disjoint from OpenOOD v1.5's 45k ID test set) enter the online ID image model before the
stream starts. On Four-OOD (50k ID = val + test) those images are left out of the metric (45k protocol), and TANL
is re-scored on the same images. Oracle arms (upper bounds, never results) feed the ID model true labels:
``full`` (true labels, true ID only), ``purity`` (true ID only, CLIP's label), ``label`` (realistic admission,
true label), ``decontam`` (realistic admission minus its OOD images), ``coverage`` (plus every ID image).
Finding (Oct 3): purity of the ID prototypes is the main lever; 5 labelled images per class recover part of it.
Rows: <RENEG_EXP_OUT>/id_prior.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from replaylab import data as D  # noqa: E402
from replaylab.paths import out  # noqa: E402
from replaylab.replay import JOBS, append, done, mode_config, replay_x  # noqa: E402
from replaylab.traces import load_trace, stream_for  # noqa: E402

OUTJ = "id_prior.jsonl"
KEYS = ("mode", "variant", "seed", "id", "set")


def run(items, path, log=print):
    have = done(path, KEYS)
    for it in items:
        mode, variant, opts = it["mode"], it["variant"], it.get("opts", {})
        prior, oracle = it.get("prior", True), it.get("oracle", False)
        cfg = mode_config(mode, opts)
        P, cl = D.pool_subset(it.get("pool", "full"))
        jobs = JOBS["screen7" if it["jobs"] == "screen" else it["jobs"]]
        for seed in it.get("seeds", [0]):
            for id_name, ds in jobs:
                if (mode, variant, seed, id_name, ds) in have:
                    continue
                d_id = D.load_set(id_name)
                n_ood = len(D.load_set(ds)["labels"])
                tr = load_trace(ds, seed, id_name)
                X, is_ood, p = stream_for(ds, seed, id_name, tr)
                id_row = np.where(p >= n_ood, p - n_ood, -1)
                y = np.where(id_row >= 0, d_id["labels"][np.maximum(id_row, 0)], -1)
                evalm = D.eval_mask_45k(p, n_ood) if id_name == "imagenet_val_all" else np.ones(len(X), bool)
                s = replay_x(tr, X, None, P, cl, cfg, opts, prior=prior, oracle_y=y if oracle else None)
                r = D.metrics(s[evalm & ~is_ood], s[evalm & is_ood])
                rt = D.metrics(tr["tanl"][evalm & ~is_ood], tr["tanl"][evalm & is_ood])
                append(path, {"stage": it.get("stage"), "mode": mode, "variant": variant, "seed": seed, "id": id_name,
                              "set": ds, "fpr95": r["fpr95"], "fpr95_idpos": r["fpr95_idpos"], "auroc": r["auroc"],
                              "tanl_fpr95_evalset": rt["fpr95"], "tanl_fpr95_idpos_evalset": rt["fpr95_idpos"]})
                have.add((mode, variant, seed, id_name, ds))
                log(f"{mode} | {variant} | order {seed} {ds}: {r['fpr95']:.2f} (TANL on the same images {rt['fpr95']:.2f})")


def report(path, col="fpr95"):
    """Group means (near, far, Four-OOD 45k) with paired differences vs TANL on the same images, then per set."""
    ood = list(D.OOD_SETS)
    four = ["4ood_" + k for k in D.FOUR]
    groups = {"near": ood[:2], "far": ood[2:], "four45": four}
    tcol = "tanl_fpr95_evalset" if col == "fpr95" else "tanl_fpr95_idpos_evalset"
    df = pd.DataFrame(map(json.loads, open(path)))
    for c in ("fpr95_idpos", "tanl_fpr95_idpos_evalset"):                   # rows of the original Oct 3 runs
        if c not in df.columns:
            df[c] = np.nan
    df["config"] = df["mode"] + " + " + df["variant"]
    df = df.drop_duplicates(["config", "seed", "id", "set"], keep="last")
    df["key"] = np.where(df.id == "imagenet_test", df.set, "4ood_" + df.set)
    tanl = df.drop_duplicates(["seed", "key"]).set_index(["seed", "key"])[tcol]
    rows = []
    base = df.drop_duplicates(["seed", "key"]).assign(**{col: lambda d: d[tcol]})
    for cfg, g in [("TANL (same images)", base)] + list(df.groupby("config")):
        row = {"method": cfg, "orders": int(g.seed.nunique())}
        for gname, keys in groups.items():
            p = g[g.key.isin(keys)].pivot_table(index="seed", columns="key", values=col).dropna()
            if len(p.columns) < len(keys) or not len(p):
                continue
            m = p[keys].mean(axis=1)
            row[gname] = round(m.mean(), 2)
            if not cfg.startswith("TANL"):
                pa = g[g.key.isin(keys)].pivot_table(index="seed", columns="key", values="auroc").dropna()
                if len(pa.columns) == len(keys):
                    row[gname + "_auc"] = round(pa[keys].mean(axis=1).mean(), 2)
                b = pd.Series({s: np.mean([tanl[(s, k)] for k in keys]) for s in m.index})
                d = m - b
                row[gname + "_d"] = f"{d.mean():+.2f} [{d.min():+.2f},{d.max():+.2f}]"
        for k in ood + four:
            v = g[g.key == k][col]
            row[k] = round(v.mean(), 2) if len(v) else np.nan
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    cols = [c for c in ["method", "orders", "near", "near_auc", "near_d", "far", "far_auc", "far_d", "four45",
                        "four45_auc", "four45_d"] if c in res.columns]
    print(f"== {col}: group means (Four-OOD on the 45k non-shot ID images), paired difference vs TANL")
    print(res[cols].fillna("").to_string(index=False))
    print()
    print(res[["method"] + [c for c in ood + four if c in res]].to_string(index=False))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--plan", default=os.path.join(HERE, "plans", "fewshot_oracle.json"))
    ap.add_argument("--stage", nargs="*", help="only these stages (prior-1, oracle-1, oracle-2, confirm)")
    ap.add_argument("--col", default="fpr95", choices=["fpr95", "fpr95_idpos"])
    ap.add_argument("--file", default=OUTJ, help="rows file in RENEG_EXP_OUT, or a path (results/replays/id_prior.jsonl)")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    path = a.file if (a.cmd == "report" and os.path.isfile(a.file)) else out(a.file)
    if a.cmd == "run":
        run([it for it in json.load(open(a.plan)) if not a.stage or it.get("stage") in a.stage], path)
    else:
        report(path, a.col)
    return 0


if __name__ == "__main__":
    sys.exit(main())
