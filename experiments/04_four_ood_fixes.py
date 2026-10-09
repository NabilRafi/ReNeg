#!/usr/bin/env python
"""Four-OOD fixes: screen variants of ReNeg's admission and prototypes, confirm the survivors on 3 stream orders.

    python experiments/04_four_ood_fixes.py run                    # every item of plans/four_ood_fixes.json
    python experiments/04_four_ood_fixes.py run --stage baseline confirm-1 confirm-2
    python experiments/04_four_ood_fixes.py report                 # 3-order table vs TANL, paired differences

Stages (as run on Oct 3): ``baseline`` the diagnosed release; ``screen-*`` one stream order on six sets (SUN,
Places, Textures-all; NINCO, Textures, OpenImage-O); ``confirm-*`` all 9 sets x 3 orders. The winner, agreed ID
admission (``id=softmax+tanl``), is the admission rule of every headline result. Rows: <RENEG_EXP_OUT>/four_fix.jsonl.
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

OUTF = "four_fix.jsonl"
KEYS = ("mode", "variant", "seed", "id", "set")


def run(items, path, log=print):
    have = done(path, KEYS)
    for it in items:
        mode, variant, opts = it["mode"], it["variant"], it.get("opts", {})
        cfg = mode_config(mode, opts)
        P, cl = D.pool_subset(it.get("pool", "full"))
        for seed in it.get("seeds", [0]):
            for id_name, ds in JOBS[it["jobs"]]:
                if (mode, variant, seed, id_name, ds) in have:
                    continue
                tr = load_trace(ds, seed, id_name)
                X, is_ood, _ = stream_for(ds, seed, id_name, tr)
                s = replay_x(tr, X, None, P, cl, cfg, opts)
                r = D.metrics(s[~is_ood], s[is_ood])
                rt = D.metrics(tr["tanl"][~is_ood], tr["tanl"][is_ood])
                append(path, {"stage": it.get("stage"), "mode": mode, "variant": variant, "opts": opts,
                              "pool": it.get("pool", "full"), "seed": seed, "id": id_name, "set": ds,
                              "fpr95": r["fpr95"], "fpr95_idpos": r["fpr95_idpos"], "auroc": r["auroc"],
                              "tanl_fpr95": rt["fpr95"], "tanl_fpr95_idpos": rt["fpr95_idpos"], "tanl_auroc": rt["auroc"]})
                have.add((mode, variant, seed, id_name, ds))
                log(f"{mode} | {variant} | order {seed} {ds}: {r['fpr95']:.2f} (TANL {rt['fpr95']:.2f})")


GROUPS = {"near": list(D.NEAR), "far": list(D.FAR), "four": ["4ood_" + k for k in D.FOUR]}


def report(path, col="fpr95", screen=False):
    """Complete configurations (9 sets x 3 orders) with paired differences against TANL on the same streams; with
    screen=True, the one-order screens (six sets) instead."""
    df = pd.DataFrame(map(json.loads, open(path)))
    for c in ("stage", "tanl_auroc", "tanl_fpr95_idpos", "fpr95_idpos"):     # rows of the original Oct 3 runs
        if c not in df.columns:
            df[c] = np.nan
    df["config"] = df["mode"] + " + " + df["variant"]
    df = df.drop_duplicates(["config", "seed", "id", "set"], keep="last")
    df["key"] = np.where(df.id == "imagenet_test", df.set, "4ood_" + df.set)
    tanl = df.drop_duplicates(["seed", "key"]).set_index(["seed", "key"])
    tcol = "tanl_" + col
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    if screen:
        scr = df.stage.astype(str).str.startswith("screen")
        keys = sorted(df[scr].key.unique()) if scr.any() else sorted(df[df.seed == 0].key.unique())
        g = df[(df.seed == 0) & df.key.isin(keys)]
        t = g.pivot_table(index="config", columns="key", values=col).round(2)
        ref = tanl.loc[0][tcol].reindex(t.columns).round(2)
        t.loc["TANL"] = ref
        print(f"== screens, order 0, {col}\n" + t.to_string())
        return t
    rows = []
    t = df.drop_duplicates(["seed", "key"]).assign(**{col: lambda d: d[tcol], "auroc": lambda d: d.tanl_auroc})
    for cfg, g in [("TANL", t)] + list(df.groupby("config")):
        n = g.groupby("seed").key.nunique()
        if cfg != "TANL" and not (len(n) == 3 and n.min() == 9):
            continue
        row = {"method": cfg}
        for gname, keys in GROUPS.items():
            p = g[g.key.isin(keys)].pivot_table(index="seed", columns="key", values=col).dropna()
            pa = g[g.key.isin(keys)].pivot_table(index="seed", columns="key", values="auroc").dropna()
            if len(p.columns) < len(keys):
                continue
            m = p[keys].mean(axis=1)
            row[gname] = round(m.mean(), 2)
            if set(keys) <= set(pa.columns):
                row[gname + "_auc"] = round(pa[keys].mean(axis=1).mean(), 2)
            if cfg != "TANL":
                b = pd.Series({s: np.mean([tanl.loc[(s, k), tcol] for k in keys]) for s in m.index})
                d = m - b
                row[gname + "_d"] = f"{d.mean():+.2f} [{d.min():+.2f},{d.max():+.2f}]"
        for k in list(D.OOD_SETS) + ["4ood_" + k for k in D.FOUR]:
            row[k] = round(g[g.key == k][col].mean(), 2)
        rows.append(row)
    res = pd.DataFrame(rows)
    cols = [c for c in ["method", "near", "near_auc", "near_d", "far", "far_auc", "far_d", "four", "four_auc", "four_d"]
            if c in res.columns]
    print(f"== 3 stream orders, {col} (paired difference vs TANL [min, max over orders])")
    print(res[cols].fillna("").to_string(index=False))
    print()
    print(res[["method"] + [c for c in list(D.OOD_SETS) + ["4ood_" + k for k in D.FOUR] if c in res]].to_string(index=False))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--plan", default=os.path.join(HERE, "plans", "four_ood_fixes.json"))
    ap.add_argument("--stage", nargs="*", help="only these stages (baseline, screen-1 ... confirm-2)")
    ap.add_argument("--col", default="fpr95", choices=["fpr95", "fpr95_idpos"])
    ap.add_argument("--screens", action="store_true", help="report: the one-order screens")
    ap.add_argument("--file", default=OUTF, help="rows file in RENEG_EXP_OUT, or a path (results/replays/four_fix.jsonl)")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    path = a.file if (a.cmd == "report" and os.path.isfile(a.file)) else out(a.file)
    if a.cmd == "run":
        items = [it for it in json.load(open(a.plan)) if not a.stage or it.get("stage") in a.stage]
        run(items, path)
    else:
        report(path, a.col, a.screens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
