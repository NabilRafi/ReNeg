#!/usr/bin/env python
"""Tables from evaluation rows: group means over stream orders, both FPR95 conventions, paired differences.

    python scripts/summarize.py runs/main_vitb16/runs.csv --ref tanl
    python scripts/summarize.py results/g3_vitb16/g3_runs.csv --ref tanl --markdown --sets

Groups: OpenOOD v1.5 near (SSB-hard, NINCO) and far (iNaturalist, Textures, OpenImage-O); Four-OOD (iNaturalist,
SUN, Places, Textures) on the 50k ID images or, for runs with labelled images, on the 45k others (``four_ood_45k``).
``fpr95`` is OpenOOD's convention (OOD positive, as TANL reports); ``fpr95_idpos`` the convention of MCM, NegLabel
and TINS (ID positive). The paired difference is computed per stream order against the reference run on the same
protocol; [min, max] over orders shows whether its sign holds in every order.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

GROUPS = {
    "near": ("openood_v15", ["ssb_hard", "ninco"]),
    "far": ("openood_v15", ["inaturalist", "textures", "openimage_o"]),
    "Four-OOD": ("four_ood", ["inaturalist", "sun", "places", "textures_all"]),
    "Four-OOD 45k": ("four_ood_45k", ["inaturalist", "sun", "places", "textures_all"]),
}
METRICS = {"fpr95": "FPR95", "fpr95_idpos": "FPR95 ID+", "auroc": "AUROC"}


def load(paths) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "label" not in df.columns:
        df["label"] = df.get("config", df.get("method"))
    return df


def group_seed(df, label, protocol, sets, col) -> pd.Series:
    g = df[(df.label == label) & (df.protocol == protocol) & df.dataset.isin(sets)]
    n = g.groupby("seed").size()
    g = g[g.seed.isin(n[n == len(sets)].index)]
    return g.groupby("seed")[col].mean()


def table(df, ref=None, metrics=("fpr95", "fpr95_idpos", "auroc")) -> pd.DataFrame:
    rows = []
    for label in dict.fromkeys(df.label):
        row = {"method": label}
        for gname, (prot, sets) in GROUPS.items():
            for col in metrics:
                s = group_seed(df, label, prot, sets, col)
                if len(s) == 0:
                    continue
                row[f"{gname} {METRICS[col]}"] = round(float(s.mean()), 2)
                if ref and label != ref and col != "auroc":
                    r = group_seed(df, ref, prot, sets, col)
                    d = (s - r).dropna()
                    if len(d):
                        row[f"{gname} {METRICS[col]} vs {ref}"] = f"{d.mean():+.2f} [{d.min():+.2f}, {d.max():+.2f}]"
            n = df[(df.label == label) & (df.protocol == prot)].seed.nunique()
            if n:
                row["orders"] = max(row.get("orders", 0), int(n))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.dropna(axis=1, how="all")


def per_set(df, col="fpr95") -> pd.DataFrame:
    pre = {"openood_v15": "", "four_ood": "4:", "four_ood_45k": "45k:"}
    m = df.groupby(["label", "protocol", "dataset"])[col].mean().reset_index()
    m["column"] = m.protocol.map(lambda p: pre.get(p, p + ":")) + m.dataset
    t = m.pivot(index="label", columns="column", values=col).round(2)
    return t.reindex(list(dict.fromkeys(df.label)))


def show(t: pd.DataFrame, markdown: bool) -> str:
    if markdown:
        cols = list(t.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        for _, r in t.iterrows():
            lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
        return "\n".join(lines)
    return t.to_string(index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", help="runs.csv files (evaluate.py) or g3_runs.csv-style files")
    ap.add_argument("--ref", default=None, help="label of the reference run for paired differences (e.g. tanl)")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS), help="which groups to print")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--sets", action="store_true", help="also print per-set tables")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    df = load(a.csv)
    t = table(df, a.ref)
    for g in a.groups:
        cols = ["method"] + [c for c in t.columns if c.startswith(g + " ") and not (g == "Four-OOD" and "45k" in c)]
        sub = t[cols].dropna(how="all", subset=cols[1:]) if len(cols) > 1 else None
        if sub is not None and len(sub):
            print(f"\n## {g}\n")
            print(show(sub, a.markdown))
    if a.sets:
        for col in ("fpr95", "fpr95_idpos"):
            print(f"\n## Per set, {METRICS[col]} (mean over stream orders)\n")
            ps = per_set(df, col)
            print(show(ps.reset_index().rename(columns={"label": "method"}), a.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
