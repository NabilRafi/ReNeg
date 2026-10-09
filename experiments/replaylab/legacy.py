"""Read the shipped replay rows (results/replays/*.jsonl, written by the original analysis scripts) in the column
format of scripts/evaluate.py, so the same table code summarises old and new runs.

  seeds.jsonl   {config, seed, id, set, fpr95, auroc, tanl_fpr95, tanl_auroc}     operating points, Oct 2
  idpos.jsonl   {config, seed, id, set, fpr95, fpr95_idpos, auroc}               both conventions, Oct 4
                (few-shot rows and "TANL (45k)" on Four-OOD are scored on the 45k non-shot ID images)
  replay.jsonl  {label, protocol, dataset, seed, ...}                            experiments/02_replay.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

COLS = ["label", "method", "protocol", "dataset", "seed", "fpr95", "fpr95_idpos", "auroc"]


def _protocol(id_name: str, label: str) -> str:
    if id_name == "imagenet_test":
        return "openood_v15"
    return "four_ood_45k" if ("5-shot" in label or label == "TANL (45k)") else "four_ood"


def to_runs(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "label" in df.columns and "protocol" in df.columns:                 # already the new format
        return df
    if "config" not in df.columns or "id" not in df.columns:
        raise ValueError("unknown rows format (expected evaluate.py columns, or config/id/set rows)")
    out = []
    if "tanl_fpr95" in df.columns:                                          # seeds.jsonl: TANL rides along
        t = df.drop_duplicates(["seed", "id", "set"])
        out.append(pd.DataFrame({"label": "TANL", "method": "tanl", "protocol": [_protocol(i, "") for i in t.id],
                                 "dataset": t.set.values, "seed": t.seed.values, "fpr95": t.tanl_fpr95.values,
                                 "fpr95_idpos": np.nan, "auroc": t.get("tanl_auroc", np.nan)}))
    lab = df.config.replace({"TANL (45k)": "TANL"})
    out.append(pd.DataFrame({"label": lab.values, "method": "replay",
                             "protocol": [_protocol(i, c) for i, c in zip(df.id, df.config)],
                             "dataset": df.set.values, "seed": df.seed.values, "fpr95": df.fpr95.values,
                             "fpr95_idpos": df["fpr95_idpos"].values if "fpr95_idpos" in df else np.nan,
                             "auroc": df.auroc.values}))
    res = pd.concat(out, ignore_index=True)
    return res.drop_duplicates(["label", "protocol", "dataset", "seed"], keep="last").reindex(columns=COLS)


def read_jsonl(path: str) -> pd.DataFrame:
    with open(path) as fh:
        return to_runs([json.loads(line) for line in fh if line.strip()])
