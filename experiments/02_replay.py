#!/usr/bin/env python
"""Replay ReNeg's operating points (or any variant) on the TANL traces: three stream orders, both FPR95 conventions.

    python experiments/02_replay.py paper                  # TANL and the paper's operating points (~2-3 h on 2 cores)
    python experiments/02_replay.py table --ref tanl       # group means, paired differences vs TANL
    python experiments/02_replay.py table --file results/replays/idpos.jsonl --ref TANL  # the shipped replays
    python experiments/02_replay.py run --label "balanced, blind, p_min 0.7" --mode balanced --pool blind \\
        --set p_min=0.7                                    # any ReNegConfig field
    python experiments/02_replay.py run --label "max, proto vote" --mode max --opts '{"proto": "vote"}'

Rows go to <RENEG_EXP_OUT>/replay.jsonl (resumable), in the columns of scripts/evaluate.py, so the table is the one
scripts/summarize.py prints. Four-OOD is also scored on the 45k ID images that are not labelled images
(protocol four_ood_45k), the protocol of every few-shot row. Agreed ID admission (CLIP's softmax and TANL's ID mask
must agree) is on by default, as in every headline configuration; --packaged-admission turns it off.
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

import reneg.core as RC  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.paths import REPO, out  # noqa: E402
from replaylab.replay import append, done, mode_config, replay_x  # noqa: E402
from replaylab.traces import ID_OF, load_trace, stream_for, trace_path  # noqa: E402

SETS = {"openood_v15": list(D.OOD_SETS), "four_ood": list(D.FOUR)}
# the configurations of the paper's main table (configs/main_vitb16.yaml), replayed
PAPER = [
    {"label": "reneg_balanced_blind", "mode": "balanced", "pool": "blind"},
    {"label": "reneg_max_blind", "mode": "max", "pool": "blind"},
    {"label": "reneg_safe_blind", "mode": "safe", "pool": "blind"},
    {"label": "reneg_balanced_full", "mode": "balanced", "pool": "full"},
    {"label": "reneg_max_full", "mode": "max", "pool": "full"},
    {"label": "reneg_balanced_blind_5shot", "mode": "balanced", "pool": "blind", "prior": True},
    {"label": "reneg_max_blind_5shot", "mode": "max", "pool": "blind", "prior": True},
]
KEYS = ("label", "seed", "protocol", "dataset")


def parse_set(items):
    """k=v pairs typed like the ReNegConfig defaults."""
    base, kw = RC.ReNegConfig(), {}
    for it in items or []:
        k, v = it.split("=", 1)
        d = getattr(base, k)
        if isinstance(d, bool):
            kw[k] = v.lower() in ("1", "true", "yes")
        elif d is None:
            kw[k] = None if v.lower() == "none" else (float(v) if v.replace(".", "", 1).isdigit() else v)
        else:
            kw[k] = type(d)(v)
    return kw


def rows_for(label, method, seed, prot, ds, s, is_ood, ev45):
    recs = []
    for ptag, ev in [(prot, np.ones(len(s), bool))] + ([("four_ood_45k", ev45)] if prot == "four_ood" else []):
        m = D.metrics(s[ev & ~is_ood], s[ev & is_ood])
        recs.append({"label": label, "method": method, "protocol": ptag, "dataset": ds, "seed": seed,
                     "fpr95": m["fpr95"], "fpr95_idpos": m["fpr95_idpos"], "auroc": m["auroc"],
                     "n_id": int((ev & ~is_ood).sum()), "n_ood": int((ev & is_ood).sum())})
    return recs


def run(spec, seeds, protocols, path, datasets=None, log=print):
    label = spec["label"]
    have = done(path, KEYS)
    rc = mode_config(spec.get("mode", "balanced"), spec.get("opts")).with_(**spec.get("set", {})) \
        if spec.get("mode") else None
    opts = dict(spec.get("opts") or {})
    if spec.get("agreed", True):
        opts.setdefault("idadm", "softmax+tanl")
    P, cl = D.pool_subset(spec.get("pool", "blind")) if rc is not None else (None, None)
    for seed in seeds:
        for prot in protocols:
            id_name = ID_OF[prot]
            for ds in SETS[prot]:
                if datasets and ds not in datasets:
                    continue
                need = [(label, seed, prot, ds)] + ([(label, seed, "four_ood_45k", ds)] if prot == "four_ood" else [])
                if all(k in have for k in need):
                    continue
                if not os.path.isfile(trace_path(ds, seed, id_name)):
                    log(f"no trace for {id_name} / {ds} order {seed} (experiments/01_build_traces.py)")
                    continue
                tr = load_trace(ds, seed, id_name)
                X, is_ood, p = stream_for(ds, seed, id_name, tr)
                ev45 = D.eval_mask_45k(p, len(D.load_set(ds)["labels"])) if prot == "four_ood" else None
                if rc is None:
                    s = np.asarray(tr["tanl"], np.float64)
                else:
                    s = replay_x(tr, X, None, P, cl, rc, opts, prior=bool(spec.get("prior")))
                recs = rows_for(label, "tanl" if rc is None else "reneg_tanl", seed, prot, ds, s, is_ood, ev45)
                for rec in recs:
                    append(path, rec)
                    have.add(tuple(rec[k] for k in KEYS))
                r = recs[0]
                log(f"{label:<30} {prot}:{ds:<13} order {seed}  FPR95 {r['fpr95']:6.2f} (ID+ {r['fpr95_idpos']:6.2f})"
                    + (f"  | 45k {recs[1]['fpr95']:6.2f}" if len(recs) > 1 else ""))


def table(path, ref, markdown=False, sets=False):
    """The tables of scripts/summarize.py for a rows file: this script's, or a shipped one (results/replays/
    seeds.jsonl, idpos.jsonl; replaylab/legacy.py converts them)."""
    import tempfile

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import summarize
    from replaylab.legacy import read_jsonl

    df = read_jsonl(path).drop_duplicates(list(KEYS), keep="last")
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "runs.csv")
        df.to_csv(csv, index=False)
        summarize.main([csv] + (["--ref", ref] if ref else []) + (["--markdown"] if markdown else []) +
                       (["--sets"] if sets else []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("paper", "run"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
        p.add_argument("--protocols", nargs="*", default=list(SETS), choices=list(SETS))
        p.add_argument("--out", default="replay.jsonl", help="file name in RENEG_EXP_OUT")
        p.add_argument("--datasets", nargs="*", help="only these OOD sets")
        if name == "run":
            p.add_argument("--label", required=True)
            p.add_argument("--mode", choices=["balanced", "max", "safe"], help="omit for TANL alone")
            p.add_argument("--pool", default="blind", choices=["blind", "full", "blind+neg", "full+neg"])
            p.add_argument("--prior", action="store_true", help="5 labelled images per class enter the ID model")
            p.add_argument("--packaged-admission", action="store_true", help="softmax within TANL's mask only")
            p.add_argument("--opts", default="{}", help="replay options as JSON (replaylab/replay.py)")
            p.add_argument("--set", nargs="*", default=[], help="ReNegConfig overrides, k=v")
    t = sub.add_parser("table")
    t.add_argument("--file", default="replay.jsonl", help="rows file in RENEG_EXP_OUT, or any path to a rows file")
    t.add_argument("--ref", default="tanl", help="reference label for paired differences (TANL in shipped files)")
    t.add_argument("--markdown", action="store_true")
    t.add_argument("--sets", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    if a.cmd == "table":
        path = a.file if os.path.isfile(a.file) else out(a.file)
        table(path, a.ref, a.markdown, a.sets)
        return 0
    path = out(a.out)
    if a.cmd == "paper":
        specs = [{"label": "tanl"}] + PAPER
    else:
        specs = [{"label": a.label, "mode": a.mode, "pool": a.pool, "prior": a.prior, "opts": json.loads(a.opts),
                  "set": parse_set(a.set), "agreed": not a.packaged_admission}]
    for spec in specs:
        run(spec, a.seeds, a.protocols, path, a.datasets)
    print(f"\nrows in {path}; table: python experiments/02_replay.py table --file {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
