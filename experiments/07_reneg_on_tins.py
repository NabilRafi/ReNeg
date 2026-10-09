#!/usr/bin/env python
"""ReNeg on top of TINS (and of TANL+TINS), replayed on the per-image TINS traces of notebook P4.

    python experiments/07_reneg_on_tins.py selftest      # the generic replay == replay_x on a TANL base
    python experiments/07_reneg_on_tins.py run           # every configuration x order x set (resumable)
    python experiments/07_reneg_on_tins.py report        # tables in both conventions and the pass rule

ReNeg never changes its base: it reads the base's score (log-odds lt), votes with it (running Otsu threshold) and
admits images to its ID image prototypes only where CLIP's softmax and the base agree. So for any base

    s = (1 - omega) * lt_base + omega * li          li = ReNeg's image score (KG visual prototypes, ID prototypes)

Bases (pre-registered on Oct 4, before the traces arrived):
  tins        lt = logit(S_TINS); vote on lt; base ID mask = not voted OOD (the same Otsu rule as the vote)
  tins@0.7    as tins, base ID mask = S_TINS >= 0.7 (TINS inverts images with S < 0.3; 0.7 is the mirror)
  tanl+tins   lt = (logit(S_TANL) + logit(S_TINS)) / 2; vote on lt; base ID mask = TANL's mask and not voted OOD
  tanl        TANL's trace (for the same-stream reference rows)

Pass rule (pre-registered): ReNeg-balanced on TINS, blind pool, 5-shot prior (TINS already uses those labelled
images) beats TINS alone in TINS's own convention (FPR95, ID positive), mean of three stream orders, on near-OOD and
far-OOD in every order and on Four-OOD (45k non-shot ID images) on average. Result (Oct 5): PASS.

TINS traces: results/p4_tins/tins_traces_1.zip (from P4), or --traces DIR_OR_ZIP. Rows: <RENEG_EXP_OUT>/tins_replay.jsonl.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import reneg.core as RC  # noqa: E402
from reneg.reneg_tanl import logit  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.paths import RESULTS, out  # noqa: E402
from replaylab.replay import MODES, append, done, replay_base, replay_x  # noqa: E402
from replaylab.traces import ID_OF, load_trace, stream_for, text_from_trace  # noqa: E402

TINS_DEFAULT = os.path.join(RESULTS, "p4_tins", "tins_traces_1.zip")
SETS = {"openood_v15": list(D.OOD_SETS), "four_ood": list(D.FOUR)}
OUTJ = "tins_replay.jsonl"
GROUPS = {"near": ("openood_v15", list(D.NEAR)), "far": ("openood_v15", list(D.FAR)),
          "Four-OOD 45k": ("four_ood_45k", list(D.FOUR))}


class TinsTraces:
    """P4's traces: <protocol>__<set>__s<order>.npz with score (S_TINS per stream image), rows (stream order),
    is_ood, pre (score before inversion), cand (inversion candidates), kept, n_inv, t_inv."""

    def __init__(self, src):
        self.src = src
        self.zip = zipfile.ZipFile(src) if src.endswith(".zip") else None

    def has(self, protocol, ds, seed):
        name = f"{protocol}__{ds}__s{seed}.npz"
        return name in self.zip.namelist() if self.zip else os.path.isfile(os.path.join(self.src, name))

    def load(self, protocol, ds, seed):
        name = f"{protocol}__{ds}__s{seed}.npz"
        return np.load(io.BytesIO(self.zip.read(name))) if self.zip else np.load(os.path.join(self.src, name))


def lgt(p, eps=1e-7):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def vote_all(lt, batch=256):
    v, res = RC.VoteHistory(), np.zeros(len(lt), bool)
    for s in range(0, len(lt), batch):
        res[s:s + batch] = v.vote(lt[s:s + batch])
    return res


def base_inputs(base, protocol, ds, seed, tins):
    """(lt_base, vote_lt, base_mask or None) of one base on one stream."""
    if base == "tanl":
        tr = load_trace(ds, seed, ID_OF[protocol])
        lt = logit(text_from_trace(tr["a"], tr["cum"], tr["step"], None, 0.0)).double().numpy()
        return lt, logit(torch.as_tensor(tr["tanl"])).double().numpy(), np.asarray(tr["id_mask"], bool)
    S = tins["score"].astype(np.float64)
    if base == "tins":
        return lgt(S), lgt(S), None
    if base == "tins@0.7":
        return lgt(S), lgt(S), S >= 0.7
    if base == "tanl+tins":
        tr = load_trace(ds, seed, ID_OF[protocol])
        lt = 0.5 * (lgt(tr["tanl"]) + lgt(S))
        return lt, lt, np.asarray(tr["id_mask"], bool) & ~vote_all(lt)
    raise ValueError(base)


def configs(tier=None):
    """name -> (base, mode or None, prior); mode None = the base alone. Tier 1 decides, tiers 2-3 explain,
    tier 0 = the TANL rows on the same streams."""
    t0 = {"TANL": ("tanl", None, False), "balanced on tanl, 5-shot": ("tanl", "balanced", True),
          "max on tanl, 5-shot": ("tanl", "max", True)}
    t1 = {"TINS": ("tins", None, False), "balanced on tins, 5-shot": ("tins", "balanced", True),
          "TANL+TINS": ("tanl+tins", None, False), "balanced on tanl+tins, 5-shot": ("tanl+tins", "balanced", True)}
    t2 = {"max on tins, 5-shot": ("tins", "max", True), "safe on tins, 5-shot": ("tins", "safe", True),
          "balanced on tins": ("tins", "balanced", False), "balanced on tins@0.7, 5-shot": ("tins@0.7", "balanced", True)}
    t3 = {"max on tanl+tins, 5-shot": ("tanl+tins", "max", True), "safe on tanl+tins, 5-shot": ("tanl+tins", "safe", True),
          "max on tins": ("tins", "max", False), "safe on tins": ("tins", "safe", False),
          "max on tins@0.7, 5-shot": ("tins@0.7", "max", True), "safe on tins@0.7, 5-shot": ("tins@0.7", "safe", True)}
    tiers = {0: t0, 1: t1, 2: t2, 3: t3}
    return tiers[tier] if tier is not None else {**t0, **t1, **t2, **t3}


def selftest():
    """The generic replay reproduces replay_x (with and without the prior) when the base is TANL."""
    P, cl = D.pool_subset("blind")
    tr = load_trace("ninco", 0, "imagenet_test")
    X, is_ood, p = stream_for("ninco", 0, "imagenet_test", tr)
    lt, vl, bm = base_inputs("tanl", "openood_v15", "ninco", 0, None)
    worst = 0.0
    for mode, prior in (("balanced", False), ("max", True)):
        cfg = RC.ReNegConfig(**MODES[mode])
        opts = {"idadm": "softmax+tanl"}
        ref = replay_x(tr, X, None, P, cl, cfg, opts, prior=prior)
        new = replay_base(lt, vl, bm, X, P, cl, cfg, opts, prior=prior)
        d = float(np.abs(ref - new).max())
        worst = max(worst, d)
        print(f"selftest {mode} prior={prior}: max |difference| {d:.2e}")
    print("PASS" if worst < 1e-9 else "FAIL")
    return worst < 1e-9


def run(traces: TinsTraces, path, seeds=(0, 1, 2), tiers=(0, 1, 2, 3), datasets=None, log=print):
    have = done(path, ("config", "seed", "protocol", "set"))
    P, cl = D.pool_subset("blind")
    for tier in tiers:
        cf = configs(tier)
        for seed in seeds:
            for protocol in ("openood_v15", "four_ood"):
                for ds in SETS[protocol]:
                    if datasets and ds not in datasets:
                        continue
                    todo = [n for n in cf if (n, seed, protocol, ds) not in have]
                    if not todo or not traces.has(protocol, ds, seed):
                        continue
                    tins = traces.load(protocol, ds, seed)
                    X, is_ood, p = D.pair_stream(D.load_set(ID_OF[protocol])["X"], D.load_set(ds)["X"], seed)
                    if not ((tins["rows"] == p).all() and (tins["is_ood"] == is_ood).all()):
                        raise ValueError(f"stream order of the TINS trace {protocol}/{ds}/s{seed} disagrees")
                    ev45 = D.eval_mask_45k(p, len(D.load_set(ds)["labels"])) if protocol == "four_ood" else None
                    for name in todo:
                        base, mode, prior = cf[name]
                        lt, vl, bm = base_inputs(base, protocol, ds, seed, tins)
                        if mode is None:
                            s = vl if base == "tanl" else lt
                        else:
                            s = replay_base(lt, vl, bm, X, P, cl, RC.ReNegConfig(**MODES[mode]), {"idadm": "softmax+tanl"},
                                            prior=prior)
                        evs = [("all", np.ones(len(X), bool))] + ([("45k", ev45)] if protocol == "four_ood" else [])
                        for ev_name, ev in evs:
                            m = D.metrics(s[ev & ~is_ood], s[ev & is_ood])
                            append(path, {"config": name, "seed": seed, "set": ds,
                                          "protocol": protocol if ev_name == "all" else "four_ood_45k",
                                          "fpr95": m["fpr95"], "fpr95_idpos": m["fpr95_idpos"], "auroc": m["auroc"]})
                        have.add((name, seed, protocol, ds))
                        log(f"[tier {tier}] {name} order {seed} {protocol} {ds}: ood+ {m['fpr95']:.2f} | "
                            f"id+ {m['fpr95_idpos']:.2f}")


def report(path, ref="TINS", cand="balanced on tins, 5-shot"):
    df = pd.DataFrame(map(json.loads, open(path))).drop_duplicates(["config", "seed", "protocol", "set"], keep="last")

    def gseed(cfg, prot, sets_, col):
        g = df[(df.config == cfg) & (df.protocol == prot) & df.set.isin(sets_)]
        n = g.groupby("seed").size()
        g = g[g.seed.isin(n[n == len(sets_)].index)]
        return g.groupby("seed")[col].mean()

    order = [c for c in configs() if c in set(df.config)]
    rows = []
    for cfg in order:
        row = {"config": cfg, "orders": int(df[df.config == cfg].seed.nunique())}
        for gname, (prot, sets_) in GROUPS.items():
            for col, short in (("fpr95_idpos", "id+"), ("fpr95", "ood+"), ("auroc", "AUROC")):
                s = gseed(cfg, prot, sets_, col)
                if len(s) == 0:
                    continue
                row[f"{gname} {short}"] = round(s.mean(), 2)
                if col == "fpr95_idpos" and cfg != ref:
                    d = (s - gseed(ref, prot, sets_, col)).dropna()
                    if len(d):
                        row[f"{gname} id+ vs {ref}"] = f"{d.mean():+.2f} [{d.min():+.2f},{d.max():+.2f}]"
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    for gname in GROUPS:
        cols = ["config", "orders"] + [c for c in res.columns if c.startswith(gname + " ")]
        print(f"\n== {gname} (mean of stream orders; paired difference vs {ref} [min, max over orders])")
        print(res[cols].to_string(index=False))
    for col, title in (("fpr95_idpos", "ID-positive"), ("fpr95", "OOD-positive")):
        sub = df[df.protocol != "four_ood"]
        ps = sub.groupby(["config", "protocol", "set"])[col].mean().unstack(["protocol", "set"]).round(2)
        print(f"\n== per set, {title} FPR95 (mean of orders)")
        print(ps.reindex([c for c in order if c in ps.index]).to_string())
    ok = True
    print(f"\n== pass rule for '{cand}' (ID-positive FPR95 vs {ref})")
    for gname, (prot, sets_) in GROUPS.items():
        d = (gseed(cand, prot, sets_, "fpr95_idpos") - gseed(ref, prot, sets_, "fpr95_idpos")).dropna()
        need_every = gname in ("near", "far")
        passed = len(d) == 3 and d.mean() < 0 and (not need_every or bool((d < 0).all()))
        ok &= passed
        print(f"{gname:14s} mean diff {d.mean():+.2f}, per order {np.round(d.values, 2).tolist()} -> "
              f"{'PASS' if passed else 'FAIL'}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["selftest", "run", "report"])
    ap.add_argument("--traces", default=os.environ.get("RENEG_TINS_TRACES", TINS_DEFAULT),
                    help="folder or zip with P4's TINS traces")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--tiers", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--datasets", nargs="*", help="only these OOD sets")
    ap.add_argument("--file", default=OUTJ, help="rows file in RENEG_EXP_OUT, or a path (results/replays/tins_replay.jsonl)")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    path = a.file if (a.cmd == "report" and os.path.isfile(a.file)) else out(a.file)
    if a.cmd == "selftest":
        return 0 if selftest() else 1
    if a.cmd == "run":
        run(TinsTraces(a.traces), path, a.seeds, a.tiers, a.datasets)
        return 0
    return 0 if report(path) else 1


if __name__ == "__main__":
    sys.exit(main())
