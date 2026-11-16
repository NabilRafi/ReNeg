#!/usr/bin/env python
"""Probe P5: ReNeg's heads on detectors other than TANL and TINS (NegLabel, AdaNeg, DDE), replayed on the 8-bit
feature exports with the packaged classes (``reneg.multi.HeadsOnBase``, ``reneg.dde.DDEFeat``).

    python experiments/12_other_bases.py run --base neglabel      # also: adaneg, dde, neglabel_fp16, ... (resumable)
    python experiments/12_other_bases.py report                   # tables in both conventions and the pass rule
    python experiments/12_other_bases.py hu --datasets ninco places   # the Hu et al. reproduction attempt

The head reads only the base's score: lt = logit(score), vote = lt below its running Otsu threshold, base ID mask =
not voted OOD (the recipe used for TINS). Same head, same settings on every base; nothing is tuned per base.

Pass rule (fixed on Nov 14, 2026 before the first number; results/p5_other_bases/PREREGISTRATION_P5.md):
balanced on <base> has a lower FPR95 than <base> on the near, far and Four-OOD group means (mean of three stream
orders) in OpenOOD's convention, with the same sign in every order for near and far. ID-positive beside it.

Bases: neglabel (10,000 negatives, 100 groups), adaneg (official settings), both in float32; dde (official score
with the paper's settings and the whole corpus as the selection bank, which reproduces the paper);
dde_release_settings and dde_released (the release's settings; the release as it is); neglabel_fp16 and adaneg_fp16
(the official half-precision arithmetic, exploratory: confident scores round to 1, smaller gain on NegLabel); mcm (exploratory). Rows: <RENEG_EXP_OUT>/other_bases.jsonl; shipped rows:
results/p5_other_bases/other_bases.jsonl (``report --file``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import reneg.core as RC  # noqa: E402
from reneg.multi import HeadsOnBase  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.paths import RESULTS, out  # noqa: E402
from replaylab.replay import MODES  # noqa: E402

PROTOS = {"openood_v15": ("imagenet_test", list(D.OOD_SETS)), "four_ood": ("imagenet_val_all", list(D.FOUR))}
GROUPS = {"near": ("openood_v15", list(D.NEAR)), "far": ("openood_v15", list(D.FAR)), "Four-OOD": ("four_ood", list(D.FOUR))}
OUTJ = "other_bases.jsonl"
PUBLISHED = {  # FPR95 as printed in the papers (near, far, Four-OOD)
    "dde": (58.64, 15.58, 9.80), "hu": (52.86, 13.22, 12.79), "neglabel": (69.45, 23.73, 25.40),
    "adaneg": (67.51, 17.31, 18.92)}


def make_base(name: str):
    """(detector, largest possible score)."""
    from oodlab.methods.adaneg import AdaNeg, AdaNegConfig
    from oodlab.methods.neglabel import MCM, MCMConfig, NegLabel, NegLabelConfig
    from reneg.dde import DDEConfig, DDEFeat

    tb = D.textbank()
    if name == "neglabel":
        return NegLabel(D.id_text, D.neg_text, NegLabelConfig()), 1.0
    if name == "adaneg":
        cfg = AdaNegConfig.official()
        return AdaNeg(D.id_text, D.neg_text, cfg), 1.0 + cfg.lambda_
    if name == "neglabel_fp16":                # the official half-precision arithmetic (Table 1's row)
        return NegLabel(D.id_text, D.neg_text, NegLabelConfig(emulate_fp16=True)), 1.0
    if name == "adaneg_fp16":
        cfg = AdaNegConfig.official().with_(emulate_fp16=True, proxy_chunk=1024)
        return AdaNeg(D.id_text, D.neg_text, cfg), 1.0 + cfg.lambda_
    if name == "dde":                          # the paper's settings, selection bank = the whole corpus
        return DDEFeat(D.id_text, tb["corpus_text"].float(), torch.as_tensor(tb["noise_feats"]), DDEConfig.paper()), 1.0
    if name == "dde_release_settings":         # whole corpus, the release's thresholds and caches
        return DDEFeat(D.id_text, tb["corpus_text"].float(), torch.as_tensor(tb["noise_feats"]), DDEConfig.release()), 1.0
    if name == "dde_released":                 # the release as it is: the 10,000 negatives twice
        return DDEFeat(D.id_text, D.neg_text, torch.as_tensor(tb["noise_feats"]),
                       DDEConfig.release().with_(duplicate_negatives=True)), 1.0
    if name == "mcm":
        return MCM(D.id_text, MCMConfig.paper()), 1.0
    if name in ("hu", "hu_union"):
        from replaylab.hu import HuConfig, HuFeat

        return HuFeat(D.id_text, D.neg_text, HuConfig(label_space="union" if name == "hu_union" else "split")), 1.0
    raise ValueError(name)


def run(base_name, seeds, modes, path, datasets=None, protocols=None, log=print):
    have = set()
    if os.path.isfile(path):
        with open(path) as fh:
            have = {(r["label"], r["protocol"], r["dataset"], r["seed"]) for r in map(json.loads, fh)}
    P, cl = D.pool_subset("blind")
    rcs = {m: RC.ReNegConfig(**MODES[m]).with_(id_admit="softmax+base") for m in modes}
    for seed in seeds:
        for proto in protocols or list(PROTOS):
            id_name, sets = PROTOS[proto]
            for ds in sets:
                if datasets and ds not in datasets:
                    continue
                labels = [base_name] + [f"{m} on {base_name}" for m in modes]
                if all((l, proto, ds, seed) in have for l in labels):
                    continue
                t0 = time.time()
                X, is_ood, _ = D.pair_stream(D.load_set(id_name)["X"], D.load_set(ds)["X"], seed)
                base, smax = make_base(base_name)
                m = HeadsOnBase(base, base_name, D.id_text, P, cl, rcs, score_max=smax)
                sc = {l: np.zeros(len(X)) for l in labels}
                for s in range(0, len(X), 256):
                    o = m.step(X[s:s + 256])
                    for l in labels:
                        sc[l][s:s + 256] = o.info["scores"][l].double().numpy()
                line = []
                for l in labels:
                    mt = D.metrics(sc[l][~is_ood], sc[l][is_ood])
                    rec = {"label": l, "base": base_name, "protocol": proto, "dataset": ds, "seed": seed,
                           "fpr95": mt["fpr95"], "fpr95_idpos": mt["fpr95_idpos"], "auroc": mt["auroc"],
                           "n_id": int((~is_ood).sum()), "n_ood": int(is_ood.sum())}
                    with open(path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    line.append(f"{l} {mt['fpr95']:.2f} ({mt['fpr95_idpos']:.2f})")
                log(f"{proto}:{ds:<13} order {seed}  " + " | ".join(line) + f"  {time.time() - t0:.0f}s")


def _per_order(df, label, col):
    res = {}
    for g, (proto, sets) in GROUPS.items():
        x = df[(df.label == label) & (df.protocol == proto) & df.dataset.isin(sets)]
        cnt = x.groupby("seed")[col].count()
        res[g] = x[x.seed.isin(cnt[cnt == len(sets)].index)].groupby("seed")[col].mean()
    return res


def report(path, modes=("balanced", "max", "safe"), log=print):
    with open(path) as fh:
        df = pd.DataFrame([json.loads(l) for l in fh]).drop_duplicates(["label", "protocol", "dataset", "seed"], keep="last")
    verdicts = {}
    for base in [b for b in ("neglabel", "adaneg", "dde", "dde_released", "mcm", "hu", "hu_union") if (df.base == b).any()]:
        b = {c: _per_order(df, base, c) for c in ("fpr95", "fpr95_idpos", "auroc")}
        log(f"\n=== {base}: stream orders " + ", ".join(f"{g} {len(b['fpr95'][g])}" for g in GROUPS)
            + (f"; paper {PUBLISHED[base][0]} / {PUBLISHED[base][1]} / {PUBLISHED[base][2]}" if base in PUBLISHED else ""))
        log(f"{'FPR95 OOD+ (ID+), AUROC':<26}" + "".join(f"{g:>26}" for g in GROUPS))
        fmt = lambda h: "".join(f"{h['fpr95'][g].mean():>11.2f} ({h['fpr95_idpos'][g].mean():5.2f}) {h['auroc'][g].mean():6.2f}" for g in GROUPS)  # noqa: E731
        log(f"{base:<26}" + fmt(b))
        tol = dict(zip(GROUPS, (1.5, 1.5, 1.0)))                 # reproduction tolerance of a re-implemented base
        off_rule = base in ("dde", "hu", "hu_union") and not all(
            len(b["fpr95"][g]) and abs(b["fpr95"][g].mean() - pub) <= tol[g]
            for g, pub in zip(GROUPS, PUBLISHED["hu" if base.startswith("hu") else base]))
        if base in ("dde", "hu", "hu_union"):
            log("    reproduction of the paper (1.5 / 1.5 / 1.0 FPR95): " + ("outside the tolerance: rows below are "
                "exploratory, not under the pass rule" if off_rule else "inside the tolerance"))
        for m in modes:
            lab = f"{m} on {base}"
            if not (df.label == lab).any():
                continue
            h = {c: _per_order(df, lab, c) for c in ("fpr95", "fpr95_idpos", "auroc")}
            log(f"{'+ ' + m:<26}" + fmt(h))
            ok = {}
            parts = []
            for conv, tag in (("fpr95", "OOD+"), ("fpr95_idpos", "ID+")):
                ok[tag] = True
                for g in GROUPS:
                    d = (h[conv][g] - b[conv][g]).dropna()
                    ok[tag] &= bool(len(d) == 3 and d.mean() < 0 and (g == "Four-OOD" or (d < 0).all()))
                    if len(d):
                        parts.append(f"{g} {tag} {d.mean():+.2f} [{d.min():+.2f}, {d.max():+.2f}]")
            log("    " + "; ".join(parts[:3]))
            log("    " + "; ".join(parts[3:]))
            log(f"    pass rule: OOD+ {'PASS' if ok['OOD+'] else 'FAIL'}, ID+ {'PASS' if ok['ID+'] else 'FAIL'}"
                + ("   (not counted: the base does not reproduce its paper)" if off_rule else
                   "   <- the pre-registered row" if m == "balanced" and base in ("neglabel", "adaneg", "dde") else ""))
            verdicts[(base, m)] = ok
    return verdicts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--modes", nargs="*", default=["balanced", "max", "safe"])
    h = sub.add_parser("hu", help="base-only run of the Hu et al. attempt (both readings)")
    for p in (r, h):
        p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
        p.add_argument("--datasets", nargs="*")
        p.add_argument("--protocols", nargs="*", choices=list(PROTOS))
        p.add_argument("--out", default=OUTJ, help="file name in RENEG_EXP_OUT")
        p.add_argument("--threads", type=int, default=0)
    t = sub.add_parser("report")
    t.add_argument("--file", default=None, help="rows file (default: RENEG_EXP_OUT, else the shipped rows)")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        path = a.file or (out(OUTJ) if os.path.isfile(out(OUTJ)) else os.path.join(RESULTS, "p5_other_bases", OUTJ))
        report(path)
        return 0
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.cmd == "hu":
        for name in ("hu", "hu_union"):
            run(name, a.seeds[:1] if a.seeds == [0, 1, 2] else a.seeds, [], out(a.out), a.datasets, a.protocols)
    else:
        run(a.base, a.seeds, a.modes, out(a.out), a.datasets, a.protocols)
    return 0


if __name__ == "__main__":
    sys.exit(main())
