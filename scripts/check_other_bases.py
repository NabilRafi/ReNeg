#!/usr/bin/env python
"""Summary and checks for configs/other_bases.yaml (ReNeg's heads on NegLabel, AdaNeg and DDE).

usage: python scripts/check_other_bases.py RUNS_CSV [--baselines results/baselines/04_reproduce_adaneg/adaneg_runs.csv]
                                                   [--out summary.csv]

Prints, per base, the FPR95 group means (near = SSB-hard + NINCO, far = iNaturalist + Textures + OpenImage-O,
Four-OOD = iNaturalist + SUN + Places + Textures-all; mean over the stream orders) in both conventions (OOD+ =
OpenOOD's, ID+ = TINS's), AUROC, the paired change of each head against its base [min, max over orders], and the
pre-registered pass rule: balanced on <base> is better than <base> on all three group means, on near and far in
every order (the three single-precision bases; the half-precision bases neglabel_fp16 and adaneg_fp16 are printed
beside them as exploratory rows). Also checks that the run is complete, that the AdaNeg and half-precision rows
agree with the shipped baseline runs (Table 1), and that the DDE row is close to the numbers in its paper.
"""
import argparse
import sys

import pandas as pd

GROUPS = {"near": ("openood_v15", ["ssb_hard", "ninco"]),
          "far": ("openood_v15", ["inaturalist", "textures", "openimage_o"]),
          "four": ("four_ood", ["inaturalist", "sun", "places", "textures_all"])}
BASES = ["neglabel", "adaneg", "dde"]                             # under the pre-registered pass rule
EXTRA = ["neglabel_fp16", "adaneg_fp16"]                          # half-precision bases: reported, not under the rule
MODES = ["balanced", "max", "safe"]
DDE_PAPER = {"near": 58.64, "far": 15.58, "four": 9.80}          # FPR95 (OOD positive) in Zhu et al., ECCV 2026
DDE_TOL = {"near": 1.5, "far": 1.5, "four": 1.0}
SHIPPED = {"neglabel_fp16": ("neglabel", "g100/fp16"), "adaneg_fp16": ("adaneg", "fp16"), "adaneg": ("adaneg", "fp32")}
MEMORYLESS = {"neglabel", "neglabel_fp16"}                        # the shipped run has one order


def per_order(df, label):
    res = {}
    for g, (proto, sets) in GROUPS.items():
        x = df[(df.label == label) & (df.protocol == proto) & df.dataset.isin(sets)]
        cnt = x.groupby("seed")["fpr95"].count()
        res[g] = x[x.seed.isin(cnt[cnt == len(sets)].index)].groupby("seed")[["fpr95", "fpr95_idpos", "auroc"]].mean()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    df = df[df.protocol.isin(["openood_v15", "four_ood"])].drop_duplicates(["label", "protocol", "dataset", "seed"], keep="last")
    ok, rows = True, []

    print("rows per label (expect 27 = 9 sets x 3 orders):")
    for b in BASES + EXTRA:
        for lab in [b] + [f"{m} on {b}" for m in MODES]:
            c = int((df.label == lab).sum())
            if b in EXTRA and c == 0:
                print(f"  {lab:<26}   -  not run (optional half-precision row)")
                continue
            print(f"  {lab:<26} {c:3d}  {'OK' if c == 27 else 'MISSING ' + str(27 - c)}")
            ok &= c == 27

    passed = {}
    for b in BASES + EXTRA:
        if not (df.label == b).any():
            continue
        pb = per_order(df, b)
        print(f"\n=== {b}" + ("   (half precision: exploratory, not under the pass rule)" if b in EXTRA else ""))
        print(f"{'FPR95 OOD+ (ID+)  AUROC':<24}" + "".join(f"{g:>24}" for g in GROUPS))
        line = lambda p: "".join(f"{p[g].fpr95.mean():>9.2f} ({p[g].fpr95_idpos.mean():5.2f}) {p[g].auroc.mean():6.2f}" for g in GROUPS)  # noqa: E731
        print(f"{b:<24}" + line(pb))
        rows.append({"label": b, **{f"{g}_{k}": pb[g][c].mean() for g in GROUPS for k, c in
                                    (("ood+", "fpr95"), ("id+", "fpr95_idpos"), ("auroc", "auroc"))}})
        for m in MODES:
            lab = f"{m} on {b}"
            if not (df.label == lab).any():
                continue
            ph = per_order(df, lab)
            print(f"{'+ ' + m:<24}" + line(ph))
            rec = {"label": lab, **{f"{g}_{k}": ph[g][c].mean() for g in GROUPS for k, c in
                                    (("ood+", "fpr95"), ("id+", "fpr95_idpos"), ("auroc", "auroc"))}}
            verdict, txt = {}, []
            for conv, tag in (("fpr95", "OOD+"), ("fpr95_idpos", "ID+")):
                verdict[tag] = True
                for g in GROUPS:
                    d = (ph[g][conv] - pb[g][conv]).dropna()
                    verdict[tag] &= bool(len(d) == 3 and d.mean() < 0 and (g == "four" or (d < 0).all()))
                    if len(d):
                        txt.append(f"{g} {tag} {d.mean():+.2f} [{d.min():+.2f}, {d.max():+.2f}]")
                        rec[f"{g}_d_{tag.lower()}"] = d.mean()
                        rec[f"{g}_d_{tag.lower()}_min"], rec[f"{g}_d_{tag.lower()}_max"] = d.min(), d.max()
            print("    " + "; ".join(txt[:3]))
            print("    " + "; ".join(txt[3:]))
            print(f"    pass rule: OOD+ {'PASS' if verdict['OOD+'] else 'FAIL'}, ID+ {'PASS' if verdict['ID+'] else 'FAIL'}"
                  + ("   <- the pre-registered row" if m == "balanced" and b in BASES else ""))
            rec["pass_ood+"], rec["pass_id+"] = verdict["OOD+"], verdict["ID+"]
            rows.append(rec)
            if m == "balanced" and b in BASES:
                passed[b] = verdict

    print("\nchecks:")
    if (df.label == "dde").any():
        pb = per_order(df, "dde")
        for g in GROUPS:
            if len(pb[g]):
                gap = pb[g].fpr95.mean() - DDE_PAPER[g]
                good = abs(gap) <= DDE_TOL[g]
                print(f"  DDE {g}: ours {pb[g].fpr95.mean():.2f}, paper {DDE_PAPER[g]:.2f} ({gap:+.2f}) "
                      f"{'OK' if good else 'CHECK: outside +-' + str(DDE_TOL[g])}")
                ok &= bool(good)
    if a.baselines:
        ref = pd.read_csv(a.baselines)
        key = ["protocol", "dataset", "seed"]
        for b, (meth, conf) in SHIPPED.items():
            r = ref[(ref.method == meth) & (ref.config == conf)]
            if b in MEMORYLESS:                      # memoryless: the shipped run has one order
                m = df[df.label == b].merge(r[["protocol", "dataset", "fpr95"]].drop_duplicates(["protocol", "dataset"]),
                                            on=["protocol", "dataset"], suffixes=("", "_ref"))
            else:
                m = df[df.label == b].merge(r[key + ["fpr95"]], on=key, suffixes=("", "_ref"))
            if len(m):
                dif = (m.fpr95 - m.fpr95_ref).abs()
                good = dif.max() <= 0.5
                print(f"  {b} vs the shipped baseline run ({len(m)} set x order pairs): max |FPR95 difference| "
                      f"{dif.max():.3f}, mean {dif.mean():.3f}  {'OK' if good else 'NOTE: differs'}")
    for b, v in passed.items():
        print(f"  pass rule, balanced on {b}: OOD+ {'PASS' if v['OOD+'] else 'FAIL'}, ID+ {'PASS' if v['ID+'] else 'FAIL'}")
    if a.out:
        pd.DataFrame(rows).round(3).to_csv(a.out, index=False)
        print("\nwrote", a.out)
    print("\nRUN COMPLETE AND CHECKS OK" if ok else "\nNOT COMPLETE OR A CHECK NEEDS A LOOK (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
