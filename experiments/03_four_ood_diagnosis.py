#!/usr/bin/env python
"""Why the first ReNeg release lost on Four-OOD (SUN, Places, Textures): per-image diagnosis of the balanced mode.

    python experiments/03_four_ood_diagnosis.py features      # per-image state at scoring time -> diag/*.npz
    python experiments/03_four_ood_diagnosis.py penalties     # who is penalised: ID or OOD, caught or not
    python experiments/03_four_ood_diagnosis.py id-images     # which ID images, where in the stream, which classes
    python experiments/03_four_ood_diagnosis.py names         # which KG names do the harm

Finding (Oct 3): the penalised ID images are hard ImageNet images that CLIP's softmax admits wrongly; their wrong
ID prototypes and nearby KG prototypes push them down. Agreed ID admission (softmax and TANL's mask) fixes it
(experiments/04_four_ood_fixes.py). Uses the packaged admission (the release that was diagnosed), stream order 0.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import reneg.core as RC  # noqa: E402
from reneg.reneg_tanl import logit  # noqa: E402
from reneg.transport import l2n  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.paths import out  # noqa: E402
from replaylab.replay import MODES  # noqa: E402
from replaylab.traces import load_trace, stream_for, text_from_trace  # noqa: E402

JOBS = [("imagenet_val_all", d) for d in ("sun", "places", "textures_all", "inaturalist")] + \
       [("imagenet_test", d) for d in ("ninco", "ssb_hard", "openimage_o", "textures")]


def diag_dir():
    d = out("diag")
    os.makedirs(d, exist_ok=True)
    return d


@torch.no_grad()
def diag_replay(trace, X, id_text, P_text, cl, rcfg, batch=256):
    """Per image, at scoring time (state left by earlier batches): lt (TANL log-odds), pen (balanced penalty),
    plain (plain image score from the same state), sid / cid (max cos to an ID prototype and its class), sid_lse,
    skg / jkg (strongest admitted KG prototype, pi-weighted, and its name), pi_j, nvis_j, rbar_j (concentration of
    j's caught images), n_c, rbar_c, tid (max cos to ID text), tkg (max cos to pool text), text_pred, fused."""
    dt = rcfg.torch_dtype
    T_id = l2n(torch.as_tensor(id_text).to(dt))
    gate = RC.KGGate(P_text, cl, rcfg)
    idm = RC.OnlineIdModel(T_id.shape[0], T_id.shape[1], rcfg)
    votes = RC.VoteHistory()
    F = {k: [] for k in ("lt", "pen", "plain", "sid", "cid", "sid_lse", "skg", "jkg", "pi_j", "nvis_j", "rbar_j",
                         "n_c", "rbar_c", "tid", "tkg", "text_pred", "fused")}
    for s in range(0, len(X), batch):
        Xr = l2n(X[s:s + batch].to(dt))
        B = len(Xr)
        a, cum = trace["a"][s:s + batch], trace["cum"][s:s + batch]
        lt = logit(text_from_trace(a, cum, trace["step"], None, 0.0)).to(dt)
        pen = RC.image_score(Xr, idm, gate, rcfg)
        plain = RC.image_score(Xr, idm, gate, rcfg.with_(image_mode="plain"))
        fused = RC.fuse(lt, pen, rcfg)
        St = Xr @ T_id.T
        tid, tpred = St.max(1)
        tkg = (Xr @ gate.T.T).max(1).values
        nan = torch.full((B,), float("nan"), dtype=dt)
        if idm.ready():
            el = np.nonzero(idm.counts >= rcfg.n_min)[0]
            Si = Xr @ idm.prototypes().T
            sid, ci = Si.max(1)
            cid = torch.as_tensor(el)[ci]
            sid_lse = torch.logsumexp(rcfg.tau * Si, 1) / rcfg.tau
            n_c = torch.as_tensor(idm.counts[cid.numpy()])
            rbar_c = idm.sums[cid].norm(dim=1) / n_c.clamp(min=1)
        else:
            sid, cid, sid_lse, n_c, rbar_c = nan, torch.full((B,), -1), nan, nan, nan
        Vp, lw = gate.visual_terms()
        if Vp is not None:
            vis = np.nonzero((gate.n_vis >= rcfg.k_vis) & (gate.pi > 1e-4))[0]
            Z = Xr @ Vp.T + lw[None, :] / rcfg.tau
            skg, jj = Z.max(1)
            jkg = torch.as_tensor(vis)[jj]
            pi_j = torch.as_tensor(gate.pi[jkg.numpy()])
            nvis_j = torch.as_tensor(gate.n_vis[jkg.numpy()])
            rbar_j = gate.fsum[jkg].norm(dim=1) / nvis_j.clamp(min=1)
        else:
            skg, jkg, pi_j, nvis_j, rbar_j = nan, torch.full((B,), -1), nan, nan, nan
        for k, v in (("lt", lt), ("pen", pen if pen is not None else nan), ("plain", plain if plain is not None else nan),
                     ("sid", sid), ("cid", cid), ("sid_lse", sid_lse), ("skg", skg), ("jkg", jkg), ("pi_j", pi_j),
                     ("nvis_j", nvis_j), ("rbar_j", rbar_j), ("n_c", n_c), ("rbar_c", rbar_c), ("tid", tid),
                     ("tkg", tkg), ("text_pred", tpred), ("fused", fused)):
            F[k].append(torch.as_tensor(v).double().numpy() if k not in ("cid", "jkg", "text_pred")
                        else torch.as_tensor(v).numpy())
        lab, adm = RC.admit_mask(Xr, T_id, rcfg, base_mask=trace["id_mask"][s:s + batch])      # as replay()
        vote = votes.vote(logit(torch.as_tensor(trace["tanl"][s:s + batch])).numpy()) if rcfg.vote else None
        caught = gate.update(Xr, T_id, ood_vote=vote)
        adm = adm & ~gate.exclusion(caught, rcfg.id_exclude)
        idm.add(Xr, lab, adm)
    return {k: np.concatenate(v) for k, v in F.items()}, gate, idm


def features(seed=0):
    cfg = RC.ReNegConfig(**MODES["balanced"])
    P, cl = D.pool_subset("full")
    for id_name, ds in JOBS:
        path = os.path.join(diag_dir(), f"{id_name}__{ds}.npz")
        if os.path.isfile(path):
            continue
        tr = load_trace(ds, seed, id_name)
        X, is_ood, _ = stream_for(ds, seed, id_name, tr)
        F, gate, idm = diag_replay(tr, X, D.id_text, P, cl, cfg)
        F.update(is_ood=is_ood, tanl_idmask=tr["id_mask"], tanl=tr["tanl"], gate_pi=gate.pi, gate_n=gate.n,
                 gate_nvis=gate.n_vis, gate_vsum=gate.vsum)
        np.savez_compressed(path, **F)
        r = D.metrics(F["fused"][~is_ood], F["fused"][is_ood])
        print(f"{ds}: balanced FPR95 {r['fpr95']:.2f} (admitted names {int((gate.pi > 0.5).sum())})", flush=True)


def fpr(conf, ood):
    """OpenOOD convention: OOD positive; FPR = share of ID at or below the score that keeps 95% of OOD."""
    c = np.quantile(conf[ood], 0.95, method="higher")
    return (conf[~ood] <= c).mean() * 100


def penalties():
    print(f"{'set':13s} {'TANL':>6s} {'bal':>6s} {'(off.)':>6s} {'OODpen':>6s} {'IDpen':>6s} {'caught':>6s} "
          f"{'%IDpen':>6s} {'%OODpen':>7s} {'%ID p&c':>7s} {'%OODp&c':>7s}")
    for p in sorted(glob.glob(os.path.join(diag_dir(), "*.npz"))):
        d = np.load(p)
        name = os.path.basename(p)[:-4].split("__")[1]
        ood = d["is_ood"].astype(bool)
        idd = ~ood
        pen = np.nan_to_num(d["pen"])
        lt, fused = d["lt"], d["fused"]
        caught = d["tkg"] > d["tid"]
        P = pen < -0.05
        r = (name, fpr(lt, ood), fpr(fused, ood),
             fpr(np.where(idd, 0.75 * lt, fused), ood),        # penalties kept on OOD images only
             fpr(np.where(ood, 0.75 * lt, fused), ood),        # penalties kept on ID images only
             fpr(np.where(caught, fused, 0.75 * lt), ood),     # safe-like: penalty only where caught in text
             P[idd].mean() * 100, P[ood].mean() * 100, (P & caught)[idd].mean() * 100, (P & caught)[ood].mean() * 100,
             D.metrics(fused[idd], fused[ood])["fpr95"])
        print(f"{r[0]:13s} {r[1]:6.2f} {r[2]:6.2f} {r[10]:6.2f} {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.2f} {r[6]:6.2f} "
              f"{r[7]:7.2f} {r[8]:7.2f} {r[9]:7.2f}")


def id_images(seed=0):
    for id_name, ds in (("imagenet_val_all", "places"), ("imagenet_val_all", "textures_all"),
                        ("imagenet_val_all", "sun"), ("imagenet_test", "ninco")):
        d = np.load(os.path.join(diag_dir(), f"{id_name}__{ds}.npz"))
        yid = D.load_set(id_name)["labels"]
        n_ood = len(D.load_set(ds)["labels"])
        _, is_ood, p = D.pair_stream(torch.zeros(len(yid), 1), torch.zeros(n_ood, 1), seed)
        n = len(d["pen"])
        is_ood, p = is_ood[:n], p[:n]
        y = np.where(is_ood, -1, np.r_[np.full(n_ood, -1), yid][p])
        idd = ~is_ood
        pen = np.nan_to_num(d["pen"])
        P = pen < -0.05
        pos = np.arange(len(pen)) / len(pen)
        cid, tp = d["cid"], d["text_pred"]
        own_nearest = cid == y
        print(f"\n{ds}: ID images penalised {P[idd].mean() * 100:.1f}%")
        for lo, hi in ((0, .1), (.1, .3), (.3, .6), (.6, 1.0)):
            m = idd & (pos >= lo) & (pos < hi)
            print(f"  stream {lo:.1f}-{hi:.1f}: penalised {P[m].mean() * 100:5.1f}%  mean pen {pen[m].mean():6.3f}")
        print(f"  penalised ID: nearest ID prototype is own class {own_nearest[idd & P].mean() * 100:.1f}% "
              f"(unpenalised {own_nearest[idd & ~P].mean() * 100:.1f}%); text-pred correct "
              f"{np.mean(tp[idd & P] == y[idd & P]) * 100:.1f}% (unpen. {np.mean(tp[idd & ~P] == y[idd & ~P]) * 100:.1f}%)")
        print(f"  support n_c of nearest ID proto: pen median {np.nanmedian(d['n_c'][idd & P]):.0f} vs unpen "
              f"{np.nanmedian(d['n_c'][idd & ~P]):.0f};  KG proto nvis: pen median {np.nanmedian(d['nvis_j'][idd & P]):.0f}; "
              f"OOD pen median {np.nanmedian(d['nvis_j'][~idd & P]):.0f}")
        print(f"  concentration rbar: ID proto (pen) {np.nanmedian(d['rbar_c'][idd & P]):.3f}; KG proto on pen-ID "
              f"{np.nanmedian(d['rbar_j'][idd & P]):.3f}; KG proto on pen-OOD {np.nanmedian(d['rbar_j'][~idd & P]):.3f}")
        print(f"  sid-skg (cos): pen-ID median {np.nanmedian((d['sid'] - d['skg'])[idd & P]):.4f}; pen-OOD "
              f"{np.nanmedian((d['sid'] - d['skg'])[~idd & P]):.4f}")
        print(f"  caught in text: pen-ID {np.mean((d['tkg'] > d['tid'])[idd & P]) * 100:.1f}%; pen-OOD "
              f"{np.mean((d['tkg'] > d['tid'])[~idd & P]) * 100:.1f}%")
        print(f"  TANL logit lt: pen-ID median {np.median(d['lt'][idd & P]):.2f}, unpen-ID {np.median(d['lt'][idd & ~P]):.2f}, "
              f"pen-OOD {np.median(d['lt'][~idd & P]):.2f}")
        C = len(D.id_names)
        cls, cnt = np.unique(y[idd & P], return_counts=True)
        tot = np.bincount(y[idd], minlength=C)
        rate = np.zeros(C)
        rate[cls] = cnt / np.maximum(tot[cls], 1)
        top = np.argsort(-rate)[:12]
        print("  most penalised classes (rate):", [(D.id_names[c], round(float(rate[c]), 2)) for c in top])
        print("  share of ID classes with rate > 0.5:", round(float((rate > 0.5).mean()) * 100, 1), "%")


def names():
    nm = D.pool_names
    for id_name, ds in (("imagenet_val_all", "places"), ("imagenet_val_all", "textures_all"),
                        ("imagenet_test", "ninco"), ("imagenet_test", "ssb_hard")):
        d = np.load(os.path.join(diag_dir(), f"{id_name}__{ds}.npz"))
        ood = d["is_ood"].astype(bool)
        pen = np.nan_to_num(d["pen"])
        P = pen < -0.05
        j = d["jkg"].astype(int)
        K = len(nm)
        sel_id, sel_ood = P & ~ood & (j >= 0), P & ood & (j >= 0)
        pid, pood = np.bincount(j[sel_id], minlength=K), np.bincount(j[sel_ood], minlength=K)
        wid = np.bincount(j[sel_id], weights=-pen[sel_id], minlength=K)
        n, nvis, vsum, pi = d["gate_n"], d["gate_nvis"], d["gate_vsum"], d["gate_pi"]
        share = (vsum + 1) / (n + 2)
        act = (pid + pood) > 0
        harm, use = act & (pid > pood), act & (pood >= pid)
        print(f"\n{ds}: names penalising anything {act.sum()}, ID-dominated {harm.sum()} (ID pens {pid[harm].sum()}), "
              f"OOD-dominated {use.sum()} (OOD pens {pood[use].sum()})")
        for lab, m in (("ID-dominated", harm), ("OOD-dominated", use)):
            if m.any():
                print(f"  {lab:14s} end n median {np.median(n[m]):5.0f}  nvis {np.median(nvis[m]):5.0f}  "
                      f"vote share {np.median(share[m]):.2f}  pi {np.median(pi[m]):.2f}")
        top = np.argsort(-wid)[:15]
        print("  top ID-harming names:", [(nm[k], int(pid[k]), int(pood[k]), round(float(pi[k]), 2), int(n[k])) for k in top])
        tot, srt = wid.sum(), np.sort(wid)[::-1]
        if tot > 0:
            print(f"  ID-penalty mass from top 50 / 200 / 500 names: {srt[:50].sum() / tot * 100:.0f}% / "
                  f"{srt[:200].sum() / tot * 100:.0f}% / {srt[:500].sum() / tot * 100:.0f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["features", "penalties", "id-images", "names", "all"])
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    for c in (["features", "penalties", "id-images", "names"] if a.cmd == "all" else [a.cmd]):
        {"features": features, "penalties": penalties, "id-images": id_images, "names": names}[c]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
