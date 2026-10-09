"""Gate G1b: corrected, pre-registered diagnostics for Seam 1 (the text-to-image transport).

Why G1b exists. G1 (v1) returned FAIL, but three of its tests could not answer its question:

1. Part B ranked image centroids for each mapped *name* (text -> image). The OOD score works the
   other way: for each *image* it compares all prototypes. In the text -> image direction a shift
   shared by every name (the gap shift R1, and much of what R2/B2 learn) adds the same number to a
   centroid's similarity for every query, which favours "generic" centroids (hubness). For R1 this
   is exact: its only effect on the text -> image ranking is the query-independent term
   (m_V - m_T) . c_k. The score never sees such a term: for one image it adds the same amount to
   every prototype and cancels in the log-sum-exp difference.
2. Part D used CLIP's text temperature (about 100) for image-space similarities, whose spread is
   several times larger, so the image-space score saturated at 0 or 1 and the fusion collapsed
   onto the text score. It also demanded a near-OOD gain, which no map can give while the
   negatives are NegLabel's far-mined words: near-OOD negatives are Seam 2's job.
3. Part E replays the chosen map; with R0 chosen it only re-scored the text score.

What G1b measures (every rule is fixed in G1bConfig before the run):

B'  Leave-subtree-out on ID classes, scored image -> names, in raw and in centred geometry:
    name top-1 among held-out classes, how often a held-out name "catches" its own images
    against all ID prototypes, how often it "steals" ID images, and a pseudo-OOD test (held-out
    subtrees as OOD, their mapped names plus 3,000 NegLabel negatives as the negatives).
H   Does a map hamper CLIP? ImageNet validation top-1 with mapped ID prototypes.
D'  Static OpenOOD v1.5 scores with the temperature and the fusion weight tuned on OpenOOD's
    validation split (ImageNet val vs OpenImage-O val), log-odds fusion, raw and centred geometry.
O   Oracle near negatives (diagnostic only: it uses OOD class names and images). The names of
    SSB-hard and NINCO classes are added as negatives: as text, as mapped prototypes, and as the
    classes' real image centroids from half of their images (a perfect transport), evaluated on
    the other half. This bounds what Seam 2 can gain and what any better map (Plan B) can add.
E'  Cold start, only if D' keeps a map.

Geometry. ``raw``: cosine of unit vectors, as CLIP does. ``cent``: cosine of deviations from the
mean, images minus the stream's mean image (no labels) and prototypes minus their own mean
(the ID-text mean for raw text, the map's m_V for mapped prototypes). In the centred geometry the
gap shift R1 is exactly raw text again (R1(t) - m_V = t - m_T), so R0-centred stands for both.

Splits. The ID stream (OpenOOD's 45,000 ImageNet test images) is split by index parity: half A
feeds the pseudo-labelled class statistics, the image mean and the maps; half B is the ID side of
every test score. Tuning uses only OpenOOD's validation split.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .concepts import build_concepts
from .scores import pair_metrics
from .selfcheck import make_folds, retrieval_scores
from .transport import Transport, candidate_name, fit_candidate, l2n, oracle_stats, pseudo_label_stats
from .wordnet import describe_groups, subtree_groups, text_cluster_groups

GEOMS = ("raw", "cent")


@dataclass
class G1bConfig:
    backbone: str = "ViT-B/16"
    id_stream_set: str = "imagenet_test"          # split by parity: half A fits, half B is the ID test side
    val_set: str = "imagenet_val"                 # OpenOOD's labelled tuning split (ID)
    val_ood_set: str = "openimage_o_val"          # OpenOOD's tuning split (OOD)
    p_min: float = 0.5
    n_min: int = 3
    n_cap: int = 20
    n_folds: int = 5
    group_max_size: Optional[int] = None
    r2_lams: tuple = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
    b1_lams: tuple = (0.1, 1.0)
    b2_hs: tuple = (0.1, 0.2)
    b2_lams: tuple = (0.1, 1.0, 3.0)
    taus: tuple = (10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0)
    omegas: tuple = (0.25, 0.5, 0.75, 1.0)
    bprime_negatives: int = 3000                  # NegLabel negatives mixed into the pseudo-OOD test
    top_maps: int = 3                             # maps (by B' pseudo-OOD AUROC) carried into D'
    near_sets: tuple = ("ssb_hard", "ninco")
    far_sets: tuple = ("inaturalist", "textures", "openimage_o")
    four_sets: tuple = ("inaturalist", "sun", "places", "textures_all")
    oracle_sets: tuple = ("ssb_hard", "ninco")
    concept_min_images: int = 6                   # at least 3 per half
    prompt: str = "The nice {}."
    # pre-registered decision thresholds (points)
    gain: float = 1.0                             # FPR95 gain that counts
    harm: float = 0.5                             # FPR95 loss tolerated on the other group
    hamper: float = 1.0                           # ImageNet top-1 a map may lose
    bprime_margin: float = 0.5                    # pseudo-OOD AUROC a map may trail raw text by
    planb_margin: float = 3.0                     # near FPR95 a perfect transport must add for Plan B
    cold_ns: tuple = (256, 1024, 4096, 16384)
    seed: int = 0
    device: str = "cpu"
    chunk: int = 4096
    max_images_per_set: Optional[int] = None      # smoke runs only


def cfg_name(cand: str, geom: str, protos: str = "T") -> str:
    return f"{cand}|{geom}|{protos}"


@torch.no_grad()
def lse_taus(X: torch.Tensor, P_pos: torch.Tensor, P_neg: torch.Tensor, taus: Sequence[float], device: str = "cpu",
             chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """LSE over positive and over negative prototypes, for every temperature: two (n, len(taus)) arrays.

    X, P_pos, P_neg must already be unit vectors in the same geometry.
    """
    Pp = P_pos.to(device, torch.float32)
    Pn = P_neg.to(device, torch.float32)
    A, B = [], []
    for s in range(0, X.shape[0], chunk):
        v = X[s : s + chunk].to(device, torch.float32)
        sp, sn = v @ Pp.t(), v @ Pn.t()
        A.append(torch.stack([torch.logsumexp(float(t) * sp, 1) for t in taus], 1).double().cpu())
        B.append(torch.stack([torch.logsumexp(float(t) * sn, 1) for t in taus], 1).double().cpu())
    if not A:
        z = np.zeros((0, len(taus)))
        return z, z
    return torch.cat(A).numpy(), torch.cat(B).numpy()


def _best(rows: List[Dict], key_fpr: str = "fpr95", key_auroc: str = "auroc") -> Dict:
    """Lowest FPR95, ties broken by the higher AUROC."""
    return min(rows, key=lambda r: (round(r[key_fpr], 6), -r[key_auroc]))


class G1b:
    def __init__(self, cache, bank, cfg: Optional[G1bConfig] = None, logger=None):
        self.cache, self.bank = cache, bank
        self.cfg = cfg or G1bConfig()
        self.log = logger.info if logger else print
        self.scale = float(getattr(bank, "logit_scale", 100.0))
        self.id_text = l2n(bank.id_text.float().cpu())
        self.neg_text = l2n(bank.neg_text.float().cpu())
        self.C, self.D = self.id_text.shape
        self.mu_text = self.id_text.double().mean(0)
        self._sets: Dict = {}
        self.results: Dict = {}
        self.tables: Dict = {}
        self.fits: Dict[str, Transport] = {}
        self.concepts: Dict = {}

    # ------------------------------------------------------------------ data
    def load(self, name: str):
        if name not in self._sets:
            fs = self.cache.load_set(name)
            n = self.cfg.max_images_per_set
            if n and len(fs) > n:
                idx = np.random.default_rng(self.cfg.seed).choice(len(fs), size=n, replace=False)
                fs = fs.subset(np.sort(idx))
            self._sets[name] = fs
        return self._sets[name]

    def has(self, name: str) -> bool:
        return self.cache.has(name)

    # ------------------------------------------------------------------ geometry
    def images(self, X: torch.Tensor, geom: str) -> torch.Tensor:
        V = l2n(torch.as_tensor(X).double())
        return (V if geom == "raw" else l2n(V - self.m_img)).float()

    def protos(self, tr: Transport, t: torch.Tensor, geom: str) -> torch.Tensor:
        U = tr.raw(t)
        if geom == "raw":
            return l2n(U).float()
        mu = self.mu_text if tr.kind == "R0" else tr.m_V
        return l2n(U - mu).float()

    def centroid_protos(self, tr: Transport, dirs: torch.Tensor, geom: str) -> torch.Tensor:
        """Image-centroid directions as prototypes, in the same geometry as a map's prototypes."""
        U = torch.as_tensor(dirs).double()
        if geom == "raw":
            return l2n(U).float()
        if tr.m_V is None:
            raise ValueError("centred centroid prototypes need a map with m_V (R1 or richer)")
        return l2n(U - tr.m_V).float()

    def candidates(self) -> List[Dict]:
        c = self.cfg
        out = [{"kind": "R0"}, {"kind": "R1"}]
        out += [{"kind": "R2", "lam": float(l)} for l in c.r2_lams]
        out += [{"kind": "B1", "lam": float(l)} for l in c.b1_lams]
        out += [{"kind": "B2", "lam": float(l), "h": float(h)} for h in c.b2_hs for l in c.b2_lams]
        return out

    @staticmethod
    def geoms_for(kind: str) -> Tuple[str, ...]:
        return ("raw",) if kind == "R1" else GEOMS          # R1 centred == R0 centred

    # ------------------------------------------------------------------ preparation
    def prepare(self) -> Dict:
        c = self.cfg
        t0 = time.time()
        fs = self.load(c.id_stream_set)
        idx = np.arange(len(fs))
        a_idx, b_idx = idx[idx % 2 == 0], idx[idx % 2 == 1]
        self.stream_A = fs.feats[torch.as_tensor(a_idx)]
        self.labels_A = np.asarray(fs.labels)[a_idx]
        self.test_B = fs.feats[torch.as_tensor(b_idx)]
        stats, info = pseudo_label_stats(self.stream_A, self.id_text, self.scale, c.p_min, true_labels=self.labels_A,
                                         device=c.device)
        self.stats = stats
        self.m_img = l2n(self.stream_A.double()).mean(0)            # mean image direction, no labels
        v = self.load(c.val_set)
        self.val_X, self.val_y = v.feats, np.asarray(v.labels)
        self.val_stats = oracle_stats(v.feats, self.val_y, self.C)
        self.val_ood = self.load(c.val_ood_set).feats if self.has(c.val_ood_set) else None
        self.fits = {candidate_name(k): fit_candidate(k, self.id_text, stats, n_min=c.n_min, n_cap=c.n_cap)
                     for k in self.candidates()}
        info.update({"half_A": int(len(a_idx)), "half_B": int(len(b_idx)),
                     "n_classes_eligible": int(len(stats.eligible(c.n_min))),
                     "image_mean_norm": float(self.m_img.norm()), "val_images": int(len(self.val_y)),
                     "val_ood_images": int(0 if self.val_ood is None else len(self.val_ood)),
                     "seconds": round(time.time() - t0, 1)})
        self.results["prep"] = info
        self.log(f"[G1b] half A {info['half_A']} images ({info['n_kept']} confident, {info['n_classes_eligible']} classes), "
                 f"half B {info['half_B']} for testing; pseudo-label accuracy "
                 f"{100 * info.get('pseudo_label_acc_kept', float('nan')):.1f}%")
        return info

    def groups(self, wn=None):
        wnids = None
        if self.C == 1000:
            try:
                from oodlab.data.imagenet import official_wnids

                wnids = official_wnids()
            except Exception:
                wnids = None
        if wnids is None:
            wnids = getattr(self.bank, "id_wnids", None)
        if wn is not None and wnids is not None and len(wnids) == self.C:
            g, names = subtree_groups(wnids, wn, max_size=self.cfg.group_max_size, min_groups=2 * self.cfg.n_folds)
            return g, names, "wordnet"
        g, names = text_cluster_groups(self.id_text, n_groups=25, seed=self.cfg.seed)
        return g, names, "text-clusters"

    # ------------------------------------------------------------------ B'
    @torch.no_grad()
    def part_b(self, wn=None) -> Dict:
        c = self.cfg
        t0 = time.time()
        dev = c.device
        groups, gnames, source = self.groups(wn)
        elig_fit = self.stats.eligible(c.n_min).numpy()
        elig = np.intersect1d(elig_fit, self.val_stats.eligible(2).numpy())
        folds = make_folds(groups, elig, n_folds=c.n_folds, seed=c.seed)
        rng = np.random.default_rng(c.seed)
        n_neg = min(c.bprime_negatives, self.neg_text.shape[0])
        neg_sub = self.neg_text[torch.as_tensor(np.sort(rng.choice(self.neg_text.shape[0], n_neg, replace=False)))]
        y = self.val_y
        X = {g: self.images(self.val_X, g).to(dev) for g in GEOMS}
        acc: Dict[Tuple[str, str], Dict] = {}
        taus = list(c.taus)
        for fi, H in enumerate(folds):
            H = np.asarray(H)
            F_all = np.setdiff1d(np.arange(self.C), H)
            fit_classes = np.setdiff1d(elig_fit, H)
            mH = np.isin(y, H)
            mF = (~mH) & (y >= 0)
            h_pos = {int(k): i for i, k in enumerate(H)}
            yH = y[mH]
            own_col = torch.as_tensor([h_pos[int(k)] for k in yH], device=dev)
            yF = torch.as_tensor(y[mF], device=dev)
            H_t, F_t = torch.as_tensor(H, device=dev), torch.as_tensor(F_all, device=dev)
            for cand in self.candidates():
                tr = fit_candidate(cand, self.id_text, self.stats, classes=fit_classes, n_min=c.n_min, n_cap=c.n_cap)
                name = candidate_name(cand)
                for geom in self.geoms_for(cand["kind"]):
                    P = self.protos(tr, self.id_text, geom).to(dev)
                    PN = self.protos(tr, neg_sub, geom).to(dev)
                    Xg = X[geom]
                    S = Xg @ P.t()                                      # (n_val, C)
                    SH = S[torch.as_tensor(mH, device=dev)]
                    SF = S[torch.as_tensor(mF, device=dev)]
                    sHH = SH[:, H_t]
                    own = sHH.gather(1, own_col[:, None]).squeeze(1)
                    top1 = float((sHH.argmax(1) == own_col).double().mean())
                    catch = float((own > SH[:, F_t].max(1).values).double().mean())
                    ownF = SF.gather(1, yF[:, None]).squeeze(1)
                    steal = float((SF[:, H_t].max(1).values > ownF).double().mean())
                    # pseudo-OOD: ID = images of classes outside H, OOD = images of H; negatives = P[H] + PN
                    Sn = torch.cat([S[:, H_t], Xg @ PN.t()], 1)
                    Sp = S[:, F_t]
                    ells = torch.stack([torch.logsumexp(t * Sp, 1) - torch.logsumexp(t * Sn, 1) for t in taus], 1).cpu().numpy()
                    ell_id, ell_ood = ells[mF], ells[mH]
                    pood = [pair_metrics(ell_id[:, i], ell_ood[:, i]) for i in range(len(taus))]
                    # the old direction, for the record: mapped names -> held-out image centroids
                    cent = torch.zeros(len(H), self.D, device=dev, dtype=torch.float32)
                    cent.index_add_(0, own_col, Xg[torch.as_tensor(mH, device=dev)])
                    cent = l2n(cent)
                    r = retrieval_scores(P[H_t].double().cpu(), cent.double().cpu())
                    a = acc.setdefault((name, geom), {"kind": cand["kind"], "lam": cand.get("lam", np.nan),
                                                      "h": cand.get("h", np.nan), "nH": 0, "nF": 0, "nC": 0,
                                                      "top1": 0.0, "catch": 0.0, "steal": 0.0, "t2c": 0.0, "align": 0.0,
                                                      "auroc": np.zeros(len(taus)), "fpr": np.zeros(len(taus)), "folds": 0})
                    nH, nF, nC = int(mH.sum()), int(mF.sum()), r["n"]
                    a["top1"] += top1 * nH
                    a["catch"] += catch * nH
                    a["steal"] += steal * nF
                    a["t2c"] += r["top1"] * nC
                    a["align"] += r["align"] * nC
                    a["nH"] += nH
                    a["nF"] += nF
                    a["nC"] += nC
                    a["auroc"] += np.array([m["auroc"] for m in pood])
                    a["fpr"] += np.array([m["fpr95"] for m in pood])
                    a["folds"] += 1
            self.log(f"[G1b-B'] fold {fi + 1}/{len(folds)}: {len(H)} held-out classes")
        rows = []
        for (name, geom), a in acc.items():
            au, fp = a["auroc"] / a["folds"], a["fpr"] / a["folds"]
            i = int(np.argmax(au))
            rows.append({"config": cfg_name(name, geom), "candidate": name, "geom": geom, "kind": a["kind"],
                         "lam": a["lam"], "h": a["h"], "name_top1": 100 * a["top1"] / a["nH"],
                         "catch": 100 * a["catch"] / a["nH"], "steal": 100 * a["steal"] / a["nF"],
                         "pood_auroc": float(au[i]), "pood_fpr95": float(fp[i]), "pood_tau": float(taus[i]),
                         "t2c_top1": 100 * a["t2c"] / a["nC"], "align": a["align"] / a["nC"]})
        import pandas as pd

        df = pd.DataFrame(rows).sort_values("pood_auroc", ascending=False).reset_index(drop=True)
        self.tables["bprime"] = df
        r0 = df[df.config == cfg_name("R0", "raw")].iloc[0]
        best_map = df[~df.kind.isin(["R0", "R1"])].iloc[0]
        info = {"group_source": source, "n_groups": int(len(set(groups.tolist()))), "n_classes": int(len(elig)),
                "negatives": n_neg, "best_config": str(df.iloc[0].config), "best_map": str(best_map.config),
                "r0_raw": {k: float(r0[k]) for k in ("name_top1", "catch", "steal", "pood_auroc", "pood_fpr95", "t2c_top1")},
                "best_map_row": {k: float(best_map[k]) for k in ("name_top1", "catch", "steal", "pood_auroc",
                                                                   "pood_fpr95", "t2c_top1")},
                "seconds": round(time.time() - t0, 1)}
        self.results["B"] = info
        self.log(f"[G1b-B'] groups from {source}: {describe_groups(groups, gnames)}")
        self.log(f"[G1b-B'] best by pseudo-OOD AUROC: {info['best_config']}; best map {info['best_map']} "
                 f"({best_map.pood_auroc:.2f} vs raw text {r0.pood_auroc:.2f})")
        return info

    # ------------------------------------------------------------------ H
    @torch.no_grad()
    def part_h(self) -> Dict:
        c = self.cfg
        rows = []
        for cand in self.candidates():
            name = candidate_name(cand)
            tr = self.fits[name]
            for geom in self.geoms_for(cand["kind"]):
                P = self.protos(tr, self.id_text, geom).to(c.device)
                X = self.images(self.val_X, geom).to(c.device)
                pred = (X @ P.t()).argmax(1).cpu().numpy()
                rows.append({"config": cfg_name(name, geom), "candidate": name, "geom": geom, "kind": cand["kind"],
                             "val_top1": 100 * float((pred == self.val_y).mean())})
        import pandas as pd

        df = pd.DataFrame(rows).sort_values("val_top1", ascending=False).reset_index(drop=True)
        self.tables["hamper"] = df
        r0 = float(df[df.config == cfg_name("R0", "raw")].val_top1.iloc[0])
        info = {"r0_raw_top1": r0, "best": str(df.iloc[0].config), "best_top1": float(df.iloc[0].val_top1)}
        self.results["H"] = info
        self.log(f"[G1b-H] ImageNet val top-1: raw text {r0:.2f}%, best {info['best']} {info['best_top1']:.2f}%")
        return info

    # ------------------------------------------------------------------ D'
    def _cfg_protos(self, cfg: Tuple[str, str, str]):
        cand, geom, pk = cfg
        tr = self.fits[cand]
        P_id = self.protos(tr, self.id_text, geom)
        if pk == "C":
            elig = self.stats.eligible(self.cfg.n_min)
            P_id = P_id.clone()
            P_id[elig] = self.centroid_protos(tr, self.stats.directions()[elig], geom)
        return tr, P_id, self.protos(tr, self.neg_text, geom), geom

    def _ell(self, X, cfg, taus, extra_neg: Optional[torch.Tensor] = None) -> np.ndarray:
        tr, P_id, P_neg, geom = self._cfg_protos(cfg)
        if extra_neg is not None:
            P_neg = torch.cat([P_neg, extra_neg])
        A, B = lse_taus(self.images(X, geom), P_id, P_neg, taus, self.cfg.device, self.cfg.chunk)
        return A - B

    def test_sets(self) -> List[str]:
        c = self.cfg
        names = list(dict.fromkeys(list(c.near_sets) + list(c.far_sets) + list(c.four_sets)))
        return [n for n in names if self.has(n)]

    def _group_metrics(self, ell_id: np.ndarray, ell_ood: Dict[str, np.ndarray]) -> Dict:
        c = self.cfg
        out = {}
        per = {ds: pair_metrics(ell_id, e) for ds, e in ell_ood.items()}
        for grp, names in (("near", c.near_sets), ("far", c.far_sets), ("four", c.four_sets)):
            ms = [per[n] for n in names if n in per]
            if len(ms) == len(names):
                out[f"{grp}_fpr95"] = float(np.mean([m["fpr95"] for m in ms]))
                out[f"{grp}_auroc"] = float(np.mean([m["auroc"] for m in ms]))
        for ds, m in per.items():
            out[f"{ds}_fpr95"] = m["fpr95"]
            out[f"{ds}_auroc"] = m["auroc"]
        return out

    def part_d(self) -> Dict:
        c = self.cfg
        t0 = time.time()
        if self.val_ood is None:
            raise RuntimeError(f"{c.val_ood_set} is not in the cache: D' needs OpenOOD's validation split")
        taus = list(c.taus)
        b = self.tables["bprime"]
        z_cfgs = [("R0", "raw", "T"), ("R0", "cent", "T")]
        maps = b[~b.kind.isin(["R0", "R1"])].head(c.top_maps)
        t_cfgs = [("R1", "raw", "T")]
        for _, m in maps.iterrows():
            t_cfgs += [(m.candidate, m.geom, "T"), (m.candidate, m.geom, "C")]
        t_cfgs = list(dict.fromkeys(t_cfgs))
        # 1) validation sweep
        val = {}
        for cfg in z_cfgs + t_cfgs:
            val[cfg] = (self._ell(self.val_X, cfg, taus), self._ell(self.val_ood, cfg, taus))
        ztune = {}
        for cfg in z_cfgs:
            li, lo = val[cfg]
            rows = [{"tau": taus[i], **pair_metrics(li[:, i], lo[:, i])} for i in range(len(taus))]
            ztune[cfg] = _best(rows)
        z_star = min(z_cfgs, key=lambda k: (round(ztune[k]["fpr95"], 6), -ztune[k]["auroc"]))
        zi = taus.index(ztune[z_star]["tau"])
        zl_id, zl_ood = val[z_star][0][:, zi], val[z_star][1][:, zi]
        ttune, talone = {}, {}
        for cfg in t_cfgs:
            li, lo = val[cfg]
            rows, alone = [], []
            for i, t in enumerate(taus):
                alone.append({"tau": t, **pair_metrics(li[:, i], lo[:, i])})
                for om in c.omegas:
                    rows.append({"tau": t, "omega": om, **pair_metrics((1 - om) * zl_id + om * li[:, i],
                                                                     (1 - om) * zl_ood + om * lo[:, i])})
            ttune[cfg], talone[cfg] = _best(rows), _best(alone)
        # 2) test
        sets = self.test_sets()
        ood_feats = {ds: self.load(ds).feats for ds in sets}
        rows = []

        def evaluate(cfg, tau_list):
            ell_b = self._ell(self.test_B, cfg, tau_list)
            ell_o = {ds: self._ell(ood_feats[ds], cfg, tau_list) for ds in sets}
            return ell_b, ell_o

        zt = {}
        for cfg in z_cfgs:
            is_r0 = cfg == ("R0", "raw", "T")
            tl = list(dict.fromkeys([ztune[cfg]["tau"]] + ([self.scale] if is_r0 else [])))
            eb, eo = evaluate(cfg, tl)
            zt[cfg] = (eb[:, 0], {ds: e[:, 0] for ds, e in eo.items()})
            rows.append({"config": cfg_name(*cfg), "role": "text (Z)", "mode": "alone", "tau": tl[0], "omega": 0.0,
                         "val_fpr95": ztune[cfg]["fpr95"], "val_auroc": ztune[cfg]["auroc"],
                         **self._group_metrics(zt[cfg][0], zt[cfg][1])})
            if is_r0:                                       # NegLabel's form at CLIP's own temperature, untuned
                j = tl.index(self.scale)
                li, lo = self._ell(self.val_X, cfg, [self.scale]), self._ell(self.val_ood, cfg, [self.scale])
                vm = pair_metrics(li[:, 0], lo[:, 0])
                rows.append({"config": cfg_name(*cfg), "role": "NegLabel form", "mode": "alone", "tau": self.scale,
                             "omega": 0.0, "val_fpr95": vm["fpr95"], "val_auroc": vm["auroc"],
                             **self._group_metrics(eb[:, j], {ds: e[:, j] for ds, e in eo.items()})})
        zb, zo = zt[z_star]
        for cfg in t_cfgs:
            tf, ta = ttune[cfg], talone[cfg]
            tl = list(dict.fromkeys([tf["tau"], ta["tau"]]))
            eb, eo = evaluate(cfg, tl)
            i_f, i_a = tl.index(tf["tau"]), tl.index(ta["tau"])
            om = tf["omega"]
            rows.append({"config": cfg_name(*cfg), "role": "map", "mode": f"+{cfg_name(*z_star)}",
                         "tau": tf["tau"], "omega": om, "val_fpr95": tf["fpr95"], "val_auroc": tf["auroc"],
                         **self._group_metrics((1 - om) * zb + om * eb[:, i_f],
                                               {ds: (1 - om) * zo[ds] + om * eo[ds][:, i_f] for ds in sets})})
            rows.append({"config": cfg_name(*cfg), "role": "map", "mode": "alone", "tau": ta["tau"], "omega": 1.0,
                         "val_fpr95": ta["fpr95"], "val_auroc": ta["auroc"],
                         **self._group_metrics(eb[:, i_a], {ds: e[:, i_a] for ds, e in eo.items()})})
        import pandas as pd

        df = pd.DataFrame(rows)
        self.tables["dprime"] = df
        fused = df[(df.role == "map") & (df["mode"] != "alone")]
        t_row = fused.sort_values(["val_fpr95", "val_auroc"], ascending=[True, False]).iloc[0]
        alone = df[(df.role == "map") & (df["mode"] == "alone")]
        a_row = alone.sort_values(["val_fpr95", "val_auroc"], ascending=[True, False]).iloc[0]
        z_row = df[(df.config == cfg_name(*z_star)) & (df.role == "text (Z)")].iloc[0]
        r0_row = df[(df.config == cfg_name("R0", "raw")) & (df.role == "text (Z)")].iloc[0]
        rc_row = df[(df.config == cfg_name("R0", "cent")) & (df.role == "text (Z)")].iloc[0]
        keep = lambda r: {k: (float(r[k]) if k in r and r[k] == r[k] else None) for k in
                          ("tau", "omega", "val_fpr95", "near_fpr95", "far_fpr95", "four_fpr95", "near_auroc", "far_auroc")}
        self.z_star = z_star
        self.t_star = tuple(t_row.config.split("|"))
        self.t_alone = tuple(a_row.config.split("|"))
        self.tuned = {"z": ztune, "fused": ttune, "alone": talone}
        info = {"Z_star": cfg_name(*z_star), "Z": keep(z_row), "R0_raw": keep(r0_row), "R0_cent": keep(rc_row),
                "T_star": t_row.config, "T": keep(t_row), "T_alone": a_row.config, "T_alone_row": keep(a_row),
                "seconds": round(time.time() - t0, 1)}
        self.results["D"] = info
        self.log(f"[G1b-D'] text {info['Z_star']} (tau {z_row.tau:g}): near {z_row.near_fpr95:.2f} / far {z_row.far_fpr95:.2f}; "
                 f"best map {t_row.config} (tau {t_row.tau:g}, omega {t_row.omega:g}): near {t_row.near_fpr95:.2f} / "
                 f"far {t_row.far_fpr95:.2f} FPR95")
        return info

    # ------------------------------------------------------------------ O
    def build_oracle_concepts(self, text_encoder: Callable[[List[str]], torch.Tensor], wn=None) -> Dict:
        c = self.cfg
        out = {}
        for ds in c.oracle_sets:
            if not self.has(ds):
                continue
            fs = self.load(ds)
            cs = build_concepts(ds, fs.feats, fs.keys, wn=wn, min_images=c.concept_min_images)
            if len(cs) < 2:
                continue
            ic = cs.image_concept
            half = np.full(len(ic), -1)
            for k in range(len(cs)):
                idx = np.nonzero(ic == k)[0]
                half[idx[0::2]] = 0                         # A: builds the real centroid
                half[idx[1::2]] = 1                         # B: evaluated
            X = l2n(fs.feats.double())
            sums = torch.zeros(len(cs), self.D, dtype=torch.float64)
            mA = torch.as_tensor((half == 0) & (ic >= 0))
            sums.index_add_(0, torch.as_tensor(ic[mA.numpy()]), X[mA])
            out[ds] = {"names": list(cs.names), "text": l2n(text_encoder(list(cs.names)).double()).float(),
                       "cent_A": l2n(sums).float(), "eval_idx": np.nonzero((half == 1) & (ic >= 0))[0],
                       "eval_concept": ic[(half == 1) & (ic >= 0)], "n_concepts": len(cs)}
        self.concepts = out
        return {ds: v["n_concepts"] for ds, v in out.items()}

    @torch.no_grad()
    def part_o(self, text_encoder: Callable[[List[str]], torch.Tensor], wn=None) -> Dict:
        c = self.cfg
        t0 = time.time()
        if not self.concepts:
            self.build_oracle_concepts(text_encoder, wn)
        if not self.concepts:
            self.results["O"] = {"note": "no concept sets available"}
            return self.results["O"]
        ds_list = list(self.concepts)
        offsets, k0 = {}, 0
        for ds in ds_list:
            offsets[ds] = k0
            k0 += self.concepts[ds]["n_concepts"]
        c_text = torch.cat([self.concepts[ds]["text"] for ds in ds_list])
        c_cent = torch.cat([self.concepts[ds]["cent_A"] for ds in ds_list])
        z, ta = self.z_star, self.t_alone
        z_tau = self.tuned["z"][z]["tau"]
        a_tau = self.tuned["alone"][ta]["tau"]
        tr_z, tr_a = self.fits[z[0]], self.fits[ta[0]]
        variants = [
            ("text", z, z_tau, None),
            ("text + names", z, z_tau, self.protos(tr_z, c_text, z[1])),
            ("map", ta, a_tau, None),
            ("map + names", ta, a_tau, self.protos(tr_a, c_text, ta[1])),
            ("map + true centroids", ta, a_tau, self.centroid_protos(tr_a, c_cent, ta[1])),
        ]
        far = [d for d in c.far_sets if self.has(d)]
        rows = []
        for label, cfg, tau, extra in variants:
            _, P_id, P_neg, geom = self._cfg_protos(cfg)
            n_base = P_neg.shape[0]
            Pn = P_neg if extra is None else torch.cat([P_neg, extra])
            ell_b = self._ell_with(self.test_B, P_id, Pn, geom, tau)
            row = {"variant": label, "config": cfg_name(*cfg), "tau": tau}
            fprs, aucs = [], []
            for ds in ds_list:
                cc = self.concepts[ds]
                Xe = self.load(ds).feats[torch.as_tensor(cc["eval_idx"])]
                m = pair_metrics(ell_b, self._ell_with(Xe, P_id, Pn, geom, tau))
                row[f"{ds}_fpr95"], row[f"{ds}_auroc"] = m["fpr95"], m["auroc"]
                fprs.append(m["fpr95"])
                aucs.append(m["auroc"])
                if extra is not None:                       # catch: the concept's own prototype is the nearest
                    Xg = self.images(Xe, geom)
                    own = torch.as_tensor(P_id.shape[0] + n_base + offsets[ds] + cc["eval_concept"])
                    Pa = torch.cat([P_id, Pn]).to(c.device)
                    hits = []
                    for s in range(0, Xg.shape[0], c.chunk):
                        am = (Xg[s : s + c.chunk].to(c.device) @ Pa.t()).argmax(1).cpu()
                        hits.append(am == own[s : s + c.chunk])
                    row[f"{ds}_catch"] = 100 * float(torch.cat(hits).double().mean())
            row["near_fpr95"], row["near_auroc"] = float(np.mean(fprs)), float(np.mean(aucs))
            if extra is not None:                           # steal: an ID test image's nearest prototype is a concept
                Xg = self.images(self.test_B, geom)
                n_id = P_id.shape[0] + n_base
                Pa = torch.cat([P_id, Pn]).to(c.device)
                hit = []
                for s in range(0, Xg.shape[0], c.chunk):
                    hit.append(((Xg[s : s + c.chunk].to(c.device) @ Pa.t()).argmax(1) >= n_id).cpu())
                row["id_stolen"] = 100 * float(torch.cat(hit).double().mean())
            if far:
                fm = [pair_metrics(ell_b, self._ell_with(self.load(d).feats, P_id, Pn, geom, tau))["fpr95"] for d in far]
                row["far_fpr95"] = float(np.mean(fm))
            rows.append(row)
        import pandas as pd

        df = pd.DataFrame(rows)
        self.tables["oracle"] = df
        g = df.set_index("variant")
        near = g["near_fpr95"]
        info = {"concepts": {ds: self.concepts[ds]["n_concepts"] for ds in ds_list},
                "text_config": cfg_name(*z), "map_config": cfg_name(*ta),
                "near_fpr95": {k: float(v) for k, v in near.items()},
                "gain_text_names": float(near["text"] - near["text + names"]),
                "gain_map_names": float(near["map"] - near["map + names"]),
                "gain_true_centroids": float(near["map"] - near["map + true centroids"]),
                "transport_headroom": float(near["map + names"] - near["map + true centroids"]),
                "visual_vs_text": float(near["text + names"] - near["map + true centroids"]),
                "seconds": round(time.time() - t0, 1)}
        self.results["O"] = info
        self.log(f"[G1b-O] near FPR95 with oracle negatives: text {near['text']:.2f} -> {near['text + names']:.2f}; "
                 f"map {near['map']:.2f} -> names {near['map + names']:.2f} / true centroids {near['map + true centroids']:.2f}")
        return info

    def _ell_with(self, X, P_id, P_neg, geom, tau) -> np.ndarray:
        A, B = lse_taus(self.images(X, geom), P_id, P_neg, [tau], self.cfg.device, self.cfg.chunk)
        return (A - B)[:, 0]

    # ------------------------------------------------------------------ E'
    def part_e(self) -> Dict:
        c = self.cfg
        dec = self.results.get("decision") or self.decide()
        if dec["verdict"] != "KEEP-TRANSPORT":
            info = {"ran": False, "note": f"skipped: verdict {dec['verdict']} (E' runs only when a map is kept)"}
            self.results["E"] = info
            return info
        cand, geom, pk = self.t_star
        tf = self.tuned["fused"][self.t_star]
        z = self.z_star
        z_tau = self.tuned["z"][z]["tau"]
        zl_id = self._ell(self.val_X, z, [z_tau])[:, 0]
        zl_ood = self._ell(self.val_ood, z, [z_tau])[:, 0]
        base = pair_metrics(zl_id, zl_ood)
        perm = np.random.default_rng(c.seed).permutation(len(self.stream_A))
        rows = [{"n_seen": 0, "n_classes": 0, "val_fpr95_text": base["fpr95"], "val_fpr95_fused": base["fpr95"]}]
        from .selfcheck import candidate_from_name

        spec = candidate_from_name(cand, self.candidates())
        full = self.fits
        for n in c.cold_ns:
            n = min(n, len(perm))
            st, _ = pseudo_label_stats(self.stream_A[torch.as_tensor(perm[:n])], self.id_text, self.scale, c.p_min,
                                       device=c.device)
            tr = fit_candidate(spec, self.id_text, st, n_min=c.n_min, n_cap=c.n_cap)
            self.fits = dict(full, **{cand: tr})
            saved = self.stats
            self.stats = st
            li = self._ell(self.val_X, self.t_star, [tf["tau"]])[:, 0]
            lo = self._ell(self.val_ood, self.t_star, [tf["tau"]])[:, 0]
            self.stats = saved
            om = tf["omega"]
            m = pair_metrics((1 - om) * zl_id + om * li, (1 - om) * zl_ood + om * lo)
            rows.append({"n_seen": n, "n_classes": int(len(st.eligible(c.n_min))), "val_fpr95_text": base["fpr95"],
                         "val_fpr95_fused": m["fpr95"]})
            if n >= len(perm):
                break
        self.fits = full
        import pandas as pd

        df = pd.DataFrame(rows)
        self.tables["coldstart"] = df
        helped = df[(df.n_seen > 0) & (df.val_fpr95_fused < df.val_fpr95_text)]
        info = {"ran": True, "first_n_that_helps": int(helped.n_seen.min()) if len(helped) else None}
        self.results["E"] = info
        return info

    # ------------------------------------------------------------------ decision
    def decide(self) -> Dict:
        c = self.cfg
        r = self.results
        d, b, h = r.get("D", {}), self.tables.get("bprime"), self.tables.get("hamper")
        z, t = d.get("Z", {}), d.get("T", {})
        near_gain = (z.get("near_fpr95") or 0) - (t.get("near_fpr95") or 0)
        far_gain = (z.get("far_fpr95") or 0) - (t.get("far_fpr95") or 0)
        score_ok = (near_gain >= c.gain and far_gain >= -c.harm) or (far_gain >= c.gain and near_gain >= -c.harm)
        t_cand, t_geom = (self.t_star[0], self.t_star[1]) if hasattr(self, "t_star") else ("R0", "raw")
        hamper_ok = bprime_ok = False
        if h is not None and len(h):
            r0_top1 = float(h[h.config == cfg_name("R0", "raw")].val_top1.iloc[0])
            ht = h[h.config == cfg_name(t_cand, t_geom)]
            hamper_ok = bool(len(ht)) and float(ht.val_top1.iloc[0]) >= r0_top1 - c.hamper
        if b is not None and len(b):
            text_best = float(b[b.kind == "R0"].pood_auroc.max())
            bt = b[b.config == cfg_name(t_cand, t_geom)]
            bprime_ok = bool(len(bt)) and float(bt.pood_auroc.iloc[0]) >= text_best - c.bprime_margin
        r0, rc = d.get("R0_raw", {}), d.get("R0_cent", {})
        c_near = (r0.get("near_fpr95") or 0) - (rc.get("near_fpr95") or 0)
        c_far = (r0.get("far_fpr95") or 0) - (rc.get("far_fpr95") or 0)
        center_ok = (c_near >= c.gain and c_far >= -c.harm) or (c_far >= c.gain and c_near >= -c.harm)
        if score_ok and hamper_ok and bprime_ok:
            verdict = "KEEP-TRANSPORT"
            action = (f"Seam 1 = {d.get('T_star')} (tau {t.get('tau')}, omega {t.get('omega')}), fused with "
                      f"{d.get('Z_star')}. Continue to Seam 2 (G2) on top of it.")
        elif center_ok:
            verdict = "CENTER-ONLY"
            action = ("No map beats text, but centring the similarities does. Seam 1 becomes the centred geometry "
                      "(no fitted map) plus test-time evidence (R3); continue to Seam 2 (G2).")
        else:
            verdict = "NO-STATIC-GAIN"
            action = ("Neither a map nor centring improves the static score. Keep raw text for scoring; use mapped "
                      "prototypes only where B' shows they help (screening near negatives), and continue to Seam 2.")
        o = r.get("O", {})
        planb = None
        if "transport_headroom" in o:
            planb = bool(o["transport_headroom"] >= c.planb_margin and o["visual_vs_text"] >= c.planb_margin)
        dec = {"verdict": verdict, "action": action,
               "criteria": {"score_gain_vs_text": score_ok, "near_gain": round(near_gain, 2), "far_gain": round(far_gain, 2),
                            "does_not_hamper_imagenet": hamper_ok, "bprime_not_worse": bprime_ok,
                            "centring_gain": {"near": round(c_near, 2), "far": round(c_far, 2), "counts": center_ok}},
               "plan_b": _planb_text(planb, o, c.planb_margin),
               "seam2_outlook": ({k: round(o[k], 2) for k in ("gain_text_names", "gain_map_names", "gain_true_centroids")}
                                 if "gain_text_names" in o else None),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.results["decision"] = dec
        self.log(f"[G1b] verdict {verdict}: {action}")
        return dec

    # ------------------------------------------------------------------ reporting
    def summary_text(self) -> str:
        """A compact plain-text block with every number of the run."""
        from . import __version__

        r = self.results
        L = [f"G1b summary (reneg {__version__})"]
        p = r.get("prep", {})
        if p:
            L.append(f"prep: half A {p['half_A']} images, {p['n_kept']} confident, {p['n_classes_eligible']} classes; "
                     f"pseudo-label acc {100 * p.get('pseudo_label_acc_kept', float('nan')):.1f}%; half B {p['half_B']}")
        b = self.tables.get("bprime")
        if b is not None:
            keep = b[b.kind.isin(["R0", "R1"])]
            top = b[~b.kind.isin(["R0", "R1"])].head(6)
            t = __import__("pandas").concat([keep, top])
            L.append("B' held-out subtrees (image -> names), best tau per row:")
            L.append(t[["config", "name_top1", "catch", "steal", "pood_auroc", "pood_fpr95", "pood_tau", "t2c_top1",
                        "align"]].round(2).to_string(index=False))
        h = self.tables.get("hamper")
        if h is not None:
            L.append("H ImageNet val top-1: " + "; ".join(f"{row.config} {row.val_top1:.2f}" for row in h.head(8).itertuples())
                     + f"; R0|raw|T {r['H']['r0_raw_top1']:.2f}")
        d = self.tables.get("dprime")
        if d is not None:
            cols = ["config", "role", "mode", "tau", "omega", "val_fpr95", "near_fpr95", "far_fpr95", "four_fpr95",
                    "near_auroc", "far_auroc"]
            L.append("D' OpenOOD v1.5 (ID = ImageNet test half B), tau/omega tuned on val:")
            L.append(d[[x for x in cols if x in d.columns]].round(2).to_string(index=False))
        o = self.tables.get("oracle")
        if o is not None:
            cols = [x for x in o.columns if x.endswith("fpr95") or x.endswith("catch") or x == "id_stolen" or x == "variant"]
            L.append("O oracle near negatives (diagnostic only):")
            L.append(o[cols].round(2).to_string(index=False))
        e = self.tables.get("coldstart")
        if e is not None:
            L.append("E' cold start:")
            L.append(e.round(2).to_string(index=False))
        dec = r.get("decision")
        if dec:
            L.append(f"VERDICT: {dec['verdict']} - {dec['action']}")
            L.append(f"criteria: {json.dumps(dec['criteria'])}")
            L.append(f"PLAN B: {dec['plan_b']}")
            if dec.get("seam2_outlook"):
                L.append(f"SEAM 2 OUTLOOK (near FPR95 gain from oracle near negatives): {json.dumps(dec['seam2_outlook'])}")
        return "\n".join(L)

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        for name, df in self.tables.items():
            if df is not None and len(df):
                df.to_csv(os.path.join(out_dir, f"g1b_{name}.csv"), index=False)
        res = dict(self.results)
        res["config"] = asdict(self.cfg)
        with open(os.path.join(out_dir, "g1b_results.json"), "w") as f:
            json.dump(_jsonable(res), f, indent=1)
        path = os.path.join(out_dir, "g1b_summary.txt")
        with open(path, "w") as f:
            f.write(self.summary_text() + "\n")
        self.log(f"[G1b] saved tables, g1b_results.json and g1b_summary.txt to {out_dir}")
        return path

    # ------------------------------------------------------------------ analysis pack
    def export_pack(self, out_dir: str, sizes: Optional[Dict[str, int]] = None) -> List[str]:
        """Two compressed .npz files (about 20 MB each) with what the later CPU analyses need (experiments/).

        Part 1: text side, class statistics, the validation split. Part 2: random subsets of test images.
        All features are fp16 like the cache. Nothing here is needed to reproduce G1b itself.
        """
        c = self.cfg
        sizes = dict({"test_B": 7000, "half_A": 5000, "ssb_hard": 3000, "ninco": 2500, "textures": 1500,
                      "inaturalist": 1500, "openimage_o": 1500}, **(sizes or {}))
        rng = np.random.default_rng(c.seed)
        os.makedirs(out_dir, exist_ok=True)
        f16 = lambda x: np.asarray(torch.as_tensor(x).float().numpy(), dtype=np.float16)
        p1 = {"id_text": f16(self.id_text), "neg_text": f16(self.neg_text), "logit_scale": np.float32(self.scale),
              "id_names": np.asarray(list(self.bank.id_names), dtype=str),
              "stats_sums": self.stats.sums.float().numpy(), "stats_counts": self.stats.counts.float().numpy(),
              "m_img": self.m_img.float().numpy(), "val_feats": f16(self.val_X), "val_labels": self.val_y.astype(np.int16),
              "p_min": np.float32(c.p_min)}
        if self.val_ood is not None:
            p1["val_ood_feats"] = f16(self.val_ood)
        for ds, cc in self.concepts.items():
            p1[f"concept_names__{ds}"] = np.asarray(cc["names"], dtype=str)
            p1[f"concept_text__{ds}"] = f16(cc["text"])
            p1[f"concept_centA__{ds}"] = f16(cc["cent_A"])
        try:
            from oodlab.data.imagenet import official_wnids

            p1["id_wnids"] = np.asarray(official_wnids(), dtype=str)
        except Exception:
            pass
        p2 = {}

        def pick(n_total, n):
            return np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))

        i = pick(len(self.test_B), sizes["test_B"])
        p2["test_B_feats"] = f16(self.test_B[torch.as_tensor(i)])
        fsB = self.load(c.id_stream_set)
        p2["test_B_labels"] = np.asarray(fsB.labels)[1::2][i].astype(np.int16)
        i = pick(len(self.stream_A), sizes["half_A"])
        p2["half_A_feats"] = f16(self.stream_A[torch.as_tensor(i)])
        p2["half_A_labels"] = self.labels_A[i].astype(np.int16)
        for ds in ("ssb_hard", "ninco", "textures", "inaturalist", "openimage_o"):
            if not self.has(ds):
                continue
            fs = self.load(ds)
            if ds in self.concepts:
                cc = self.concepts[ds]
                j = pick(len(cc["eval_idx"]), sizes[ds])
                p2[f"{ds}_feats"] = f16(fs.feats[torch.as_tensor(cc["eval_idx"][j])])
                p2[f"{ds}_concept"] = cc["eval_concept"][j].astype(np.int16)
            else:
                j = pick(len(fs), sizes[ds])
                p2[f"{ds}_feats"] = f16(fs.feats[torch.as_tensor(j)])
                if ds == "textures":
                    from .concepts import folder_of

                    fold = [folder_of(fs.keys[k]) for k in j]
                    p2["textures_folder"] = np.asarray(fold, dtype=str)
        paths = []
        for k, d in (("1", p1), ("2", p2)):
            path = os.path.join(out_dir, f"reneg_pack_{k}.npz")
            np.savez_compressed(path, **d)
            paths.append(path)
            self.log(f"[G1b] analysis pack part {k}: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
        return paths


def _planb_text(planb: Optional[bool], o: Dict, margin: float) -> str:
    if planb is None:
        return "not tested (part O did not run)"
    head, vis = o["transport_headroom"], o["visual_vs_text"]
    if planb:
        return (f"justified: the classes' real image centroids beat their mapped names by {head:.1f} near FPR95 points "
                f"and their text names by {vis:.1f}, so a better map could pay")
    if head <= 0:
        return (f"not justified: the mapped names already do as well as the classes' real image centroids "
                f"({-head:.1f} near FPR95 points better), so a better map cannot add anything here")
    return (f"not justified: a map that reproduced the real centroids exactly would gain {head:.1f} near FPR95 points "
            f"over this map and {vis:.1f} over text names (threshold {margin:g} for both)")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x
