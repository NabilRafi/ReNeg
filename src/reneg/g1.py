"""Gate G1: does the text-to-image transport carry over from ID classes to real OOD concepts?

Run the parts in order (one notebook cell each)::

    g = G1(cache, bank, G1Config(device="cuda"), logger=log)
    g.part_a()                 # class statistics from confident ID test images (no labels used)
    g.part_b(wn)               # leave-subtree-out self-check -> chosen rung, lambda, kappa*
    g.part_c(text_encoder, wn) # OOD concept test (NINCO, Textures, SSB-hard names vs their images)
    g.part_d()                 # static image-space score vs text-space score
    g.part_e()                 # cold-start curve
    g.decide(); g.save(out_dir)

Only part A's *report* and parts C/D/E's *evaluation* touch labels; nothing that decides the
method (part B) uses a label or an OOD image.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .concepts import ConceptSet, build_concepts
from .scores import fuse, lse_parts, pair_metrics, score_from_parts
from .selfcheck import (candidate_from_name, choose_ladder, choose_online, clearly_better, kappa_star,
                        leave_group_out, retrieval_scores)
from .transport import (DEFAULT_HS, DEFAULT_LAMS, ClassStats, Transport, candidate_grid, candidate_name,
                        change_profile, fit_candidate, l2n, oracle_stats, pseudo_label_stats)
from .wordnet import describe_groups, subtree_groups, text_cluster_groups


@dataclass
class G1Config:
    backbone: str = "ViT-B/16"
    id_stream_set: str = "imagenet_test"          # confident ID images for fitting come from here
    val_set: str = "imagenet_val"                 # OpenOOD's labelled tuning split: judges the maps in part B
    p_min: float = 0.5                            # minimum zero-shot probability of a confident ID image
    n_min: int = 3                                # images before a class joins the fit
    n_cap: int = 20                               # cap on a class's weight
    lams: tuple = DEFAULT_LAMS
    hs: tuple = DEFAULT_HS
    n_folds: int = 5
    group_max_size: Optional[int] = None          # None -> min(120, max(8, classes // 8))
    eps: float = 0.01                             # a richer rung must win by 1 point of top-1 ...
    eps_align: float = 0.01                       # ... or tie on top-1 and gain 0.01 cosine alignment
    online_guard: float = 0.10                    # test-time rule: top-1 may trail R0 by at most this much
    concept_sets: tuple = ("ninco", "textures_all", "ssb_hard")
    concept_min_images: int = 5
    prompt: str = "The nice {}."
    score_scale: Optional[float] = None           # None -> the text bank's logit scale (about 100)
    omegas: tuple = (0.25, 0.5, 0.75, 1.0)        # 1.0 = image-space score alone
    protocols: Dict = field(default_factory=lambda: {
        "four_ood": ["imagenet_val_all", ["inaturalist", "sun", "places", "textures_all"]],
        "openood_v15": ["imagenet_test", ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]],
    })
    near_sets: tuple = ("ssb_hard", "ninco")
    far_sets: tuple = ("inaturalist", "textures", "openimage_o")
    cold_ns: tuple = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    cold_eval: tuple = ("imagenet_val", "openimage_o_val")   # the tuning split: never a test set
    cold_min_classes: int = 50                    # below this many classes the ladder stays on R1
    seed: int = 0
    device: str = "cpu"
    max_images_per_set: Optional[int] = None      # only for smoke runs


def clip_text_encoder(model, prompt: str = "The nice {}.", batch_size: int = 1000) -> Callable[[List[str]], torch.Tensor]:
    """Encode names exactly like oodlab's text bank encodes ID names (one template, normalised)."""
    from oodlab.clipwrap import label_text_features

    @torch.no_grad()
    def enc(names: List[str]) -> torch.Tensor:
        return l2n(label_text_features(model, list(names), [prompt], batch_size=batch_size).float().cpu())

    return enc


class G1:
    def __init__(self, cache, bank, cfg: Optional[G1Config] = None, logger=None):
        self.cache, self.bank = cache, bank
        self.cfg = cfg or G1Config()
        self.log = logger.info if logger else print
        self.scale = float(self.cfg.score_scale or getattr(bank, "logit_scale", 100.0))
        self.id_text = l2n(bank.id_text.float().cpu())
        self.neg_text = l2n(bank.neg_text.float().cpu())
        self.C = self.id_text.shape[0]
        self._sets: Dict = {}
        self.results: Dict = {}
        self.tables: Dict = {}
        self.fits: Dict[str, Transport] = {}
        self.stats: Optional[ClassStats] = None

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

    # ------------------------------------------------------------------ part A
    def part_a(self) -> Dict:
        c = self.cfg
        t0 = time.time()
        fs = self.load(c.id_stream_set)
        stats, info = pseudo_label_stats(fs.feats, self.id_text, self.scale, c.p_min, true_labels=fs.labels,
                                         device=c.device)
        self.stats = stats
        if self.has(c.val_set):
            v = self.load(c.val_set)
            self.val_stats = oracle_stats(v.feats, v.labels, self.C)
            info["val_set"] = c.val_set
            info["val_classes_with_2_images"] = int(len(self.val_stats.eligible(2)))
        else:
            self.val_stats = None
        elig = stats.eligible(c.n_min)
        info.update({"set": c.id_stream_set, "n_classes_eligible": int(len(elig)), "n_classes": self.C,
                     "median_images_per_class": float(np.median(stats.counts.numpy())),
                     "seconds": round(time.time() - t0, 1)})
        info["pass"] = bool(len(elig) >= min(900, int(0.9 * self.C)))
        self.results["A"] = info
        self.log(f"[G1-A] {info['n_kept']}/{info['n_images']} confident ID images (p >= {c.p_min}); "
                 f"{len(elig)}/{self.C} classes with >= {c.n_min} images; pseudo-label accuracy "
                 f"{100 * info.get('pseudo_label_acc_kept', float('nan')):.1f}%")
        return info

    # ------------------------------------------------------------------ part B
    def groups(self, wn=None):
        wnids = None
        if self.C == 1000:
            try:
                from oodlab.data.imagenet import official_wnids

                wnids = official_wnids()
            except Exception as e:  # pragma: no cover - depends on oodlab
                self.log(f"[G1-B] could not read ImageNet wnids from oodlab ({e})")
        if wnids is None:
            wnids = getattr(self.bank, "id_wnids", None)
        if wn is not None and wnids is not None and len(wnids) == self.C:
            g, names = subtree_groups(wnids, wn, max_size=self.cfg.group_max_size, min_groups=2 * self.cfg.n_folds)
            return g, names, "wordnet"
        g, names = text_cluster_groups(self.id_text, n_groups=25, seed=self.cfg.seed)
        return g, names, "text-clusters"

    def part_b(self, wn=None) -> Dict:
        assert self.stats is not None, "run part_a first"
        c = self.cfg
        t0 = time.time()
        groups, gnames, source = self.groups(wn)
        self.log(f"[G1-B] groups from {source}: {describe_groups(groups, gnames)}")
        cands = candidate_grid(c.lams, c.hs)
        # 1) the decision: maps fitted like the method (pseudo-labelled stream), judged on the labelled
        #    validation split's centroids of held-out subtrees (unbiased targets; the split is for tuning)
        judge = self.val_stats if self.val_stats is not None else None
        df = leave_group_out(self.id_text, self.stats, groups, cands, n_min=c.n_min, n_cap=c.n_cap,
                             n_folds=c.n_folds, seed=c.seed, eval_stats=judge)
        choice = choose_ladder(df, eps=c.eps, eps_align=c.eps_align)
        kap = kappa_star(self.stats, c.n_min, choice["chosen_row"]["b2"])
        self.tables["selfcheck"] = df
        # 2) what the method could decide alone at test time (pseudo-labelled targets, alignment first)
        dfo = leave_group_out(self.id_text, self.stats, groups, cands, n_min=c.n_min, n_cap=c.n_cap,
                              n_folds=c.n_folds, seed=c.seed)
        self.tables["selfcheck_online"] = dfo
        online = choose_online(dfo, guard=c.online_guard, eps_align=c.eps_align)
        # final maps: every candidate fitted on all eligible classes
        self.fits = {candidate_name(k): fit_candidate(k, self.id_text, self.stats, n_min=c.n_min, n_cap=c.n_cap)
                     for k in cands}
        chosen = self.fits[choice["chosen"]]
        prof = change_profile(chosen, self.id_text, self.stats, n_min=c.n_min, n_cap=c.n_cap)
        self.chosen_name = choice["chosen"]
        info = {"group_source": source, "n_groups": int(len(set(groups.tolist()))),
                "judged_on": c.val_set if judge is not None else "pseudo-labels (validation split missing)",
                **choice, **kap, "online_choice": online["chosen"],
                "online_agrees": online["chosen"] == choice["chosen"] or
                                 online["chosen"].split("(")[0] == choice["chosen"].split("(")[0],
                "change_profile": prof, "pass": choice["chosen"] != "R0", "seconds": round(time.time() - t0, 1)}
        info.pop("chosen_row", None)
        self.results["B"] = info
        r0 = df[df.candidate == "R0"].iloc[0]
        ch = df[df.candidate == choice["chosen"]].iloc[0]
        self.log(f"[G1-B] chosen {choice['chosen']} (path {choice['path']}): top-1 {100 * ch['top1']:.1f}% vs R0 "
                 f"{100 * r0['top1']:.1f}%, alignment {ch['align']:.3f} vs {r0['align']:.3f}; "
                 f"kappa* = {kap['kappa_star']:.2f}")
        return info

    # ------------------------------------------------------------------ part C
    def part_c(self, text_encoder: Callable[[List[str]], torch.Tensor], wn=None) -> Dict:
        assert self.fits, "run part_b first"
        c = self.cfg
        t0 = time.time()
        # the encoder must reproduce the bank's ID embeddings, or every comparison below is off
        probe = text_encoder(list(self.bank.id_names[:20]))
        enc_cos = float((probe * self.id_text[:20]).sum(1).min())
        id_dirs = self.stats.directions()[self.stats.eligible(c.n_min)].float()
        rows, notes, previews = [], {}, {}
        for ds in c.concept_sets:
            if not self.has(ds):
                notes[ds] = "not in the cache"
                continue
            fs = self.load(ds)
            cs = build_concepts(ds, fs.feats, fs.keys, wn=wn, min_images=c.concept_min_images)
            if len(cs) < 2:
                notes[ds] = cs.note or "fewer than 2 concepts"
                continue
            previews[ds] = cs.preview()
            if cs.note:
                notes[ds] = cs.note
            t = text_encoder(cs.names)
            for name, tr in self.fits.items():
                P = tr.apply(t)
                r = retrieval_scores(P, cs.centroids)
                own = (P * cs.centroids).sum(1)
                best_id = (P @ id_dirs.t()).max(1).values if len(id_dirs) else torch.full_like(own, -1.0)
                rows.append({"dataset": ds, "candidate": name, "kind": tr.kind, "top1": r["top1"], "mrr": r["mrr"],
                             "align": r["align"], "own_vs_id": float((own > best_id).float().mean()),
                             "n_concepts": r["n"]})
        import pandas as pd

        df = pd.DataFrame(rows)
        self.tables["concepts"] = df
        info = {"encoder_matches_bank_cos_min": enc_cos, "encoder_ok": enc_cos > 0.99, "notes": notes,
                "previews": previews, "seconds": round(time.time() - t0, 1)}
        if len(df):
            chosen, datasets = self.chosen_name, sorted(df.dataset.unique())
            ch = df[df.candidate == chosen].set_index("dataset")
            r0 = df[df.candidate == "R0"].set_index("dataset")
            wins = [d for d in datasets if clearly_better(ch.loc[d].to_dict(), r0.loc[d].to_dict(), c.eps)]
            no_drop = all(ch.loc[d, "align"] >= r0.loc[d, "align"] - 1e-9 for d in datasets)
            need = 2 if len(datasets) >= 3 else len(datasets)
            info.update({"datasets": datasets, "chosen": chosen, "wins": wins, "align_never_lower": no_drop,
                         "pass": bool(len(wins) >= need and no_drop)})
            mean_ood = df.groupby("candidate").top1.mean()
            sc = self.tables["selfcheck"].set_index("candidate").top1
            common = mean_ood.index.intersection(sc.index)
            info["ood_best_candidate"] = str(mean_ood.idxmax())
            info["selfcheck_vs_ood_spearman"] = _spearman(sc[common], mean_ood[common])
            al_ood = df.groupby("candidate").align.mean()
            al_sc = self.tables["selfcheck"].set_index("candidate")["align"]
            info["selfcheck_vs_ood_spearman_align"] = _spearman(al_sc[common], al_ood[common])
            info["chosen_ood_top1_gap_to_best"] = float(mean_ood.max() - mean_ood.get(chosen, np.nan))
            for d in datasets:
                self.log(f"[G1-C] {d}: {chosen} top-1 {100 * ch.loc[d, 'top1']:.1f}% vs R0 {100 * r0.loc[d, 'top1']:.1f}%, "
                         f"alignment {ch.loc[d, 'align']:.3f} vs {r0.loc[d, 'align']:.3f}")
        else:
            info["pass"] = False
        self.results["C"] = info
        return info

    # ------------------------------------------------------------------ part D
    def _id_protos(self, tr: Transport, kind: str) -> torch.Tensor:
        P = tr.apply(self.id_text)
        if kind == "C":                                   # centroids where we have them
            elig = self.stats.eligible(self.cfg.n_min)
            P = P.clone()
            P[elig] = self.stats.directions()[elig].float()
        return P

    def score_candidates(self) -> List[str]:
        b = self.results.get("B", {})
        names = ["R0", "R1", self.chosen_name] + list(b.get("best_per_kind", {}).values())
        out = []
        for n in names:
            if n in self.fits and n not in out:
                out.append(n)
        return out

    def part_d(self, candidates: Optional[List[str]] = None) -> Dict:
        assert self.fits, "run part_b first"
        c = self.cfg
        t0 = time.time()
        candidates = candidates or self.score_candidates()
        sets_needed = []
        for id_set, ood in c.protocols.values():
            sets_needed += [id_set] + list(ood)
        sets_needed = [s for s in dict.fromkeys(sets_needed) if self.has(s)]
        text_parts = {s: lse_parts(self.load(s).feats, self.id_text, self.neg_text, self.scale, device=c.device)
                      for s in sets_needed}
        s_text = {s: score_from_parts(*text_parts[s]) for s in sets_needed}
        rows = []
        for pname, (id_set, ood_sets) in c.protocols.items():
            if id_set not in s_text:
                continue
            for ds in ood_sets:
                if ds in s_text:
                    m = pair_metrics(s_text[id_set], s_text[ds])
                    rows.append({"protocol": pname, "dataset": ds, "scorer": "text (NegLabel form)", "candidate": "R0",
                                 "protos": "-", "omega": 0.0, **m})
        for name in candidates:
            if name == "R0":
                continue
            tr = self.fits[name]
            negs = tr.apply(self.neg_text)
            for protos in ("T", "C"):
                pos = self._id_protos(tr, protos)
                s_vis = {}
                for s in sets_needed:
                    a, b = lse_parts(self.load(s).feats, pos, negs, self.scale, device=c.device)
                    s_vis[s] = score_from_parts(a, b)
                for pname, (id_set, ood_sets) in c.protocols.items():
                    if id_set not in s_vis:
                        continue
                    for om in c.omegas:
                        sid = fuse(s_text[id_set], s_vis[id_set], om)
                        for ds in ood_sets:
                            if ds not in s_vis:
                                continue
                            m = pair_metrics(sid, fuse(s_text[ds], s_vis[ds], om))
                            rows.append({"protocol": pname, "dataset": ds, "scorer": "fused" if om < 1 else "image space",
                                         "candidate": name, "protos": protos, "omega": om, **m})
            self.log(f"[G1-D] scored {name}")
        import pandas as pd

        df = pd.DataFrame(rows)
        self.tables["scores"] = df
        summ = self._summarise_scores(df)
        self.tables["scores_summary"] = summ
        info = self._score_verdict(summ)
        info["seconds"] = round(time.time() - t0, 1)
        self.results["D"] = info
        return info

    def _summarise_scores(self, df):
        import pandas as pd

        c = self.cfg
        keys = ["protocol", "candidate", "protos", "omega", "scorer"]
        out = []
        for k, sub in df.groupby(keys):
            row = dict(zip(keys, k))
            if row["protocol"] == "openood_v15":
                for grp, names in (("near", c.near_sets), ("far", c.far_sets)):
                    s = sub[sub.dataset.isin(names)]
                    if len(s) == len(names):
                        out.append({**row, "group": grp, "fpr95": s.fpr95.mean(), "auroc": s.auroc.mean()})
            else:
                out.append({**row, "group": "all", "fpr95": sub.fpr95.mean(), "auroc": sub.auroc.mean()})
        return pd.DataFrame(out)

    def _score_verdict(self, summ) -> Dict:
        info = {"pass": False}
        o = summ[summ.protocol == "openood_v15"]
        if not len(o):
            info["note"] = "OpenOOD v1.5 sets missing"
            return info
        base = o[o.candidate == "R0"].set_index("group")
        if not {"near", "far"} <= set(base.index):
            info["note"] = "near or far sets missing"
            return info
        near0, far0 = float(base.loc["near", "fpr95"]), float(base.loc["far", "fpr95"])
        best = None
        for (cand, protos, om), sub in o[o.candidate != "R0"].groupby(["candidate", "protos", "omega"]):
            g = sub.set_index("group")
            if not {"near", "far"} <= set(g.index):
                continue
            near, far = float(g.loc["near", "fpr95"]), float(g.loc["far", "fpr95"])
            # 1 point on real benchmarks; a tenth of the baseline when the baseline is already tiny
            ok = near <= near0 - min(1.0, 0.1 * near0) and far <= far0 + max(1.0, 0.1 * far0)
            key = (ok, -(near + far))
            if best is None or key > best[0]:
                best = (key, {"candidate": cand, "protos": protos, "omega": float(om), "near_fpr95": near,
                              "far_fpr95": far})
        info.update({"text_near_fpr95": near0, "text_far_fpr95": far0})
        if best:
            info.update({"best": best[1], "pass": bool(best[0][0])})
            b = best[1]
            self.log(f"[G1-D] text score: near {near0:.2f} / far {far0:.2f} FPR95; best transported: {b['candidate']} "
                     f"({b['protos']}, omega {b['omega']}) near {b['near_fpr95']:.2f} / far {b['far_fpr95']:.2f}")
        return info

    # ------------------------------------------------------------------ part E
    def ladder_fit(self, stats: ClassStats) -> Transport:
        """What the method would use after seeing ``stats``: R0 -> R1 -> chosen rung."""
        c = self.cfg
        n_el = int(len(stats.eligible(c.n_min)))
        if stats.counts.sum() <= 0:
            return self.fits["R0"] if "R0" in self.fits else Transport("R0")
        if n_el < c.cold_min_classes or self.chosen_name in ("R0", "R1"):
            k = "R1" if self.chosen_name != "R0" else "R0"
            return fit_candidate({"kind": k}, self.id_text, stats, n_min=1, n_cap=c.n_cap)
        return fit_candidate(candidate_from_name(self.chosen_name, candidate_grid(c.lams, c.hs)), self.id_text, stats,
                             n_min=c.n_min, n_cap=c.n_cap)

    def part_e(self, omega: Optional[float] = None) -> Dict:
        assert self.fits, "run part_b first"
        c = self.cfg
        t0 = time.time()
        id_eval, ood_eval = c.cold_eval
        if not (self.has(id_eval) and self.has(ood_eval)):
            self.results["E"] = {"pass": False, "note": f"{id_eval} / {ood_eval} not in the cache"}
            return self.results["E"]
        if omega is None:
            omega = float(self.results.get("D", {}).get("best", {}).get("omega", 0.5))
            omega = 0.5 if omega >= 1.0 else omega
        fs = self.load(c.id_stream_set)
        perm = np.random.default_rng(c.seed).permutation(len(fs))
        ev_id, ev_ood = self.load(id_eval).feats, self.load(ood_eval).feats
        t_id = score_from_parts(*lse_parts(ev_id, self.id_text, self.neg_text, self.scale, device=c.device))
        t_ood = score_from_parts(*lse_parts(ev_ood, self.id_text, self.neg_text, self.scale, device=c.device))
        base = pair_metrics(t_id, t_ood)
        rows = [{"n_seen": 0, "rung": "R0", "n_classes": 0, "fpr95_text": base["fpr95"], "fpr95_vis": base["fpr95"],
                 "fpr95_fused": base["fpr95"], "auroc_fused": base["auroc"]}]
        for n in c.cold_ns:
            n = min(n, len(fs))
            idx = torch.as_tensor(perm[:n])
            st, _ = pseudo_label_stats(fs.feats[idx], self.id_text, self.scale, c.p_min, device=c.device)
            tr = self.ladder_fit(st)
            pos, negs = tr.apply(self.id_text), tr.apply(self.neg_text)
            v_id = score_from_parts(*lse_parts(ev_id, pos, negs, self.scale, device=c.device))
            v_ood = score_from_parts(*lse_parts(ev_ood, pos, negs, self.scale, device=c.device))
            mv, mf = pair_metrics(v_id, v_ood), pair_metrics(fuse(t_id, v_id, omega), fuse(t_ood, v_ood, omega))
            rows.append({"n_seen": n, "rung": tr.name, "n_classes": int(len(st.eligible(c.n_min))),
                         "fpr95_text": base["fpr95"], "fpr95_vis": mv["fpr95"], "fpr95_fused": mf["fpr95"],
                         "auroc_fused": mf["auroc"]})
            if n >= len(fs):
                break
        import pandas as pd

        df = pd.DataFrame(rows)
        self.tables["coldstart"] = df
        early = df[(df.n_seen > 0) & (df.n_seen <= 2048)]
        helped = early[early.fpr95_fused < early.fpr95_text]
        no_room = base["fpr95"] < 1.0                     # text score already near-perfect on this pair
        vacuous = getattr(self, "chosen_name", "R0") == "R0"   # the ladder never leaves raw text
        info = {"eval": f"{id_eval} vs {ood_eval}", "omega": omega,
                "pass": None if (no_room or vacuous) else bool(len(helped) > 0),
                "first_n_that_helps": int(helped.n_seen.min()) if len(helped) else None,
                "seconds": round(time.time() - t0, 1)}
        if no_room:
            info["note"] = f"text FPR95 is {base['fpr95']:.2f}% on {id_eval} vs {ood_eval}: nothing to gain"
        elif vacuous:
            info["note"] = "part B chose R0, so this curve only re-scores the text score (no map was tested)"
        self.results["E"] = info
        if info["pass"] is None:
            self.log(f"[G1-E] {info['note']}")
        else:
            self.log(f"[G1-E] fused score beats text from n = {info['first_n_that_helps']} confident-stream images"
                     if info["pass"] else "[G1-E] no gain within the first 2,048 images")
        return info

    # ------------------------------------------------------------------ decision and report
    def decide(self) -> Dict:
        r = self.results
        c1 = r.get("B", {}).get("pass", False)
        c2 = r.get("C", {}).get("pass", False)
        c3 = r.get("D", {}).get("pass", False)
        c4 = r.get("E", {}).get("pass", False)
        chosen = getattr(self, "chosen_name", "R0")
        if c1 and c2 and c3:
            verdict = "PASS"
            action = f"Use {chosen} as ReNeg's transport and continue to Seam 2 (checklist 2.1)."
        elif c1 and c2:
            verdict = "PARTIAL"
            action = (f"{chosen} generalises to OOD concepts but the static score gain is small. Use it to screen "
                      "near negatives in Seam 2 and re-test the score inside TANL at G2.")
        elif chosen.startswith("R1") or (r.get("C", {}).get("wins") and not c1):
            verdict = "R1-ONLY"
            action = "Only the gap shift is reliable. Use R1, and try Plan B for a richer map (checklist 1.6)."
        else:
            verdict = "FAIL"
            action = ("No map passed G1's tests. Before Plan B, run G1b_transport_diagnostics (checklist 1.5b): G1's "
                      "part B ranks centroids per name, not prototypes per image, and part D scores image space at the "
                      "text temperature; G1b corrects both and tests whether a better map could pay at all.")
        dec = {"verdict": verdict, "action": action, "chosen": chosen,
               "criteria": {"B_selfcheck_beats_R0": c1, "C_ood_concepts": c2, "D_static_score": c3,
                            "E_cold_start": "n/a" if c4 is None else c4},
               "kappa_star": r.get("B", {}).get("kappa_star"), "config": _jsonable(asdict(self.cfg)),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.results["decision"] = dec
        self.log(f"[G1] verdict {verdict}: {action}")
        return dec

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        for name, df in self.tables.items():
            if df is not None and len(df):
                df.to_csv(os.path.join(out_dir, f"g1_{name}.csv"), index=False)
        with open(os.path.join(out_dir, "g1_results.json"), "w") as f:
            json.dump(_jsonable(self.results), f, indent=1)
        path = os.path.join(out_dir, "g1_report.md")
        with open(path, "w") as f:
            f.write(self.report_markdown())
        self.log(f"[G1] saved tables, g1_results.json and g1_report.md to {out_dir}")
        return path

    def report_markdown(self) -> str:
        r = self.results
        d = r.get("decision", {})
        L = ["# Gate G1 report", "", f"**Verdict: {d.get('verdict', '-')}.** {d.get('action', '')}", ""]
        crit = d.get("criteria", {})
        L += ["| Criterion | Result |", "| --- | --- |"]
        for k, v in crit.items():
            L.append(f"| {k} | {v if isinstance(v, str) else ('pass' if v else 'fail')} |")
        L.append("")
        if "A" in r:
            a = r["A"]
            L += ["## A · Confident ID images", "",
                  f"{a['n_kept']} of {a['n_images']} images kept at p >= {a['p_min']}; {a['n_classes_eligible']} of "
                  f"{a['n_classes']} classes have at least {self.cfg.n_min}. Pseudo-label accuracy on kept images: "
                  f"{100 * a.get('pseudo_label_acc_kept', float('nan')):.1f}%.", ""]
        if "B" in r:
            b = r["B"]
            L += ["## B · Self-check on ID classes (no OOD data)", "",
                  f"Maps fitted on pseudo-labelled stream images, judged on {b.get('judged_on')} centroids of held-out "
                  f"WordNet subtrees. The test-time rule (pseudo-labels, alignment first) picks {b.get('online_choice')}.", "",
                  f"Groups from {b['group_source']} ({b['n_groups']} groups). Chosen: **{b['chosen']}** via {b['path']}; "
                  f"top-1 gain over R0 {100 * b['gain_top1_over_R0']:.1f} points, alignment gain "
                  f"{b['gain_align_over_R0']:+.3f}; kappa* = {b['kappa_star']:.2f}.", ""]
            L += _md_table(self.tables.get("selfcheck"), ["candidate", "top1", "mrr", "align", "b2"], 12)
        if "C" in r:
            cc = r["C"]
            L += ["## C · OOD concepts (evaluation only)", "",
                  f"Encoder check: min cosine to the bank's ID embeddings {cc.get('encoder_matches_bank_cos_min', float('nan')):.4f}. "
                  f"Best candidate on OOD concepts: {cc.get('ood_best_candidate', '-')}. Rank correlation between the "
                  f"self-check and the OOD test: {_fmt(cc.get('selfcheck_vs_ood_spearman'))} on top-1, "
                  f"{_fmt(cc.get('selfcheck_vs_ood_spearman_align'))} on alignment.", ""]
            df = self.tables.get("concepts")
            if df is not None and len(df):
                keep = df[df.candidate.isin(["R0", "R1", getattr(self, "chosen_name", "R0")])]
                L += _md_table(keep, ["dataset", "candidate", "top1", "align", "own_vs_id", "n_concepts"], 30)
            for ds, note in cc.get("notes", {}).items():
                L.append(f"- {ds}: {note}")
            L.append("")
        if "D" in r:
            L += ["## D · Static scores (FPR95 %, OpenOOD convention)", ""]
            s = self.tables.get("scores_summary")
            if s is not None and len(s):
                L += _md_table(s.sort_values(["protocol", "group", "fpr95"]), ["protocol", "group", "candidate", "protos",
                                                                              "omega", "fpr95", "auroc"], 40)
        if "E" in r:
            L += ["## E · Cold start", ""]
            L += _md_table(self.tables.get("coldstart"), ["n_seen", "rung", "n_classes", "fpr95_text", "fpr95_vis",
                                                          "fpr95_fused"], 20)
        return "\n".join(L) + "\n"


def _spearman(a, b) -> Optional[float]:
    """Rank correlation, or None when either side is constant (e.g. every candidate at 100% top-1)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _md_table(df, cols, n) -> List[str]:
    if df is None or not len(df):
        return ["(no rows)", ""]
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.head(n).iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines + [""]


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x
