"""Choose the transport rung and its lambda using ID classes only (no OOD data, no labels).

Leave-subtree-out validation: ID classes are grouped by WordNet subtree (dogs, birds,
vehicles, ...). Each fold holds out whole subtrees, fits the map on the remaining classes and
predicts the held-out classes' image centroids from their names alone. Holding out whole
subtrees makes the held-out names semantically far from the fitted ones, like negative labels.

Two scores per candidate, averaged over all held-out classes:

* discrimination (top1): T(t_c) retrieves its own centroid among all held-out centroids;
* alignment (align): mean cosine between T(t_c) and the centroid direction.

``b2`` (squared distance between the scaled prediction and the raw centroid) feeds
kappa* = sigma^2 / b^2, the trust R3 should give the transported prior.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .transport import ClassStats, candidate_grid, candidate_name, fit_candidate, l2n


def make_folds(groups: np.ndarray, classes: np.ndarray, n_folds: int = 5, seed: int = 0) -> List[np.ndarray]:
    """Assign whole groups to folds so that folds hold similar numbers of classes."""
    classes = np.asarray(classes)
    g = np.asarray(groups)[classes]
    uniq, counts = np.unique(g, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(uniq)), -counts))           # big groups first, random ties
    load = np.zeros(n_folds, dtype=np.int64)
    fold_of = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += counts[i]
    return [classes[np.array([fold_of[x] == f for x in g])] for f in range(n_folds)]


@torch.no_grad()
def retrieval_scores(pred: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """pred, targets: (n, D) unit vectors, row i of pred should find row i of targets."""
    sim = pred.double() @ targets.double().t()
    n = sim.shape[0]
    own = sim.diagonal()
    rank = (sim > own[:, None]).sum(1)                              # 0 = retrieved first
    return {"top1": float((rank == 0).double().mean()), "mrr": float((1.0 / (rank.double() + 1)).mean()),
            "align": float(own.mean()), "n": int(n)}


def leave_group_out(id_text: torch.Tensor, stats: ClassStats, groups: np.ndarray, candidates: Optional[list] = None,
                    n_min: float = 3, n_cap: float = 20, n_folds: int = 5, seed: int = 0, logger=None,
                    eval_stats: Optional[ClassStats] = None, n_min_eval: float = 2):
    """Score every candidate map by leave-subtree-out prediction of ID class centroids.

    Maps are always fitted on ``stats`` (what the method sees: pseudo-labelled stream images).
    Held-out targets come from ``eval_stats`` when given (e.g. the labelled OpenOOD validation
    split, which is meant for tuning), otherwise from ``stats`` itself. Pseudo-label targets favour
    raw text, because CLIP picked those images by their similarity to the class name.
    """
    import pandas as pd

    candidates = candidates or candidate_grid()
    elig = stats.eligible(n_min).numpy()
    if eval_stats is not None:
        elig = np.intersect1d(elig, eval_stats.eligible(n_min_eval).numpy())
    if len(elig) < 2 * n_folds:
        raise ValueError(f"only {len(elig)} ID classes have enough images; lower p_min or n_min")
    folds = make_folds(groups, elig, n_folds=n_folds, seed=seed)
    ev = eval_stats if eval_stats is not None else stats
    dirs, raw = ev.directions(), ev.means()
    T = l2n(torch.as_tensor(id_text).detach().to("cpu", torch.float64))
    acc: Dict[str, Dict[str, float]] = {}
    for fi, test in enumerate(folds):
        if len(test) < 2:
            continue
        train = np.setdiff1d(stats.eligible(n_min).numpy(), test)       # every fitted class outside the fold
        rho = float(raw[torch.as_tensor(np.intersect1d(train, elig))].norm(dim=1).mean())   # typical centroid norm
        tgt = dirs[torch.as_tensor(test)]
        tgt_raw = raw[torch.as_tensor(test)]
        for cand in candidates:
            tr = fit_candidate(cand, T, stats, classes=train, n_min=n_min, n_cap=n_cap)
            P = tr.apply(T[torch.as_tensor(test)]).double()
            r = retrieval_scores(P, tgt)
            b2 = float(((rho * P - tgt_raw) ** 2).sum(1).mean())
            a = acc.setdefault(candidate_name(cand), {"kind": cand["kind"], "lam": cand.get("lam", np.nan),
                                                      "h": cand.get("h", np.nan), "top1": 0.0, "mrr": 0.0,
                                                      "align": 0.0, "b2": 0.0, "n_eval": 0})
            n = r["n"]
            for k in ("top1", "mrr", "align"):
                a[k] += r[k] * n
            a["b2"] += b2 * n
            a["n_eval"] += n
        if logger:
            logger.info(f"[selfcheck] fold {fi + 1}/{len(folds)}: {len(test)} held-out classes, {len(train)} fitted")
    rows = []
    for name, a in acc.items():
        n = max(a["n_eval"], 1)
        rows.append({"candidate": name, "kind": a["kind"], "lam": a["lam"], "h": a["h"], "top1": a["top1"] / n,
                     "mrr": a["mrr"] / n, "align": a["align"] / n, "b2": a["b2"] / n, "n_eval": a["n_eval"]})
    df = pd.DataFrame(rows)
    return df.sort_values(["top1", "align"], ascending=False).reset_index(drop=True)


def clearly_better(a: Dict, b: Dict, eps: float = 0.01, eps_align: float = 0.01, tie: float = 0.005) -> bool:
    """a beats b if it retrieves better by ``eps`` top-1, or ties on top-1 (within ``tie``) and
    sits closer to the real centroids by ``eps_align`` cosine. The second case matters when
    retrieval is already near-perfect but the prototypes are still far from the images."""
    if a["top1"] >= b["top1"] + eps:
        return True
    return a["top1"] >= b["top1"] - tie and a["align"] >= b["align"] + eps_align


def choose_ladder(df, eps: float = 0.01, eps_align: float = 0.01) -> Dict:
    """Walk up the ladder: a richer map is used only if it is clearly better than the current one.

    Returns the chosen candidate plus, for the record, the best candidate of every kind.
    """
    best = {}
    for kind, sub in df.groupby("kind"):
        s = sub.sort_values(["top1", "align"], ascending=False).iloc[0]
        best[kind] = s.to_dict()
    if "R0" not in best:
        raise ValueError("the candidate list must contain R0")
    current = best["R0"]
    steps = ["R0"]
    if "R1" in best and clearly_better(best["R1"], current, eps, eps_align):
        current = best["R1"]
        steps.append("R1")
    richer = [best[k] for k in ("R2", "B1", "B2") if k in best]
    richer = [r for r in richer if clearly_better(r, current, eps, eps_align)]
    if richer:
        current = max(richer, key=lambda r: (round(r["top1"], 3), r["align"]))
        steps.append(current["kind"])
    return {"chosen": current["candidate"], "chosen_row": current, "path": " -> ".join(steps),
            "best_per_kind": {k: v["candidate"] for k, v in best.items()}, "eps": eps, "eps_align": eps_align,
            "gain_top1_over_R0": float(current["top1"] - best["R0"]["top1"]),
            "gain_align_over_R0": float(current["align"] - best["R0"]["align"])}


def choose_online(df, guard: float = 0.10, eps_align: float = 0.01) -> Dict:
    """The rule the method can use at test time, when the only targets are pseudo-labelled centroids.

    Those targets favour raw text on top-1, so alignment decides: the candidate with the highest
    alignment wins if it beats R0's alignment by ``eps_align`` and its top-1 is within ``guard``
    of R0's (a guard against maps that collapse all prototypes together).
    """
    r0 = df[df.candidate == "R0"].iloc[0]
    ok = df[df.top1 >= r0["top1"] - guard]
    best = ok.sort_values(["align", "top1"], ascending=False).iloc[0]
    if best["align"] < r0["align"] + eps_align:
        best = r0
    return {"chosen": best["candidate"], "top1": float(best["top1"]), "align": float(best["align"]),
            "r0_top1": float(r0["top1"]), "r0_align": float(r0["align"]), "guard": guard}


def kappa_star(stats: ClassStats, n_min: float, b2: float) -> Dict[str, float]:
    """kappa* = sigma^2 / b^2: how many single test images the transported prior is worth."""
    elig = stats.eligible(n_min)
    sigma2 = stats.spread(elig)
    return {"sigma2": sigma2, "b2": float(b2), "kappa_star": float(sigma2 / max(b2, 1e-12))}


def candidate_from_name(name: str, candidates: Optional[list] = None) -> Dict:
    for c in candidates or candidate_grid():
        if candidate_name(c) == name:
            return c
    raise KeyError(name)
