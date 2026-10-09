"""Seam 2: knowledge-graph near-OOD negatives, grounded in the test stream, admitted by an LCB gate.

Per test batch:

1. Score every image with what is admitted so far::

       S(v) = (1 - omega) * [LSE_tau(v . T_id) - LSE_tau(v . [T_neg ; T_adm])]
              +     omega * [LSE_tau(v . V_id) - LSE_tau(v . [V_bg ; V_adm])]

   T_* are text prototypes (ID names, NegLabel's far negatives, admitted KG candidates), V_* image prototypes
   (ID image centroids, a background set, and the visual prototypes of admitted candidates).
2. Let the batch vote: an image whose nearest text prototype (over ID names and the whole KG pool) is a pool
   candidate j is "caught" by j. j keeps a running sum of the caught images (its visual prototype) and of their
   evidence x(v): how untypical v is for its nearest ID image centroid, in units of that class's spread. Stolen
   ID images are typical of an ID class (x near the ID level); real near-OOD images are not.
3. Gate: the evidence mean is shrunk toward the candidate's KG cluster (graph sharing), and a lower confidence
   bound decides::

       m_j   = (sum_x_j + gamma * mean_x_cluster) / (n_j + gamma)
       LCB_j = m_j - mu_id - beta * sigma / sqrt(n_j + gamma)

   admit if LCB_j > delta_in (and n_j >= n_ev), evict if LCB_j < delta_out, at most q per cluster and M_near in
   total. mu_id is the ID level of x, measured on confident ID images.

Nothing here uses labels or OOD data; the ID statistics come from confident (pseudo-labelled) ID images.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from .transport import l2n


@dataclass
class Seam2Config:
    tau: float = 100.0
    omega: float = 0.25
    batch: int = 256
    k_vis: int = 2            # caught images before a candidate gets a visual prototype
    n_ev: int = 3             # caught images before the gate looks at a candidate
    gamma: float = 2.0        # cluster pseudo-count
    beta: float = 1.0         # LCB width
    delta_in: float = 0.25    # admit above this (units of the ID spread of x)
    delta_out: float = 0.0    # evict below this
    q: int = 5                # admitted per cluster
    m_near: int = 500         # admitted in total
    vis_only_untypical: bool = False  # visual prototype from caught images above the ID level only
    gate: bool = True         # False: every pool candidate is admitted from the start (ungated ablation)
    evidence: str = "margin"  # "margin": candidate's text similarity minus the best ID name's, on its caught images
                              # "untypical": image-side untypicality for the nearest ID centroid
    soft: bool = True         # soft gate: each candidate weighs pi_j in the negative mass instead of in/out
    prior: float = 0.05       # pi before any evidence (per candidate, or per KG band via pool_prior)
    temp: float = 0.5         # soft gate temperature (LCB units)
    seed: int = 0
    dtype: str = "float64"    # "float32" is about twice as fast on CPU (scores agree to ~1e-5)


@dataclass
class IdModel:
    """What the method knows about ID data in image space (from confident ID images, no labels)."""
    centroids: torch.Tensor   # (C', D) unit ID image centroids
    member_sim: torch.Tensor  # (C',) mean similarity of members to their centroid
    sigma: float              # spread of member similarity
    mu_x: float = 0.0         # mean untypicality of ID images (the ID level of the evidence)
    sd_x: float = 1.0

    def untypicality(self, X: torch.Tensor) -> torch.Tensor:
        S = X.to(self.centroids.dtype) @ self.centroids.T
        best, idx = S.max(1)
        return (self.member_sim[idx] - best) / self.sigma


def id_model_from_stats(stats, n_min: float, X_id_sample: Optional[torch.Tensor] = None) -> IdModel:
    """ID image centroids from pseudo-labelled class sums; spread and ID level from a sample of ID images."""
    el = stats.eligible(n_min)
    C = stats.directions()[el].double()
    r = stats.means()[el].norm(dim=1).double()
    sigma = 0.043
    m = IdModel(C, r, sigma)
    if X_id_sample is not None and len(X_id_sample):
        X = l2n(X_id_sample.double())
        S = X @ C.T
        best, idx = S.max(1)
        sigma = float((best - r[idx]).std())
        m = IdModel(C, r, sigma)
        x = m.untypicality(X)
        m.mu_x, m.sd_x = float(x.mean()), float(x.std())
    return m


class Seam2:
    def __init__(self, id_text, neg_text, pool_text, pool_cluster, idm: IdModel, background: torch.Tensor,
                 cfg: Optional[Seam2Config] = None, pool_prior: Optional[np.ndarray] = None):
        self.cfg = cfg or Seam2Config()
        self.dt = dt = torch.float64 if self.cfg.dtype == "float64" else torch.float32
        self.T_id = l2n(torch.as_tensor(id_text).to(dt))
        self.T_neg = l2n(torch.as_tensor(neg_text).to(dt))
        self.T_pool = l2n(torch.as_tensor(pool_text).to(dt))
        self.cluster = np.asarray(pool_cluster)
        self.n_clusters = int(self.cluster.max()) + 1 if len(self.cluster) else 0
        self.idm = idm
        self.V_id = idm.centroids.to(dt)
        self.V_bg = l2n(torch.as_tensor(background).to(dt))
        K, D = self.T_pool.shape
        self.n = np.zeros(K)
        self.n_vis = np.zeros(K)
        self.fsum = torch.zeros(K, D, dtype=dt)
        self.xsum = np.zeros(K)
        self.admitted = np.ones(K, bool) if not self.cfg.gate else np.zeros(K, bool)
        self.lcb = np.full(K, -np.inf)
        self.prior = np.full(K, self.cfg.prior) if pool_prior is None else np.asarray(pool_prior, float)
        self.pi = self.prior.copy() if self.cfg.gate else np.ones(K)
        self.x2sum = np.zeros(K)
        self.history = []

    # ------------------------------------------------------------------ scoring
    def _lse(self, X, P):
        return torch.logsumexp(self.cfg.tau * X @ P.T, 1)

    def _lse_w(self, X, P, logw):
        return torch.logsumexp(self.cfg.tau * X @ P.T + logw[None, :], 1)

    def text_part(self, X: torch.Tensor) -> torch.Tensor:
        """LSE over ID names minus LSE over far negatives and (weighted / admitted) pool names."""
        c = self.cfg
        if c.soft and c.gate:
            w = torch.as_tensor(self.pi, dtype=self.dt)
            use = w > 1e-4
            logw = torch.cat([torch.zeros(self.T_neg.shape[0], dtype=self.dt), torch.log(w[use])])
            return self._lse(X, self.T_id) - self._lse_w(X, torch.cat([self.T_neg, self.T_pool[use]]), logw)
        adm = torch.as_tensor(self.admitted)
        T_negs = torch.cat([self.T_neg, self.T_pool[adm]]) if adm.any() else self.T_neg
        return self._lse(X, self.T_id) - self._lse(X, T_negs)

    def image_part(self, X: torch.Tensor) -> torch.Tensor:
        """LSE over ID image prototypes minus LSE over the background and the candidates' visual prototypes."""
        c = self.cfg
        if c.soft and c.gate:
            vis = (self.n_vis >= c.k_vis) & (self.pi > 1e-4)
            if vis.any():
                logw = torch.cat([torch.zeros(self.V_bg.shape[0], dtype=self.dt),
                                  torch.log(torch.as_tensor(self.pi[vis], dtype=self.dt))])
                return self._lse(X, self.V_id) - self._lse_w(X, torch.cat([self.V_bg, l2n(self.fsum[torch.as_tensor(vis)])]), logw)
            return self._lse(X, self.V_id) - self._lse(X, self.V_bg)
        vis = torch.as_tensor(self.admitted & (self.n_vis >= c.k_vis))
        V_negs = torch.cat([self.V_bg, l2n(self.fsum[vis])]) if vis.any() else self.V_bg
        return self._lse(X, self.V_id) - self._lse(X, V_negs)

    def score(self, X: torch.Tensor) -> np.ndarray:
        c = self.cfg
        lt = self.text_part(X)
        if c.omega <= 0:
            return lt.numpy()
        return ((1 - c.omega) * lt + c.omega * self.image_part(X)).numpy()

    # ------------------------------------------------------------------ evidence and gate
    def update(self, X: torch.Tensor):
        c = self.cfg
        P = torch.cat([self.T_id, self.T_pool])
        top = (X @ P.T).argmax(1).numpy() - self.T_id.shape[0]
        hit = np.nonzero(top >= 0)[0]
        if len(hit) == 0:
            return
        j = top[hit]
        Xh = X[torch.as_tensor(hit)]
        if c.evidence == "margin":
            x = ((Xh * self.T_pool[torch.as_tensor(j)]).sum(1) - (Xh @ self.T_id.T).max(1).values).numpy()
        else:
            x = self.idm.untypicality(Xh).numpy()
        np.add.at(self.n, j, 1)
        np.add.at(self.xsum, j, x)
        np.add.at(self.x2sum, j, x * x)
        keep = (self.idm.untypicality(Xh).numpy() > self.idm.mu_x) if c.vis_only_untypical else np.ones(len(x), bool)
        if keep.any():
            self.fsum.index_add_(0, torch.as_tensor(j[keep]), X[torch.as_tensor(hit[keep])])
            np.add.at(self.n_vis, j[keep], 1)
        if c.gate:
            self._gate()

    def _gate(self):
        c, idm = self.cfg, self.idm
        n_k = np.bincount(self.cluster, weights=self.n, minlength=self.n_clusters)
        x_k = np.bincount(self.cluster, weights=self.xsum, minlength=self.n_clusters)
        if c.evidence == "margin":
            tot = max(self.n.sum(), 1.0)
            mu0 = 0.0                                      # a name that only ties the best ID name
            sd = float(np.sqrt(max(self.x2sum.sum() / tot - (self.xsum.sum() / tot) ** 2, 1e-8)))
        else:
            mu0, sd = idm.mu_x, idm.sd_x
        mean_k = np.where(n_k > 0, x_k / np.maximum(n_k, 1), mu0)[self.cluster]
        m = (self.xsum + c.gamma * mean_k) / (self.n + c.gamma)
        lcb = (m - mu0) / sd - c.beta / np.sqrt(self.n + c.gamma)
        lcb = np.where(self.n >= c.n_ev, lcb, -np.inf)
        self.lcb = lcb
        if c.soft:
            z = np.clip((lcb - c.delta_in) / c.temp, -30, 30)
            post = 1.0 / (1.0 + np.exp(-z))
            self.pi = np.where(np.isfinite(lcb), post, self.prior)
            self.admitted = self.pi > 0.5
            return
        want = np.where(self.admitted, lcb > c.delta_out, lcb > c.delta_in)
        idx = np.nonzero(want)[0]
        if len(idx):                                    # at most q per cluster, M_near in total, best LCB first
            order = idx[np.lexsort((-lcb[idx], self.cluster[idx]))]
            cl = self.cluster[order]
            first = np.r_[0, np.nonzero(np.diff(cl))[0] + 1]
            rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
            order = order[rank < c.q]
            order = order[np.argsort(-lcb[order])][: c.m_near]
            new = np.zeros_like(self.admitted)
            new[order] = True
            self.admitted = new
        else:
            self.admitted[:] = False

    # ------------------------------------------------------------------ stream
    def run(self, X: torch.Tensor, seed: Optional[int] = None) -> np.ndarray:
        """Score a stream in batches (each batch is scored before it updates the state). Returns scores in the
        stream's own order."""
        c = self.cfg
        X = l2n(torch.as_tensor(X).to(self.dt))
        out = np.zeros(len(X))
        for s in range(0, len(X), c.batch):
            v = X[s : s + c.batch]
            out[s : s + c.batch] = self.score(v)
            self.update(v)
            self.history.append(int(self.admitted.sum()))
        return out

    def summary(self) -> Dict:
        return {"admitted": int(self.admitted.sum()), "candidates_with_evidence": int((self.n > 0).sum()),
                "images_caught": int(self.n.sum())}
