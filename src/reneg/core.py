"""ReNeg's two test-time components, shared by every base method (NegLabel, TANL).

KGGate
    The WordNet near-OOD candidate pool with a soft LCB gate. Every test image is "caught" by its nearest text
    prototype over [ID names ; pool names]. A pool name collects the caught images' text margin (its cosine minus
    the best ID name's) and their sum (its visual prototype). The evidence mean is shrunk toward the name's KG
    cluster, and the gate weight is::

        LCB_j = (m_j - 0) / sd - beta / sqrt(n_j + gamma)          (sd: pooled sd of all margins)
        pi_j  = sigmoid((LCB_j - delta) / T)    once n_j >= n_ev,  prior pi_0 before

    pi_j weighs name j in the negative mass of the text score and its visual prototype in the image score.

OnlineIdModel
    ID image prototypes built from the test stream itself (reset with the method): an image predicted as ID class
    c is added to c's sum when it is admitted (rule chosen by the caller); c has a prototype once its admitted
    weight reaches n_min. A background of random directions around the running image mean stands for "an
    image" in general.

Scores are log-odds (higher = more ID):
    text  lt = LSE(tau v.T_id) - LSE(tau v.[T_neg ; T_pool] + [0 ; log pi])        (or TANL's version)
    image li = LSE(tau v.V_id) - LSE(tau v.[V_bg ; V_pool] + [0 ; log pi])
    fused s  = (1 - omega) lt + omega li
Every batch is scored with the state left by earlier batches, then updates it (score-then-update).

Devices: the tensors (pool text, visual sums, ID sums, background) live on ``device`` (a GPU when the caller passes
one); the small per-name bookkeeping (counts, margins, pi) stays in numpy on the CPU. The arithmetic is the same on
either device, so a GPU run differs from a CPU run only by float rounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .transport import l2n


@dataclass
class ReNegConfig:
    tau: float = 100.0
    omega: float = 0.25        # weight of the image score in the fused log-odds
    use_pool: bool = True      # KG candidate pool (gate, catching, visual prototypes)
    pool_in_text: bool = True  # gated KG names also join the text score's negatives (False when the base method
                               # already scores them, e.g. TANL with the KG names appended to its corpus)
    pool_frac: float = 1.0     # TANL base: share of TANL's averaged terms (those over the most negatives) that
                               # also count the KG mass; 1 = every term
    use_image: bool = True     # image score (ID prototypes vs background + visual prototypes of KG names)
    use_vis: bool = True       # visual prototypes of KG names in the image score
    image_mode: str = "plain"  # "plain": li as below; "kg_clip": li = min(0, LSE(V_id) - LSE(V_kg + log pi)),
                               # a pure penalty for looking more like an admitted KG name's images than like ID
    clip_margin: float = 0.0   # kg_clip modes: penalise only beyond this margin (LSE units; 1 = 0.01 in cosine)
    id_weight: float = 1.0     # image score = id_weight * [ID prototypes vs background]
    ood_weight: float = 1.0    #             + ood_weight * [background vs background + KG visual prototypes]
    # gate
    k_vis: int = 2             # caught images before a name gets a visual prototype
    n_ev: int = 2              # caught images before the gate looks at a name
    gamma: float = 2.0         # cluster pseudo-count
    beta: float = 0.5          # LCB width
    delta: float = 1.0         # gate centre (LCB units)
    temp: float = 0.5          # gate temperature (LCB units)
    prior: float = 0.2         # pi before any evidence
    gate: bool = True          # False: every name has pi = 1 (ungated ablation)
    vote: bool = False         # second gate test: the share of a name's caught images that the base method's own
                               # score places below its adaptive (Otsu) threshold; pi *= sigmoid((share - rho) / vote_temp)
    rho: float = 0.5
    vote_temp: float = 0.1
    # online ID image model
    id_admit: str = "softmax"  # "softmax" (p >= p_min), "all", "base" (the base method's own ID mask, e.g. TANL's),
                               # "softmax+base" (both agree: CLIP's softmax >= p_min AND the base method's ID mask)
    id_exclude: str = "none"   # also keep out images caught by a pool name: "none", "caught" (any name),
                               # "admitted" (a name with pi > 0.5)
    p_min: float = 0.5
    n_min: float = 1.0
    bg_noise: float = 0.25
    n_bg: int = 10000
    seed: int = 0
    dtype: str = "float32"

    def with_(self, **kw) -> "ReNegConfig":
        return replace(self, **kw)

    @property
    def torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32


def lse_w(S: torch.Tensor, tau: float, logw: Optional[torch.Tensor] = None) -> torch.Tensor:
    Z = tau * S
    if logw is not None:
        Z = Z + logw[None, :]
    return torch.logsumexp(Z, 1)


def _rows(n_or_mask: np.ndarray, device) -> torch.Tensor:
    """Row indices of a numpy bool mask, as a long tensor on ``device`` (indexing with it = indexing with the mask)."""
    return torch.as_tensor(np.nonzero(np.asarray(n_or_mask))[0], dtype=torch.long, device=device)


class KGGate:
    def __init__(self, pool_text: torch.Tensor, pool_cluster, cfg: ReNegConfig, pool_prior=None, device="cpu"):
        self.cfg = cfg
        dt = cfg.torch_dtype
        self.device = torch.device(device)
        self.T = l2n(torch.as_tensor(pool_text).to(device=self.device, dtype=dt))
        self.cluster = np.asarray(pool_cluster, dtype=np.int64)
        self.n_clusters = int(self.cluster.max()) + 1 if len(self.cluster) else 0
        K = self.T.shape[0]
        self.prior = np.full(K, cfg.prior, float) if pool_prior is None else np.asarray(pool_prior, float)
        self.reset()

    @property
    def K(self) -> int:
        return int(self.T.shape[0])

    def reset(self) -> None:
        K, D = self.T.shape
        self.n = np.zeros(K)
        self.xsum = np.zeros(K)
        self.x2sum = np.zeros(K)
        self.n_vis = np.zeros(K)
        self.vsum = np.zeros(K)
        self.fsum = torch.zeros(K, D, dtype=self.T.dtype, device=self.device)
        self.lcb = np.full(K, -np.inf)
        self.pi = self.prior.copy() if self.cfg.gate else np.ones(K)

    def _log_pi(self, mask: np.ndarray) -> torch.Tensor:
        return torch.log(torch.as_tensor(self.pi[mask], dtype=self.T.dtype, device=self.device))

    # ---------------------------------------------------------------- what the scores use
    def text_terms(self, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
        use = self.pi > eps
        return self.T[_rows(use, self.device)], self._log_pi(use)

    def visual_terms(self, eps: float = 1e-4) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        vis = (self.n_vis >= self.cfg.k_vis) & (self.pi > eps)
        if not vis.any():
            return None, None
        return l2n(self.fsum[_rows(vis, self.device)]), self._log_pi(vis)

    def admitted(self) -> np.ndarray:
        return self.pi > 0.5

    def catch_weight(self, X: torch.Tensor, idm_text=None) -> torch.Tensor:
        """Per image: pi of the pool name that catches it in text (nearest prototype over [ID ; pool]), else 0."""
        T_id = self._T_id
        best_id = (X @ T_id.T).max(1).values
        bp, j = (X @ self.T.T).max(1)
        w = torch.as_tensor(self.pi, dtype=X.dtype, device=X.device)[j]
        return torch.where(bp > best_id, w, torch.zeros_like(w))

    # ---------------------------------------------------------------- evidence and gate
    def update(self, X: torch.Tensor, T_id: torch.Tensor, ood_vote: Optional[np.ndarray] = None) -> np.ndarray:
        """Catch each image by its nearest text prototype over [ID ; pool]. Returns, per image, the index of the
        pool name that caught it, or -1. ood_vote (bool per image): the base method's own verdict, for the vote."""
        self._T_id = T_id
        S_id = X @ T_id.T
        S_pool = X @ self.T.T
        best_id = S_id.max(1).values
        bp, j = S_pool.max(1)
        hit_t = bp > best_id
        hit = hit_t.cpu().numpy()
        jn = j.cpu().numpy()
        caught_by = np.where(hit, jn, -1)
        if not hit.any():
            return caught_by
        jh = jn[hit]
        x = (bp[hit_t] - best_id[hit_t]).double().cpu().numpy()
        np.add.at(self.n, jh, 1)
        np.add.at(self.xsum, jh, x)
        np.add.at(self.x2sum, jh, x * x)
        self.fsum.index_add_(0, torch.as_tensor(jh, dtype=torch.long, device=self.device),
                             X[hit_t].to(device=self.device, dtype=self.fsum.dtype))
        np.add.at(self.n_vis, jh, 1)
        if ood_vote is not None:
            np.add.at(self.vsum, jh, np.asarray(ood_vote, float)[hit])
        if self.cfg.gate:
            self._gate()
        return caught_by

    def exclusion(self, caught_by: np.ndarray, rule: str) -> np.ndarray:
        """Images to keep out of the ID model: caught by any pool name, or by an admitted one (pi > 0.5)."""
        if rule == "none":
            return np.zeros(len(caught_by), bool)
        hit = caught_by >= 0
        if rule == "caught":
            return hit
        if rule == "admitted":
            out = np.zeros(len(caught_by), bool)
            out[hit] = self.pi[caught_by[hit]] > 0.5
            return out
        raise ValueError(f"unknown id_exclude rule {rule!r}")

    def _gate(self) -> None:
        c = self.cfg
        tot = max(self.n.sum(), 1.0)
        sd = float(np.sqrt(max(self.x2sum.sum() / tot - (self.xsum.sum() / tot) ** 2, 1e-12)))
        n_k = np.bincount(self.cluster, weights=self.n, minlength=self.n_clusters)
        x_k = np.bincount(self.cluster, weights=self.xsum, minlength=self.n_clusters)
        mean_k = np.where(n_k > 0, x_k / np.maximum(n_k, 1), 0.0)[self.cluster]
        m = (self.xsum + c.gamma * mean_k) / (self.n + c.gamma)
        lcb = m / sd - c.beta / np.sqrt(self.n + c.gamma)
        lcb = np.where(self.n >= c.n_ev, lcb, -np.inf)
        self.lcb = lcb
        z = np.clip((lcb - c.delta) / c.temp, -30, 30)
        pi = 1.0 / (1.0 + np.exp(-z))
        if c.vote:
            share = (self.vsum + c.gamma * 0.5) / (self.n + c.gamma)      # shrunk toward an undecided 0.5
            pi = pi / (1.0 + np.exp(-np.clip((share - c.rho) / c.vote_temp, -30, 30)))
        self.pi = np.where(np.isfinite(lcb), pi, self.prior)

    def summary(self) -> Dict:
        return {"admitted": int(self.admitted().sum()), "with_evidence": int((self.n > 0).sum()),
                "caught": int(self.n.sum()), "pi_mass": round(float(self.pi.sum()), 1)}


class OnlineIdModel:
    def __init__(self, n_classes: int, dim: int, cfg: ReNegConfig, device="cpu"):
        self.cfg = cfg
        dt = cfg.torch_dtype
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(int(cfg.seed))           # drawn on the CPU: the same noise on any device
        self.noise = l2n(torch.randn(cfg.n_bg, dim, generator=g)).to(device=self.device, dtype=dt)
        self.C, self.D, self.dt = n_classes, dim, dt
        self.reset()

    def reset(self) -> None:
        self.sums = torch.zeros(self.C, self.D, dtype=self.dt, device=self.device)
        self.counts = np.zeros(self.C)
        self.img_sum = torch.zeros(self.D, dtype=self.dt, device=self.device)
        self.img_n = 0
        self._V = None
        self._bg = None

    def ready(self) -> bool:
        return self._V is not None and self._bg is not None

    def prototypes(self) -> Optional[torch.Tensor]:
        return self._V

    def background(self) -> Optional[torch.Tensor]:
        return self._bg

    def add(self, X: torch.Tensor, labels: torch.Tensor, admit: np.ndarray, weights: Optional[np.ndarray] = None):
        X = X.to(device=self.device, dtype=self.dt)
        admit = np.asarray(admit, bool)
        if admit.any():
            idx = _rows(admit, self.device)
            lab = torch.as_tensor(labels).to(self.device).long()[idx]
            lab_np = lab.cpu().numpy()
            if weights is None:
                self.sums.index_add_(0, lab, X[idx])
                np.add.at(self.counts, lab_np, 1.0)
            else:
                w = torch.as_tensor(weights[admit], dtype=X.dtype, device=self.device)
                self.sums.index_add_(0, lab, X[idx] * w[:, None])
                np.add.at(self.counts, lab_np, weights[admit])
        self.img_sum += X.sum(0)
        self.img_n += len(X)
        el = self.counts >= self.cfg.n_min
        self._V = l2n(self.sums[_rows(el, self.device)]) if el.any() else None
        m = l2n(self.img_sum[None])[0]
        self._bg = l2n(m[None, :] + self.cfg.bg_noise * self.noise)

    def summary(self) -> Dict:
        return {"id_classes": int((self.counts >= self.cfg.n_min).sum()), "id_images": round(float(self.counts.sum()), 1)}


def image_score(X: torch.Tensor, idm: OnlineIdModel, gate: Optional[KGGate], cfg: ReNegConfig) -> Optional[torch.Tensor]:
    """li, or None while the ID model has no prototype yet.

    li = LSE(V_id) - LSE([V_bg ; V_kg] + [0 ; log pi]) splits exactly into
         [LSE(V_id) - LSE(V_bg)]                          (ID prototypes vs background: helps far-OOD)
       + [LSE(V_bg) - LSE([V_bg ; V_kg] + [0 ; log pi])]  (penalty for resembling a KG visual prototype: near-OOD)
    weighted by id_weight and ood_weight (both 1 = the plain form)."""
    if not idm.ready():
        return None
    V, bg = idm.prototypes(), idm.background()
    if cfg.image_mode == "kg_clip_max":                     # nearest KG prototype vs nearest ID prototype
        if gate is None or not cfg.use_vis:
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        Vp, lw = gate.visual_terms()
        if Vp is None:
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        s_id = (X @ V.T).max(1).values
        s_kg = (X @ Vp.T + lw[None, :] / cfg.tau).max(1).values
        return torch.clamp(cfg.tau * (s_id - s_kg) + cfg.clip_margin, max=0.0)
    if cfg.image_mode in ("kg_clip", "kg_clip_caught"):
        if gate is None or not cfg.use_vis:
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        Vp, lw = gate.visual_terms()
        if Vp is None:
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        pen = torch.clamp(lse_w(X @ V.T, cfg.tau) - lse_w(X @ Vp.T, cfg.tau, lw) + cfg.clip_margin, max=0.0)
        if cfg.image_mode == "kg_clip_caught":            # only images a KG name also catches in text, by its pi
            pen = pen * gate.catch_weight(X, idm_text=None)
        return pen
    l_bg = lse_w(X @ bg.T, cfg.tau)
    li_id = lse_w(X @ V.T, cfg.tau) - l_bg
    li_ood = torch.zeros_like(li_id)
    if gate is not None and cfg.use_vis:
        Vp, lw = gate.visual_terms()
        if Vp is not None:
            neg = torch.cat([bg, Vp])
            logw = torch.cat([torch.zeros(bg.shape[0], dtype=lw.dtype, device=lw.device), lw])
            li_ood = l_bg - lse_w(X @ neg.T, cfg.tau, logw)
    return cfg.id_weight * li_id + cfg.ood_weight * li_ood


def otsu_threshold(v: np.ndarray, n_cand: int = 200) -> float:
    """Threshold maximising the between-class variance of v (Otsu), over candidate quantiles."""
    v = np.sort(np.asarray(v, float))
    if len(v) < 10:
        return -np.inf
    cand = np.quantile(v, np.linspace(0.01, 0.99, n_cand))
    c1, n = np.cumsum(v), len(v)
    k = np.clip(np.searchsorted(v, cand, side="left"), 1, n - 1)
    w0 = k / n
    m0, m1 = c1[k - 1] / k, (c1[-1] - c1[k - 1]) / (n - k)
    return float(cand[int(np.argmax(w0 * (1 - w0) * (m0 - m1) ** 2))])


class VoteHistory:
    """Running history of the base score; vote(v) = v below the Otsu threshold of the history so far."""

    def __init__(self, length: int = 20000):
        self.length = length
        self.h = np.zeros(0)

    def vote(self, v: np.ndarray) -> np.ndarray:
        self.h = np.r_[self.h, np.asarray(v, float)][-self.length:]
        return np.asarray(v, float) < otsu_threshold(self.h)


def admit_mask(X: torch.Tensor, T_id: torch.Tensor, cfg: ReNegConfig, base_mask: Optional[np.ndarray] = None):
    """(labels, mask) of the images that update the ID model."""
    p = torch.softmax(cfg.tau * X @ T_id.T, 1)
    conf, lab = p.max(1)
    if cfg.id_admit == "all":
        return lab, np.ones(len(X), bool)
    soft = (conf >= cfg.p_min).cpu().numpy()
    if cfg.id_admit in ("base", "softmax+base"):
        if base_mask is None:
            raise ValueError(f"id_admit={cfg.id_admit!r} needs the base method's ID mask")
        base = np.asarray(base_mask, bool)
        return lab, base if cfg.id_admit == "base" else soft & base
    if cfg.id_admit != "softmax":
        raise ValueError(f"unknown id_admit rule {cfg.id_admit!r}")
    return lab, soft


def fuse(lt: torch.Tensor, li: Optional[torch.Tensor], cfg: ReNegConfig) -> torch.Tensor:
    if li is None or not cfg.use_image or cfg.omega <= 0:
        return lt
    return (1 - cfg.omega) * lt + cfg.omega * li


class ReNegNegLabel:
    """ReNeg on NegLabel's fixed 10,000 negatives (the ablation base). Same interface as oodlab's StreamMethod."""
    name = "reneg_neglabel"

    def __init__(self, id_text, neg_text, pool_text=None, pool_cluster=None, cfg: Optional[ReNegConfig] = None,
                 pool_prior=None, device: str = "cpu"):
        self.cfg = cfg or ReNegConfig()
        dt = self.cfg.torch_dtype
        self.device = torch.device(device)
        self.T_id = l2n(torch.as_tensor(id_text).to(device=self.device, dtype=dt))
        self.T_neg = l2n(torch.as_tensor(neg_text).to(device=self.device, dtype=dt))
        self.gate = None
        if pool_text is not None and self.cfg.use_pool and len(pool_text):
            self.gate = KGGate(pool_text, pool_cluster, self.cfg, pool_prior, device=self.device)
        self.idm = OnlineIdModel(self.T_id.shape[0], self.T_id.shape[1], self.cfg, device=self.device)
        self.votes = VoteHistory()

    def reset(self) -> None:
        if self.gate is not None:
            self.gate.reset()
        self.idm.reset()
        self.votes = VoteHistory()

    def text_score(self, X: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if self.gate is None or not c.pool_in_text:
            return lse_w(X @ self.T_id.T, c.tau) - lse_w(X @ self.T_neg.T, c.tau)
        Tp, lw = self.gate.text_terms()
        negs = torch.cat([self.T_neg, Tp])
        logw = torch.cat([torch.zeros(self.T_neg.shape[0], dtype=lw.dtype, device=lw.device), lw])
        return lse_w(X @ self.T_id.T, c.tau) - lse_w(X @ negs.T, c.tau, logw)

    @torch.no_grad()
    def step(self, feats: torch.Tensor):
        c = self.cfg
        X = l2n(torch.as_tensor(feats).to(device=self.device, dtype=c.torch_dtype))
        lt = self.text_score(X)
        li = image_score(X, self.idm, self.gate, c) if c.use_image else None
        s = fuse(lt, li, c)
        pred = (X @ self.T_id.T).argmax(1)
        lab, adm = admit_mask(X, self.T_id, c)
        if self.gate is not None:
            vote = self.votes.vote(lt.double().cpu().numpy()) if c.vote else None
            caught_by = self.gate.update(X, self.T_id, ood_vote=vote)
            adm = adm & ~self.gate.exclusion(caught_by, c.id_exclude)
        self.idm.add(X, lab, adm)
        return _StepOutput(pred=pred, score=s.float(), info={})

    def run(self, X: torch.Tensor, batch: int = 256) -> np.ndarray:
        self.reset()
        out = np.zeros(len(X))
        for s in range(0, len(X), batch):
            out[s : s + batch] = self.step(X[s : s + batch]).score.cpu().numpy()
        return out


@dataclass
class _StepOutput:
    pred: torch.Tensor
    score: torch.Tensor
    info: Dict = field(default_factory=dict)
