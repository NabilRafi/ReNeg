"""Seam 1: move label embeddings from CLIP's text space into CLIP's image space.

Every map here is fitted from *ID classes only*: for each ID class c we know its name's
text embedding t_c and (from confident test images) the direction of its image centroid v_c.
A map that turns t_c into v_c for classes it was fitted on is then applied to the names of
negative labels, which have no images yet.

The ladder (each rung contains the previous one as a special case)::

    R0  identity            T(t) = t                                    (what TANL does)
    R1  gap shift           T(t) = n(t - m_T + m_V)                      one vector
    R2  ridge map           T(t) = n(m_V + W (t - m_T))                  W pulled toward I
    B1  Procrustes          T(t) = n(m_V + Q (t - m_T))                  Q orthogonal, pulled toward I
    B2  kernel ridge        T(t) = n(t - m_T + m_V + sum_c a_c(t) r_c)   non-linear version of R2

``n(.)`` is L2 normalisation. m_T, m_V are weighted means of the fitted classes' text and
image directions, r_c = (v_c - m_V) - (t_c - m_T) is class c's leftover shift after the gap shift.

R2 (the method in the spec)::

    W* = argmin_W  sum_c w_c ||(v_c - m_V) - W (t_c - m_T)||^2 + lam ||W - I||_F^2
       = I + R^T Om X (X^T Om X + lam I)^-1          rows of X: t_c - m_T, rows of R: r_c, Om = diag(w)

As lam -> infinity, W -> I and R2 becomes R1. The same prediction can be written as
``gap shift + sum_c a_c(t) r_c`` (kernel view), which is exactly B2 with a linear kernel.

All fitting is done in float64 on the CPU (D <= 1024, at most 1,000 classes: milliseconds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

KINDS = ("R0", "R1", "R2", "B1", "B2")


def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


# ---------------------------------------------------------------------------
# per-class statistics from (pseudo-)labelled ID images
# ---------------------------------------------------------------------------


@dataclass
class ClassStats:
    """Running sums of L2-normalised image features per ID class. No images are stored."""

    sums: torch.Tensor     # (C, D) float64
    counts: torch.Tensor   # (C,) float64

    @classmethod
    def empty(cls, n_classes: int, dim: int) -> "ClassStats":
        return cls(torch.zeros(n_classes, dim, dtype=torch.float64), torch.zeros(n_classes, dtype=torch.float64))

    @property
    def n_classes(self) -> int:
        return int(self.sums.shape[0])

    @property
    def dim(self) -> int:
        return int(self.sums.shape[1])

    def add(self, feats: torch.Tensor, classes, weights=None) -> "ClassStats":
        f = torch.as_tensor(feats).detach().to("cpu", torch.float64)
        f = l2n(f)
        c = torch.as_tensor(np.asarray(classes), dtype=torch.long).cpu()
        if len(c) == 0:
            return self
        w = torch.ones(len(c), dtype=torch.float64) if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        self.sums.index_add_(0, c, f * w[:, None])
        self.counts.index_add_(0, c, w)
        return self

    def copy(self) -> "ClassStats":
        return ClassStats(self.sums.clone(), self.counts.clone())

    def means(self) -> torch.Tensor:
        """Raw class means of unit vectors (norm <= 1; the norm measures how tight the class is)."""
        return self.sums / self.counts.clamp_min(1e-12)[:, None]

    def directions(self) -> torch.Tensor:
        """Unit-length class centroids (rows of zeros stay zero)."""
        m = self.means()
        n = m.norm(dim=1, keepdim=True)
        return torch.where(n > 0, m / n.clamp_min(1e-12), m)

    def eligible(self, n_min: float) -> torch.Tensor:
        return torch.nonzero(self.counts >= n_min).flatten()

    def spread(self, idx: Optional[torch.Tensor] = None) -> float:
        """Mean squared distance of one image to its class mean: E||v - vbar||^2 = 1 - ||vbar||^2."""
        m = self.means()
        if idx is not None:
            m = m[idx]
        return float((1.0 - (m * m).sum(1)).clamp_min(0).mean())


@torch.no_grad()
def zero_shot(feats: torch.Tensor, id_text: torch.Tensor, logit_scale: float = 100.0, chunk: int = 8192,
              device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """CLIP zero-shot prediction over the ID classes: (top-1 class, its softmax probability)."""
    T = id_text.to(device, torch.float32)
    preds, probs = [], []
    for s in range(0, feats.shape[0], chunk):
        v = l2n(feats[s : s + chunk].to(device, torch.float32))
        p = torch.softmax(logit_scale * v @ T.t(), dim=1)
        pr, ix = p.max(dim=1)
        preds.append(ix.cpu())
        probs.append(pr.cpu())
    if not preds:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    return torch.cat(preds).numpy().astype(np.int64), torch.cat(probs).numpy().astype(np.float32)


def pseudo_label_stats(feats: torch.Tensor, id_text: torch.Tensor, logit_scale: float, p_min: float,
                       true_labels: Optional[np.ndarray] = None, device: str = "cpu") -> Tuple[ClassStats, Dict]:
    """Class statistics from images whose zero-shot top-1 probability is at least ``p_min``.

    This is exactly what the method can do at test time (no labels). ``true_labels`` is only
    used to *report* how clean the pseudo-labels are.
    """
    pred, prob = zero_shot(feats, id_text, logit_scale, device=device)
    keep = prob >= p_min
    st = ClassStats.empty(id_text.shape[0], id_text.shape[1]).add(feats[torch.as_tensor(np.nonzero(keep)[0])], pred[keep])
    info = {"n_images": int(len(pred)), "n_kept": int(keep.sum()), "kept_frac": float(keep.mean()) if len(keep) else 0.0,
            "p_min": float(p_min)}
    if true_labels is not None and len(true_labels) == len(pred):
        tl = np.asarray(true_labels)
        info["zero_shot_acc_all"] = float((pred == tl).mean())
        info["pseudo_label_acc_kept"] = float((pred[keep] == tl[keep]).mean()) if keep.any() else float("nan")
    return st, info


def oracle_stats(feats: torch.Tensor, labels: np.ndarray, n_classes: int) -> ClassStats:
    """Class statistics from the true labels. Analysis only: the method never sees labels."""
    labels = np.asarray(labels)
    m = labels >= 0
    return ClassStats.empty(n_classes, feats.shape[1]).add(feats[torch.as_tensor(np.nonzero(m)[0])], labels[m])


# ---------------------------------------------------------------------------
# the transport maps
# ---------------------------------------------------------------------------


@dataclass
class Transport:
    kind: str
    params: Dict = field(default_factory=dict)
    m_T: Optional[torch.Tensor] = None
    m_V: Optional[torch.Tensor] = None
    W: Optional[torch.Tensor] = None          # (D, D), column convention y = W x   (R2, B1)
    anchors: Optional[torch.Tensor] = None    # (K, D) ID name embeddings           (B2)
    coef: Optional[torch.Tensor] = None       # (K, D) kernel-ridge coefficients   (B2)
    n_fit: int = 0                            # number of ID classes the map was fitted on

    @property
    def name(self) -> str:
        if self.kind in ("R0", "R1"):
            return self.kind
        if self.kind == "B2":
            ker = self.params.get("kernel", "rbf")
            h = self.params.get("h")
            tag = f"lin" if ker == "linear" else f"h={h:g}"
            return f"B2({tag},lam={self.params.get('lam'):g})"
        return f"{self.kind}(lam={self.params.get('lam'):g})"

    @torch.no_grad()
    def raw(self, t: torch.Tensor, chunk: int = 20000) -> torch.Tensor:
        """The map's output before L2 normalisation, float64, CPU.

        ``raw(t) - m_V`` is the predicted deviation of the label's images from the mean image
        direction: the quantity compared in the centred geometry (G1b).
        """
        t = torch.as_tensor(t)
        outs = []
        for s in range(0, t.shape[0], chunk):
            x = l2n(t[s : s + chunk].detach().to("cpu", torch.float64))
            if self.kind == "R0":
                y = x
            elif self.kind == "R1":
                y = x - self.m_T + self.m_V
            elif self.kind in ("R2", "B1"):
                y = self.m_V + (x - self.m_T) @ self.W.t()
            elif self.kind == "B2":
                y = x - self.m_T + self.m_V + _kernel(x, self.anchors, self.params, self.m_T) @ self.coef
            else:
                raise ValueError(self.kind)
            outs.append(y)
        return torch.cat(outs) if outs else torch.empty(0, t.shape[1], dtype=torch.float64)

    @torch.no_grad()
    def apply(self, t: torch.Tensor, chunk: int = 20000) -> torch.Tensor:
        """Transport label embeddings (N, D) into image space. Returns unit vectors, float32, CPU."""
        return l2n(self.raw(t, chunk)).float()

    def summary(self) -> Dict:
        d = {"transport": self.name, "kind": self.kind, "n_fit_classes": self.n_fit, **self.params}
        if self.m_T is not None and self.m_V is not None:
            d["gap_norm"] = float((self.m_V - self.m_T).norm())
        if self.W is not None:
            d["W_minus_I_fro"] = float((self.W - torch.eye(self.W.shape[0], dtype=self.W.dtype)).norm())
        return d


def _kernel(x: torch.Tensor, anchors: torch.Tensor, params: Dict, m_T: torch.Tensor) -> torch.Tensor:
    if params.get("kernel", "rbf") == "linear":
        return (x - m_T) @ (anchors - m_T).t()
    h = float(params["h"])
    return torch.exp((x @ anchors.t() - 1.0) / h)   # = exp(-||x - a||^2 / 2h) for unit vectors


def _prepare(id_text: torch.Tensor, stats: ClassStats, classes, n_min: float, n_cap: float):
    elig = stats.eligible(n_min)
    if classes is not None:
        keep = torch.as_tensor(np.asarray(classes), dtype=torch.long)
        mask = torch.zeros(stats.n_classes, dtype=torch.bool)
        mask[keep] = True
        elig = elig[mask[elig]]
    T = l2n(torch.as_tensor(id_text).detach().to("cpu", torch.float64))[elig]
    V = stats.directions()[elig]
    w = stats.counts[elig].clamp(max=float(n_cap))
    return elig, T, V, w


def fit_transport(kind: str, id_text: torch.Tensor, stats: ClassStats, classes=None, n_min: float = 3,
                  n_cap: float = 20, lam: float = 1.0, h: float = 0.1, kernel: str = "rbf",
                  lam_is_relative: bool = True, min_classes: int = 2) -> Transport:
    """Fit one transport map on the eligible ID classes (optionally restricted to ``classes``).

    ``lam`` is relative by default: for R2/B1 it is multiplied by tr(X^T Om X)/D (the average
    eigenvalue of the weighted text spread), for B2 by the mean kernel diagonal (1 for RBF). So
    lam = 1 means "the pull toward the identity is as strong as an average direction of the data".
    """
    if kind not in KINDS:
        raise ValueError(f"unknown transport {kind}; choose from {KINDS}")
    if kind == "R0":
        return Transport("R0")
    elig, T, V, w = _prepare(id_text, stats, classes, n_min, n_cap)
    if len(elig) < min_classes:
        return Transport("R0", params={"fallback": f"only {len(elig)} eligible classes"})
    wsum = w.sum()
    m_T = (w[:, None] * T).sum(0) / wsum
    m_V = (w[:, None] * V).sum(0) / wsum
    if kind == "R1":
        return Transport("R1", m_T=m_T, m_V=m_V, n_fit=len(elig))
    X, Y = T - m_T, V - m_V
    R = Y - X
    D = X.shape[1]
    I = torch.eye(D, dtype=torch.float64)
    if kind in ("R2", "B1"):
        scale = float((w[:, None] * X * X).sum() / D) if lam_is_relative else 1.0
        lam_abs = lam * scale
        if kind == "R2":
            A = X.t() @ (w[:, None] * X) + lam_abs * I
            B = R.t() @ (w[:, None] * X)
            Wm = I + torch.linalg.solve(A, B.t()).t()               # I + B A^-1  (A symmetric)
        else:
            M = Y.t() @ (w[:, None] * X) + lam_abs * I              # sum_c w_c y_c x_c^T + lam I
            U, _, Vh = torch.linalg.svd(M)
            Wm = U @ Vh                                              # argmax tr(Q^T M) over rotations
        return Transport(kind, params={"lam": float(lam), "lam_abs": float(lam_abs)}, m_T=m_T, m_V=m_V, W=Wm,
                         n_fit=len(elig))
    # B2: weighted kernel ridge on the leftover shifts r_c
    params = {"lam": float(lam), "h": float(h), "kernel": kernel}
    K = _kernel(T, T, params, m_T)
    scale = float(K.diagonal().mean()) if lam_is_relative else 1.0
    lam_abs = lam * scale
    coef = torch.linalg.solve(K + lam_abs * torch.diag(1.0 / w), R)     # (K + lam Om^-1)^-1 R
    params["lam_abs"] = float(lam_abs)
    return Transport("B2", params=params, m_T=m_T, m_V=m_V, anchors=T, coef=coef, n_fit=len(elig))


# ---------------------------------------------------------------------------
# candidates and the "does not hamper CLIP" diagnostic
# ---------------------------------------------------------------------------

DEFAULT_LAMS = (1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_HS = (0.05, 0.1, 0.2)


def candidate_grid(lams: Sequence[float] = DEFAULT_LAMS, hs: Sequence[float] = DEFAULT_HS,
                   kinds: Iterable[str] = KINDS) -> list:
    """Every (kind, hyper-parameter) combination the self-check compares."""
    kinds = list(kinds)
    out = []
    if "R0" in kinds:
        out.append({"kind": "R0"})
    if "R1" in kinds:
        out.append({"kind": "R1"})
    for k in ("R2", "B1"):
        if k in kinds:
            out += [{"kind": k, "lam": float(l)} for l in lams]
    if "B2" in kinds:
        out += [{"kind": "B2", "lam": float(l), "h": float(h)} for h in hs for l in lams]
    return out


def fit_candidate(cand: Dict, id_text, stats, classes=None, n_min=3, n_cap=20) -> Transport:
    return fit_transport(cand["kind"], id_text, stats, classes=classes, n_min=n_min, n_cap=n_cap,
                         lam=cand.get("lam", 1.0), h=cand.get("h", 0.1), kernel=cand.get("kernel", "rbf"))


def candidate_name(cand: Dict) -> str:
    k = cand["kind"]
    if k in ("R0", "R1"):
        return k
    if k == "B2":
        if cand.get("kernel") == "linear":
            return f"B2(lin,lam={cand['lam']:g})"
        return f"B2(h={cand['h']:g},lam={cand['lam']:g})"
    return f"{k}(lam={cand['lam']:g})"


@torch.no_grad()
def change_profile(tr: Transport, id_text, stats: ClassStats, n_min=3, n_cap=20) -> Dict:
    """How much W changes each eigen-direction of the ID text spread (R2/B1 only).

    Returns the largest relative change in the low-variance half of the directions and the mean
    change in the high-variance half. For R2 the change along direction i is bounded by
    ||R Om^1/2|| sqrt(s_i) / (s_i + lam): directions the ID data do not cover stay unchanged.
    """
    if tr.W is None:
        return {}
    _, T, V, w = _prepare(id_text, stats, None, n_min, n_cap)
    X = T - tr.m_T
    S = X.t() @ (w[:, None] * X)
    evals, evecs = torch.linalg.eigh(S)
    D = S.shape[0]
    change = ((tr.W - torch.eye(D, dtype=tr.W.dtype)) @ evecs).norm(dim=0)
    order = torch.argsort(evals)
    low, high = order[: D // 2], order[D // 2 :]
    out = {"low_var_change_max": float(change[low].max()), "high_var_change_mean": float(change[high].mean())}
    if tr.kind == "R2":
        R = (V - tr.m_V) - X
        bound = float(torch.linalg.matrix_norm(R.t() * w.sqrt()[None, :], ord=2)) * evals.clamp_min(0).sqrt() / (
            evals.clamp_min(0) + tr.params["lam_abs"])
        out["bound_holds"] = bool((change <= bound + 1e-8).all())
    return out
