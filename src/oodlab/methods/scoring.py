"""Scoring primitives shared by the methods, with exact equivalents of the official code.

All functions take *logits* already multiplied by CLIP's logit scale (100 for
OpenAI CLIP), laid out as ``[ID classes | negative labels]``.
"""
from __future__ import annotations

from typing import List, Optional

import torch

# ---------------------------------------------------------------------------
# logits, with optional emulation of the official fp16 arithmetic
# ---------------------------------------------------------------------------


def scaled_logits(img: torch.Tensor, txt_T: torch.Tensor, scale: float, emulate_fp16: bool = False) -> torch.Tensor:
    """``(scale * img) @ txt_T`` as float32.

    ``img``: (B, D) L2-normalised image features. ``txt_T``: (D, K) text features, i.e. the
    *column* layout the official networks store (``net.text_features`` is D x K). Keeping the
    same layout and operation order as the official expression
    ``logit_scale * image_features @ text_features`` means the same BLAS kernels are used,
    which is what makes the fp16 emulation bit-exact on GPU as well as on CPU.

    With ``emulate_fp16=True`` the product is computed natively in half precision (as CLIP
    runs on the GPU) and returned as float32 so downstream code can choose its precision.
    """
    if emulate_fp16:
        return scaled_logits_half(img, txt_T, scale).float()
    s = torch.tensor(scale, dtype=torch.float32, device=img.device)
    return (s * img.float()) @ txt_T.float()


def scaled_logits_half(img: torch.Tensor, txt_T: torch.Tensor, scale: float) -> torch.Tensor:
    """Official expression ``logit_scale * image_features @ text_features`` in fp16 (txt_T is D x K)."""
    s = torch.tensor(scale, dtype=torch.float32, device=img.device)  # official: 0-dim fp32 tensor
    return (s * img.to(torch.float16)) @ txt_T.to(torch.float16)


def to_columns(text: torch.Tensor) -> torch.Tensor:
    """(K, D) row-per-label features -> contiguous (D, K), the official layout."""
    return text.t().contiguous()


# ---------------------------------------------------------------------------
# NegLabel-style scores
# ---------------------------------------------------------------------------


def id_mass(logits: torch.Tensor, n_id: int) -> torch.Tensor:
    """Softmax mass on the ID columns: sum_{i<C} softmax(logits)_i  (NegLabel's S_nl)."""
    a = torch.logsumexp(logits[:, :n_id], dim=1)
    b = torch.logsumexp(logits, dim=1)
    return torch.exp(a - b)


def activation_aware_score(logits: torch.Tensor, n_id: int, step: int = 1) -> torch.Tensor:
    """TANL's activation-aware score (paper Eq. 15), closed form.

    ``logits``: (B, C+M) with the M negatives sorted by *descending* activation.

    Official loop (``activation_aware_score`` in the TANL postprocessor):
        for i in range(C, C+M, step):
            conf += softmax(logits[:, :i+step])[:, :C].sum(1)
        conf /= number_of_terms
    Each term equals ``sigmoid(logsumexp(ID) - logsumexp(first m negatives))`` with
    m = min(i - C + step, M), which we evaluate with one ``logcumsumexp``.
    ``step == 0`` gives the plain NegLabel score over all M selected negatives.
    """
    n_neg = logits.shape[1] - n_id
    a = torch.logsumexp(logits[:, :n_id], dim=1, keepdim=True)  # (B,1)
    if n_neg <= 0:
        return torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
    if step == 0:
        s = torch.logsumexp(logits[:, n_id:], dim=1, keepdim=True)
        return torch.sigmoid(a - s).squeeze(1)
    cum = torch.logcumsumexp(logits[:, n_id:], dim=1)  # (B, M): logsumexp of first m negatives
    step = int(step)
    ms = [min(j + step, n_neg) for j in range(0, n_neg, step)]
    idx = torch.tensor([m - 1 for m in ms], device=logits.device)
    return torch.sigmoid(a - cum.index_select(1, idx)).mean(dim=1)


def activation_aware_score_fp16_loop(logits_half: torch.Tensor, n_id: int, step: int = 1) -> torch.Tensor:
    """The official loop, op-for-op, on fp16 logits (slow; for bit-exact comparisons)."""
    n_neg = logits_half.shape[1] - n_id
    if step == 0:
        return logits_half.softmax(dim=-1)[:, :n_id].sum(dim=-1)
    sums = []
    for i in range(n_id, n_id + n_neg, int(step)):
        sums.append(logits_half[:, : i + int(step)].softmax(dim=-1)[:, :n_id].sum(dim=-1))
    return torch.stack(sums, dim=-1).mean(dim=-1)


def neglabel_group_permutation(n_neg: int, group_num: int, random_permute: bool = True, seed: int = 0) -> torch.Tensor:
    """The permutation the official NegLabel/AdaNeg code applies before grouping.

    Official: ``torch.manual_seed(0); idx = torch.randperm(n_neg_after_drop)`` on CPU.
    A fresh ``torch.Generator`` seeded identically produces the same permutation.
    """
    drop = n_neg % group_num
    n = n_neg - drop
    if not random_permute:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)


def neglabel_grouped_score(
    logits: torch.Tensor, n_id: int, group_num: int, perm: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """NegLabel's grouping strategy: mean over groups of the ID mass in [ID | group].

    Matches ``OneOodPromptDevelopPostprocessor.postprocess`` and the AdaNeg scorer:
    negatives beyond a multiple of ``group_num`` are dropped (from the end), the rest
    are permuted by ``perm`` and reshaped into ``group_num`` equal groups.
    """
    pos = logits[:, :n_id]
    neg = logits[:, n_id:]
    drop = neg.shape[1] % group_num
    if drop > 0:
        neg = neg[:, :-drop]
    if perm is not None:
        neg = neg[:, perm.to(neg.device)]
    B = logits.shape[0]
    neg = neg.reshape(B, group_num, -1)
    a = torch.logsumexp(pos, dim=1, keepdim=True)  # (B,1)
    s = torch.logsumexp(neg, dim=2)  # (B,G)
    return torch.sigmoid(a - s).mean(dim=1)


def neglabel_grouped_score_fp16(
    logits_half: torch.Tensor, n_id: int, group_num: int, perm: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The official grouped score executed op-for-op in fp16 (returns an fp16 tensor)."""
    pos = logits_half[:, :n_id]
    neg = logits_half[:, n_id:]
    drop = neg.shape[1] % group_num
    if drop > 0:
        neg = neg[:, :-drop]
    if perm is not None:
        neg = neg.T[perm.to(neg.device)].T
    neg = neg.reshape(pos.shape[0], group_num, -1).contiguous()
    scores = []
    for i in range(group_num):
        full = torch.cat([pos, neg[:, i, :]], dim=-1).softmax(dim=-1)
        scores.append(full[:, :n_id].sum(dim=-1).unsqueeze(-1))
    return torch.cat(scores, dim=-1).mean(dim=-1)


# ---------------------------------------------------------------------------
# TANL's automatic threshold (Otsu-style), verbatim port
# ---------------------------------------------------------------------------


def _os_variance(os_tensor: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    device = os_tensor.device
    mask = (os_tensor >= th).float()
    n_pixels = os_tensor.numel()
    n_pixels1 = torch.sum(mask)
    weight1 = n_pixels1 / n_pixels
    weight0 = 1 - weight1
    if weight1 == 0 or weight0 == 0:
        return torch.tensor(float("inf"), device=device)
    class1 = os_tensor[mask.bool()]
    class0 = os_tensor[~mask.bool()]
    var0 = torch.var(class0, unbiased=False) if class0.numel() > 0 else torch.tensor(0.0, device=device)
    var1 = torch.var(class1, unbiased=False) if class1.numel() > 0 else torch.tensor(0.0, device=device)
    return weight0 * var0 + weight1 * var1


def find_best_threshold(scores: torch.Tensor) -> float:
    """Port of ``find_best_threshold`` in the official TANL code (ties -> middle candidate)."""
    threshold_range = torch.arange(0, 1, 0.01, device=scores.device)
    criterias = torch.stack([_os_variance(scores, th) for th in threshold_range])
    min_val = torch.min(criterias)
    candidate = torch.where(criterias == min_val)[0]
    if len(candidate) == 0:
        return threshold_range[torch.argmin(criterias)].item()
    return threshold_range[candidate[len(candidate) // 2]].item()


def find_best_threshold_fast(scores: torch.Tensor) -> float:
    """Vectorised version of :func:`find_best_threshold` (same criterion, float64 sums).

    Gives the same threshold except in rare exact ties caused by float rounding.
    Use it when the stream is long and you do not need bit-level fidelity.
    """
    s = scores.double().flatten()
    n = s.numel()
    th = torch.arange(0, 1, 0.01, device=s.device, dtype=torch.float32).double()
    m1 = (s[None, :] >= th[:, None]).double()  # (T, n)
    n1 = m1.sum(1)
    n0 = n - n1
    s1 = m1 @ s
    s0 = s.sum() - s1
    q1 = m1 @ (s * s)
    q0 = (s * s).sum() - q1
    mean1 = torch.where(n1 > 0, s1 / n1.clamp(min=1), torch.zeros_like(s1))
    mean0 = torch.where(n0 > 0, s0 / n0.clamp(min=1), torch.zeros_like(s0))
    var1 = torch.where(n1 > 0, q1 / n1.clamp(min=1) - mean1**2, torch.zeros_like(s1))
    var0 = torch.where(n0 > 0, q0 / n0.clamp(min=1) - mean0**2, torch.zeros_like(s0))
    crit = (n0 / n) * var0 + (n1 / n) * var1
    crit = torch.where((n1 == 0) | (n0 == 0), torch.full_like(crit, float("inf")), crit)
    min_val = crit.min()
    cand = torch.where(crit == min_val)[0]
    return float(th[cand[len(cand) // 2]].item())


def entropy(prob: torch.Tensor) -> torch.Tensor:
    """Official AdaNeg entropy: -(p * log(p + 1e-8)).sum()."""
    return -(prob * torch.log(prob + 1e-8)).sum(dim=-1)


def ranks_to_list(idx: torch.Tensor) -> List[int]:
    return [int(i) for i in idx.tolist()]
