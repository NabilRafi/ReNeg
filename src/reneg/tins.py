"""TINS on cached CLIP features, and the pieces ReNeg needs to sit on top of it.

TINS: "Test-time ID-prototype-separated Negative Semantics Learning for OOD Detection" (arXiv 2605.10756, May 2026).
This is a re-implementation that follows the official evaluation code (github.com/zxk1212/tins, eval_tins_w_init.py)
step by step, on the oodlab feature cache instead of image folders:

1. ID prototypes mu_c: mean of a few labelled ID images per class (TINS: 16 training images; its Fig. 4b shows the
   gain plateaus after 4). base_sim_c = cos(t_c, mu_c) with t_c the ID class text.
2. Static negatives: WordNet nouns ("The nice {}.") and adjectives ("This is a {} photo."), kept only if, for every
   class, cos(word, mu_c) < base_sim_c; the 2,000 with the largest mean (base_sim_c - cos(word, mu_c)) are used,
   split between adjectives and nouns in proportion to the survivors.
3. Score (group-wise aggregation): negatives (static + dynamic bank) are permuted with a fixed permutation and split
   into G groups; S = mean_g sum_c e^{s v.t_c} / (sum_c e^{s v.t_c} + C * mean_{t in g} e^{s v.t}), s = 100.
4. Test-time inversion: images with S < 0.3 are candidates. For each, a pseudo-token z in "a photo of a <z>." starts
   from the token of the static negative word that minimises 1 - cos(t, v) + lambda * mean_c(1 + cos(t, mu_c)) and is
   optimised for 30 AdamW steps (lr 2e-2, weight decay 1e-2) on the same objective (lambda 0.3). The inverted text t
   is kept only if cos(t, mu_c) < base_sim_c for every class; kept ones enter the dynamic bank (2,000, ranked by
   mean(base_sim_c - cos(t, mu_c)); optional overflow buffer), and the batch is scored again with the new bank.

Everything runs on the device of the CLIP model; the text encoder is used with frozen weights (gradients flow only to
the pseudo-tokens).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .transport import l2n

NOUN_PROMPT = "The nice {}."
ADJ_PROMPT = "This is a {} photo."
INV_TEMPLATE = "a photo of a {}."
INV_TEMPLATE_NO_PERIOD = "a photo of a {}"
_LOCAL_WORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "wordnet_txt")


def _default_word_dir() -> str:
    """NegLabel's WordNet word lists (noun.*.txt, adj.*.txt), the same files TINS uses. oodlab ships them
    (oodlab.CORPUS_DIR, Apache-2.0); a local copy under reneg/resources/wordnet_txt is used if present."""
    if os.path.isdir(_LOCAL_WORDS):
        return _LOCAL_WORDS
    try:
        import oodlab

        return oodlab.CORPUS_DIR
    except Exception:                                           # noqa: BLE001
        return _LOCAL_WORDS


WORD_DIR = _default_word_dir()

try:                                                            # the real oodlab (Colab / Kaggle / local copy)
    from oodlab.methods.base import StepOutput, StreamMethod
except Exception:                                               # noqa: BLE001
    StepOutput, StreamMethod = None, object


@dataclass
class TINSConfig:
    ood_number: int = 2000            # static negatives
    extra_text_length: int = 2000     # dynamic bank size
    group_num: int = 5
    ood_threshold: float = 0.3        # candidates: S < threshold
    inversion_steps: int = 30
    inversion_lr: float = 2e-2
    inversion_weight_decay: float = 1e-2
    reg_lambda: float = 0.3
    random_permute: bool = True
    use_buffer: bool = False          # official default off (Table 4: 6.68 without, 6.72 with)
    bank_buffer_size: int = 2000
    text_batch_size: int = 1000
    max_candidates: int = 512         # inversion runs in chunks of this many candidates (memory)
    record: bool = True

    def with_(self, **kw) -> "TINSConfig":
        return replace(self, **kw)


# ---------------------------------------------------------------------------------------------------- text side
def tokenize(texts: Sequence[str]) -> torch.Tensor:
    from oodlab.clipwrap import import_clip
    return import_clip().tokenize(list(texts), truncate=True)


@torch.no_grad()
def encode_texts(model, texts: Sequence[str], batch_size: int = 1000) -> torch.Tensor:
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(texts), batch_size):
        f = model.encode_text(tokenize(texts[s:s + batch_size]).to(dev)).float()
        out.append(l2n(f).cpu())
    return torch.cat(out) if out else torch.zeros(0, model.text_projection.shape[1])


def collect_negative_words(word_dir: str = WORD_DIR, positive_labels: Sequence[str] = ()) -> Tuple[List[str], List[str]]:
    """Nouns and adjectives from the WordNet word files (noun.*.txt, adj.*.txt), minus the ID class names;
    a word that is both keeps the noun prompt (as the official code)."""
    nouns, adjs = [], []
    for fn in sorted(os.listdir(word_dir)):
        kind = fn.split(".")[0]
        if kind not in ("noun", "adj") or not fn.endswith(".txt"):
            continue
        words = [w.strip() for w in open(os.path.join(word_dir, fn)) if w.strip()]
        (nouns if kind == "noun" else adjs).extend(words)
    pos = set(positive_labels)
    nouns = [w for w in dict.fromkeys(nouns) if w not in pos]
    noun_set = set(nouns)
    adjs = [w for w in dict.fromkeys(adjs) if w not in pos and w not in noun_set]
    return nouns, adjs


def separation(cand: torch.Tensor, prototypes: torch.Tensor, base_sim: torch.Tensor):
    """(keep mask, score): keep if cos(cand, mu_c) < base_sim_c for every class; score = mean(base_sim - cos)."""
    sim = cand @ prototypes.T
    return torch.all(sim < base_sim[None, :], dim=1), (base_sim[None, :] - sim).mean(1)


def build_static_negatives(model, pos_feats: torch.Tensor, prototypes: torch.Tensor, n: int = 2000,
                           positive_labels: Sequence[str] = (), word_dir: str = WORD_DIR, log=print):
    nouns, adjs = collect_negative_words(word_dir, positive_labels)
    base_sim = (pos_feats.cpu() * prototypes.cpu()).sum(1)
    P = prototypes.cpu()
    fa = encode_texts(model, [ADJ_PROMPT.format(w) for w in adjs])
    ma, sa = separation(fa, P, base_sim)
    fn = encode_texts(model, [NOUN_PROMPT.format(w) for w in nouns])
    mn, sn = separation(fn, P, base_sim)
    adjs_k = [w for w, k in zip(adjs, ma.tolist()) if k]
    nouns_k = [w for w, k in zip(nouns, mn.tolist()) if k]
    tot = len(adjs_k) + len(nouns_k)
    if tot == 0:
        raise RuntimeError("no WordNet word passes the ID-prototype separation test")
    n_adj = int(n * len(adjs_k) / tot)
    n_noun = n - n_adj
    if adjs_k:
        n_adj = max(1, n_adj)
    if nouns_k:
        n_noun = max(1, n_noun)

    def top(words, feats, scores, k):
        k = min(k, len(words))
        idx = torch.topk(scores, k).indices.tolist() if k else []
        return [words[i] for i in idx], feats[idx] if k else feats[:0]

    wa, Fa = top(adjs_k, fa[ma], sa[ma], n_adj)
    wn, Fn = top(nouns_k, fn[mn], sn[mn], n_noun)
    log(f"static negatives: {len(nouns)} nouns / {len(adjs)} adjectives -> {len(nouns_k)} / {len(adjs_k)} pass the "
        f"separation test -> {len(wn)} nouns + {len(wa)} adjectives")
    return torch.cat([Fa, Fn]), wa + wn


@torch.no_grad()
def build_init_candidates(model, words: Sequence[str], prototypes: torch.Tensor, batch_size: int = 1000) -> Dict:
    """For every static negative word: the prompt "a photo of a <word>.", the position of its last content token,
    that token's embedding (initial pseudo-token), the prompt's text feature and its regulariser value."""
    dev = next(model.parameters()).device
    tok = tokenize([INV_TEMPLATE.format(w) for w in words])
    tok_np = tokenize([INV_TEMPLATE_NO_PERIOD.format(w) for w in words])
    pos = (tok_np != 0).sum(1) - 2                               # last content token (before <eot>)
    emb = model.token_embedding(tok.to(dev))
    init = emb[torch.arange(len(words), device=dev), pos.to(dev)].float().cpu()
    feats = encode_texts(model, [INV_TEMPLATE.format(w) for w in words], batch_size)
    reg = (1 + feats @ prototypes.cpu().T).mean(1)
    return {"tok": tok, "pos": pos, "init": init, "feats": feats, "reg": reg}


def encode_with_pseudo_tokens(model, tok: torch.Tensor, pseudo: torch.Tensor, pos: torch.Tensor,
                              trim: bool = True) -> torch.Tensor:
    """CLIP's text encoder with the token at ``pos`` replaced by ``pseudo`` (differentiable in ``pseudo``).

    trim=True drops the padding after the last <eot> of the batch. CLIP's text transformer is causal (position i only
    attends to positions <= i) and the feature is read at <eot>, so the result is the same as with all 77 positions
    (up to float rounding), about 7x cheaper for "a photo of a <z>." (TINS's 30 gradient steps per image dominate
    its cost). trim=False is the official code's computation."""
    rows = torch.arange(tok.shape[0], device=tok.device)
    eot = tok.argmax(dim=-1)
    if not trim:
        x = model.token_embedding(tok).type(model.dtype).clone()
        x[rows, pos] = pseudo.to(x.dtype)
        x = x + model.positional_embedding.type(model.dtype)
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x).type(model.dtype)
        return x[rows, eot] @ model.text_projection
    L = int(eot.max()) + 1
    tok = tok[:, :L]
    x = model.token_embedding(tok).type(model.dtype).clone()
    x[rows, pos] = pseudo.to(x.dtype)
    x = (x + model.positional_embedding[:L].type(model.dtype)).permute(1, 0, 2)
    mask = torch.full((L, L), float("-inf"), device=x.device).triu_(1).to(x.dtype)   # CLIP's causal mask, cut to L
    for blk in model.transformer.resblocks:
        h = blk.ln_1(x)
        x = x + blk.attn(h, h, h, need_weights=False, attn_mask=mask)[0]
        x = x + blk.mlp(blk.ln_2(x))
    x = model.ln_final(x.permute(1, 0, 2)).type(model.dtype)
    return x[rows, eot] @ model.text_projection


def grouped_score(v: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, logit_scale: float, groups: int,
                  permute: bool) -> torch.Tensor:
    """TINS's group-wise aggregation score (higher = more ID)."""
    pl = logit_scale * (v @ pos.T)
    nl = logit_scale * (v @ neg.T)
    drop = nl.shape[1] % groups
    if drop:
        nl = nl[:, :-drop]
    if permute:
        g = torch.Generator(device="cpu").manual_seed(0)
        nl = nl[:, torch.randperm(nl.shape[1], generator=g).to(nl.device)]
    nl = nl.reshape(v.shape[0], groups, -1)
    lp = torch.logsumexp(pl, -1)
    ln = torch.logsumexp(nl, -1) - np.log(nl.shape[-1]) + np.log(pl.shape[1])
    return torch.exp(lp[:, None] - torch.logaddexp(lp[:, None], ln)).mean(1)


class TINSFeat(StreamMethod):
    """TINS as an oodlab stream method (one batch of image features at a time)."""
    name = "tins"

    def __init__(self, model, pos_feats: torch.Tensor, prototypes: torch.Tensor, neg_feats: torch.Tensor,
                 init_cands: Dict, cfg: Optional[TINSConfig] = None):
        self.model = model
        self.cfg = cfg or TINSConfig()
        self.device = next(model.parameters()).device
        self.pos = l2n(pos_feats.float().to(self.device))
        self.proto = l2n(prototypes.float().to(self.device))
        self.base_sim = (self.pos * self.proto).sum(1)
        self.fixed_neg = l2n(neg_feats.float().to(self.device))
        self.cands = {k: (v.to(self.device) if k in ("feats", "reg") else v) for k, v in init_cands.items()}
        self.logit_scale = float(model.logit_scale.exp().detach().float().cpu())
        self.C = self.pos.shape[0]
        self.reset()

    def reset(self) -> None:
        D = self.pos.shape[1]
        self.bank = torch.zeros(0, D, dtype=torch.float16)
        self.bank_s = torch.zeros(0)
        self.buf = torch.zeros(0, D, dtype=torch.float16)
        self.buf_s = torch.zeros(0)
        self.n_inv, self.t_inv = 0, 0.0              # candidates inverted and seconds spent on it (this stream)

    def negatives(self) -> torch.Tensor:
        if self.bank.shape[0] == 0:
            return self.fixed_neg
        return torch.cat([self.fixed_neg, self.bank[: self.cfg.extra_text_length].float().to(self.device)])

    def score(self, v: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return grouped_score(v, self.pos, self.negatives(), self.logit_scale, c.group_num, c.random_permute)

    def invert(self, v: torch.Tensor) -> torch.Tensor:
        """Inverted (normalised) text features for the candidate images v (n, D)."""
        c = self.cfg
        obj = 1.0 - v @ self.cands["feats"].T + c.reg_lambda * self.cands["reg"][None, :]
        best = obj.argmin(1).cpu()
        tok = self.cands["tok"][best].to(self.device)
        pos = self.cands["pos"][best].to(self.device)
        z = torch.nn.Parameter(self.cands["init"][best].to(self.device).clone())
        opt = torch.optim.AdamW([z], lr=c.inversion_lr, weight_decay=c.inversion_weight_decay)
        with torch.enable_grad():
            for _ in range(c.inversion_steps):
                opt.zero_grad(set_to_none=True)
                t = l2n(encode_with_pseudo_tokens(self.model, tok, z, pos).float())
                loss = (1.0 - (t * v).sum(1)).mean() + c.reg_lambda * (1 + t @ self.proto.T).mean()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return l2n(encode_with_pseudo_tokens(self.model, tok, z, pos).float())

    def _invert_chunked(self, v: torch.Tensor) -> torch.Tensor:
        """invert() in chunks of cfg.max_candidates; on a CUDA out-of-memory error the chunk size is halved for the
        rest of the run (the result does not depend on the chunk size: every candidate has its own pseudo-token)."""
        while True:
            k = max(1, int(self.cfg.max_candidates))
            try:
                return torch.cat([self.invert(v[s:s + k]) for s in range(0, len(v), k)])
            except torch.cuda.OutOfMemoryError:
                if k == 1:
                    raise
                torch.cuda.empty_cache()
                self.cfg = self.cfg.with_(max_candidates=k // 2)
                print(f"TINS: GPU memory full, inverting {k // 2} candidates at a time from now on", flush=True)

    def _update_bank(self, feats: torch.Tensor, scores: torch.Tensor) -> None:
        c = self.cfg
        allf = torch.cat([self.bank, feats.half().cpu()])
        alls = torch.cat([self.bank_s, scores.cpu()])
        order = torch.argsort(alls, descending=True)
        allf, alls = allf[order], alls[order]
        keep = c.extra_text_length
        self.bank, self.bank_s = allf[:keep], alls[:keep]
        if c.use_buffer and allf.shape[0] > keep:
            self.buf = torch.cat([self.buf, allf[keep:]])
            self.buf_s = torch.cat([self.buf_s, alls[keep:]])
        if c.use_buffer and c.bank_buffer_size > 0 and self.buf.shape[0] >= c.bank_buffer_size:
            half = self.buf.shape[0] // 2
            o = torch.argsort(self.buf_s, descending=True)[:half]
            pool_f = torch.cat([self.bank, self.buf[o]])
            pool_s = torch.cat([self.bank_s, self.buf_s[o]])
            pick = torch.randperm(pool_f.shape[0])[: min(c.extra_text_length, pool_f.shape[0])]
            self.bank, self.bank_s = pool_f[pick], pool_s[pick]
            self.buf = torch.zeros(0, self.bank.shape[1], dtype=torch.float16)
            self.buf_s = torch.zeros(0)

    def step(self, feats: torch.Tensor):
        c = self.cfg
        v = l2n(feats.to(self.device).float())
        with torch.no_grad():
            s0 = self.score(v)
        cand = s0 < c.ood_threshold
        kept_full = torch.zeros_like(cand)
        s = s0
        if cand.any():
            idx = torch.nonzero(cand).flatten()
            t0 = time.perf_counter()
            t = self._invert_chunked(v[idx])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.n_inv += int(len(idx))
            self.t_inv += time.perf_counter() - t0
            keep, sep = separation(t, self.proto, self.base_sim)
            if keep.any():
                self._update_bank(t[keep], sep[keep])
                kept_full[idx[keep]] = True
                with torch.no_grad():
                    s = self.score(v)                                 # the batch again, with the new bank
        with torch.no_grad():
            pred = (v @ self.pos.T).argmax(1)
        info = {"pre": s0.float().cpu(), "cand": cand.cpu(), "kept": kept_full.cpu(),
                "bank": int(self.bank.shape[0])} if c.record else {}
        return StepOutput(pred=pred, score=s.float(), info=info)
