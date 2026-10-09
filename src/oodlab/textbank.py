"""The text side shared by every method: ID label features, the WordNet corpus,
NegMining (NegLabel's selection of 10,000 negatives) and the 15 noise-image features.

Reference: OpenOOD-VLM ``networks/clip_for_wordnet_prepare.py`` (corpus + cosine
matrix, always computed with ViT-B/16) and ``networks/clip_fixed_ood_prompt.py``
(``get_selected_ood_text_list`` + ``get_text_features_neg`` + noise images).

Corpus layout (identical to the official ``text_features_all[:, 1000:]``)::

    [ selected adj (1,646) | selected noun (8,354) | unselected adj | unselected noun ]
      \\________ NegLabel / AdaNeg negatives (10,000) ______/

"selected" = the 10,000 words whose 95th-percentile cosine similarity to the ID
class names is lowest, split between adjectives and nouns in proportion to their
counts. TANL scores every one of the 69,554 words, so its selection does not
depend on this split.

Deliberate deviation: the official code builds the word lists with ``list(set(...))``,
whose order changes from run to run (Python hash randomisation), and sorts with an
unstable sort, so ties in the fp16 mining score are broken arbitrarily. We sort the
words alphabetically first and use a stable sort, which makes the corpus identical
on every run. Only words tied exactly at the 10,000 boundary can differ.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .clipwrap import ADJ_TEMPLATE, TEMPLATES, centered_label_features, encode_prompts, l2n, noise_features

OFFICIAL_N_ADJ = 11453    # adjectives left after the official de-duplication (ImageNet-1K names)
OFFICIAL_N_NOUN = 58101   # nouns left (11,453 + 58,101 = 69,554; see TANL's "70K-word corpus")


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def read_wordnet_corpus(txt_dir: str, id_names: Sequence[str]) -> Tuple[List[str], List[str], Dict]:
    """Adjective and noun lists exactly as ``generate_cossim_idname_wordnet_dedup`` builds them
    (then sorted, see module docstring)."""
    adj, noun = [], []
    files = sorted(os.listdir(txt_dir))
    used = []
    for fn in files:
        path = os.path.join(txt_dir, fn)
        if fn[:3] == "adj" and fn[-3:] == "txt":
            with open(path, "r", encoding="utf-8") as f:
                adj += f.read().split("\n")[:-1]
            used.append(fn)
        elif fn[:4] == "noun" and fn[-3:] == "txt":
            with open(path, "r", encoding="utf-8") as f:
                noun += f.read().split("\n")[:-1]
            used.append(fn)
    if not used:
        raise FileNotFoundError(f"no adj*.txt / noun*.txt files in {txt_dir}")
    stats = {"files": used, "raw_adj": len(adj), "raw_noun": len(noun)}
    adj_s, noun_s = set(adj), set(noun)
    ids = set(id_names)
    adj_s -= ids
    noun_s -= ids
    adj_s -= noun_s  # "when a word is both a noun and an adj, remain the noun only"
    adj_l, noun_l = sorted(adj_s), sorted(noun_s)
    stats.update({"n_adj": len(adj_l), "n_noun": len(noun_l)})
    return adj_l, noun_l, stats


def subsample_corpus(adj: List[str], noun: List[str], limit: int, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Deterministic smaller corpus with the same adj/noun proportion (smoke tests only)."""
    g = torch.Generator().manual_seed(seed)
    n_adj = max(1, round(limit * len(adj) / (len(adj) + len(noun))))
    n_noun = max(1, limit - n_adj)
    ia = torch.randperm(len(adj), generator=g)[:n_adj].sort().values.tolist()
    inn = torch.randperm(len(noun), generator=g)[:n_noun].sort().values.tolist()
    return [adj[i] for i in ia], [noun[i] for i in inn]


def words_digest(words: Sequence[str]) -> str:
    h = hashlib.sha1()
    for w in words:
        h.update(w.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# NegMining
# ---------------------------------------------------------------------------


@torch.no_grad()
def percentile_cosine(word_feats: torch.Tensor, id_feats: torch.Tensor, q: float = 0.95, chunk: int = 4096) -> torch.Tensor:
    """For every word: its ``int(C*q)``-th smallest cosine similarity to the C ID labels
    (official: ``cos.sort(1)[0][:, int(C*0.95)]``), computed in the features' dtype."""
    C = id_feats.shape[0]
    k = int(C * q)
    out = []
    for s in range(0, word_feats.shape[0], chunk):
        cos = word_feats[s : s + chunk] @ id_feats.t()
        out.append(cos.sort(dim=1).values[:, k].float().cpu())
    return torch.cat(out) if out else torch.empty(0)


def split_counts(n_adj: int, n_noun: int, total: int) -> Tuple[int, int]:
    """Official: adj_num = int(total * n_adj / (n_adj + n_noun)); noun_num = total - adj_num."""
    adj_num = int(total * (n_adj / (n_adj + n_noun)))
    return adj_num, total - adj_num


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------


@dataclass
class TextBank:
    backbone: str
    id_names: List[str]
    id_text: torch.Tensor          # (C, D) prompt "The nice {}."   (TANL, NegLabel, AdaNeg)
    id_text_simple: torch.Tensor   # (C, D) prompt "a photo of a {}." (MCM)
    words: List[str]               # corpus in the official layout (see module docstring)
    is_adj: torch.Tensor           # (N,) bool
    corpus_text: torch.Tensor      # (N, D)
    mining_score: torch.Tensor     # (N,) 95th-percentile cosine to ID names (low = far from ID)
    n_selected: int                # corpus_text[:n_selected] are NegLabel's negatives
    n_sel_adj: int
    noise_feats: torch.Tensor      # (15, D) official noise-image features
    meta: Dict = field(default_factory=dict)

    @property
    def neg_text(self) -> torch.Tensor:
        """NegLabel / AdaNeg negatives: the first ``n_selected`` corpus rows."""
        return self.corpus_text[: self.n_selected]

    @property
    def dim(self) -> int:
        return int(self.id_text.shape[1])

    @property
    def logit_scale(self) -> float:
        """``clip_model.logit_scale.exp()`` of the backbone (official code uses this, ~100)."""
        return float(self.meta.get("logit_scale", 100.0))

    def __len__(self) -> int:
        return len(self.words)

    def summary(self) -> Dict:
        return {
            "backbone": self.backbone,
            "n_id": len(self.id_names),
            "n_corpus": len(self.words),
            "n_adj": int(self.is_adj.sum()),
            "n_noun": int((~self.is_adj).sum()),
            "n_selected": self.n_selected,
            "n_sel_adj": self.n_sel_adj,
            "dim": self.dim,
            "dtype": str(self.corpus_text.dtype),
            "logit_scale": self.logit_scale,
            **{k: v for k, v in self.meta.items() if k in ("corpus_digest", "mining_backbone", "noise_seed", "prompt")},
        }

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        from .utils import atomic_torch_save

        atomic_torch_save({
            "format": "oodlab.textbank.v1",
            "backbone": self.backbone,
            "id_names": list(self.id_names),
            "id_text": self.id_text.cpu(),
            "id_text_simple": self.id_text_simple.cpu(),
            "words": list(self.words),
            "is_adj": self.is_adj.cpu(),
            "corpus_text": self.corpus_text.cpu(),
            "mining_score": self.mining_score.cpu(),
            "n_selected": int(self.n_selected),
            "n_sel_adj": int(self.n_sel_adj),
            "noise_feats": self.noise_feats.cpu(),
            "meta": dict(self.meta),
        }, path)

    @classmethod
    def load(cls, path: str) -> "TextBank":
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # very old torch
            d = torch.load(path, map_location="cpu")
        if d.get("format") != "oodlab.textbank.v1":
            raise ValueError(f"{path} is not an oodlab text bank")
        return cls(
            backbone=d["backbone"], id_names=d["id_names"], id_text=d["id_text"], id_text_simple=d["id_text_simple"],
            words=d["words"], is_adj=d["is_adj"], corpus_text=d["corpus_text"], mining_score=d["mining_score"],
            n_selected=d["n_selected"], n_sel_adj=d["n_sel_adj"], noise_feats=d["noise_feats"], meta=d["meta"],
        )


@torch.no_grad()
def build_textbank(
    model,
    backbone: str,
    id_names: Sequence[str],
    txt_dir: str,
    mining_model=None,
    mining_backbone: str = "ViT-B/16",
    total_neg: int = 10000,
    prompt: str = "nice",
    noise_seed: int = 0,
    batch_size: int = 1000,
    corpus_limit: Optional[int] = None,
    logger=None,
) -> TextBank:
    """Compute every text feature the methods need, in the official layout.

    ``mining_model`` must be CLIP ViT-B/16 (the official NegMining always uses it).
    If ``backbone`` is ViT-B/16 you can leave it as None and ``model`` is reused, which
    also lets us reuse the corpus encodings (they use the same prompts).
    """
    t0 = time.time()
    log = logger.info if logger else print
    if mining_model is None:
        if backbone != mining_backbone:
            raise ValueError(f"NegMining uses {mining_backbone}; pass mining_model= for backbone {backbone}")
        mining_model = model
    same = mining_model is model

    adj, noun, stats = read_wordnet_corpus(txt_dir, id_names)
    log(f"[textbank] corpus: {stats['n_adj']} adjectives + {stats['n_noun']} nouns "
        f"(official: {OFFICIAL_N_ADJ} + {OFFICIAL_N_NOUN})")
    if corpus_limit:
        adj, noun = subsample_corpus(adj, noun, corpus_limit)
        log(f"[textbank] SMOKE: corpus subsampled to {len(adj)} + {len(noun)}")
    if total_neg > len(adj) + len(noun):
        raise ValueError(f"total_neg={total_neg} exceeds the corpus size {len(adj) + len(noun)}")

    templates = TEMPLATES[prompt]
    if len(templates) != 1:
        raise ValueError("NegMining reuse assumes a single-template prompt ('nice')")
    noun_prompts = [templates[0].format(w) for w in noun]
    adj_prompts = [ADJ_TEMPLATE.format(w) for w in adj]

    # ---- raw encodings with the mining model (ViT-B/16) --------------------------
    id_raw_m = encode_prompts(mining_model, ["The nice " + n + "." for n in id_names], batch_size)
    adj_raw_m = encode_prompts(mining_model, adj_prompts, batch_size, progress="adjectives", logger=logger)
    noun_raw_m = encode_prompts(mining_model, noun_prompts, batch_size, progress="nouns", logger=logger)
    id_m = l2n(id_raw_m)
    score_adj = percentile_cosine(l2n(adj_raw_m), id_m)
    score_noun = percentile_cosine(l2n(noun_raw_m), id_m)

    adj_num, noun_num = split_counts(len(adj), len(noun), total_neg)
    order_adj = torch.sort(score_adj, stable=True).indices   # ascending: far from ID first
    order_noun = torch.sort(score_noun, stable=True).indices
    log(f"[textbank] NegMining: {adj_num} adjectives + {noun_num} nouns selected (total {total_neg})")

    # ---- corpus features with the scoring backbone --------------------------------
    if same:
        adj_raw, noun_raw = adj_raw_m, noun_raw_m
    else:
        adj_raw = encode_prompts(model, adj_prompts, batch_size, progress="adjectives (backbone)", logger=logger)
        noun_raw = encode_prompts(model, noun_prompts, batch_size, progress="nouns (backbone)", logger=logger)
    adj_f = centered_label_features(adj_raw, 1)
    noun_f = centered_label_features(noun_raw, 1)
    del adj_raw_m, noun_raw_m

    sa, ua = order_adj[:adj_num], order_adj[adj_num:]
    sn, un = order_noun[:noun_num], order_noun[noun_num:]
    dev_a, dev_n = adj_f.device, noun_f.device
    corpus = torch.cat([adj_f[sa.to(dev_a)], noun_f[sn.to(dev_n)], adj_f[ua.to(dev_a)], noun_f[un.to(dev_n)]], dim=0)
    words = [adj[i] for i in sa.tolist()] + [noun[i] for i in sn.tolist()] + \
            [adj[i] for i in ua.tolist()] + [noun[i] for i in un.tolist()]
    is_adj = torch.cat([torch.ones(adj_num, dtype=torch.bool), torch.zeros(noun_num, dtype=torch.bool),
                        torch.ones(len(ua), dtype=torch.bool), torch.zeros(len(un), dtype=torch.bool)])
    mining = torch.cat([score_adj[sa], score_noun[sn], score_adj[ua], score_noun[un]])

    # ---- ID label features --------------------------------------------------------
    id_text = centered_label_features(
        id_raw_m if same else encode_prompts(model, [templates[0].format(n) for n in id_names], batch_size), 1)
    id_text_simple = centered_label_features(
        encode_prompts(model, [TEMPLATES["simple"][0].format(n) for n in id_names], batch_size), 1)

    noise = noise_features(model, seed=noise_seed)
    meta = {
        "logit_scale": float(model.logit_scale.detach().float().exp().item()),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": backbone,
        "mining_backbone": mining_backbone,
        "prompt": prompt,
        "adj_template": ADJ_TEMPLATE,
        "total_neg": total_neg,
        "noise_seed": noise_seed,
        "corpus_digest": words_digest(adj + noun),
        "corpus_limit": corpus_limit,
        "stats": stats,
        "torch": torch.__version__,
        "dtype": str(corpus.dtype),
        "seconds": round(time.time() - t0, 1),
    }
    bank = TextBank(
        backbone=backbone, id_names=list(id_names), id_text=id_text.cpu(), id_text_simple=id_text_simple.cpu(),
        words=words, is_adj=is_adj, corpus_text=corpus.cpu(), mining_score=mining, n_selected=total_neg,
        n_sel_adj=adj_num, noise_feats=noise.cpu(), meta=meta,
    )
    log(f"[textbank] done in {meta['seconds']}s: {len(words)} corpus words, dim {bank.dim}, dtype {corpus.dtype}")
    return bank
