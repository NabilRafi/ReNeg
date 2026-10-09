"""Building blocks of the command-line pipeline (``scripts/evaluate.py``).

Everything here is what the Colab notebooks G3, G3b, G4 and D1 do in their cells, packaged for a local run on a
feature cache built by ``scripts/build_cache.py`` (or oodlab's Kaggle notebook 01):

* the ReNeg configurations (operating points ``balanced``, ``max``, ``safe`` on the TANL base, and the plain
  configuration on NegLabel's negatives used by the ablations);
* the knowledge-graph (KG) pool: CLIP text embeddings of the 14,526 WordNet names, cached next to the features,
  and its ``full`` and ``blind`` variants (blind = without the SSB-hard and NINCO class names);
* the few-shot setting: OpenOOD's ImageNet ID-val split (about 5 labelled images per class) and the Four-OOD
  protocol on the 45,000 ID images that are not those images (``four_ood_45k``);
* a factory that builds every method of the paper from a :class:`RunSpec`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .core import ReNegConfig, ReNegNegLabel
from .packs import export_pool, load_kg_pool
from .transport import l2n

# ReNeg on TANL (the paper's main method): KG names enter only through visual prototypes (pool_frac 0), and a
# stream image joins the ID image prototypes only when CLIP's softmax and TANL's confident-ID mask agree.
TANL_BASE = dict(n_min=1, pool_frac=0.0, pool_in_text=False, id_admit="softmax+base")
OPERATING_POINTS = {
    "balanced": dict(image_mode="kg_clip", vote=True),        # clipped KG penalty + the base's vote (main)
    "max": dict(),                                            # plain image score (near-OOD optimised)
    "safe": dict(image_mode="kg_clip_caught", vote=True),     # penalty only on KG-caught images (do no harm)
}
# ReNeg on NegLabel's 10,000 static negatives (the ablation base): the plain configuration.
NEGLABEL_BASE = dict(n_min=1)

METHODS = ("tanl", "tanl_official_sh", "neglabel", "mcm", "reneg_tanl", "reneg_neglabel", "tins", "multi")
POOLS = ("blind", "full", "none", "neglabel_words")
FOUR_OOD_SETS = ["inaturalist", "sun", "places", "textures_all"]


def reneg_config(mode: Optional[str] = "balanced", base: str = "tanl", **overrides) -> ReNegConfig:
    """ReNegConfig for an operating point on a base ("tanl" or "neglabel"), with field overrides."""
    kw = dict(TANL_BASE if base == "tanl" else NEGLABEL_BASE)
    if mode is not None:
        if mode not in OPERATING_POINTS:
            raise ValueError(f"unknown operating point {mode!r}; choose from {sorted(OPERATING_POINTS)}")
        kw.update(OPERATING_POINTS[mode])
    kw.update(overrides)
    return ReNegConfig(**kw)


@dataclass
class RunSpec:
    """One row of a configuration file (configs/*.yaml)."""
    label: str
    method: str
    mode: Optional[str] = None                    # balanced | max | safe (reneg_tanl); None = plain (reneg_neglabel)
    pool: str = "blind"                           # blind | full | none | neglabel_words
    few_shot: bool = False                        # labelled ID images enter ReNeg's ID image prototypes
    overrides: Dict = field(default_factory=dict) # ReNegConfig fields (reneg_*) or TINSConfig fields (tins, multi)
    protocols: Optional[List[str]] = None         # overrides the file's protocols for this row

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"{self.label}: unknown method {self.method!r}; choose from {METHODS}")
        if self.pool not in POOLS:
            raise ValueError(f"{self.label}: unknown pool {self.pool!r}; choose from {POOLS}")

    @property
    def uses_labels(self) -> bool:
        return self.few_shot or self.method in ("tins", "multi")


# ---------------------------------------------------------------------------------------------- KG pool
def pool_path(cache) -> str:
    return os.path.join(cache.dir, "reneg_pool.npz")


def ensure_pool(cache, bank, model_fn: Callable, log=print) -> str:
    """The KG pool's CLIP text embeddings ("The nice {}.", as the text bank), cached in the cache folder."""
    path = pool_path(cache)
    names = [r["name"] for r in load_kg_pool()["pool"]]
    if os.path.isfile(path) and list(np.load(path)["pool_names"]) == names:
        return path
    from .g1 import clip_text_encoder

    model = model_fn()
    enc = clip_text_encoder(model, prompt="The nice {}.")
    ref = l2n(torch.as_tensor(bank.id_text[:20]).float().cpu())
    check = float((enc(list(bank.id_names[:20])).float() * ref).sum(1).min())
    log(f"[pool] encoder reproduces the bank's ID embeddings: min cosine {check:.6f} (expect > 0.999)")
    if check < 0.999:
        log("[pool] WARNING: the CLIP model does not match the text bank (another backbone or weights?)")
    return export_pool(bank, enc, cache.dir, prompt_note="The nice {}.", log=log)


def load_pools(path: str) -> Dict[str, Tuple[torch.Tensor, np.ndarray]]:
    kg = load_kg_pool()["pool"]
    z = np.load(path)
    if [r["name"] for r in kg] != list(z["pool_names"]):
        raise ValueError(f"{path} and resources/kg_pool.json disagree; delete the file to re-encode the pool")
    text = l2n(torch.tensor(z["pool_text"]).float())
    cluster = np.array([r["cluster"] for r in kg])
    blind = ~(np.asarray(z["pool_in_ssb_hard"], bool) | np.asarray(z["pool_in_ninco"], bool))
    idx = torch.as_tensor(np.nonzero(blind)[0])
    return {"full": (text, cluster), "blind": (text[idx], cluster[blind])}


# ---------------------------------------------------------------------------------------- few-shot setting
def labelled_images(runner) -> Tuple[torch.Tensor, np.ndarray]:
    """OpenOOD's ImageNet ID-val split (5,000 images, about 5 per class; not in OpenOOD v1.5's 45k test split)."""
    v = runner.get("imagenet_val")
    return v.feats, np.asarray(v.labels)


def shot_mask(runner, device: str = "cpu", thr: float = 0.999) -> np.ndarray:
    """Which of Four-OOD's 50k ID images are the labelled images (feature match, cosine > thr)."""
    va = runner.get("imagenet_val_all")
    A = l2n(va.feats.float().to(device))
    B = l2n(runner.get("imagenet_val").feats.float().to(device))
    shot = torch.zeros(len(va), dtype=torch.bool)
    for s in range(0, len(va), 4096):
        shot[s:s + 4096] = ((A[s:s + 4096] @ B.T).max(1).values > thr).cpu()
    return shot.numpy()


def add_four_ood_45k(runner, shot: np.ndarray):
    """Register ``four_ood_45k``: Four-OOD with the labelled images removed from the ID set (and the stream)."""
    from oodlab.runner import Protocol

    va = runner.get("imagenet_val_all")
    runner.extra_sets["imagenet_val_45k"] = va.subset(np.nonzero(~shot)[0], name="imagenet_val_45k")
    return Protocol("four_ood_45k", "imagenet_val_45k", {"four45": list(FOUR_OOD_SETS)},
                    "Four-OOD without OpenOOD's 5k ID-val images (the labelled images of the few-shot setting)")


# ------------------------------------------------------------------------------------------------- factory
DEFAULT_HEADS = (
    ("balanced on tins", "tins", "balanced", "blind", True),
    ("max on tins", "tins", "max", "blind", True),
    ("safe on tins", "tins", "safe", "blind", True),
    ("balanced on tins, full pool", "tins", "balanced", "full", True),
    ("balanced on tins, no labelled images", "tins", "balanced", "blind", False),
    ("balanced on tanl+tins", "tanl+tins", "balanced", "blind", True),
    ("max on tanl+tins", "tanl+tins", "max", "blind", True),
    ("safe on tanl+tins", "tanl+tins", "safe", "blind", True),
    ("balanced on tanl", "tanl", "balanced", "blind", True),
)


class Factory:
    """Builds the methods of the paper on one feature cache. Expensive pieces (KG pool, labelled images, TINS's
    static negatives) are computed once, on first use, and cached in the cache folder where that is useful."""

    def __init__(self, cache, runner, device: str = "cpu", model_fn: Optional[Callable] = None, log=print):
        self.cache, self.runner, self.device, self.log = cache, runner, device, log
        self.bank = cache.load_textbank()
        self._model_fn = model_fn
        self._model = None
        self._pools = None
        self._prior = None
        self._tins = None
        self._p45 = None

    # lazily loaded pieces
    def model(self):
        if self._model is None:
            if self._model_fn is None:
                raise RuntimeError("this method needs the CLIP model (pass model_fn)")
            self._model = self._model_fn()
        return self._model

    def pools(self):
        if self._pools is None:
            self._pools = load_pools(ensure_pool(self.cache, self.bank, self.model, self.log))
        return self._pools

    def prior(self):
        if self._prior is None:
            self._prior = labelled_images(self.runner)
        return self._prior

    def protocol(self, name: str, spec: RunSpec):
        """Four-OOD runs that use labelled images are scored on the 45k streams without them; ``four_ood_45k`` can
        also be asked for by name (e.g. TANL as the reference of the few-shot rows)."""
        if name == "four_ood_45k" or (name == "four_ood" and spec.uses_labels):
            if self._p45 is None:
                shot = shot_mask(self.runner, self.device)
                self.log(f"[few-shot] {int(shot.sum())} labelled images found in Four-OOD's 50k ID set (expect ~5,000)")
                self._p45 = add_four_ood_45k(self.runner, shot)
            return self._p45
        return name

    def tins_static(self):
        if self._tins is None:
            from .tins import build_init_candidates, build_static_negatives

            path = os.path.join(self.cache.dir, "tins_static.pt")
            X5, y5 = self.prior()
            X5 = l2n(X5.float())
            y5 = torch.as_tensor(y5).long()
            C = len(self.bank.id_names)
            cnt = torch.bincount(y5, minlength=C)
            proto = l2n(torch.zeros(C, X5.shape[1]).index_add_(0, y5, X5))
            if (cnt == 0).any():
                proto[cnt == 0] = l2n(self.bank.id_text.float())[cnt == 0]
            if os.path.isfile(path):
                st = torch.load(path, weights_only=False)
                if st["proto"].shape == proto.shape and torch.allclose(st["proto"], proto, atol=1e-5):
                    self._tins = st
                    return st
            model = self.model()
            neg, words = build_static_negatives(model, self.bank.id_text.float(), proto, n=2000,
                                                positive_labels=list(self.bank.id_names), log=self.log)
            st = {"neg": neg, "words": words, "cands": build_init_candidates(model, words, proto), "proto": proto}
            torch.save(st, path)
            self._tins = st
        return self._tins

    # methods
    def _tanl(self, official_sh: bool = False, record: bool = False):
        from oodlab.methods import TANL, TANLConfig

        cfg = TANLConfig.official_sh() if official_sh else TANLConfig.paper()
        return TANL(self.bank.id_text, self.bank.corpus_text, self.bank.noise_feats, cfg.with_(record=record),
                    device=self.device, n_neglabel=int(self.bank.n_selected))

    def _tins_method(self, overrides: Dict):
        from .tins import TINSConfig, TINSFeat

        st = self.tins_static()
        return TINSFeat(self.model(), self.bank.id_text.float(), st["proto"], st["neg"], st["cands"],
                        TINSConfig().with_(**overrides))

    def make(self, spec: RunSpec):
        b, m = self.bank, spec.method
        if m == "tanl":
            return self._tanl()
        if m == "tanl_official_sh":
            return self._tanl(official_sh=True)
        if m == "neglabel":
            from oodlab.methods import NegLabel, NegLabelConfig

            return NegLabel(b.id_text, b.corpus_text[: int(b.n_selected)], NegLabelConfig(), device=self.device)
        if m == "mcm":
            from oodlab.methods import MCM, MCMConfig

            return MCM(b.id_text_simple, MCMConfig.paper(), device=self.device)      # "a photo of a {}." as MCM
        if m == "reneg_tanl":
            from oodlab.methods import TANLConfig

            from .reneg_tanl import ReNegTANL

            pt, pc = self.pools()[spec.pool] if spec.pool in ("blind", "full") else (None, None)
            return ReNegTANL(b.id_text, b.corpus_text, b.noise_feats, pool_text=pt, pool_cluster=pc,
                             cfg=TANLConfig.paper(), rcfg=reneg_config(spec.mode or "balanced", "tanl", **spec.overrides),
                             device=self.device, n_neglabel=int(b.n_selected),
                             id_prior=self.prior() if spec.few_shot else None)
        if m == "reneg_neglabel":
            neg = b.corpus_text[: int(b.n_selected)]
            if spec.pool in ("blind", "full"):
                pt, pc = self.pools()[spec.pool]
            elif spec.pool == "neglabel_words":                  # NegLabel's own words as the pool, own clusters
                pt, pc = neg, np.arange(neg.shape[0])
            else:
                pt, pc = None, None
            return ReNegNegLabel(b.id_text, neg, pt, pc, reneg_config(spec.mode, "neglabel", **spec.overrides),
                                 device=self.device)
        if m == "tins":
            return self._tins_method(spec.overrides)
        if m == "multi":
            from .multi import HeadSpec, MultiBase

            heads = [HeadSpec(name, base, reneg_config(mode, "tanl"), pool=pool, prior=prior)
                     for name, base, mode, pool, prior in DEFAULT_HEADS]
            return MultiBase(self._tins_method(spec.overrides), self._tanl(record=True), b.id_text, self.pools(),
                             heads, primary="balanced on tins", device=self.device, id_prior=self.prior())
        raise ValueError(m)


def multi_labels() -> List[str]:
    return ["tins", "tanl", "tanl+tins"] + [h[0] for h in DEFAULT_HEADS]
