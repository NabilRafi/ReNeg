"""FAKE TANL: a static NegLabel score (enough to test the notebook plumbing)."""
from dataclasses import dataclass, replace


@dataclass
class TANLConfig:
    logit_scale: float = 100.0
    num_neg: int = 1000

    @classmethod
    def paper(cls):
        return cls()

    def with_(self, **kw):
        return replace(self, **kw)


class TANL:
    name = "tanl"

    def __init__(self, id_text, corpus_text, noise_feats, cfg, device="cpu", **kw):
        self.id_text, self.neg = id_text, corpus_text[: cfg.num_neg]
        self.cfg = cfg
