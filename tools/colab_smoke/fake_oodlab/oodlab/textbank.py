from dataclasses import dataclass, field

import torch


@dataclass
class TextBank:
    backbone: str
    id_names: list
    id_text: torch.Tensor
    id_text_simple: torch.Tensor
    words: list
    is_adj: torch.Tensor
    corpus_text: torch.Tensor
    mining_score: torch.Tensor
    n_selected: int
    n_sel_adj: int
    noise_feats: torch.Tensor
    meta: dict = field(default_factory=dict)

    @property
    def neg_text(self):
        return self.corpus_text[: self.n_selected]

    @property
    def logit_scale(self):
        return float(self.meta.get("logit_scale", 100.0))

    def summary(self):
        return {"backbone": self.backbone, "n_id": len(self.id_names), "n_corpus": len(self.words),
                "n_selected": self.n_selected, "dim": int(self.id_text.shape[1])}

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert d.get("format") == "oodlab.textbank.v1"
        return cls(backbone=d["backbone"], id_names=d["id_names"], id_text=d["id_text"],
                   id_text_simple=d["id_text_simple"], words=d["words"], is_adj=d["is_adj"],
                   corpus_text=d["corpus_text"], mining_score=d["mining_score"], n_selected=d["n_selected"],
                   n_sel_adj=d["n_sel_adj"], noise_feats=d["noise_feats"], meta=d["meta"])
