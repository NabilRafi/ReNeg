"""Worked examples of modifying TANL by overriding one method.

These are *starting points for your own research code*, not reproductions: nothing
here is from the TANL paper. Each class changes exactly one step of
``TANL.step`` (see the override points listed in ``oodlab/methods/tanl.py``).

    from oodlab.methods.variants import TANL_LCB
    m = TANL_LCB(bank.id_text, bank.corpus_text, bank.noise_feats, TANLConfig.paper(), kappa=1.0)
    runner.evaluate(m, "four_ood", seeds=[0])
"""
from __future__ import annotations

import torch

from .tanl import TANL


class TANL_LCB(TANL):
    """Select negatives by a lower confidence bound of the activation difference.

    TANL picks the top-M labels by ``mean(act | OOD queue) - mean(act | ID queue)``.
    Here each label is ranked by that difference *minus* ``kappa`` standard errors,
    so a label that fired strongly on only one or two queued images is not trusted
    until its evidence is consistent (pessimism under a noisy, small queue).
    ``kappa=0`` recovers TANL exactly.
    """

    name = "tanl_lcb"

    def __init__(self, *args, kappa: float = 1.0, **kwargs):
        self.kappa = float(kappa)
        super().__init__(*args, **kwargs)

    def select(self, combined: torch.Tensor) -> torch.Tensor:
        if self.kappa == 0:
            return super().select(combined)
        n_pos, n_neg = self.pos_q.shape[0], self.neg_q.shape[0]
        se = torch.sqrt(self.neg_q.float().var(0, unbiased=False) / max(n_neg, 1)
                        + self.pos_q.float().var(0, unbiased=False) / max(n_pos, 1))
        return (combined - self.kappa * se.to(combined.dtype)).sort(descending=True)[1][: self.cfg.num_neg]

    def config_dict(self):
        d = super().config_dict()
        d["kappa"] = self.kappa
        return d


class TANL_FixedGap(TANL):
    """Example of changing the admission rule: absolute margins instead of the relative gap.

    Admit as ID if score > thr + margin_id, as OOD if score < thr - margin_ood.
    """

    name = "tanl_fixedgap"

    def __init__(self, *args, margin_id: float = 0.1, margin_ood: float = 0.1, **kwargs):
        self.margin_id, self.margin_ood = float(margin_id), float(margin_ood)
        super().__init__(*args, **kwargs)

    def admit(self, conf: torch.Tensor, thr: float):
        return conf > thr + self.margin_id, conf < thr - self.margin_ood
