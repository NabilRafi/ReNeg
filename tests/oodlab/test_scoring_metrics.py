import numpy as np
import pytest
import torch

from oodlab.methods.scoring import (
    activation_aware_score,
    activation_aware_score_fp16_loop,
    find_best_threshold,
    find_best_threshold_fast,
)
from oodlab.metrics import openood_metrics


def _loop_reference(logits, n_id, step):
    """Direct transcription of the official scoring loop in float64."""
    lg = logits.double()
    n_neg = lg.shape[1] - n_id
    if step == 0:
        p = torch.softmax(lg, dim=1)
        return p[:, :n_id].sum(1)
    outs = []
    for j in range(0, n_neg, step):
        m = min(j + step, n_neg)
        p = torch.softmax(torch.cat([lg[:, :n_id], lg[:, n_id:n_id + m]], 1), dim=1)
        outs.append(p[:, :n_id].sum(1))
    return torch.stack(outs, 1).mean(1)


@pytest.mark.parametrize("step", [0, 1, 2, 7])
def test_activation_aware_closed_form(step):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(16, 10 + 45, generator=g) * 5
    ours = activation_aware_score(logits, 10, step).double()
    ref = _loop_reference(logits, 10, step)
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_activation_aware_fp16_loop_runs():
    logits = (torch.randn(8, 30) * 10).half()
    s = activation_aware_score_fp16_loop(logits, 10, 1)
    assert s.dtype == torch.float16 and s.shape == (8,)


def test_thresholds_agree_including_ties():
    rng = np.random.default_rng(0)
    for trial in range(20):
        x = np.concatenate([rng.normal(0.8, 0.05, 700), rng.normal(0.2, 0.1, 300)]).clip(0, 1)
        if trial % 2:
            x = np.round(x, 2)  # heavy ties
        t = torch.tensor(x, dtype=torch.float32)
        a, b = find_best_threshold(t), find_best_threshold_fast(t)
        assert abs(a - b) < 1e-6, (trial, a, b)


def test_metrics_perfect_and_conventions():
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    label = np.array([3, 1, 2, -1, -1])
    m = openood_metrics(conf, label, np.array([3, 1, 0, 5, 5]))
    assert m["fpr95"] == 0.0 and m["auroc"] == 100.0 and m["fpr95_idpos"] == 0.0
    assert abs(m["acc"] - 200 / 3) < 1e-9
    # one OOD sample scores like ID: OOD-positive FPR (ID flagged) vs ID-positive FPR (OOD accepted) differ
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.1, 0.2, 0.3])
    label = np.array([0, 0, 0, 0, -1, -1, -1, -1])
    m = openood_metrics(conf, label)
    assert m["fpr95"] == 100.0        # to catch 95 % of OOD we must flag every ID sample
    assert m["fpr95_idpos"] == 25.0   # at 95 % ID recall one of four OOD samples is accepted


def test_targets_and_conventions():
    from oodlab.targets import ACCEPTANCE, TARGETS, targets_for

    t = {x.key: x for x in targets_for("tanl", "four_ood")}
    assert t["mean:four"].fpr95 == 9.81 and t["mean:four"].convention == "ood_pos"
    assert abs(sum(t[k].fpr95 for k in ["inaturalist", "sun", "places", "textures_all"]) / 4 - 9.81) < 0.01
    nl = {x.key: x for x in targets_for("neglabel", "four_ood", "neglabel_paper")}
    assert nl["mean:four"].convention == "id_pos"
    assert {a[0] for a in ACCEPTANCE} == {"tanl", "adaneg"}
    for m, p, src in [("adaneg", "four_ood", "adaneg_paper"), ("adaneg", "four_ood", "adaneg_repo"),
                      ("neglabel", "four_ood", "neglabel_repo"), ("mcm", "four_ood", "mcm_paper")]:
        d = {x.key: x.fpr95 for x in targets_for(m, p, src)}
        mean = sum(d[k] for k in ["inaturalist", "sun", "places", "textures_all"]) / 4
        assert abs(mean - d["mean:four"]) < 0.02, (m, src, mean, d["mean:four"])
