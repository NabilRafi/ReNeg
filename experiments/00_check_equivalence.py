#!/usr/bin/env python
"""Check that the replay engine equals the packaged method: replay(TANL trace) == reneg.ReNegTANL.step.

    python experiments/00_check_equivalence.py              # NINCO stream, order 0, first 2,048 images
    python experiments/00_check_equivalence.py --n 6144

TANL runs on the first N images of the stream with its real corpus; ReNeg is then run (a) inside ReNegTANL and
(b) replayed on the stored trace, for the packaged admission rule, the agreed admission (balanced and max) and the
few-shot prior. Differences should be at float32 rounding level (< 1e-4). With the pool and the image score off,
the replay's text score is TANL's own score (identical metrics).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replaylab  # noqa: E402,F401  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import reneg.core as RC  # noqa: E402
from oodlab.methods.tanl import TANL, TANLConfig  # noqa: E402
from reneg.reneg_tanl import ReNegTANL  # noqa: E402
from replaylab import data as D  # noqa: E402
from replaylab.replay import MODES, replay_x  # noqa: E402
from replaylab.traces import make_trace, replay, tanl_inputs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="ninco")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2048, help="stream images to compare")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    tid, corpus, noise, n_sel = tanl_inputs()
    X, is_ood, _ = D.pair_stream(D.load_set("imagenet_test")["X"], D.load_set(a.set)["X"], a.seed)
    X, is_ood = X[:a.n], is_ood[:a.n]
    P, cl = D.pool_subset("blind")
    cfg = TANLConfig().with_(init_seed=a.seed)
    tr = make_trace(TANL(tid, corpus, noise, cfg.with_(record=True), n_neglabel=n_sel), X)   # fp32 trace
    print(f"{a.set} order {a.seed}: {len(X)} images ({int(is_ood.sum())} OOD), TANL trace {tr['sec']:.0f}s")
    worst = 0.0

    def direct(rc, prior=None):
        m = ReNegTANL(tid, corpus, noise, pool_text=P, pool_cluster=cl, cfg=cfg, rcfg=rc, n_neglabel=n_sel,
                      id_prior=prior)
        m.reset()
        return np.concatenate([m.step(X[s:s + 256]).score.double().numpy() for s in range(0, len(X), 256)])

    def report(name, x, y):
        nonlocal worst
        d = np.abs(x - y)
        worst = max(worst, float(d.max()))
        rc = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        print(f"  {name:<44} max |diff| {d.max():.2e}  mean {d.mean():.2e}  rank corr {rc:.6f}", flush=True)

    rc0 = RC.ReNegConfig(n_min=1)
    report("packaged admission (softmax in TANL's mask)", direct(rc0), replay(tr, X, tid, P, cl, rc0)[0])
    shots = D.shots()
    for mode in ("balanced", "max"):
        for prior in (False, True):
            ref = direct(RC.ReNegConfig(**MODES[mode], id_admit="softmax+base"), (shots[0], shots[1]) if prior else None)
            new = replay_x(tr, X, tid, P, cl, RC.ReNegConfig(**MODES[mode]), {"idadm": "softmax+tanl"},
                           prior=(shots[0], shots[1]) if prior else None)
            report(f"{mode}, agreed admission{', 5-shot prior' if prior else ''}", ref, new)
    s0 = replay(tr, X, tid, None, None, rc0.with_(use_pool=False, use_image=False))[0]
    if is_ood.any() and (~is_ood).any():
        r_t, r_0 = D.metrics(tr["tanl"][~is_ood], tr["tanl"][is_ood]), D.metrics(s0[~is_ood], s0[is_ood])
        print(f"  TANL alone FPR95 {r_t['fpr95']:.3f} AUROC {r_t['auroc']:.3f} | replay with pool and image off "
              f"FPR95 {r_0['fpr95']:.3f} AUROC {r_0['auroc']:.3f}")
    ok = worst < a.tol
    print("PASS" if ok else f"FAIL: largest difference {worst:.2e} >= {a.tol:g}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
