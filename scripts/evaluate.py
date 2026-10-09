#!/usr/bin/env python
"""Evaluate OOD detectors on a CLIP feature cache: the paper's tables from a YAML configuration.

    python scripts/evaluate.py --config configs/main_vitb16.yaml --cache oodlab_work/working/cache \\
        --out runs/main_vitb16

``--cache`` is the folder that holds one sub-folder per backbone (``vit-b-16/``, ``vit-l-14/``), as written by
``scripts/build_cache.py`` or oodlab's Kaggle notebook 01. Every (run, protocol, OOD set, stream order) row is
appended to ``<out>/runs.csv`` as soon as it finishes, so a stopped evaluation resumes where it stopped. Then

    python scripts/summarize.py runs/main_vitb16/runs.csv --ref tanl

prints the group means (both FPR95 conventions) and the paired differences against the reference.

Configuration file (see configs/)::

    backbone: ViT-B/16
    protocols: [openood_v15, four_ood]
    seeds: [0, 1, 2]
    runs:
      - {label: tanl, method: tanl}
      - {label: reneg_balanced_blind, method: reneg_tanl, mode: balanced, pool: blind}
      - {label: reneg_balanced_blind_5shot, method: reneg_tanl, mode: balanced, pool: blind, few_shot: true}

Methods: tanl, tanl_official_sh, neglabel, mcm, reneg_tanl (ReNeg on TANL), reneg_neglabel (ReNeg on NegLabel's
negatives, the ablation base), tins (TINS re-implementation, 5 labelled images per class) and multi (TINS and TANL
once per batch with nine ReNeg heads; writes one row per head). Runs that use labelled images are scored on
Four-OOD's 45k ID images that are not those images (protocol four_ood_45k).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))                  # run from a clone without `pip install -e .`

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from oodlab.cache import FeatureCache  # noqa: E402
from oodlab.metrics import openood_metrics  # noqa: E402
from oodlab.runner import PROTOCOLS, Runner  # noqa: E402
from reneg.pipeline import Factory, RunSpec, multi_labels  # noqa: E402

COLUMNS = ["label", "method", "protocol", "group", "dataset", "seed", "fpr95", "fpr95_idpos", "auroc", "aupr_in",
           "aupr_out", "acc", "n_id", "n_ood", "seconds", "backbone"]


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("backbone", "runs"):
        if key not in cfg:
            raise ValueError(f"{path}: missing '{key}'")
    cfg.setdefault("protocols", ["openood_v15", "four_ood"])
    cfg.setdefault("seeds", [0, 1, 2])
    cfg["runs"] = [RunSpec(**r) for r in cfg["runs"]]
    labels = [r.label for r in cfg["runs"]]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: run labels must be unique")
    return cfg


def protocol_sets(p) -> list:
    p = PROTOCOLS[p] if isinstance(p, str) else p
    return [(g, d) for g, names in p.groups.items() for d in names]


class ScoreHook:
    """Collects every score a multi-head method reports, per image in stream order."""

    def __init__(self):
        self.parts = []

    def on_reset(self, method, seg):
        self.parts = []

    def on_step(self, method, seg, start, end, out, stream):
        self.parts.append((start, {k: v.numpy() for k, v in out.info["scores"].items()}))

    def on_end(self, method, stream):
        self.labels = np.asarray(stream.labels)
        n = len(stream)
        names = list(self.parts[0][1]) if self.parts else []
        self.scores = {k: np.zeros(n, np.float64) for k in names}
        for s, d in self.parts:
            for k, v in d.items():
                self.scores[k][s:s + len(v)] = v


def truncate_sets(runner, cache, n: int, log=print) -> None:
    """Quick mode: the first n images of every cached set (for smoke tests, not for numbers)."""
    for name in cache.available():
        fs = runner.get(name)
        if len(fs) > n:
            runner.extra_sets[name] = fs.subset(np.arange(n))
    log(f"[quick] every set truncated to {n} images: numbers are NOT comparable to the paper")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="YAML file in configs/")
    ap.add_argument("--cache", required=True, help="feature cache root (contains vit-b-16/, vit-l-14/)")
    ap.add_argument("--out", required=True, help="output folder (runs.csv is appended)")
    ap.add_argument("--device", default="auto", help="cuda, cpu or auto")
    ap.add_argument("--seeds", type=int, nargs="*", help="override the stream orders")
    ap.add_argument("--datasets", nargs="*", help="only these OOD sets")
    ap.add_argument("--runs", nargs="*", help="only these run labels")
    ap.add_argument("--clip-weights", default=None, help="folder with (or for) the CLIP checkpoint")
    ap.add_argument("--random-clip", action="store_true", help="random tiny CLIP (tests with a smoke cache only)")
    ap.add_argument("--max-images", type=int, default=None, help="quick mode: truncate every set (tests only)")
    ap.add_argument("--save-scores", action="store_true", help="multi: save per-image scores of every head")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    cache = FeatureCache(a.cache, cfg["backbone"])
    if not (os.path.isdir(cache.dir) and cache.has_textbank()):
        raise SystemExit(f"no feature cache with a text bank at {cache.dir} (build it with scripts/build_cache.py)")
    os.makedirs(a.out, exist_ok=True)
    runs_path = os.path.join(a.out, "runs.csv")
    done = pd.read_csv(runs_path) if os.path.isfile(runs_path) else pd.DataFrame(columns=COLUMNS)
    have = set(zip(done.label, done.protocol, done.dataset, done.seed))
    quiet = logging.getLogger("oodlab.runner")
    quiet.setLevel(logging.WARNING)
    runner = Runner(cache, device=device, logger=quiet)
    if a.max_images:
        truncate_sets(runner, cache, a.max_images)

    def model_fn():
        from oodlab.clipwrap import load_clip

        w = a.clip_weights
        return load_clip(cfg["backbone"], device=device, input_roots=[w] if w else [], download_root=w,
                         random_init=a.random_clip)

    factory = Factory(cache, runner, device=device, model_fn=model_fn, log=print)
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    print(f"{cfg['backbone']} | cache {cache.dir} | device {device} | seeds {seeds} | sets {cache.available()}")

    for spec in cfg["runs"]:
        if a.runs and spec.label not in a.runs:
            continue
        labels = multi_labels() if spec.method == "multi" else [spec.label]
        if spec.method == "multi" and spec.label != "multi":
            labels = [f"{spec.label}/{x}" for x in labels]
        for pname in spec.protocols or cfg["protocols"]:
            proto = factory.protocol(pname, spec)
            miss = runner.missing(proto)
            if miss:
                print(f"skipped {spec.label} on {pname}: sets missing from the cache {miss}")
                continue
            ptag = proto if isinstance(proto, str) else proto.name
            for group, ds in protocol_sets(proto):
                if a.datasets and ds not in a.datasets:
                    continue
                for seed in seeds:
                    if all((lab, ptag, ds, seed) in have for lab in labels):
                        continue
                    t0 = time.time()
                    method = factory.make(spec)
                    hook = ScoreHook() if spec.method == "multi" else None
                    df = runner.evaluate(method, proto, seeds=[seed], datasets=[ds], config=spec.label,
                                         hooks_factory=(lambda n, s: [hook]) if hook else None)
                    sec = round(time.time() - t0, 1)
                    if hook is None:
                        rows = df.assign(label=spec.label, method=spec.method, seconds=sec, backbone=cfg["backbone"])
                        rows = rows.reindex(columns=COLUMNS)
                    else:
                        recs = []
                        for lab, key in zip(labels, multi_labels()):
                            m = openood_metrics(hook.scores[key], hook.labels)
                            recs.append({"label": lab, "method": "multi", "protocol": ptag, "group": group,
                                         "dataset": ds, "seed": seed, "seconds": sec, "backbone": cfg["backbone"],
                                         **{k: m[k] for k in ("fpr95", "fpr95_idpos", "auroc", "aupr_in", "aupr_out",
                                                              "acc", "n_id", "n_ood")}})
                            if a.save_scores:
                                sd = os.path.join(a.out, "scores")
                                os.makedirs(sd, exist_ok=True)
                                np.savez_compressed(os.path.join(sd, f"{ptag}__{ds}__s{seed}.npz"),
                                                    labels=hook.labels, names=np.asarray(multi_labels()),
                                                    scores=np.stack([hook.scores[k] for k in multi_labels()]))
                        rows = pd.DataFrame(recs, columns=COLUMNS)
                    done = pd.concat([done, rows], ignore_index=True) if len(done) else rows
                    done.to_csv(runs_path, index=False)
                    have |= set(zip(rows.label, rows.protocol, rows.dataset, rows.seed))
                    head = rows.iloc[0]
                    print(f"{spec.label:<34} {ptag}:{ds:<13} seed {seed}  FPR95 {head.fpr95:6.2f}  "
                          f"(ID-positive {head.fpr95_idpos:6.2f})  AUROC {head.auroc:6.2f}  {sec:7.1f}s", flush=True)
    print(f"\nrows in {runs_path}: {len(done)}. Tables: python scripts/summarize.py {runs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
