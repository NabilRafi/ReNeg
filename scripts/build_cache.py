#!/usr/bin/env python
"""Build the CLIP feature cache: encode every benchmark image once (what oodlab's Kaggle notebook 01 does).

    python scripts/build_cache.py --home oodlab_work --inputs /data/imagenet /data/openood --backbone ViT-B/16

Inputs (searched under ``--inputs`` and ``<home>/input``):

* ImageNet-1K validation in the Kaggle layout of the *ImageNet Object Localization Challenge*
  (``LOC_synset_mapping.txt``, ``LOC_val_solution.csv``, ``ILSVRC/Data/CLS-LOC/val/*.JPEG``);
* the OpenOOD v1.5 image lists and OOD sets (``benchmark_imglist.zip``, ``inaturalist.zip``, ``sun.zip``,
  ``places.zip``, ``texture.zip``, ``openimage_o.zip``, ``ssb_hard.zip``, ``ninco.zip``). Zips (or folders with
  the same names) found in the inputs are used; otherwise they are downloaded from OpenOOD's Google Drive files.

Output: ``<home>/working/cache/<backbone>/`` (features as fp16 .npz per set, manifests, ``textbank.pt``). Encoding
is resumable. A GPU is strongly recommended (about 1-2 hours for ViT-B/16 on a T4). docs/DATA.md has the details.

``OODLAB_SMOKE=1`` runs the whole pipeline on a tiny fake dataset with a random CLIP (tests only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

SETS = ["imagenet_val_all", "imagenet_test", "imagenet_val",            # ID (ImageNet-1K val, OpenOOD's splits)
        "ninco", "textures_all", "textures", "inaturalist", "sun", "places",
        "openimage_o", "openimage_o_val", "ssb_hard"]                    # OOD (OpenOOD v1.5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--home", default="oodlab_work", help="working folder (cache in <home>/working/cache)")
    ap.add_argument("--inputs", nargs="*", default=[], help="folders that contain ImageNet and the OpenOOD zips")
    ap.add_argument("--backbone", default="ViT-B/16", help="ViT-B/16 (paper) or ViT-L/14")
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--clip-weights", default=None, help="folder with (or for) the CLIP checkpoint")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-images", action="store_true", help="keep the extracted image folders")
    a = ap.parse_args(argv)
    os.environ["OODLAB_HOME"] = os.path.abspath(a.home)

    import oodlab
    import pandas as pd
    from oodlab.cache import FeatureCache, encode_manifest, find_existing_caches, import_from
    from oodlab.clipwrap import load_clip
    from oodlab.data import SPECS, build_manifest
    from oodlab.data.openood import cleanup, obtain_imglist_dir
    from oodlab.session import finish, start
    from oodlab.textbank import build_textbank

    ctx = start("build_cache", need_gpu=False, device=a.device)
    log = ctx.log
    roots = list(ctx.input_roots) + [os.path.abspath(p) for p in a.inputs]
    if ctx.device != "cuda" and not ctx.smoke:
        log.warning("no GPU: encoding ~500k images on a CPU takes a day or more")
    batch, workers, shard, total_neg, corpus_limit = a.batch_size, a.workers, a.shard_size, 10_000, None
    if ctx.smoke:
        batch, workers, shard, total_neg, corpus_limit = 32, 0, 150, 500, 2000

    cache = FeatureCache(ctx.dirs.cache, a.backbone).make()
    for r in find_existing_caches([ctx.dirs.inputs] + [os.path.abspath(p) for p in a.inputs]):
        import_from(r, cache, logger=log)
    weights = [a.clip_weights] if a.clip_weights else []
    model = load_clip(a.backbone, device=ctx.device, input_roots=roots + weights, download_root=a.clip_weights,
                      random_init=ctx.smoke)
    log.info(f"CLIP {a.backbone}: cache at {cache.dir}; already done: {cache.available()}")

    if not cache.has_textbank():
        mining = None
        if a.backbone != "ViT-B/16":                    # NegLabel's mining always uses ViT-B/16 (official code)
            mining = load_clip("ViT-B/16", device=ctx.device, input_roots=roots + weights,
                               download_root=a.clip_weights, random_init=ctx.smoke)
        bank = build_textbank(model, a.backbone, oodlab.imagenet_classnames(), oodlab.CORPUS_DIR,
                              total_neg=total_neg, corpus_limit=corpus_limit, logger=log, mining_model=mining)
        cache.save_textbank(bank)
    log.info(f"text bank: {cache.load_textbank().summary()}")

    try:
        imglist_dir = obtain_imglist_dir(ctx.dirs.scratch, roots, logger=log)
    except Exception:                                    # noqa: BLE001  sets that need the lists fail one by one
        log.exception("OpenOOD image lists unavailable (docs/DATA.md)")
        imglist_dir = None
    status, memo = [], {}
    for name in a.sets:
        t0 = time.time()
        if cache.has(name):
            status.append({"set": name, "status": "cached", "images": len(cache.load_set(name))})
            continue
        try:
            man = build_manifest(name, roots, ctx.dirs.scratch, imglist_dir, logger=log, strict_counts=not ctx.smoke,
                                 _imagenet_cache=memo)
            missing = man.missing_files(sample=500)
            if missing:
                raise FileNotFoundError(f"{len(missing)} of 500 sampled files missing, e.g. {missing[:2]}")
            fs = encode_manifest(model, man, cache, batch_size=batch, num_workers=workers, shard_size=shard, logger=log)
            status.append({"set": name, "status": "encoded", "images": len(fs), "expected": SPECS[name].expected,
                           "minutes": round((time.time() - t0) / 60, 1)})
        except Exception as e:                           # noqa: BLE001
            log.exception(f"[{name}] FAILED")
            status.append({"set": name, "status": f"FAILED: {type(e).__name__}: {str(e)[:150]}"})
        finally:
            key = SPECS[name].drive_key
            users = [n for n in a.sets if SPECS[n].drive_key == key]
            if key and not a.keep_images and all(cache.has(n) for n in users):
                cleanup(os.path.join(ctx.dirs.scratch, "extracted", key))
                cleanup(os.path.join(ctx.dirs.scratch, "zips", f"{key}.zip"))
    print(pd.DataFrame(status).to_string(index=False))
    print(pd.DataFrame(cache.summary()).to_string(index=False))
    missing = [n for n in a.sets if not cache.has(n)]
    finish(ctx, {"cache_dir": cache.dir, "missing_sets": missing})
    print(f"\ncache: {cache.dir}\nmissing sets: {missing or 'none'}")
    print(f"next: python scripts/evaluate.py --config configs/main_vitb16.yaml --cache {ctx.dirs.cache} --out runs/main")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
