#!/usr/bin/env python
"""Export compact 8-bit copies of the feature cache: the inputs of the CPU replay experiments in experiments/.

    python scripts/export_features.py --cache oodlab_work/working/cache --out exports/vit-b-16

This is what Colab notebooks G2a and G2b (and the optional analysis pack of G1b) write, as one command. The CPU is
enough (about 5 minutes); ``pool`` and ``pack`` load CLIP to encode text.

  streams    reneg_stream_*.npz      ImageNet test (45k), SSB-hard, NINCO, Textures, iNaturalist, OpenImage-O, and
                                     OpenOOD's validation splits (ImageNet val 5k, OpenImage-O val 1,763)
  four_ood   reneg_stream4ood_*.npz  ImageNet val (50k, Four-OOD's ID set), SUN, Places, Textures (all 5,640)
  textbank   reneg_textbank_*.npz    TANL's 69,554-word corpus in the official order (8-bit), NegLabel's split,
                                     the ID text and the 15 noise-image features TANL starts from
  pool       reneg_pool.npz          CLIP text embeddings of the 14,526 WordNet near-OOD candidates
  pack       reneg_pack_{1,2}.npz    G1b's analysis pack: confident class statistics, the mean image, concept names
                                     of SSB-hard and NINCO (only the Seam 1 analyses need it; needs NLTK WordNet)

Image features are stored as int8 codes with one fp16 scale per image (cosine error about 3e-4, docs/DATA.md).
Every file stays under ``--max-mb``. Then point the experiments at the folder:

    export RENEG_EXPORTS=exports/vit-b-16
    python experiments/01_build_traces.py
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from oodlab.cache import FeatureCache  # noqa: E402
from reneg.packs import export_pool, export_streams, export_textbank, load_kg_pool, load_textbank_export  # noqa: E402
from reneg.transport import l2n  # noqa: E402

STREAMS = ["imagenet_test", "ssb_hard", "ninco", "textures", "inaturalist", "openimage_o", "imagenet_val",
           "openimage_o_val"]
FOUR_OOD = ["imagenet_val_all", "sun", "places", "textures_all"]
WHAT = ["streams", "four_ood", "textbank", "pool", "pack"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="feature cache root (contains vit-b-16/, vit-l-14/)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--backbone", default="ViT-B/16")
    ap.add_argument("--what", nargs="*", default=["streams", "four_ood", "textbank", "pool"], choices=WHAT)
    ap.add_argument("--max-mb", type=float, default=24.0, help="largest file size")
    ap.add_argument("--clip-weights", default=None, help="folder with (or for) the CLIP checkpoint")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--random-clip", action="store_true", help="random tiny CLIP (tests with a smoke cache only)")
    a = ap.parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    cache = FeatureCache(a.cache, a.backbone)
    if not (os.path.isdir(cache.dir) and cache.has_textbank()):
        raise SystemExit(f"no feature cache with a text bank at {cache.dir} (build it with scripts/build_cache.py)")
    bank = cache.load_textbank()
    os.makedirs(a.out, exist_ok=True)
    written = []
    print(f"{a.backbone} | cache {cache.dir} | sets {cache.available()}")

    model = None

    def clip():
        nonlocal model
        if model is None:
            from oodlab.clipwrap import load_clip

            w = a.clip_weights
            model = load_clip(a.backbone, device=device, input_roots=[w] if w else [], download_root=w,
                              random_init=a.random_clip)
        return model

    if "streams" in a.what:
        written += export_streams(cache, a.out, STREAMS, max_mb=a.max_mb)
    if "four_ood" in a.what:
        written += export_streams(cache, a.out, FOUR_OOD, max_mb=a.max_mb, prefix="reneg_stream4ood")
    if "textbank" in a.what:
        paths = export_textbank(bank, a.out, max_mb=a.max_mb)
        d = load_textbank_export(paths)
        ref = l2n(bank.corpus_text.float())
        rows = torch.randperm(ref.shape[0], generator=torch.Generator().manual_seed(0))[:2000]
        err = float((d["corpus_text"][rows] @ ref[rows].T - ref[rows] @ ref[rows].T).abs().max())
        print(f"text bank round trip: largest cosine error {err:.5f} (fine below 0.005)")
        written += paths
    if "pool" in a.what:
        from reneg.g1 import clip_text_encoder

        enc = clip_text_encoder(clip(), prompt="The nice {}.")
        ref = l2n(torch.as_tensor(bank.id_text[:20]).float().cpu())
        check = float((enc(list(bank.id_names[:20])).float() * ref).sum(1).min())
        print(f"pool: the encoder reproduces the bank's ID embeddings with min cosine {check:.6f} (expect > 0.999)")
        if check < 0.999 and not a.random_clip:
            print("WARNING: this CLIP model does not match the text bank (another backbone or other weights?)")
        written.append(export_pool(bank, enc, a.out, pool=load_kg_pool(), prompt_note="The nice {}."))
    if "pack" in a.what:
        from reneg.g1 import clip_text_encoder
        from reneg.g1b import G1b, G1bConfig
        from reneg.wordnet import get_wordnet

        wn = get_wordnet()
        if wn is None:
            print("WARNING: NLTK WordNet unavailable: concept names fall back to folder names")
        cfg = G1bConfig(backbone=a.backbone, device=device, p_min=0.5, n_min=3, n_cap=20)
        g = G1b(cache, bank, cfg)
        g.prepare()
        g.build_oracle_concepts(clip_text_encoder(clip(), prompt=cfg.prompt), wn)
        written += g.export_pack(a.out)
    total = sum(os.path.getsize(p) for p in written) / 1e6
    print(f"\n{len(written)} files, {total:.0f} MB in {a.out}")
    print(f"next: RENEG_EXPORTS={a.out} python experiments/01_build_traces.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
