# Data

No images or image features are stored in this repository. Everything is computed from public benchmarks: the
images are encoded once with CLIP into a feature cache, and every experiment replays test streams from that cache.

## Benchmarks

| Set (cache name) | Role | Images | Source |
| --- | --- | --- | --- |
| `imagenet_val_all` | ID of Four-OOD | 50,000 | ImageNet-1K validation, from Kaggle's *ImageNet Object Localization Challenge* |
| `imagenet_test` | ID of OpenOOD v1.5 | 45,000 | the same images, OpenOOD's `test_imagenet.txt` split |
| `imagenet_val` | OpenOOD's ID validation split; the labelled images of the few-shot setting | 5,000 | OpenOOD's `val_imagenet.txt` |
| `ssb_hard` | near-OOD | 49,000 | OpenOOD v1.5 (SSB-hard) |
| `ninco` | near-OOD | 5,879 | OpenOOD v1.5 (NINCO) |
| `inaturalist` | far-OOD (both benchmarks) | 10,000 | OpenOOD v1.5 / MOS subset |
| `textures` | far-OOD (OpenOOD) | 5,160 | DTD, OpenOOD's `test_textures.txt` |
| `openimage_o` | far-OOD (OpenOOD) | 15,869 | OpenImage-O, OpenOOD's test split |
| `openimage_o_val` | OpenOOD's OOD validation split (tuning only) | 1,763 | OpenImage-O, OpenOOD's validation split |
| `sun`, `places` | Four-OOD | 10,000 each | MOS subsets |
| `textures_all` | Four-OOD | 5,640 | all DTD images |

`src/oodlab/data/registry.py` lists them; `src/oodlab/data/openood.py` holds the OpenOOD v1.5 Google Drive file
ids (the image lists `benchmark_imglist.zip` and one zip per OOD dataset) and downloads them with `gdown` when no
local copy is given. ImageNet is read in the Kaggle layout (`LOC_synset_mapping.txt`, `LOC_val_solution.csv`,
`ILSVRC/Data/CLS-LOC/val/*.JPEG`); on Kaggle it is mounted read-only after joining the competition.

Class order and names follow OpenOOD-VLM (`src/oodlab/resources/imagenet_wnids.json`, `imagenet_classes.json`).

## Building the feature cache

```bash
python scripts/build_cache.py --home oodlab_work --inputs /path/to/imagenet /path/to/openood_zips
python scripts/build_cache.py --home oodlab_work --inputs ... --backbone ViT-L/14
```

or Kaggle notebook 01 (docs/KAGGLE_GUIDE.md). About 1-2 hours for ViT-B/16 on a T4, resumable. Missing OpenOOD
zips are downloaded from Google Drive. `OODLAB_SMOKE=1` runs the same pipeline on a tiny fake dataset.

Layout (`src/oodlab/cache.py`):

    oodlab_work/working/cache/<backbone>/          vit-b-16/ or vit-l-14/
        CACHE_INFO.json                            contents and versions
        textbank.pt                                text side: ID names, TANL's 69,554-word corpus, NegLabel's
                                                   10,000 negatives (the first corpus rows), 15 noise images
        images/<set>.npz                           feats (N, D) fp16, L2-normalised | labels (N,) | keys (N,)
        images/<set>.json, manifests/<set>.csv     the exact image list that was encoded
        reneg_pool.npz                             CLIP text embeddings of the KG pool (written on first use)
        tins_static.pt                             TINS's static negatives (written on first use)

The ViT-B/16 cache is about 0.3 GB. The text bank follows NegLabel's and TANL's official code (WordNet word lists
from NegLabel, Apache-2.0, in `src/oodlab/resources/neglabel_txtfiles/`; NegLabel's mining always uses ViT-B/16).

## 8-bit exports for the CPU experiments

`scripts/export_features.py` (or Colab notebooks G2a and G2b) writes compact copies of the cache for
`experiments/`: image features as int8 codes with one fp16 scale per image (`x = q * scale / 127`, cosine error
about 3e-4), split into files under 24 MB.

| File | Contents |
| --- | --- |
| `reneg_stream_*.npz` | ImageNet test, SSB-hard, NINCO, Textures, iNaturalist, OpenImage-O, ImageNet val, OpenImage-O val |
| `reneg_stream4ood_*.npz` | ImageNet val (50k), SUN, Places, Textures (all) |
| `reneg_textbank_*.npz` | TANL's corpus (8-bit), its words, NegLabel's split, the ID text, the noise images |
| `reneg_pool.npz` | KG pool text embeddings (fp16) and metadata (WordNet id, Wu-Palmer similarity, nearest ImageNet class, SSB-hard / NINCO membership) |
| `reneg_pack_{1,2}.npz` | optional (`--what pack`): G1b's analysis pack for the Seam 1 analyses |

The ViT-B/16 exports are about 140 MB. They are derived from the benchmark images; they are not part of this
repository. If you publish them (for example as a release asset), check the licences of the source datasets first.

## Files in `results/` that are derived from the data

* `results/p4_tins/tins_traces_1.zip`: TINS's per-image scores on every benchmark stream (no features).
* `results/p4_tins/shot_mask.npy`, `results/replays/val5k_mask.npy`: which of Four-OOD's 50k ID images are the
  labelled images (boolean masks).
* `results/seam1/reneg_gen_probe.npz`: ViT-B/16 embeddings of images we generated with SDXL-Turbo (notebook P1).
* `results/seam1/p2/`: ViT-L/14 text embeddings, per-concept image centroids (NINCO, SSB-hard, ImageNet, half of
  each concept's images) and the L/14-to-B/16 map of notebook P2.

## Licences of the sources

ImageNet (terms of access of image-net.org / the Kaggle competition rules), OpenOOD v1.5 benchmark lists (MIT)
and its datasets (SSB-hard, NINCO, iNaturalist, DTD, OpenImage-O, SUN, Places; each under its own terms),
WordNet 3.0 (WordNet licence), NegLabel's word lists (Apache-2.0). See THIRD_PARTY_NOTICES.md.
