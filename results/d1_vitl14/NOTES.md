# D1 (ViT-L/14 check) — run notes (2026-10-06)

Files: `d1_runs.csv`, `d1_info.json`; logs `D1_run.log`, `encode_l14.log`.

- The ViT-L/14 cache in this Drive drop again had only 7 sets. iNaturalist, Textures, Textures-all, OpenImage-O and
  OpenImage-O-val were encoded locally with oodlab's `encode_manifest` on the ViT-B/16 manifests' exact image lists
  (check: 512 Textures images re-encoded with ViT-B/16 vs the Kaggle cache, cosine min 0.99906, mean 1.00001).
- The `tanl` (paper settings) reference row is missing: it reads G3b's `g3_runs.csv`, which is not in this drop.

| method | backbone | convention | iNat | SUN | Places | Textures | mean | published | gap |
|---|---|---|---|---|---|---|---|---|---|
| MCM | B/16 | ID+ | 34.38 | 38.48 | 47.91 | 58.65 | 44.86 | 42.74 | +2.12 |
| MCM | L/14 | ID+ | 31.57 | 26.36 | 35.08 | 58.17 | 37.80 | 38.17 | -0.37 |
| NegLabel | B/16 | ID+ | 1.80 | 20.13 | 32.76 | 42.22 | 24.23 | 25.40 | -1.17 |
| NegLabel | L/14 | ID+ | 1.77 | 23.61 | 33.81 | 42.80 | 25.50 | 24.81 | +0.69 |
| TANL official_sh | L/14 | OOD+ | 0.58 | 4.60 | 24.74 | 28.16 | 14.52 | 9.52 | +5.00 |

**Reading (the notebook's own rule):** NegLabel and MCM land on their published ViT-L/14 numbers, so the L/14
features and text bank are right; the gap is TANL's own (mostly Textures: 28.16 vs 13.57).

Side note: `d1_info.json` shows the L/14 text bank's `mining_backbone` is ViT-B/16 (the 10,000 negatives were
selected with B/16). NegLabel still matches its paper with them, so this is not a bank problem for NegLabel, but TANL
re-selects negatives per batch from the 69,554-word corpus and may be more sensitive to it.
