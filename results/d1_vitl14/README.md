# D1: is our ViT-L/14 set-up right?

Notebook `notebooks/4_evaluation/D1_vitl14_check.ipynb` (command-line equivalent: `configs/baselines_vitl14.yaml`),
run on Oct 6, 2026 (1.7 min). Question: our TANL on ViT-L/14 gives Four-OOD 14.79 (G3b) against 9.52 in TANL's
paper, while the same code reproduces ViT-B/16 (9.89 vs 9.81). Is that a problem of our ViT-L/14 features and text
bank, or specific to TANL? Rule fixed before the run: if NegLabel and MCM land on their published ViT-L/14 numbers,
the features and text bank are right and the gap is TANL's own.

* `d1_runs.csv`: Four-OOD, order 0 (NegLabel and MCM are memoryless), both conventions.
* `d1_info.json`: zero-shot top-1 on OpenOOD's 5k ImageNet ID-val images (ViT-B/16 67.26%, ViT-L/14 73.86%) and the
  text banks (69,554-word corpus, 10,000 NegLabel negatives mined with ViT-B/16 for both backbones, as NegLabel's code
  does). `D1_run.log`, `encode_l14.log`, `NOTES.md`: logs and run notes (iNaturalist, Textures, Textures-all,
  OpenImage-O and its validation split were encoded locally with ViT-L/14 on the ViT-B/16 image lists).

| Method | Backbone | Convention | iNat | SUN | Places | Textures | Four-OOD | Published | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCM | ViT-B/16 | ID+ | 34.38 | 38.48 | 47.91 | 58.65 | 44.86 | 42.74 | +2.12 |
| MCM | ViT-L/14 | ID+ | 31.57 | 26.36 | 35.08 | 58.17 | 37.80 | 38.17 | -0.37 |
| NegLabel | ViT-B/16 | ID+ | 1.80 | 20.13 | 32.76 | 42.22 | 24.23 | 25.40 | -1.17 |
| NegLabel | ViT-L/14 | ID+ | 1.77 | 23.61 | 33.81 | 42.80 | 25.50 | 24.81 | +0.69 |
| TANL, released script's settings | ViT-L/14 | OOD+ | 0.58 | 4.60 | 24.74 | 28.16 | 14.52 | 9.52 | +5.00 |
| TANL, paper's settings (G3b) | ViT-L/14 | OOD+ | 0.37 | 5.67 | 25.47 | 27.65 | 14.79 | 9.52 | +5.27 |

**Reading:** NegLabel and MCM reproduce their published ViT-L/14 numbers (within 0.7 and 0.4 points), so the ViT-L/14
features and text bank are right. The gap is specific to TANL (mostly Textures, 28 vs 13.57, and Places, 25 vs
20.79 in TANL's Table A12) and does not depend on the two sets of TANL settings. Open possibilities: TANL's
ViT-L/14 run may use other settings or negatives mined with ViT-L/14 (TANL re-selects negatives from the corpus at
every batch, so it may depend more on the mining backbone than NegLabel does). ReNeg's gains over our TANL on
ViT-L/14 (-3.20 / -2.81 / -1.29, `results/g3b_vitl14/`) hold either way; the paper reports our reproduction and the
gap.
