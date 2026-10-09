# Negative results

What did not work, with the rule each idea had to pass. All of these are reproducible from this repository
(the script is named in each section); the paper reports them because they shaped the method.

## 1 · Translating negative labels into image space (Seam 1): closed

CLIP's text and image embeddings occupy different regions (the modality gap), so a negative *name* is a weak
proxy for the images it stands for. Three routes tried to give each candidate name an image-space prototype before
the stream starts. All failed rules fixed before the run; the method instead grounds names in the test stream.

| Route | Rule | Result | Code / results |
| --- | --- | --- | --- |
| Fitted maps from text to image space (gap shift, identity-anchored ridge, Procrustes, kernel ridge, low-rank), G1 / G1b | beat raw text without hurting ImageNet | every map lowers ImageNet top-1 by 5+ points (67.26 -> 61.5-62.3); in the stream, near FPR95 75-77 vs 68.15 for raw text | notebooks G1, G1b; `results/seam1/g1*/` |
| Generated images (SDXL-Turbo, 4 per name), P1 | NINCO FPR95 gain >= 3 at weight 4 | +1.6; generated prototypes find their own real concept less often than text (NINCO 59% vs 78%, SSB-hard 44% vs 60%) | `experiments/08_seam1_generation.py` |
| Diffusion prior (Kandinsky 2.1, ViT-L/14) with a ViT-L/14 -> ViT-B/16 map, P2 | prior beats text by 10 points at finding the concept; map cosine >= 0.9; NINCO gain >= 3 | screen 1 fails (NINCO 68.8% vs 79.7%; SSB-hard 53.3% vs 66.0%); the map passes (0.91); stream gain -0.56 | `experiments/09_seam1_prior.py` |
| Six training-free bridges on P1's images: re-centring, CLIP-consistency filter, text + image ensembles, relative representations, TIP-X signatures, gap-shifted text | find the real concept more often than text (NINCO 78.1%, SSB-hard 60.0%) | best: signatures 68.8% / 48.7%, text + re-centred 67.2% / 60.0%; nothing beats text | `experiments/10_seam1_bridges.py` |
| Ceiling: real-photo centroids of the exact test concepts as priors (an oracle, not a method) | - | balanced NINCO 50.45 -> 46.26 (w = 4); max 45.78 -> 40.50: even a perfect bridge adds 4-5 points on NINCO and little on SSB-hard for balanced | `experiments/11_solved_seam1_ceiling.py` |

## 2 · Design choices that were tested and dropped (ReNeg on TANL)

| Idea | Result (order 0 unless stated) | Code / results |
| --- | --- | --- |
| KG names in TANL's text score (gated mass added to every term of the activation-aware average) | Textures 11.4 -> 25.6; adding them to only the last 10-50% of the terms trades NINCO for Textures | `results/replays/tanl_eval_full.jsonl`, `tanl_eval_frac.jsonl`; ablation `kg_in_text_score` |
| KG names appended to TANL's corpus | NINCO 58.35 (no gain alone); with the image score 45.66 but Textures 12.61 | `results/replays/tanl_eval_ext_first.jsonl` |
| Clipped penalty without the base's vote | NINCO 49.92, Textures 11.14, SUN 4.19, Places 22.05; without the vote, 1,344 names admitted on SUN that caught mostly ID images | `results/replays/tanl_eval_kgclip.jsonl`, `tanl_eval_vote*.jsonl`; ablation `no_vote` |
| Clip margins 1 / 2 / 4 | NINCO 53.1 / 53.8 / 55.1 for Places 21.15 / 20.94 / 20.74: the same trade-off as the operating points | `results/replays/tanl_eval_margin.jsonl` |
| Re-weighting the ID and OOD terms of the image score (pseudo tasks) | the default 1 / 1 was best (objective 101.32 vs 106.9-116.4 for the other weights) | `results/replays/weights_tune.jsonl` |

## 3 · Four-OOD fixes that did not survive (Oct 3)

The first release lost 0.32 FPR95 on Four-OOD in every order. Screened on one order (Places / Textures-all / SUN /
NINCO / Textures / OpenImage-O; baseline 21.39 / 15.57 / 3.97 / 51.01 / 11.13 / 34.47; TANL 20.67 / 15.13 / 3.60 /
57.01 / 11.35 / 39.22):

| Variant | Result | Verdict |
| --- | --- | --- |
| **agreed admission** (softmax and TANL's mask) | 21.26 / 14.53 / 3.55 / 51.23 / 9.27 / 34.31 | adopted; confirmed on 3 orders: -3.30 / -2.45 / -0.36 |
| TANL's mask alone | 20.85 / 13.66 / 3.28 / 53.46 / 8.97 / 37.10 | better Four-OOD, gives back near-OOD |
| vote-filtered visual prototypes | 21.09 / 15.34 / 3.96 / 52.52 / 10.93 / 36.60 | no fix |
| centred image space | 23.32 / 19.16 / 7.08 / 48.33 / 14.22 / 31.21 | worse on scenes |
| soft text-catch weight | 20.80 / 15.48 / 3.49 / 54.91 / 11.34 / 37.73 | gives back near-OOD |
| class-calibrated penalty | 21.01 / 15.57 / 3.88 / 52.28 / 10.78 / 35.32 | no fix |
| k_vis = 8 | 20.50 / 14.71 / 3.39 / 53.46 / 10.93 / 38.06 | gives back near-OOD |
| KG pool + NegLabel's 10k words | no gain | dropped |
| on top of agreed: image-prototype agreement, KG-admitted / KG-caught / penalised exclusion, p_min 0.7, lagged admission | each trades scene and texture sets for near-OOD (e.g. KG-caught exclusion NINCO 46.14, Places 23.14) | dropped |

`experiments/04_four_ood_fixes.py report --screens --file results/replays/four_fix.jsonl` prints the full table.

## 4 · Open problem: purity of the ID image prototypes

With an oracle that admits only true ID images, with their true labels, to the ID prototypes, max reaches near
39.49 (NINCO 31.72) against 51.07 with the realistic admission (full pool, order 0); with CLIP's own labels instead
of the true ones it still reaches 40.97 (purity is the lever, not labels). Every label-free signal we tried that removes near-OOD images from the ID prototypes (KG catching,
the KG penalty, lag) also removes hard ID images and costs scene and texture sets
(`experiments/05_fewshot_and_oracle.py`).

## 5 · Limits of the pseudo-OOD tuning

Held-out ImageNet classes as pseudo near-OOD cannot see the scene trade-off of Four-OOD, and TANL rates held-out
ImageNet classes as ID, so the voted modes show no pseudo-near gain (`experiments/06_pseudo_ood_tuning.py`). Every
ID-admission rule lands within about one point of the others on the pseudo tasks (objective = pseudo-near +
validation-far FPR95; TANL 122.71): balanced 120.61 with CLIP's softmax alone, 121.32 with the agreed rule, 122.41
with TANL's mask alone; max 113.99 / 114.30 / 114.45 (the agreed-rule rows were added on Oct 5, after the fact). The
agreed admission was therefore chosen on the benchmark streams after the mechanism analysis, and the paper says so.

## 6 · Our ViT-L/14 TANL baseline

The same code reproduces TANL on ViT-B/16 (Four-OOD 9.89 vs 9.81) but gives 14.79 on ViT-L/14 against 9.52 in
TANL's paper (worst on SUN, Places and Textures). D1 (`configs/baselines_vitl14.yaml`) compares NegLabel and MCM
with their published ViT-L/14 numbers to tell a feature or text-bank problem from a TANL-specific one. ReNeg's
gain over our TANL holds on ViT-L/14 either way (-3.20 / -2.81 / -1.29).
