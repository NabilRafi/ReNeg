# P5 · ReNeg on other base detectors: NegLabel, AdaNeg, DDE (Nov 14, 2026)

Question: the method reads only a detector's score, but the paper showed it on TANL and TINS only. Does the same
head, unchanged, improve other detectors?

Everything here is a **CPU replay on the 8-bit feature exports** (`experiments/12_other_bases.py`), float32, three
stream orders. The run on the exact features is `configs/other_bases.yaml` (`scripts/check_other_bases.py` prints
these tables from its `runs.csv`). TANL and TINS rows are the shipped GPU runs (`../g3_vitb16/`, `../g4_tins/`),
shown for comparison.

## Fixed before the first number (`PREREGISTRATION_P5.md`)

* Head: ReNeg-balanced, blind pool, default settings, the packaged generic head (`reneg/multi.py`): it reads the
  base's score as log-odds, votes with a running Otsu threshold on it, and takes "not voted OOD" as the base's ID
  mask. Nothing is tuned per base. AdaNeg's score (at most 1 + lambda) is divided by 1 + lambda first.
* Pass rule per base: ReNeg-balanced on the base has a lower FPR95 than the base on the near, far and Four-OOD group
  means (mean of three orders, OOD positive), with the same sign in every order for near and far. ID positive is
  reported beside it.
* DDE and Hu et al. are re-implementations; each must first reproduce its paper (group means within 1.5 FPR95 on
  near and far, 1.0 on Four-OOD).

## Result

| Base | pass rule, OOD+ | pass rule, ID+ |
|---|---|---|
| NegLabel | **PASS** | PASS |
| AdaNeg | **PASS** | PASS |
| DDE (our re-implementation) | not under the rule: the base misses its paper on far-OOD (below); the head is better on every group in every order | better on every group in every order |

FPR95, OOD positive; change against the base alone [min, max over the three orders]:

| Base detector | Row | Near-OOD | Far-OOD | Four-OOD |
|---|---|---|---|---|
| NegLabel | alone | 68.04 | 23.39 | 24.96 |
| | + ReNeg-balanced | 66.34 (-1.70 [-1.74, -1.62]) | 17.52 (-5.87 [-6.25, -5.52]) | 20.75 (-4.21 [-4.58, -3.72]) |
| | + ReNeg-max | 61.62 (-6.42 [-6.54, -6.33]) | 17.20 (-6.19 [-6.44, -5.93]) | 21.89 (-3.07 [-3.30, -2.86]) |
| | + ReNeg-safe | 67.16 (-0.88 [-1.00, -0.73]) | 20.41 (-2.98 [-3.12, -2.90]) | 23.16 (-1.80 [-1.93, -1.54]) |
| AdaNeg | alone | 66.63 | 16.59 | 17.68 |
| | + ReNeg-balanced | 66.11 (-0.52 [-0.66, -0.41]) | 15.62 (-0.97 [-1.01, -0.92]) | 17.23 (-0.46 [-0.61, -0.26]) |
| | + ReNeg-max | 60.01 (-6.62 [-6.74, -6.55]) | 15.22 (-1.37 [-1.62, -1.20]) | 18.73 (+1.05 [+0.77, +1.42]) |
| | + ReNeg-safe | 66.53 (-0.10 [-0.18, +0.00]) | 16.16 (-0.43 [-0.49, -0.37]) | 17.29 (-0.39 [-0.47, -0.35]) |
| DDE | alone | 59.28 | 17.35 | 9.57 |
| | + ReNeg-balanced | 56.32 (-2.96 [-3.47, -2.46]) | 15.75 (-1.60 [-1.70, -1.50]) | 9.28 (-0.29 [-0.34, -0.25]) |
| | + ReNeg-max | 52.62 (-6.66 [-7.20, -5.93]) | 12.81 (-4.54 [-4.67, -4.47]) | 10.11 (+0.54 [+0.42, +0.74]) |
| | + ReNeg-safe | 58.39 (-0.89 [-1.10, -0.68]) | 17.12 (-0.23 [-0.35, -0.11]) | 9.60 (+0.03 [-0.02, +0.12]) |
| TANL | alone | 60.05 | 16.98 | 9.89 |
| | + ReNeg-balanced | 56.57 (-3.49 [-4.36, -2.83]) | 14.66 (-2.31 [-2.47, -2.11]) | 9.61 (-0.28 [-0.51, -0.11]) |
| | + ReNeg-max | 52.76 (-7.29 [-7.81, -6.88]) | 12.73 (-4.24 [-4.45, -4.05]) | 10.56 (+0.67 [+0.37, +0.82]) |
| | + ReNeg-safe | 58.50 (-1.55 [-2.14, -1.19]) | 16.43 (-0.55 [-0.61, -0.50]) | 9.96 (+0.07 [-0.03, +0.18]) |
| TINS (5 labelled images per class) | alone | 66.64 | 15.20 | 7.63 |
| | + ReNeg-balanced | 63.85 (-2.79 [-2.86, -2.71]) | 13.62 (-1.58 [-1.75, -1.28]) | 7.05 (-0.57 [-0.79, -0.42]) |
| | + ReNeg-max | 56.78 (-9.85 [-9.99, -9.67]) | 10.67 (-4.53 [-4.70, -4.27]) | 7.62 (-0.01 [-0.18, +0.08]) |
| | + ReNeg-safe | 66.01 (-0.63 [-0.68, -0.58]) | 14.75 (-0.44 [-0.63, -0.32]) | 7.25 (-0.38 [-0.49, -0.29]) |

FPR95, ID positive:

| Base detector | Row | Near-OOD | Far-OOD | Four-OOD |
|---|---|---|---|---|
| NegLabel | alone | 75.76 | 24.24 | 24.35 |
| | + ReNeg-balanced | 67.81 (-7.95 [-8.11, -7.70]) | 16.93 (-7.30 [-7.64, -6.80]) | 17.36 (-6.99 [-7.29, -6.67]) |
| | + ReNeg-max | 69.51 (-6.25 [-6.38, -6.04]) | 18.19 (-6.05 [-6.52, -5.22]) | 18.73 (-5.62 [-5.84, -5.51]) |
| | + ReNeg-safe | 64.57 (-11.19 [-11.27, -11.07]) | 17.31 (-6.93 [-7.07, -6.82]) | 17.60 (-6.75 [-6.93, -6.56]) |
| AdaNeg | alone | 73.05 | 15.96 | 13.16 |
| | + ReNeg-balanced | 67.65 (-5.40 [-5.82, -4.94]) | 14.20 (-1.76 [-1.94, -1.67]) | 12.01 (-1.15 [-1.32, -0.98]) |
| | + ReNeg-max | 68.84 (-4.21 [-4.57, -3.70]) | 14.29 (-1.66 [-1.97, -1.38]) | 12.57 (-0.59 [-0.75, -0.33]) |
| | + ReNeg-safe | 66.74 (-6.31 [-6.60, -6.07]) | 14.87 (-1.08 [-1.17, -1.02]) | 11.85 (-1.31 [-1.42, -1.21]) |
| DDE | alone | 54.29 | 16.92 | 8.51 |
| | + ReNeg-balanced | 47.52 (-6.77 [-7.00, -6.65]) | 13.27 (-3.65 [-3.83, -3.38]) | 7.31 (-1.20 [-1.30, -1.08]) |
| | + ReNeg-max | 49.67 (-4.62 [-5.02, -4.34]) | 12.65 (-4.27 [-4.43, -4.02]) | 7.75 (-0.75 [-0.88, -0.69]) |
| | + ReNeg-safe | 48.68 (-5.61 [-6.06, -5.38]) | 16.23 (-0.69 [-0.72, -0.64]) | 8.25 (-0.26 [-0.36, -0.19]) |
| TANL | alone | 53.47 | 16.98 | 9.27 |
| | + ReNeg-balanced | 44.91 (-8.56 [-8.66, -8.44]) | 12.27 (-4.71 [-4.81, -4.64]) | 7.79 (-1.47 [-1.63, -1.25]) |
| | + ReNeg-max | 47.10 (-6.37 [-6.44, -6.31]) | 12.22 (-4.76 [-4.85, -4.69]) | 8.12 (-1.15 [-1.32, -0.84]) |
| | + ReNeg-safe | 46.27 (-7.20 [-7.40, -7.01]) | 14.80 (-2.18 [-2.28, -2.06]) | 8.80 (-0.47 [-0.55, -0.38]) |
| TINS (5 labelled images per class) | alone | 56.63 | 11.79 | 6.80 |
| | + ReNeg-balanced | 48.34 (-8.29 [-8.70, -7.67]) | 9.95 (-1.84 [-2.02, -1.67]) | 6.03 (-0.77 [-0.86, -0.66]) |
| | + ReNeg-max | 48.30 (-8.33 [-8.88, -7.55]) | 9.03 (-2.76 [-2.89, -2.63]) | 6.32 (-0.48 [-0.58, -0.34]) |
| | + ReNeg-safe | 51.05 (-5.58 [-6.23, -5.07]) | 11.25 (-0.54 [-0.62, -0.46]) | 6.35 (-0.46 [-0.49, -0.42]) |

All columns (AUROC included) and the rule for every operating point:

| Detector | Row | near OOD+ | near ID+ | near AUROC | far OOD+ | far ID+ | far AUROC | Four OOD+ | Four ID+ | Four AUROC | rule OOD+ | rule ID+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NegLabel | alone | 68.04 | 75.76 | 75.29 | 23.39 | 24.24 | 94.92 | 24.96 | 24.35 | 94.54 |  |  |
|  | + ReNeg-balanced | 66.34 | 67.81 | 78.71 | 17.52 | 16.93 | 96.38 | 20.75 | 17.36 | 95.86 | PASS | PASS |
|  | + ReNeg-max | 61.62 | 69.51 | 78.90 | 17.20 | 18.19 | 96.34 | 21.89 | 18.73 | 95.52 | PASS | PASS |
|  | + ReNeg-safe | 67.16 | 64.57 | 78.77 | 20.41 | 17.31 | 96.13 | 23.16 | 17.60 | 95.72 | PASS | PASS |
| AdaNeg | alone | 66.63 | 73.05 | 77.22 | 16.59 | 15.96 | 96.56 | 17.68 | 13.16 | 96.65 |  |  |
|  | + ReNeg-balanced | 66.11 | 67.65 | 79.19 | 15.62 | 14.20 | 96.90 | 17.23 | 12.01 | 96.84 | PASS | PASS |
|  | + ReNeg-max | 60.01 | 68.84 | 80.25 | 15.22 | 14.29 | 96.96 | 18.73 | 12.57 | 96.56 | FAIL | PASS |
|  | + ReNeg-safe | 66.53 | 66.74 | 78.60 | 16.16 | 14.87 | 96.76 | 17.29 | 11.85 | 96.88 | FAIL | PASS |
| DDE | alone | 59.28 | 54.29 | 84.37 | 17.35 | 16.92 | 96.13 | 9.57 | 8.51 | 97.92 |  |  |
|  | + ReNeg-balanced | 56.32 | 47.52 | 86.62 | 15.75 | 13.27 | 96.90 | 9.28 | 7.31 | 98.18 | PASS | PASS |
|  | + ReNeg-max | 52.62 | 49.67 | 86.84 | 12.81 | 12.65 | 97.34 | 10.11 | 7.75 | 98.01 | FAIL | PASS |
|  | + ReNeg-safe | 58.39 | 48.68 | 85.77 | 17.12 | 16.23 | 96.36 | 9.60 | 8.25 | 98.07 | FAIL | PASS |
| TANL | alone | 60.05 | 53.47 | 84.53 | 16.98 | 16.98 | 96.23 | 9.89 | 9.27 | 97.82 |  |  |
|  | + ReNeg-balanced | 56.57 | 44.91 | 87.12 | 14.66 | 12.27 | 97.19 | 9.61 | 7.79 | 98.10 | PASS | PASS |
|  | + ReNeg-max | 52.76 | 47.10 | 87.26 | 12.73 | 12.22 | 97.41 | 10.56 | 8.12 | 97.90 | FAIL | PASS |
|  | + ReNeg-safe | 58.50 | 46.27 | 86.37 | 16.43 | 14.80 | 96.69 | 9.96 | 8.80 | 97.99 | FAIL | PASS |
| TINS (5 labelled images per class) | alone | 66.64 | 56.63 | 82.20 | 15.20 | 11.79 | 97.19 | 7.63 | 6.80 | 98.53 |  |  |
|  | + ReNeg-balanced | 63.85 | 48.34 | 85.39 | 13.62 | 9.95 | 97.61 | 7.05 | 6.03 | 98.69 | PASS | PASS |
|  | + ReNeg-max | 56.78 | 48.30 | 86.76 | 10.67 | 9.03 | 98.05 | 7.62 | 6.32 | 98.59 | PASS | PASS |
|  | + ReNeg-safe | 66.01 | 51.05 | 84.04 | 14.75 | 11.25 | 97.32 | 7.25 | 6.35 | 98.63 | PASS | PASS |

Per OOD set, base -> + ReNeg-balanced (mean of three orders, OOD positive), and in how many (set, order) pairs the
head is better:

| Detector | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O | iNaturalist (4) | SUN (4) | Places (4) | Textures (4) | pairs better |
|---|---|---|---|---|---|---|---|---|---|---|
| NegLabel | 75.95 → 75.77 | 60.12 → 56.90 | 1.24 → 0.63 | 34.55 → 23.24 | 34.38 → 28.68 | 1.23 → 0.74 | 21.52 → 17.13 | 38.45 → 35.77 | 38.66 → 29.37 | 27/27 |
| AdaNeg | 73.69 → 73.74 | 59.57 → 58.49 | 0.56 → 0.30 | 18.88 → 17.96 | 30.32 → 28.59 | 0.65 → 0.36 | 10.03 → 9.24 | 33.54 → 33.58 | 26.51 → 25.73 | 22/27 |
| DDE | 61.87 → 61.57 | 56.68 → 51.07 | 0.48 → 0.21 | 10.64 → 9.18 | 40.93 → 37.85 | 0.48 → 0.22 | 3.55 → 3.13 | 20.08 → 20.11 | 14.17 → 13.68 | 25/27 |
| TANL | 62.51 → 62.16 | 57.60 → 50.97 | 0.47 → 0.22 | 11.38 → 9.34 | 39.09 → 34.44 | 0.47 → 0.23 | 3.74 → 3.39 | 21.36 → 21.54 | 14.00 → 13.30 | 25/27 |
| TINS (5 labelled images per class) | 64.68 → 63.72 | 68.60 → 63.98 | 0.20 → 0.13 | 8.24 → 6.38 | 37.15 → 34.35 | 0.19 → 0.12 | 5.08 → 4.32 | 16.80 → 16.56 | 8.43 → 7.22 | 27/27 |

Readings:

* The gain is largest on NegLabel, which has no memory of the stream (far -5.9, Four-OOD -4.2), and smallest on
  AdaNeg (about half a point per group in OpenOOD's convention; SSB-hard and Places do not move), whose negative
  proxies already adapt to the stream.
* As on TANL, the ID-positive convention shows larger gains than OpenOOD's: the head pushes down the bulk of the
  OOD images more than the most ID-like 5%.
* max trades Four-OOD for near-OOD on AdaNeg (+1.05), as it does on TANL; balanced does not.

## DDE: which configuration, and how close to the paper

DDE's score is re-implemented from the official postprocessor (`reneg/dde.py`; `selfcheck` equals the official
recomputation to 1e-6). Four configurations were compared on the base alone (stream order 0, SSB-hard / NINCO,
OOD positive; paper 61.02 / 56.26), before any ReNeg number on the chosen one:

| Configuration | SSB-hard | NINCO | near | gap |
|---|---|---|---|---|
| A. the release as it is (10,000 negatives twice, OOD threshold 0.10, caches 5,000) | - | 66.20 | - | - |
| B. whole 69,554-word corpus, the release's thresholds and caches | 65.68 | 56.74 | 61.21 | +2.57 |
| **C. whole corpus, the paper's settings (lambda_neg 0.25, queue 1,000)** | 62.15 | 55.88 | 59.02 | +0.38 |
| D. 10,000-negative bank, the paper's settings (duplicated / not) | 63.51 / 65.94 | 59.42 / 58.53 | 61.47 / 62.24 | +2.83 / +3.60 |

C was chosen (closest on near-OOD). Over three orders, configuration C does not reproduce the paper inside the tolerance fixed beforehand on every group (table below), so by the rule fixed beforehand it is reported as our re-implementation and ReNeg on it is exploratory:

| Group | ours, OOD+ | paper | gap | tolerance | | ours, ID+ |
|---|---|---|---|---|---|---|
| near | 59.28 | 58.64 | +0.64 | 1.5 | inside | 54.29 |
| far | 17.35 | 15.58 | +1.77 | 1.5 | OUTSIDE | 16.92 |
| four | 9.57 | 9.80 | -0.23 | 1.0 | inside | 8.51 |

Per set, ours (paper): SSB-hard 61.87 (61.02); NINCO 56.68 (56.26); iNaturalist 0.48 (0.47); Textures 10.64 (14.42); OpenImage-O 40.93 (31.84); SUN 3.55 (3.45); Places 20.08 (20.85); Textures, full set, Four-OOD stream 14.17 (14.42 printed for Textures)

Where the far-OOD gap comes from: five of the seven sets are within one point of the paper. Textures is below it
(10.64 vs 14.42): the paper's two group means are both reproduced by one iNaturalist and one Textures number
(0.47 and 14.42 give 15.58 for far-OOD and 9.80 for Four-OOD), so it probably scores the full Textures set in both
benchmarks; on the full set (our Four-OOD stream) our DDE gives 14.17. OpenImage-O is 9 points above the paper (40.93 vs 31.84), and that does not depend on the configuration
(order 0, base alone: A 40.56, B 40.34, C 40.39, D 38.37); our TANL run gives 39.09 on the same images. The cause
is not known (a different OpenImage-O list or ID set in the paper cannot be ruled out).

ReNeg-balanced on this DDE: near -2.96 [-3.47, -2.46], far
-1.60 [-1.70, -1.50], Four-OOD -0.29
[-0.34, -0.25] (OOD positive); ID positive -6.77 /
-3.65 / -1.20.
Seen before the choice (exploratory, configuration B, order 0, four sets): balanced lowered FPR95 on all four
(SSB-hard 65.68 -> 65.36, NINCO 56.74 -> 54.17, iNaturalist 0.42 -> 0.18, Textures 9.79 -> 8.41).

## The base's score must be in single precision

The head reads log-odds. The official NegLabel and AdaNeg code runs in half precision, where a confident score
rounds to exactly 1 and its log-odds say nothing about the ranking among confident images. One exploratory stream
(NINCO, order 0), NegLabel in half precision: base 60.87, + balanced 61.47 (ID positive 68.80 -> 65.69), against
60.12 -> 57.06 in float32. The rule above was fixed for float32 and stays; `configs/other_bases.yaml` runs the
half-precision bases beside the float32 ones (`neglabel_fp16`, `adaneg_fp16`) so that the exact-feature run reports
both. Table 1's NegLabel and AdaNeg rows are the half-precision reproductions; the float32 rows here differ from them
by at most 1.3 FPR95 (NegLabel near 68.04 vs 69.34).

## Hu et al. (ICML 2026): not reproduced, so not used as a base

No code is public (arXiv 2605.26661, the ICML page and a web search, Nov 14). A re-implementation from the paper
(`experiments/replaylab/hu.py`; ambiguous points fixed beforehand in the pre-registration) is far from the published
numbers under both readings of its label space (order 0, OOD positive / ID positive):

| Reading | NINCO | Places | published NINCO / Places |
|---|---|---|---|
| each branch's loss over its own label set | 73.79 / 64.33 | 55.17 / 50.18 | 50.60 / 28.55 |
| one label space | 76.43 / 67.29 | - | |

Stopped after two readings (the rule: a re-implementation must reproduce its paper before ReNeg is put on it).

## Files

* `PREREGISTRATION_P5.md`: the rules, with the two addenda (DDE and Hu et al.; the DDE configuration and the
  precision of the base), each written before the numbers it refers to.
* `other_bases.jsonl`: 324 rows, one per (label, protocol, OOD set, stream order): `neglabel`, `adaneg`, `dde` and
  `balanced|max|safe on <base>`; `fpr95` (OOD positive), `fpr95_idpos`, `auroc`.
* `p5_summary.csv`: group means, paired changes and the rule for every row of the tables above.
* `exploratory.jsonl`: 34 rows that are not under the rule (DDE configurations, the half-precision stream, the
  Hu et al. attempts).
* Reproduce: `python experiments/12_other_bases.py run --base neglabel` (also `adaneg`, `dde`), then `report`.
