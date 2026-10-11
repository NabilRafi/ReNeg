# G4: ReNeg on TINS and on TANL + TINS (the few-shot table)

Notebook `notebooks/4_evaluation/G4_reneg_on_tins.ipynb` (command-line equivalent: `configs/fewshot_tins.yaml`),
run on Oct 6, 2026 on an RTX 4060 (1 h 59 min for 27 streams), reneg 0.2.12. One pass per stream runs TINS (our
re-implementation, 5 labelled images per class) and TANL once, and nine ReNeg heads on top (`src/reneg/multi.py`).
OpenOOD v1.5 and Four-OOD with the labelled images removed from the stream (`four_ood_45k`: 44,998 ID images; the
same 5,002-image mask as `results/p4_tins/shot_mask.npy`), three stream orders.

* `g4_runs.csv`: one row per (head, protocol, OOD set, order): `fpr95` (OOD positive), `fpr95_idpos` (ID positive),
  `auroc`, image counts, TINS's inverted images and time. Tables: `python scripts/summarize.py
  results/g4_tins/g4_runs.csv --ref tins` (or `--ref tanl+tins`).
* `g4_summary.csv`: group means as printed by the notebook. `G4_run.log`, `NOTES.md`: log and run notes.
* The per-image scores of every head (`g4_scores_1..4.zip`, 73 MB) are not in the repository.

## Result (mean of three orders; FPR95 ID positive / OOD positive; paired difference [min, max] over orders)

| Method | near | far | Four-OOD 45k |
| --- | --- | --- | --- |
| TANL (no labels) | 53.47 / 60.05 | 16.98 / 16.98 | 8.78 / 9.49 |
| ReNeg-balanced on TANL | 41.00 / 55.86 | 11.44 / 14.36 | 6.90 / 8.63 |
| TINS | 56.63 / 66.64 | 11.79 / 15.20 | 6.80 / 7.63 |
| ReNeg-balanced on TINS | 48.34 / 63.85 | 9.95 / 13.62 | 6.03 / 7.05 |
| vs TINS, ID+ (pre-registered) | -8.29 [-8.70, -7.67] | -1.84 [-2.02, -1.67] | -0.77 [-0.86, -0.66] |
| vs TINS, OOD+ | -2.79 [-2.86, -2.71] | -1.58 [-1.75, -1.28] | -0.57 [-0.79, -0.42] |
| TANL + TINS (mean log-odds) | 50.62 / 59.50 | 11.55 / 12.79 | 5.57 / 6.34 |
| ReNeg-balanced on TANL + TINS | **39.32** / 56.00 | 9.58 / 11.89 | **5.32** / **6.24** |
| vs TANL + TINS, ID+ | -11.30 [-11.68, -11.00] | -1.97 [-2.07, -1.89] | -0.26 [-0.30, -0.20] |
| vs TANL + TINS, OOD+ | -3.50 [-3.78, -3.14] | -0.90 [-0.95, -0.84] | -0.11 [-0.25, +0.09] |
| ReNeg-max on TANL + TINS | 39.57 / **50.43** | **8.68** / **9.60** | 5.61 / 6.99 |

**Pre-registered pass rule** (ReNeg-balanced on TINS better than TINS in ID-positive FPR95 on all three groups, near
and far in every order): **PASS**, as in the replays (`results/p4_tins/`).

Reading:

* Consistency: ReNeg-balanced on TANL equals G3's 5-shot row to two decimals (55.86 / 14.36 / 8.63, OOD positive),
  and on OpenOOD v1.5 the heads match the 8-bit replays of `experiments/07_reneg_on_tins.py` to a mean absolute
  difference of 0.06-0.08 FPR95 per stream (TINS itself identically). The Four-OOD numbers differ slightly from the
  replays because G4 removes the labelled images from the stream, the replays only from the metric.
* The size of the near-OOD gain depends on the convention: on TINS it is 8.3 points in ID-positive FPR95 but 2.8 in
  OpenOOD's convention. This is consistent with ReNeg pushing down the bulk of the near-OOD images (the ID-positive
  metric counts the OOD images above the ID threshold) more than the most ID-like 5% (OpenOOD's metric places its
  threshold among them). Both conventions should be reported.
* TANL and TINS are complementary: averaging their log-odds is already a strong base (Four-OOD 5.57 / 6.34). ReNeg
  adds 11.3 near-OOD points (ID+) on top of it; on Four-OOD its contribution there is small (-0.26 ID+, every
  order; -0.11 OOD+, not every order).
* ReNeg-max trades Four-OOD for near- and far-OOD here too (6.99 vs 6.34 OOD+ on TANL + TINS), as on TANL (G3).
* Blind and full pool agree within 0.6 everywhere; without the labelled images in ReNeg's ID model, balanced on TINS
  is better on Four-OOD (5.87 vs 6.03 ID+) but much worse on near-OOD (54.32 vs 48.34).
