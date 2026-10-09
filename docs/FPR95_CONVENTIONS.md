# Two FPR95 conventions

FPR95 is "the false positive rate at 95% true positive rate", but papers in this area disagree on which class is
*positive*, and the two choices give different numbers for the same scores.

| Column in our files | Positive class | Meaning | Used by |
| --- | --- | --- | --- |
| `fpr95` | **OOD** | the share of ID images scored as OOD at the threshold that detects 95% of the OOD images | OpenOOD, TANL, AdaNeg (OpenOOD-VLM code) |
| `fpr95_idpos` | **ID** | the share of OOD images scored as ID at the threshold that keeps 95% of the ID images | MCM, NegLabel, TINS (`get_measures(ID scores, OOD scores)`) |

AUROC is the same in both conventions.

Every evaluation in this repository records both (`oodlab.metrics.openood_metrics`, `reneg.scores.pair_metrics`,
`experiments/replaylab/data.metrics`), and every table states its convention. Our main tables use OpenOOD's
convention, as TANL (the base) does; comparisons with TINS use the ID-positive column.

## Why it matters

* On near-OOD the two differ by up to 12 points for the same scores (TINS on our features: 66.64 OOD-positive vs
  56.63 ID-positive; ReNeg-balanced: 56.57 vs 44.91).
* TINS's Table 1 mixes the two: its own rows are ID-positive, while its TANL row (Four-OOD 9.81) is copied from
  TANL's paper, which is OOD-positive. In one convention TINS's Four-OOD gain over TANL is 2.7 points, not 3.1.
* MCM's and NegLabel's published numbers are ID-positive, so a reproduction compared with them must report
  `fpr95_idpos` (notebook D1 does this for ViT-L/14).

## Where the numbers come from

* G3 (`results/g3_vitb16/g3_runs.csv`) and P4 (`results/p4_tins/tins_runs.csv`) carry both columns per stream.
* `results/replays/idpos.jsonl` holds both conventions for TANL and every operating point on the replays;
  `results/replays/idpos_report.txt` is its table; `results/p4_tins/fpr95_conventions_vs_tins.txt` puts TANL,
  ReNeg and TINS side by side in each convention.
