# Results

Every number in the paper and in docs/ comes from a file in this folder. Run logs are kept as written (local paths
replaced by `<project>`). docs/REPRODUCE.md shows how to print the tables from these files and how to re-run each
experiment.

| Folder | Produced by | Paper | Status |
| --- | --- | --- | --- |
| `g3_vitb16/` | notebook G3 (`configs/main_vitb16.yaml`), GPU, exact fp16 features | Table 1 (zero-shot, ViT-B/16), few-shot rows on TANL | final |
| `g3b_vitl14/` | notebook G3b (`configs/vitl14.yaml`) | second backbone (ViT-L/14) | final; D1 checks the L/14 TANL baseline |
| `p4_tins/` | notebook P4 (TINS on our features) + `experiments/07_reneg_on_tins.py` | few-shot table (preview) | TINS final; ReNeg-on-TINS rows are replays until G4 |
| `g4_d1/` | notebooks G4 (`configs/fewshot_tins.yaml`) and D1 (`configs/baselines_vitl14.yaml`) | few-shot table; L/14 check | not run yet |
| `baselines/` | Kaggle notebooks 02-04 (oodlab) | baselines: TANL, AdaNeg, NegLabel, MCM reproductions | final |
| `replays/` | the CPU replays (`experiments/`), on 8-bit features | ablations, analyses, supplementary | final |
| `seam1/` | notebooks G1, G1b, P1, P2 + `experiments/08`-`11` | negative results (Seam 1) | closed |

## `g3_vitb16/`, `g3b_vitl14/`

* `g3_runs.csv`: one row per (method, protocol, OOD set, stream order): `fpr95` (OOD positive), `fpr95_idpos`
  (ID positive), `auroc`, `aupr_in`, `aupr_out`, ID accuracy, image counts, `cold_*` (cold-start metrics on the
  first images of the stream), timing. `label` is the run name used in `configs/` (`tanl`, `reneg_balanced_blind`, ...).
* `g3_summary.csv`: group means (near, far, Four-OOD, Four-OOD 45k) as printed by the notebook.
* `g3_check.json`: the GPU-vs-CPU check run before the table (identical metrics), library versions, timing.
* `G3_run.log` / `G3b_run.log`: full logs. `g3_vitb16/NOTES.md`: run notes (hardware, the ViT-L/14 cache completion,
  the 5,002 matched labelled images). `g3_report.txt`: the tables with paired differences. `encode_l14.log`: the
  ViT-L/14 features of iNaturalist, Textures, Textures-all and OpenImage-O encoded locally with the ViT-B/16 image
  lists (the Kaggle L/14 cache lacked them).

## `p4_tins/`

* `tins_runs.csv`, `tins_summary.csv`: TINS (our re-implementation, 5 labelled images per class) on every stream,
  both conventions, with the number of inverted images and the inversion time.
* `tins_traces_1.zip`: per-image TINS traces of all 27 streams (`<protocol>__<set>__s<order>.npz`: `score`, the
  score before the batch's own inversions `pre`, inversion candidates `cand`, `kept`, stream `rows`, `is_ood`). The
  input of `experiments/07_reneg_on_tins.py`.
* `shot_mask.npy`: which of Four-OOD's 50k ID images are labelled images (5,002 matches of OpenOOD's 5,000 ID-val
  images at cosine > 0.999).
* `reneg_on_tins_replays.jsonl`: ReNeg on TINS and on TANL+TINS, replayed (= `replays/tins_replay.jsonl`).
* `p4_report.txt`: the pre-registered pass rule (PASS) and the tables; `fpr95_conventions_vs_tins.txt`: TANL,
  ReNeg and TINS side by side in both conventions. `NOTES.md`, `P4_run.log`: run notes and log.

## `g4_d1/`

Empty until notebooks G4 and D1 run (their code is in `notebooks/4_evaluation/` and `configs/`).

## `baselines/`

Kaggle notebooks 02 (TANL: `tanl_runs.csv` every stream, `tanl_summary.csv`, `tanl_acceptance.csv` against the
paper), 03 (TANL's churn: per set, per batch, temporal shift) and 04 (AdaNeg, NegLabel, MCM: runs, summary,
acceptance against the papers, leaderboard). `logs/` holds two set-up logs.

## `replays/`

Rows written by the original analysis scripts (Oct 1-5), read by the `report` / `table` commands of `experiments/`.

| File | Contents | Read with |
| --- | --- | --- |
| `seeds.jsonl` | TANL and the first release's operating points, 3 orders (Oct 2) | `02_replay.py table --file` |
| `idpos.jsonl`, `idpos_report.txt` | the final operating points (agreed admission), few-shot rows, both conventions (Oct 4) | `02_replay.py table --file` |
| `four_fix.jsonl` | Four-OOD fixes: screens and confirmations (Oct 3) | `04_four_ood_fixes.py report --file` |
| `id_prior.jsonl` | few-shot prior and oracle ID models (Oct 3) | `05_fewshot_and_oracle.py report --file` |
| `tins_replay.jsonl` | ReNeg on TINS and on TANL+TINS, tiers 1-3 (Oct 5) | `07_reneg_on_tins.py report --file` |
| `pseudo_tanl.jsonl`, `pseudo_admit.jsonl` | pseudo near-OOD tuning on the TANL base (Oct 2) | json lines |
| `pseudo_admit_rerun.jsonl` | the admission rules re-run with this repository's `experiments/06_pseudo_ood_tuning.py`, plus the agreed rule (Oct 5) | json lines |
| `tanl_eval_*.jsonl` | design tests on order 0: KG mass in TANL's text score, corpus extension, clip margins, the vote (Oct 2) | json lines |
| `s2full_*.jsonl`, `ablate_online.jsonl`, `admit_exp.jsonl`, `dual_exp.jsonl`, `weights_tune.jsonl` | ReNeg on NegLabel's negatives: the ablation ladder and early design tests (Oct 1-2) | json lines |
| `val5k_mask.npy` | the labelled-image mask used by the replays (4,998 rows: no match among the 45k test images) | numpy |
| `plans/` | the original per-stage plan files (merged into `experiments/plans/`) | |
| `logs/` | the printed output of every replay run | |

## `seam1/`

`g1/`, `g1b/` (notebook outputs), `p1/`, `p2/` (probe notes and P2's inputs), `reneg_gen_probe.npz` (P1's
generated-image embeddings), `planb_results.json` (P1 evaluation), `prior_eval.json` (P2 evaluation),
`seam1_rethink.json` (six more bridges), `oracle_prior.jsonl` (ceiling of a solved Seam 1). All failed their
pre-registered rules (docs/NEGATIVE_RESULTS.md).
