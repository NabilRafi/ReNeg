# P4 (TINS probe) — run notes (2026-10-05)

Files the notebook asks for: `tins_runs.csv`, `tins_summary.csv`, `shot_mask.npy`, `tins_traces_1.zip`
(all 27 traces fit in one zip). Full log: `P4_run.log`.

## How it was run

- Locally on an RTX 4060 (8 GB), torch 2.5.1+cu121, Python 3.12, Windows 11; the notebook's cells executed unchanged
  through its non-Colab path. oodlab 0.2.1, reneg 0.2.11 (installed from `uploads/ReNeg_Colab_bundle.zip` by the
  set-up cell).
- 5,000 labelled ID images, 1,000 classes (5 per class); zero-shot accuracy on them 67.3%.
  5,002 matched inside Four-OOD's 50k set (same as in G3).
- Static negatives: 58,101 nouns / 11,453 adjectives -> 46,508 / 4,057 pass separation -> 1,840 nouns + 160 adjectives.
- 27 streams (9 sets x 3 orders), 1 h 48 min total, about 42 ms per inverted image. The notebook's 5.2 h estimate is
  an upper bound (it assumes every OOD image is inverted).
- The first attempt died once with no traceback (exit 127) in cell 3, before anything was saved; the re-run passed
  the same step and everything after it. Treated as a one-off.

## Results (ViT-B/16, mean of 3 stream orders)

| | FPR95 (OpenOOD, OOD positive) | FPR95 (ID positive, TINS's code) | AUROC |
|---|---|---|---|
| OpenOOD v1.5 near (SSB-hard, NINCO) | 66.64 | 56.63 | 82.20 |
| OpenOOD v1.5 far (iNat, Textures, OI-O) | 15.20 | 11.79 | 97.19 |
| Four-OOD (50k ID) | 7.42 | 6.79 | 98.56 |
| Four-OOD 45k (shots removed) | 7.45 | 6.84 | 98.55 |

Per order (OpenOOD convention): near 66.37 / 67.52 / 66.02; far 15.47 / 14.81 / 15.30; Four-OOD-45k 7.24 / 7.31 / 7.80.

Sanity check against TINS's paper (16 shots, ID-positive convention): Four-OOD 6.72 (here 6.84 with 5 shots, 45k),
near 57.88 (here 56.63), far 12.99 (here 11.79). The re-implementation reproduces the paper's level with 5 shots.

## Against the G3 rows (same features, OpenOOD convention, FPR95)

| method | near | far | Four-OOD-45k |
|---|---|---|---|
| TANL | 60.05 | 16.98 | 9.49 |
| ReNeg balanced, blind, 5-shot | 55.86 | 14.36 | 8.63 |
| ReNeg max, blind, 5-shot | **50.01** | **11.48** | 9.24 |
| TINS, 5-shot | 66.64 | 15.20 | **7.45** |

- TINS is the best on Four-OOD (Places 16.1 vs about 21 for TANL/ReNeg) and the worst on near-OOD
  (NINCO 68.6, SSB-hard 64.7). Its near-OOD gap between the two FPR conventions is large (66.6 vs 56.6), so tables
  must use one convention throughout.
- So ReNeg on top of TINS (the replay this probe enables) has a clear target: keep TINS's Four-OOD and fix its
  near-OOD.
