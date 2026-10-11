# G4 (ReNeg on TINS) — run notes (2026-10-06)

Files the notebook asks for: `g4_runs.csv`, `g4_summary.csv`, `g4_scores_1..4.zip` (all 27 streams, every head).
Also `shot_mask.npy` and the full log `G4_run.log`.

## How it was run

- Locally, RTX 4060 (8 GB), torch 2.5.1+cu121, Python 3.12, Windows 11; cells executed unchanged through the
  non-Colab path. oodlab 0.2.1, reneg 0.2.12.
- 5,000 labelled images; 5,002 matched inside the 50k Four-OOD set. TINS static negatives rebuilt (identical counts
  to P4: 1,840 nouns + 160 adjectives). 14,526 KG names, 13,878 blind; 9 heads.
- 27 streams, 19:22 -> 21:21 (1 h 59 min). TINS on NINCO s0 gives exactly P4's value (58.14 ID-positive).
- The G3 consistency check in the summary cell did not print (G3's `g3_runs.csv` is not in this Drive drop).
  Checked by hand instead: `balanced on tanl` here = G3's `reneg_balanced_blind_5shot` to two decimals
  (OOD-positive near 55.86, far 14.36, Four-OOD-45k 8.63 in both).

## Pre-registered pass rule (ID-positive FPR95, ReNeg-balanced on TINS minus TINS)

| group | mean | per order | |
|---|---|---|---|
| near | -8.29 | -8.51 / -7.67 / -8.70 | PASS |
| far | -1.84 | -1.84 / -1.67 / -2.02 | PASS |
| Four-OOD 45k | -0.77 | -0.66 / -0.86 / -0.80 | PASS |

**Overall: PASS.**

## Summary (mean of 3 orders; FPR95, lower is better)

| method | near ID+ | far ID+ | 4OOD45 ID+ | near OOD+ | far OOD+ | 4OOD45 OOD+ |
|---|---|---|---|---|---|---|
| tins | 56.63 | 11.79 | 6.80 | 66.64 | 15.20 | 7.63 |
| tanl | 53.47 | 16.98 | 8.78 | 60.05 | 16.98 | 9.49 |
| tanl+tins (no ReNeg) | 50.62 | 11.55 | 5.57 | 59.50 | 12.79 | 6.34 |
| balanced on tins | 48.34 | 9.95 | 6.03 | 63.85 | 13.62 | 7.05 |
| max on tins | 48.30 | 9.03 | 6.32 | 56.78 | 10.67 | 7.62 |
| safe on tins | 51.05 | 11.25 | 6.35 | 66.01 | 14.75 | 7.25 |
| balanced on tins, full pool | 47.75 | 9.94 | 6.04 | 63.76 | 13.58 | 7.05 |
| balanced on tins, no labelled images | 54.32 | 9.85 | 5.87 | 64.43 | 13.40 | 6.71 |
| **balanced on tanl+tins** | **39.32** | 9.58 | **5.32** | 56.00 | 11.89 | **6.24** |
| max on tanl+tins | 39.57 | **8.68** | 5.61 | **50.43** | **9.60** | 6.99 |
| safe on tanl+tins | 42.60 | 11.09 | 5.46 | 58.23 | 12.67 | 6.24 |
| balanced on tanl | 41.00 | 11.44 | 6.90 | 55.86 | 14.36 | 8.63 |

## Things worth a look before writing them up

1. **The convention changes the size of the near-OOD gain a lot.** Balanced on TINS vs TINS: near -8.29 in the
   ID-positive convention (the pre-registered one) but only -2.79 in the OpenOOD (OOD-positive) convention
   (63.85 vs 66.64). TANL's and G3's tables use OOD-positive, so one convention has to be chosen for the paper.
2. **`tanl+tins` without ReNeg is already a strong baseline** (Four-OOD-45k 5.57 ID+ / 6.34 OOD+, better than
   balanced on TINS). ReNeg's own contribution should be measured on that base too: balanced on tanl+tins vs
   tanl+tins is near -11.30 / far -1.97 / Four-OOD -0.25 (ID+), and -3.50 / -0.90 / -0.10 (OOD+).
3. **Best head depends on the group.** balanced on tanl+tins is best on near (ID+) and Four-OOD (both);
   max on tanl+tins is best on far (both) and near (OOD+), but worse than its base on Four-OOD in OOD+
   (6.99 vs 6.34) — the same max/Four-OOD trade-off as in G3.
4. Blind vs full pool: same within 0.6 everywhere (again no benchmark-name leakage effect).
5. Without labelled images, balanced on TINS is slightly *better* on Four-OOD (5.87 vs 6.03 ID+) but much worse on
   near (54.32 vs 48.34).
