# Experiment log (Sep 27 to Oct 5, 2026)

Every step in the order it was run, with the pass rule fixed **before** the run and the decision taken. Failed steps
are kept: they are the negative results the paper reports (docs/NEGATIVE_RESULTS.md).

Benchmarks: OpenOOD v1.5 (ID ImageNet test 45k; near = SSB-hard, NINCO; far = iNaturalist, Textures, OpenImage-O)
and Four-OOD (ID ImageNet val 50k; iNaturalist, SUN, Places, Textures). CLIP ViT-B/16 unless stated. FPR95 in
OpenOOD's convention (OOD positive) unless marked ID+ (docs/FPR95_CONVENTIONS.md). "Local" steps ran on a CPU on
8-bit exports of the features (`experiments/`); their numbers match the GPU runs to about 0.2 FPR95 per set.

## Baselines (Kaggle, oodlab)

| Date | Step | What | Result | In this repository |
| --- | --- | --- | --- | --- |
| Sep 27-30 | 00, 01 | Set-up check; encode the 12 image sets once (feature cache) | Done | `notebooks/0_baselines_kaggle/`, `scripts/build_cache.py` |
| Sep 30 | 02 | Reproduce TANL, 3 orders; pass if within 1 point (Four-OOD) / 1.5 (near, far) of the paper | PASS: Four-OOD 9.89 vs 9.81, near 60.05 vs 60.06, far 16.98 vs 17.21 (paper configuration, fp32); the released script's configuration gives 10.75 | `results/baselines/02_reproduce_tanl/` |
| Sep 30 | 03 | Churn diagnostic: how fast TANL's selected negatives change | Done (per-set and temporal-shift tables) | `results/baselines/03_churn_diagnostic/` |
| Sep 30 | 04 | Reproduce AdaNeg, NegLabel, MCM | AdaNeg Four-OOD 17.35 (fp16) vs 18.92 published (the repository's own log shows 17.95), near 67.29 vs 67.51; NegLabel 24.96; MCM 37.68 | `results/baselines/04_reproduce_adaneg/` |

## ReNeg

| Date | Step | Question | Pass rule (fixed before the run) | Result | Decision | In this repository |
| --- | --- | --- | --- | --- | --- | --- |
| Sep 30 | C0, C1 | Move cache and code from Kaggle to Drive; does Colab reproduce Kaggle? | All checks pass | PASS | Start G1 | `notebooks/1_setup_colab/` |
| Sep 30 | G1 | Seam 1: does a fitted text-to-image map (R1 gap shift, R2 ridge, B1 Procrustes, B2 kernel ridge) beat raw text (R0)? | A map must beat R0 on the ImageNet self-check and carry over to OOD concepts | **FAIL**: both parts chose R0 (gain 0) | Run the corrected G1b | `notebooks/2_seam1_transport/G1*`, `results/seam1/g1/` |
| Sep 30 | G1b | Corrected diagnostics (per-image prototypes, image-space temperature, oracle) | A map must help without hurting ImageNet | **FAIL**: every map lowers ImageNet top-1 by 5+ points (R0 67.26 vs 61.5-62.3); fused into the stream, near FPR95 75-77 vs 68.15 for R0; the static far gain came from ID image prototypes, not the map | Fitted maps dropped; Plan B next; Seam 2 (KG) continues | `results/seam1/g1b/` |
| Oct 1 | KG pool, G2a | 14,526 WordNet names around the ImageNet classes (648 clusters; blind pool without SSB-hard / NINCO names: 13,878); 8-bit stream export | Built | TANL's corpus has no animal or food word lists: 3,406 KG names (2,330 animals, 1,076 foods) are new to it | Local analysis starts | `src/reneg/resources/kg_pool.json`, `scripts/export_features.py` |
| Oct 1 | P1 | Plan B: do SDXL-Turbo images (4 per name) give KG names a usable image prototype? | NINCO FPR95 gain of 3+ points at weight 4 | **FAIL**: +1.6; generated prototypes find their own real concept less often than text (NINCO 59% vs 78%, SSB-hard 44% vs 60%) | Negative result; P2 next | `experiments/08_seam1_generation.py`, `results/seam1/` |
| Oct 1-2 | local | ReNeg on NegLabel's 10,000 negatives (order 0) | Exploration | KG names worth about 11 near-OOD points over NegLabel's own words as the pool (near 54.62 vs 65.76; NegLabel alone 68.15) | Move ReNeg onto TANL | `configs/ablation_neglabel_base.yaml`, `results/replays/s2full_*.jsonl` |
| Oct 2 | G2b | Export TANL's text bank and the Four-OOD streams | Built | Done | ReNeg on TANL | `scripts/export_features.py` |
| Oct 2 | local | ReNeg on TANL: where the KG names enter | Exploration (test sets, order 0, reported as such) | KG mass in TANL's text score wrecked Textures (11.4 to 25.6) | KG names only through visual prototypes; operating points balanced / max / safe | `results/replays/tanl_eval_*.jsonl` |
| Oct 2 | local | Settings on pseudo near-OOD tasks (held-out ImageNet classes + OpenImage-O val), no benchmark OOD data | Exploration | ID admission rules within about 1 point; the voted modes show no pseudo-near gain | softmax >= 0.5 kept | `experiments/06_pseudo_ood_tuning.py`, `results/replays/pseudo_*.jsonl` |
| Oct 2 | ViT-L/14, P2 | Plan B2: Kandinsky 2.1 diffusion prior + ViT-L/14 to B/16 image map | Screen 1: prior finds its concept more often than text (+10); screen 2: map cosine >= 0.9; stream: NINCO gain 3+ at weight 4 | **FAIL**: screen 1 fails (NINCO 68.8% vs 79.7%), screen 2 passes (0.91), stream gain -0.56 | Seam 1 closed | `experiments/09_seam1_prior.py`, `results/seam1/p2/` |
| Oct 2 | local | Three stream orders for TANL and the operating points | Must-bar: never worse than TANL on any benchmark | Balanced: near -3.33, far -1.84, Four-OOD **+0.32** (worse in every order) | Diagnose Four-OOD | `experiments/02_replay.py`, `results/replays/seeds.jsonl` |
| Oct 3 | local | Four-OOD diagnosis and fixes | Same must-bar | The cost came from OOD look-alikes in the ID image prototypes; agreed admission (CLIP softmax >= 0.5 AND TANL's confident-ID mask) gives balanced -3.30 / -2.45 / -0.36, better in every order | Agreed admission adopted (v0.2.10) | `experiments/03_*.py`, `04_*.py`, `results/replays/four_fix.jsonl` |
| Oct 3 | local | Few-shot setting; oracle ID models; Seam 1 re-analysis | Exploration | 5 labelled images per class help every group; oracle shows ID-prototype purity is the main lever; six more bridges do not beat text; a perfect bridge adds little | Few-shot rows added; Seam 1 stays closed | `experiments/05_*.py`, `10_*.py`, `11_*.py` |
| Oct 4 | G3 | Final table on the exact features (GPU), 3 orders | Must-bar on every group, every order | **PASS**: balanced 56.57 / 14.66 / 9.61 vs TANL 60.05 / 16.98 / 9.89, better in every order; matches the replays to about 0.2 | Main table done | `configs/main_vitb16.yaml`, `results/g3_vitb16/` |
| Oct 4 | G3b | Same on ViT-L/14 | Same | Balanced beats TANL (-3.20 / -2.81 / -1.29) | Our TANL L/14 gives 14.79 on Four-OOD vs 9.52 in TANL's paper: D1 built to check | `configs/vitl14.yaml`, `results/g3b_vitl14/` |
| Oct 4 | local | TINS's reporting convention | - | TINS reports ID+ FPR95. In ID+, balanced (no labels) 45.27 / 12.27 / 7.82 vs TINS's paper 57.88 / 12.99 / 6.72 | Compare with TINS in its own convention | `results/replays/idpos.jsonl` |
| Oct 4-5 | P4 | Does ReNeg on top of TINS (labelled images + test-time backprop) beat TINS? | ReNeg-balanced on TINS better than our TINS run on near, far and Four-OOD 45k (ID+, mean of 3 orders), near and far in every order | **PASS**: -8.32 / -1.84 / -0.64, every order; TINS reproduced at its paper's level with 5 labelled images (Four-OOD 6.79 vs 6.72) | G4 (official few-shot run) and D1 built (v0.2.12) | `notebooks/4_evaluation/P4*`, `experiments/07_reneg_on_tins.py`, `results/p4_tins/` |
| Oct 5 | G4, D1 | Official few-shot table (TINS + TANL + nine ReNeg heads per pass); ViT-L/14 set-up check | - | Built and tested (smoke run, real-cell tests); not run yet | Run on the GPU | `notebooks/4_evaluation/G4*`, `D1*`, `configs/fewshot_tins.yaml`, `configs/baselines_vitl14.yaml` |

## Code versions

The package version of each run is printed at the top of its log. G1 ran reneg 0.1.0, G1b 0.2.0, G3 / G3b 0.2.10,
P4 0.2.11; G4 and D1 were built with 0.2.12. This repository is 0.3.0: the same method code, plus the
command-line pipeline (`reneg/pipeline.py`, `scripts/`) and the cleaned local analysis (`experiments/`). Each
change of method code was checked against the previous version's scores (bit-identical on the CPU, or
equal to float rounding).
