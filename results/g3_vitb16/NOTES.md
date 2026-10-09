# G3 / G3b results — run notes (2026-10-04)

Files requested by the notebooks: `g3_final_b16/` and `g3_final_l14/`, each with `g3_runs.csv`, `g3_summary.csv`,
`g3_check.json`. Full logs: `G3_run.log`, `G3b_run.log`, `encode_l14.log`.

## How it was run (differences from GUIDE_COLAB.md)

- **Locally, not on Colab**: RTX 4060 (8 GB), torch 2.5.1+cu121, Python 3.12, Windows 11.
  The notebooks were executed cell by cell, unchanged, through their own non-Colab path
  (`RENEG_DRIVE_ROOT`, `RENEG_OODLAB_HOME`). Code: oodlab 0.2.1, reneg 0.2.10 (auto-installed from
  `uploads/ReNeg_Colab_bundle.zip` by the set-up cell).
- **ViT-L/14 cache completed locally.** The L/14 cache from Kaggle notebook 01 lacked `inaturalist`, `textures`,
  `textures_all`, `openimage_o`, `openimage_o_val`, so G3b would have skipped both protocols. They were encoded
  with oodlab's own `encode_manifest`, using the **ViT-B/16 manifests' exact image lists** (same keys, order,
  labels). iNaturalist came from OpenOOD's Drive file (`DRIVE_IDS["inaturalist"]`, all 10,000 keys matched);
  DTD and OpenImage-O from local copies (all keys matched).
  Sanity check: 512 Textures images re-encoded locally with ViT-B/16 vs the Kaggle B/16 cache:
  cosine min 0.99906, mean 1.00001.
- GPU/CPU check passed on both backbones (identical FPR95/AUROC), so ReNeg ran on CUDA.
- Few-shot: 5,002 shot images were matched in the 50k set (expected ~5,000; probably near-duplicate val images
  under the cosine > 0.999 rule); zero-shot accuracy on them 67.2%.

## Headline (FPR95 %, lower is better; mean of 3 stream orders)

### ViT-B/16

| method | OpenOOD near | OpenOOD far | Four-OOD | Four-OOD 45k |
|---|---|---|---|---|
| TANL | 60.05 | 16.98 | 9.89 | 9.49 |
| balanced, blind | 56.57 | 14.66 | **9.61** | — |
| balanced, full | 56.39 | 14.72 | 9.68 | — |
| max, blind | 52.76 | 12.73 | 10.56 | — |
| max, full | 51.18 | 12.81 | 10.65 | — |
| safe, blind | 58.50 | 16.43 | 9.96 | — |
| balanced, blind, 5-shot | 55.86 | 14.36 | — | **8.63** |
| max, blind, 5-shot | **50.01** | **11.48** | — | 9.24 |

TANL reproduces the paper (Four-OOD 9.89 vs 9.81; near 60.05 vs 60.06; far 16.98 vs 17.21).

### ViT-L/14

| method | OpenOOD near | OpenOOD far | Four-OOD |
|---|---|---|---|
| TANL | 54.71 | 16.87 | 14.79 |
| balanced, blind | 51.52 | 14.06 | 13.50 |
| balanced, full | 50.96 | 14.10 | 13.49 |

## Observations

1. **balanced-blind beats TANL on every group mean, on both backbones**, and on every one of the 3 stream orders
   (e.g. B/16 Four-OOD per order 9.79/9.60/9.45 vs TANL 10.00/9.71/9.96). The Four-OOD margin on B/16 is small
   (−0.28); the OpenOOD margins are larger (near −3.5, far −2.3). Only B/16 Places is a tie (21.54 vs 21.36).
2. **blind ≈ full** everywhere (≤ 1.6 points on any group), so the gains do not come from benchmark class names in
   the KG pool.
3. **max** is the strongest on OpenOOD (near 52.76, far 12.73) but is **worse than TANL on Four-OOD**
   (10.56 vs 9.89; Places 24.36 vs 21.36) — consistent with the notebook's description.
4. **safe** is the weakest ReNeg variant and roughly ties TANL on Four-OOD (9.96 vs 9.89).
5. **5-shot** helps both variants; balanced 5-shot gives the best Four-OOD-45k (8.63 vs TANL 9.49).
6. TANL is notably worse on ViT-L/14 than on ViT-B/16 for Four-OOD (14.79 vs 9.89), driven by Textures
   (27.65 vs 14.00) and SUN — worth checking against TANL's reported L/14 numbers before the paper uses them.
7. Minor: `reneg/reneg_tanl.py` docstring cites TANL as CVPR 2026; it is CVPR 2025.
   *(Correction added later: TANL is CVPR 2026, arXiv 2603.25250, "CVPR 2026 main track"; the docstring is right.)*
