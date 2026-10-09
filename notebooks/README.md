# Notebooks

The notebooks that produced the results, in the order they were run. They are generated from
`tools/build_kaggle_notebooks.py` (00-04) and `tools/build_colab_notebooks.py` (the rest), so every notebook of a
family shares the same set-up cell; edit the builders, not the `.ipynb` files. Logic lives in `src/` (the notebooks
only call it), and the same code runs without notebooks through `scripts/` (docs/REPRODUCE.md).

How to run them: docs/KAGGLE_GUIDE.md (00-04, on Kaggle) and docs/COLAB_GUIDE.md (the rest, on Colab or a local
GPU). Both families have a smoke test on fake data: `python tools/smoke_kaggle_notebooks.py` and
`tools/colab_smoke/` (docs/REPRODUCE.md, section 1).

| Folder | Notebook | What it does | Result |
| --- | --- | --- | --- |
| `0_baselines_kaggle/` | `00_setup_check` | GPU, tests, ImageNet layout, CLIP zero-shot sanity check | PASS table |
| | `01_build_feature_cache` | encode the 12 image sets once with CLIP; TANL's text bank | the feature cache |
| | `02_reproduce_tanl` | TANL on Four-OOD and OpenOOD v1.5, 3 orders, against the paper | `results/baselines/02_*` |
| | `03_churn_diagnostic` | how fast TANL's selected negatives change; temporal shift | `results/baselines/03_*` |
| | `04_reproduce_adaneg` | AdaNeg (fp32 and bit-exact fp16), NegLabel, MCM | `results/baselines/04_*` |
| `1_setup_colab/` | `C0_transfer_from_kaggle` | Kaggle -> Google Drive (code, cache, results) | - |
| | `C1_colab_setup_check` | Colab reproduces Kaggle (tests, cache, zero-shot, TANL within 0.2) | - |
| `2_seam1_transport/` | `G1_transport_gate` | Seam 1: fitted text-to-image maps vs raw text | `results/seam1/g1/` (failed) |
| | `G1b_transport_diagnostics` | corrected re-test with an oracle; optional analysis pack | `results/seam1/g1b/` (failed) |
| | `P1_generation_probe` | SDXL-Turbo images as prototypes for names | `results/seam1/` (failed) |
| | `P2_prior_probe` | Kandinsky 2.1 prior + ViT-L/14 -> B/16 map | `results/seam1/p2/` (failed) |
| `3_exports/` | `G2a_pool_and_streams` | KG pool text embeddings; 8-bit OpenOOD v1.5 streams | inputs of `experiments/` |
| | `G2b_textbank_and_four_ood` | TANL's corpus and the Four-OOD streams, 8-bit | inputs of `experiments/` |
| `4_evaluation/` | `G3_final_eval` | **Table 1**: ReNeg on TANL, ViT-B/16, 3 orders, + 5-shot rows | `results/g3_vitb16/` |
| | `G3b_final_eval_vitl14` | the same on ViT-L/14 | `results/g3b_vitl14/` |
| | `P4_tins_probe` | TINS on our features, per-image traces | `results/p4_tins/` |
| | `D1_vitl14_check` | NegLabel / MCM vs their published ViT-L/14 numbers | pending |
| | `G4_reneg_on_tins` | **few-shot table**: TINS + TANL + nine ReNeg heads | pending |

The notebooks are stored without outputs; the outputs of each run are in `results/` (logs included).
