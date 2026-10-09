# Running the notebooks (Google Colab or a local GPU)

The paper's GPU results were produced by the notebooks in `notebooks/` (G3, G3b, P4; G4 and D1 are the next
runs), on top of a CLIP feature cache built once on Kaggle (`notebooks/0_baselines_kaggle/`, docs/KAGGLE_GUIDE.md).
The command-line pipeline (`scripts/`, docs/REPRODUCE.md) runs the same code without notebooks; this guide is for
re-running the notebooks exactly as they were run.

The notebooks run on Colab, or on any machine with a GPU through their non-Colab path (G3, G3b and P4 were run that
way on an RTX 4060 under Windows): set `RENEG_DRIVE_ROOT` (the project folder) and `RENEG_OODLAB_HOME` (a local
working folder) before starting Jupyter, and lay out `RENEG_DRIVE_ROOT` as below.

## 0 · What you need

* A Kaggle account with phone verification (ImageNet is mounted there; notebook 01 builds the feature cache).
* A Google account with Colab (Colab Pro is recommended for the GPU runs), or a local GPU.
* The two zips built from this repository:

  ```bash
  python tools/make_kaggle_dataset.py     # dist/oodlab_code.zip        (oodlab + Kaggle notebooks 00-04)
  python tools/make_colab_bundle.py       # dist/ReNeg_Colab_bundle.zip (reneg + the Colab notebooks)
  ```

## 1 · On Kaggle: the feature cache and the baselines

Follow docs/KAGGLE_GUIDE.md: upload `oodlab_code.zip` as a private dataset named `oodlab-code`, then run notebooks
00 (set-up check), 01 (feature cache, about 1-2 hours on a T4), 02 (TANL), 03 (churn diagnostic) and 04 (AdaNeg,
NegLabel, MCM). Their outputs are in `results/baselines/`. For ViT-L/14, run a copy of notebook 01 with
`BACKBONE = "ViT-L/14"` (step 11 below).

## 2 · The project folder

Everything lives in one Drive folder, `My Drive/ReNeg/` (or `RENEG_DRIVE_ROOT`). Notebook C0 creates the sub-folders.

| Folder | What is inside | Filled by |
| --- | --- | --- |
| `uploads/` | zips you upload by hand (the bundle; Kaggle output zips for route C) | you |
| `code/oodlab_code/` | the oodlab package (TANL, AdaNeg, NegLabel, MCM, the cache reader) | C0 |
| `code/reneg_code/` | the reneg package | C0, from the bundle |
| `notebooks/` | the Colab notebooks | C0, from the bundle |
| `notebooks_colab/` | Colab copies of the Kaggle notebooks 00-04 | C0 |
| `oodlab_working/cache/vit-b-16/` | the feature cache: 12 image sets and the text bank, about 0.3 GB | C0, from Kaggle notebook 01 |
| `oodlab_working/results/` | results of every notebook | C0, then every notebook |
| `transfers/` | outputs meant for the local analysis and the paper tables | the notebooks |
| `weights/` | CLIP weights, downloaded once | C1 and later |

1. Open [drive.google.com](https://drive.google.com), **New → New folder** `ReNeg` (in *My Drive*), and inside it
   a folder `uploads`.
2. Upload `ReNeg_Colab_bundle.zip` **as it is (not unzipped)** into `My Drive/ReNeg/uploads/`. Unzip a copy on your
   computer: you need `notebooks/C0_transfer_from_kaggle.ipynb` from it in step 4.

## 3 · Give Colab your Kaggle key

1. On [kaggle.com](https://www.kaggle.com): profile picture → **Settings** → **API** → **Create New Token**.
   A file `kaggle.json` downloads (`{"username":"…","key":"…"}`).
2. On [colab.research.google.com](https://colab.research.google.com): **File → Upload notebook** →
   `C0_transfer_from_kaggle.ipynb`.
3. Click the **key icon** in the left sidebar (*Secrets*) → **Add new secret** twice: `KAGGLE_USERNAME` (the
   username) and `KAGGLE_KEY` (the key). Switch on **Notebook access** for both.

Secrets stay in your Google account. Never paste the key into a cell and never share `kaggle.json`. If Kaggle gives
you a single API token instead, add it as `KAGGLE_API_TOKEN` together with `KAGGLE_USERNAME`.

## 4 · C0: transfer from Kaggle (CPU, about 10 minutes)

1. **Runtime → Change runtime type → CPU**, then **Runtime → Run all**; allow access to Google Drive.
2. Step 3 of the notebook lists the notebooks on your Kaggle account and marks each of the five it needs as
   *found* or *guessed*; step 5 shows `downloaded` or `FAILED`. A notebook's slug is the end of its URL
   (`kaggle.com/code/<user>/<slug>`); put the real slug into `NOTEBOOKS` in step 3 if one fails.
3. The last cell prints PASS or FAIL for each item and `ALL DONE` when everything arrived. C0 can be re-run at
   any time; it copies only what is new (run it again after building the ViT-L/14 cache).

If a download fails: (B) turn the notebook's output into a private Kaggle dataset and put its name into
`EXTRA_DATASETS` in step 3; or (C) download each notebook's output as a zip from Kaggle, upload the zips and
`oodlab_code.zip` into `uploads/`, and run C0 again (it unpacks everything in `uploads/`).

## 5 · C1: set-up check (T4 GPU, 10-20 minutes)

Open `My Drive/ReNeg/notebooks/C1_colab_setup_check.ipynb` from Drive (**Open with → Google Colaboratory**; always
open notebooks from Drive so edits are saved), choose a GPU runtime, **Run all**. Expect PASS on checks 2-7:

| Check | What PASS means |
| --- | --- |
| 2 | oodlab's and reneg's unit tests pass on Colab |
| 3 | the cache has all 12 sets; the text bank has 69,554 words and 10,000 negatives |
| 4 | CLIP zero-shot top-1 on the 5,000 ImageNet validation images is between 60% and 75% |
| 5 | WordNet is available |
| 6 | the CLIP text encoder reproduces the text bank's ID embeddings |
| 7 | TANL on Colab gives the same Four-OOD numbers as the Kaggle run (within 0.2 points) |

## 6 · The notebooks, in the order they were run

Open each from `My Drive/ReNeg/notebooks/` and **Run all**. Every notebook saves as it goes and skips finished
parts after a dropped session; every notebook ends with a cell that flushes Drive (run it before disconnecting).

| Step | Notebook | Question | Runtime | Outputs (`transfers/` or `oodlab_working/results/`) | In this repository |
| --- | --- | --- | --- | --- | --- |
| 6 | `G1_transport_gate` | Seam 1: can a fitted text-to-image map move negative labels into image space? | T4, 5-10 min | `results/G1_transport_gate/` | `results/seam1/g1/` (failed) |
| 7 | `G1b_transport_diagnostics` | Corrected re-test of G1 (per-image prototypes, oracle) | T4, 5-10 min | `results/G1b_transport_diagnostics/`, optional `transfers/analysis_pack/` | `results/seam1/g1b/` (failed) |
| 8 | `G2a_pool_and_streams` | Encode the 14,526-name WordNet pool; export 8-bit OpenOOD v1.5 streams | CPU, 5 min | `transfers/analysis_pack2/` | `scripts/export_features.py` |
| 9 | `P1_generation_probe` | Plan B: SDXL-Turbo images as prototypes for names | A100/L4/T4, 15-40 min | `transfers/analysis_pack2/reneg_gen_probe.npz` | `results/seam1/` (failed) |
| 10 | `G2b_textbank_and_four_ood` | Export TANL's corpus and the Four-OOD streams | CPU, 3 min | `transfers/analysis_pack2/` | `scripts/export_features.py` |
| 11 | Kaggle copy of notebook 01 | ViT-L/14 feature cache (`BACKBONE = "ViT-L/14"`, `LIVE_CHECK = False`, `build_textbank(..., mining_model=load_clip("ViT-B/16", ...))`) | Kaggle GPU, 2-3 h | Kaggle output | `scripts/build_cache.py --backbone ViT-L/14` |
| 12 | `P2_prior_probe` | Plan B2: a diffusion prior (Kandinsky 2.1, ViT-L/14) with an L/14-to-B/16 map | A100/L4/T4, 35-55 min | `transfers/analysis_pack3/` | `results/seam1/p2/` (failed) |
| 13 | `G3_final_eval` | **Main table**: ReNeg on TANL, ViT-B/16, three stream orders, plus 5-shot rows | GPU, 2-4 h | `transfers/g3_final_b16/` | `results/g3_vitb16/` |
| 13b | `G3b_final_eval_vitl14` | The same on ViT-L/14 | GPU, 1-2 h | `transfers/g3_final_l14/` | `results/g3b_vitl14/` |
| 14 | `P4_tins_probe` | TINS on our features; per-image traces for ReNeg on top of TINS | GPU, about 2 h | `transfers/p4_tins/` | `results/p4_tins/` |
| 15 | `D1_vitl14_check` | Is the ViT-L/14 set-up right? NegLabel and MCM against their published L/14 numbers | about 10 min | `transfers/d1_vitl14_check/` | `results/g4_d1/` (pending) |
| 16 | `G4_reneg_on_tins` | **Few-shot table**: ReNeg on TINS and on TANL+TINS (one pass: TINS + TANL + 9 ReNeg heads) | GPU, 2-3 h | `transfers/g4_tins/` | `results/g4_d1/` (pending) |

Notes:

* G3 starts with a check that runs ReNeg once on the GPU and once on the CPU (identical metrics are required) and
  prints an estimate of the total time.
* P1 downloads SDXL-Turbo (about 7 GB) and P2 the Kandinsky 2.1 prior; P2 also downloads the ViT-L/14 cache from
  Kaggle (about 0.6 GB).
* G4 reuses P4's `transfers/p4_tins/tins_static.pt` (TINS's static negatives) when present.
* P4 and G4 halve the TINS inversion chunk automatically after a CUDA out-of-memory error.

## Everyday rules on Colab

* **New code?** Rebuild the bundle (`python tools/make_colab_bundle.py`) and put it into `uploads/`. The first cell
  of every notebook installs a bundle with a higher `reneg.__version__` by itself, so C0 need not run again.
* **Start every session from the top.** The first cell mounts Drive and links the project folder.
* **Results are safe on Drive once flushed.** Colab uploads in the background: run the last cell of every notebook
  (**Finish saving to Google Drive**) before you disconnect.
* **Save compute units.** CPU for C0 and for reading results, a GPU only for runs; **Runtime → Disconnect and
  delete runtime** when done.
* **Avoid many small files on Drive.** The cache is a few large files, which is fine; copy new image datasets to the
  Colab disk as zips and unzip them there.

## The Kaggle notebooks on Colab

`My Drive/ReNeg/notebooks_colab/` holds Colab copies of notebooks 00-04 whose first cell finds the code on Drive.
Notebooks 02-04 run as they are; notebook 01 encodes images and needs ImageNet, which Kaggle mounts, so encode new
backbones on Kaggle and run C0 again.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `ReNeg_Colab_bundle.zip not found` | Upload the zip, not unzipped, into `My Drive/ReNeg/uploads/` |
| `No Kaggle credentials found` | Add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Secrets with Notebook access on, or use route C |
| `401` or `403` from Kaggle | The key is wrong or was replaced: create a new token and update both secrets |
| C0 step 5: `notebook1: FAILED` | Copy the slug from the notebook's URL into `NOTEBOOKS` in C0 step 3 and run from there; else route B or C |
| C0: `FAIL all 12 feature sets` | The table above it shows which set is MISSING or SHORT: re-run Kaggle notebook 01 (it resumes), then C0 |
| C1 check 4 outside 60-75% | The cache or its labels are wrong: run C0 again and compare with notebook 00's accuracy on Kaggle |
| C1 check 7 difference above 0.2 | Compare with Kaggle rows of config `paper`, fp32, seed 0; another GPU explains a few hundredths, not more |
| `code/oodlab_code not found` | Run C0 first |
| G1 part A fails (fewer than 900 classes) | Set `p_min=0.3` in G1's settings cell and run from there |
| Output files printed but the Drive folder is empty | The runtime disconnected before the upload finished: run again and end with the last cell |
| `reneg was already imported in this session` | **Runtime → Restart session**, then run again from the top |
| `CUDA out of memory` | **Runtime → Restart session** and run again; P4/G4 shrink their inversion chunk by themselves |
