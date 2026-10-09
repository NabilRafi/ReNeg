# Reproducing the results

Three levels, from minutes to a full re-run:

1. **Check that everything runs** (no data, CPU, ~10 minutes): unit tests and the whole pipeline on fake data.
2. **Re-derive every table from the shipped result files** (`results/`, seconds).
3. **Re-run the experiments**: the GPU evaluations from a CLIP feature cache, and the CPU replay experiments
   (ablations, diagnostics, supplementary analyses) from 8-bit exports of that cache.

Numbers are FPR95 (%) in OpenOOD's convention (OOD positive) unless marked ID+ (docs/FPR95_CONVENTIONS.md);
means over three stream orders; differences are paired per order against TANL, `mean [min, max]`.

## 0 · Install

```bash
git clone https://github.com/NabilRafi/ReNeg.git && cd ReNeg
python -m venv .venv && source .venv/bin/activate          # Python 3.9-3.12
pip install -e ".[all]"                                     # or: pip install -r requirements.txt
```

A CUDA build of PyTorch is needed only for the GPU evaluations (`scripts/build_cache.py`, `scripts/evaluate.py`
with TINS); everything else runs on a CPU. OpenAI CLIP is vendored (`src/oodlab/_vendor/clip`), so no Git
dependency is installed; the CLIP weights are downloaded on first use (or pass `--clip-weights DIR`).

## 1 · Check that everything runs (no data needed)

```bash
pytest -q                                   # unit tests: official TANL/AdaNeg/NegLabel == ours, ReNeg, TINS, ...
bash tools/smoke_pipeline.sh                # build_cache -> evaluate -> summarize -> export -> experiments,
                                            # on a tiny fake dataset with a random CLIP (numbers are meaningless)
python tools/smoke_kaggle_notebooks.py      # optional: Kaggle notebooks 00-04 on fake data
python tools/colab_smoke/make_fake_drive.py runs/fake_drive && \
python tools/colab_smoke/run_smoke.py runs/fake_drive        # optional: every Colab notebook on a fake Drive
```

## 2 · Tables from the shipped results

```bash
# GPU runs (results/g3_*, results/p4_tins)
python scripts/summarize.py results/g3_vitb16/g3_runs.csv --ref tanl          # Table 1, ViT-B/16 (+ few-shot rows)
python scripts/summarize.py results/g3b_vitl14/g3_runs.csv --ref tanl         # ViT-L/14
python scripts/summarize.py results/p4_tins/tins_runs.csv                     # TINS on our features
python scripts/summarize.py results/g3_vitb16/g3_runs.csv --sets              # per-set tables

# CPU replays (results/replays)
python experiments/07_reneg_on_tins.py report --file results/replays/tins_replay.jsonl   # ReNeg on TINS, pass rule
python experiments/02_replay.py table --file results/replays/idpos.jsonl --ref TANL      # both conventions
python experiments/02_replay.py table --file results/replays/seeds.jsonl --ref TANL      # first release (Oct 2)
python experiments/04_four_ood_fixes.py report --file results/replays/four_fix.jsonl     # Four-OOD fixes
python experiments/05_fewshot_and_oracle.py report --file results/replays/id_prior.jsonl # few-shot, oracle
```

`results/README.md` lists every file and the run that produced it.

## 3 · Re-running the experiments

### 3.1 · Feature cache (GPU, once per backbone)

```bash
python scripts/build_cache.py --home oodlab_work --inputs /data/imagenet /data/openood      # ViT-B/16, 1-2 h on a T4
python scripts/build_cache.py --home oodlab_work --inputs ... --backbone ViT-L/14          # ViT-L/14
```

docs/DATA.md says where the images come from. Kaggle notebook 01 does the same (docs/KAGGLE_GUIDE.md).

### 3.2 · GPU evaluations (the paper's tables)

`scripts/evaluate.py` appends one row per (run, protocol, OOD set, stream order) to `<out>/runs.csv` and resumes
where it stopped; `scripts/summarize.py` prints the tables. Times are for an RTX 4060.

| Paper | Command | Time | Shipped result |
| --- | --- | --- | --- |
| Table 1 (zero-shot, ViT-B/16) + few-shot rows on TANL | `python scripts/evaluate.py --config configs/main_vitb16.yaml --cache oodlab_work/working/cache --out runs/main_vitb16` | 1.6 h | `results/g3_vitb16/` |
| Second backbone (ViT-L/14) | `... --config configs/vitl14.yaml --out runs/vitl14` | 45 min | `results/g3b_vitl14/` |
| Few-shot table: TINS, ReNeg on TINS and on TANL+TINS | `... --config configs/fewshot_tins.yaml --out runs/fewshot_tins` | 2-3 h | replays in `results/p4_tins/`, `results/replays/tins_replay.jsonl` (G4 pending) |
| Baselines (MCM, NegLabel, TANL, TANL with the released script) | `... --config configs/baselines_vitb16.yaml --out runs/baselines` | 1 h | `results/baselines/` (Kaggle 02-04) |
| ViT-L/14 set-up check (D1) | `... --config configs/baselines_vitl14.yaml --out runs/d1` | 10 min | pending |
| Ablations on TANL | `... --config configs/ablation_tanl_base.yaml --out runs/ablation_tanl` | 2 h | replays in `results/replays/` |
| Ablation ladder on NegLabel's negatives | `... --config configs/ablation_neglabel_base.yaml --out runs/ablation_neglabel` | 30 min | `results/replays/s2full_*.jsonl` |

Then, for example: `python scripts/summarize.py runs/main_vitb16/runs.csv --ref tanl`.

**Expected (Table 1, ViT-B/16, `results/g3_vitb16/`):**

| Method | near | far | Four-OOD | near ID+ | far ID+ | Four-OOD ID+ |
| --- | --- | --- | --- | --- | --- | --- |
| TANL (our reproduction; paper 60.06 / 17.21 / 9.81) | 60.05 | 16.98 | 9.89 | 53.47 | 16.98 | 9.27 |
| ReNeg-balanced, blind pool | **56.57** -3.49 [-4.36, -2.83] | **14.66** -2.31 [-2.47, -2.11] | **9.61** -0.28 [-0.51, -0.11] | 44.91 | 12.27 | 7.79 |
| ReNeg-balanced, full pool | 56.39 | 14.72 | 9.68 | 44.31 | 12.22 | 7.82 |
| ReNeg-max, blind pool | 52.76 | 12.73 | 10.56 | 47.10 | 12.22 | 8.12 |
| ReNeg-max, full pool | 51.18 | 12.81 | 10.65 | 46.44 | 12.19 | 8.16 |
| ReNeg-safe, blind pool | 58.50 | 16.43 | 9.96 | 46.27 | 14.80 | 8.80 |

Few-shot rows (5 labelled images per class; Four-OOD on the 44,998 non-shot ID images): TANL 60.05 / 16.98 / 9.49;
ReNeg-balanced 55.86 / 14.36 / 8.63; ReNeg-max 50.01 / 11.48 / 9.24.

**ViT-L/14 (`results/g3b_vitl14/`):** TANL 54.71 / 16.87 / 14.79; ReNeg-balanced, blind 51.52 / 14.06 / 13.50
(-3.20 / -2.81 / -1.29). TANL's paper reports 9.52 on Four-OOD with ViT-L/14; our ViT-L/14 TANL is 5 points
worse while the same code reproduces ViT-B/16, which D1 (`configs/baselines_vitl14.yaml`) checks.

**ReNeg on TINS (replays on P4's TINS scores, ID+; Four-OOD 45k):** TINS 56.63 / 11.79 / 6.83; ReNeg-balanced on
TINS 48.31 / 9.95 / 6.20 (pre-registered pass rule: PASS); TANL+TINS 50.75 / 11.57 / 5.70; ReNeg-balanced on
TANL+TINS 39.32 / 9.60 / 5.55. `configs/fewshot_tins.yaml` (notebook G4) gives the official numbers.

GPU and CPU give identical metrics (checked at the start of G3). Runs on the 8-bit exports (experiments below)
differ from the GPU runs by about 0.2 FPR95 per set (mean absolute difference 0.17-0.20).

### 3.3 · CPU replay experiments (ablations, diagnostics, supplementary material)

TANL's dynamics do not depend on ReNeg, so TANL runs once per stream and ReNeg's variants are replayed on its
stored trace in seconds to minutes (`experiments/00_check_equivalence.py` shows the replay equals the packaged
`ReNegTANL` to 3e-5). Inputs: 8-bit exports of the ViT-B/16 cache (docs/DATA.md).

```bash
python scripts/export_features.py --cache oodlab_work/working/cache --out exports/vit-b-16    # ~5 min
export RENEG_EXPORTS=exports/vit-b-16                 # outputs go to runs/experiments (RENEG_EXP_OUT)
python experiments/00_check_equivalence.py            # replay == ReNegTANL                     ~2 min
python experiments/01_build_traces.py                 # TANL traces, 27 streams, ~3.5 GB          ~1.5 h (2 cores)
python experiments/02_replay.py paper                 # TANL + operating points, both conventions ~2-3 h
python experiments/02_replay.py table --ref tanl
```

| Script | Question (paper section) | Shipped rows |
| --- | --- | --- |
| `02_replay.py run ...` | any ReNeg variant (`--set k=v`, `--opts`), e.g. the ablations | `results/replays/seeds.jsonl`, `idpos.jsonl`, `tanl_eval_*.jsonl` |
| `03_four_ood_diagnosis.py all` | why the first release lost on Four-OOD (analysis) | `results/replays/logs/diag_four.log` |
| `04_four_ood_fixes.py run/report` | the fixes screened and confirmed; agreed admission (method, ablation) | `results/replays/four_fix.jsonl` |
| `05_fewshot_and_oracle.py run/report` | few-shot prior; oracle ID models (analysis, limitations) | `results/replays/id_prior.jsonl` |
| `06_pseudo_ood_tuning.py traces/modes/admission` | settings chosen without benchmark OOD data (method) | `results/replays/pseudo_*.jsonl` |
| `07_reneg_on_tins.py selftest/run/report` | ReNeg on TINS and on TANL+TINS; pass rule (few-shot table) | `results/replays/tins_replay.jsonl` |
| `08_seam1_generation.py [--gap]` | generated images as prototypes, P1 (negative result) | `results/seam1/planb_results.json` |
| `09_seam1_prior.py` | diffusion prior, P2 (negative result) | `results/seam1/prior_eval.json` |
| `10_seam1_bridges.py` | six more training-free bridges (negative result) | `results/seam1/seam1_rethink.json` |
| `11_solved_seam1_ceiling.py` | ceiling of a perfect text-to-image bridge (analysis) | `results/seam1/oracle_prior.jsonl` |

`07` reads P4's TINS traces from `results/p4_tins/tins_traces_1.zip`; `08`-`11` need G1b's analysis pack
(`scripts/export_features.py --what pack`) and the probe files in `results/seam1/`. experiments/README.md has the
details of each script.

## 4 · The notebooks

The GPU results above were produced by the notebooks in `notebooks/` (G3, G3b, P4), run cell by cell, unchanged;
docs/COLAB_GUIDE.md shows how to run them on Colab or on a local GPU.
