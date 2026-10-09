# Experiments: CPU replays on exported features

The design tests, ablations, diagnostics and supplementary analyses behind the paper. They run on a CPU, on 8-bit
copies of the CLIP features (`scripts/export_features.py`), and rely on one fact: TANL's dynamics do not depend on
ReNeg (ReNeg only reads TANL's selection, score and ID mask). So TANL runs **once** per stream and stores a trace
(`01_build_traces.py`), and every ReNeg variant is then **replayed** on the trace in seconds to minutes.
`00_check_equivalence.py` shows the replay equals the packaged `reneg.ReNegTANL` (max |difference| 3e-5).

The paper's headline numbers come from the GPU runs (`scripts/evaluate.py`, `results/g3_*`); these replays agree
with them to about 0.2 FPR95 per set (8-bit features).

## Set-up

```bash
python scripts/export_features.py --cache oodlab_work/working/cache --out exports/vit-b-16   # ~140 MB, ~5 min
export RENEG_EXPORTS=exports/vit-b-16         # inputs   (default: exports/vit-b-16)
export RENEG_EXP_OUT=runs/experiments         # outputs  (default: runs/experiments; traces in <out>/traces)
python experiments/00_check_equivalence.py
python experiments/01_build_traces.py         # 27 TANL traces (5 OpenOOD v1.5 + 4 Four-OOD sets x 3 orders), ~1.5 h on 2 cores, ~3.5 GB
```

Each script appends rows to a `.jsonl` file in `RENEG_EXP_OUT` and skips rows that exist, so a stopped run
resumes. Every `report` / `table` command also reads the rows shipped in `results/` (`--file`), which is how the
numbers in docs/ were produced.

## Scripts

| Script | What it answers | Paper | Time (2 cores) | Rows / output |
| --- | --- | --- | --- | --- |
| `00_check_equivalence.py` | replay == `ReNegTANL` (packaged admission, agreed admission, few-shot prior) | reproducibility | 2 min | stdout, PASS / FAIL |
| `01_build_traces.py` | TANL once per stream (paper hyper-parameters, official stream order) | - | 1.5 h | `traces/*.pt` |
| `02_replay.py paper` | TANL and the operating points of Table 1 on 3 orders, both conventions | Table 1 (replay), ablations | 2-3 h | `replay.jsonl` |
| `02_replay.py run --label L --mode M [--set k=v] [--opts JSON]` | any single variant: an ablation, a setting, a pool | ablations | 10-20 min | `replay.jsonl` |
| `03_four_ood_diagnosis.py all` | per-image state at scoring time: which ID images are penalised, by which KG names, when | analysis (Four-OOD cost) | 15 min | `diag/*.npz`, stdout |
| `04_four_ood_fixes.py run / report [--screens]` | variants of admission and prototypes screened on one order, confirmed on three | method (agreed admission), supplementary | 30 min per screen, 1.5 h per confirmation | `four_fix.jsonl` |
| `05_fewshot_and_oracle.py run / report` | 5 labelled images per class; oracle ID models (purity vs labels vs coverage) | few-shot rows, limitations | 20-60 min per stage | `id_prior.jsonl` |
| `06_pseudo_ood_tuning.py traces / modes / admission` | settings chosen without benchmark OOD data (held-out ImageNet classes) | method (tuning protocol) | 10 min + 20-40 min | `pseudo_tanl.jsonl`, `pseudo_admit.jsonl` |
| `07_reneg_on_tins.py selftest / run / report` | ReNeg on TINS and on TANL+TINS from P4's TINS scores; pre-registered pass rule | few-shot table | 1-2 h | `tins_replay.jsonl` |
| `08_seam1_generation.py [--gap]` | P1: SDXL-Turbo images as prototypes for names | negative result | 20 min | `planb_results.json` |
| `09_seam1_prior.py` | P2: diffusion prior with a ViT-L/14 -> B/16 map | negative result | 15 min | `prior_eval.json` |
| `10_seam1_bridges.py` | six more training-free bridges between names and images | negative result | 2 min | `seam1_rethink.json` |
| `11_solved_seam1_ceiling.py` | what a perfect text-to-image bridge would add to ReNeg | analysis | 15 min | `oracle_prior.jsonl` |

Extra inputs: `07` reads P4's TINS traces from `results/p4_tins/tins_traces_1.zip` (`--traces`); `08`-`11` need
G1b's analysis pack (`scripts/export_features.py --what pack`, needs NLTK WordNet) and the probe files in
`results/seam1/` (`RENEG_GEN_PROBE`, `--p2-dir`).

## Plans

`plans/four_ood_fixes.json` and `plans/fewshot_oracle.json` list every variant in the order it was run on Oct 3
(`--stage` selects a stage). Each item: `mode` (operating point), `variant` (a name), `opts` (replay options and
config overrides, see `replaylab/replay.py`), `jobs` (`screen`: six sets on order 0; `all`: nine sets), `seeds`,
`pool` (`full`, `blind`, `full+neg`). The original per-stage files are in `results/replays/plans/`.

## replaylab

| Module | Contents |
| --- | --- |
| `paths.py` | `RENEG_EXPORTS`, `RENEG_EXP_OUT`, `RENEG_TRACES` |
| `data.py` | lazy loading of the exports, the official stream order, both FPR95 conventions, the few-shot images |
| `traces.py` | `make_trace`, `text_from_trace` (TANL's score in closed form), `replay` (ReNeg as packaged), trace files |
| `replay.py` | `replay_x` (ReNeg with the experimental options, few-shot prior, oracle ID models), `replay_base` (ReNeg on any base score) |
| `seam1.py` | concept tables, P1's generated prototypes, P1's online stream test, `replay_prior` |
| `legacy.py` | reads the shipped `results/replays/*.jsonl` in the column format of `scripts/evaluate.py` |

## Reading the outputs

FPR95 (%) in OpenOOD's convention unless the column says `idpos` / ID+ (docs/FPR95_CONVENTIONS.md). Paired
differences are per stream order against TANL on the same streams, printed as `mean [min, max]` over orders. Rows
that use labelled images score Four-OOD on the 45k ID images that are not those images (TANL re-scored on the same
images).

Small differences to the shipped rows (at most a few hundredths of a point on the TANL-base replays) come from the
ID text: the original runs read it from G1b's fp16 pack, these scripts from the exported text bank (float32).
