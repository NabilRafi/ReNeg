# Kaggle: the feature cache and the baselines (notebooks 00–04)

This guide takes you from an empty Kaggle account to: a reproduced TANL (Four-OOD, OpenOOD v1.5,
temporal shift, sample order), a CLIP feature cache of every benchmark image, NINCO / SSB-hard /
OpenImage-O added, the churn diagnostic, and a reproduced AdaNeg with NegLabel and MCM baselines.

Everything is organised so that **images are decoded and encoded exactly once** (notebook 01). All
experiments afterwards replay test streams from cached features, which takes minutes instead of hours
and never touches the ImageNet folders again.

---

## 1. What is in `oodlab_code.zip`

Build it from the repository with `python tools/make_kaggle_dataset.py` (it is written to `dist/`).

```
oodlab_code/
├── GUIDE.md                  this file
├── README.md                 one-page overview
├── oodlab/                   the Python package (all logic lives here, notebooks only call it)
│   ├── methods/tanl.py       TANL, one overridable method per step (select, score, admit, ...)
│   ├── methods/variants.py   worked examples of modifying TANL (TANL_LCB, TANL_FixedGap)
│   ├── methods/adaneg.py     AdaNeg (fp32 and bit-exact fp16 emulation of the official run)
│   ├── methods/neglabel.py   NegLabel and MCM
│   ├── methods/scoring.py    shared scores (activation-aware score, grouped NegLabel score, Otsu threshold)
│   ├── official/             the official OpenOOD-VLM postprocessors, vendored verbatim, used only to prove equivalence
│   ├── data/                 ImageNet on Kaggle, OpenOOD downloads and image lists, Fi-ImageNet-1k hook
│   ├── clipwrap.py           CLIP loading, the official transform, text/image encoders, noise images
│   ├── textbank.py           WordNet corpus, NegMining, all text features
│   ├── cache.py              feature cache (encode once, resumable)
│   ├── stream.py             test streams (shuffled pairs, temporal shift, ID-first / OOD-first)
│   ├── runner.py             protocols, evaluation, summaries, comparison with published numbers
│   ├── diagnostics/churn.py  churn / survival / queue-purity tracker for TANL
│   ├── targets.py            every published number we compare against, with its source
│   └── resources/            ImageNet class names + wnids (official order), NegLabel word lists (Apache-2.0)
├── notebooks/                00 → 04, run in this order on Kaggle
├── tests/                    unit tests: official == ours, metrics, data resolution, full pipeline
├── tools/                    build_notebooks.py (source of the notebooks), run_notebooks_smoke.py
└── licenses/                 OpenOOD-VLM (MIT), OpenAI CLIP (MIT, bundled in oodlab/_vendor), NegLabel word lists (Apache-2.0)
```

The notebooks are generated from the repository's `tools/build_kaggle_notebooks.py`; if you change a notebook
permanently, change it there so all five keep the same set-up cell.

---

## 2. One-time Kaggle set-up (≈ 15 minutes)

1. **Verify your phone number** (Kaggle → Settings → Phone verification). Without it you get neither
   GPUs nor Internet in notebooks.
2. **Join the ImageNet competition.** Open *ImageNet Object Localization Challenge*
   (`kaggle.com/c/imagenet-object-localization-challenge`) → *Late Submission / Join* → accept the rules.
   Its 50,000 validation images are our ID set; nothing is downloaded, Kaggle mounts it read-only.
3. **Upload the code as a private dataset.** Kaggle → *Datasets* → *New Dataset* → drag in
   `oodlab_code.zip` → title `oodlab-code` → keep it **Private** → *Create*. Kaggle unpacks the zip;
   if it does not, the notebooks unpack it themselves.
4. **Import the five notebooks.** Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* → upload
   `notebooks/00_setup_check.ipynb`. Repeat for 01–04 (one Kaggle notebook each).
5. In every notebook: *Add Input* → *Your Datasets* → `oodlab-code`. In 00 and 01 also *Add Input* →
   *Competitions* → *ImageNet Object Localization Challenge*.

**Updating the code later:** edit locally, run `pytest`, zip, then on the dataset page *New Version* →
upload the new zip. Notebooks pick up the new version when you re-run them (check the printed
`code_hash`; each run also stores a copy of the code it used under `results/<notebook>/code_snapshot`).

---

## 3. Run order

| # | notebook | accelerator | Internet | inputs | time | produces |
|---|---|---|---|---|---|---|
| 00 | setup check | GPU | on | code, ImageNet | 5–10 min | PASS/FAIL table |
| 01 | build feature cache | GPU | on | code, ImageNet (+ zips if Drive fails) | 1–2 h | `cache/vit-b-16/` ≈ 0.3 GB |
| 02 | reproduce TANL | GPU (CPU: fp32 only, slow) | off | code, output of 01 | 40–60 min | tables, report, stream orders |
| 03 | churn diagnostic | GPU | off | code, output of 01 | 20–40 min | per-batch CSVs, plots, verdict |
| 04 | reproduce AdaNeg (+ NegLabel, MCM) | GPU | off | code, output of 01 (+ 02) | 30–45 min | tables, report, leaderboard |

OpenAI CLIP's code is **bundled** in `oodlab/_vendor/clip` (upstream commit d05afc4, MIT), so nothing is
installed from GitHub. Internet is needed in 00/01 only for the CLIP ViT-B/16 weights
(`openaipublic.azureedge.net`, 335 MB) and the OpenOOD downloads. Without Internet: attach a dataset
containing `ViT-B-16.pt` (it is picked up automatically) and the OpenOOD zips (§5).

**Always run 01–04 with *Save Version → Save & Run All (Commit)***, not interactively. A committed run
keeps going when you close the browser, and its `/kaggle/working` becomes a versioned output you can
attach to the next notebook (*Add Input → Your Work → 01-build-feature-cache*).

Kaggle limits to keep in mind: 12 h per session, about 30 GPU-hours per week, 20 GB of saved output.
Notebooks 00–04 need roughly 3–4 GPU-hours in total.

---

## 4. Notebook by notebook

### 00 · Setup check
Runs, and reports PASS/FAIL for each, without stopping at the first failure:
GPU present · scratch disk > 25 GB · the unit tests · official == ours on the GPU (synthetic data) ·
ImageNet located (flat Kaggle layout, 50,000 images, class order identical to OpenOOD-VLM's) · CLIP
ViT-B/16 loads in fp16 · **zero-shot top-1 on 1,000 val images between 60 and 75 %** (catches label or
transform mistakes) · a Google-Drive test download. Do not continue until everything passes.

### 01 · Build the feature cache
1. **Resume:** imports finished sets (and unfinished shards) from any attached earlier output.
2. **CLIP** ViT-B/16, fp16 on GPU — exactly `clip.build_model(state_dict)` + `.cuda()` as the official code.
3. **Text bank:** the WordNet corpus de-duplicated like the official code (must print **11,453 adjectives
   + 58,101 nouns = 69,554 words**), NegMining (95th-percentile cosine to the 1,000 ID names; keeps
   **1,646 + 8,354 = 10,000**), ID prompts "The nice {}." and "a photo of a {}.", 15 noise images.
   Asserted, so a wrong corpus stops the run.
4. **Images, one set at a time:** manifest → download/extract (OpenOOD sets) → check files → encode
   (fp16, batch 256, 10k-image shards) → save → delete the images. A failing set is logged and skipped;
   re-run later to fill gaps. Expected counts:

   | set | images | role |
   |---|---|---|
   | imagenet_val_all | 50,000 | ID, Four-OOD and near-OOD (50k) protocols |
   | imagenet_test / imagenet_val | 45,000 / 5,000 | ID, OpenOOD v1.5 test / validation |
   | inaturalist · sun · places | 10,000 each | Four-OOD (iNaturalist also OpenOOD far) |
   | textures_all | 5,640 | Four-OOD "Textures" (all of DTD) |
   | textures | OpenOOD list | OpenOOD v1.5 far |
   | openimage_o / openimage_o_val | 15,869 / 1,763 | OpenOOD v1.5 far / validation |
   | ssb_hard | 49,000 | OpenOOD v1.5 near |
   | ninco | 5,879 | OpenOOD v1.5 near |

5. **Fi-ImageNet-1k** (optional): see §6.
6. **Acceptance check — live vs cache:** the *official* TANL postprocessor on images decoded and encoded
   on the fly vs *our* TANL on the cache, same stream (NINCO + 5,000 ImageNet images). Must agree on
   FPR95 and AUROC to two decimals (`LIVE == CACHE: True`). On GPU the scores are normally identical.
7. **Summary** of the cache. Then *Save Version*; attach this output to 02–04.

### 02 · Reproduce TANL
* **Equivalence on real features:** official vs ours on 6,000 real samples; fp32 max |Δ| ≈ 1e-6, fp16
  normally 0 (the check accepts up to one fp16 rounding step, in case your GPU picks a different kernel).
* **Benchmarks:** Four-OOD, OpenOOD v1.5 (near/far), near-OOD with 50k ID; 3 stream seeds; two
  hyper-parameter sets × two precisions:
  * `paper` = M 1,000 · L 300 · g 0.2 · α 0.95 (Sec. 4.1 of the paper);
  * `official_sh` = g 0.5 · α 0 (what `scripts/ood/TANL/official.sh` actually runs);
  * `fp32` = float32 arithmetic; `fp16` = bit-exact emulation of the official GPU arithmetic.
* **Temporal shift** (Tab. A15, four orders, no queue reset between OOD sets) and **ID-first /
  OOD-first** orders (Tab. A16), run with whichever configuration came closest to 9.81 (notebook 03
  picks the same one when 02's output is attached).
* **Acceptance table:** Four-OOD mean FPR95 within ±1 of 9.81; OpenOOD near/far within ±1.5 of
  60.06 / 17.21. `CHECK` means outside the tolerance — see §8.
* **Worked example of a modification** (`TANL_LCB`, §7).
* Saves `tanl_runs.csv` (every dataset × seed), `tanl_summary.csv`, `tanl_acceptance.csv`,
  `stream_orders.npz/json` (the exact stream orders used) and `report.md`.

### 03 · Churn diagnostic
Measurement only — step 1 proves the scores are bit-identical with and without the tracker, so the
figures and the FPR95 come from the same run. Per batch: churn, survival of the first-batch and initial
selections, overlap with the oracle set (top-M by the true OOD−ID activation difference), distinct labels
so far, queue purity (FIFO mirrored with ground truth), admission precision/recall, threshold. Per run:
dwell times, batches to 90 % of the oracle-overlap plateau; after each shift in temporal streams: churn,
overlap, recovery time and post-shift FPR95. Outputs: `per_batch/*.csv`, `plots/*.png`, `churn_*.csv`,
`report.md` with the **decision** (> 50 % final survival → cold-start framing; < 10 % → shift-only).

### 04 · Reproduce AdaNeg (+ NegLabel, MCM)
AdaNeg as in `scripts/ood/adaneg/imagenet.sh` (memory 10, threshold 0.5, gap 0.5, λ 0.1, 5 groups,
random permutation, `combine`), in two precisions:
* `fp16` reproduces the official run, including a quirk: its entropy is computed on fp16 probabilities,
  which underflow, so every entropy is NaN and memory slots are never *replaced* (each label keeps its
  first 9 confident features next to its text feature). This is what produced the published numbers.
* `fp32` is the rule as described in the paper (replace the highest-entropy slot).
Acceptance: Four-OOD within ±1 of 18.92 (the repository's own log shows 17.95), OpenOOD near-OOD
close to 67.51. Baselines: NegLabel (10,000 static negatives, 100 groups) and MCM, twice: the paper version
(softmax of cosine / T, T = 1) and the `scripts/ood/mcm/official.sh` version (logit scale 100, τ chosen by
OpenOOD's automatic parameter search on the validation split). The leaderboard merges TANL's results if
notebook 02's output is attached.

---

## 5. Where the data comes from, and what to do when Google Drive says no

* **ImageNet-1K val:** Kaggle competition data (flat `ILSVRC/Data/CLS-LOC/val/` + `LOC_val_solution.csv`).
  OpenOOD's 45k/5k split lists are matched to it by the `ILSVRC2012_val_XXXXXXXX` id, with a label check.
  The training folder (1.28 M files) is never listed.
* **OpenOOD v1.5 sets** (image lists, iNaturalist, SUN, Places, Textures, OpenImage-O, SSB-hard, NINCO):
  the Google-Drive files used by `scripts/download/download.py` of OpenOOD / OpenOOD-VLM. Downloads are
  retried and validated as real zip files.

**If a download fails** ("Too many users have viewed or downloaded this file recently", or an HTML page
instead of a zip), use one of these, then re-run notebook 01 — finished sets are kept:

1. **Your own copy.** Open `https://drive.google.com/file/d/<id>` (ids in `oodlab/data/openood.py`,
   `DRIVE_IDS`), *Make a copy* into your Drive, share the copy as "anyone with the link", and put its id in
   `DRIVE_ID_OVERRIDES` in notebook 01's settings cell, e.g. `{"ssb_hard": "1AbC..."}`.
2. **Upload the zips to Kaggle.** Download `<name>.zip` in your browser (names: `benchmark_imglist`,
   `inaturalist`, `sun`, `places`, `texture`, `openimage_o`, `ssb_hard`, `ninco`), create a private dataset
   `openood-raw-zips` with them and attach it. Zips (or already-extracted folders named like the key) in
   any attached input are used before Drive is tried.

Disk: each archive is downloaded to scratch, extracted, encoded and deleted before the next one, so peak
scratch use is about one dataset (SSB-hard is the largest, roughly 10–15 GB while extracted).

---

## 6. Fi-ImageNet-1k (when the authors send the lists)

Its 655 near-OOD images are *inside* ImageNet-1K val, so nothing new is encoded: the images are already
in `imagenet_val_all`. Save the lists as `fi_imagenet_1k_ood.txt` (655 lines) and
`fi_imagenet_1k_ambiguous.txt` (980 lines) — any line containing `ILSVRC2012_val_XXXXXXXX` works — in a
dataset attached to notebook 01 (copied into the cache) or directly to 02–04. The protocol
`fi_imagenet_1k` uses the 50k ID set **minus both lists**, as the authors do:

```python
from oodlab.data.fi_imagenet import find_lists, read_val_ids, fi_imagenet_sets
ood_p, amb_p = find_lists(ctx.input_roots + [cache.dir])
id_clean, fi = fi_imagenet_sets(cache.load_set("imagenet_val_all"), read_val_ids(ood_p), read_val_ids(amb_p))
runner.extra_sets.update({id_clean.name: id_clean, fi.name: fi})
runner.evaluate(tanl, "fi_imagenet_1k", seeds=[0, 1, 2])
```

---

## 7. Modifying TANL

`oodlab/methods/tanl.py` implements `ActivatedNegPostprocessor.postprocess` as one method per step:

| method | paper | default behaviour |
|---|---|---|
| `reset()` | init | ID queue ← activations of 300 random ID label texts; OOD queue ← 15 noise images |
| `activation(feats)` | Eq. 5 | softmax over [1,000 ID ∣ 69,554 corpus] logits, corpus part |
| `select(combined)` | Eq. 6 | top-M of (OOD-queue mean − ID-queue mean) |
| `score(feats, selected)` | Eq. 15 | activation-aware score (closed form of the official loop) |
| `threshold()` | Eq. 9 | Otsu-style threshold over the last 20,000 scores |
| `admit(conf, thr)` | Eq. 9 | ID if > thr + g(1−thr); OOD if < thr − g·thr |

To change a step, subclass and override that one method — everything else, including the evaluation,
the churn tracker and the comparison tables, keeps working. `oodlab/methods/variants.py` has two
examples; `TANL_LCB` ranks labels by *difference − κ·standard error* in ten lines:

```python
class TANL_LCB(TANL):
    def __init__(self, *a, kappa=1.0, **k):
        self.kappa = kappa; super().__init__(*a, **k)
    def select(self, combined):
        se = torch.sqrt(self.neg_q.var(0, unbiased=False) / len(self.neg_q)
                        + self.pos_q.var(0, unbiased=False) / len(self.pos_q))
        return (combined - self.kappa * se).sort(descending=True)[1][: self.cfg.num_neg]
```

Rules that keep modifications trustworthy:
1. **A variant must reduce to TANL** for some setting (`kappa=0`) — check it gives identical scores.
2. **Run the tests** (`pytest -q`) after touching anything in `methods/` — the equivalence tests fail
   loudly if the baseline moved.
3. **Tune only on the OpenOOD validation split** (`runner.evaluate(m, "openood_val", ...)`: 5,000 ID +
   1,763 OpenImage-O images), never on test sets.
4. Name the variant (`name = "tanl_lcb"`) and pass `config=` to `runner.evaluate` so rows are traceable.

---

## 8. Reading the numbers

**FPR95 has two conventions.** OpenOOD (TANL, AdaNeg, the repository logs) treats OOD as positive: FPR95
= share of *ID* images flagged OOD when 95 % of OOD images are caught — our `fpr95`. MCM and NegLabel's
papers treat ID as positive: share of *OOD* images accepted when 95 % of ID images are kept — our
`fpr95_idpos`. The comparison tables pick the matching column automatically (`fpr_convention`).

**fp32 vs fp16.** The official code runs CLIP in fp16 and computes several scores in fp16. `fp16` rows
reproduce that arithmetic bit for bit (proven against the vendored official code); `fp32` rows are the
same algorithm in float32. For TANL the two usually differ by a few tenths; for AdaNeg they differ more
because of the NaN-entropy quirk (§4). The cache stores fp16 features — that is lossless, because the
official model produces fp16 features.

**If TANL misses the ±1 target:** (i) compare `paper` vs `official_sh` and fp16 vs fp32 rows — the paper
text and the released script disagree on g and α; (ii) look at the seed spread (`fpr95_std`); the paper
reports 9.81 ± 0.01, so a spread of ±0.5 means something stream-dependent is off; (iii) check the
per-dataset rows — Places and Textures dominate the mean; (iv) verify notebook 01's counts and the
live-vs-cache check. Report what you find; a documented, explained gap is a valid reproduction outcome.

**Deliberate differences from the official code** (all documented in the code):
* Stream order, TANL's initial-queue permutation and the noise images are **seeded** (official: unseeded
  global RNG). Seed *k* sets both the stream order and the queue permutation, so one seed = one run; we
  report 3 seeds (the paper: 9.81 ± 0.01 over three runs).
* The corpus is sorted alphabetically before mining and sorted with a stable sort (official:
  `list(set())`, order changes per run). Only exact ties at the 10,000-word boundary can differ.
* The temporal-shift protocol really carries the queues across OOD sets (the official `temshift` configs
  reset memory per dataset, so they do not implement it).
* ID set: Kaggle's 50,000 val images; OpenOOD's 45k/5k lists are mapped onto them.

**Official-repository problems worked around:** `openood/postprocessors/utils.py` imports a missing
module (`activated_neg_postprocessor_sigclip`); `configs/postprocessors/tanl.yml` uses the name `tanl`,
which is not registered (`actneg` is); the download script assumes Drive names the file `<dataset>.zip`;
the NegLabel word lists (`data/txtfiles`) are not shipped; SUN/Places image lists are missing; the Four-OOD
config points to `test_imagenet_all.txt`, which is not one of OpenOOD v1.5's standard lists (we build the
50,000-image ID set directly from Kaggle's validation folder instead).

---

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| `oodlab code not found` | attach the `oodlab-code` dataset (§2.5) |
| `This notebook needs a GPU` | Settings → Accelerator → GPU; phone-verify the account |
| `git clone ... CLIP.git ... exit code 128` | only in code older than v0.2.1 — re-upload the new `oodlab_code.zip` (CLIP is bundled now) |
| CLIP weights download fails | Settings → Internet → On (phone-verified account), or attach a dataset with `ViT-B-16.pt` |
| `pip install regex` fails | Internet is off and the image lacks `regex`: turn Internet on for this run |
| `Could not find LOC_synset_mapping.txt` | attach the ImageNet competition and accept its rules |
| `DownloadError: ... quota` | §5 (your own Drive copy, or zips as a Kaggle dataset) |
| a set has fewer images than expected | the warning names the set; check the zip content, then delete `cache/.../images/<set>.npz` and re-run |
| `N of M listed images not found` | the archive layout differs from the list; the resolver tries joins and a basename index — open an issue with two example list lines and `ls` of the extracted folder |
| session died in 01 | attach 01's last output and re-run; it resumes |
| `No feature cache found` in 02–04 | attach notebook 01's output (Add Input → Your Work) |
| zero-shot accuracy check fails in 00 | labels or transform are wrong — do not continue; check the synset file and the val solution |
| `CUDA out of memory` | lower `BATCH_SIZE` in 01; in 02–04 restart the session (another notebook holds the GPU) |

Logs of every notebook are in `/kaggle/working/logs/<notebook>.log` (also in the saved output).

---

## 10. Running outside Kaggle (a local GPU)

From a clone of the repository:

```bash
pip install -e ".[clip,data,test]"
export OODLAB_HOME=~/oodlab_work          # working/, scratch/, input/ are created under it
pytest -q tests/oodlab                    # oodlab's unit tests, about 30 s on a CPU
python tools/smoke_kaggle_notebooks.py    # executes notebooks 00-04 on fake data (no GPU, no downloads)
```

`scripts/build_cache.py` does what notebook 01 does, from the command line (docs/REPRODUCE.md).

Put the ImageNet competition folder (or any folder with `LOC_synset_mapping.txt`, `LOC_val_solution.csv`
and `ILSVRC/Data/CLS-LOC/val/`) under `$OODLAB_HOME/input/`, and optionally the OpenOOD zips; then run
the notebooks with Jupyter. Paths are resolved the same way as on Kaggle.
