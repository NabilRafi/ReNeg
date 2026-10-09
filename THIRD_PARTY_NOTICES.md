# Third-party notices

This repository's own code is under the MIT licence (LICENSE). It includes or derives from the following
third-party material, each under its own terms; the licence texts are in `licenses/`.

| Component | Where | Licence | Notes |
| --- | --- | --- | --- |
| OpenAI CLIP (github.com/openai/CLIP, commit d05afc4) | `src/oodlab/_vendor/clip/` | MIT (`licenses/OpenAI_CLIP_MIT.txt`) | vendored; one change: the tokenizer works without `ftfy` (`VENDORED.txt`) |
| OpenOOD-VLM (github.com/YBZh/OpenOOD-VLM, commit c6fef2f) | `src/oodlab/official/` | MIT, Copyright (c) 2021 Jingkang Yang (`licenses/OpenOOD-VLM_MIT.txt`) | TANL's, AdaNeg's and NegLabel's official postprocessors, vendored verbatim and used only to test that our implementations are equivalent; ImageNet class names (`src/oodlab/resources/imagenet_classes.json`) |
| NegLabel's WordNet word lists (as distributed with NegLabel's official code and OpenOOD-VLM) | `src/oodlab/resources/neglabel_txtfiles/` | Apache-2.0 (`licenses/NegLabel_Apache-2.0.txt`, also next to the files) | the WordNet-derived word corpus that NegLabel, AdaNeg and TANL draw negatives from |
| WordNet 3.0 (Princeton University) | `src/reneg/resources/kg_pool.json`, `kg_hood.json`; the NegLabel word lists | WordNet 3.0 licence (`licenses/WordNet_3.0_license.txt`) | the KG pool is built from WordNet synsets, lemmas and relations (via NLTK) |

## Re-implementations (no code copied)

* **TANL** (`src/oodlab/methods/tanl.py`), **AdaNeg** (`adaneg.py`), **NegLabel** and **MCM** (`neglabel.py`):
  re-implemented on cached features; equivalence with the vendored official code is tested in
  `tests/oodlab/test_equivalence.py`.
* **TINS** (`src/reneg/tins.py`): re-implemented from the description and the official evaluation script
  (github.com/zxk1212/tins, `eval_tins_w_init.py`); our version runs on cached image features and reuses NegLabel's
  WordNet word files for its static negatives.

## Data and models used at run time (not included)

ImageNet-1K (image-net.org terms of access / Kaggle competition rules), the OpenOOD v1.5 benchmark lists and its
datasets (SSB-hard, NINCO, iNaturalist, DTD / Textures, OpenImage-O, SUN, Places), CLIP weights (OpenAI), and, for
the Seam 1 probes only, SDXL-Turbo (Stability AI) and the Kandinsky 2.1 prior (Kandinsky Lab). Each is downloaded
from its source and used under its own terms; docs/DATA.md lists them.
