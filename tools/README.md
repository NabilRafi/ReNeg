# Tools

| Tool | What it does |
| --- | --- |
| `smoke_pipeline.sh` | the whole command-line pipeline (build cache, evaluate, summarize, export, experiments) on a tiny fake dataset with a random CLIP; CPU, about 3 minutes; run by CI |
| `smoke_kaggle_notebooks.py` | executes the Kaggle notebooks 00-04 on fake data (`OODLAB_SMOKE=1`) |
| `colab_smoke/` | `make_fake_drive.py` builds a fake Google Drive (fake Kaggle outputs, a stand-in oodlab on a synthetic world, this repository's bundle); `run_smoke.py` executes every Colab notebook against it |
| `build_kaggle_notebooks.py`, `build_colab_notebooks.py` | generate `notebooks/` from their cell lists (deterministic: rebuilding changes nothing) |
| `make_kaggle_dataset.py` | `dist/oodlab_code.zip`: oodlab, Kaggle notebooks 00-04, tests, licences (upload as the private Kaggle dataset `oodlab-code`) |
| `make_colab_bundle.py` | `dist/ReNeg_Colab_bundle.zip`: reneg, its tests and the Colab notebooks, in the layout the notebooks' set-up cell installs |

```bash
bash tools/smoke_pipeline.sh                                   # runs/smoke
python tools/smoke_kaggle_notebooks.py                         # runs/smoke_kaggle
python tools/colab_smoke/make_fake_drive.py runs/fake_drive && python tools/colab_smoke/run_smoke.py runs/fake_drive
```
