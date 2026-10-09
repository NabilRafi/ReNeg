# G4 and D1 (pending)

Both are built and tested (smoke run on a fake Drive; their cells run on the real oodlab with a tiny random CLIP;
G4's ReNeg heads give exactly the replay scores of `experiments/07_reneg_on_tins.py` on real TINS traces). Their
outputs go here once they have run.

* **G4** (`notebooks/4_evaluation/G4_reneg_on_tins.ipynb`, or `configs/fewshot_tins.yaml` with
  `scripts/evaluate.py`): the official few-shot table. One pass per stream runs TINS and TANL once and nine ReNeg
  heads on top (`reneg/multi.py`): balanced / max / safe on TINS, balanced on TINS with the full pool and without the
  labelled images, balanced / max / safe on TANL+TINS, balanced on TANL. OpenOOD v1.5 and the 45k Four-OOD streams,
  three orders; about 2-3 hours on an RTX 4060. Outputs: `g4_runs.csv`, `g4_summary.csv`, per-image scores.
  Expected from the replays (ID+, near / far / Four-OOD 45k): TINS 56.63 / 11.79 / 6.83; balanced on TINS
  48.31 / 9.95 / 6.20; balanced on TANL+TINS 39.32 / 9.60 / 5.55.
* **D1** (`notebooks/4_evaluation/D1_vitl14_check.ipynb`, or `configs/baselines_vitl14.yaml`): NegLabel and MCM on
  the ViT-B/16 and ViT-L/14 caches against their published numbers (Four-OOD, ID+: NegLabel B/16 25.40, L/14 24.81;
  MCM B/16 42.74, L/14 38.17), plus TANL with its released script's settings on ViT-L/14; about 10 minutes.
  Outputs: `d1_runs.csv`, `d1_info.json`. It tells whether our ViT-L/14 TANL baseline (Four-OOD 14.79 vs 9.52 in
  TANL's paper) is a feature / text-bank problem or a TANL-specific one.
