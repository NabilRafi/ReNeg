# P1 · generation probe (Oct 1): FAILED

Plan B for Seam 1: generate 4 images per KG name with SDXL-Turbo and use their CLIP embeddings as the name's image
prototype.

* `../reneg_gen_probe.npz`: the probe output (embeddings of 1,656 generated images for 414 names: NINCO, SSB-hard and
  ImageNet controls; `names`, `tags`, `prompt`, `model_id`).
* `../planb_results.json`: the evaluation (now `experiments/08_seam1_generation.py`). Generated prototypes
  find their own real concept less often than text: NINCO 59.4% vs 78.1%, SSB-hard 44.3% vs 60.0%. The stream test
  gains 1.6 FPR95 points on NINCO at weight 4; the rule needed 3.
* `P1_first_attempt_out_of_memory_log.txt`: the first attempt, which ran out of GPU memory in the VAE decoder;
  the re-run completed and produced the files above.
