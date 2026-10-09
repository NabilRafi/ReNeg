# P2 · diffusion-prior probe (Oct 2): FAILED, Seam 1 closed

Plan B2: the Kandinsky 2.1 diffusion prior maps a name's ViT-L/14 text embedding to an image embedding; a ridge map
takes it from ViT-L/14 to our ViT-B/16 space.

* `reneg_prior_probe.npz`: the probe output (8-bit prior image embeddings, 4 per name, for 414 names; settings).
* `../prior_eval.json` and `../prior_eval.log`: the pre-registered evaluation (now `experiments/09_seam1_prior.py`).
  Screen 1 (prior prototype finds its own real concept more often than text): FAIL (NINCO 68.8% vs 79.7%, SSB-hard
  53.3% vs 66.0%). Screen 2 (L/14-to-B/16 map): PASS (cosine 0.91-0.92). Stream test on NINCO: FAIL (gain at weight
  4: -0.56; needed 3).
* `reneg_l14_centroids.npz`, `reneg_l14_text.npz`, `reneg_l14_to_b16_map.npz`: the other inputs of the evaluation
  (ViT-L/14 per-concept image centroids of NINCO, SSB-hard and ImageNet from two halves of each concept's images;
  8-bit ViT-L/14 text embeddings of the ID names, probe names and pool names; the ridge map from ViT-L/14 to ViT-B/16
  image space with its held-out checks). P2's prototypes for all 14,526 pool names (`reneg_prior_pool.npz`, 25 MB)
  are not included: no evaluation reads them.
