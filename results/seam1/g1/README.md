# G1 · transport gate (Sep 30, bundle v0.1.0): FAILED

Seam 1 asked whether a map fitted on ImageNet (R1 gap shift, R2 identity-anchored ridge, B1 Procrustes, B2 kernel
ridge) moves text embeddings of negative labels to where their images are, better than raw text (R0).

* `G1_cell_outputs_part_B_selfcheck.txt`: leave-subtree-out self-check on ImageNet (81 WordNet groups). Chosen: R0,
  gain 0 (top-1 88.4% for R0 and the chosen map alike).
* `G1_cell_outputs_part_C_ood_concepts.txt`: the same on the OOD concepts of NINCO, SSB-hard and Textures. Chosen:
  R0; `"pass": false`.

These are the Colab cell outputs as pasted (some HTML from Colab's table widget is in them). Decision: run G1b.
