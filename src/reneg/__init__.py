"""ReNeg: negative labels moved from CLIP text space into CLIP image space (Seam 1),
plus knowledge-graph near negatives behind an LCB gate (Seam 2).

This package sits next to ``oodlab`` (the TANL / AdaNeg / NegLabel reproduction) and only
reads oodlab's feature cache and text bank. Nothing here changes CLIP or oodlab.

Modules
-------
transport   the transport ladder R0-R3 and the Plan B maps B1, B2 (closed forms)
selfcheck   leave-subtree-out validation on ID classes (no OOD data, no labels)
wordnet     WordNet helpers: ImageNet subtrees, names for wnids
concepts    class names and image centroids of OOD sets (NINCO, Textures, SSB-hard)
scores      NegLabel-style scores in text or image space, fused scores, metrics
g1          gate G1 (v1): parts A-E and the decision
g1b         gate G1b: corrected, pre-registered Seam 1 diagnostics (B', H, D', O, E')
packs       compact 8-bit exports of the cache and the WordNet near-OOD candidate pool (resources/kg_pool.json)
gen         Plan B probe: generate images for label names (SDXL-Turbo) and encode them with our CLIP
seam2       Seam 2 prototype: knowledge-graph near negatives grounded in the test stream (superseded by core)
core        the method: KG gate (LCB + TANL vote), online ID image model, image score, ReNeg on NegLabel
reneg_tanl  ReNeg on top of oodlab's TANL (the paper's main method); runs on TANL's device (GPU on Colab)
tins        TINS (arXiv 2605.10756) re-implemented on the feature cache: static negatives, grouped score,
            test-time inversion of OOD-looking images into text (probe P4: ReNeg on top of TINS)
multi       ReNeg on any base score (ReNegHead) and one pass of TINS + TANL with several ReNeg heads (MultiBase, G4)
prior       Plan B2 probe: Kandinsky 2.1 diffusion prior + ViT-L/14 -> ViT-B/16 image map (closed: did not pass)
colab       Google Drive / Colab / Kaggle transfer helpers
"""

__version__ = "0.3.0"
