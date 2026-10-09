# G1b · corrected transport diagnostics (Sep 30, reneg 0.2.0): FAILED

G1b fixed two flaws of G1 (per-name centroids instead of per-image prototypes; image space scored at the text
temperature) and added an oracle. Files: `g1b_summary.txt` (all tables), `g1b_results.json`, and the per-test CSVs
(B' held-out subtrees, D' OpenOOD stream test, the hamper test, the oracle).

Result: every map lowers ImageNet val top-1 by 5+ points (R0 67.26 vs 61.5-62.3), and fused into the stream the maps
give near-OOD FPR95 75-77 vs 68.15 for raw text. Decision (Sep 30): fitted maps dropped; Plan B (P1) next.
