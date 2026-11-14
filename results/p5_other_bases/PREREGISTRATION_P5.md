<!-- Written on Nov 14, 2026 in the working notes, before the numbers it refers to. The only edit for the
repository: the method's working name is replaced by the repository's name (ReNeg). -->
# Probe P5: ReNeg on more base detectors (written Nov 14, 2026, before any P5 number)

Question (supervisor, Nov 2026): the paper says ReNeg plugs into any detector but shows only TANL and TINS.
Does the same head, unchanged, improve other detectors?

## What is fixed before running
- Head: ReNeg-balanced, blind pool (13,878 relatives), default settings (omega 0.25, delta 1.0, beta 0.5, gamma 2,
  T 0.5, n_ev 2, k_vis 2, n_min 1, agreed ID admission). Nothing is tuned per base.
- Interface: the packaged generic head (`reneg/multi.py`, `ReNegHead.step(X, lt)`), as used for TINS in G4. It
  reads only the base's score: lt = logit(score); vote = lt below the running Otsu threshold of lt; base ID mask =
  not voted OOD. A base score above 1 is rescaled into (0, 1) first (AdaNeg: divide by 1 + lambda).
- Bases under the pass rule: NegLabel (10,000 mined negatives, 100 groups) and AdaNeg (official settings),
  i.e. the two rows of Table 1 the supervisor named.
- Data: OpenOOD v1.5 (near: SSB-hard, NINCO; far: iNaturalist, Textures, OpenImage-O) and Four-OOD (50k ID),
  stream orders 0-2, batch 256, state reset per stream. CPU replay on the 8-bit feature exports, float32
  arithmetic (the GPU runs use the exact fp16 features; they differ by about 0.2 FPR95 per set).

## Pass rule (per base)
ReNeg-balanced on base B has a lower FPR95 than B on the near, far and Four-OOD group means (mean of the three
orders) in OpenOOD's convention (OOD positive), with the same sign in every order for near and far.
The ID-positive convention is reported beside it; "passes in both conventions" is stated separately.

## What happens after
- PASS on a base: it gets a row in a "ReNeg on other base detectors" table; final numbers from a GPU run on the
  exact features (Colab), same rule.
- FAIL on Four-OOD only: reported as it is (same trade-off as ReNeg-max on TANL); the safe operating point is
  shown as the do-no-harm variant. No setting is changed to rescue a base.
- FAIL because the base's log-odds have a different spread than TANL's (fusion weight): a scale-free fusion would
  be a change of the method; it may only be chosen on the held-out pseudo-OOD / validation tasks and must then be
  re-run on every base, TANL and TINS included.

## Exploratory (reported, not under the rule)
max and safe operating points on the same bases; MCM as a base; Hu et al. (ICML 2026) and DDE (ECCV 2026) if
they can be re-implemented faithfully from the papers or released code (each needs its own reproduction check
against the published numbers before ReNeg is put on top).
Sat Nov 14 13:41:51 UTC 2026

## Addendum (Nov 14, before any DDE or Hu et al. number): the two extra bases
- DDE (ECCV 2026): official code is public (github.com/PolyU-VCLab/OpenOOD-VLM, commit 48b0233,
  openood/postprocessors/dde_postprocessor.py, scripts/ood/dde/imagenet.sh). Its OOD score is re-implemented on
  cached features, op for op from that file (float32): 10,000 NegLabel negatives duplicated in the selection bank
  as in the official network class, hat_M 500, ID threshold 0.75, OOD threshold 0.10, caches of 5,000, noise
  features as the initial OOD cache. Its Gaussian branch only changes the class prediction, not the score.
- Hu et al. (ICML 2026): no public code found; re-implemented from the paper (Alg. 1, Eqs. 3, 7-15; tau 0.01,
  kappa 0.05, rho 0.1, beta 0.95, 10,000 negatives, one sample per step). Ambiguous points are fixed here:
  step size eta = rho / sqrt(count) (Eq. 8's form); soft pseudo-labels = CLIP's text softmax (tau) over the ID
  names for positives and over the negative labels for negatives; each branch's loss is over its own label set.
- Reproduction rule before ReNeg is put on a re-implemented base: group means within 1.0 FPR95 of the paper on
  Four-OOD and within 1.5 on OpenOOD near and far, in the convention that matches best (the match also tells us
  which convention the paper uses). Reproduced -> same pass rule as above. Not reproduced -> the base is shown
  only as "our re-implementation" in the supplement, with the gap, and ReNeg on it is reported as exploratory.

## Addendum 2 (Nov 14 14:51 UTC, before any ReNeg number on the chosen DDE configuration or on a half-precision base)
### Which DDE is "our DDE": chosen by reproduction only (base alone, no ReNeg head)
FPR95 (OOD positive), stream order 0, SSB-hard / NINCO; paper: 61.02 / 56.26 (near 58.64).
- A. release as is (10,000 negatives duplicated as the selection bank, OOD threshold 0.10, caches 5,000): NINCO 66.20.
- B. selection bank = the whole 69,554-word corpus, release thresholds and caches: 65.68 / 56.74 (near 61.21, +2.57).
- C. whole corpus, the paper's stated settings (Sec. 4.1: lambda_pos 0.75, lambda_neg 0.25, queue Q 1,000, hat_M 500):
  62.15 / 55.88 (near 59.02, +0.38).
- D. 10,000-negative bank with the paper's settings: duplicated 63.51 / 59.42 (near 61.47); not duplicated
  65.94 / 58.53 (near 62.24).
Decision: C (closest to the paper; inside the 1.5 tolerance on near for order 0). The reproduction rule of the first
addendum is applied to C on all three group means over the three orders. Batch 256 as for every method in our
protocol (the DDE paper states 128).
Already seen before this decision (exploratory, not used for it): ReNeg heads on configuration B, order 0, first
four sets: balanced lowered FPR95 on all four (SSB-hard 65.68 -> 65.36, NINCO 56.74 -> 54.17, iNaturalist
0.42 -> 0.18, Textures 9.79 -> 8.41).

### Arithmetic of the base: single precision (as fixed above), and why it matters
The pass rule was fixed for float32 base scores. Table 1's NegLabel and AdaNeg rows use the official half-precision
arithmetic. One exploratory stream (NINCO, order 0) on half-precision NegLabel: base 60.87, + balanced 61.47
(worse by 0.60; ID-positive 68.80 -> 65.69), against 60.12 -> 57.06 in float32. Cause: in half precision the
NegLabel score rounds to exactly 1 for confident images, so its log-odds carry no ranking there.
Decision: nothing changes in the rule. The head reads the base's score in single precision (this is stated in the
paper as a requirement of the interface); the GPU confirmation runs the bases in float32. Half-precision bases
(Table 1's rows) are run beside them as exploratory rows and reported whatever they show. No setting of the head is
changed for them.
