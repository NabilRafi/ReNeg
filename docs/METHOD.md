# ReNeg: method

ReNeg is a training-free plug-in for test-time OOD detection with CLIP. It sits on top of a **base detector**
(TANL in the zero-shot table; TINS, or TANL and TINS together, in the few-shot table) and never changes the
base's own computation: it reads the base's score and, on the TANL base, TANL's confident-ID mask. Everything
ReNeg learns comes from the unlabeled test stream and is reset with the stream (per OOD set, as TANL does).

Code: `src/reneg/core.py` (gate, ID model, image score, fusion), `src/reneg/reneg_tanl.py` (ReNeg on TANL),
`src/reneg/multi.py` (ReNeg heads on any base, used for TINS), `src/reneg/resources/kg_pool.json` (the pool).

## 1 · The near-OOD candidate pool (knowledge graph)

Negative-label methods (NegLabel, AdaNeg, TANL) choose their negatives among general words that are *far* from
the ID classes. Near-OOD images (SSB-hard, NINCO) are close relatives of ImageNet classes, and their names are
largely missing from those corpora: TANL's 69,554-word corpus has no WordNet animal or food file, so 3,406 of the
pool's names (2,330 animals, 1,076 foods) are new to it; 39% of the SSB-hard class names are in TANL's corpus
against 66% in the pool.

The pool holds **14,526 WordNet 3.0 nouns** around the 1,000 ImageNet synsets: siblings, parents' siblings,
siblings' children, cousins and cousins' children, minus the ImageNet classes themselves, all their ancestors and
descendants, and any name equal to an ImageNet lemma. The names form **648 clusters** (shared WordNet parents).
Each name is encoded with CLIP's text encoder and the text bank's prompt ("The nice {}.").

* **Blind pool** (every headline result): the pool without the names that are SSB-hard or NINCO classes
  (13,878 names), so the method never sees the near-OOD benchmarks' own class names.
* **Full pool**: all 14,526 names (reported beside the blind pool; it gains 0.2-1.6 near-OOD points).

## 2 · Per batch: score with the current state, then update it

For a batch of image features `x` (unit vectors), ReNeg first scores every image with the state left by earlier
batches, then updates the state with the batch.

**Catching.** Each image goes to its nearest text prototype over `[ID class names ; pool names]`. If a pool name
`j` wins, `j` *catches* the image: it records the text margin `m = cos(x, t_j) - max_c cos(x, t_c)` and adds `x`
to its running sum (its **visual prototype** `v_j`, used once `n_j >= k_vis`).

**Soft LCB gate.** A name is trusted (`pi_j` near 1) only when its caught images are consistently far on the
pool side:

    m_bar_j = (sum of j's margins + gamma * cluster mean margin) / (n_j + gamma)
    LCB_j   = m_bar_j / sd_pooled - beta / sqrt(n_j + gamma)
    pi_j    = sigmoid((LCB_j - delta) / temp)              once n_j >= n_ev (else the prior pi = 0.2)

**Vote (TANL base).** A second test uses the base's own verdict: `share_j` is the shrunk fraction of `j`'s caught
images whose base score lies below the base's running Otsu threshold, and `pi_j *= sigmoid((share_j - 0.5) / 0.1)`.
Without the vote, 1,344 names were admitted on the SUN stream although they caught mostly ID images.

**Online ID image model.** Images predicted as class `c` join `c`'s running sum (the **ID image prototype**
`V_c`) when CLIP's softmax over the ID names is at least `p_min = 0.5` **and** the base's confident-ID mask agrees
(`id_admit = "softmax+base"`, the *agreed* admission; for the TINS base the mask is "not voted OOD"). The
background is 10,000 random directions around the running mean image. **Few-shot setting:** labelled ID images
(OpenOOD's ImageNet ID-val split, about 5 per class, disjoint from the test images) enter the ID model at every
reset.

**Image score** (three operating points):

    plain  (max):       li = LSE(x V_id) - LSE([x V_bg ; x V_kg + log pi])
    kg_clip (balanced): li = min(0, LSE(x V_id) - LSE(x V_kg + log pi))          a pure penalty
    kg_clip_caught (safe): kg_clip only on images a KG name catches in text, weighted by that name's pi

with `LSE(.) = logsumexp(tau * .)`, `tau = 100`.

**Fusion** with the base's log-odds `lt = logit(S_base)`:

    s = (1 - omega) * lt + omega * li,      omega = 0.25

On the TANL base the KG names enter **only** through visual prototypes (`pool_frac = 0`, `pool_in_text = False`):
adding the gated KG mass to TANL's activation-aware text score wrecked Textures (FPR95 11.4 -> 25.6).

## 3 · Operating points and defaults

| Name | Image score | Vote | Character |
| --- | --- | --- | --- |
| **balanced** (main) | `kg_clip` | yes | beats TANL on every benchmark in every stream order |
| max | `plain` | no | best near-/far-OOD, costs Four-OOD (scene sets) |
| safe | `kg_clip_caught` | yes | smallest changes to TANL |

Defaults (`ReNegConfig`, the same for every operating point): `tau 100`, `omega 0.25`, `prior 0.2`, `delta 1.0`,
`beta 0.5`, `gamma 2`, `temp 0.5`, `n_ev 2`, `k_vis 2`, `n_min 1`, `p_min 0.5`, `id_admit "softmax+base"`,
`pool_frac 0`, `pool_in_text False`. `reneg.pipeline.reneg_config(mode, base)` builds them; `configs/*.yaml`
name them.

How the settings were chosen (docs/EXPERIMENT_LOG.md): the operating points and the gate on pseudo near-OOD tasks
built from held-out ImageNet classes (experiments/06); the agreed admission after a mechanism analysis of the
Four-OOD cost on the benchmark streams (experiments/03 and 04, every variant tried is listed). The paper states
which choices touched benchmark data.

## 4 · ReNeg on other bases

ReNeg needs from its base only a per-image score (higher = more ID) and, optionally, an ID mask:

| Base | Score `lt` | Vote on | ID mask for the agreed admission |
| --- | --- | --- | --- |
| TANL | `logit(S_TANL)` | `logit(S_TANL)` | TANL's confident-ID mask |
| TINS | `logit(S_TINS)` | `lt` | not voted OOD (the same Otsu rule as the vote) |
| TANL + TINS | `(logit S_TANL + logit S_TINS) / 2` | `lt` | TANL's mask and not voted OOD |
| NegLabel (ablation base) | NegLabel's score | - | softmax only |

`reneg.multi.ReNegHead` runs ReNeg on any base; `reneg.multi.MultiBase` runs TINS and TANL once per batch with
several heads (notebook G4, `configs/fewshot_tins.yaml`). `src/reneg/tins.py` re-implements TINS's official
evaluation code on cached features (static WordNet negatives filtered by prototype separation, grouped score,
textual inversion of candidate OOD images with 30 AdamW steps, a dynamic bank of 2,000 negatives).

## 5 · Evaluation protocol

* **OpenOOD v1.5** (ImageNet-1K test split, 45,000 ID images): near-OOD SSB-hard and NINCO; far-OOD iNaturalist,
  Textures, OpenImage-O. **Four-OOD** (ImageNet val, 50,000 ID images): iNaturalist, SUN, Places, Textures.
* One stream per (ID set, OOD set): `ConcatDataset([OOD, ID])` shuffled with `torch.randperm(seed)`, batches of
  256, state reset per stream (TANL's protocol); three stream orders (seeds 0, 1, 2), reported as means with the
  paired difference to TANL `[min, max]` over orders.
* **Few-shot rows**: Four-OOD on the ID images that are not labelled images (protocol `four_ood_45k`: the 5,002
  images of the 50k set that match OpenOOD's 5,000 ID-val images, cosine > 0.999, are removed from the stream,
  leaving 44,998). Compare few-shot rows only within that protocol: TANL scores 9.49 on
  the 45k streams but 10.00 when the 50k streams are scored on the same 45k images.
* **Metrics**: FPR95 in both conventions (docs/FPR95_CONVENTIONS.md) and AUROC, as percentages.
* CLIP ViT-B/16 at 224 px (main) and ViT-L/14 at 224 px; image features are computed once and cached (fp16).
