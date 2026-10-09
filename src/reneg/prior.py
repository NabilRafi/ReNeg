"""Plan B2 probe: a diffusion prior that maps a CLIP text embedding straight to a CLIP image embedding.

Public priors (Kandinsky 2.1, Karlo) target OpenAI CLIP ViT-L/14, not ViT-B/16. The probe therefore works in two
spaces:

* in ViT-L/14 space, the prior's prototype of a name is compared with the real image centroid of that concept
  (from a ViT-L/14 feature cache), against the plain text embedding as the baseline;
* to plug into the ViT-B/16 pipeline, a ridge map from ViT-L/14 image embeddings to ViT-B/16 image embeddings is
  fitted on the same ImageNet images encoded by both backbones (image to image, no text involved) and checked on
  NINCO and SSB-hard images, which it never saw.

Nothing here touches OOD test images except the map check, which only measures how faithful the map is.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .transport import l2n

DEFAULT_PRIOR = "kandinsky-community/kandinsky-2-1-prior"


def load_kandinsky_prior(model_id: str = DEFAULT_PRIOR, device: str = "cuda", log=print):
    """prior(prompts, n_per, seed, guidance, steps) -> (len(prompts) * n_per, 768) unit rows, grouped by prompt."""
    from diffusers import KandinskyPriorPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = KandinskyPriorPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def prior(prompts: Sequence[str], n_per: int = 4, seed: int = 0, guidance: float = 4.0, steps: int = 25):
        g = torch.Generator(device=device).manual_seed(int(seed))
        out = pipe(prompt=list(prompts), num_images_per_prompt=int(n_per), guidance_scale=float(guidance),
                   num_inference_steps=int(steps), generator=g)
        return l2n(out.image_embeds.float()).cpu()

    prior.pipe = pipe
    return prior


def fake_prior(dim: int):
    """Deterministic random unit vectors per prompt (tests and offline smoke runs only)."""
    import zlib

    def prior(prompts, n_per=4, seed=0, guidance=4.0, steps=25):
        out = []
        for p in prompts:
            g = torch.Generator().manual_seed(zlib.crc32(f"{p}|{guidance}".encode()) + int(seed) % 997)
            out.append(l2n(torch.randn(n_per, dim, generator=g)))
        return torch.cat(out)

    return prior


def image_encoder_agreement(pipe, clip_model, device: str = "cuda", n: int = 4, seed: int = 0) -> float:
    """Smallest cosine between the prior's own image encoder and our CLIP model on the same random images.
    About 1.0 means the prior predicts embeddings in our ViT-L/14 space."""
    from PIL import Image

    from .gen import clip_image_encoder

    rng = np.random.default_rng(seed)
    ims = [Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(n)]
    ours = clip_image_encoder(clip_model, device=device)(ims)
    proc = pipe.image_processor(ims, return_tensors="pt").pixel_values.to(device, pipe.image_encoder.dtype)
    with torch.inference_mode():
        theirs = l2n(pipe.image_encoder(proc).image_embeds.float()).cpu()
    return float((ours * theirs).sum(1).min())


def run_prior(prior: Callable, names: Sequence[str], out_path: str, n_per: int = 4, batch: int = 32,
              prompt: str = "a photo of a {}.", guidance: float = 4.0, steps: int = 25, seed: int = 0,
              model_id: str = DEFAULT_PRIOR, save_every: int = 10, log=print) -> Dict:
    """Sample n_per prior embeddings for every name and save (resumably) as int8 codes in out_path (.npz)."""
    from .packs import dq8, q8

    names = list(names)
    q_parts, s_parts, done = [], [], 0
    if os.path.isfile(out_path):
        z = np.load(out_path, allow_pickle=False)
        if list(z["names"]) == names and str(z["prompt"]) == prompt and int(z["n_per"]) == n_per \
                and float(z["guidance"]) == guidance:
            q_parts, s_parts, done = [z["q"]], [z["s"]], int(z["done"])
            log(f"resuming: {done} of {len(names)} names already done")

    def save(k):
        tmp = out_path + ".tmp.npz"
        q = np.concatenate(q_parts) if q_parts else np.zeros((0, 1), np.int8)
        s = np.concatenate(s_parts) if s_parts else np.zeros(0, np.float16)
        np.savez(tmp, q=q, s=s, done=np.asarray(k), names=np.asarray(names, dtype=str), n_per=np.asarray(n_per),
                 prompt=np.asarray(prompt), guidance=np.asarray(guidance), steps=np.asarray(steps),
                 model_id=np.asarray(model_id))
        os.replace(tmp, out_path)

    t0, k0 = time.time(), done
    for b, i in enumerate(range(done, len(names), batch)):
        chunk = names[i : i + batch]
        E = prior([prompt.format(n) for n in chunk], n_per=n_per, seed=seed * 1_000_003 + i, guidance=guidance,
                  steps=steps)
        q, s = q8(E)
        q_parts.append(q)
        s_parts.append(s)
        done = i + len(chunk)
        if b == 0 or (b + 1) % save_every == 0 or done >= len(names):
            save(done)
            rate = (time.time() - t0) / max(done - k0, 1)
            log(f"{done}/{len(names)} names, {rate:.2f} s per name, about {(len(names) - done) * rate / 60:.0f} min left")
    save(done)
    return {"names": len(names), "path": out_path}


def load_prior_embeddings(path: str) -> Tuple[torch.Tensor, List[str], int]:
    """(prototypes (N, D): the normalised mean of each name's samples, names, n_per)."""
    from .packs import dq8

    z = np.load(path, allow_pickle=False)
    n_per, names = int(z["n_per"]), list(z["names"])
    E = dq8(z["q"], z["s"])
    k = E.shape[0] // n_per
    P = l2n(E[: k * n_per].reshape(k, n_per, -1).mean(1))
    return P, names[:k], n_per


def fit_image_map(X_src: torch.Tensor, Y_tgt: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Ridge map W (D_src + 1, D_tgt) from unit source image embeddings to unit target image embeddings
    (one backbone to another, same images). apply_image_map adds the bias column."""
    X = torch.cat([l2n(X_src.double()), torch.ones(X_src.shape[0], 1, dtype=torch.float64)], 1)
    Y = l2n(Y_tgt.double())
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=torch.float64)
    A[-1, -1] -= lam                                     # do not shrink the bias
    return torch.linalg.solve(A, X.T @ Y)


def apply_image_map(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    X1 = torch.cat([l2n(X.double()), torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
    return l2n(X1 @ W).float()


def align_by_keys(keys_a: Sequence[str], keys_b: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (ia, ib) of the images present in both lists, in a's order."""
    pos = {k: i for i, k in enumerate(keys_b)}
    ia = np.array([i for i, k in enumerate(keys_a) if k in pos], dtype=np.int64)
    ib = np.array([pos[keys_a[i]] for i in ia], dtype=np.int64)
    return ia, ib


def concept_centroids(feats: torch.Tensor, groups: np.ndarray, n_groups: int, split: Optional[np.ndarray] = None):
    """Unit centroid per group (rows with group -1 ignored). With split (bool per row), returns (A, B) halves."""
    def cent(mask):
        out = torch.zeros(n_groups, feats.shape[1])
        g = torch.as_tensor(groups[mask])
        out.index_add_(0, g, feats[torch.as_tensor(np.nonzero(mask)[0])].float())
        return l2n(out)
    ok = groups >= 0
    if split is None:
        return cent(ok)
    return cent(ok & split), cent(ok & ~split)
