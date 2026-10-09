"""Plan B probe: draw a few images per label name with a text-to-image model and encode them with our CLIP.

The encoded images give each name an image-space prototype without any fitted map (generate-then-encode,
Plan B5). The probe runs on names whose real images we have (NINCO and SSB-hard classes, plus a few ImageNet
classes as a sanity check), so the generated prototypes can be compared with the real centroids and with the
perfect-map bound measured on the analysis pack.

Nothing here touches test images: generation depends only on the names.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .transport import l2n

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_image_encoder(model, device: str = "cpu", size: int = 224) -> Callable[[list], torch.Tensor]:
    """OpenAI CLIP's own preprocessing (bicubic resize, centre crop, normalise) + model.encode_image, unit rows."""
    import torchvision.transforms as T

    tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size), T.ToTensor(),
                    T.Normalize(CLIP_MEAN, CLIP_STD)])
    dtype = getattr(model, "dtype", None) or next(model.parameters()).dtype

    @torch.no_grad()
    def enc(images: list) -> torch.Tensor:
        x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device, dtype)
        return l2n(model.encode_image(x).float()).cpu()

    return enc


def load_pipeline(model_id: str = "stabilityai/sdxl-turbo", device: str = "cuda", log=print):
    """A diffusers text-to-image pipeline wrapped as gen(prompts, seed) -> list of PIL images.

    SDXL-Turbo and SD-Turbo make a usable image in one step without classifier-free guidance.
    """
    from diffusers import AutoPipelineForText2Image

    kw = dict(torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, variant="fp16", **kw)
    except Exception as e:                                           # noqa: BLE001
        log(f"(no fp16 variant: {type(e).__name__}; loading the default weights)")
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kw)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # SDXL's VAE decodes in float32; decoding a whole batch at 512 px needs more than a T4's 15 GB.
    # Slicing decodes one image at a time (same images, a fraction of the memory).
    for enable in (lambda: pipe.vae.enable_slicing(), lambda: pipe.enable_vae_slicing()):
        try:
            enable()
            break
        except Exception:                                            # noqa: BLE001
            continue

    @torch.inference_mode()
    def gen(prompts: Sequence[str], seed: int, size: int = 512, steps: int = 1) -> list:
        g = torch.Generator(device=device).manual_seed(int(seed))
        out = pipe(prompt=list(prompts), num_inference_steps=steps, guidance_scale=0.0, height=size, width=size,
                   generator=g)
        return out.images

    gen.pipe = pipe
    return gen


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def generate_safely(gen: Callable, prompts: Sequence[str], seed: int, log=print) -> list:
    """gen(prompts) that halves the batch and retries when the GPU runs out of memory."""
    try:
        return gen(list(prompts), seed=seed)
    except Exception as e:                                           # noqa: BLE001
        if not _is_oom(e) or len(prompts) <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = len(prompts) // 2
        log(f"GPU out of memory with {len(prompts)} prompts; retrying in halves")
        return (generate_safely(gen, prompts[:half], seed, log) +
                generate_safely(gen, prompts[half:], seed + 7919, log))


def fake_generator(size: int = 64):
    """Random images, for tests and smoke runs without a GPU."""
    from PIL import Image

    def gen(prompts, seed, **_):
        rng = np.random.default_rng(int(seed))
        return [Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in prompts]

    return gen


def run_probe(gen: Callable, enc: Callable, names: Sequence[str], tags: Sequence[str], folders: Sequence[str],
              out_path: str, n_per: int = 4, batch: int = 8, prompt: str = "a photo of a {}.", seed: int = 0,
              model_id: str = "", save_every: int = 10, log=print) -> Dict:
    """Generate n_per images for every name, encode them, and save (resumably) to out_path (.npz).

    Saved arrays: emb (N, D) fp16 unit rows, name_idx (N,), rep (N,), names, tags, folders, prompt, model_id.
    """
    names, tags, folders = list(names), list(tags), list(folders)
    jobs = [(i, r) for r in range(n_per) for i in range(len(names))]      # round-robin: every name gets 1 image first
    done_emb: List[np.ndarray] = []
    done_idx: List[int] = []
    done_rep: List[int] = []
    if os.path.isfile(out_path):
        z = np.load(out_path, allow_pickle=False)
        if list(z["names"]) == names and str(z["prompt"]) == prompt:
            done_emb, done_idx, done_rep = [z["emb"]], list(z["name_idx"]), list(z["rep"])
            log(f"resuming: {len(done_idx)} of {len(jobs)} images already done")
    have = set(zip(done_idx, done_rep))
    todo = [j for j in jobs if j not in have]

    def save():
        emb = np.concatenate(done_emb) if done_emb else np.zeros((0, 1), np.float16)
        tmp = out_path + ".tmp.npz"
        np.savez(tmp, emb=emb.astype(np.float16), name_idx=np.asarray(done_idx, np.int32),
                 rep=np.asarray(done_rep, np.int16), names=np.asarray(names, dtype=str), tags=np.asarray(tags, dtype=str),
                 folders=np.asarray(folders, dtype=str), prompt=np.asarray(prompt), model_id=np.asarray(model_id))
        os.replace(tmp, out_path)

    t0, n0 = time.time(), len(done_idx)
    for b, s in enumerate(range(0, len(todo), batch)):
        chunk = todo[s : s + batch]
        prompts = [prompt.format(names[i]) for i, _ in chunk]
        images = generate_safely(gen, prompts, seed=seed * 1_000_003 + chunk[0][1] * 10_007 + chunk[0][0], log=log)
        done_emb.append(enc(images).numpy().astype(np.float16))
        done_idx += [i for i, _ in chunk]
        done_rep += [r for _, r in chunk]
        if b == 0 or (b + 1) % save_every == 0 or s + batch >= len(todo):
            save()
            rate = (time.time() - t0) / max(len(done_idx) - n0, 1)
            left = (len(todo) - (s + len(chunk))) * rate
            log(f"{len(done_idx)}/{len(jobs)} images, {rate:.2f} s per image, about {left / 60:.0f} min left")
    save()
    return {"images": len(done_idx), "names": len(names), "path": out_path}


def probe_names(cache, bank, wn=None, ssb_max: Optional[int] = 300, n_imagenet: int = 50, seed: int = 0,
                min_images: int = 6):
    """NINCO's and SSB-hard's class names (from the cache's image keys) plus a few ImageNet classes."""
    from .concepts import build_concepts

    rng = np.random.default_rng(seed)
    names, tags, folders = [], [], []
    for ds in ("ninco", "ssb_hard"):
        if not cache.has(ds):
            continue
        fs = cache.load_set(ds)
        cs = build_concepts(ds, fs.feats, fs.keys, wn=wn, min_images=min_images)
        idx = np.arange(len(cs))
        if ds == "ssb_hard" and ssb_max is not None and len(idx) > ssb_max:
            idx = np.sort(rng.choice(idx, size=ssb_max, replace=False))
        for i in idx:
            names.append(cs.names[i])
            tags.append(ds)
            folders.append(cs.folders[i])
    ids = np.sort(rng.choice(len(bank.id_names), size=min(n_imagenet, len(bank.id_names)), replace=False))
    for i in ids:
        names.append(bank.id_names[i])
        tags.append("imagenet")
        folders.append(str(int(i)))
    return names, tags, folders
