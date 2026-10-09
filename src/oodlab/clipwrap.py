"""CLIP: loading, the official test-time transform, and feature encoders.

Everything here mirrors what OpenOOD-VLM (commit c6fef2f) does for the
``fixedclip_negoodprompt`` network used by TANL / AdaNeg / NegLabel:

* model      ``clip.build_model(state_dict)`` -> fp16 weights -> ``.cuda()``
             (``clip.load(name, device="cuda", jit=False)`` gives the same thing)
* transform  Convert('RGB') -> Resize(256, bicubic) -> CenterCrop(224) -> ToTensor
             -> Normalize(CLIP mean/std)                (preprocessors/test_preprocessor.py)
* image      ``f = encode_image(x); f /= f.norm(dim=-1, keepdim=True)``   (fp16 on GPU)
* text       per label: encode every prompt, L2-normalise each, average, L2-normalise
             again (``text_center=True``)                (networks/clip_fixed_ood_prompt.py)
* noise      15 synthetic noise images (Gaussian, uniform, Poisson, Gamma, salt&pepper)
             fed *without* CLIP normalisation; features L2-normalised.

Because the model runs in fp16 on the GPU, image/text features are fp16 numbers;
storing them as fp16 in the cache is therefore lossless w.r.t. the official pipeline.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Iterable, List, Optional, Sequence

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # same as OpenOOD's ImglistDataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# prompt templates used by OpenOOD-VLM (get_templates) and by NegMining
TEMPLATES = {
    "nice": ["The nice {}."],          # TANL / NegLabel / AdaNeg on ImageNet (scripts: prompt=nice)
    "simple": ["a photo of a {}."],     # MCM script (prompt=simple)
    "vanilla": ["{}."],
}
ADJ_TEMPLATE = "This is a {} photo."    # adjectives of the WordNet corpus (both mining and scoring)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def import_clip():
    """OpenAI CLIP, vendored in ``oodlab/_vendor/clip`` (upstream commit d05afc4, MIT).

    Using the bundled copy means nothing has to be installed from GitHub on Kaggle (``pip install
    git+https://github.com/openai/CLIP.git`` fails when git cannot reach GitHub) and no other package
    called ``clip`` can shadow it. Only ``regex`` is needed (``ftfy`` is optional).
    """
    from ._vendor import clip  # noqa: WPS433

    return clip


def backbone_tag(name: str) -> str:
    """'ViT-B/16' -> 'vit-b-16' (used in folder names)."""
    return name.lower().replace("/", "-").replace("@", "-")


def weights_filename(name: str) -> str:
    clip = import_clip()
    return os.path.basename(clip.clip._MODELS[name])  # the private URL table lives in the submodule


def resolve_weights(name: str, input_roots: Sequence[str] = ()) -> str:
    """Use a checkpoint file from an attached dataset if there is one (e.g. ``ViT-B-16.pt``),
    otherwise return ``name`` so ``clip.load`` downloads it (needs Internet)."""
    if os.path.isfile(name):
        return name
    from .paths import find_file

    try:
        fn = weights_filename(name)
    except KeyError:
        return name
    hit = find_file(fn, input_roots, max_depth=4) if input_roots else None
    return hit or name


def tiny_random_clip(seed: int = 0, embed_dim: int = 64, image_resolution: int = 224):
    """A randomly initialised miniature CLIP with the real tokenizer interface.

    Only for smoke tests (no weights download needed). Features are meaningless.
    """
    CLIP = import_clip().model.CLIP

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CLIP(
            embed_dim=embed_dim, image_resolution=image_resolution, vision_layers=2, vision_width=64,
            vision_patch_size=32, context_length=77, vocab_size=49408,
            transformer_width=64, transformer_heads=2, transformer_layers=2,
        )
    return model


def load_clip(
    name: str = "ViT-B/16",
    device: str = "cuda",
    input_roots: Sequence[str] = (),
    download_root: Optional[str] = None,
    random_init: bool = False,
):
    """Load a frozen CLIP model exactly like the official code.

    On CUDA the weights stay fp16 (as in OpenOOD-VLM); on CPU ``clip.load`` casts
    to fp32 (only used for tests / emergencies).
    """
    if random_init:
        model = tiny_random_clip()
        if str(device).startswith("cuda"):
            import_clip().model.convert_weights(model)
        model = model.to(device)
    else:
        clip = import_clip()
        src = resolve_weights(name, input_roots)
        model, _ = clip.load(src, device=device, jit=False, download_root=download_root)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def model_dtype(model) -> torch.dtype:
    return model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype


def input_resolution(model) -> int:
    return int(model.visual.input_resolution)


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


class Convert:
    def __init__(self, mode: str = "RGB"):
        self.mode = mode

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert(self.mode)


def build_transform(image_size: int = 224, pre_size: int = 256):
    """OpenOOD's TestStandardPreProcessor with CLIP normalisation and bicubic resize."""
    from torchvision import transforms as T

    return T.Compose([
        Convert("RGB"),
        T.Resize(pre_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def load_image(path: str) -> Image.Image:
    """Read bytes then decode (same as ImglistDataset: open -> BytesIO -> Image.open)."""
    import io

    with open(path, "rb") as f:
        buff = io.BytesIO(f.read())
    return Image.open(buff).convert("RGB")


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_image_batch(model, images: torch.Tensor) -> torch.Tensor:
    """L2-normalised image features in the model dtype (official: ``f /= f.norm(...)``)."""
    dev = next(model.parameters()).device
    f = model.encode_image(images.to(dev, non_blocking=True))
    f = f / f.norm(dim=-1, keepdim=True)
    return f


@torch.no_grad()
def encode_prompts(model, prompts: Sequence[str], batch_size: int = 1000, progress: Optional[str] = None,
                   logger=None) -> torch.Tensor:
    """Raw (un-normalised) text-encoder outputs for a list of prompts, model dtype, on the model device."""
    clip = import_clip()
    dev = next(model.parameters()).device
    outs = []
    n = len(prompts)
    for i in range(0, n, batch_size):
        tok = clip.tokenize(list(prompts[i : i + batch_size])).to(dev)
        outs.append(model.encode_text(tok))
        if progress and logger and (i // batch_size) % 20 == 0:
            logger.info(f"[text] {progress}: {min(i + batch_size, n)}/{n}")
    return torch.cat(outs, dim=0) if outs else torch.empty(0, model.text_projection.shape[1], device=dev)


def l2n(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def centered_label_features(raw: torch.Tensor, n_templates: int) -> torch.Tensor:
    """Official ``text_center=True`` recipe from raw encodings ordered label-major.

    raw: (n_labels * n_templates, D) -> (n_labels, D):
    normalise each prompt, average over the templates of a label, normalise again.
    """
    D = raw.shape[-1]
    e = raw.view(-1, n_templates, D)
    e = e / e.norm(dim=-1, keepdim=True)
    c = e.mean(dim=1)
    return c / c.norm(dim=-1, keepdim=True)


@torch.no_grad()
def label_text_features(model, labels: Sequence[str], templates: Sequence[str], batch_size: int = 1000,
                        logger=None, progress: Optional[str] = None) -> torch.Tensor:
    prompts = [t.format(l) for l in labels for t in templates]
    raw = encode_prompts(model, prompts, batch_size=batch_size, logger=logger, progress=progress)
    return centered_label_features(raw, len(templates))


def official_noise_images(seed: int = 0, size: int = 224) -> torch.Tensor:
    """The 15 noise images of FixedCLIP_NegOODPrompt, generated on CPU like the official code.

    The official code does not seed this; we do (``seed``), inside a forked RNG so the
    global random state is untouched.
    """
    shape = (3, 3, size, size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        g = torch.randn(shape, dtype=torch.float)
        g = (g - g.min()) / (g.max() - g.min())
        u = torch.rand(shape, dtype=torch.float)
        p = torch.poisson(5.0 * torch.ones(shape))
        p = (p - p.min()) / (p.max() - p.min())
        ga = torch.distributions.Gamma(2, 1).sample(shape)
        ga = (ga - ga.min()) / (ga.max() - ga.min())
        sp = torch.rand(shape, dtype=torch.float)
        sp[sp >= 0.5] = 1.0
        sp[sp < 0.5] = 0
    return torch.cat((g, u, p, ga, sp), dim=0)


@torch.no_grad()
def noise_features(model, seed: int = 0) -> torch.Tensor:
    x = official_noise_images(seed, size=input_resolution(model))
    dev = next(model.parameters()).device
    f = model.encode_image(x.to(dev))
    return f / f.norm(dim=-1, keepdim=True)


def autocast_ctx(device: str):
    """No autocast: the official model is already fp16 on GPU. Kept for symmetry."""
    return nullcontext()
