"""Build a fake MyDrive/ReNeg with fake Kaggle outputs, a fake oodlab and the real Colab bundle.

    python tools/colab_smoke/make_fake_drive.py <root>

Then run tools/colab_smoke/run_smoke.py <root> to execute every Colab notebook against it (no Colab, no Kaggle,
no GPU). The fake oodlab (fake_oodlab/) stands in for the real one with a synthetic world (tests/reneg/
synthetic_world.py); the bundle is the repository's own code and notebooks (tools/make_colab_bundle.py).
"""
import json
import os
import shutil
import sys
import zipfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tests", "reneg"))
sys.path.insert(0, os.path.dirname(HERE))
from make_colab_bundle import build_bundle  # noqa: E402
from synthetic_world import make_world  # noqa: E402

N_CORPUS, N_SEL = 69554, 10000


def zip_dir(src_dir, zip_path, arc_root):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fns in os.walk(src_dir):
            for fn in fns:
                if "__pycache__" in dp or fn.endswith(".pyc"):
                    continue
                p = os.path.join(dp, fn)
                z.write(p, os.path.join(arc_root, os.path.relpath(p, src_dir)))


def build(root, seed=0):
    root = os.path.abspath(root)
    drive = os.path.join(root, "MyDrive", "ReNeg")
    shutil.rmtree(root, ignore_errors=True)
    uploads = os.path.join(drive, "uploads")
    os.makedirs(uploads)
    tmp = os.path.join(root, "_build")
    w = make_world(C=1000, D=128, n_neg=3000, imgs_per_class=12, seed=seed, spread=0.5, distortion=0.6, noise=1.0,
                   n_ninco=64, n_tex=26, n_ssb=120)
    rng = np.random.default_rng(seed)
    D = w.dim
    # ---- text bank in oodlab's dict format: first 10,000 corpus rows are the negatives
    pad = N_CORPUS - len(w.neg_names)
    extra = rng.normal(size=(pad, D))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    words = list(w.neg_names) + [f"filler word {i}" for i in range(pad)]
    corpus = np.concatenate([w.neg_text, extra]).astype(np.float32)
    bank = {"format": "oodlab.textbank.v1", "backbone": "ViT-B/16", "id_names": list(w.id_names),
            "id_text": torch.tensor(w.id_text, dtype=torch.float16),
            "id_text_simple": torch.tensor(w.id_text, dtype=torch.float16), "words": words,
            "is_adj": torch.zeros(len(words), dtype=torch.bool), "corpus_text": torch.tensor(corpus, dtype=torch.float16),
            "mining_score": torch.zeros(len(words)), "n_selected": N_SEL, "n_sel_adj": 0,
            "noise_feats": torch.zeros(15, D, dtype=torch.float16), "meta": {"logit_scale": 100.0}}
    # ---- notebook 01's /kaggle/working
    k01 = os.path.join(tmp, "k01")
    cdir = os.path.join(k01, "cache", "vit-b-16")
    os.makedirs(os.path.join(cdir, "images"))
    os.makedirs(os.path.join(cdir, "manifests"))
    torch.save(bank, os.path.join(cdir, "textbank.pt"))
    for name, s in w.sets.items():
        np.savez(os.path.join(cdir, "images", f"{name}.npz"), feats=np.asarray(s["feats"], dtype=np.float16),
                 labels=np.asarray(s["labels"], dtype=np.int64), keys=np.asarray(s["keys"], dtype=str))
        with open(os.path.join(cdir, "images", f"{name}.json"), "w") as f:
            json.dump({"name": name, "n": len(s["keys"]), "dim": D}, f)
    with open(os.path.join(cdir, "CACHE_INFO.json"), "w") as f:
        json.dump({"format": "oodlab.cache.v1"}, f)
    os.makedirs(os.path.join(k01, "results", "01_build_feature_cache"))
    open(os.path.join(k01, "results", "01_build_feature_cache", "report.md"), "w").write("# 01 (fake)\n")
    os.makedirs(os.path.join(k01, "logs"))
    zip_dir(k01, os.path.join(uploads, "01-build-feature-cache-output.zip"), "")
    # ---- fake oodlab code (resources hold the fake "CLIP" text table and the class lists)
    code = os.path.join(tmp, "oodlab_code")
    shutil.copytree(os.path.join(HERE, "fake_oodlab"), code)
    res = os.path.join(code, "oodlab", "resources")
    os.makedirs(res, exist_ok=True)
    json.dump(list(w.id_wnids), open(os.path.join(res, "imagenet_wnids.json"), "w"))
    json.dump(list(w.id_names), open(os.path.join(res, "imagenet_classes.json"), "w"))
    names = list(w.name_to_text)
    np.savez(os.path.join(res, "fake_text.npz"), names=np.asarray(names, dtype=str),
             vecs=np.stack([w.name_to_text[n] for n in names]).astype(np.float32))
    zip_dir(code, os.path.join(uploads, "oodlab_code.zip"), "oodlab_code")
    # ---- notebook 02's output: TANL rows computed by the (fake) runner, as on Kaggle
    sys.path.insert(0, code)
    from oodlab.cache import FeatureCache
    from oodlab.methods import TANL, TANLConfig
    from oodlab.runner import Runner
    from oodlab.textbank import TextBank

    c = FeatureCache(os.path.join(k01, "cache"), "ViT-B/16")
    b = TextBank.load(c.textbank_path)
    df = Runner(c).evaluate(TANL(b.id_text, b.corpus_text, b.noise_feats, TANLConfig.paper()), "four_ood", seeds=[0, 1, 2],
                            config="paper")
    df["precision"] = "fp32"
    k02 = os.path.join(tmp, "k02", "results", "02_reproduce_tanl")
    os.makedirs(k02)
    df.to_csv(os.path.join(k02, "tanl_runs.csv"), index=False)
    zip_dir(os.path.join(tmp, "k02"), os.path.join(uploads, "02-reproduce-tanl-output.zip"), "")
    # ---- the real bundle (this repository's code and notebooks)
    build_bundle(os.path.join(uploads, "ReNeg_Colab_bundle.zip"))
    shutil.rmtree(tmp)
    # zero-shot accuracy of the fake world, for reference
    s = w.sets["imagenet_val"]
    pred = (s["feats"] @ w.id_text.T).argmax(1)
    print("fake drive ready:", drive, "| zero-shot top-1 on fake imagenet_val:", round(100 * float((pred == s["labels"]).mean()), 1))
    return drive


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_drive")
