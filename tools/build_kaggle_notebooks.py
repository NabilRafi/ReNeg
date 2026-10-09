"""Generate the Kaggle notebooks (00-04, oodlab: feature cache and the baselines) from the cell lists below.

    python tools/build_kaggle_notebooks.py

Edit the cells here (not the .ipynb files) so every notebook shares the same set-up cell.
"""
from __future__ import annotations

import hashlib
import os
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "notebooks", "0_baselines_kaggle")


def md(s: str):
    return new_markdown_cell(textwrap.dedent(s).strip("\n"))


def code(s: str):
    return new_code_cell(textwrap.dedent(s).strip("\n"))


FIND_CODE = '''
# ── Set-up 1/2: find the oodlab code (attached as a private Kaggle dataset) ─────────
import os, sys

def find_oodlab(roots, max_depth=4):
    """Bounded breadth-first search for a folder containing oodlab/__init__.py.
    Never walks image folders (the old notebook hung on a recursive glob over ImageNet)."""
    skip = {"ILSVRC", "train", "val", "test", "images", "__pycache__", "cache", "results", "logs"}
    for root in [r for r in roots if r and os.path.isdir(r)]:
        frontier = [(root, 0)]
        while frontier:
            d, depth = frontier.pop(0)
            if os.path.isfile(os.path.join(d, "oodlab", "__init__.py")):
                return d
            if depth < max_depth:
                try:
                    subs = sorted(e.path for e in os.scandir(d)
                                  if e.is_dir(follow_symlinks=False) and e.name not in skip)
                except OSError:
                    continue
                frontier += [(s, depth + 1) for s in subs]
    return None

ROOTS = [os.environ.get("OODLAB_CODE"), "/kaggle/input", "/kaggle/working", os.getcwd()]
CODE_ROOT = find_oodlab(ROOTS)
if CODE_ROOT is None:   # the dataset was uploaded as a zip that Kaggle did not unpack -> unpack it ourselves
    import zipfile
    for root in [r for r in ROOTS[:2] if r and os.path.isdir(r)]:
        for dp, dn, fns in os.walk(root):
            dn[:] = [d for d in dn if d not in {"ILSVRC", "train", "val", "test", "images"}] if dp.count(os.sep) - root.count(os.sep) < 3 else []
            for fn in fns:
                if fn.startswith("oodlab") and fn.endswith(".zip"):
                    zipfile.ZipFile(os.path.join(dp, fn)).extractall("/kaggle/working/_oodlab_code")
    CODE_ROOT = find_oodlab(["/kaggle/working/_oodlab_code"])
if CODE_ROOT is None:
    raise FileNotFoundError("oodlab code not found: attach your private dataset with the code "
                            "(Add Input -> Your Datasets -> oodlab-code). See GUIDE.md, section 2.")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
import oodlab
print("oodlab", oodlab.__version__, "loaded from", CODE_ROOT)
'''


def setup(name: str, need_gpu: bool, gdown: bool = False) -> list:
    return [
        code(FIND_CODE),
        code(f'''
        # ── Set-up 2/2: packages, folders, logging ─────────────────────────────────────────
        from oodlab.session import start, finish, ensure_packages
        ensure_packages(clip=True, gdown={gdown})      # CLIP ships inside oodlab; this only adds small pip packages if missing
        ctx = start("{name}", need_gpu={need_gpu})
        log = ctx.log                                   # log.info(...) -> screen + /kaggle/working/logs/{name}.log
        import numpy as np, pandas as pd, torch, time, json
        from oodlab.report import show, save_tables, write_report
        '''),
    ]


# ============================================================================
# 00 setup check
# ============================================================================
NB00 = [
    md('''
    # 00 · Setup check

    Run this once per new environment, before spending GPU hours. **≈ 5–10 minutes.**

    **Notebook settings (right-hand panel):** Accelerator **GPU T4 ×2** (or P100) · Internet **On** · Persistence *off* is fine.

    **Inputs to attach (Add Input):**
    1. your private dataset with this code (e.g. `oodlab-code`) — GUIDE.md §2
    2. Competition **ImageNet Object Localization Challenge** (accept its rules on the competition page first)

    Every check prints PASS/FAIL and the notebook keeps going, so you see *all* problems in one run.
    '''),
    *setup("00_setup_check", need_gpu=False, gdown=True),
    code('''
    import traceback, subprocess
    checks = []

    def check(name, fn):
        t0 = time.time()
        try:
            detail = fn()
            checks.append({"check": name, "status": "PASS", "detail": str(detail)[:300], "sec": round(time.time() - t0, 1)})
        except Exception as e:
            log.error(traceback.format_exc())
            checks.append({"check": name, "status": "FAIL", "detail": f"{type(e).__name__}: {e}"[:300], "sec": round(time.time() - t0, 1)})
        print(f"[{checks[-1]['status']}] {name}: {checks[-1]['detail']}")
    '''),
    md('''
    ### 1 · GPU and disk
    '''),
    code('''
    def gpu():
        if ctx.smoke:
            return "skipped in smoke mode"
        assert torch.cuda.is_available(), "no GPU: Settings -> Accelerator -> GPU"
        p = torch.cuda.get_device_properties(0)
        return f"{torch.cuda.device_count()} x {p.name}, {p.total_memory / 1e9:.1f} GB, torch {torch.__version__}"
    check("GPU", gpu)

    def disk():
        import shutil
        free = {d: shutil.disk_usage(d).free / 1e9 for d in [ctx.dirs.work, ctx.dirs.scratch]}
        msg = ", ".join(f"{k}: {v:.0f} GB free" for k, v in free.items())
        assert ctx.smoke or free[ctx.dirs.scratch] > 25, "need > 25 GB of scratch space for the OpenOOD downloads"
        return msg
    check("disk space", disk)
    '''),
    md('''
    ### 2 · Unit tests (synthetic data, CPU) and GPU equivalence with the official code

    The tests run the **vendored official** TANL / AdaNeg / NegLabel postprocessors and our
    re-implementations on the same stream and require identical results (≈1e-6 in fp32, bit-exact in fp16).
    '''),
    code('''
    def unit_tests():
        try:
            import pytest  # noqa: F401
        except ImportError:
            from oodlab.session import pip_install
            pip_install("pytest")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider",
                            os.path.join(CODE_ROOT, "tests")], capture_output=True, text=True, cwd=ctx.dirs.scratch)
        out = (r.stdout + r.stderr).strip().splitlines()
        assert r.returncode == 0, "\\n".join(out[-40:])
        return out[-1]
    check("unit tests", unit_tests)

    def gpu_equivalence():
        from oodlab.synthetic import make_bank, make_images
        from oodlab.methods import TANLConfig, AdaNegConfig
        from oodlab.official.equivalence import compare_tanl, compare_adaneg
        dev = ctx.device
        b = make_bank(C=100, N=5000, D=512, n_neglabel=1000)
        fi, _ = make_images(b, 3000, "id", seed=1); fo, _ = make_images(b, 2000, "ood", seed=2)
        x = torch.cat([fo, fi])[torch.randperm(5000, generator=torch.Generator().manual_seed(0))]
        batches = list(x.split(256))
        cfg = TANLConfig.paper().with_(num_neg=300, queue_len=100)
        r32 = compare_tanl(b.id_text, b.corpus, b.noise, batches, 1000, cfg, device=dev)
        h = lambda t: t.half()
        r16 = compare_tanl(h(b.id_text), h(b.corpus), h(b.noise), [h(t) for t in batches], 1000,
                           cfg.with_(emulate_fp16=True, exact_fp16_final=True), device=dev)
        a16 = compare_adaneg(h(b.id_text), h(b.corpus[:1000]), [h(t) for t in batches], AdaNegConfig(emulate_fp16=True), device=dev)
        # fp16: bit-identical is expected; allow ~1 fp16 ulp in case a GPU picks another kernel
        ok = (r32["max_abs_diff"] < 1e-4 and r16["max_abs_diff"] <= 2e-3 and a16["max_abs_diff"] <= 2e-3
              and min(r32["pred_agreement"], r16["pred_agreement"], a16["pred_agreement"]) >= 0.999)
        assert ok, (r32, r16, a16)
        exact = "bit-identical" if r16["max_abs_diff"] == 0 and a16["max_abs_diff"] == 0 else "within 1 fp16 ulp"
        return f"{dev}: TANL fp32 max diff {r32['max_abs_diff']:.1e}; fp16 TANL/AdaNeg {exact}"
    check("official == ours on " + ctx.device, gpu_equivalence)
    '''),
    md('''
    ### 3 · ImageNet from the Kaggle competition

    Kaggle's validation folder is flat; labels come from `LOC_val_solution.csv`. The class order is
    checked against OpenOOD-VLM's (`all_wnids`), so label *i* ↔ class name *i* of the text prompts.
    '''),
    code('''
    from oodlab.data.imagenet import locate_imagenet, imagenet_val_manifest
    state = {}

    def imagenet():
        loc = locate_imagenet(ctx.input_roots)
        m = imagenet_val_manifest(loc, expected=None if ctx.smoke else 50000)
        miss = m.missing_files(sample=300)
        assert not miss, f"missing files, e.g. {miss[:3]}"
        state["imagenet"] = m
        return f"{loc.base} ({loc.layout}), {len(m)} images, {len(set(m.labels.tolist()))} classes"
    check("ImageNet val on Kaggle", imagenet)
    '''),
    md('''
    ### 4 · CLIP ViT-B/16 and a zero-shot sanity check

    Loads CLIP exactly like the official code (fp16 on GPU) and classifies 1,000 random validation
    images with the prompt *"The nice {class}."*. Expect roughly **64–70 %** top-1 (ViT-B/16 zero-shot
    is ≈ 67 % on the full set; 1,000 images give ± 3 %).
    '''),
    code('''
    from oodlab.clipwrap import load_clip, label_text_features, TEMPLATES, build_transform, encode_image_batch, load_image

    def clip_check():
        model = load_clip("ViT-B/16", device=ctx.device, input_roots=ctx.input_roots, random_init=ctx.smoke)
        state["model"] = model
        ls = model.logit_scale.exp().item()
        return f"dtype {model.dtype}, input {model.visual.input_resolution}px, logit scale {ls:.6f}"
    check("CLIP ViT-B/16 loads", clip_check)

    def zero_shot():
        model, m = state["model"], state["imagenet"]
        text = label_text_features(model, oodlab.imagenet_classnames(), TEMPLATES["nice"])
        idx = np.random.default_rng(0).choice(len(m), size=min(1000, len(m)), replace=False)
        sub = m.subset(idx)
        tf = build_transform(model.visual.input_resolution, round(model.visual.input_resolution * 256 / 224))

        class DS(torch.utils.data.Dataset):
            def __len__(self): return len(sub)
            def __getitem__(self, i): return tf(load_image(sub.paths[i])), int(sub.labels[i])

        dl = torch.utils.data.DataLoader(DS(), batch_size=250, num_workers=0 if ctx.smoke else 4)
        correct, n, t0 = 0, 0, time.time()
        for x, y in dl:
            f = encode_image_batch(model, x)
            correct += int(((f @ text.t()).argmax(1).cpu() == y).sum()); n += len(y)
        acc = 100 * correct / n
        rate = n / (time.time() - t0)
        assert ctx.smoke or 60 <= acc <= 75, f"zero-shot top-1 {acc:.1f}% is outside 60-75%: check labels/transform"
        return f"top-1 {acc:.1f}% on {n} images, {rate:.0f} img/s (decode + encode)"
    check("zero-shot accuracy", zero_shot)
    '''),
    md('''
    ### 5 · Google Drive (OpenOOD downloads)

    Downloads OpenOOD's small image-list archive. If this fails with a quota message, Drive is
    rate-limiting the file: use the manual route in GUIDE.md §5 (upload the zips as a Kaggle dataset).
    '''),
    code('''
    def drive():
        if ctx.smoke:
            return "skipped in smoke mode"
        from oodlab.data.openood import gdown_zip
        import zipfile
        z = gdown_zip("benchmark_imglist", os.path.join(ctx.dirs.scratch, "zips"), retries=2, wait_s=10, logger=log)
        names = zipfile.ZipFile(z).namelist()
        assert any(n.endswith("imagenet/test_ninco.txt") for n in names), "unexpected archive content"
        return f"{os.path.getsize(z) / 1e6:.1f} MB, {len(names)} entries"
    check("Google Drive download", drive)
    '''),
    md('''
    ### Summary
    '''),
    code('''
    table = pd.DataFrame(checks)
    show(table, "setup checks")
    table.to_csv(ctx.path("checks.csv"), index=False)
    n_fail = int((table.status == "FAIL").sum())
    finish(ctx, {"failed_checks": n_fail})
    print("ALL CHECKS PASSED - continue with notebook 01" if n_fail == 0 else f"{n_fail} check(s) FAILED - see GUIDE.md §8 (troubleshooting)")
    '''),
]


# ============================================================================
# 01 build feature cache
# ============================================================================
NB01 = [
    md('''
    # 01 · Build the CLIP feature cache

    Encodes every ID and OOD image **once** with CLIP ViT-B/16 (fp16, exactly like OpenOOD-VLM) and computes
    all text-side features. Every later notebook replays test streams from this cache — no images needed.

    **Settings:** GPU T4 ×2 / P100 · Internet On. **Inputs:** `oodlab-code`, the ImageNet competition, and
    (optional) a dataset with OpenOOD zips if Google Drive refuses downloads (GUIDE.md §5).
    **Time:** ≈ 1–2 h (mostly JPEG decoding and downloads). **Run it with "Save Version → Save & Run All"**
    so it survives a closed browser.

    **Output** (`/kaggle/working/cache/vit-b-16/`, ≈ 0.3 GB):

    | file | content |
    |---|---|
    | `textbank.pt` | 1,000 ID prompts ("The nice …", "a photo of a …"), the 69,554-word WordNet corpus in the official layout, NegLabel's 10,000 mined negatives, 15 noise-image features |
    | `images/<set>.npz` | fp16 features, labels (−1 = OOD), image keys |
    | `manifests/<set>.csv` | the exact image list that was encoded |

    **If the session dies:** add this notebook's own previous output as an input and run again — finished
    sets are imported and unfinished ones resume from their last 10k-image shard.
    '''),
    *setup("01_build_feature_cache", need_gpu=True, gdown=True),
    md('''
    ### Settings
    '''),
    code('''
    BACKBONE = "ViT-B/16"
    SETS = ["imagenet_val_all", "imagenet_test", "imagenet_val",      # ID (Kaggle ImageNet val)
            "ninco", "textures_all", "textures", "inaturalist", "sun", "places",
            "openimage_o", "openimage_o_val", "ssb_hard"]               # OOD (OpenOOD v1.5 downloads)
    BATCH_SIZE = 256
    NUM_WORKERS = min(4, os.cpu_count() or 2)   # Kaggle GPU machines have 4 vCPUs
    SHARD_SIZE = 10_000
    TOTAL_NEG = 10_000                           # NegLabel's mined negatives (official ood_number)
    LIVE_CHECK = True                            # official TANL on live images vs our TANL on the cache
    LIVE_CHECK_N_ID = 5_000
    CORPUS_LIMIT = None
    # If Google Drive refuses a file, copy it to your own Drive ("Make a copy"), share it with
    # "anyone with the link", and put the copy's id here, e.g. {"ssb_hard": "1AbC..."}  (GUIDE.md §5)
    DRIVE_ID_OVERRIDES = {}
    from oodlab.data.openood import DRIVE_IDS
    DRIVE_IDS.update(DRIVE_ID_OVERRIDES)
    if ctx.smoke:
        BATCH_SIZE, NUM_WORKERS, SHARD_SIZE, TOTAL_NEG, LIVE_CHECK_N_ID, CORPUS_LIMIT = 32, 0, 150, 500, 100, 2000
    '''),
    md('''
    ### 1 · Resume: import anything a previous run already finished
    '''),
    code('''
    from oodlab.cache import FeatureCache, find_existing_caches, import_from, encode_manifest
    cache = FeatureCache(ctx.dirs.cache, BACKBONE).make()
    for root in find_existing_caches([ctx.dirs.inputs]):
        import_from(root, cache, logger=log)
    log.info(f"cache at {cache.dir}; already done: {cache.available()} textbank={cache.has_textbank()}")
    '''),
    md('''
    ### 2 · CLIP
    '''),
    code('''
    from oodlab.clipwrap import load_clip
    model = load_clip(BACKBONE, device=ctx.device, input_roots=ctx.input_roots, random_init=ctx.smoke)
    log.info(f"CLIP {BACKBONE}: dtype {model.dtype}, logit scale {model.logit_scale.exp().item():.6f}")
    '''),
    md('''
    ### 3 · Text bank (≈ 1 min on GPU)

    Word lists are NegLabel's WordNet files shipped inside the package (`oodlab/resources/neglabel_txtfiles`,
    Apache-2.0). De-duplication follows the official code and must give **11,453 adjectives + 58,101 nouns
    = 69,554 words**; NegMining (95th-percentile cosine to the ID names, ViT-B/16) keeps **1,646 + 8,354**.
    '''),
    code('''
    from oodlab.textbank import build_textbank, OFFICIAL_N_ADJ, OFFICIAL_N_NOUN
    if not cache.has_textbank():
        bank = build_textbank(model, BACKBONE, oodlab.imagenet_classnames(), oodlab.CORPUS_DIR,
                              total_neg=TOTAL_NEG, corpus_limit=CORPUS_LIMIT, logger=log)
        cache.save_textbank(bank)
    bank = cache.load_textbank()
    print(json.dumps(bank.summary(), indent=1))
    st = bank.meta["stats"]
    if not ctx.smoke:
        assert (st["n_adj"], st["n_noun"]) == (OFFICIAL_N_ADJ, OFFICIAL_N_NOUN), st
        assert bank.n_sel_adj == 1646 and bank.n_selected == 10000, (bank.n_sel_adj, bank.n_selected)
    print("most OOD-like selected words:", bank.words[:8], bank.words[bank.n_sel_adj:bank.n_sel_adj + 8])
    print("closest-to-ID unselected words:", bank.words[-8:])
    '''),
    md('''
    ### 4 · Image lists and images, one set at a time

    For each set: build the manifest (downloading and extracting the OpenOOD zip if needed), check files
    exist, encode, save, and delete the extracted images to keep disk use low. A failure (e.g. a Drive
    quota error) is logged and the loop moves on; re-run the notebook later to fill the gaps.
    '''),
    code('''
    from oodlab.data import build_manifest, SPECS
    from oodlab.data.openood import obtain_imglist_dir, cleanup
    try:
        imglist_dir = obtain_imglist_dir(ctx.dirs.scratch, ctx.input_roots, logger=log)
        log.info(f"OpenOOD image lists: {imglist_dir}: {sorted(os.listdir(imglist_dir))}")
    except Exception as e:   # sets that need a list will fail individually below; the others still run
        log.exception("OpenOOD image lists unavailable (GUIDE.md §5)")
        imglist_dir = None

    def cleanup_images(key):
        users = [n for n in SETS if SPECS[n].drive_key == key]
        if key and all(cache.has(n) for n in users) and not (LIVE_CHECK and key == "ninco"):
            cleanup(os.path.join(ctx.dirs.scratch, "extracted", key))
            cleanup(os.path.join(ctx.dirs.scratch, "zips", f"{key}.zip"))

    status, imagenet_memo = [], {}
    for name in SETS:
        t0 = time.time()
        if cache.has(name):
            status.append({"set": name, "status": "cached", "images": len(cache.load_set(name)), "min": 0.0})
            continue
        try:
            man = build_manifest(name, ctx.input_roots, ctx.dirs.scratch, imglist_dir, logger=log,
                                 strict_counts=not ctx.smoke, _imagenet_cache=imagenet_memo)
            missing = man.missing_files(sample=500)
            if missing:
                raise FileNotFoundError(f"{len(missing)} of 500 sampled files missing, e.g. {missing[:2]}")
            fs = encode_manifest(model, man, cache, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                 shard_size=SHARD_SIZE, logger=log)
            status.append({"set": name, "status": "encoded", "images": len(fs), "expected": SPECS[name].expected,
                           "min": round((time.time() - t0) / 60, 1)})
        except Exception as e:
            log.exception(f"[{name}] FAILED")
            status.append({"set": name, "status": f"FAILED: {type(e).__name__}: {str(e)[:150]}", "min": round((time.time() - t0) / 60, 1)})
        finally:
            cleanup_images(SPECS[name].drive_key)
        log.info(f"[{name}] {status[-1]['status']} ({ctx.elapsed()} since start)")
    show(pd.DataFrame(status), "build status")
    '''),
    md('''
    ### 5 · Fi-ImageNet-1k (optional, when you have the authors' lists)

    Its images are ImageNet validation images, already encoded above, so nothing new is encoded. Put
    `fi_imagenet_1k_ood.txt` (655 images) and `fi_imagenet_1k_ambiguous.txt` (980) in any attached dataset.
    '''),
    code('''
    from oodlab.data.fi_imagenet import find_lists, read_val_ids, fi_imagenet_sets
    import shutil
    ood_list, amb_list = find_lists(ctx.input_roots)
    if ood_list and cache.has("imagenet_val_all"):
        id_clean, fi_ood = fi_imagenet_sets(cache.load_set("imagenet_val_all"), read_val_ids(ood_list),
                                            read_val_ids(amb_list) if amb_list else [], logger=log)
        os.makedirs(os.path.join(cache.dir, "fi_imagenet_1k"), exist_ok=True)
        for p in [ood_list, amb_list]:
            if p:
                shutil.copy2(p, os.path.join(cache.dir, "fi_imagenet_1k", os.path.basename(p)))
    else:
        print("Fi-ImageNet-1k lists not attached - skipped (expected until the data is released).")
    '''),
    md('''
    ### 6 · Acceptance check: live run vs cache replay

    The **official** TANL postprocessor runs on images decoded and encoded on the fly; **our** TANL runs on
    the cached features; both see the same stream (all NINCO images + 5,000 ImageNet images). The plan's
    criterion: FPR95 and AUROC agree to two decimals. On GPU the scores should be bit-identical.
    '''),
    code('''
    live = None
    if LIVE_CHECK and cache.has("ninco") and cache.has("imagenet_val_all"):
        try:   # a failure here must not lose the hours of encoding above
            from oodlab.official.live import live_vs_cache
            from oodlab.data.manifest import Manifest
            ninco_man = build_manifest("ninco", ctx.input_roots, ctx.dirs.scratch, imglist_dir, logger=log,
                                       strict_counts=not ctx.smoke)   # re-extracts NINCO if it was imported
            live = live_vs_cache(model, bank, Manifest.load(cache.manifests_dir, "imagenet_val_all"),
                                 cache.load_set("imagenet_val_all"), ninco_man, cache.load_set("ninco"),
                                 n_id=LIVE_CHECK_N_ID, device=ctx.device, num_workers=NUM_WORKERS, logger=log)
            oodlab.utils.write_json(live, ctx.path("live_vs_cache.json"))
            print("LIVE == CACHE (two decimals):", live["pass"], "| identical scores:", live["max_abs_score_diff"] == 0)
        except Exception as e:
            log.exception("live-vs-cache check failed (the cache itself is fine; re-run this cell or notebook 01 later)")
            live = {"error": f"{type(e).__name__}: {e}"}
        finally:
            cleanup(os.path.join(ctx.dirs.scratch, "extracted", "ninco"))
    else:
        print("live check skipped")
    '''),
    md('''
    ### 7 · Summary
    '''),
    code('''
    summary = pd.DataFrame(cache.summary())
    show(summary, "feature cache")
    summary.to_csv(ctx.path("cache_summary.csv"), index=False)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(cache.dir) for f in fs)
    missing = [n for n in SETS if not cache.has(n)]
    finish(ctx, {"cache_dir": cache.dir, "cache_GB": round(size / 1e9, 3), "missing_sets": missing,
                 "live_check": live})
    print(f"cache size {size / 1e9:.2f} GB; missing sets: {missing or 'none'}")
    print("NEXT: 'Save Version' finished -> attach this notebook's output to notebooks 02-04.")
    '''),
]


# ============================================================================
# 02 reproduce TANL
# ============================================================================
NB02 = [
    md('''
    # 02 · Reproduce TANL (CVPR 2026) from the feature cache

    **Acceptance (pre-G1 plan):** Four-OOD FPR95 within **±1** of **9.81**; OpenOOD v1.5 near / far FPR95
    close to **60.06 / 17.21** (ViT-B/16, M = 1,000, L = 300).

    **What runs** (all from cached features, 3 stream seeds):

    1. equivalence of our TANL and the official postprocessor on *real* features;
    2. Four-OOD, OpenOOD v1.5 near/far, and near-OOD with the 50k ID set;
    3. two hyper-parameter sets — **paper** (g = 0.2, α = 0.95, as in Sec. 4.1) and **official_sh**
       (g = 0.5, α = 0, what `scripts/ood/TANL/official.sh` runs) — each in **fp32** and **fp16**
       (bit-exact emulation of the official GPU arithmetic);
    4. temporal shift (Tab. A15) and ID-first / OOD-first orders (Tab. A16);
    5. a worked example of modifying TANL (`TANL_LCB`, one overridden method).

    **Settings:** GPU recommended (≈ 30–45 min; CPU works for fp32 only, several hours). Internet not needed.
    **Inputs:** `oodlab-code` + the output of notebook 01.
    '''),
    *setup("02_reproduce_tanl", need_gpu=False),
    md('''
    ### Settings
    '''),
    code('''
    from oodlab.methods import TANL, TANLConfig
    from oodlab.runner import Runner, summarize, compare_with_targets, acceptance_table
    from oodlab.report import find_cache

    BACKBONE = "ViT-B/16"
    SEEDS = [0, 1, 2]
    CONFIGS = {"paper": TANLConfig.paper(), "official_sh": TANLConfig.official_sh()}
    PRECISIONS = ["fp32", "fp16"] if ctx.device == "cuda" else ["fp32"]   # fp16 emulation is slow on CPU
    RUN = dict(equivalence=True, four_ood=True, openood_v15=True, near_50k=True, temporal=True, orders=True, variant_demo=True)
    STREAM_CONFIG = "auto"           # config for temporal / order runs; "auto" = whichever matches 9.81 best
    if ctx.smoke:
        SEEDS = [0, 1]
        CONFIGS = {k: v.with_(num_neg=100) for k, v in CONFIGS.items()}
    '''),
    code('''
    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    LS = bank.logit_scale
    print(json.dumps(bank.summary(), indent=1))
    runner = Runner(cache, device=ctx.device, logger=log, keep_orders=True)

    def make_tanl(cfg, precision, cls=TANL, **kw):
        c = cfg.with_(logit_scale=LS)
        if precision == "fp16":
            c = c.with_(emulate_fp16=True, exact_fp16_final=True)
        return cls(bank.id_text, bank.corpus_text, bank.noise_feats, c, device=ctx.device, n_neglabel=bank.n_selected, **kw)

    runs = []   # every per-(dataset, seed) row goes here
    '''),
    md('''
    ### 1 · Our TANL == official TANL on real features

    Same stream (4,000 ImageNet + 2,000 iNaturalist images) through the vendored official postprocessor and
    through our class. Expect max |Δscore| ≈ 1e-6 in fp32 and exactly 0 in fp16.
    '''),
    code('''
    eq_sets = [n for n in ["inaturalist", "ninco", "sun", "places", "textures_all", "ssb_hard", "openimage_o"] if cache.has(n)]
    if RUN["equivalence"] and cache.has("imagenet_val_all") and eq_sets:
        from oodlab.official.equivalence import compare_tanl
        from oodlab.stream import pair_stream
        ids, ood = cache.load_set("imagenet_val_all"), cache.load_set(eq_sets[0])
        n_id, n_ood = min(4000, len(ids)), min(2000, len(ood))
        s = pair_stream(ids.subset(range(n_id)), ood.subset(range(n_ood)), seed=0)
        batches = [s.feats(a, b) for _, a, b, _ in s.batches()]
        cfg = CONFIGS["paper"].with_(logit_scale=LS)
        eq = {"fp32": compare_tanl(bank.id_text.float(), bank.corpus_text.float(), bank.noise_feats.float(),
                                   [b.float() for b in batches], bank.n_selected, cfg, device=ctx.device)}
        if "fp16" in PRECISIONS:
            eq["fp16"] = compare_tanl(bank.id_text.half(), bank.corpus_text.half(), bank.noise_feats.half(),
                                      [b.half() for b in batches], bank.n_selected,
                                      cfg.with_(emulate_fp16=True, exact_fp16_final=True), device=ctx.device)
        show(pd.DataFrame(eq).T, f"official vs ours (real features: ImageNet + {eq_sets[0]})", digits=8)
        ok32 = eq["fp32"]["max_abs_diff"] < 1e-4 and eq["fp32"]["pred_agreement"] >= 0.999
        ok16 = "fp16" not in eq or (eq["fp16"]["max_abs_diff"] <= 2e-3 and eq["fp16"]["pred_agreement"] >= 0.999)
        if ok32 and ok16:
            print("EQUIVALENCE OK", "(fp16 bit-identical)" if eq.get("fp16", {}).get("max_abs_diff", 1) == 0 else "")
        else:
            log.warning("EQUIVALENCE CHECK FAILED - results below are still computed, but investigate before trusting them")
        oodlab.utils.write_json(eq, ctx.path("equivalence_real_features.json"))
    else:
        print("equivalence check skipped (needs imagenet_val_all and one OOD set in the cache)")
    '''),
    md('''
    ### 2 · Benchmarks
    '''),
    code('''
    for protocol in ["four_ood", "openood_v15", "near_50k"]:
        if not RUN[protocol]:
            continue
        miss = runner.missing(protocol)
        if miss:
            log.warning(f"skip {protocol}: sets not cached yet {miss}")
            continue
        for cname, cfg in CONFIGS.items():
            for prec in PRECISIONS:
                runs.append(runner.evaluate(make_tanl(cfg, prec), protocol, SEEDS, config=f"{cname}/{prec}"))
    if not runs:
        raise RuntimeError(f"no protocol could run - cached sets: {cache.available()}; finish notebook 01 first")
    df = pd.concat(runs, ignore_index=True)
    summary = summarize(df)
    show(summary[summary.key.str.startswith("mean:")],
         "group means (FPR95 lower is better)", cols=["config", "protocol", "key", "fpr95", "fpr95_std", "auroc", "fpr95_idpos", "acc"])
    '''),
    code('''
    for protocol in ["four_ood", "openood_v15"]:
        show(compare_with_targets(summary, "tanl", protocol), f"TANL vs paper: {protocol}")
    acc_table = acceptance_table(summary)
    show(acc_table, "pre-G1 acceptance (TANL)")
    '''),
    md('''
    ### 3 · Temporal shift (Tab. A15) and sample order (Tab. A16)

    Temporal shift: the four OOD sets arrive one after another (each mixed with the full ID set) and
    the queues are **not** reset between them. Sample order: all ID first, or all OOD first.
    '''),
    code('''
    from oodlab.report import best_tanl_config
    stream_rows = []
    best_prec = "fp16" if "fp16" in PRECISIONS else "fp32"
    if STREAM_CONFIG == "auto":
        STREAM_CONFIG = best_tanl_config(summary, precision=best_prec) or "paper"
        log.info(f"temporal / order experiments use the '{STREAM_CONFIG}' configuration")
    if RUN["temporal"] and not runner.missing("four_ood"):
        t = runner.evaluate_temporal(make_tanl(CONFIGS[STREAM_CONFIG], best_prec), seeds=SEEDS,
                                     config=f"{STREAM_CONFIG}/{best_prec}")
        stream_rows.append(t)
    if RUN["orders"] and not runner.missing("four_ood"):
        for order in ["id_first", "ood_first"]:
            stream_rows.append(runner.evaluate(make_tanl(CONFIGS[STREAM_CONFIG], best_prec), "four_ood", SEEDS,
                                               config=f"{STREAM_CONFIG}/{best_prec}", order=order, tag=f"order:{order}"))
    if stream_rows:
        sdf = pd.concat(stream_rows, ignore_index=True)
        ssum = summarize(sdf)
        for protocol in sorted(ssum.protocol.unique()):
            show(compare_with_targets(ssum, "tanl", protocol), f"TANL vs paper: {protocol}")
        show(sdf[sdf.protocol.str.startswith("temporal")].groupby(["protocol", "position", "dataset"])[["fpr95", "post_shift_fpr95"]].mean().reset_index(),
             "temporal shift: whole-segment FPR95 and FPR95 on the first 2,560 samples after each shift")
    else:
        sdf, ssum = pd.DataFrame(), pd.DataFrame()
    '''),
    md('''
    ### 4 · Modifying TANL: a worked example

    Every step of TANL is a method you can override (`activation`, `select`, `score`, `threshold`, `admit`).
    `oodlab/methods/variants.py` contains `TANL_LCB`, which ranks labels by *activation difference − κ·SE*
    (a lower confidence bound) instead of the plain difference — about ten lines. `κ = 0` is exactly TANL.
    Write your own variant the same way, in that file or right here in a cell.
    '''),
    code('''
    if RUN["variant_demo"] and not runner.missing("four_ood"):
        from oodlab.methods.variants import TANL_LCB
        demo = [runner.evaluate(make_tanl(CONFIGS["paper"], "fp32", cls=TANL_LCB, kappa=k), "four_ood", SEEDS[:1],
                                config=f"paper/fp32/lcb{k}") for k in [0.0, 1.0]]
        demo_sum = summarize(pd.concat(demo, ignore_index=True))
        show(demo_sum[demo_sum.key == "mean:four"], "TANL_LCB example (seed 0)", cols=["method", "config", "fpr95", "auroc"])
    '''),
    md('''
    ### 5 · Save
    '''),
    code('''
    all_df = pd.concat([df] + ([sdf] if len(sdf) else []), ignore_index=True)
    all_sum = summarize(all_df)
    save_tables(ctx, tanl_runs=all_df, tanl_summary=all_sum, tanl_acceptance=acc_table)
    runner.save_orders(ctx.path("stream_orders.npz"))
    sections = [("Acceptance", acc_table)]
    for p in ["four_ood", "openood_v15", "near_50k"] + sorted(ssum.protocol.unique() if len(ssum) else []):
        c = compare_with_targets(all_sum, "tanl", p)
        if len(c):
            sections.append((f"TANL vs published: {p}", c))
    sections.append(("All group means", all_sum[all_sum.key.str.startswith("mean:")][
        ["config", "protocol", "key", "fpr95", "fpr95_std", "auroc", "auroc_std", "fpr95_idpos", "acc", "n_seeds"]]))
    write_report(ctx, "TANL reproduction", sections)
    finish(ctx, {"acceptance": acc_table.to_dict("records")})
    '''),
]


# ============================================================================
# 03 churn diagnostic
# ============================================================================
NB03 = [
    md('''
    # 03 · Churn diagnostic for TANL's negative-label selection

    Measures how TANL's set of M = 1,000 selected negatives evolves over the test stream, **without changing
    anything TANL computes** (checked in step 1: scores with and without the tracker are bit-identical, so the
    diagnostic figures and the headline numbers come from the same run).

    Per batch: **churn** (1 − overlap with the previous batch's set) · **survival** of the first-batch set and
    of the initial text/noise set · **overlap with the oracle** set (top-M by the *true* OOD−ID activation
    difference of the stream) · distinct labels used so far · **queue purity** (true-ID share of the ID
    queue, true-OOD share of the OOD queue, by mirroring the FIFOs with ground truth) · admission precision ·
    the automatic threshold. Per run: dwell times and batches to reach 90 % of the oracle-overlap plateau;
    for temporal-shift streams the same after every shift.

    **Decision rule (plan):** final survival of the first-batch selection **> 50 %** supports the cold-start
    framing; **< 10 %** means switching to a shift-only framing before writing.

    **Settings:** GPU recommended (≈ 20–30 min). **Inputs:** `oodlab-code` + notebook 01's output.
    '''),
    *setup("03_churn_diagnostic", need_gpu=False),
    code('''
    from oodlab.methods import TANL, TANLConfig
    from oodlab.runner import Runner, run_stream, summarize, seed_method
    from oodlab.stream import pair_stream, temporal_stream, TEMPORAL_ORDERS
    from oodlab.diagnostics import ChurnTracker, oracle_selection, assert_passive, plot_run, plot_dwell, decision
    from oodlab.report import find_cache

    from oodlab.report import best_tanl_config
    BACKBONE = "ViT-B/16"
    CONFIG_NAME = "auto"   # "paper" | "official_sh" | "auto" (= the one closest to 9.81 in notebook 02, if attached)
    PRECISION = "fp16" if ctx.device == "cuda" else "fp32"
    if CONFIG_NAME == "auto":
        prev = oodlab.paths.find_file("tanl_summary.csv", [ctx.dirs.results, ctx.dirs.inputs], max_depth=5)
        CONFIG_NAME = (best_tanl_config(prev, precision=PRECISION) if prev else None) or "paper"
    CONFIG = {"paper": TANLConfig.paper(), "official_sh": TANLConfig.official_sh()}[CONFIG_NAME]
    log.info(f"TANL configuration for the diagnostic: {CONFIG_NAME} ({PRECISION})")
    SEEDS = [0, 1, 2]
    DEFAULT_STREAMS = [("imagenet_val_all", ["inaturalist", "sun", "places", "textures_all"]),   # Four-OOD
                       ("imagenet_test", ["ssb_hard", "ninco"])]                                  # OpenOOD near
    RUN_TEMPORAL = True
    if ctx.smoke:
        CONFIG, SEEDS = CONFIG.with_(num_neg=100), [0, 1]

    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    runner = Runner(cache, device=ctx.device, logger=log)

    def make(record):
        c = CONFIG.with_(logit_scale=bank.logit_scale, record=record)
        if PRECISION == "fp16":
            c = c.with_(emulate_fp16=True, exact_fp16_final=True)
        return TANL(bank.id_text, bank.corpus_text, bank.noise_feats, c, device=ctx.device, n_neglabel=bank.n_selected)
    os.makedirs(ctx.path("plots", "x"), exist_ok=True)
    '''),
    md('''
    ### 1 · The tracker is passive
    '''),
    code('''
    pairs = [(i, o) for i, oods in DEFAULT_STREAMS for o in oods if cache.has(i) and cache.has(o)]
    if not pairs:
        raise RuntimeError(f"no ID/OOD pair in the cache ({cache.available()}); finish notebook 01 first")
    s = pair_stream(runner.get(pairs[0][0]), runner.get(pairs[0][1]), seed=0).to(ctx.device)
    print(assert_passive(make(False), make(True), s, device=ctx.device))
    del s
    '''),
    md('''
    ### 2 · Default (shuffled) streams, one reset per ID/OOD pair
    '''),
    code('''
    rows, frames = [], {}
    for id_name, oods in DEFAULT_STREAMS:
        for ood_name in oods:
            if any(n not in cache.available() for n in [id_name, ood_name]):
                log.warning(f"skip {ood_name}: not cached"); continue
            for seed in SEEDS:
                s = pair_stream(runner.get(id_name), runner.get(ood_name), seed=seed).to(ctx.device)
                m = make(True); seed_method(m, seed); m.reset()
                tr = ChurnTracker({0: oracle_selection(m, s, 0, device=ctx.device)}, name=f"{ood_name}/s{seed}")
                res = run_stream(m, s, ctx.device, hooks=[tr])
                met = res.segment_metrics(0)
                summ = tr.summary()
                rows.append({"id": id_name, "ood": ood_name, "seed": seed, "fpr95": met["fpr95"], "auroc": met["auroc"],
                             **{k: v for k, v in summ.items() if k not in ("name", "shifts", "decision")}})
                frames[(ood_name, seed)] = tr.frame()
                tr.frame().to_csv(ctx.path("per_batch", f"{ood_name}_seed{seed}.csv"), index=False)
                if seed == SEEDS[0]:
                    plot_run(tr.frame(), f"TANL {PRECISION}: {id_name} + {ood_name} (seed {seed})",
                             ctx.path("plots", f"churn_{ood_name}.png"))
                    plot_dwell(tr.dwell_times(), f"dwell times: {ood_name}", ctx.path("plots", f"dwell_{ood_name}.png"))
                log.info(f"{ood_name} seed {seed}: final survival {summ['final_surv_first']:.3f}, "
                         f"churn(first10) {summ['churn_first10']:.3f}, FPR95 {met['fpr95']:.2f}")
                del s
    churn = pd.DataFrame(rows)
    per_set = churn.groupby("ood").mean(numeric_only=True).drop(columns=["seed"]).reset_index()
    show(per_set, "churn summary (mean over seeds)", cols=["ood", "fpr95", "final_surv_first", "final_surv_init",
         "final_oracle_overlap", "batches_to_90pct_oracle_plateau", "churn_first10", "churn_second_half",
         "unique_labels", "dwell_median", "final_pos_purity", "final_neg_purity"], digits=3)
    '''),
    code('''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for (name, seed), f in frames.items():
        if seed != SEEDS[0]:
            continue
        ax[0].plot(f.batch, f.surv_first, label=name); ax[1].plot(f.batch, f.churn, label=name)
        ax[2].plot(f.batch, f.oracle_overlap, label=name)
    for a, t in zip(ax, ["survival of first-batch set", "churn", "overlap with oracle set"]):
        a.set_title(t); a.set_xlabel("batch"); a.legend(fontsize=7)
    ax[0].axhline(0.5, ls="--", c="gray"); ax[0].axhline(0.1, ls=":", c="gray")
    fig.tight_layout(); fig.savefig(ctx.path("plots", "overview.png"), dpi=110); plt.show()
    '''),
    md('''
    ### 3 · Temporal-shift orders (no reset between OOD sets)
    '''),
    code('''
    shift_rows = []
    if RUN_TEMPORAL and all(n in cache.available() for n in ["imagenet_val_all", *TEMPORAL_ORDERS["I-S-P-T"]]):
        for oname, names in TEMPORAL_ORDERS.items():
            for seed in SEEDS[:1] if ctx.smoke else SEEDS:
                ts = temporal_stream(runner.get("imagenet_val_all"), [runner.get(n) for n in names], seed=seed).to(ctx.device)
                m = make(True); seed_method(m, seed); m.reset()
                orc = {k: oracle_selection(m, ts, k, device=ctx.device) for k in range(len(ts.segments))}
                tr = ChurnTracker(orc, name=f"{oname}/s{seed}")
                res = run_stream(m, ts, ctx.device, hooks=[tr])
                summ = tr.summary()
                for k, seg in enumerate(res.segments):
                    met, win = res.segment_metrics(k), res.window_metrics(k, runner.window)
                    sh = next((x for x in summ.get("shifts", []) if x["segment"] == k), {})
                    shift_rows.append({"order": oname, "seed": seed, "position": k, "ood": seg.name,
                                       "fpr95": met["fpr95"], "post_shift_fpr95": win["fpr95"],
                                       "churn_at_shift": sh.get("churn_at_shift"), "overlap_at_shift": sh.get("overlap_at_shift"),
                                       "batches_to_90pct": sh.get("batches_to_90pct_plateau", summ["batches_to_90pct_oracle_plateau"] if k == 0 else None),
                                       "final_overlap": sh.get("final_overlap", summ["final_oracle_overlap"] if k == 0 else None)})
                if seed == SEEDS[0]:
                    f = tr.frame()
                    f.to_csv(ctx.path("per_batch", f"temporal_{oname}.csv"), index=False)
                    bounds = [int(f[f.segment == k].batch.min()) for k in range(1, len(ts.segments))]
                    plot_run(f, f"temporal {oname} (seed {seed})", ctx.path("plots", f"temporal_{oname}.png"), boundaries=bounds)
                del ts
    shifts = pd.DataFrame(shift_rows)
    if len(shifts):
        show(shifts.groupby(["order", "position", "ood"]).mean(numeric_only=True).drop(columns=["seed"]).reset_index(),
             "post-shift behaviour (mean over seeds)", digits=3)
    '''),
    md('''
    ### 4 · Decision
    '''),
    code('''
    surv = float(churn["final_surv_first"].mean()) if len(churn) else float("nan")
    verdict = decision(surv)
    text = (f"Mean final survival of the first-batch selection over {len(churn)} default streams: **{surv:.1%}** "
            f"(per set: " + ", ".join(f"{r.ood} {r.final_surv_first:.1%}" for r in per_set.itertuples()) + ").\\n\\n"
            f"Rule: > 50 % supports the cold-start framing; < 10 % means a shift-only framing.\\n\\n**Verdict: {verdict}**")
    print(text.replace("**", ""))
    save_tables(ctx, churn_runs=churn, churn_per_set=per_set, churn_shifts=shifts)
    write_report(ctx, "TANL churn diagnostic", [("Decision", text), ("Per OOD set (mean over seeds)", per_set),
                                               ("Temporal shift", shifts if len(shifts) else "not run")])
    finish(ctx, {"mean_final_survival": surv, "verdict": verdict})
    '''),
]


# ============================================================================
# 04 reproduce AdaNeg (+ NegLabel, MCM baselines)
# ============================================================================
NB04 = [
    md('''
    # 04 · Reproduce AdaNeg (NeurIPS 2024), with NegLabel and MCM baselines

    **Acceptance (pre-G1 plan):** Four-OOD FPR95 within **±1** of **18.92** (paper; the repository's own log
    shows 17.95) and OpenOOD near-OOD close to **67.51**.

    AdaNeg runs as in `scripts/ood/adaneg/imagenet.sh` (memory 10 per label, threshold 0.5, gap 0.5,
    λ = 0.1, 5 groups, random permutation, `combine` score). Two precisions:

    * **fp16** — bit-exact emulation of the official GPU run. Its per-sample entropy is NaN in fp16, so
      memory slots are never *replaced*: each label keeps its first 9 confident features (slot 0 holds its
      text feature). This is what produced the published numbers.
    * **fp32** — the entropy rule as described in the paper (replace the most uncertain slot).

    **Settings:** GPU recommended (≈ 30 min). **Inputs:** `oodlab-code` + notebook 01's output
    (+ notebook 02's output, optional, for the final leaderboard).
    '''),
    *setup("04_reproduce_adaneg", need_gpu=False),
    code('''
    from oodlab.methods import AdaNeg, AdaNegConfig, NegLabel, NegLabelConfig, MCM, MCMConfig
    from oodlab.runner import Runner, summarize, compare_with_targets, acceptance_table
    from oodlab.report import find_cache

    BACKBONE = "ViT-B/16"
    SEEDS = [0, 1, 2]
    ADANEG_PRECISIONS = ["fp16", "fp32"] if ctx.device == "cuda" else ["fp32"]
    PROTOCOLS = ["four_ood", "near_50k", "openood_v15"]
    if ctx.smoke:
        SEEDS = [0, 1]

    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    LS = bank.logit_scale
    runner = Runner(cache, device=ctx.device, logger=log)
    runs = []
    '''),
    md('''
    ### 1 · Our AdaNeg / NegLabel == official on real features
    '''),
    code('''
    from oodlab.official.equivalence import compare_adaneg, compare_neglabel
    from oodlab.stream import pair_stream
    eq_sets = [n for n in ["inaturalist", "ninco", "sun", "places", "textures_all", "ssb_hard"] if cache.has(n)]
    assert cache.has("imagenet_val_all") and eq_sets, f"cache incomplete: {cache.available()}"
    ids, ood = cache.load_set("imagenet_val_all"), cache.load_set(eq_sets[0])
    s = pair_stream(ids.subset(range(min(4000, len(ids)))), ood.subset(range(min(2000, len(ood)))), seed=0)
    batches = [s.feats(a, b) for _, a, b, _ in s.batches()]
    eq = {"adaneg fp32": compare_adaneg(bank.id_text.float(), bank.neg_text.float(), [b.float() for b in batches],
                                        AdaNegConfig(logit_scale=LS), device=ctx.device),
          "neglabel fp32": compare_neglabel(bank.id_text.float(), bank.neg_text.float(), [b.float() for b in batches],
                                            NegLabelConfig(logit_scale=LS), device=ctx.device)}
    if ctx.device == "cuda":
        eq["adaneg fp16"] = compare_adaneg(bank.id_text.half(), bank.neg_text.half(), [b.half() for b in batches],
                                           AdaNegConfig(logit_scale=LS, emulate_fp16=True), device=ctx.device)
        eq["neglabel fp16"] = compare_neglabel(bank.id_text.half(), bank.neg_text.half(), [b.half() for b in batches],
                                               NegLabelConfig(logit_scale=LS, emulate_fp16=True), device=ctx.device)
    show(pd.DataFrame(eq).T, f"official vs ours (real features: ImageNet + {eq_sets[0]})", digits=8)
    bad = [k for k, v in eq.items() if v["max_abs_diff"] > (1e-4 if "fp32" in k else 2e-3) or v["pred_agreement"] < 0.999]
    print("EQUIVALENCE OK" if not bad else f"EQUIVALENCE CHECK FAILED for {bad} - investigate before trusting results")
    '''),
    md('''
    ### 2 · AdaNeg
    '''),
    code('''
    for protocol in PROTOCOLS:
        if runner.missing(protocol):
            log.warning(f"skip {protocol}: missing {runner.missing(protocol)}"); continue
        for prec in ADANEG_PRECISIONS:
            m = AdaNeg(bank.id_text, bank.neg_text, AdaNegConfig(logit_scale=LS, emulate_fp16=(prec == "fp16")), device=ctx.device)
            runs.append(runner.evaluate(m, protocol, SEEDS, config=prec))
    '''),
    md('''
    ### 3 · Baselines: NegLabel (static negatives) and MCM (ID labels only)

    MCM twice: as in its paper (softmax of cosine / T, T = 1) and as `scripts/ood/mcm/official.sh` runs it
    (softmax of 100·cosine / τ with τ chosen by OpenOOD's automatic parameter search on the validation
    split: 5,000 ImageNet + 1,763 OpenImage-O images, τ ∈ {0.5, 1, 1.5, 2, 3, 5}, best AUROC).
    '''),
    code('''
    MCM_TAU = None
    if not runner.missing("openood_val"):
        MCM_TAU, aps = runner.tune_on_val(lambda t: MCM(bank.id_text_simple, MCMConfig.openood_vlm(t, LS), device=ctx.device),
                                          [0.5, 1, 1.5, 2.0, 3.0, 5.0])
        show(aps, f"MCM tau search on the OpenOOD validation split -> tau = {MCM_TAU}")
    '''),
    code('''
    for protocol in PROTOCOLS:
        if runner.missing(protocol):
            continue
        nl_prec = "fp16" if ctx.device == "cuda" else "fp32"
        runs.append(runner.evaluate(NegLabel(bank.id_text, bank.neg_text,
                                             NegLabelConfig(logit_scale=LS, emulate_fp16=(nl_prec == "fp16")), device=ctx.device),
                                    protocol, SEEDS, config=f"g100/{nl_prec}"))
        runs.append(runner.evaluate(MCM(bank.id_text_simple, MCMConfig.paper(), device=ctx.device), protocol, SEEDS, config="paper_T1"))
        if MCM_TAU is not None:
            runs.append(runner.evaluate(MCM(bank.id_text_simple, MCMConfig.openood_vlm(MCM_TAU, LS), device=ctx.device),
                                        protocol, SEEDS, config=f"openood_aps_tau{MCM_TAU}"))
    if not runs:
        raise RuntimeError(f"no protocol could run - cached sets: {cache.available()}; finish notebook 01 first")
    df = pd.concat(runs, ignore_index=True)
    summary = summarize(df)
    '''),
    code('''
    for method in ["adaneg", "neglabel", "mcm"]:
        for protocol in PROTOCOLS:
            c = compare_with_targets(summary, method, protocol)
            if len(c):
                show(c, f"{method} vs published: {protocol}")
    acc_table = acceptance_table(summary)
    show(acc_table[acc_table.method == "adaneg"], "pre-G1 acceptance (AdaNeg)")
    '''),
    md('''
    ### 4 · Leaderboard (adds TANL if notebook 02's output is attached)
    '''),
    code('''
    tanl_csv = oodlab.paths.find_file("tanl_summary.csv", [ctx.dirs.results, ctx.dirs.inputs], max_depth=5)
    board = summary.copy()
    if tanl_csv:
        board = pd.concat([board, pd.read_csv(tanl_csv)], ignore_index=True)
        log.info(f"added TANL results from {tanl_csv}")
    board = board[board.key.str.startswith("mean:")]
    pivot = board.pivot_table(index=["method", "config"], columns=["protocol", "key"], values="fpr95").round(2)
    show(pivot.reset_index(), "FPR95 (OOD-positive, OpenOOD convention) - lower is better")
    save_tables(ctx, adaneg_runs=df, adaneg_summary=summary, adaneg_acceptance=acc_table, leaderboard=pivot.reset_index())
    write_report(ctx, "AdaNeg / NegLabel / MCM reproduction",
                 [("Acceptance", acc_table)] +
                 [(f"{m} vs published: {p}", compare_with_targets(summary, m, p)) for m in ["adaneg", "neglabel", "mcm"]
                  for p in PROTOCOLS if len(compare_with_targets(summary, m, p))])
    finish(ctx)
    '''),
]


NOTEBOOKS = {
    "00_setup_check.ipynb": NB00,
    "01_build_feature_cache.ipynb": NB01,
    "02_reproduce_tanl.ipynb": NB02,
    "03_churn_diagnostic.ipynb": NB03,
    "04_reproduce_adaneg.ipynb": NB04,
}


def build(out_dir: str = OUT) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fn, cells in NOTEBOOKS.items():
        nb = new_notebook(cells=cells)
        for i, c in enumerate(nb.cells):                   # stable ids: rebuilding changes nothing
            c["id"] = hashlib.md5(f"{fn}:{i}".encode()).hexdigest()[:8]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        p = os.path.join(out_dir, fn)
        nbformat.write(nb, p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
