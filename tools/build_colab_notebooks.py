"""Generate the Colab notebooks (C0, C1, G1, G1b, G2a, G2b, P1, P2, G3, G3b, P4, D1, G4) from the cell lists below.

    python tools/build_colab_notebooks.py

Edit cells here, not in the .ipynb files, so every notebook keeps the same set-up cell. The notebooks are written
into notebooks/<stage>/; tools/make_colab_bundle.py packs them with the code for Google Drive.
"""
import hashlib
import os
import sys
import textwrap

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from reneg.colab import COLAB_SETUP_CELL  # noqa: E402

NB_DIR = os.path.join(ROOT, "notebooks")
STAGE = {"C0": "1_setup_colab", "C1": "1_setup_colab", "G1": "2_seam1_transport", "P1": "2_seam1_transport",
         "P2": "2_seam1_transport", "G2": "3_exports", "G3": "4_evaluation", "P4": "4_evaluation",
         "D1": "4_evaluation", "G4": "4_evaluation"}


def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


def save(name, cells, gpu=False):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    for i, c in enumerate(nb.cells):                       # stable ids: rebuilding changes nothing
        c["id"] = hashlib.md5(f"{name}:{i}".encode()).hexdigest()[:8]
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        nb.metadata["accelerator"] = "GPU"
    out = os.path.join(NB_DIR, STAGE[name[:2]])
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    nbf.write(nb, path)
    print("wrote", path)


# The last cell of every notebook: Colab uploads files to Drive in the background, and a runtime that is
# disconnected too early loses the files that were still waiting. This cell waits until everything is written.
DRIVE_FLUSH = [
    md("""
    ### Finish saving to Google Drive

    Colab copies files to Drive in the background. This cell waits until every file is written, so run it before
    you disconnect the runtime. (It unmounts Drive: to run more cells afterwards, run the first cell again.)
    """),
    code(r'''
    try:
        from google.colab import drive
        drive.flush_and_unmount()
        print("All files are written to Google Drive. You can disconnect the runtime now.")
    except ImportError:
        print("Not on Colab: nothing to flush.")
    '''),
]


# ============================================================================
# C0 · transfer from Kaggle
# ============================================================================
C0 = [
    md("""
    # C0 · Move the pre-G1 work from Kaggle to Colab

    Run this **once**. It puts everything G1 and G2 need into your Google Drive, under `MyDrive/ReNeg/`:

    | What | From | To |
    |---|---|---|
    | oodlab code | your private Kaggle dataset `oodlab-code` (or `oodlab_code.zip` you upload) | `code/oodlab_code/` |
    | reneg code, notebooks C1 and G1 | `ReNeg_Colab_bundle.zip` you upload | `code/reneg_code/`, `notebooks/` |
    | ViT-B/16 feature cache (12 sets + text bank, about 0.3 GB) | output of Kaggle notebook 01 | `oodlab_working/cache/vit-b-16/` |
    | results of Kaggle notebooks 02–04 | their outputs | `oodlab_working/results/` |
    | Colab copies of Kaggle notebooks 00–04 | the notebooks themselves | `notebooks_colab/` |

    **Before running** (GUIDE_COLAB.md steps 1–3): upload `ReNeg_Colab_bundle.zip` into `MyDrive/ReNeg/uploads/`
    and add your Kaggle key to Colab Secrets. Without a Kaggle key the notebook uses whatever zips you put in
    `uploads/` instead (guide route C).

    **Runtime:** CPU is enough (*Runtime → Change runtime type → CPU*). About 5–10 minutes. Run the cells in order
    with *Runtime → Run all*; every cell can be run again safely.
    """),
    code(r'''
    # ── Step 1 · Drive folders and the ReNeg code ──────────────────────────────────────────
    import os, sys, shutil, zipfile
    DRIVE_ROOT = "/content/drive/MyDrive/ReNeg"          # change only if your project folder is elsewhere
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        IN_COLAB = True
    except ImportError:                                    # running outside Colab (tests)
        IN_COLAB = False
        DRIVE_ROOT = os.environ.get("RENEG_DRIVE_ROOT", os.path.abspath("ReNeg"))
    for sub in ("uploads", "code", "notebooks", "transfers"):
        os.makedirs(os.path.join(DRIVE_ROOT, sub), exist_ok=True)

    def find_bundle_root(folder, depth=4):
        """Folder that contains reneg_code/reneg/__init__.py (searched a few levels deep)."""
        for dp, dn, fn in os.walk(folder):
            if dp[len(folder):].count(os.sep) > depth:
                dn[:] = []
                continue
            if os.path.isfile(os.path.join(dp, "reneg_code", "reneg", "__init__.py")):
                return dp
        return None

    uploads = os.path.join(DRIVE_ROOT, "uploads")
    bundle_root = None
    for f in sorted(os.listdir(uploads), reverse=True):     # newest bundle name last-sorted wins
        p = os.path.join(uploads, f)
        if f.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                if any(n.endswith("reneg_code/reneg/__init__.py") for n in z.namelist()):
                    dest = os.path.join(DRIVE_ROOT, "transfers", "bundle")
                    shutil.rmtree(dest, ignore_errors=True)
                    z.extractall(dest)
                    bundle_root = find_bundle_root(dest)
                    print("unpacked", f)
                    break
        elif os.path.isdir(p) and find_bundle_root(p):
            bundle_root = find_bundle_root(p)
            break
    RENEG_CODE = os.path.join(DRIVE_ROOT, "code", "reneg_code")
    if bundle_root:
        shutil.rmtree(RENEG_CODE, ignore_errors=True)
        shutil.copytree(os.path.join(bundle_root, "reneg_code"), RENEG_CODE)
        for f in os.listdir(os.path.join(bundle_root, "notebooks")):
            if f.endswith(".ipynb") and not f.startswith("C0"):
                shutil.copy2(os.path.join(bundle_root, "notebooks", f), os.path.join(DRIVE_ROOT, "notebooks", f))
        if os.path.isfile(os.path.join(bundle_root, "GUIDE_COLAB.md")):
            shutil.copy2(os.path.join(bundle_root, "GUIDE_COLAB.md"), os.path.join(DRIVE_ROOT, "GUIDE_COLAB.md"))
    if not os.path.isfile(os.path.join(RENEG_CODE, "reneg", "__init__.py")):
        raise FileNotFoundError("ReNeg_Colab_bundle.zip not found in MyDrive/ReNeg/uploads/ (GUIDE_COLAB.md step 2)")
    if RENEG_CODE not in sys.path:
        sys.path.insert(0, RENEG_CODE)
    import importlib
    import reneg
    from reneg import colab as rc
    reneg, rc = importlib.reload(reneg), importlib.reload(rc)   # pick up a newer bundle without restarting
    L = rc.layout(DRIVE_ROOT)
    print(f"reneg {reneg.__version__} installed in {RENEG_CODE}; notebooks copied to {L['notebooks']}")
    '''),
    md("""
    ## Step 2 · Kaggle access

    Reads `KAGGLE_USERNAME` and `KAGGLE_KEY` from Colab Secrets (the key icon on the left; switch on *Notebook
    access*). If they are missing, the notebook keeps going with the zips in `uploads/`.
    """),
    code(r'''
    KAGGLE_USER = rc.setup_kaggle_credentials()
    USE_KAGGLE_API = bool(KAGGLE_USER)
    if USE_KAGGLE_API:
        rc.ensure_kaggle_cli()
    print("Kaggle API:", f"on (user {KAGGLE_USER})" if USE_KAGGLE_API else "off -> using the files in MyDrive/ReNeg/uploads")
    '''),
    md("""
    ## Step 3 · Your Kaggle names

    A notebook's slug is the last part of its URL: `kaggle.com/code/<user>/<slug>`. The list printed below shows
    notebooks on your account. Each of the five is marked **found** (matched in the list) or **guessed** (not in
    the list, so the slug is made the way Kaggle makes it: `Notebook_three` → `notebook-three`). A wrong guess
    shows up as a FAILED download in step 5; then copy the real slug from the notebook's URL into `NOTEBOOKS`
    and run from this cell again.
    """),
    code(r'''
    CODE_DATASET = "oodlab-code"                # your private Kaggle dataset holding oodlab_code.zip
    NOTEBOOKS = {                               # your Kaggle notebooks (title or slug; case, _ and - don't matter)
        "00": "notebook0",
        "01": "notebook1",                      # builds the feature cache: the one that matters most
        "02": "notebook2",
        "03": "Notebook_three",
        "04": "Notebook_four",
    }
    EXTRA_DATASETS = []                         # guide route B, e.g. ["your-user/reneg-01-output"]
    if USE_KAGGLE_API:
        rows = rc.list_my_kernels()
        print(f"{len(rows)} notebooks listed on your Kaggle account:")
        for r in rows[:40]:
            print("   ", r.get("ref"), "|", r.get("title"), "|", r.get("lastRunTime"))
        print()
        NOTEBOOKS = rc.resolve_or_guess(NOTEBOOKS, rows)      # real (or guessed) slugs from here on
    '''),
    md("""
    ## Step 4 · oodlab code
    """),
    code(r'''
    kaggle_dir = os.path.join(L["transfers"], "kaggle")
    if USE_KAGGLE_API:
        rc.download_dataset(f"{KAGGLE_USER}/{CODE_DATASET}", os.path.join(kaggle_dir, CODE_DATASET))
    rc.unpack_uploads(L)                        # zips you uploaded by hand (oodlab_code.zip, notebook outputs)
    rc.expand_nested_zips(L["transfers"])       # a Kaggle dataset may contain oodlab_code.zip itself
    got = rc.install_code([os.path.join(kaggle_dir, CODE_DATASET), os.path.join(L["transfers"], "uploads")],
                          "oodlab", L["oodlab_code"])
    print("oodlab code:", got or "NOT FOUND -> upload oodlab_code.zip to MyDrive/ReNeg/uploads and run this cell again")
    '''),
    md("""
    ## Step 5 · Feature cache and results

    Downloads the outputs of Kaggle notebooks 01–04 (their `/kaggle/working` folders) and installs the cache and
    the results. Notebook 01's output is the big one (about 0.3 GB).
    """),
    code(r'''
    if USE_KAGGLE_API:
        for k, slug in NOTEBOOKS.items():
            if k == "00":
                continue
            ok = rc.download_kernel_output(f"{KAGGLE_USER}/{slug}", os.path.join(kaggle_dir, slug))
            print(f"{slug}: {'downloaded' if ok else 'FAILED -> GUIDE_COLAB.md route B or C'}")
    for slug in EXTRA_DATASETS:                 # route B: a Kaggle dataset made from a notebook's output
        ok = rc.download_dataset(slug, os.path.join(kaggle_dir, "datasets", slug.split("/")[-1]))
        print(f"{slug}: {'downloaded' if ok else 'FAILED'}")
    rc.unpack_uploads(L)
    search = [kaggle_dir, os.path.join(L["transfers"], "uploads")]
    tags = rc.install_cache(search, L["cache"])
    names = rc.install_results(search, L["results"])
    print("caches installed:", tags or "nothing new")
    print("results installed:", names or "nothing new")
    '''),
    md("""
    ## Step 6 · Colab copies of your Kaggle notebooks (optional)

    Each copy gets the Colab set-up cell in place of the Kaggle one, so it runs from Drive. Notebooks you upload
    by hand (`*.ipynb` in `uploads/`) are converted too.
    """),
    code(r'''
    converted = []
    if USE_KAGGLE_API:
        for k, slug in NOTEBOOKS.items():
            src = rc.pull_kernel(f"{KAGGLE_USER}/{slug}", os.path.join(L["transfers"], "kaggle_notebooks", slug))
            if src:
                dst = os.path.join(L["notebooks_colab"], f"{slug}_colab.ipynb")
                converted.append((dst, rc.convert_notebook(src, dst)))
    for f in sorted(os.listdir(L["uploads"])):
        if f.endswith(".ipynb"):
            dst = os.path.join(L["notebooks_colab"], f.replace(".ipynb", "_colab.ipynb"))
            converted.append((dst, rc.convert_notebook(os.path.join(L["uploads"], f), dst)))
    for dst, swapped in converted:
        print(("converted  " if swapped else "copied (no Kaggle set-up cell found)  ") + dst)
    print(f"{len(converted)} notebooks in {L['notebooks_colab']}")
    '''),
    md("""
    ## Step 7 · Check everything
    """),
    code(r'''
    import json
    import pandas as pd
    if L["oodlab_code"] not in sys.path:
        sys.path.insert(0, L["oodlab_code"])
    import oodlab
    rows, bank, cache_dir = rc.verify_cache(L["cache"], "ViT-B/16")
    print(pd.DataFrame(rows).to_string(index=False))
    checks = {
        "oodlab code": os.path.isfile(os.path.join(L["oodlab_code"], "oodlab", "__init__.py")),
        "reneg code": os.path.isfile(os.path.join(L["reneg_code"], "reneg", "__init__.py")),
        "all 12 feature sets": all(r["status"] == "OK" for r in rows),
        "text bank: 69,554 words, 10,000 negatives": bool(bank) and bank.get("n_corpus") == 69554 and bank.get("n_selected") == 10000,
        "TANL results from notebook 02": os.path.isfile(os.path.join(L["results"], "02_reproduce_tanl", "tanl_runs.csv")),
        "C1 and G1 notebooks on Drive": all(os.path.isfile(os.path.join(L["notebooks"], n))
                                             for n in ("C1_colab_setup_check.ipynb", "G1_transport_gate.ipynb")),
    }
    print()
    for k, v in checks.items():
        print(("PASS  " if v else "FAIL  ") + k)
    print("\ntext bank:", json.dumps(bank) if bank else "missing")
    with open(os.path.join(L["root"], "transfer_report.json"), "w") as f:
        json.dump({"oodlab": oodlab.__version__, "cache_dir": cache_dir, "sets": rows, "bank": bank, "checks": checks}, f, indent=1)
    print("\nALL DONE -> open MyDrive/ReNeg/notebooks/C1_colab_setup_check.ipynb" if all(checks.values())
          else "\nSomething is missing -> see the FAIL lines and GUIDE_COLAB.md, troubleshooting")
    '''),
    md("""
    **After C1 passes** you can delete `MyDrive/ReNeg/transfers/` to free Drive space; everything needed was copied
    out of it. Run this notebook again later to bring over the ViT-L/14 cache (checklist step 3.1).
    """),
]

# ============================================================================
# C1 · setup check
# ============================================================================
C1 = [
    md("""
    # C1 · Colab set-up check

    Run this at the start of the project on Colab, and again whenever something looks wrong. Every check prints
    PASS, WARN or FAIL, and the notebook keeps going so you see all problems in one run.

    1. GPU (a warning only: G1 also runs on a CPU, just slower)
    2. oodlab's and reneg's unit tests
    3. the feature cache and the text bank
    4. CLIP zero-shot accuracy from the cache (catches label or cache mix-ups)
    5. WordNet (for the subtree groups and SSB-hard class names)
    6. the CLIP text encoder reproduces the text bank
    7. TANL re-run on Four-OOD matches the Kaggle run from notebook 02

    **Runtime:** GPU (*Runtime → Change runtime type → T4 or L4*). About 10–20 minutes, mostly check 7.
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("C1_colab_setup_check", need_gpu=False)
    log = ctx.log
    import subprocess, time, json
    import numpy as np, pandas as pd, torch
    from oodlab.report import find_cache
    BACKBONE = "ViT-B/16"
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    RUN_TANL_CHECK = True            # False skips check 7 (the slow one)
    checks = []

    def check(name, fn, warn_only=False):
        t0 = time.time()
        try:
            detail, status = fn(), "PASS"
        except Exception as e:
            detail, status = f"{type(e).__name__}: {e}", ("WARN" if warn_only else "FAIL")
        checks.append({"check": name, "status": status, "detail": str(detail)[:300], "seconds": round(time.time() - t0, 1)})
        print(f"[{status}] {name}: {detail}")
    '''),
    code(r'''
    def gpu():
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU: Runtime -> Change runtime type -> T4 or L4 (G1 still runs on CPU)")
        return torch.cuda.get_device_name(0)
    check("1 GPU", gpu, warn_only=True)

    def run_tests(folder):
        def fn():
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", "tests"],
                               cwd=folder, capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
            if r.returncode != 0:
                raise RuntimeError(last + "\n" + r.stdout[-1500:])
            return last
        return fn
    try:
        import pytest  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"])
    check("2a oodlab tests", run_tests(CODE_ROOT))
    check("2b reneg tests", run_tests(f"{DRIVE_ROOT}/code/reneg_code"))

    state = {}
    def cache_ok():
        c = find_cache(ctx, BACKBONE)
        b = c.load_textbank()
        s = b.summary()
        assert s["n_corpus"] == 69554 and s["n_selected"] == 10000, s
        missing = [n for n in ("imagenet_val_all", "imagenet_test", "imagenet_val", "ninco", "ssb_hard", "textures_all",
                               "textures", "inaturalist", "sun", "places", "openimage_o", "openimage_o_val") if not c.has(n)]
        assert not missing, f"missing sets: {missing}"
        state["cache"], state["bank"] = c, b
        return f"{c.dir}: {len(c.available())} sets, {s['n_corpus']} corpus words, dim {s['dim']}"
    check("3 feature cache and text bank", cache_ok)

    def zero_shot_acc():
        from reneg.transport import zero_shot
        fs, b = state["cache"].load_set("imagenet_val"), state["bank"]
        pred, _ = zero_shot(fs.feats, b.id_text.float(), b.logit_scale, device=ctx.device)
        acc = 100 * float((pred == fs.labels).mean())
        assert 60 <= acc <= 75, f"{acc:.1f}% is outside 60-75%"
        return f"{acc:.1f}% top-1 on the 5,000 validation images"
    check("4 zero-shot accuracy", zero_shot_acc)

    def wordnet():
        from reneg.wordnet import get_wordnet, wnid_name
        wn = get_wordnet()
        assert wn is not None, "nltk could not download WordNet"
        return f"n02099601 -> {wnid_name(wn, 'n02099601')}"
    check("5 WordNet", wordnet, warn_only=True)

    def text_encoder():
        from oodlab.clipwrap import load_clip
        from reneg.g1 import clip_text_encoder
        model = load_clip(BACKBONE, device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS)
        enc = clip_text_encoder(model)
        b = state["bank"]
        cos = float((enc(list(b.id_names[:50])) * torch.nn.functional.normalize(b.id_text[:50].float(), dim=1)).sum(1).min())
        assert cos > 0.99, f"min cosine {cos:.4f}"
        return f"min cosine to the bank's ID embeddings {cos:.5f}"
    check("6 CLIP text encoder", text_encoder)
    '''),
    md("""
    ### 7 · TANL on Colab matches TANL on Kaggle

    Same code, same cache, same seed (0), paper settings, fp32. Small differences (≤ 0.2 points) can come from
    a different GPU; larger ones mean the cache or the code differ from the Kaggle run.
    """),
    code(r'''
    def tanl_matches():
        from oodlab.methods import TANL, TANLConfig
        from oodlab.runner import Runner
        c, b = state["cache"], state["bank"]
        runner = Runner(c, device=ctx.device, logger=log)
        tanl = TANL(b.id_text, b.corpus_text, b.noise_feats, TANLConfig.paper().with_(logit_scale=b.logit_scale), device=ctx.device)
        new = runner.evaluate(tanl, "four_ood", seeds=[0], config="paper")
        new.to_csv(ctx.path("tanl_four_ood_seed0.csv"), index=False)
        old_path = os.path.join(ctx.dirs.results, "02_reproduce_tanl", "tanl_runs.csv")
        if not os.path.isfile(old_path):
            raise FileNotFoundError(f"{old_path} (run C0 step 5)")
        old = pd.read_csv(old_path)
        m = old[old["protocol"] == "four_ood"] if "protocol" in old else old
        if "method" in m:
            m = m[m["method"].astype(str).str.lower().str.startswith("tanl")]
        if "seed" in m:
            m = m[m["seed"] == 0]
        if "precision" in m:
            m = m[m["precision"].astype(str) == "fp32"]
        if "config" in m:
            cfgs = [x for x in m["config"].astype(str).unique() if x.startswith("paper") and "fp16" not in x]
            m = m[m["config"].astype(str).isin(["paper"] if "paper" in cfgs else cfgs[:1])]
        both = new[["dataset", "fpr95", "auroc"]].merge(m[["dataset", "fpr95", "auroc"]], on="dataset", suffixes=("_colab", "_kaggle"))
        print(both.round(3).to_string(index=False))
        if not len(both):
            raise RuntimeError("no matching Kaggle rows (protocol four_ood, seed 0, paper, fp32): compare by eye")
        gap = float((both.fpr95_colab - both.fpr95_kaggle).abs().max())
        assert gap <= 0.2, f"largest FPR95 difference {gap:.3f} points"
        return f"largest FPR95 difference {gap:.3f} points over {len(both)} datasets"
    if RUN_TANL_CHECK:
        check("7 TANL Colab == Kaggle", tanl_matches)
    '''),
    code(r'''
    table = pd.DataFrame(checks)
    print(table.to_string(index=False))
    table.to_csv(ctx.path("checks.csv"), index=False)
    n_fail = int((table.status == "FAIL").sum())
    finish(ctx, {"failed_checks": n_fail})
    print("\nALL CHECKS PASSED -> checklist step 0.3 is Done; open G1_transport_gate.ipynb" if n_fail == 0
          else f"\n{n_fail} check(s) FAILED -> GUIDE_COLAB.md, troubleshooting")
    '''),
]

# ============================================================================
# G1 · transport gate
# ============================================================================
G1 = [
    md("""
    # G1 · Does the text-to-image transport work? (checklist steps 1.1–1.7)

    Seam 1 moves every negative label from CLIP's text space into its image space with a map fitted on confident
    ID images. G1 decides which map ReNeg uses — or whether we need Plan B — **before** Seam 2 is built on it.

    | Part | Question | Uses labels? |
    |---|---|---|
    | A | Which test images are confident ID, and which ID classes have enough of them? | no (labels only reported) |
    | B | Which map and λ predict *held-out* ID classes best? (leave-subtree-out) | no |
    | C | Do mapped OOD class names land next to their real images? (NINCO, Textures, SSB-hard) | evaluation only |
    | D | Does an image-space score beat the text-space score? | evaluation only |
    | E | How many confident ID images does the map need? (cold start) | evaluation only |

    **Runtime:** GPU recommended (*T4 or L4*; about 5–10 minutes). On a CPU it takes about 15–30 minutes.
    **Output:** `MyDrive/ReNeg/oodlab_working/results/G1_transport_gate/` — `g1_report.md`, `g1_results.json`,
    and one CSV per table.
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("G1_transport_gate", need_gpu=False)
    log = ctx.log
    import json, time
    import numpy as np, pandas as pd, torch
    from oodlab.report import find_cache
    from reneg.g1 import G1, G1Config, clip_text_encoder
    from reneg.wordnet import get_wordnet
    pd.set_option("display.width", 200)

    BACKBONE = "ViT-B/16"
    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    wn = get_wordnet()
    print("WordNet:", "ready" if wn is not None else "unavailable -> text-cluster groups; SSB-hard names skipped")
    '''),
    md("""
    ### Settings

    The defaults are the spec's starting values. Change them only for a reason you would write in the paper;
    every value is saved with the results.
    """),
    code(r'''
    cfg = G1Config(
        backbone=BACKBONE,
        device=ctx.device,
        id_stream_set="imagenet_test",   # confident ID images for fitting (45,000 test images, no labels used)
        p_min=0.5,                       # checklist 1.1 fallback: 0.3
        n_min=3, n_cap=20,
        eps=0.01, eps_align=0.01,        # a richer map must win by 1 point of top-1, or tie and gain 0.01 cosine
    )
    g = G1(cache, bank, cfg, logger=log)
    print(json.dumps({k: v for k, v in cfg.__dict__.items() if k not in ("protocols",)}, indent=1, default=str))
    '''),
    md("""
    ## Part A · Confident ID images (checklist 1.1)

    **Pass:** at least 900 of the 1,000 classes have 3 or more confident images. **If not:** set `p_min=0.3` above.
    """),
    code(r'''
    a = g.part_a()
    print(json.dumps(a, indent=1))
    print("PASS" if a["pass"] else "FAIL -> lower p_min to 0.3 in the settings cell and re-run from there")
    '''),
    md("""
    ## Part B · Self-check on ID classes (checklist 1.2)

    Whole WordNet subtrees (dogs, birds, vehicles, …) are held out in turn. Each candidate map is fitted on the
    rest, exactly as the method would fit it (pseudo-labelled stream images, no labels), and must predict the
    held-out classes' image centroids from their names. The held-out centroids come from OpenOOD's **labelled
    validation split** (5,000 ImageNet images, the split meant for tuning): pseudo-labelled centroids would favour
    raw text, because CLIP picked those images by their similarity to the class name. No OOD data is used.

    **Pass:** some map is clearly better than raw text (R0). The line `online_choice` shows what the method would
    pick on its own at test time (pseudo-labels, alignment first); ideally it is the same kind of map.
    """),
    code(r'''
    b = g.part_b(wn)
    print(json.dumps({k: v for k, v in b.items() if k != "best_per_kind"}, indent=1, default=str))
    print("best of each kind:", b["best_per_kind"])
    g.tables["selfcheck"].head(15).round(4)
    '''),
    md("""
    ## Part C · OOD concepts (checklist 1.3)

    The names of NINCO, Textures and SSB-hard classes are mapped and compared with those classes' real image
    centroids. These labels only evaluate; nothing is tuned on them. **Pass:** the chosen map beats R0 on 2 of 3
    sets and its alignment never drops.
    """),
    code(r'''
    from oodlab.clipwrap import load_clip
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    model = load_clip(BACKBONE, device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS)
    enc = clip_text_encoder(model, prompt=cfg.prompt)
    c = g.part_c(enc, wn)
    print(json.dumps({k: v for k, v in c.items() if k != "previews"}, indent=1, default=str))
    for p in c.get("previews", {}).values():
        print(p)
    ct = g.tables["concepts"]
    ct[ct.candidate.isin(["R0", "R1", g.chosen_name])].round(4)
    '''),
    md("""
    ## Part D · Image-space score vs text-space score (checklist 1.4)

    NegLabel's score with the 10,000 negatives in text space, against the same score with transported
    prototypes (`T`) or ID centroids plus transported negatives (`C`), alone (ω = 1) or fused. **Pass:** near-OOD
    FPR95 drops by at least 1 point and far-OOD rises by at most 1 point.
    """),
    code(r'''
    d = g.part_d()
    print(json.dumps(d, indent=1, default=str))
    s = g.tables["scores_summary"]
    s[s.protocol == "openood_v15"].pivot_table(index=["candidate", "protos", "omega"], columns="group", values="fpr95").round(2).sort_values("near").head(15)
    '''),
    md("""
    ## Part E · Cold start (checklist 1.5)

    The map is refitted from the first *n* confident images of a shuffled ID stream and scored on the tuning split
    (ImageNet validation vs OpenImage-O validation). **Pass:** the fused score beats text before 2,048 images.
    """),
    code(r'''
    e = g.part_e()
    print(json.dumps(e, indent=1, default=str))
    g.tables["coldstart"].round(3)
    '''),
    md("""
    ## Decision (checklist 1.7)
    """),
    code(r'''
    dec = g.decide()
    path = g.save(ctx.out)
    print(json.dumps({k: v for k, v in dec.items() if k != "config"}, indent=1))
    print("\nReport:", path)
    finish(ctx, {"verdict": dec["verdict"], "chosen": dec["chosen"]})
    '''),
    code(r'''
    import matplotlib.pyplot as plt
    sc = g.tables["selfcheck"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for kind, sub in sc[sc.kind.isin(["R2", "B1", "B2"])].groupby("kind"):
        best = sub.sort_values("top1").groupby("lam").tail(1).sort_values("lam")
        ax[0].plot(best.lam, 100 * best.top1, marker="o", label=kind)
        ax[1].plot(best.lam, best["align"], marker="o", label=kind)
    for k in ("R0", "R1"):
        r = sc[sc.candidate == k].iloc[0]
        ax[0].axhline(100 * r["top1"], ls="--", c="grey" if k == "R0" else "k", label=k)
        ax[1].axhline(r["align"], ls="--", c="grey" if k == "R0" else "k", label=k)
    ax[0].set(xscale="log", xlabel="λ (relative)", ylabel="held-out top-1 (%)", title="Self-check: discrimination")
    ax[1].set(xscale="log", xlabel="λ (relative)", ylabel="cosine to real centroid", title="Self-check: alignment")
    cs = g.tables.get("coldstart")
    if cs is not None and len(cs):
        cs = cs[cs.n_seen > 0]
        ax[2].plot(cs.n_seen, cs.fpr95_text, "--", c="grey", label="text")
        ax[2].plot(cs.n_seen, cs.fpr95_fused, marker="o", label="fused")
        ax[2].set(xscale="log", xlabel="confident ID images seen", ylabel="FPR95 (%)", title="Cold start")
    for a_ in ax:
        a_.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ctx.path("g1_plots.png"), dpi=120)
    plt.show()
    '''),
    md("""
    ### What to do with the verdict

    * **PASS** — mark checklist steps 1.1–1.7 Done, write the chosen map, λ and κ* into step 1.7, and start Seam 2.
    * **PARTIAL** — the map generalises but the static score gain is small: keep it for screening near negatives
      (Seam 2) and re-test the score inside TANL at G2.
    * **R1-ONLY** — use the gap shift, and try Plan B for a richer map (checklist 1.6).
    * **FAIL** — run `G1b_transport_diagnostics` first (checklist 1.5b). It re-tests the maps in the direction the
      score uses, with tuned temperatures, and tells whether Plan B (a learned prior on ViT-L/14) could pay at all.

    `g1_report.md` summarises the gate's decision.
    """),
]

# ============================================================================
# G1b · corrected transport diagnostics
# ============================================================================
G1B = [
    md("""
    # G1b · Seam 1 diagnostics, corrected and pre-registered (checklist 1.5b)

    G1 (v1) said FAIL, but three of its tests could not answer its question:

    1. **Part B looked in the wrong direction.** It ranked image centroids for each mapped *name*. The OOD score does
       the opposite: for each *image* it compares all prototypes. In the name → image direction, a shift shared by
       every name (the gap shift, and much of what the maps learn) favours "generic" centroids for every query.
       The score never sees that shift: for one image it adds the same amount to every prototype and cancels.
    2. **Part D used the text temperature (100) for image-space similarities**, whose spread is several times larger,
       so the image score saturated at 0 or 1 and the fusion fell back to the text score. It also asked for a
       near-OOD gain, which no map can give while the negatives are NegLabel's far words (that is Seam 2's job).
    3. **Part E only re-scored text**, because Part B had chosen raw text.

    G1b fixes these and adds the question that decides Plan B. Every rule is fixed in the settings cell below
    **before** you run it.

    | Part | Question | Labels / OOD data |
    |---|---|---|
    | B′ | Held-out ImageNet subtrees: do mapped names pick out their own images (image → names), catch them against the ID prototypes, and separate them as pseudo-OOD? Raw and centred geometry. | ImageNet val labels judge; no OOD data |
    | H | Does a map hamper CLIP? ImageNet val top-1 with mapped prototypes. | ImageNet val labels |
    | D′ | OpenOOD v1.5 scores, temperature and fusion weight tuned on OpenOOD's validation split. | evaluation only |
    | O | **Oracle near negatives** (diagnostic only): SSB-hard and NINCO class names added as negatives, as text, as mapped prototypes, and as the classes' real image centroids (a perfect transport). | evaluation only |
    | E′ | Cold start, only if D′ keeps a map. | evaluation only |

    **Runtime:** T4 GPU, about 5–10 minutes. **Output:** `MyDrive/ReNeg/oodlab_working/results/G1b_transport_diagnostics/`
    — `g1b_summary.txt` (every number in plain text), `g1b_results.json` and one CSV per table.
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("G1b_transport_diagnostics", need_gpu=False)
    log = ctx.log
    import json, time
    import numpy as np, pandas as pd, torch
    from oodlab.report import find_cache
    from reneg.g1b import G1b, G1bConfig
    from reneg.g1 import clip_text_encoder
    from reneg.wordnet import get_wordnet
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    BACKBONE = "ViT-B/16"
    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    wn = get_wordnet()
    print("device:", ctx.device, "| WordNet:", "ready" if wn is not None else "unavailable (text-cluster groups; SSB-hard names skipped)")
    '''),
    md("""
    ### Settings (pre-registered: do not change after seeing results)

    The decision rules: a map is kept only if, on the OpenOOD v1.5 test sets, it improves the best text score by
    ≥ 1.0 FPR95 point on near- or far-OOD **and** loses ≤ 0.5 on the other; **and** keeps ImageNet val top-1
    within 1.0 point of raw text; **and** its held-out pseudo-OOD AUROC is within 0.5 of raw text's. Plan B (a
    learned prior) is justified only if the real image centroids of the oracle classes beat their mapped names by
    ≥ 3 near FPR95 points **and** beat the text names by ≥ 3 points.
    """),
    code(r'''
    cfg = G1bConfig(
        backbone=BACKBONE,
        device=ctx.device,
        p_min=0.5, n_min=3, n_cap=20,
        gain=1.0, harm=0.5, hamper=1.0, bprime_margin=0.5, planb_margin=3.0,
    )
    g = G1b(cache, bank, cfg, logger=log)
    print(json.dumps({k: v for k, v in cfg.__dict__.items()}, indent=1, default=str))
    '''),
    md("""
    ## Preparation

    The 45,000 ImageNet test images are split by index parity: half A gives the confident (pseudo-labelled) class
    statistics, the mean image and the maps; half B is the ID side of every test score.
    """),
    code(r'''
    prep = g.prepare()
    print(json.dumps(prep, indent=1))
    '''),
    md("""
    ## B′ · Held-out subtrees, scored image → names

    Columns: `name_top1` = a held-out image's nearest held-out name is its own (%); `catch` = its own name beats every
    ID prototype (%); `steal` = an ID image's nearest held-out name beats its own class (%, lower is better);
    `pood_*` = held-out subtrees as OOD, their names plus 3,000 NegLabel negatives as negatives (best temperature);
    `t2c_top1`, `align` = G1's old direction, for comparison.
    """),
    code(r'''
    b = g.part_b(wn)
    print(g.tables["bprime"].round(2).to_string(index=False))
    '''),
    md("""
    ## H · Does a map hamper CLIP?
    """),
    code(r'''
    h = g.part_h()
    print(g.tables["hamper"].round(2).to_string(index=False))
    '''),
    md("""
    ## D′ · OpenOOD v1.5 static scores (tuned on the validation split)

    `text (Z)` rows: raw text and centred text, each at its tuned temperature; `NegLabel form`: raw text at CLIP's
    temperature (G1's baseline). `map` rows: the three best maps from B′ plus the gap shift, with transported ID
    names (`T`) or ID centroids (`C`), alone or fused (log-odds) with the better text row.
    """),
    code(r'''
    d = g.part_d()
    cols = ["config", "role", "mode", "tau", "omega", "val_fpr95", "near_fpr95", "far_fpr95", "four_fpr95", "near_auroc", "far_auroc"]
    print(g.tables["dprime"][cols].round(2).to_string(index=False))
    '''),
    md("""
    ## O · Oracle near negatives (diagnostic only, never a result)

    The class names of SSB-hard and NINCO are added to the negatives. Half of each class's images builds its real
    centroid; the other half is scored. `catch` = the image's nearest prototype is its own class (%); `id_stolen` =
    an ID image's nearest prototype is one of the added classes (%).
    """),
    code(r'''
    from oodlab.clipwrap import load_clip
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    model = load_clip(BACKBONE, device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS)
    enc = clip_text_encoder(model, prompt=cfg.prompt)
    o = g.part_o(enc, wn)
    print(g.tables["oracle"].round(2).to_string(index=False))
    '''),
    md("""
    ## Decision, cold start, report
    """),
    code(r'''
    dec = g.decide()
    e = g.part_e()
    path = g.save(ctx.out)
    finish(ctx, {"verdict": dec["verdict"]})
    print(json.dumps({k: v for k, v in dec.items()}, indent=1, default=str))
    '''),
    md("""
    ## Summary

    One plain-text block with every number above (also saved as `g1b_summary.txt`).
    """),
    code(r'''
    print(g.summary_text())
    '''),
    md("""
    ## Optional · analysis pack for the local analysis

    Two files of about 20 MB each with the text embeddings, class statistics, the validation split and random
    subsets of the test sets (fp16, like the cache), so ideas can be tested on the real features on a CPU
    (experiments/ in the repository; the Seam 1 analyses need it). They are written to
    `MyDrive/ReNeg/transfers/analysis_pack/`.
    """),
    code(r'''
    EXPORT_PACK = True
    if EXPORT_PACK:
        for p in g.export_pack(f"{DRIVE_ROOT}/transfers/analysis_pack"):
            print(p, f"{os.path.getsize(p) / 1e6:.1f} MB")
    '''),
]

# ============================================================================
# G2a · candidate pool and full streams for Seam 2
# ============================================================================
G2A = [
    md("""
    # G2a · Candidate pool and full test streams for Seam 2 (checklist 2.0)

    G1b showed that a fitted text-to-image map adds nothing, and that the gain lies in near-OOD negative labels
    that are **grounded in the test stream**: each label's prototype becomes the centroid of the test images it
    picks up. To build and tune that (Seam 2) on the real features, the local analysis needs two things this notebook
    exports to `MyDrive/ReNeg/transfers/analysis_pack2/`:

    1. `reneg_pool.npz` — CLIP text embeddings of the 14,526 WordNet near-OOD candidates (siblings and cousins of
       the ImageNet classes; the classes themselves, their ancestors and descendants removed), plus the words
       behind NegLabel's 10,000 negatives.
    2. `reneg_stream_*.npz` — every image of ImageNet test, SSB-hard, NINCO, Textures, iNaturalist and
       OpenImage-O, stored as 8-bit codes (half the size of the cache, accurate to about 0.0003 in cosine).
    3. `baseline_results.zip` — the CSVs from your Kaggle notebooks 02–04 (TANL and the other baselines), so
       Seam 2 is compared against your own runs, plus the oodlab code (ReNeg is built on its TANL).

    **Runtime:** CPU or GPU, about 5 minutes. Then attach every file the last cell lists (each under 25 MB).
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("G2a_pool_and_streams", need_gpu=False)
    log = ctx.log
    import glob, json, zipfile
    from oodlab.report import find_cache
    from oodlab.clipwrap import load_clip
    from reneg.g1 import clip_text_encoder
    from reneg.packs import export_pool, export_streams, load_kg_pool
    BACKBONE = "ViT-B/16"
    OUT = f"{DRIVE_ROOT}/transfers/analysis_pack2"
    os.makedirs(OUT, exist_ok=True)
    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    pool = load_kg_pool()
    print(len(pool["pool"]), "candidate names; e.g.", [r["name"] for r in pool["pool"][:5]])
    '''),
    md("""
    ## 1 · Encode the candidate pool (same prompt as the text bank)
    """),
    code(r'''
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    model = load_clip(BACKBONE, device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS)
    enc = clip_text_encoder(model, prompt="The nice {}.")
    import torch
    check = float((enc(list(bank.id_names[:20])) * torch.nn.functional.normalize(bank.id_text[:20].float(), dim=1)).sum(1).min())
    print(f"encoder reproduces the bank's ID embeddings: min cosine {check:.6f}")
    pool_path = export_pool(bank, enc, OUT, pool=pool, prompt_note="The nice {}.", log=log.info)
    '''),
    md("""
    ## 2 · Export the full test streams (8-bit)
    """),
    code(r'''
    SETS = ["imagenet_test", "ssb_hard", "ninco", "textures", "inaturalist", "openimage_o"]
    stream_paths = export_streams(cache, OUT, SETS, max_mb=24.0, log=log.info)
    '''),
    md("""
    ## 3 · Your baseline results (Kaggle notebooks 02–04)
    """),
    code(r'''
    csvs = sorted(p for p in glob.glob(f"{ctx.dirs.results}/0[2-4]*/**/*.csv", recursive=True))
    zpath = os.path.join(OUT, "baseline_results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in csvs:
            z.write(p, os.path.relpath(p, ctx.dirs.results))
        # the oodlab code too (TANL, AdaNeg, NegLabel): ReNeg is built as an extension of this TANL
        for dp, dn, fn in os.walk(CODE_ROOT):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".git", ".pytest_cache")]
            for f in fn:
                if not f.endswith((".pyc", ".npz", ".pt")):
                    q = os.path.join(dp, f)
                    z.write(q, os.path.join("oodlab_code", os.path.relpath(q, CODE_ROOT)))
    print(f"{len(csvs)} CSV files and the oodlab code -> {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")
    for p in csvs:
        print("   ", os.path.relpath(p, ctx.dirs.results))
    '''),
    md("""
    ## Files written
    """),
    code(r'''
    files = [pool_path] + stream_paths + [zpath]
    total = 0
    for p in files:
        mb = os.path.getsize(p) / 1e6
        total += mb
        print(f"{mb:6.1f} MB  {p}")
    print(f"{len(files)} files, {total:.0f} MB in total, all in {OUT}")
    finish(ctx, {"files": [os.path.basename(p) for p in files]})
    '''),
]

# ============================================================================
# G2b · TANL's corpus and the Four-OOD streams
# ============================================================================
G2B = [
    md("""
    # G2b · TANL's word corpus and the Four-OOD streams (checklist 2.3)

    Seam 2 now runs on the exported streams. To build ReNeg **on top of TANL** (and to compare on TANL's second
    benchmark, Four-OOD) the local analysis needs two more things, exported to `MyDrive/ReNeg/transfers/analysis_pack2/`:

    1. `reneg_textbank_*.npz` — the CLIP text embeddings of TANL's whole 69,554-word corpus (official order, 8-bit),
       with the words, NegLabel's 10,000-word split and the 15 noise-image features TANL starts from.
    2. `reneg_stream4ood_*.npz` — ImageNet val (50,000, the ID set of Four-OOD), SUN, Places and Textures (all), 8-bit.

    **Runtime:** CPU is fine, about 3 minutes. Then attach every file the last cell lists (each under 25 MB).
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish
    ctx = start("G2b_textbank_and_four_ood", need_gpu=False)
    log = ctx.log
    import numpy as np, torch
    from oodlab.report import find_cache
    from reneg.packs import export_streams, export_textbank, load_textbank_export
    BACKBONE = "ViT-B/16"
    OUT = f"{DRIVE_ROOT}/transfers/analysis_pack2"
    os.makedirs(OUT, exist_ok=True)
    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    print(bank.summary())
    '''),
    md("""
    ## 1 · TANL's corpus (8-bit) and a round-trip check
    """),
    code(r'''
    tb_paths = export_textbank(bank, OUT, max_mb=24.0, log=log.info)
    d = load_textbank_export(tb_paths)
    ref = torch.nn.functional.normalize(bank.corpus_text.float(), dim=1)
    rows = torch.randperm(ref.shape[0], generator=torch.Generator().manual_seed(0))[:2000]
    err = float((d["corpus_text"][rows] @ ref[rows].T - ref[rows] @ ref[rows].T).abs().max())
    print(f"{d['corpus_text'].shape[0]} corpus rows; largest cosine error after 8-bit coding: {err:.5f} (fine below 0.005)")
    '''),
    md("""
    ## 2 · Four-OOD streams (8-bit)
    """),
    code(r'''
    SETS = ["imagenet_val_all", "sun", "places", "textures_all"]
    print({s: cache.has(s) for s in SETS})
    four_paths = export_streams(cache, OUT, SETS, max_mb=24.0, prefix="reneg_stream4ood", log=log.info)
    '''),
    md("""
    ## Files written
    """),
    code(r'''
    files = tb_paths + four_paths
    total = 0
    for p in files:
        mb = os.path.getsize(p) / 1e6
        total += mb
        print(f"{mb:6.1f} MB  {p}")
    print(f"{len(files)} files, {total:.0f} MB in total, all in {OUT}")
    finish(ctx, {"files": [os.path.basename(p) for p in files]})
    '''),
]

# ============================================================================
# P1 · Plan B probe: generate-then-encode
# ============================================================================
P1 = [
    md("""
    # P1 · Plan B probe: generated images as prototypes for label names (checklist 1.6)

    No map fitted on ImageNet classes can know what an unseen name looks like, but a perfect map would still help
    Seam 2 (NINCO FPR95 36.6 → 30.3 in the stream simulation). This probe asks whether a text-to-image model can
    supply that knowledge: it draws a few images for each name with **SDXL-Turbo**, encodes them with our CLIP
    ViT-B/16, and saves the embeddings. experiments/08_seam1_generation.py then compares these generated prototypes
    with the real image centroids and with the perfect-map bound.

    Names: all 64 NINCO classes, 300 random SSB-hard classes, and 50 random ImageNet classes (a sanity check:
    CLIP should recognise generated ImageNet images). 4 images per name, 1,656 images in total.

    **Runtime:** GPU. A100 or L4 if Colab offers one (5–15 minutes), otherwise T4 (about 30–40 minutes). The run
    saves as it goes, so a dropped session resumes where it stopped. **Output:**
    `MyDrive/ReNeg/transfers/analysis_pack2/reneg_gen_probe.npz` (about 3 MB), the input of the local evaluation.
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("P1_generation_probe", need_gpu=False)
    log = ctx.log
    import json, subprocess, time
    import numpy as np, torch
    if not torch.cuda.is_available():
        print("No GPU: Runtime -> Change runtime type -> A100, L4 or T4 (generation on a CPU takes hours)")
    from oodlab.report import find_cache
    from oodlab.clipwrap import load_clip
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation on small GPUs
    from reneg.gen import clip_image_encoder, fake_generator, generate_safely, load_pipeline, probe_names, run_probe
    from reneg.transport import l2n
    from reneg.wordnet import get_wordnet
    FAKE = os.environ.get("RENEG_FAKE_GEN") == "1"          # only for offline tests of this notebook
    BACKBONE = "ViT-B/16"
    OUT = f"{DRIVE_ROOT}/transfers/analysis_pack2"
    os.makedirs(OUT, exist_ok=True)
    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    wn = get_wordnet()
    print("device:", ctx.device, "| WordNet:", "ready" if wn is not None else "unavailable (SSB-hard names skipped)")
    '''),
    md("""
    ### Settings

    `MODEL = "stabilityai/sd-turbo"` is about three times faster than SDXL-Turbo, if the T4 is too slow.
    """),
    code(r'''
    MODEL = "stabilityai/sdxl-turbo"
    N_PER = 4                 # images per name
    SSB_MAX = 300             # random SSB-hard classes (all 64 NINCO classes are always included)
    N_IMAGENET = 50           # ImageNet classes for the sanity check
    PROMPT = "a photo of a {}."
    BATCH = 4                 # halves itself automatically if the GPU runs out of memory
    names, tags, folders = probe_names(cache, bank, wn=wn, ssb_max=SSB_MAX, n_imagenet=N_IMAGENET)
    print({t: tags.count(t) for t in dict.fromkeys(tags)}, "names;", N_PER * len(names), "images to draw")
    print("e.g.", [n for n, t in zip(names, tags) if t == "ninco"][:4], [n for n, t in zip(names, tags) if t == "ssb_hard"][:4])
    '''),
    md("""
    ### Load the generator and our CLIP image encoder

    The generator downloads once per session (about 7 GB for SDXL-Turbo, 1–2 minutes). If the download asks for a
    token, add a Hugging Face token as the Colab secret `HF_TOKEN` (key icon, Notebook access on) and run this cell
    again. If an earlier run stopped with `CUDA out of memory`, first use **Runtime → Restart session**.
    """),
    code(r'''
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
    except Exception:
        pass
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    if FAKE:
        gen = fake_generator(64)
        dim = int(bank.id_text.shape[1])
        proj = torch.randn(3, dim, generator=torch.Generator().manual_seed(0))
        enc = lambda ims: l2n(torch.tensor(np.stack([np.asarray(im, np.float32).mean((0, 1)) for im in ims])) @ proj)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.27", "accelerate"], check=False)
        gen = load_pipeline(MODEL, device=ctx.device, log=log.info)
        model = load_clip(BACKBONE, device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS)
        enc = clip_image_encoder(model, device=ctx.device)
    # quick check before the long run: CLIP should recognise generated ImageNet classes
    probe_ids = [i for i, t in enumerate(tags) if t == "imagenet"][:8]
    t0 = time.time()
    ims = generate_safely(gen, [PROMPT.format(names[i]) for i in probe_ids], seed=123, log=print)
    e = enc(ims)
    pred = (e @ l2n(bank.id_text.float()).T).argmax(1).numpy()
    ok = [int(folders[i]) == int(p) for i, p in zip(probe_ids, pred)]
    print(f"{sum(ok)}/{len(ok)} generated ImageNet images recognised as their class; {time.time() - t0:.1f}s for 8 images")
    if not FAKE and sum(ok) < 4:
        print("WARNING: CLIP recognises few generated images; check the model and the image preprocessing before the long run")
    '''),
    md("""
    ### Draw and encode (resumable)
    """),
    code(r'''
    out_path = os.path.join(OUT, "reneg_gen_probe.npz")
    info = run_probe(gen, enc, names, tags, folders, out_path, n_per=N_PER, batch=BATCH, prompt=PROMPT,
                     model_id=MODEL, log=log.info)
    print(json.dumps(info, indent=1))
    '''),
    md("""
    ### Summary
    """),
    code(r'''
    z = np.load(out_path)
    E = torch.tensor(z["emb"]).float()
    tg = np.asarray(z["tags"])[z["name_idx"]]
    im = tg == "imagenet"
    if im.any():
        truth = np.asarray([int(f) for f in np.asarray(z["folders"])[z["name_idx"][im]]])
        acc = 100 * float(((E[torch.as_tensor(im)] @ l2n(bank.id_text.float()).T).argmax(1).numpy() == truth).mean())
        print(f"generated ImageNet images recognised by zero-shot CLIP: {acc:.1f}%")
    for t in ("ninco", "ssb_hard", "imagenet"):
        print(f"{t:9s} {int((tg == t).sum()):5d} images")
    print(f"{out_path}: {os.path.getsize(out_path) / 1e6:.1f} MB (input of experiments/08_seam1_generation.py)")
    finish(ctx, {"images": int(len(tg)), "model": MODEL})
    '''),
]

# ============================================================================
# P2 · Plan B2 probe: a diffusion prior (text embedding -> image embedding), ViT-L/14
# ============================================================================
P2 = [
    md("""
    # P2 · Plan B2 probe: a diffusion prior maps label names into image space (checklist 1.6b)

    A *prior* is the half of a DALL·E-2-style model that turns a CLIP **text** embedding into a CLIP **image**
    embedding, with no picture in between. Kandinsky 2.1's prior predicts OpenAI CLIP **ViT-L/14** image
    embeddings (no public prior exists for ViT-B/16), so this notebook works with the ViT-L/14 cache from your
    Kaggle run `notebook1-vitl14` and exports, to `MyDrive/ReNeg/transfers/analysis_pack3/`:

    1. `reneg_prior_probe.npz`: 4 prior embeddings for each P1 probe name (NINCO, SSB-hard, ImageNet).
    2. `reneg_prior_pool.npz`: 2 prior embeddings for each of the 14,526 knowledge-graph names.
    3. `reneg_l14_text.npz`: ViT-L/14 text embeddings of the ID, probe and pool names (the baseline).
    4. `reneg_l14_centroids.npz`: ViT-L/14 image centroids of every NINCO, SSB-hard and ImageNet class.
    5. `reneg_l14_to_b16_map.npz`: an image-to-image map from ViT-L/14 to ViT-B/16, fitted on ImageNet val
       images encoded by both, and checked on NINCO and SSB-hard images it never saw.

    **Runtime:** GPU (A100 / L4 / T4). About 15 minutes for the probe, plus 20–40 minutes for the pool names
    (resumable). The first run also downloads your Kaggle ViT-L/14 cache (about 0.6 GB).
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("P2_prior_probe", need_gpu=False)
    log = ctx.log
    import json, subprocess, time
    import numpy as np, torch
    from reneg import colab as rc
    if not torch.cuda.is_available():
        print("No GPU: Runtime -> Change runtime type -> A100, L4 or T4 (the prior is slow on a CPU)")
    L = rc.layout(DRIVE_ROOT)
    OUT = f"{DRIVE_ROOT}/transfers/analysis_pack3"
    os.makedirs(OUT, exist_ok=True)
    KAGGLE_SLUG = "notebook1-vitl14"          # your Kaggle copy of notebook1 that built the ViT-L/14 cache
    '''),
    md("""
    ## 1 · The ViT-L/14 cache (downloaded from Kaggle on the first run)
    """),
    code(r'''
    have = os.path.isfile(os.path.join(L["cache"], "vit-l-14", "textbank.pt"))
    if not have:
        user = rc.setup_kaggle_credentials()
        if user:
            rc.ensure_kaggle_cli()
            slug = rc.resolve_or_guess({"01L": KAGGLE_SLUG}, rc.list_my_kernels())["01L"]
            d = os.path.join(L["transfers"], "kaggle", slug)
            ok = rc.download_kernel_output(f"{user}/{slug}", d)
            print(f"{slug}: {'downloaded' if ok else 'FAILED -> download its output zip on Kaggle and put it in uploads/'}")
        rc.unpack_uploads(L)
        tags = rc.install_cache([os.path.join(L["transfers"], "kaggle"), L["uploads"]], L["cache"])
        print("caches installed:", tags or "nothing new")
    from oodlab.report import find_cache
    cache_l = find_cache(ctx, "ViT-L/14")
    cache_b = find_cache(ctx, "ViT-B/16")
    bank_l = cache_l.load_textbank()
    print("ViT-L/14 sets:", cache_l.available())
    '''),
    md("""
    ## 2 · Load the prior and check that it speaks our ViT-L/14
    """),
    code(r'''
    from oodlab.clipwrap import load_clip
    from reneg.g1 import clip_text_encoder
    from reneg.prior import load_kandinsky_prior, image_encoder_agreement, DEFAULT_PRIOR
    FAKE = os.environ.get("RENEG_FAKE_GEN") == "1"          # offline tests of this notebook only
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    model = load_clip("ViT-L/14", device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS,
                      random_init=FAKE)
    enc_text = clip_text_encoder(model, prompt="The nice {}.")
    if FAKE:
        from reneg.prior import fake_prior
        prior = fake_prior(int(bank_l.id_text.shape[1]))
        agree = 1.0
    else:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
            if tok:
                os.environ["HF_TOKEN"] = tok
        except Exception:
            pass
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.27", "transformers", "accelerate"],
                       check=False)
        prior = load_kandinsky_prior(DEFAULT_PRIOR, device=ctx.device, log=log.info)
        agree = image_encoder_agreement(prior.pipe, model, device=ctx.device)
    print(f"prior's image encoder vs our ViT-L/14: min cosine {agree:.4f} (should be above 0.99)")
    '''),
    md("""
    ## 3 · Real ViT-L/14 centroids and the guidance setting (chosen on ImageNet classes only)
    """),
    code(r'''
    from reneg.concepts import folder_of
    from reneg.prior import concept_centroids, run_prior, load_prior_embeddings
    from reneg.transport import l2n
    cents = {}
    for ds in ("ninco", "ssb_hard"):
        fs = cache_l.load_set(ds)
        f = np.asarray([folder_of(k) for k in fs.keys])
        uniq = sorted(set(f))
        idx = {u: i for i, u in enumerate(uniq)}
        g = np.asarray([idx[u] for u in f])
        A, B = concept_centroids(fs.feats, g, len(uniq), split=(np.arange(len(g)) % 2 == 0))
        cents[ds] = (uniq, A, B)
    fs = cache_l.load_set("imagenet_test")
    y = np.asarray(fs.labels).astype(int)
    A, B = concept_centroids(fs.feats, y, int(bank_l.id_text.shape[0]), split=(np.arange(len(y)) % 2 == 0))
    cents["imagenet"] = ([str(i) for i in range(A.shape[0])], A, B)
    np.savez_compressed(os.path.join(OUT, "reneg_l14_centroids.npz"),
                        **{f"{k}__folders": np.asarray(v[0], dtype=str) for k, v in cents.items()},
                        **{f"{k}__A": v[1].half().numpy() for k, v in cents.items()},
                        **{f"{k}__B": v[2].half().numpy() for k, v in cents.items()})
    # probe names = the P1 names (same 414 concepts), so P1 and P2 compare directly
    p1 = f"{DRIVE_ROOT}/transfers/analysis_pack2/reneg_gen_probe.npz"
    if os.path.isfile(p1):
        z = np.load(p1)
        names, tags, folders = list(z["names"]), list(z["tags"]), list(z["folders"])
    else:
        from reneg.gen import probe_names
        from reneg.wordnet import get_wordnet
        names, tags, folders = probe_names(cache_b, cache_b.load_textbank(), wn=get_wordnet())
    print({t: tags.count(t) for t in dict.fromkeys(tags)}, "probe names")
    im = [i for i, t in enumerate(tags) if t == "imagenet"]
    real = cents["imagenet"][2]
    truth = np.asarray([int(folders[i]) for i in im])
    best = None
    for gscale in (1.0, 4.0):
        E = prior([f"a photo of a {names[i]}." for i in im], n_per=4, seed=1, guidance=gscale)
        P = l2n(E.reshape(len(im), 4, -1).mean(1))
        acc = 100 * float(((P @ real.T).argmax(1).numpy() == truth).mean())
        print(f"guidance {gscale}: prior prototype finds its ImageNet class (of 1000) {acc:.1f}% of the time")
        if best is None or acc > best[1]:
            best = (gscale, acc)
    T = l2n(enc_text([names[i] for i in im]).float())
    print(f"text embedding finds its class {100 * float(((T @ real.T).argmax(1).numpy() == truth).mean()):.1f}% of the time")
    GUIDANCE = best[0]
    print("guidance used from here on:", GUIDANCE)
    '''),
    md("""
    ## 4 · Prior embeddings for the probe names and for the knowledge-graph pool (resumable)
    """),
    code(r'''
    p_probe = os.path.join(OUT, "reneg_prior_probe.npz")
    run_prior(prior, names, p_probe, n_per=4, batch=32, guidance=GUIDANCE, log=log.info)
    from reneg.packs import load_kg_pool
    pool_names = [r["name"] for r in load_kg_pool()["pool"]]
    POOL = True                       # set False to skip the 14,526 pool names (20-40 minutes)
    p_pool = os.path.join(OUT, "reneg_prior_pool.npz")
    if POOL:
        run_prior(prior, pool_names, p_pool, n_per=2, batch=64, guidance=GUIDANCE, log=log.info)
    '''),
    md("""
    ## 5 · ViT-L/14 text embeddings (the baseline) and the ViT-L/14 → ViT-B/16 image map
    """),
    code(r'''
    from reneg.packs import q8
    from reneg.prior import align_by_keys, apply_image_map, fit_image_map
    texts = {"id": l2n(bank_l.id_text.float()), "probe": l2n(enc_text(names).float())}
    texts["pool"] = l2n(torch.cat([enc_text(pool_names[s:s + 2000]).float() for s in range(0, len(pool_names), 2000)]))
    arr = {}
    for k, v in texts.items():
        q, s = q8(v)
        arr[f"{k}__q"], arr[f"{k}__s"] = q, s
    np.savez_compressed(os.path.join(OUT, "reneg_l14_text.npz"), probe_names=np.asarray(names, dtype=str), **arr)
    # image-to-image map, fitted on ImageNet val (both backbones encoded the same images)
    a, b = cache_l.load_set("imagenet_val_all"), cache_b.load_set("imagenet_val_all")
    ia, ib = align_by_keys(list(a.keys), list(b.keys))
    perm = np.random.default_rng(0).permutation(len(ia))
    tr, te = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
    XA, XB = a.feats[torch.as_tensor(ia)].float(), b.feats[torch.as_tensor(ib)].float()
    checks = {}
    best_map = None
    for lam in (0.1, 1.0, 10.0):
        W = fit_image_map(XA[torch.as_tensor(tr)], XB[torch.as_tensor(tr)], lam)
        c = float((apply_image_map(W, XA[torch.as_tensor(te)]) * l2n(XB[torch.as_tensor(te)])).sum(1).mean())
        print(f"lambda {lam}: held-out ImageNet images, mean cosine(mapped ViT-L/14, real ViT-B/16) = {c:.4f}")
        if best_map is None or c > best_map[1]:
            best_map = (lam, c, W)
    lam, c_in, W = best_map
    checks["imagenet_heldout"] = c_in
    for ds in ("ninco", "ssb_hard"):
        sa, sb = cache_l.load_set(ds), cache_b.load_set(ds)
        ja, jb = align_by_keys(list(sa.keys), list(sb.keys))
        cc = (apply_image_map(W, sa.feats[torch.as_tensor(ja)].float()) * l2n(sb.feats[torch.as_tensor(jb)].float())).sum(1)
        checks[ds] = float(cc.mean())
        print(f"{ds}: mean cosine(mapped, real ViT-B/16) on {len(ja)} images it never saw = {checks[ds]:.4f}")
    np.savez_compressed(os.path.join(OUT, "reneg_l14_to_b16_map.npz"), W=W.float().numpy(), lam=np.asarray(lam),
                        checks=np.asarray(json.dumps(checks)))
    '''),
    md("""
    ## Files written
    """),
    code(r'''
    files = sorted(os.path.join(OUT, f) for f in os.listdir(OUT) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    for p in files:
        print(f"{os.path.getsize(p) / 1e6:6.1f} MB  {p}")
    finish(ctx, {"files": [os.path.basename(p) for p in files], "guidance": GUIDANCE, "encoder_agreement": agree,
                 "map_checks": checks})
    '''),
]

# ============================================================================
# G3 · Final evaluation through oodlab's Runner on the exact cached features (G3b: the same for ViT-L/14)
# ============================================================================
G3_INTRO = {
    "ViT-B/16": """
    # G3 · Final evaluation: TANL and ReNeg on the exact features (checklist 3.2)

    The numbers so far came from 8-bit copies of your features, replayed outside Colab. This notebook produces the
    paper's table the official way: oodlab's `Runner` replays OpenOOD v1.5 and Four-OOD from your **exact** cache,
    three stream orders (seeds 0, 1, 2), memory reset for every OOD set, for TANL and for ReNeg on top of TANL.

    ReNeg settings (fixed before this run, nothing is tuned here). In all three, the KG names act only through their
    visual prototypes, the gate also asks TANL whether a name's caught images look OOD, and an image enters the ID
    image prototypes only when CLIP (softmax at least 0.5) and TANL (its confident-ID mask) agree that it is ID:
    * **balanced** (proposed main method): the image score is a penalty clipped at zero;
    * **max**: the plain image score (strongest on near-OOD, weaker on scene datasets);
    * **safe**: the penalty only on images that a KG name also catches in text.
    Balanced and max with the **blind** pool (benchmark class names removed) and the **full** pool, safe blind.

    After the main rows, optional few-shot rows give ReNeg 5 labelled ImageNet images per class (TINS uses 16).

    **Runtime:** GPU needed (A100 / L4 / T4). A short check first runs ReNeg once on the GPU and once on the CPU
    (NINCO, one stream order) and keeps the GPU only if both give the same result; it then prints an estimate of the
    main rows' time (roughly 2–4 hours on a T4; the few-shot rows add about half of that). Results are saved after
    every method and protocol, so if the session drops, run all cells again: finished parts are skipped.
    **Output:** `MyDrive/ReNeg/transfers/g3_final_b16/`
    `g3_summary.csv` and `g3_runs.csv` (plus `g3_check.json`), as in results/g3_vitb16/.
    """,
    "ViT-L/14": """
    # G3b · Second backbone: TANL and ReNeg with CLIP ViT-L/14 (optional, after G3)

    The same evaluation as G3 on the ViT-L/14 cache (built on Kaggle, step 11; P2 already copied it to Drive): TANL
    and ReNeg-balanced with the blind and the full pool, three stream orders, both protocols. Nothing is tuned: the
    settings are the ViT-B/16 ones. The knowledge-graph names are encoded here with the ViT-L/14 text encoder
    (same prompt as the text bank). A protocol whose sets are missing from the cache is skipped with a message.

    **Runtime:** GPU needed; a check cell as in G3, then roughly 1–3 hours on a T4. Saved after every method and
    protocol: after a dropped session, run all cells again. **Output:** `MyDrive/ReNeg/transfers/g3_final_l14/`
    `g3_summary.csv`, `g3_runs.csv` and `g3_check.json`, as in results/g3b_vitl14/.
    """,
}

G3_POOL = {
    "ViT-B/16": r'''
    z = np.load(f"{DRIVE_ROOT}/transfers/analysis_pack2/reneg_pool.npz")
    assert [r["name"] for r in kg] == list(z["pool_names"]), "pool file and kg_pool.json disagree"
    pool_text = l2n(torch.tensor(z["pool_text"]).float())
    ''',
    "ViT-L/14": r'''
    pool_file = os.path.join(OUT, "pool_text_vitl14.npz")
    names = [r["name"] for r in kg]
    if os.path.isfile(pool_file) and list(np.load(pool_file)["pool_names"]) == names:
        pool_text = l2n(torch.tensor(np.load(pool_file)["pool_text"]).float())
    else:
        from oodlab.clipwrap import load_clip
        from reneg.g1 import clip_text_encoder
        WEIGHTS = f"{DRIVE_ROOT}/weights"
        model = load_clip(BACKBONE, device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS,
                          random_init=os.environ.get("RENEG_FAKE_GEN") == "1")
        enc = clip_text_encoder(model, prompt="The nice {}.")
        ok = float((enc(list(bank.id_names[:20])).float() * l2n(bank.id_text[:20].float().cpu())).sum(1).min())
        print(f"encoder reproduces the bank's ID embeddings: min cosine {ok:.6f} (should be above 0.999)")
        pool_text = l2n(torch.cat([enc(names[s:s + 2000]).float().cpu() for s in range(0, len(names), 2000)]))
        np.savez_compressed(pool_file, pool_text=pool_text.half().numpy(), pool_names=np.asarray(names))
        del model
    ''',
}


# Few-shot rows of G3 (ViT-B/16 only): 5 labelled ID images per class, OpenOOD's ImageNet ID-val split
FEWSHOT = [
    md("""
    ## Few-shot rows: 5 labelled ImageNet images per class (optional, runs after the main rows)

    TINS and InterNeg use 16 labelled ImageNet training images per class. Here ReNeg gets 5 per class: OpenOOD's
    ImageNet ID-validation split (5,000 images), which OpenOOD v1.5's 45k test split does not contain. They enter
    ReNeg's ID image prototypes before every stream. Four-OOD's 50k ID set does contain them, so Four-OOD is
    re-scored without those 5,000 images (protocol `four_ood_45k`, with TANL on the same images for comparison).
    """),
    code(r'''
    FEW = [("tanl", None, None, ["four_ood_45k"])] + [(c, c, "blind", ["openood_v15", "four_ood_45k"]) for c in ("max", "balanced")]
    if len(RUNS) > 1 and "imagenet_val" in cache.available() and "four_ood" in PROTOCOLS:
        from oodlab.runner import Protocol
        va, v5 = runner.get("imagenet_val_all"), runner.get("imagenet_val")
        A = l2n(va.feats.float().to(ctx.device)); B = l2n(v5.feats.float().to(ctx.device))
        shot = torch.zeros(len(va), dtype=torch.bool)
        for s in range(0, len(va), 4096):
            shot[s:s + 4096] = ((A[s:s + 4096] @ B.T).max(1).values > 0.999).cpu()
        shot = shot.numpy()
        prior = (va.feats[torch.as_tensor(np.nonzero(shot)[0])], va.labels[shot])
        acc = float(((l2n(prior[0].float()) @ l2n(bank.id_text.float()).T).argmax(1).numpy() == prior[1]).mean() * 100)
        print(f"shots found in the 50k set: {int(shot.sum())} (expect about 5,000); zero-shot accuracy on them {acc:.1f}% "
              f"(about 70% means the labels line up)")
        runner.extra_sets["imagenet_val_45k"] = va.subset(np.nonzero(~shot)[0], name="imagenet_val_45k")
        P45 = Protocol("four_ood_45k", "imagenet_val_45k", {"four45": ["inaturalist", "sun", "places", "textures_all"]},
                       "Four-OOD without OpenOOD's 5k ID-val images (they are the few-shot ID images)")
        done = pd.read_csv(runs_path) if os.path.isfile(runs_path) else pd.DataFrame()
        for name, cfg_name, pool, protos in FEW:
            label = name if pool is None else f"reneg_{cfg_name}_{pool}_5shot"
            for protocol in protos:
                if len(done) and ((done["label"] == label) & (done["protocol"] == protocol)).any():
                    print("done already:", label, protocol)
                    continue
                t0 = time.time()
                method = make(name, cfg_name, pool, reneg_device=RENEG_DEVICE, id_prior=None if pool is None else prior)
                df = runner.evaluate(method, P45 if protocol == "four_ood_45k" else protocol, seeds=SEEDS, config=label)
                df["label"] = label
                df["backbone"] = BACKBONE
                done = pd.concat([done, df], ignore_index=True)
                done.to_csv(runs_path, index=False)
                print(f"{label} {protocol}: {(time.time() - t0) / 60:.1f} min")
    else:
        print("few-shot rows skipped (no imagenet_val in the cache, no Four-OOD, or the oodlab test stub)")
    '''),
]


def g3_cells(backbone):
    l14 = backbone == "ViT-L/14"
    out_dir = "g3_final_l14" if l14 else "g3_final_b16"
    runs = ('[("tanl", None, None)] + [("balanced", "balanced", p) for p in ("blind", "full")]' if l14 else
            '[("tanl", None, None)] + [(c, c, p) for c in ("balanced", "max") for p in ("blind", "full")]'
            ' + [("safe", "safe", "blind")]')
    setup = r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("@@NAME@@", need_gpu=False)
    log = ctx.log
    import json, time
    import numpy as np, pandas as pd, torch
    from oodlab.report import find_cache
    from oodlab.runner import Runner
    from oodlab.methods import TANL, TANLConfig
    from reneg.core import ReNegConfig
    from reneg.packs import load_kg_pool
    from reneg.reneg_tanl import ReNegTANL
    from reneg.transport import l2n
    BACKBONE = "@@BACKBONE@@"
    cache = find_cache(ctx, BACKBONE)
    bank = cache.load_textbank()
    OUT = f"{DRIVE_ROOT}/transfers/@@OUT@@"
    os.makedirs(OUT, exist_ok=True)
    SEEDS = [0, 1, 2]
    runner = Runner(cache, device=ctx.device, logger=log)
    PROTOCOLS = []
    for p in ("openood_v15", "four_ood"):
        if runner.missing(p):
            print(f"{p}: skipped, sets missing from the {BACKBONE} cache: {runner.missing(p)}")
        else:
            PROTOCOLS.append(p)
    print(BACKBONE, "| device:", ctx.device, "| sets:", cache.available(), "| protocols:", PROTOCOLS)
    '''.replace("@@NAME@@", "G3b_final_eval_vitl14" if l14 else "G3_final_eval").replace(
        "@@BACKBONE@@", backbone).replace("@@OUT@@", out_dir)
    pool_cell = r'''
    kg = load_kg_pool()["pool"]
    @@POOL@@
    pool_cluster = np.array([r["cluster"] for r in kg])
    blind = ~np.array([bool(r["in_ssb_hard"]) or bool(r["in_ninco"]) for r in kg])
    POOLS = {"full": np.arange(len(kg)), "blind": np.nonzero(blind)[0]}
    BASE = dict(n_min=1, pool_frac=0.0, pool_in_text=False, id_admit="softmax+base")
    CONFIGS = {"balanced": ReNegConfig(**BASE, image_mode="kg_clip", vote=True),
               "max": ReNegConfig(**BASE),
               "safe": ReNegConfig(**BASE, image_mode="kg_clip_caught", vote=True)}
    RUNS = @@RUNS@@
    if not hasattr(TANL, "selected_T"):                       # the offline test stub of oodlab, not the real one
        RUNS = RUNS[:1]
        print("oodlab test stub: only the TANL row runs")
    print(len(kg), "KG names;", int(blind.sum()), "in the blind pool;", tuple(pool_text.shape))

    def make(name, cfg_name, pool, reneg_device=None, id_prior=None):
        if pool is None:
            return TANL(bank.id_text, bank.corpus_text, bank.noise_feats, TANLConfig.paper(),
                        device=ctx.device, n_neglabel=int(bank.n_selected))
        idx = torch.as_tensor(POOLS[pool])
        return ReNegTANL(bank.id_text, bank.corpus_text, bank.noise_feats, pool_text=pool_text[idx],
                         pool_cluster=pool_cluster[POOLS[pool]], cfg=TANLConfig.paper(), rcfg=CONFIGS[cfg_name],
                         device=ctx.device, n_neglabel=int(bank.n_selected), reneg_device=reneg_device,
                         id_prior=id_prior)

    runs_path = os.path.join(OUT, "g3_runs.csv")
    check_path = os.path.join(OUT, "g3_check.json")
    '''
    pool_src = "\n    ".join(textwrap.dedent(G3_POOL[backbone]).strip().splitlines())
    pool_cell = pool_cell.replace("@@POOL@@", pool_src).replace("@@RUNS@@", runs)
    return [
        md(G3_INTRO[backbone]),
        code(COLAB_SETUP_CELL),
        code(setup),
        md("""
        ## The knowledge-graph pool and the ReNeg settings
        """),
        code(pool_cell),
        md("""
        ## Check: ReNeg on the GPU gives the same result as on the CPU (about 5 minutes)

        TANL always runs on the GPU. ReNeg's own part (the knowledge-graph gate and the image prototypes) can run on
        the GPU too, which is much faster. This cell runs ReNeg-balanced on NINCO (one stream order) both ways and
        keeps the GPU only when both give the same FPR95 and AUROC (within float rounding). Otherwise everything still
        runs, with ReNeg's part on the CPU (slower). The decision is saved, so a re-run skips this check.
        """),
        code(r'''
        RENEG_DEVICE = ctx.device
        if os.path.isfile(check_path):
            chk = json.load(open(check_path))
            RENEG_DEVICE = chk["reneg_device"]
            print("check done already:", chk)
        elif ctx.device == "cpu" or len(RUNS) == 1 or "openood_v15" not in PROTOCOLS:
            print("no GPU (or no ReNeg rows): nothing to check")
        else:
            res = {}
            for dev in (ctx.device, "cpu"):
                t0 = time.time()
                try:
                    d = runner.evaluate(make("balanced", "balanced", "full", reneg_device=dev), "openood_v15",
                                        seeds=[0], datasets=["ninco"], config=f"check_{dev}")
                    res[dev] = {"fpr95": round(float(d.fpr95.iloc[0]), 3), "auroc": round(float(d.auroc.iloc[0]), 3),
                                "img_per_s": float(d.img_per_s.iloc[0]), "minutes": round((time.time() - t0) / 60, 1)}
                except Exception as e:                                   # noqa: BLE001
                    res[dev] = {"error": f"{type(e).__name__}: {e}"}
                print(dev, res[dev])
            g, c = res[ctx.device], res["cpu"]
            if "error" in c and "error" in g:
                raise RuntimeError(f"ReNeg fails on both devices: {res} - please report this output")
            same = ("error" not in g and "error" not in c and abs(g["fpr95"] - c["fpr95"]) <= 0.5
                    and abs(g["auroc"] - c["auroc"]) <= 0.2)
            RENEG_DEVICE = ctx.device if same else "cpu"
            chk = {"reneg_device": RENEG_DEVICE, "results": res}
            json.dump(chk, open(check_path, "w"), indent=1)
            print("ReNeg runs on:", RENEG_DEVICE, "(GPU and CPU agree)" if same else "(GPU result differs or failed: CPU)")
        if os.path.isfile(check_path):              # time estimate (upper bound: TANL alone is faster)
            sets = {"openood_v15": (["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"], "imagenet_test"),
                    "four_ood": (["inaturalist", "sun", "places", "textures_all"], "imagenet_val_all")}
            n_img = sum(sum(len(runner.get(n)) for n in sets[p][0]) + len(sets[p][0]) * len(runner.get(sets[p][1]))
                        for p in PROTOCOLS)
            speed = json.load(open(check_path))["results"][RENEG_DEVICE]["img_per_s"]
            print(f"estimate: {len(RUNS)} methods x {len(SEEDS)} orders x {n_img:,} images at about {speed:.0f} img/s "
                  f"= {len(RUNS) * len(SEEDS) * n_img / speed / 3600:.1f} hours at most")
        '''),
        md("""
        ## Run (saved after every method and protocol; re-running skips what is done)
        """),
        code(r'''
        done = pd.read_csv(runs_path) if os.path.isfile(runs_path) else pd.DataFrame()
        for name, cfg_name, pool in RUNS:
            label = name if pool is None else f"reneg_{cfg_name}_{pool}"
            for protocol in PROTOCOLS:
                if len(done) and ((done["label"] == label) & (done["protocol"] == protocol)).any():
                    print("done already:", label, protocol)
                    continue
                t0 = time.time()
                method = make(name, cfg_name, pool, reneg_device=RENEG_DEVICE)
                df = runner.evaluate(method, protocol, seeds=SEEDS, config=label)
                df["label"] = label
                df["backbone"] = BACKBONE
                done = pd.concat([done, df], ignore_index=True)
                done.to_csv(runs_path, index=False)
                print(f"{label} {protocol}: {(time.time() - t0) / 60:.1f} min")
        '''),
    ] + (FEWSHOT if not l14 else []) + [
        md("""
        ## Summary (mean over the three stream orders), then attach the three files
        """),
        code(r'''
        df = pd.read_csv(runs_path)
        per = df.groupby(["label", "protocol", "group", "dataset"])[["fpr95", "auroc"]].mean().reset_index()
        rows = []
        for label, g in per.groupby("label"):
            row = {"method": label}
            for (protocol, group), gg in g.groupby(["protocol", "group"]):
                row[f"{group} FPR95"] = round(gg["fpr95"].mean(), 2)
                row[f"{group} AUROC"] = round(gg["auroc"].mean(), 2)
            for _, r in g.iterrows():                              # iNaturalist is in several protocols: prefix them
                pre = {"openood_v15": "", "four_ood": "4ood_", "four_ood_45k": "4ood45_"}.get(r["protocol"], r["protocol"] + "_")
                row[pre + r["dataset"]] = round(r["fpr95"], 2)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(OUT, "g3_summary.csv"), index=False)
        pd.set_option("display.width", 250)
        print(BACKBONE)
        print(summary.to_string(index=False))
        print("\nresult files:")
        for f in (runs_path, os.path.join(OUT, "g3_summary.csv"), check_path):
            if os.path.isfile(f):
                print("  ", f)
        finish(ctx, {"runs": int(len(df)), "labels": sorted(df["label"].unique().tolist()), "reneg_device": RENEG_DEVICE,
                     "backbone": BACKBONE})
        '''),
    ]


G3 = g3_cells("ViT-B/16")
G3B = g3_cells("ViT-L/14")


# ============================================================================
# P4 · TINS on our features, with per-image traces for ReNeg-on-TINS
# ============================================================================
P4 = [
    md("""
    # P4 · Probe: TINS on our features (does ReNeg on top of TINS beat TINS?)

    TINS (arXiv 2605.10756, May 2026) is the strongest published method on Four-OOD. It uses labelled ImageNet
    images (16 per class; its paper shows no gain beyond 4) and, at test time, turns images that look OOD into text
    embeddings with 30 gradient steps through CLIP's text encoder. `reneg/tins.py` re-implements its official code on
    our feature cache. This notebook runs TINS on OpenOOD v1.5 and Four-OOD (three stream orders) and saves, for every
    image, TINS's score and whether it was inverted. With those files I replay ReNeg on top of TINS locally, so the
    combination can be tested without more GPU runs.

    * ID prototypes: 5 labelled images per class, OpenOOD's ImageNet ID-val split. OpenOOD v1.5's test split does not
      contain them; Four-OOD's 50k ID set does, so the summary also scores Four-OOD without them (`four_ood_45k`).
    * GPU needed (any CUDA GPU; less than 8 GB of memory is handled by inverting fewer images at a time). The cell
      after the set-up times TINS on one stream and prints an estimate for the whole run. Results are saved per
      stream; after a dropped session, run all cells again and finished streams are skipped.

    **Output:** `MyDrive/ReNeg/transfers/p4_tins/`: `tins_runs.csv`, `tins_summary.csv`, `shot_mask.npy` and
    `tins_traces_*.zip` (attach all of them).
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("P4_tins_probe", need_gpu=True)
    log = ctx.log
    import json, time, glob, zipfile
    import numpy as np, pandas as pd, torch
    from oodlab.report import find_cache
    from oodlab.runner import Runner
    from oodlab.metrics import openood_metrics
    from oodlab.clipwrap import load_clip
    from reneg.tins import TINSConfig, TINSFeat, build_init_candidates, build_static_negatives
    from reneg.transport import l2n
    cache = find_cache(ctx, "ViT-B/16")
    bank = cache.load_textbank()
    runner = Runner(cache, device=ctx.device, logger=log)
    OUT = f"{DRIVE_ROOT}/transfers/p4_tins"
    TR = os.path.join(OUT, "tins_traces")
    os.makedirs(TR, exist_ok=True)
    SEEDS = [0, 1, 2]
    PROTOCOLS = [p for p in ("openood_v15", "four_ood") if not runner.missing(p)]
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    model = load_clip("ViT-B/16", device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS,
                      random_init=os.environ.get("RENEG_FAKE_GEN") == "1")
    REAL = hasattr(model, "token_embedding") and "imagenet_val" in cache.available()
    print("device:", ctx.device, "| protocols:", PROTOCOLS, "| real CLIP text encoder:", REAL)
    if REAL and ctx.device == "cpu":
        print("WARNING: no GPU. TINS inverts images with 30 gradient steps each; on a CPU this takes days.")
    '''),
    md("""
    ## ID prototypes (5 per class) and TINS's static negatives (a few minutes, cached)
    """),
    code(r'''
    static_path = os.path.join(OUT, "tins_static.pt")
    shot_path = os.path.join(OUT, "shot_mask.npy")
    if REAL:
        v5 = runner.get("imagenet_val")
        X5, y5 = l2n(v5.feats.float()), torch.as_tensor(np.asarray(v5.labels)).long()
        C = len(bank.id_names)
        cnt = torch.bincount(y5, minlength=C)
        proto = l2n(torch.zeros(C, X5.shape[1]).index_add_(0, y5, X5))
        if (cnt == 0).any():                     # TINS needs one prototype per class (the official code stops here)
            print(f"WARNING: {int((cnt == 0).sum())} classes have no labelled image: their text embedding is used")
            proto[cnt == 0] = l2n(bank.id_text.float())[cnt == 0]
        acc = float(((X5 @ l2n(bank.id_text.float()).T).argmax(1) == y5).float().mean() * 100)
        print(f"{len(X5)} labelled ID images, {int((cnt > 0).sum())} classes (min {int(cnt.min())} per class); "
              f"zero-shot accuracy on them {acc:.1f}% (about 70% means the labels line up)")
        if "imagenet_val_all" in cache.available() and not os.path.isfile(shot_path):
            A = l2n(runner.get("imagenet_val_all").feats.float().to(ctx.device)); B = X5.to(ctx.device)
            shot = torch.zeros(len(A), dtype=torch.bool)
            for s in range(0, len(A), 4096):
                shot[s:s + 4096] = ((A[s:s + 4096] @ B.T).max(1).values > 0.999).cpu()
            np.save(shot_path, shot.numpy())
            del A, B
        if os.path.isfile(shot_path):
            print("labelled images found inside Four-OOD's 50k ID set:", int(np.load(shot_path).sum()), "(expect about 5,000)")
        if os.path.isfile(static_path):
            st = torch.load(static_path, weights_only=False)
        else:
            t0 = time.time()
            neg, words = build_static_negatives(model, bank.id_text.float(), proto, n=2000,
                                                positive_labels=list(bank.id_names), log=print)
            st = {"neg": neg, "words": words, "cands": build_init_candidates(model, words, proto), "proto": proto}
            torch.save(st, static_path)
            print(f"built in {(time.time() - t0) / 60:.1f} min")
        print(len(st["words"]), "static negatives, e.g.", st["words"][:8])
    else:
        print("test stub of oodlab (no CLIP text encoder): the TINS cells are skipped")
    '''),
    md("""
    ## Timing check on one stream (NINCO, stream order 0) and an estimate for the whole run
    """),
    code(r'''
    def make_tins():
        return TINSFeat(model, bank.id_text.float(), st["proto"], st["neg"], st["cands"], TINSConfig())

    class TraceHook:
        """Saves, per image in stream order: TINS's score, its score before the batch's own inversions, whether it
        was a candidate and whether its inverted text entered the bank, plus the stream rows (for alignment)."""
        def __init__(self, path):
            self.path, self.parts = path, []
        def on_reset(self, method, seg):
            self.parts = []
        def on_step(self, method, seg, start, end, out, stream):
            i = out.info
            self.parts.append((start, out.score.float().cpu().numpy(), i["pre"].numpy(), i["cand"].numpy(),
                               i["kept"].numpy()))
        def on_end(self, method, stream):
            n = len(stream)
            arr = {k: np.zeros(n, dt) for k, dt in (("score", np.float32), ("pre", np.float32), ("cand", bool),
                                                    ("kept", bool))}
            for s, sc, pre, cand, kept in self.parts:
                for k, a in (("score", sc), ("pre", pre), ("cand", cand), ("kept", kept)):
                    arr[k][s:s + len(a)] = a
            tmp = self.path + ".tmp.npz"
            np.savez_compressed(tmp, is_ood=(np.asarray(stream.labels) == -1), rows=stream.rows.astype(np.int32),
                                n_inv=method.n_inv, t_inv=method.t_inv, **arr)
            os.replace(tmp, self.path)

    def trace_path(protocol, ds, seed):
        return os.path.join(TR, f"{protocol}__{ds}__s{seed}.npz")

    runs_path = os.path.join(OUT, "tins_runs.csv")

    def run_one(protocol, ds, seed):
        global done
        if os.path.isfile(trace_path(protocol, ds, seed)):
            return None
        t0 = time.time()
        df = runner.evaluate(make_tins(), protocol, seeds=[seed], datasets=[ds], config="tins_5shot",
                             hooks_factory=lambda name, sd: [TraceHook(trace_path(protocol, name, sd))])
        tr = np.load(trace_path(protocol, ds, seed))
        df["label"] = "tins_5shot"
        df["n_inverted"] = int(tr["n_inv"])
        df["inversion_seconds"] = float(tr["t_inv"])
        done = pd.concat([done, df], ignore_index=True)
        done.to_csv(runs_path, index=False)
        r = df.iloc[0]
        print(f"{protocol} {ds} s{seed}: FPR95 {r.fpr95:.2f} (ID-positive {r.fpr95_idpos:.2f}), AUROC {r.auroc:.2f}, "
              f"{r.n_inverted} images inverted, {(time.time() - t0) / 60:.1f} min", flush=True)
        return r

    SETS = {"openood_v15": (["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"], "imagenet_test"),
            "four_ood": (["inaturalist", "sun", "places", "textures_all"], "imagenet_val_all")}
    done = pd.read_csv(runs_path) if os.path.isfile(runs_path) else pd.DataFrame()
    if REAL and "openood_v15" in PROTOCOLS:
        run_one("openood_v15", "ninco", 0)
    sel = done[(done.protocol == "openood_v15") & (done.dataset == "ninco") & (done.seed == 0)] if len(done) else done
    if REAL and len(sel):
        r = sel.iloc[-1]
        tr = np.load(trace_path("openood_v15", "ninco", 0))
        n = len(tr["score"])
        per_img = (r.method_seconds - r.inversion_seconds) / n                 # scoring, per image
        per_inv = r.inversion_seconds / max(1, r.n_inverted)                   # inversion, per candidate image
        f_id = float(tr["cand"][~tr["is_ood"]].mean())                         # ID images that become candidates
        hours = 0.0
        for p in PROTOCOLS:
            n_id = len(runner.get(SETS[p][1]))
            for ds in SETS[p][0]:
                n_ood = len(runner.get(ds))                                    # upper bound: every OOD image inverted
                hours += ((n_ood + n_id) * per_img + (n_ood + f_id * n_id) * per_inv) / 3600
        print(f"NINCO: {per_inv * 1000:.1f} ms per inverted image, {100 * f_id:.1f}% of ID images inverted, "
              f"{100 * tr['cand'][tr['is_ood']].mean():.1f}% of OOD images")
        print(f"estimate for everything: at most {len(SEEDS) * hours:.1f} hours ({hours:.1f} h per stream order)")
    '''),
    md("""
    ## Run TINS on every stream (stream order 0 for all sets first, then 1 and 2; saved per stream)
    """),
    code(r'''
    if REAL:
        for seed in SEEDS:
            for protocol in PROTOCOLS:
                for ds in SETS[protocol][0]:
                    run_one(protocol, ds, seed)
    '''),
    md("""
    ## Summary from the saved traces (both FPR95 conventions), and the files to attach

    `fpr95` is OpenOOD's convention (OOD positive; what TANL reports and our tables use). `fpr95_idpos` is the
    convention of TINS's code (ID positive: OOD images accepted when 95% of ID images are kept). TINS's paper
    (16 labelled images per class, ID positive): Four-OOD 6.72 (iNaturalist 0.21, SUN 3.84, Places 12.73, Textures
    10.09), OpenOOD v1.5 near-OOD 57.88, far-OOD 12.99.
    """),
    code(r'''
    def TRACES():
        return sorted(f for f in glob.glob(os.path.join(TR, "*.npz")) if not f.endswith(".tmp.npz"))

    shot = np.load(shot_path) if os.path.isfile(shot_path) else None
    rows = []
    for f in TRACES():
        protocol, ds, s = os.path.basename(f)[:-4].split("__")
        tr = np.load(f)
        lab = np.where(tr["is_ood"], -1, 0)
        variants = [(protocol, np.ones(len(lab), bool))]
        if protocol == "four_ood" and shot is not None:
            n_ood = len(runner.get(ds))
            idr = tr["rows"].astype(np.int64) - n_ood                          # ID row, or negative for OOD images
            keep = np.ones(len(lab), bool)
            keep[idr >= 0] = ~shot[idr[idr >= 0]]
            variants.append(("four_ood_45k", keep))
        for name, keep in variants:
            m = openood_metrics(tr["score"][keep], lab[keep])
            rows.append({"protocol": name, "dataset": ds, "seed": int(s[1:]), "fpr95": m["fpr95"],
                         "fpr95_idpos": m["fpr95_idpos"], "auroc": m["auroc"], "inverted": int(tr["cand"].sum()),
                         "kept": int(tr["kept"].sum())})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, "tins_summary.csv"), index=False)
        per = df.groupby(["protocol", "dataset"])[["fpr95", "fpr95_idpos", "auroc"]].agg(["mean", "std"]).round(2)
        pd.set_option("display.width", 200)
        print(per.to_string())
        groups = {"near": ["ssb_hard", "ninco"], "far": ["inaturalist", "textures", "openimage_o"]}
        mean = df.groupby(["protocol", "dataset"])[["fpr95", "fpr95_idpos", "auroc"]].mean()
        for p, sets in [("openood_v15", groups["near"]), ("openood_v15", groups["far"]),
                        ("four_ood", SETS["four_ood"][0]), ("four_ood_45k", SETS["four_ood"][0])]:
            try:
                g = mean.loc[[(p, d) for d in sets]].mean()
                print(f"{p} {sets}: FPR95 {g.fpr95:.2f} | ID-positive {g.fpr95_idpos:.2f} | AUROC {g.auroc:.2f}")
            except KeyError:
                pass
    parts, cur, size = [], [], 0
    for f in TRACES():
        if cur and size + os.path.getsize(f) > 22e6:
            parts.append(cur); cur, size = [], 0
        cur.append(f); size += os.path.getsize(f)
    if cur:
        parts.append(cur)
    for old in glob.glob(os.path.join(OUT, "tins_traces_*.zip")):
        os.remove(old)
    for k, fs in enumerate(parts, 1):
        with zipfile.ZipFile(os.path.join(OUT, f"tins_traces_{k}.zip"), "w") as z:
            for f in fs:
                z.write(f, os.path.basename(f))
    print("\nresult files:")
    for f in [runs_path, os.path.join(OUT, "tins_summary.csv"), shot_path] + sorted(glob.glob(os.path.join(OUT, "tins_traces_*.zip"))):
        if os.path.isfile(f):
            print("  ", f, f"({os.path.getsize(f) / 1e6:.1f} MB)")
    finish(ctx, {"traces": len(TRACES())})
    '''),
]


# ============================================================================
# D1 · ViT-L/14 check: NegLabel and MCM against their published L/14 numbers
# ============================================================================
D1 = [
    md("""
    # D1 · Check of the ViT-L/14 set-up (about 10 minutes)

    In G3b, TANL with ViT-L/14 reaches Four-OOD FPR95 14.79, while TANL's paper reports 9.52 for ViT-L/14
    (iNaturalist 0.29, SUN 3.42, Places 20.79, Textures 13.57). With ViT-B/16 the same code reproduces TANL (9.89 vs
    9.81). This notebook tells a feature or text-bank problem apart from a TANL-specific one. It runs the two
    methods without memory, NegLabel and MCM, whose ViT-L/14 numbers are published, on both backbones' caches
    (one stream order is enough: they give the same result for every order), plus TANL with its released script's
    settings (`official_sh`) on ViT-L/14.

    * If NegLabel and MCM land near their published ViT-L/14 numbers, the L/14 features and text bank are right and
      the gap is TANL's own (the paper then reports our reproduction as it is).
    * If they are far off on L/14 but not on B/16, the L/14 cache or text bank is the problem.

    **Output:** `MyDrive/ReNeg/transfers/d1_vitl14_check/d1_runs.csv` (attach it).
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ctx = start("D1_vitl14_check", need_gpu=False)
    log = ctx.log
    import json, time
    import numpy as np, pandas as pd, torch
    from oodlab.report import find_cache
    from oodlab.runner import Runner
    from oodlab.methods import TANL, TANLConfig
    try:
        from oodlab.methods import MCM, MCMConfig, NegLabel, NegLabelConfig
    except ImportError:                                                  # the offline test stub of oodlab
        MCM = NegLabel = None
    STUB = NegLabel is None or not hasattr(TANL, "selected_T")
    from reneg.transport import l2n
    OUT = f"{DRIVE_ROOT}/transfers/d1_vitl14_check"
    os.makedirs(OUT, exist_ok=True)
    runs_path = os.path.join(OUT, "d1_runs.csv")
    # published Four-OOD FPR95 (iNaturalist, SUN, Places, Textures). NegLabel and MCM: ID positive (their papers);
    # TANL: OpenOOD's convention (OOD positive). NegLabel L/14 as listed in TANL's Table A12.
    PUB = {("neglabel", "ViT-B/16"): ([1.91, 20.53, 35.59, 43.56], "fpr95_idpos"),
           ("neglabel", "ViT-L/14"): ([1.77, 22.33, 32.22, 42.92], "fpr95_idpos"),
           ("mcm", "ViT-B/16"): ([30.91, 37.59, 44.69, 57.77], "fpr95_idpos"),
           ("mcm", "ViT-L/14"): ([28.38, 29.00, 35.42, 59.88], "fpr95_idpos"),
           ("tanl", "ViT-L/14"): ([0.29, 3.42, 20.79, 13.57], "fpr95"),
           ("tanl_official_sh", "ViT-L/14"): ([0.29, 3.42, 20.79, 13.57], "fpr95")}
    FOUR = ["inaturalist", "sun", "places", "textures_all"]
    print("device:", ctx.device)
    '''),
    md("""
    ## Run (saved after every method; re-running skips what is done)
    """),
    code(r'''
    done = pd.read_csv(runs_path) if os.path.isfile(runs_path) else pd.DataFrame()
    info = {}
    for bb in (() if STUB else ("ViT-B/16", "ViT-L/14")):
        try:
            cache = find_cache(ctx, bb)
        except Exception as e:                                           # noqa: BLE001
            print(bb, "cache not found:", e)
            continue
        runner = Runner(cache, device=ctx.device, logger=log)
        if runner.missing("four_ood"):
            print(bb, "Four-OOD sets missing:", runner.missing("four_ood"))
            continue
        bank = cache.load_textbank()
        v = runner.get("imagenet_val")
        acc = float(((l2n(v.feats.float()) @ l2n(bank.id_text.float()).T).argmax(1).numpy() == np.asarray(v.labels)).mean() * 100)
        info[bb] = {"zero_shot_top1_val5k": round(acc, 2), "dim": int(bank.id_text.shape[1]),
                    "corpus": int(bank.corpus_text.shape[0]), "n_selected": int(bank.n_selected),
                    **{k: bank.meta.get(k) for k in ("backbone", "mining_backbone", "prompt", "noise_seed")}}
        print(bb, info[bb])
        neg = bank.corpus_text[: int(bank.n_selected)]
        todo = {"neglabel": lambda: NegLabel(bank.id_text, neg, NegLabelConfig(), device=ctx.device),
                "mcm": lambda: MCM(bank.id_text_simple, MCMConfig.paper(), device=ctx.device)}   # "a photo of a {}."
        if bb == "ViT-L/14":
            todo["tanl_official_sh"] = lambda: TANL(bank.id_text, bank.corpus_text, bank.noise_feats,
                                                    TANLConfig.official_sh(), device=ctx.device,
                                                    n_neglabel=int(bank.n_selected))
        for name, make in todo.items():
            if len(done) and ((done["label"] == name) & (done["backbone"] == bb)).any():
                print("done already:", name, bb)
                continue
            t0 = time.time()
            df = runner.evaluate(make(), "four_ood", seeds=[0], config=name)
            df["label"], df["backbone"] = name, bb
            done = pd.concat([done, df], ignore_index=True)
            done.to_csv(runs_path, index=False)
            print(f"{name} {bb}: {(time.time() - t0) / 60:.1f} min")
    json.dump(info, open(os.path.join(OUT, "d1_info.json"), "w"), indent=1)
    if STUB:
        print("test stub of oodlab: nothing to run")
    '''),
    md("""
    ## Ours against the published numbers, then attach the files
    """),
    code(r'''
    df = pd.read_csv(runs_path) if os.path.isfile(runs_path) else pd.DataFrame(columns=["label", "backbone", "dataset"])
    g3 = f"{DRIVE_ROOT}/transfers/g3_final_l14/g3_runs.csv"           # TANL (paper settings) from G3b, for reference
    if os.path.isfile(g3):
        t = pd.read_csv(g3)
        t = t[(t.label == "tanl") & (t.protocol == "four_ood") & (t.seed == 0)].copy()
        t["backbone"] = "ViT-L/14"
        df = pd.concat([df, t], ignore_index=True)
    rows = []
    for (name, bb), g in df.groupby(["label", "backbone"]):
        g = g.set_index("dataset")
        pub, col = PUB.get((name, bb), (None, "fpr95"))
        ours = [float(g.loc[d, col]) if d in g.index else float("nan") for d in FOUR]
        row = {"method": name, "backbone": bb, "convention": "ID positive" if col == "fpr95_idpos" else "OOD positive",
               **{d: round(x, 2) for d, x in zip(FOUR, ours)}, "mean": round(float(np.mean(ours)), 2)}
        if pub:
            row["published mean"] = round(float(np.mean(pub)), 2)
            row["gap"] = round(row["mean"] - row["published mean"], 2)
        rows.append(row)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nresult files:")
    for f in (runs_path, os.path.join(OUT, "d1_info.json")):
        if os.path.isfile(f):
            print("  ", f)
    finish(ctx, {"rows": int(len(df))})
    '''),
]


# ============================================================================
# G4 · ReNeg on top of TINS (and of TANL+TINS): the official run of the probe that passed
# ============================================================================
G4 = [
    md("""
    # G4 · ReNeg on top of TINS: the final run (checklist 3.2b)

    The probe P4 passed its pre-registered test: replayed on TINS's per-image scores, ReNeg-balanced on top of TINS
    beat TINS on near-OOD, far-OOD and Four-OOD in every stream order. This notebook produces those numbers the
    official way, on the exact features and in one pass per stream: TINS and TANL run once per batch, and several
    ReNeg heads read their scores (ReNeg never changes its base, so each head scores exactly as it would alone).

    * Labelled images: the same 5 per class as P4 (OpenOOD's ImageNet ID-val split). They are TINS's prototypes and
      enter every ReNeg head's ID image prototypes at each reset ("5-shot").
    * Protocols: OpenOOD v1.5, and Four-OOD on the 45k ID images that are not labelled images (`four_ood_45k`, the
      same streams as G3's few-shot rows). Three stream orders.
    * Scores saved per image for every head, so nothing has to run again for a new table.

    **Time:** about 2–3 hours on an RTX 4060-class GPU (TINS dominates; P4 took 1 h 48 min). Every stream saves when it
    finishes; after a stop, **Run all** again and finished streams are skipped.

    **Output:** `MyDrive/ReNeg/transfers/g4_tins/`: `g4_runs.csv`, `g4_summary.csv` and `g4_scores_*.zip` (attach all).
    """),
    code(COLAB_SETUP_CELL),
    code(r'''
    from oodlab.session import start, finish, ensure_packages
    ensure_packages(clip=True)
    ctx = start("G4_reneg_on_tins", need_gpu=True)
    log = ctx.log
    import json, time, glob, zipfile, shutil
    import numpy as np, pandas as pd, torch
    from oodlab.report import find_cache
    from oodlab.runner import Runner, Protocol
    from oodlab.metrics import openood_metrics
    from oodlab.clipwrap import load_clip
    from oodlab.methods import TANL, TANLConfig
    from reneg.core import ReNegConfig
    from reneg.multi import HeadSpec, MultiBase
    from reneg.packs import load_kg_pool
    from reneg.tins import TINSConfig, TINSFeat, build_init_candidates, build_static_negatives
    from reneg.transport import l2n
    cache = find_cache(ctx, "ViT-B/16")
    bank = cache.load_textbank()
    runner = Runner(cache, device=ctx.device, logger=log)
    OUT = f"{DRIVE_ROOT}/transfers/g4_tins"
    SC = os.path.join(OUT, "g4_scores")
    os.makedirs(SC, exist_ok=True)
    SEEDS = [0, 1, 2]
    WEIGHTS = f"{DRIVE_ROOT}/weights"
    model = load_clip("ViT-B/16", device=ctx.device, input_roots=ctx.input_roots + [WEIGHTS], download_root=WEIGHTS,
                      random_init=os.environ.get("RENEG_FAKE_GEN") == "1")
    REAL = (hasattr(model, "token_embedding") and hasattr(TANL, "selected_T")
            and all(s in cache.available() for s in ("imagenet_val", "imagenet_val_all")))
    print("device:", ctx.device, "| real oodlab and CLIP text encoder:", REAL)
    '''),
    md("""
    ## Labelled images, TINS's negatives (reused from P4), the KG pool and the 45k Four-OOD protocol
    """),
    code(r'''
    if REAL:
        v5 = runner.get("imagenet_val")
        X5, y5 = l2n(v5.feats.float()), torch.as_tensor(np.asarray(v5.labels)).long()
        C = len(bank.id_names)
        cnt = torch.bincount(y5, minlength=C)
        proto = l2n(torch.zeros(C, X5.shape[1]).index_add_(0, y5, X5))
        if (cnt == 0).any():
            print(f"WARNING: {int((cnt == 0).sum())} classes have no labelled image: their text embedding is used")
            proto[cnt == 0] = l2n(bank.id_text.float())[cnt == 0]
        prior = (v5.feats, np.asarray(v5.labels))
        static_path = os.path.join(OUT, "tins_static.pt")
        p4_static = f"{DRIVE_ROOT}/transfers/p4_tins/tins_static.pt"
        if not os.path.isfile(static_path) and os.path.isfile(p4_static):
            shutil.copy2(p4_static, static_path)
        if os.path.isfile(static_path):
            st = torch.load(static_path, weights_only=False)
            assert torch.allclose(st["proto"], proto, atol=1e-5), "the labelled images differ from P4's"
        else:
            neg, words = build_static_negatives(model, bank.id_text.float(), proto, n=2000,
                                                positive_labels=list(bank.id_names), log=print)
            st = {"neg": neg, "words": words, "cands": build_init_candidates(model, words, proto), "proto": proto}
            torch.save(st, static_path)
        print(len(st["words"]), "TINS static negatives;", len(X5), "labelled ID images")
        # Four-OOD without the labelled images (as G3's few-shot rows)
        va = runner.get("imagenet_val_all")
        A, B = l2n(va.feats.float().to(ctx.device)), X5.to(ctx.device)
        shot = torch.zeros(len(va), dtype=torch.bool)
        for s in range(0, len(va), 4096):
            shot[s:s + 4096] = ((A[s:s + 4096] @ B.T).max(1).values > 0.999).cpu()
        shot = shot.numpy()
        np.save(os.path.join(OUT, "shot_mask.npy"), shot)
        del A, B
        runner.extra_sets["imagenet_val_45k"] = va.subset(np.nonzero(~shot)[0], name="imagenet_val_45k")
        P45 = Protocol("four_ood_45k", "imagenet_val_45k", {"four45": ["inaturalist", "sun", "places", "textures_all"]},
                       "Four-OOD without OpenOOD's 5k ID-val images (they are the labelled images)")
        print("labelled images found in the 50k Four-OOD ID set:", int(shot.sum()))
        # KG pool (as G3)
        kg = load_kg_pool()["pool"]
        z = np.load(f"{DRIVE_ROOT}/transfers/analysis_pack2/reneg_pool.npz")
        assert [r["name"] for r in kg] == list(z["pool_names"]), "pool file and kg_pool.json disagree"
        pool_text = l2n(torch.tensor(z["pool_text"]).float())
        pool_cluster = np.array([r["cluster"] for r in kg])
        blind = ~np.array([bool(r["in_ssb_hard"]) or bool(r["in_ninco"]) for r in kg])
        pools = {"full": (pool_text, pool_cluster),
                 "blind": (pool_text[torch.as_tensor(np.nonzero(blind)[0])], pool_cluster[blind])}
        BASE = dict(n_min=1, pool_frac=0.0, pool_in_text=False, id_admit="softmax+base")
        MODES = {"balanced": ReNegConfig(**BASE, image_mode="kg_clip", vote=True), "max": ReNegConfig(**BASE),
                 "safe": ReNegConfig(**BASE, image_mode="kg_clip_caught", vote=True)}
        HEADS = [HeadSpec("balanced on tins", "tins", MODES["balanced"]),
                 HeadSpec("max on tins", "tins", MODES["max"]),
                 HeadSpec("safe on tins", "tins", MODES["safe"]),
                 HeadSpec("balanced on tins, full pool", "tins", MODES["balanced"], pool="full"),
                 HeadSpec("balanced on tins, no labelled images", "tins", MODES["balanced"], prior=False),
                 HeadSpec("balanced on tanl+tins", "tanl+tins", MODES["balanced"]),
                 HeadSpec("max on tanl+tins", "tanl+tins", MODES["max"]),
                 HeadSpec("safe on tanl+tins", "tanl+tins", MODES["safe"]),
                 HeadSpec("balanced on tanl", "tanl", MODES["balanced"])]
        NAMES = ["tins", "tanl", "tanl+tins"] + [h.name for h in HEADS]
        print(len(kg), "KG names,", int(blind.sum()), "in the blind pool;", len(HEADS), "ReNeg heads")
    else:
        print("test stub of oodlab (no TANL / CLIP text encoder): the G4 cells are skipped")
    '''),
    md("""
    ## Timing check on one stream (NINCO, stream order 0) and an estimate for the whole run
    """),
    code(r'''
    def make_multi():
        tins = TINSFeat(model, bank.id_text.float(), st["proto"], st["neg"], st["cands"], TINSConfig())
        tanl = TANL(bank.id_text, bank.corpus_text, bank.noise_feats, TANLConfig.paper().with_(record=True),
                    device=ctx.device, n_neglabel=int(bank.n_selected))
        return MultiBase(tins, tanl, bank.id_text, pools, HEADS, primary="balanced on tins", device=ctx.device,
                         id_prior=prior)

    class ScoreHook:
        """Saves every score (bases and heads) per image in stream order, with the labels and stream rows."""
        def __init__(self, path):
            self.path, self.parts = path, []
        def on_reset(self, method, seg):
            self.parts = []
        def on_step(self, method, seg, start, end, out, stream):
            self.parts.append((start, {k: v.numpy() for k, v in out.info["scores"].items()}))
        def on_end(self, method, stream):
            n = len(stream)
            arr = np.zeros((len(NAMES), n), np.float32)
            for s, d in self.parts:
                for i, k in enumerate(NAMES):
                    arr[i, s:s + len(d[k])] = d[k]
            tmp = self.path + ".tmp.npz"
            np.savez_compressed(tmp, scores=arr, names=np.asarray(NAMES), is_ood=(np.asarray(stream.labels) == -1),
                                rows=stream.rows.astype(np.int32), n_inv=method.tins.n_inv, t_inv=method.tins.t_inv)
            os.replace(tmp, self.path)

    def score_path(protocol, ds, seed):
        return os.path.join(SC, f"{protocol}__{ds}__s{seed}.npz")

    runs_path = os.path.join(OUT, "g4_runs.csv")

    def rows_from(path, protocol, ds, seed, seconds=None):
        z = np.load(path)
        lab = np.where(z["is_ood"], -1, 0)
        out = []
        for i, k in enumerate(z["names"]):
            m = openood_metrics(z["scores"][i], lab)
            out.append({"label": str(k), "protocol": protocol, "dataset": ds, "seed": seed, "fpr95": m["fpr95"],
                        "fpr95_idpos": m["fpr95_idpos"], "auroc": m["auroc"], "n_id": m["n_id"], "n_ood": m["n_ood"],
                        "n_inverted": int(z["n_inv"]), "inversion_seconds": float(z["t_inv"]), "stream_seconds": seconds})
        return out

    def run_one(protocol, ds, seed):
        global done
        path = score_path(protocol, ds, seed)
        if os.path.isfile(path):
            return None
        t0 = time.time()
        runner.evaluate(make_multi(), P45 if protocol == "four_ood_45k" else protocol, seeds=[seed], datasets=[ds],
                        config="g4", hooks_factory=lambda name, sd: [ScoreHook(score_path(protocol, name, sd))])
        rows = pd.DataFrame(rows_from(path, protocol, ds, seed, time.time() - t0))
        done = pd.concat([done, rows], ignore_index=True)
        done.to_csv(runs_path, index=False)
        r = rows.set_index("label")
        print(f"{protocol} {ds} s{seed}: TINS {r.loc['tins', 'fpr95_idpos']:.2f} | ReNeg-balanced on TINS "
              f"{r.loc['balanced on tins', 'fpr95_idpos']:.2f} | on TANL+TINS {r.loc['balanced on tanl+tins', 'fpr95_idpos']:.2f}"
              f" (FPR95, ID positive); {(time.time() - t0) / 60:.1f} min", flush=True)
        return rows

    SETS = {"openood_v15": (["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"], "imagenet_test"),
            "four_ood_45k": (["inaturalist", "sun", "places", "textures_all"], "imagenet_val_45k")}
    done = pd.read_csv(runs_path) if os.path.isfile(runs_path) else pd.DataFrame()
    if REAL:
        run_one("openood_v15", "ninco", 0)
        sel = done[(done.protocol == "openood_v15") & (done.dataset == "ninco") & (done.seed == 0)]
        if len(sel) and pd.notna(sel.stream_seconds.iloc[0]):
            per = float(sel.stream_seconds.iloc[0]) / len(np.load(score_path("openood_v15", "ninco", 0))["is_ood"])
            n_img = sum(len(runner.get(d)) + len(runner.get(SETS[p][1])) for p in SETS for d in SETS[p][0])
            print(f"rough estimate: {len(SEEDS) * n_img * per / 3600:.1f} hours for everything (a lower bound: NINCO "
                  f"has few OOD images; P4, TINS alone, took 1 h 48 min on an RTX 4060)")
    '''),
    md("""
    ## Run every stream (stream order 0 for all sets first, then 1 and 2; saved per stream)
    """),
    code(r'''
    if REAL:
        for seed in SEEDS:
            for protocol in SETS:
                for ds in SETS[protocol][0]:
                    run_one(protocol, ds, seed)
    '''),
    md("""
    ## Summary (both FPR95 conventions), checks, and the files to attach

    Pass rule fixed before P4's numbers: ReNeg-balanced on TINS (blind pool, 5-shot) beats TINS on near-OOD, far-OOD
    and Four-OOD in ID-positive FPR95 (mean of three orders), and on near and far in every order. `balanced on tanl`
    should equal G3's `reneg_balanced_blind_5shot` on the same streams (same code, same seeds).
    """),
    code(r'''
    def TRACES():
        return sorted(f for f in glob.glob(os.path.join(SC, "*.npz")) if not f.endswith(".tmp.npz"))

    allrows = []
    for f in TRACES():
        protocol, ds, s = os.path.basename(f)[:-4].split("__")
        allrows += rows_from(f, protocol, ds, int(s[1:]))
    if allrows:
        df = pd.DataFrame(allrows)
        GROUPS = {"near": ("openood_v15", ["ssb_hard", "ninco"]),
                  "far": ("openood_v15", ["inaturalist", "textures", "openimage_o"]),
                  "Four-OOD 45k": ("four_ood_45k", ["inaturalist", "sun", "places", "textures_all"])}

        def gseed(lab, prot, sets_, col):
            g = df[(df.label == lab) & (df.protocol == prot) & df.dataset.isin(sets_)]
            n = g.groupby("seed").size()
            return g[g.seed.isin(n[n == len(sets_)].index)].groupby("seed")[col].mean()

        rows = []
        for lab in NAMES:
            row = {"method": lab}
            for gname, (prot, sets_) in GROUPS.items():
                for col, short in (("fpr95_idpos", "FPR95 ID+"), ("fpr95", "FPR95 OOD+"), ("auroc", "AUROC")):
                    v = gseed(lab, prot, sets_, col)
                    if len(v):
                        row[f"{gname} {short}"] = round(v.mean(), 2)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(OUT, "g4_summary.csv"), index=False)
        pd.set_option("display.width", 250)
        print(summary.to_string(index=False))
        print("\npass rule (ID-positive FPR95, ReNeg-balanced on TINS minus TINS, per order):")
        ok = True
        for gname, (prot, sets_) in GROUPS.items():
            d = (gseed("balanced on tins", prot, sets_, "fpr95_idpos") - gseed("tins", prot, sets_, "fpr95_idpos")).dropna()
            good = len(d) == len(SEEDS) and d.mean() < 0 and (gname == "Four-OOD 45k" or (d < 0).all())
            ok &= good
            print(f"  {gname}: mean {d.mean():+.2f}, per order {np.round(d.values, 2).tolist()} -> {'PASS' if good else 'FAIL'}")
        print("  overall:", "PASS" if ok else "FAIL (or not finished)")
        g3 = f"{DRIVE_ROOT}/transfers/g3_final_b16/g3_runs.csv"
        if os.path.isfile(g3):
            t = pd.read_csv(g3)
            for mine, theirs in (("balanced on tanl", "reneg_balanced_blind_5shot"), ("tanl", "tanl")):
                a = df[df.label == mine].merge(t[t.label == theirs], on=["protocol", "dataset", "seed"], suffixes=("", "_g3"))
                if len(a):
                    print(f"check vs G3 {theirs}: {len(a)} streams, largest FPR95 difference "
                          f"{(a.fpr95 - a.fpr95_g3).abs().max():.3f} (expect a few tenths at most: G3 used the "
                          f"5,002 matched copies inside the 50k set as its labelled images)")
    parts, cur, size = [], [], 0
    for f in TRACES():
        if cur and size + os.path.getsize(f) > 22e6:
            parts.append(cur); cur, size = [], 0
        cur.append(f); size += os.path.getsize(f)
    if cur:
        parts.append(cur)
    for old in glob.glob(os.path.join(OUT, "g4_scores_*.zip")):
        os.remove(old)
    for k, fs in enumerate(parts, 1):
        with zipfile.ZipFile(os.path.join(OUT, f"g4_scores_{k}.zip"), "w") as zf:
            for f in fs:
                zf.write(f, os.path.basename(f))
    print("\nresult files:")
    for f in [runs_path, os.path.join(OUT, "g4_summary.csv")] + sorted(glob.glob(os.path.join(OUT, "g4_scores_*.zip"))):
        if os.path.isfile(f):
            print("  ", f, f"({os.path.getsize(f) / 1e6:.1f} MB)")
    finish(ctx, {"streams": len(TRACES())})
    '''),
]


if __name__ == "__main__":
    save("C0_transfer_from_kaggle.ipynb", C0 + DRIVE_FLUSH)
    save("C1_colab_setup_check.ipynb", C1 + DRIVE_FLUSH, gpu=True)
    save("G1_transport_gate.ipynb", G1 + DRIVE_FLUSH, gpu=True)
    save("G1b_transport_diagnostics.ipynb", G1B + DRIVE_FLUSH, gpu=True)
    save("G2a_pool_and_streams.ipynb", G2A + DRIVE_FLUSH)
    save("G2b_textbank_and_four_ood.ipynb", G2B + DRIVE_FLUSH)
    save("P1_generation_probe.ipynb", P1 + DRIVE_FLUSH, gpu=True)
    save("P2_prior_probe.ipynb", P2 + DRIVE_FLUSH, gpu=True)
    save("G3_final_eval.ipynb", G3 + DRIVE_FLUSH, gpu=True)
    save("G3b_final_eval_vitl14.ipynb", G3B + DRIVE_FLUSH, gpu=True)
    save("P4_tins_probe.ipynb", P4 + DRIVE_FLUSH, gpu=True)
    save("D1_vitl14_check.ipynb", D1 + DRIVE_FLUSH)
    save("G4_reneg_on_tins.ipynb", G4 + DRIVE_FLUSH, gpu=True)
