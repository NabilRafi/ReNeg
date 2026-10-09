"""Google Drive / Colab layout and the Kaggle -> Colab transfer (used by notebook C0).

Drive layout (everything that must survive a Colab session lives under DRIVE_ROOT)::

    MyDrive/ReNeg/
        uploads/            zips you upload by hand (this bundle, oodlab_code.zip, Kaggle output zips)
        code/oodlab_code/   the oodlab package (oodlab/, tests/, tools/, ...)
        code/reneg_code/    this package (reneg/, tests/)
        notebooks/          C0, C1, G1 (unpacked from the bundle)
        notebooks_colab/    your Kaggle notebooks 00-04 converted for Colab
        oodlab_working/     OODLAB_HOME/working: cache/, results/, logs/   (what oodlab reads and writes)
        transfers/          raw downloads from Kaggle (safe to delete once C0 passes)
        weights/            CLIP weights, downloaded once

On Colab, OODLAB_HOME is /content/oodlab_home (fast local disk) and its ``working`` folder is a
link to MyDrive/ReNeg/oodlab_working, so caches and results persist while scratch files do not.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/ReNeg"

EXPECTED_SETS = {
    "imagenet_val_all": 50000, "imagenet_test": 45000, "imagenet_val": 5000,
    "inaturalist": 10000, "sun": 10000, "places": 10000, "textures_all": 5640, "textures": None,
    "openimage_o": 15869, "openimage_o_val": 1763, "ssb_hard": 49000, "ninco": 5879,
}

KAGGLE_NOTEBOOKS = {        # key -> slug on Kaggle (the end of the notebook's URL)
    "00": "notebook0",
    "01": "notebook1",
    "02": "notebook2",
    "03": "notebook-three",     # titled "Notebook_three" on Kaggle
    "04": "notebook-four",      # titled "Notebook_four" on Kaggle
}


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def resolve_slugs(wanted: Dict[str, str], rows: List[Dict]) -> Dict[str, Optional[str]]:
    """Match each wanted name to a real notebook on the account, by slug or by title.

    "Notebook_three", "notebook-three" and "notebook three" all match, because case, spaces,
    hyphens and underscores are ignored. Returns key -> real slug, or None if nothing matches.
    """
    slugs = [(str(r.get("ref", "")).split("/")[-1], str(r.get("title") or "")) for r in rows]
    out = {}
    for k, v in wanted.items():
        g = guess_slug(v)
        tests = (lambda s, t: s == v,                    # exact slug first: "notebook1" is not "notebook-1"
                 lambda s, t: s == g,
                 lambda s, t: t == v,
                 lambda s, t: t.lower() == str(v).lower(),
                 lambda s, t: _norm(s) == _norm(v) or _norm(t) == _norm(v))
        out[k] = next((s for test in tests for s, t in slugs if test(s, t)), None)
    return out

# The first code cell of every Colab notebook. Kept here so C1, G1 and the converted Kaggle
# notebooks use exactly the same text.
COLAB_SETUP_CELL = r'''# ── Colab set-up (the same cell in every ReNeg notebook) ─────────────────────────────
import os, sys
DRIVE_ROOT = "/content/drive/MyDrive/ReNeg"          # change only if your project folder is elsewhere
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:                                    # running outside Colab (tests)
    IN_COLAB = False
    DRIVE_ROOT = os.environ.get("RENEG_DRIVE_ROOT", os.path.abspath("ReNeg"))
HOME = os.environ.get("RENEG_OODLAB_HOME", "/content/oodlab_home" if IN_COLAB else os.path.abspath("oodlab_home"))
for d in (f"{DRIVE_ROOT}/oodlab_working", f"{HOME}/scratch", f"{HOME}/input"):
    os.makedirs(d, exist_ok=True)
if not os.path.islink(f"{HOME}/working"):              # working/ lives on Drive, scratch/ stays local
    if os.path.isdir(f"{HOME}/working") and not os.listdir(f"{HOME}/working"):
        os.rmdir(f"{HOME}/working")
    if not os.path.exists(f"{HOME}/working"):
        os.symlink(f"{DRIVE_ROOT}/oodlab_working", f"{HOME}/working")
os.environ["OODLAB_HOME"] = HOME
# a newer ReNeg_Colab_bundle*.zip in uploads/ is installed here, so a new bundle needs no C0 re-run
import glob, re, shutil, zipfile
def _reneg_version(text):
    m = re.search(r'__version__ = "([0-9.]+)"', text or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else None
_init = f"{DRIVE_ROOT}/code/reneg_code/reneg/__init__.py"
_installed = _reneg_version(open(_init).read()) if os.path.isfile(_init) else None
_newest = None
for _zp in glob.glob(f"{DRIVE_ROOT}/uploads/*.zip"):
    try:
        with zipfile.ZipFile(_zp) as _z:
            _hit = [n for n in _z.namelist() if n.endswith("reneg_code/reneg/__init__.py")]
            _v = _reneg_version(_z.read(_hit[0]).decode()) if _hit else None
        if _v and (_newest is None or _v > _newest[0]):
            _newest = (_v, _zp, _hit[0][: -len("reneg_code/reneg/__init__.py")])
    except (zipfile.BadZipFile, OSError):
        pass
if _newest and (_installed is None or _newest[0] > _installed):
    _tmp = os.path.join(HOME, "scratch", "_bundle")
    shutil.rmtree(_tmp, ignore_errors=True)
    with zipfile.ZipFile(_newest[1]) as _z:
        _z.extractall(_tmp)
    _src = os.path.join(_tmp, _newest[2])
    shutil.rmtree(f"{DRIVE_ROOT}/code/reneg_code", ignore_errors=True)
    shutil.copytree(os.path.join(_src, "reneg_code"), f"{DRIVE_ROOT}/code/reneg_code")
    os.makedirs(f"{DRIVE_ROOT}/notebooks", exist_ok=True)
    for _f in os.listdir(os.path.join(_src, "notebooks")):
        if _f.endswith(".ipynb") and not _f.startswith("C0"):
            shutil.copy2(os.path.join(_src, "notebooks", _f), f"{DRIVE_ROOT}/notebooks/{_f}")
    _fmt = lambda v: ".".join(map(str, v)) if v else "none"
    print(f"reneg code updated {_fmt(_installed)} -> {_fmt(_newest[0])} from uploads/{os.path.basename(_newest[1])}")
    if "reneg" in sys.modules:
        print("reneg was already imported in this session: Runtime -> Restart session, then run again")
CODE_ROOT = f"{DRIVE_ROOT}/code/oodlab_code"           # same name as in the Kaggle notebooks
for p in (CODE_ROOT, f"{DRIVE_ROOT}/code/reneg_code"):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{p} not found: run C0_transfer_from_kaggle first (GUIDE_COLAB.md, step 4)")
    if p not in sys.path:
        sys.path.insert(0, p)
import oodlab, reneg
print(f"oodlab {oodlab.__version__} | reneg {reneg.__version__} | project folder {DRIVE_ROOT}")
'''


def layout(drive_root: str = DEFAULT_DRIVE_ROOT, make: bool = True) -> Dict[str, str]:
    d = {k: os.path.join(drive_root, v) for k, v in {
        "root": "", "uploads": "uploads", "code": "code", "oodlab_code": "code/oodlab_code",
        "reneg_code": "code/reneg_code", "notebooks": "notebooks", "notebooks_colab": "notebooks_colab",
        "working": "oodlab_working", "cache": "oodlab_working/cache", "results": "oodlab_working/results",
        "transfers": "transfers", "weights": "weights"}.items()}
    d["root"] = drive_root
    if make:
        for k in ("uploads", "code", "notebooks", "notebooks_colab", "working", "cache", "results", "transfers",
                  "weights"):
            os.makedirs(d[k], exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# small file-system helpers (bounded, never follow symlinks)
# ---------------------------------------------------------------------------


def walk_bounded(root: str, max_depth: int = 8, max_entries: int = 20000) -> Iterable[str]:
    """Yield file paths under root breadth-first, without following symlinked directories."""
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop(0)
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        if len(entries) > max_entries:
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                if depth < max_depth and e.name not in ("__MACOSX", ".git", "__pycache__"):
                    stack.append((e.path, depth + 1))
            elif e.is_file(follow_symlinks=False):
                yield e.path


def find_file(root: str, name: str, max_depth: int = 8) -> Optional[str]:
    for p in walk_bounded(root, max_depth=max_depth):
        if os.path.basename(p) == name:
            return p
    return None


def find_package_root(root: str, package: str, max_depth: int = 6) -> Optional[str]:
    """The folder that contains ``<package>/__init__.py`` (e.g. .../oodlab_code for oodlab)."""
    for p in walk_bounded(root, max_depth=max_depth):
        if p.endswith(os.path.join(package, "__init__.py")) and os.path.basename(os.path.dirname(p)) == package:
            return os.path.dirname(os.path.dirname(p))
    return None


def copy_tree(src: str, dst: str, overwrite: bool = False) -> int:
    """Copy files from src into dst; returns the number of files copied (same-size files are skipped)."""
    n = 0
    for p in walk_bounded(src, max_depth=12):
        rel = os.path.relpath(p, src)
        q = os.path.join(dst, rel)
        if os.path.exists(q) and not overwrite and os.path.getsize(q) == os.path.getsize(p):
            continue
        os.makedirs(os.path.dirname(q), exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


def unzip(zip_path: str, dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def expand_nested_zips(root: str, max_depth: int = 6, log=print) -> List[str]:
    """Unzip zips found inside downloaded folders once (e.g. a Kaggle dataset that holds oodlab_code.zip)."""
    out = []
    for p in list(walk_bounded(root, max_depth=max_depth)):
        if not p.lower().endswith(".zip"):
            continue
        dest = p[:-4]
        if os.path.isdir(dest) and os.listdir(dest):
            continue
        try:
            unzip(p, dest)
            out.append(dest)
            log(f"expanded {os.path.relpath(p, root)}")
        except zipfile.BadZipFile:
            log(f"skipped (not a zip): {p}")
    return out


def unpack_uploads(L: Dict[str, str], log=print) -> List[str]:
    """Unzip every zip in uploads/ into transfers/<zip name>/ (once)."""
    done = []
    for f in sorted(os.listdir(L["uploads"])):
        if not f.lower().endswith(".zip"):
            continue
        dest = os.path.join(L["transfers"], "uploads", f[:-4])
        if os.path.isdir(dest) and os.listdir(dest):
            done.append(dest)
            continue
        log(f"unzipping {f} ...")
        unzip(os.path.join(L["uploads"], f), dest)
        done.append(dest)
    return done


# ---------------------------------------------------------------------------
# Kaggle API
# ---------------------------------------------------------------------------


def run(cmd: Sequence[str], check: bool = False, quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print("$", " ".join(cmd), flush=True)
    r = subprocess.run(list(cmd), capture_output=True, text=True)
    if not quiet:
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if out:
            print(out[-3000:])
        if err:
            print(err[-3000:])
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r


def setup_kaggle_credentials(log=print) -> Optional[str]:
    """Read Kaggle credentials from Colab Secrets (or an uploaded kaggle.json). Returns the username.

    Colab: the key icon in the left sidebar -> add KAGGLE_USERNAME and KAGGLE_KEY (or a single
    KAGGLE_API_TOKEN), and switch on "Notebook access" for both.
    """
    user = key = token = None
    try:
        from google.colab import userdata  # type: ignore

        for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            try:
                val = userdata.get(name)
            except Exception:
                val = None
            if val:
                if name == "KAGGLE_USERNAME":
                    user = val
                elif name == "KAGGLE_KEY":
                    key = val
                else:
                    token = val
    except ImportError:
        pass
    user = user or os.environ.get("KAGGLE_USERNAME")
    key = key or os.environ.get("KAGGLE_KEY")
    token = token or os.environ.get("KAGGLE_API_TOKEN")
    kj = os.path.expanduser("~/.kaggle/kaggle.json")
    if user and key:
        os.makedirs(os.path.dirname(kj), exist_ok=True)
        with open(kj, "w") as f:
            json.dump({"username": user, "key": key}, f)
        os.chmod(kj, 0o600)
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, key
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    if not user and os.path.isfile(kj):
        with open(kj) as f:
            user = json.load(f).get("username")
    if not (user or token):
        log("No Kaggle credentials found. Add KAGGLE_USERNAME and KAGGLE_KEY in Colab Secrets (key icon), "
            "switch on notebook access, and run this cell again. GUIDE_COLAB.md step 3.")
    return user


def ensure_kaggle_cli() -> None:
    try:
        import kaggle  # noqa: F401
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])


def guess_slug(name: str) -> str:
    """What Kaggle turns a notebook title into: 'Notebook_three' -> 'notebook-three'."""
    import re

    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return s


def _field(obj, *names):
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v not in (None, ""):
            return v
    return None


def _kernels_via_python_api(log=print) -> List[Dict]:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

    api = KaggleApi()
    api.authenticate()
    out, page = [], 1
    while page <= 10:
        try:
            ks = api.kernels_list(mine=True, page=page, page_size=100)
        except TypeError:                                   # very old clients
            ks = api.kernels_list(mine=True, page=page)
        ks = list(ks or [])
        for k in ks:
            ref = _field(k, "ref")
            if not ref:
                author, slug = _field(k, "author", "author_name"), _field(k, "slug", "url_slug")
                ref = f"{author}/{slug}" if author and slug else None
            out.append({"ref": str(ref) if ref else None,
                        "title": _field(k, "title"),
                        "lastRunTime": _field(k, "lastRunTime", "last_run_time")})
        if len(ks) < 100:
            break
        page += 1
    return out


def _kernels_via_cli(log=print) -> List[Dict]:
    import csv
    import io

    r = run(["kaggle", "kernels", "list", "--mine", "--page-size", "100", "--csv"], quiet=True)
    text = r.stdout or ""
    if r.returncode != 0:
        log((r.stderr or text).strip()[-2000:])
        return []
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "ref" in ln.lower() and "title" in ln.lower()), None)
    if start is None:                                        # show what came back, so it can be fixed
        log("Could not read `kaggle kernels list` output. First lines:\n  " + "\n  ".join(lines[:6]))
        return []
    out = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        low = {str(k).strip().strip('"').lower(): v for k, v in row.items() if k is not None}
        out.append({"ref": low.get("ref"), "title": low.get("title"),
                    "lastRunTime": low.get("lastruntime") or low.get("last_run_time")})
    return out


def list_my_kernels(log=print) -> List[Dict]:
    """Your Kaggle notebooks as dicts with ref (user/slug), title and lastRunTime.

    Tries the Kaggle Python API first and the command line second; rows without a ref are dropped.
    """
    rows: List[Dict] = []
    try:
        rows = _kernels_via_python_api(log)
    except Exception as e:                                   # noqa: BLE001
        log(f"(Kaggle Python API listing failed: {type(e).__name__}: {str(e)[:200]}; trying the command line)")
    if not any(r.get("ref") for r in rows):
        try:
            rows = _kernels_via_cli(log)
        except Exception as e:                               # noqa: BLE001
            log(f"(command-line listing failed: {e})")
    return [r for r in rows if r.get("ref")]


def resolve_or_guess(wanted: Dict[str, str], rows: List[Dict], log=print) -> Dict[str, str]:
    """resolve_slugs, then Kaggle's own slug rule for anything the listing did not show."""
    found = resolve_slugs(wanted, rows)
    out = {}
    for k, name in wanted.items():
        if found.get(k):
            log(f"    found    {k}: {name} -> {found[k]}")
            out[k] = found[k]
        else:
            g = guess_slug(name)
            log(f"    guessed  {k}: {name} -> {g}   (not in the listing; the download below will tell if it is right)")
            out[k] = g
    return out


def download_dataset(slug: str, dest: str) -> bool:
    os.makedirs(dest, exist_ok=True)
    return run(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip", "-o"]).returncode == 0


def download_kernel_output(ref: str, dest: str) -> bool:
    os.makedirs(dest, exist_ok=True)
    return run(["kaggle", "kernels", "output", ref, "-p", dest, "-o"]).returncode == 0


def pull_kernel(ref: str, dest: str) -> Optional[str]:
    os.makedirs(dest, exist_ok=True)
    if run(["kaggle", "kernels", "pull", ref, "-p", dest]).returncode != 0:
        return None
    nbs = [f for f in os.listdir(dest) if f.endswith(".ipynb")]
    slug = ref.split("/")[-1]
    exact = [f for f in nbs if f.startswith(slug)]
    pick = (exact or nbs or [None])[0]
    return os.path.join(dest, pick) if pick else None


# ---------------------------------------------------------------------------
# installing what was downloaded
# ---------------------------------------------------------------------------


def install_code(search_roots: Sequence[str], package: str, dest: str, log=print) -> Optional[str]:
    """Find a folder holding ``<package>/__init__.py`` under search_roots and copy it to dest."""
    for r in search_roots:
        if not os.path.isdir(r):
            continue
        src = find_package_root(r, package)
        if src:
            if os.path.abspath(src) != os.path.abspath(dest):
                n = copy_tree(src, dest, overwrite=True)
                log(f"{package}: copied {n} files from {src} to {dest}")
            return dest
    return None


def install_cache(search_roots: Sequence[str], cache_root: str, log=print) -> List[str]:
    """Find every oodlab cache folder (the folder holding textbank.pt) and copy it under cache_root."""
    installed = []
    for r in search_roots:
        if not os.path.isdir(r):
            continue
        for p in walk_bounded(r, max_depth=8):
            if os.path.basename(p) != "textbank.pt":
                continue
            src = os.path.dirname(p)                         # .../cache/vit-b-16
            tag = os.path.basename(src)
            if os.path.abspath(src) == os.path.abspath(os.path.join(cache_root, tag)):
                continue
            n = copy_tree(src, os.path.join(cache_root, tag))
            log(f"cache {tag}: copied {n} files from {src}")
            installed.append(tag)
    return sorted(set(installed))


def install_results(search_roots: Sequence[str], results_root: str, log=print) -> List[str]:
    """Copy every notebook's results/<name>/ folder found under search_roots."""
    names = []
    for r in search_roots:
        if not os.path.isdir(r):
            continue
        for p in walk_bounded(r, max_depth=6):
            parts = os.path.relpath(p, r).split(os.sep)
            if "results" in parts:
                i = parts.index("results")
                if i + 1 < len(parts) - 1:
                    name = parts[i + 1]
                    src = os.path.join(r, *parts[: i + 2])
                    dst = os.path.join(results_root, name)
                    if name not in names and os.path.abspath(src) != os.path.abspath(dst):
                        n = copy_tree(src, dst)
                        log(f"results/{name}: copied {n} files")
                        names.append(name)
    return names


def verify_cache(cache_root: str, backbone: str = "ViT-B/16"):
    """Rows (set, images, expected, status) for the cache, plus the text-bank summary."""
    from oodlab.cache import FeatureCache

    c = FeatureCache(cache_root, backbone)
    rows = []
    for name, exp in EXPECTED_SETS.items():
        n = None
        meta = os.path.join(c.images_dir, f"{name}.json")
        if os.path.isfile(meta):
            with open(meta) as f:
                n = json.load(f).get("n")
        if n is None and c.has(name):
            import numpy as np

            with np.load(c.set_path(name), allow_pickle=False) as z:
                n = int(z["labels"].shape[0])
        if n is None:
            status = "MISSING"
        elif exp is None or n >= 0.995 * exp:
            status = "OK"
        else:
            status = "SHORT"
        rows.append({"set": name, "images": n, "expected": exp, "status": status})
    bank = None
    if c.has_textbank():
        bank = c.load_textbank().summary()
    return rows, bank, c.dir


def convert_notebook(src: str, dst: str, setup_cell: str = COLAB_SETUP_CELL) -> bool:
    """Swap a Kaggle oodlab notebook's code-finding cell for the Colab set-up cell."""
    with open(src) as f:
        nb = json.load(f)
    swapped = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if "CODE_ROOT" in text or ("/kaggle/input" in text and "oodlab" in text):
            cell["source"] = setup_cell.splitlines(keepends=True)
            cell["outputs"], cell["execution_count"] = [], None
            swapped = True
            break
    nb.setdefault("metadata", {})["accelerator"] = "GPU"
    nb["metadata"].setdefault("colab", {"provenance": []})
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1)
    return swapped
