"""Notebook set-up shared by every Kaggle notebook.

    from oodlab.session import start
    ctx = start("02_reproduce_tanl", need_gpu=False)
    ctx.log.info("hello")          # printed *and* written to /kaggle/working/logs/02_reproduce_tanl.log

Smoke mode (``OODLAB_SMOKE=1``) swaps in a miniature fake Kaggle input tree and a random
tiny CLIP so the notebooks can be executed end-to-end without data or a GPU. It exists
to test the notebooks; never use it for results.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .paths import Dirs, default_dirs, on_kaggle
from .utils import env_report, get_logger, pick_device, write_json


def smoke_mode() -> bool:
    return os.environ.get("OODLAB_SMOKE", "0") not in ("", "0", "false", "False")


@dataclass
class Context:
    name: str
    dirs: Dirs
    log: object
    device: str
    smoke: bool
    input_roots: List[str]
    out: str                       # this notebook's result folder (inside /kaggle/working/results)
    started: float = field(default_factory=time.time)
    info: Dict = field(default_factory=dict)

    def path(self, *parts: str) -> str:
        p = os.path.join(self.out, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def elapsed(self) -> str:
        return f"{(time.time() - self.started) / 60:.1f} min"


def code_hash() -> str:
    """Short hash of every .py file of the package (logged so results can be traced to code)."""
    root = os.path.dirname(__file__)
    h = hashlib.sha1()
    for dp, dn, fns in os.walk(root):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith(".py"):
                with open(os.path.join(dp, fn), "rb") as f:
                    h.update(fn.encode())
                    h.update(f.read())
    return h.hexdigest()[:12]


def snapshot_code(dst: str) -> str:
    """Copy the package (without the large resources) next to the outputs of this run."""
    src = os.path.dirname(__file__)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "neglabel_txtfiles", "*.pyc"))
    return dst


def pip_install(*pkgs: str, quiet: bool = True) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *(["-q"] if quiet else []), *pkgs]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(
            f"pip install {' '.join(pkgs)} failed. On Kaggle: Settings -> Internet -> On (needs a phone-verified "
            "account), then run the cell again.")


def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def ensure_packages(clip: bool = True, gdown: bool = False) -> None:
    """Make sure the few extra packages exist. CLIP itself ships inside oodlab (``oodlab/_vendor/clip``),
    so nothing is installed from GitHub. Missing optional packages only produce a warning."""
    if clip:
        if not _importable("regex"):              # required by CLIP's tokenizer
            pip_install("regex")
        if not _importable("ftfy"):               # optional (text clean-up; our prompts are plain ASCII)
            try:
                pip_install("ftfy")
            except RuntimeError:
                print("note: ftfy not installed - fine, CLIP's tokenizer works without it for these prompts")
    if gdown and not _importable("gdown"):
        try:
            pip_install("gdown")
        except RuntimeError:
            print("WARNING: gdown could not be installed (Internet off?). Downloads from Google Drive will fail; "
                  "attach the OpenOOD zips as a dataset instead (GUIDE.md section 5).")


def start(name: str, need_gpu: bool = False, device: str = "auto") -> Context:
    smoke = smoke_mode()
    dirs = default_dirs().make()
    log = get_logger(name, os.path.join(dirs.logs, f"{name}.log"))
    dev = pick_device("cpu" if smoke else device)
    if need_gpu and dev != "cuda" and not smoke:
        raise RuntimeError("This notebook needs a GPU: Settings -> Accelerator -> GPU T4 x2 (or P100).")
    if smoke:
        from .smoke import make_fake_kaggle

        make_fake_kaggle(dirs.inputs)
    input_roots = [dirs.inputs, dirs.work] if os.path.isdir(dirs.inputs) else [dirs.work]
    out = os.path.join(dirs.results, name)
    os.makedirs(out, exist_ok=True)
    ctx = Context(name=name, dirs=dirs, log=log, device=dev, smoke=smoke, input_roots=input_roots, out=out)
    from . import __version__

    ctx.info = {"notebook": name, "oodlab": __version__, "code_hash": code_hash(), "device": dev,
                "smoke": smoke, "on_kaggle": on_kaggle(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                **env_report()}
    for k, v in ctx.info.items():
        log.info(f"{k:>14}: {v}")
    if smoke:
        log.warning("SMOKE MODE: fake data + random tiny CLIP. Numbers are meaningless.")
    write_json(ctx.info, os.path.join(out, "run_info.json"))
    snapshot_code(os.path.join(out, "code_snapshot"))
    return ctx


def finish(ctx: Context, extra: Optional[Dict] = None) -> None:
    info = dict(ctx.info)
    info.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed": ctx.elapsed(), **(extra or {})})
    write_json(info, os.path.join(ctx.out, "run_info.json"))
    ctx.log.info(f"done in {ctx.elapsed()} - outputs in {ctx.out}")
