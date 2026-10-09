"""Small shared helpers: logging, seeding, timing, atomic saves, environment report."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "oodlab", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Logger that writes to stdout *and* (optionally) to a file, flushing every line.

    Kaggle's committed runs only keep what reaches stdout or files in /kaggle/working,
    so every long step logs through this.
    """
    key = f"{name}|{log_file}"
    if key in _LOGGERS:
        return _LOGGERS[key]
    logger = logging.getLogger(key)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGERS[key] = logger
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timer(msg: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    line = f"{msg}: {dt:.1f}s"
    (logger.info if logger else print)(line)


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def disk_free(path: str) -> str:
    try:
        return human_bytes(shutil.disk_usage(path).free)
    except (FileNotFoundError, OSError):
        return "n/a"


def atomic_write_bytes(path: str, data: bytes) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), prefix=".tmp_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_torch_save(obj: Any, path: str) -> None:
    import torch

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # torch's zip writer derives an archive folder name from the file name and rejects
    # names starting with "." -> use a plain "tmp_*.pt" name
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), prefix="tmp_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_np_savez(path: str, **arrays) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), prefix=".tmp_", suffix=".npz")
    os.close(fd)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_json(obj: Any, path: str) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, default=str).encode())


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def env_report() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        info["torch"] = None
    try:
        import regex  # noqa: F401  (the only extra dependency of the vendored CLIP tokenizer)

        info["regex"] = True
    except ImportError:
        info["regex"] = False
    for p in ["/kaggle/working", "/kaggle/tmp", "/tmp"]:
        if os.path.exists(p):
            info[f"free[{p}]"] = disk_free(p)
    return info


def pick_device(prefer: str = "auto") -> str:
    import torch

    if prefer == "cpu":
        return "cpu"
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if prefer == "cuda":
        raise RuntimeError("CUDA requested but no GPU is available (turn on the GPU accelerator in Kaggle).")
    return "cpu"
