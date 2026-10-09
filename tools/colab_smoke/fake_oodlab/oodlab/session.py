import logging
import os
import sys
import time
from dataclasses import dataclass, field

from .paths import default_dirs


@dataclass
class Context:
    name: str
    dirs: object
    log: object
    device: str
    smoke: bool
    input_roots: list
    out: str
    info: dict = field(default_factory=dict)

    def path(self, fname):
        return os.path.join(self.out, fname)


def ensure_packages(clip=True, gdown=False):
    return None


def start(name, need_gpu=False, device="auto"):
    dirs = default_dirs().make()
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
        log.addHandler(logging.FileHandler(os.path.join(dirs.logs, f"{name}.log")))
    out = os.path.join(dirs.results, name)
    os.makedirs(out, exist_ok=True)
    ctx = Context(name, dirs, log, "cpu", False, [dirs.inputs, dirs.work], out)
    from . import __version__
    ctx.info = {"notebook": name, "oodlab": __version__, "code_hash": "fake", "device": "cpu",
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    return ctx


def finish(ctx, extra=None):
    ctx.log.info(f"finished {ctx.name}: {extra}")
