import os
from dataclasses import dataclass


@dataclass
class Dirs:
    work: str
    scratch: str
    inputs: str

    @property
    def cache(self):
        return os.path.join(self.work, "cache")

    @property
    def logs(self):
        return os.path.join(self.work, "logs")

    @property
    def results(self):
        return os.path.join(self.work, "results")

    def make(self):
        for d in [self.work, self.scratch, self.cache, self.logs, self.results]:
            os.makedirs(d, exist_ok=True)
        return self


def default_dirs():
    base = os.environ.get("OODLAB_HOME", os.path.abspath("oodlab_work"))
    return Dirs(os.path.join(base, "working"), os.path.join(base, "scratch"), os.path.join(base, "input"))
