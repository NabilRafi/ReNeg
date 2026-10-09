import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# the repository (src/oodlab) or the Kaggle dataset layout (oodlab_code/oodlab next to oodlab_code/tests)
for ROOT in (os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src"), os.path.dirname(_HERE)):
    if os.path.isfile(os.path.join(ROOT, "oodlab", "__init__.py")):
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        break

torch.set_num_threads(min(4, os.cpu_count() or 1))


@pytest.fixture(scope="session")
def bank():
    from oodlab.synthetic import make_bank

    return make_bank(C=50, N=3000, D=256, n_neglabel=500)


@pytest.fixture(scope="session")
def stream_batches(bank):
    """2,000 synthetic images (1,200 ID + 800 OOD), shuffled, in batches of 64."""
    from oodlab.synthetic import make_images

    fi, _ = make_images(bank, 1200, "id", seed=1)
    fo, _ = make_images(bank, 800, "ood", seed=2)
    x = torch.cat([fo, fi])
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    return list(x[perm].split(64))


@pytest.fixture(scope="session")
def fake_kaggle(tmp_path_factory):
    """A miniature Kaggle input tree (fake ImageNet + OpenOOD zips)."""
    from oodlab.smoke import make_fake_kaggle

    root = tmp_path_factory.mktemp("kaggle")
    inputs = root / "input"
    make_fake_kaggle(str(inputs))
    return {"root": str(root), "inputs": str(inputs), "scratch": str(root / "scratch"), "work": str(root / "working")}


@pytest.fixture(scope="session")
def tiny_clip():
    from oodlab.clipwrap import load_clip

    return load_clip(device="cpu", random_init=True)
