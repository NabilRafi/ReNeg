"""oodlab: a small, testable toolkit for test-time negative-label OOD detection.

Layout
------
- ``oodlab.data``        : locating ImageNet on Kaggle, downloading OpenOOD sets, manifests
- ``oodlab.clipwrap``    : loading CLIP and encoding images/text exactly like OpenOOD-VLM
- ``oodlab.textbank``    : ID prompts, the NegLabel WordNet corpus, NegLabel mining, noise images
- ``oodlab.cache``       : the on-disk CLIP feature cache (encode once, replay many times)
- ``oodlab.stream``      : test streams (shuffled ID+OOD, temporal-shift sequences)
- ``oodlab.methods``     : MCM, NegLabel, TANL, AdaNeg as stream methods on cached features
- ``oodlab.metrics``     : OpenOOD metrics (exact) plus the ID-positive FPR95 convention
- ``oodlab.runner``      : benchmark protocols (Four-OOD, OpenOOD v1.5, temporal shift)
- ``oodlab.diagnostics`` : churn / survival / queue-purity diagnostics for TANL
- ``oodlab.official``    : vendored official code, used only to prove our code matches it
- ``oodlab.session``     : notebook set-up (paths, logging, packages, code snapshot)
"""
import json as _json
import os as _os

__version__ = "0.2.1"

RESOURCES = _os.path.join(_os.path.dirname(__file__), "resources")
CORPUS_DIR = _os.path.join(RESOURCES, "neglabel_txtfiles")   # NegLabel's WordNet word lists (Apache-2.0)


def imagenet_classnames():
    """The 1,000 class names OpenOOD-VLM uses for ImageNet (``get_class_names('imagenet')``)."""
    with open(_os.path.join(RESOURCES, "imagenet_classes.json")) as f:
        names = _json.load(f)
    assert len(names) == 1000
    return names
