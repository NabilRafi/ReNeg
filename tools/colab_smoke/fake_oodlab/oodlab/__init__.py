"""FAKE oodlab for local notebook smoke tests only. Same public names as the real package."""
import json as _json
import os as _os

__version__ = "0.2.1-fake"
RESOURCES = _os.path.join(_os.path.dirname(__file__), "resources")


def imagenet_classnames():
    with open(_os.path.join(RESOURCES, "imagenet_classes.json")) as f:
        return _json.load(f)
