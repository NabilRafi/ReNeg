import json
import os

from .. import RESOURCES


def official_wnids():
    with open(os.path.join(RESOURCES, "imagenet_wnids.json")) as f:
        return json.load(f)
