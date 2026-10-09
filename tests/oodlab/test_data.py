import os

import numpy as np
import pytest

from oodlab.data import DEFAULT_BUILD, Manifest, build_manifest
from oodlab.data.imagenet import locate_imagenet
from oodlab.data.openood import obtain_imglist_dir, read_imglist, resolve_list_paths
from oodlab.paths import find_file
from oodlab.smoke import SMOKE_SIZES


def test_locate_imagenet(fake_kaggle):
    loc = locate_imagenet([fake_kaggle["inputs"]])
    assert loc.layout == "flat"
    assert loc.val_solution is not None


def test_all_manifests(fake_kaggle, tmp_path):
    inputs, scratch = fake_kaggle["inputs"], fake_kaggle["scratch"]
    imgl = obtain_imglist_dir(scratch, [inputs], allow_download=False)
    cache = {}
    got = {}
    for name in DEFAULT_BUILD:
        m = build_manifest(name, [inputs], scratch, imgl, allow_download=False, strict_counts=False, _imagenet_cache=cache)
        assert m.missing_files(sample=None) == [], name
        assert len(set(m.keys)) == len(m), f"{name}: duplicate keys"
        got[name] = m
    n_val = SMOKE_SIZES["val"]
    assert len(got["imagenet_val_all"]) == n_val
    assert len(got["imagenet_test"]) + len(got["imagenet_val"]) == n_val
    assert not set(got["imagenet_test"].keys) & set(got["imagenet_val"].keys)
    assert len(got["ninco"]) == SMOKE_SIZES["ninco"]           # resolved through the basename fallback
    assert len(got["textures"]) == SMOKE_SIZES["texture"] // 2  # list subset
    assert len(got["textures_all"]) == SMOKE_SIZES["texture"]
    assert (got["imagenet_val_all"].labels >= 0).all() and (got["ssb_hard"].labels == -1).all()
    # labels of the ID lists agree with Kaggle's solution file (checked inside), round trip:
    got["imagenet_test"].save(str(tmp_path))
    back = Manifest.load(str(tmp_path), "imagenet_test")
    assert back.keys == got["imagenet_test"].keys and np.array_equal(back.labels, got["imagenet_test"].labels)


def test_resolver_suffix_disambiguation(tmp_path):
    for sub in ["a/x", "b/x"]:
        os.makedirs(tmp_path / "ds" / sub, exist_ok=True)
        (tmp_path / "ds" / sub / "img.jpg").write_bytes(b"0")
    paths, missing = resolve_list_paths([("zzz/b/x/img.jpg", -1)], str(tmp_path / "ds"))
    assert not missing and paths[0].endswith("b/x/img.jpg")


def test_find_file_is_bounded(tmp_path):
    deep = tmp_path / "a" / "train" / "b"
    deep.mkdir(parents=True)
    (deep / "target.txt").write_text("x")
    assert find_file("target.txt", [str(tmp_path)]) is None           # never descends into train/
    (tmp_path / "a" / "target.txt").write_text("x")
    assert find_file("target.txt", [str(tmp_path)]).endswith("a/target.txt")


def test_imglist_reader(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("a/b.jpg -1\nc/d.JPEG 7\n\n")
    assert read_imglist(str(p)) == [("a/b.jpg", -1), ("c/d.JPEG", 7)]


def test_list_required_sets_never_fall_back(fake_kaggle):
    """Sets defined by an OpenOOD list must fail loudly without the list (no silent 'all images')."""
    with pytest.raises(FileNotFoundError, match="needs the OpenOOD list"):
        build_manifest("textures", [fake_kaggle["inputs"]], fake_kaggle["scratch"], None, allow_download=False,
                       strict_counts=False, _imagenet_cache={})
    m = build_manifest("sun", [fake_kaggle["inputs"]], fake_kaggle["scratch"], None, allow_download=False,
                       strict_counts=False, _imagenet_cache={})
    assert len(m) == SMOKE_SIZES["sun"]
