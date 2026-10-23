# vs-sim copy: tests/fast/test_assets.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The asset library, USD path resolution and the scale table.

Guards `vssim/assets.py`. Pure python: no Isaac, no volume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vsbench.vs.assets import (
    ASSET_ROOT_ENV,
    DEFAULT_ASSET_ROOT,
    DEFAULT_LIBRARY,
    DEFAULT_LONGEST_M,
    DEFAULT_YCB_ROOT,
    NUCLEUS_PREFIXES,
    RESCUE_TARGETS_M,
    YCB_ROOT_ENV,
    asset_root,
    is_nucleus_entry,
    load_library,
    load_ycb_library,
    merge_ycb_library,
    resolve_isaac_usd_path,
    resolve_usd_path,
    target_for,
    ycb_root,
)
from vsbench.vs.camera import REPO_ROOT
from vsbench.vs.scenes import EVAL_SCENES, load_scenes


def test_the_library_is_in_this_repo_and_loads():
    assert DEFAULT_LIBRARY.is_file()
    lib = load_library()
    assert len(lib) > 100
    assert all(k == v["instance_id"] for k, v in lib.items())
    assert all("usd_path" in v or "asset_path" in v for v in lib.values())


def test_every_curated_object_of_every_scene_is_in_the_library():
    """A scene's non-YCB objects must resolve through the committed library, or the
    build would fail on the station and nowhere earlier."""
    lib = load_library()
    scenes, _ = load_scenes()
    for name, spec in scenes.items():
        for iid in spec.object_ids():
            if iid.startswith("ycb/"):
                continue
            assert iid in lib, f"{name}: {iid} is not in assets/asset_library_v2.json"
    assert "handal_drill_25" in lib                 # the drill room's one object
    assert all(scenes[n].kind in ("poster", "multi_object") for n in EVAL_SCENES)


def test_the_ycb_merge_only_runs_when_an_id_needs_it(tmp_path):
    lib = {}
    assert merge_ycb_library(lib, ["mug/a", "texture/t"]) == 0 and lib == {}
    root = tmp_path / "ycb"
    root.mkdir()
    (root / "ycb_library.json").write_text(json.dumps(
        {"assets": [{"instance_id": "ycb/003_cracker_box", "usd_path": "/x/a.usd"}]}))
    assert merge_ycb_library(lib, ["ycb/003_cracker_box"], root=root) == 1
    assert "ycb/003_cracker_box" in lib             # the namespaced key is kept


def test_a_missing_ycb_library_is_a_loud_refusal(tmp_path):
    """A cluttered build must fail before the first frame, not render degraded."""
    with pytest.raises(FileNotFoundError, match="YCB library not found"):
        load_ycb_library(tmp_path)
    with pytest.raises(FileNotFoundError):
        merge_ycb_library({}, ["ycb/anything"], root=tmp_path)


def test_the_ycb_root_is_an_environment_setting(monkeypatch, tmp_path):
    assert ycb_root() == Path(DEFAULT_YCB_ROOT)
    monkeypatch.setenv(YCB_ROOT_ENV, str(tmp_path))
    assert ycb_root() == tmp_path


# -------------------------------------------------------------- path resolver
def test_the_three_resolution_branches():
    nucleus = {"instance_id": "n", "usd_path": "/Isaac/Props/thing.usd"}
    absolute = {"instance_id": "a", "usd_path": "/workspace/assets/x.usd"}
    relative = {"instance_id": "r", "usd_path": "assets/gso_usd/y.usd"}
    assert resolve_usd_path(nucleus) == ("/Isaac/Props/thing.usd", True)
    assert resolve_usd_path(absolute) == ("/workspace/assets/x.usd", False)
    assert resolve_usd_path(relative) == (
        str(DEFAULT_ASSET_ROOT / "assets/gso_usd/y.usd"), False)
    assert resolve_isaac_usd_path(nucleus, "omniverse://host") == (
        "omniverse://host/Isaac/Props/thing.usd")
    assert resolve_isaac_usd_path(absolute, "omniverse://host") == "/workspace/assets/x.usd"


def test_source_isaac_native_marks_a_nucleus_entry():
    assert is_nucleus_entry({"source": "isaac_native", "usd_path": "/anything.usd"})
    assert not is_nucleus_entry({"source": "gso", "usd_path": "/local/x.usd"})
    assert all(is_nucleus_entry({"usd_path": p + "/x.usd"}) for p in NUCLEUS_PREFIXES)
    with pytest.raises(ValueError, match="no usd_path"):
        resolve_usd_path({"instance_id": "empty"})
    assert resolve_usd_path({"asset_path": "/legacy/x.usd"}) == ("/legacy/x.usd", False)


def test_relative_paths_resolve_against_the_asset_root_not_this_repo(monkeypatch,
                                                                    tmp_path):
    """317 of 323 entries are relative to the asset root (`assets/usd/` by default,
    git-ignored); this repository holds the library JSON and no USD."""
    entry = {"instance_id": "r", "usd_path": "assets/gso_usd/y.usd"}
    assert asset_root() == DEFAULT_ASSET_ROOT
    assert DEFAULT_ASSET_ROOT != REPO_ROOT
    monkeypatch.setenv(ASSET_ROOT_ENV, str(tmp_path))
    assert asset_root() == tmp_path
    assert resolve_usd_path(entry) == (str(tmp_path / "assets/gso_usd/y.usd"), False)
    assert resolve_usd_path(entry, root=REPO_ROOT)[0].startswith(str(REPO_ROOT))


def test_most_of_the_library_is_repo_relative_and_none_of_it_is_here():
    """How many entries need the asset root at all."""
    lib = load_library()
    rel = [e for e in lib.values()
           if not is_nucleus_entry(e)
           and not str(e.get("usd_path") or e.get("asset_path")).startswith("/")]
    assert len(rel) > 300
    # and not one of them resolves inside this repository
    assert not any((REPO_ROOT / (e.get("usd_path") or "")).exists() for e in rel)


# ------------------------------------------------------------- the scale table
def test_the_scale_table_is_the_frozen_one():
    assert target_for("mug") == ("longest", 0.10)
    assert target_for("drill") == ("longest", 0.25)
    assert target_for("hammer") == ("longest", 0.30)
    assert target_for("drinking_glass") == ("height", 0.14)   # force-uniform height
    assert target_for("stapler") == ("longest", 0.18)         # force-uniform size
    assert target_for("never_seen") == ("longest", DEFAULT_LONGEST_M)
    assert RESCUE_TARGETS_M["shoe"] == 0.25
