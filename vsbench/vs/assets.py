# vs-sim copy: vssim/assets.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Asset libraries and USD path resolution - pure python, no Isaac.

Purpose
-------
One library names every curated object a scene can place (``assets/``), the YCB
scans live on the station volume, and one function turns a library entry into a
path a USD stage can reference. All of it is pure python so a scene can be
checked on a laptop before a GPU boots.

Fences
------
- **Loud refusal beats a degraded scene.** A missing YCB library raises; a
  cluttered build must fail before the first frame.
- **Repo-relative ``usd_path`` values resolve against the asset root, not this
  repo's top level.** :func:`asset_root` reads ``$VSSIM_ASSET_ROOT`` (default
  ``assets/usd/`` here, git-ignored) and the Isaac scene refuses a path that does
  not exist, because a missing reference makes an empty prim and therefore a wrong
  goal distance, silently.

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/data/assets.py``);
the supplementary GSO library and the texture-pool lookup that lived beside these
functions belong to that project's dataset generation and are not here.

Golden tests: ``tests/fast/test_assets.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vsbench.vs.camera import REPO_ROOT

DEFAULT_LIBRARY = REPO_ROOT / "assets" / "asset_library_v2.json"

#: Where the USD trees live. 317 of the 323 library entries are **repo-relative**
#: (``experiments/ingest/usd/...``, ``assets/gso_usd/...``): the trees are heavy and
#: are not in git, so the root is a setting - ``$VSSIM_ASSET_ROOT`` - defaulting to
#: ``assets/usd/`` inside this repository (git-ignored), the place a standalone
#: checkout stages them. On the 4090 station they sit in the old ``genvs-sim``
#: checkout and the runbooks set ``VSSIM_ASSET_ROOT=/workspace/genvs-sim``.
#: :meth:`vsbench.vs.isaac_scene.IsaacScene.place_objects` checks the file exists before
#: referencing it - a missing USD makes an empty prim, a garbage bounding box and a
#: wrong goal distance, in silence.
ASSET_ROOT_ENV = "VSSIM_ASSET_ROOT"
DEFAULT_ASSET_ROOT = REPO_ROOT / "assets" / "usd"

#: The external YCB scans (``ycb/<object>`` ids; the two cluttered rooms use them)
#: live on the station volume, never in git: ``$VSSIM_YCB_ROOT`` overrides.
YCB_ROOT_ENV = "VSSIM_YCB_ROOT"
DEFAULT_YCB_ROOT = "/workspace/assets/ycb"

#: Prefixes marking a Nucleus / Isaac-builtin asset: a suffix to append to
#: ``get_assets_root_path()`` at render time, never a local file.
NUCLEUS_PREFIXES = ("/Isaac", "/NVIDIA")

#: Longest-extent rescue targets in metres. Unit-free sources are rescaled to
#: these; metric scans (YCB, any id containing ``/``) are placed at native scale.
RESCUE_TARGETS_M = {
    "mug": 0.10, "drinking_glass": 0.14, "tube": 0.15, "can": 0.10,
    "hairbrush": 0.20, "stapler": 0.18, "tv_remote": 0.18, "toy_car": 0.15,
    "teapot": 0.20, "shoe": 0.25, "drill": 0.25, "bottle": 0.22,
    "wine_bottle": 0.30, "hammer": 0.30,
}
FORCE_UNIFORM_HEIGHT_M = {"drinking_glass": 0.14, "wine_bottle": 0.30}
FORCE_UNIFORM_SIZE_M = {"stapler": 0.18}
DEFAULT_LONGEST_M = 0.20


def target_for(category: str) -> tuple[str, float]:
    """``(mode, target_m)``: ``'height'`` scales the post-orient Z extent,
    ``'longest'`` the maximum extent."""
    if category in FORCE_UNIFORM_HEIGHT_M:
        return ("height", FORCE_UNIFORM_HEIGHT_M[category])
    if category in FORCE_UNIFORM_SIZE_M:
        return ("longest", FORCE_UNIFORM_SIZE_M[category])
    return ("longest", RESCUE_TARGETS_M.get(category, DEFAULT_LONGEST_M))


# ------------------------------------------------------------------ libraries
def load_library(path: Path | str | None = None) -> dict:
    """``{instance_id: entry}`` from an asset-library json."""
    lib = json.loads(Path(path or DEFAULT_LIBRARY).read_text())
    assets = lib["assets"] if isinstance(lib, dict) and "assets" in lib else lib
    return {a["instance_id"]: a for a in assets}


def ycb_root(override: Path | str | None = None) -> Path:
    """The YCB scans' directory: ``$VSSIM_YCB_ROOT``, else :data:`DEFAULT_YCB_ROOT`."""
    return Path(override or os.environ.get(YCB_ROOT_ENV, DEFAULT_YCB_ROOT))


def load_ycb_library(root: Path | str | None = None) -> dict:
    """``{instance_id: entry}`` for the external YCB assets (station volume)."""
    p = ycb_root(root) / "ycb_library.json"
    if not p.exists():
        raise FileNotFoundError(
            f"YCB library not found: {p} - run the fetch + convert scripts first; "
            f"a cluttered build must fail here, not render a degraded scene")
    return {a["instance_id"]: a for a in json.loads(p.read_text())["assets"]}


def merge_ycb_library(library: dict, asset_ids, root: Path | str | None = None) -> int:
    """Merge the YCB library into ``library`` **iff** an id needs it.

    The merged library is keyed ``ycb/<oid>`` (curated ids never contain ``/``),
    which is what the per-category metric-scale rule keys on.
    """
    if not any(str(a).startswith("ycb/") for a in asset_ids):
        return 0
    merged = 0
    for iid, e in load_ycb_library(root).items():
        if iid not in library:
            library[iid] = e
            merged += 1
    return merged


# -------------------------------------------------------------- path resolver
def asset_root(override: Path | str | None = None) -> Path:
    """The checkout that holds the USD trees: ``$VSSIM_ASSET_ROOT``, else
    :data:`DEFAULT_ASSET_ROOT`. Never this repository unless it is set to it."""
    return Path(override or os.environ.get(ASSET_ROOT_ENV, DEFAULT_ASSET_ROOT))


def is_nucleus_entry(entry: dict) -> bool:
    """True iff the entry names a Nucleus / Isaac-builtin asset."""
    up = entry.get("usd_path") or entry.get("asset_path") or ""
    return entry.get("source") == "isaac_native" or up.startswith(NUCLEUS_PREFIXES)


def resolve_usd_path(entry: dict, root: Path | str | None = None) -> tuple[str, bool]:
    """``(path, is_nucleus)`` for a library entry.

    Nucleus entries return the **raw** ``/Isaac/...`` suffix for the caller to
    rebase; absolute paths come back verbatim (back-compat with the frozen v1
    library); repo-relative paths are absolutised against :func:`asset_root`,
    which is ``$VSSIM_ASSET_ROOT`` unless ``root`` says otherwise - **not** this
    repository, which holds no USD trees.
    """
    up = entry.get("usd_path") or entry.get("asset_path")
    if not up:
        raise ValueError(
            f"library entry has no usd_path/asset_path: "
            f"{entry.get('instance_id', entry)!r}")
    if is_nucleus_entry(entry):
        return (up, True)
    if os.path.isabs(up):
        return (up, False)
    return (str(asset_root(root) / up), False)


def resolve_isaac_usd_path(entry: dict, assets_root: str,
                           root: Path | str | None = None) -> str:
    """In-Isaac convenience: one path ready for ``add_reference_to_stage``."""
    path, nucleus = resolve_usd_path(entry, root)
    return assets_root + path if nucleus else path
