"""Produce the U3 parity fixtures with the FROZEN OLD CODE. Run once; see
``u3_fixture_provenance.md`` for the exact command and the resulting md5s.

Never imports ``refinevs``. It drives two frozen old matchers on the same
(goal, current) pair U1 and U2 used:

1. ``genvs-sim/src/control/matchers/classical.py::ClassicalMatcher`` for each of
   the three detectors crossed with each of the three RANSAC models, exactly as
   ``src/eval/baseline_check.py::build_controller`` builds a classical arm
   (``ransac = scene_spec.ransac``, ``num_pairs`` = the ``--K`` of the ViT path).
   Recorded per cell: the keypoint counts on both images, the ratio-test match
   count, the pre- and post-RANSAC counts, whether a model was actually fitted,
   and every matched pixel coordinate and confidence.

2. ``genvs-sim/src/control/matchers/vit_dense.py::ViTDenseMatcher(backbone="dinov2")``
   - the published ViT-VS arm (``baseline_check.py`` builds it with ``ransac="none"``
   and the seeded per-episode draw). Its torch.hub weights **are** on this laptop
   (only the HF ``facebook/dinov2-small`` copy is missing), so this is a real parity
   fixture: the full 3456-d log-binned chain, its pool and its 24 drawn landmarks.

3. ``genvs-sim/src/control/matchers/vit_tiled.py::ARTVSMatcher`` for one coarse
   frame and one tiled frame, plus the tile grid. **On DINOv3**, deliberately, not
   on ART-VS's own ``backbone="dinov2"`` default: what the fixture pins is this
   unit's arithmetic - the tile grid, the tile-local -> global mapping with its
   ``(y, x)`` flip, the per-tile budget and the latch - and a patch-**16** backbone
   is the case ``TilingConfig.patch_size = 14`` broke (trap 1). Running it on
   DINOv2 would put 14 on both sides of the fence and prove nothing. The tiled
   frame is forced with the vendored ``switch_iteration=1`` so one run captures
   both phases.

The old ART-VS draws its subsample from torch's global RNG; the vendored
``set_subsample_seed`` hook makes it reproducible, and it is given the new code's
``episode_seed("mug1", 7, 0)`` so both sides draw the same landmarks.

Regenerating a fixture is a decision, not a fix: it means the port changed a
number (``tests/fixtures/README.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zlib
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD_REPO = REPO.parent / "genvs-sim"

GOAL = HERE / "u1_frame_mug1_goal.jpg"
CURRENT = HERE / "u2_frame_mug1_roll010.jpg"
SCENE, START_ID, SEED = "mug1", 7, 0
NUM_PAIRS = 24

METHODS = ("sift", "orb", "akaze")
RANSAC_MODELS = ("none", "homography", "fundamental")


def episode_subsample_seed(scene: str, start_id, base: int) -> int:
    """``baseline_check.episode_subsample_seed`` - retyped so this file imports nothing."""
    return (int(base) + zlib.crc32(f"{scene}/{start_id}".encode())) % (2**31)


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def classical_cells(goal: np.ndarray, current: np.ndarray) -> dict:
    import cv2
    from src.control.matchers.classical import ClassicalMatcher

    cells = {}
    for method in METHODS:
        for model in RANSAC_MODELS:
            # OpenCV's SIFT and AKAZE are deterministic and its RANSAC seeds its own
            # generator per call (measured: four repeats identical with and without
            # this line, both environments). Seeded anyway, so the fixture does not
            # depend on that staying true.
            cv2.setRNGSeed(0)
            matcher = ClassicalMatcher(method=method, num_pairs=NUM_PAIRS,
                                       ransac=model, device="cpu")
            matcher.reset(goal)
            corr = matcher.match(current)
            extractor = matcher._extractor
            cells[f"{method}/{model}"] = {
                "n_keypoints_goal": len(extractor.goal_keypoints),
                "n_keypoints_cur": len(extractor.current_keypoints),
                "n_raw": int(corr.debug["n_raw"]),
                "n_inliers": int(corr.debug["n_inliers"]),
                "ransac_applied": bool(corr.debug["ransac_applied"]),
                "n": int(len(corr.pts_goal)),
                "pts_goal": np.asarray(corr.pts_goal, np.float64).tolist(),
                "pts_cur": np.asarray(corr.pts_cur, np.float64).tolist(),
                "conf": np.asarray(corr.conf, np.float64).tolist(),
            }
    return cells


def artvs_cells(goal: np.ndarray, current: np.ndarray) -> dict:
    from src.control.matchers.vit_tiled import ARTVSMatcher

    seed = episode_subsample_seed(SCENE, START_ID, SEED)
    out: dict = {"subsample_seed": seed}
    for tag, kwargs in (("coarse", {}), ("tiled", {"switch_iteration": 1})):
        matcher = ARTVSMatcher(backbone="dinov3", num_pairs=NUM_PAIRS, device="cpu", **kwargs)
        matcher.set_subsample_seed(seed)
        matcher.reset(goal)
        corr = matcher.match(current)
        out[tag] = {
            "mode": corr.debug.get("mode"),
            "n": int(len(corr.pts_goal)),
            "pts_goal": np.asarray(corr.pts_goal, np.float64).tolist(),
            "pts_cur": np.asarray(corr.pts_cur, np.float64).tolist(),
            "conf": np.asarray(corr.conf, np.float64).tolist(),
        }
        if tag == "coarse":
            cfg = matcher._tcfg
            out["tiling"] = {
                "slice_number": cfg.slice_number, "grid_size": cfg.grid_size,
                "input_resolution": cfg.input_resolution,
                "slice_resolution": cfg.slice_resolution,
                "vit_input_size": cfg.vit_input_size, "patch_size": cfg.patch_size,
                "patches_per_vit_input": cfg.patches_per_vit_input,
                "effective_patches": cfg.effective_patches,
                "backbone_input_size": matcher._input_size,
                "backbone_patch_size": matcher._patch_size,
            }
            out["tile_coords"] = [[int(y), int(x)] for y, x in matcher._tile_coords]
        else:
            out[tag]["per_tile_pairs"] = int(corr.debug["per_tile_pairs"])
            out[tag]["n_tiles"] = int(corr.debug["n_tiles"])
    return out


def vitvs_dinov2_cell(goal: np.ndarray, current: np.ndarray) -> dict:
    from src.control.matchers.vit_dense import ViTDenseMatcher

    seed = episode_subsample_seed(SCENE, START_ID, SEED)
    matcher = ViTDenseMatcher(backbone="dinov2", num_pairs=NUM_PAIRS, device="cpu",
                              ransac="none")
    matcher.set_subsample_seed(seed)
    matcher.reset(goal)
    corr = matcher.match(current)
    goal_desc = matcher._ref_descriptors
    return {
        "subsample_seed": seed,
        "descriptor_shape": [int(v) for v in goal_desc.shape],
        "goal_descriptor_sum": float(goal_desc.double().sum()),
        "goal_descriptor_absmax": float(goal_desc.abs().max()),
        "num_patches_side": int(corr.debug["num_patches_side"]),
        "input_size": int(corr.debug["input_size"]),
        "patch_size": int(corr.debug["patch_size"]),
        "n": int(len(corr.pts_goal)),
        "pts_goal": np.asarray(corr.pts_goal, np.float64).tolist(),
        "pts_cur": np.asarray(corr.pts_cur, np.float64).tolist(),
        "conf": np.asarray(corr.conf, np.float64).tolist(),
    }


def _write(path: Path, payload: dict) -> None:
    """One compact line per fixture: these are data, not files anyone reads by eye
    (`u3_fixture_provenance.md` is what a human reads), and pretty-printing 24 pairs
    of float64 coordinates nine times over costs 3,000 lines of diff for nothing."""
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-repo", type=Path, default=OLD_REPO)
    ap.add_argument("--skip-vit", action="store_true",
                    help="classical only (no backbone weights needed)")
    args = ap.parse_args()

    sys.path.insert(0, str(args.old_repo))
    goal, current = _rgb(GOAL), _rgb(CURRENT)
    git = subprocess.run(["git", "-C", str(args.old_repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()

    import cv2

    common = {
        "made": date.today().isoformat(),
        "old_repo_commit": git,
        "goal_image": GOAL.name, "current_image": CURRENT.name,
        "num_pairs": NUM_PAIRS,
        "cv2_version": cv2.__version__,
        "python": sys.version.split()[0],
    }

    _write(HERE / "u3_classical_mug1.json", {**common, "cells": classical_cells(goal, current)})

    if not args.skip_vit:
        import torch

        episode = {"scene": SCENE, "start_id": START_ID, "seed": SEED,
                   "torch": torch.__version__}
        _write(HERE / "u3_artvs_dinov3.json",
               {**common, **episode, "backbone": "dinov3", **artvs_cells(goal, current)})
        _write(HERE / "u3_vitvs_dinov2.json",
               {**common, **episode, "backbone": "dinov2",
                **vitvs_dinov2_cell(goal, current)})

    for name in ("u3_classical_mug1.json", "u3_artvs_dinov3.json", "u3_vitvs_dinov2.json"):
        path = HERE / name
        if path.is_file():
            digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - a pin, not a secret
            print(f"{name}  {path.stat().st_size / 1024:.1f} KB  md5 {digest}")


if __name__ == "__main__":
    main()
