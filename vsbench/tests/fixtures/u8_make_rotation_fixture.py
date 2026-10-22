"""Build ``u8_rotation_parity.json`` - U8's stored parity fixture (review R19 item 7).

Run once, on a machine that has the frozen old repository checked out at
``../genvs-sim`` @ ``legacy-pre-cleanup`` and the DINOv3 weights in the local cache.
Everything it stores comes from the **frozen** code or the real backbone, so the
committed file is what U8 is proven equal to on a machine that has neither:

- ``unwarp_points``: 64 random points x 7 angles x both call shapes, from
  ``genvs-sim/src/control/matchers/rot_aware.py`` ``unwarp_points``;
- ``rotate_hypothesis``: md5 of the rotated bytes at 5 angles on a fixed synthetic
  image, from ``genvs-sim/src/control/prealign.py``;
- the frozen ``RollPrealigner.pick`` record on the synthetic goal rolled 90 degrees
  (seed 2026, ``tie_order: legacy``), driven through the **vendored**
  ``find_correspondences_batch``;
- the real-image bank counts on the U1 fixture at 50 and 60 degrees.

    cd refine-vs && HF_HUB_OFFLINE=1 python tests/fixtures/u8_make_rotation_fixture.py

Never re-run to "fix" a failing test: a changed number is the finding.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = REPO.parent / "genvs-sim"
OUT = HERE / "u8_rotation_parity.json"
sys.path.insert(0, str(REPO))

from refinevs import rotation as rot  # noqa: E402
from refinevs.match import DenseMatcher  # noqa: E402

ANGLES = (0.0, 7.5, 30.0, 90.0, 180.0, -137.25, 359.9)
ROTATE_ANGLES = (0.0, 90.0, 180.0, 270.0, 17.0)
FRAME_WH = (960, 540)


def _legacy(mod: str):
    for p in (str(OLD), str(OLD / "src/control/_vitvs_vendor"),
              str(OLD / "src/control/_vitvs_vendor/visual_servoing")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "rospy" not in sys.modules:
        sys.modules["rospy"] = importlib.import_module("src.control._rospy_stub")
    return importlib.import_module(mod)


def _synthetic_goal(seed: int = 8) -> np.ndarray:
    """Byte-identical to ``tests/fast/test_rotation.py::_synthetic_goal``."""
    rng = np.random.default_rng(seed)
    w, h = FRAME_WH
    blocks = rng.integers(0, 256, size=(h // 5, w // 5, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.repeat(np.repeat(blocks, 5, axis=0), 5, axis=1))


def main() -> None:
    if not OLD.is_dir():
        raise SystemExit(f"the frozen old repo is not at {OLD}; this maker needs it")
    prealign = _legacy("src.control.prealign")
    rot_aware = _legacy("src.control.matchers.rot_aware")
    sys.path.insert(0, str(HERE))
    tests = importlib.import_module("tests.fast.test_rotation")

    rng = np.random.default_rng(0)
    pts = rng.uniform(-50, 2000, size=(64, 2))
    doc: dict = {
        "_what": "U8 parity, from genvs-sim@legacy-pre-cleanup and the real DINOv3",
        "unwarp_points": {
            "pts": pts.tolist(),
            "image_wh": [1920, 1080],
            "raw_wh": [1920, 1080],
            "squashed_wh": [512, 512],
            "angles_deg": list(ANGLES),
            "plain": {f"{a:g}": rot_aware.unwarp_points(pts, a, (1920, 1080)).tolist()
                      for a in ANGLES},
            "raw": {f"{a:g}": rot_aware.unwarp_points(pts, a, (512, 512),
                                                      raw_wh=(1920, 1080)).tolist()
                    for a in ANGLES},
        },
    }

    goal = _synthetic_goal()
    doc["rotate_hypothesis"] = {
        "frame_md5": hashlib.md5(goal.tobytes()).hexdigest(),
        "frame_shape": list(goal.shape),
        "md5": {f"{a:g}": hashlib.md5(prealign.rotate_hypothesis(goal, a).tobytes())
                .hexdigest() for a in ROTATE_ANGLES},
    }

    fake = tests._FakeViTDenseMatcher()
    doc["prealign_pick"] = {
        "seed": 2026, "tie_order": "legacy", "current": "bank_frame(goal, 90)",
        "record": prealign.RollPrealigner(fake).pick(
            goal, rot.bank_frame(goal, 90.0), seed=2026),
    }

    from PIL import Image
    from refinevs.features import build_backbone

    frame = np.asarray(Image.open(HERE / "u1_frame_mug1_goal.jpg").convert("RGB"))
    bb = build_backbone("dinov3_vits16", device="cpu")
    counts = {}
    for roll in (50.0, 60.0):
        inner = DenseMatcher(num_landmarks=24, seed=0, backbone=bb)
        inner.set_episode("u1_mug1", f"roll{int(roll):03d}")
        banked = rot.BankedMatcher(inner, angles=rot.DEFAULT_BANK_ANGLES)
        banked.reset(frame)
        rec = banked.match(rot.bank_frame(frame, roll)).extras["rotation"]
        counts[f"{roll:g}"] = {"n_matches_per_bank": rec["n_matches_per_bank"],
                               "picked_deg": rec["bank_angle_deg"]}
    doc["bank_counts_u1_fixture"] = {
        "frame": "u1_frame_mug1_goal.jpg", "backbone": "dinov3_vits16",
        "angles_deg": list(rot.DEFAULT_BANK_ANGLES),
        "valid_per_bank": [int(b.n_valid) for b in banked.banks.banks],
        "by_roll": counts,
    }
    doc["torch"] = torch.__version__
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
