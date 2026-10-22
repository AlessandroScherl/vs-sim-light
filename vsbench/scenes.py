"""Named scenes for the servo loop, and the poster target pose.

The poster, marker and clutter scenes already live in library code
(``vsbench/stdscenes.py``), so nothing is moved; ``make_scene(name)`` is the one entry
point the episode loop, the checks, the replay and the demo use to build a scene from
its name. ``poster_target_pose`` is the demo target: straight at the poster centre from
the distance where the poster spans ``fill`` of the image width.
"""

from __future__ import annotations

import numpy as np

from vsbench.camera import CameraModel
from vsbench.poses import look_at
from vsbench.scene import SceneSpec
from vsbench.stdscenes import MARKER_POINTS, POSTER_SIZE, STD_SCENES

SCENE_NAMES = tuple(STD_SCENES)
MARKER_DISTANCE = 0.7  # m, all 13 markers inside the image with a wide margin


def make_scene(name: str) -> SceneSpec:
    """The standard scene called ``name`` (``poster``, ``marker``, ``clutter``, ...)."""
    try:
        return STD_SCENES[name]()
    except KeyError:
        raise KeyError(f"unknown scene {name!r}; known: {list(SCENE_NAMES)}") from None


def poster_target_pose(cam: CameraModel, fill: float = 0.7) -> np.ndarray:
    """Camera-to-world pose looking straight down the poster normal at its centre.

    The poster (``POSTER_SIZE`` metres, plane z = 0, upright along +y) spans ``fill`` of
    the image width: distance ``fx * poster_width / (fill * width)``. Image-up is the
    poster's +y, so the picture is upright.
    """
    if not 0.0 < fill <= 1.0:
        raise ValueError("fill must be in (0, 1]")
    d = cam.fx * POSTER_SIZE[0] / (fill * cam.width)
    return look_at((0.0, 0.0, d), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))


def marker_target_pose(cam: CameraModel, distance: float = MARKER_DISTANCE) -> np.ndarray:
    """Straight down at the marker board centre from ``distance`` metres, +y up."""
    if not distance > 0.0:
        raise ValueError("distance must be positive")
    return look_at((0.0, 0.0, distance), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))


def target_pose(scene_name: str, cam: CameraModel) -> np.ndarray:
    """The demo target of a scene: the marker pose for ``marker``, the poster pose otherwise
    (the poster sits at the origin of the poster and clutter scenes)."""
    if scene_name == "marker":
        return marker_target_pose(cam)
    return poster_target_pose(cam)


def world_points(scene_name: str) -> np.ndarray:
    """The known 3D points of a scene for ``true_point_ibvs``: the 13 marker centres."""
    if scene_name == "marker":
        return MARKER_POINTS.copy()
    raise KeyError(f"scene {scene_name!r} has no known point set (only 'marker' does)")
