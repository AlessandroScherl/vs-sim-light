# vs-sim copy: vssim/camera.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The single source of camera intrinsics.

Purpose
-------
``configs/camera.yaml`` is the one place the camera lives: an Intel RealSense D435i
RGB sensor modelled as a 1920x1080 pinhole with a 42 deg vertical field of view and
a square 20.955 mm aperture. Everything else - the pinhole ``K``, the frame-window
coefficients, the intrinsics a controller solves with - is **derived here and
nowhere else**. A hard-coded focal length once meant a D455 fallback silently
servoing a D435i, so :class:`CameraConfig` raises on a missing key rather than
defaulting.

Fences
------
- **Single source.** Nothing outside this module and :data:`vsbench.vs.scene.D435I` may
  state ``1406.75``, ``960`` or ``540``; ``tests/fast/test_camera.py`` asserts the
  two homes agree.
- **No fallback camera.** A missing key in ``camera.yaml`` raises.

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/data/camera.py`` and
``refinevs.control.Intrinsics``), itself ported from ``genvs-sim``
``src/data_gen/camera_config.py`` and ``src/eval/free_flying_scene.camera_info``.
The encoder-input and dataset-storage contracts that lived beside these numbers
belong to the learning project and are not part of the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

#: repo root: vssim/camera.py -> vs-sim/
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_YAML = REPO_ROOT / "configs" / "camera.yaml"


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in **full-image** pixels.

    The fielded camera is the D435i at 1920x1080 with a 42 deg vertical field of
    view, so ``fx = fy = height / (2 tan 21 deg) = 1406.75`` and the principal point
    is the image centre. A matcher that works at a reduced resolution maps its
    points back itself; the simulator and the control law never rescale.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1920
    height: int = 1080

    @classmethod
    def from_camera_info(cls, camera_info: dict,
                         fallback: Intrinsics | None = None) -> Intrinsics:
        """Read a ``camera_info`` dict, keeping ``fallback`` for absent keys.

        There is **no** built-in fallback camera. A caller that gives neither a
        complete dict nor a fallback gets a ``KeyError`` naming the field, because
        a silently wrong focal length is a silently wrong millimetre.
        """
        info = camera_info or {}
        keys = (("f_x", "fx"), ("f_y", "fy"), ("c_x", "cx"), ("c_y", "cy"),
                ("width", "width"), ("height", "height"))
        out: dict[str, Any] = {}
        for src, dst in keys:
            if src in info and info[src] is not None:
                out[dst] = info[src]
            elif fallback is not None:
                out[dst] = getattr(fallback, dst)
            else:
                raise KeyError(f"camera_info has no {src!r} and no fallback Intrinsics was given")
        return cls(fx=float(out["fx"]), fy=float(out["fy"]), cx=float(out["cx"]),
                   cy=float(out["cy"]), width=int(out["width"]), height=int(out["height"]))

    @property
    def K(self) -> np.ndarray:
        """The 3x3 pinhole matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def as_camera_info(self) -> dict:
        """The inverse of :meth:`from_camera_info`."""
        return {"f_x": self.fx, "f_y": self.fy, "c_x": self.cx, "c_y": self.cy,
                "width": self.width, "height": self.height}


class CameraConfig:
    """Parsed ``configs/camera.yaml`` plus the optics derived from it."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_CAMERA_YAML
        self.raw = yaml.safe_load(self.path.read_text())
        intr = self.raw["intrinsics"]
        self.width = int(intr["resolution"]["width"])
        self.height = int(intr["resolution"]["height"])
        self.fov_vertical_deg = float(intr["fov_vertical_deg"])
        self.vert_aperture = float(intr["vert_aperture_mm"])
        # Square aperture; horizontal follows the pixel aspect.
        self.horiz_aperture = self.vert_aperture * (self.width / self.height)
        self.focal_length = (self.vert_aperture / 2.0) / math.tan(
            math.radians(self.fov_vertical_deg / 2.0))
        # Frame-window coefficients: metres of frame per metre of distance.
        self.v_coeff = 2.0 * math.tan(math.radians(self.fov_vertical_deg / 2.0))
        self.h_coeff = self.v_coeff * (self.width / self.height)

        wd = self.raw.get("working_distance", {}) or {}
        self.frame_fill_default = float(wd.get("frame_fill", 0.70))
        self._per_cat_fill = dict(wd.get("per_category_frame_fill", {}) or {})

    # ------------------------------------------------------------- intrinsics
    def camera_info(self) -> dict:
        """The pinhole camera in pixels (``fx = W / h_coeff``, ``fy = H / v_coeff``,
        principal point at the centre)."""
        return {"f_x": float(self.width / self.h_coeff),
                "f_y": float(self.height / self.v_coeff),
                "c_x": self.width / 2.0, "c_y": self.height / 2.0,
                "width": int(self.width), "height": int(self.height)}

    def intrinsics(self) -> Intrinsics:
        """The same camera as an :class:`Intrinsics` object."""
        return Intrinsics.from_camera_info(self.camera_info())

    # ---------------------------------------------------------------- framing
    def frame_fill(self, category: str) -> float:
        return float(self._per_cat_fill.get(category, self.frame_fill_default))

    def frame_windows_m(self, distance_m: float) -> tuple[float, float]:
        """``(vertical, horizontal)`` frame extents in metres at this distance."""
        return distance_m * self.v_coeff, distance_m * self.h_coeff


def native_K(cam: CameraConfig) -> np.ndarray:
    """Pinhole ``K`` at render resolution."""
    return np.array([[cam.width / cam.h_coeff, 0.0, cam.width / 2.0],
                     [0.0, cam.height / cam.v_coeff, cam.height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
