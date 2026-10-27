"""The D435i camera of vs-sim, for the MuJoCo world.

Two cameras, on purpose:

* :func:`d435i_intrinsics` - what the world **reports** (``MujocoScene.intrinsics``): the
  copied ``vsbench.vs.camera.CameraConfig().intrinsics()``, exactly what
  ``IsaacScene.intrinsics`` reports (``fx = fy = 1406.748...``, principal point
  ``(960, 540)``, 1920 x 1080). Every baseline, record and frozen start file was made
  with these numbers.
* :func:`d435i_camera_model` - what the world **renders with**: the same focal length and
  size, principal point ``(959.5, 539.5)`` in vsbench's pixel-centre (OpenCV) terms.
  That is Isaac's geometry: a USD camera's aperture is centred on the window, so the ray
  through integer pixel ``u`` is ``((u + 0.5) - 960) / fx`` (the world follows Isaac). The copied mock (``MockPlaneScene``) puts the
  principal point on pixel 960's centre instead, half a pixel right and down of Isaac;
  ``tests/world/test_scene_posters.py`` measures that half pixel by phase correlation.

Depth: z in the optical frame, metres (Isaac's ``distance_to_image_plane``), clipping
0.05 - 10 m (vsbench's gate-tested planes), no noise model.
"""

from __future__ import annotations

from vsbench.camera import CameraModel

#: The pixel-centre convention the MuJoCo world renders with: Isaac's.
CONVENTION = "isaac_half_pixel"
#: Isaac's principal point minus the reported one, in pixel-centre terms (both axes).
HALF_PIXEL = 0.5
NEAR_M = 0.05
FAR_M = 10.0


def d435i_intrinsics():
    """The reported camera: ``vsbench.vs.camera.CameraConfig().intrinsics()``."""
    from vsbench.vs.camera import CameraConfig

    return CameraConfig().intrinsics()


def d435i_camera_model(intrinsics=None, *, near: float = NEAR_M, far: float = FAR_M) -> CameraModel:
    """The rendered camera: the reported one with Isaac's half-pixel principal point."""
    intr = d435i_intrinsics() if intrinsics is None else intrinsics
    return CameraModel.from_focal(int(intr.width), int(intr.height), fx=float(intr.fx),
                                  fy=float(intr.fy), cx=float(intr.cx) - HALF_PIXEL,
                                  cy=float(intr.cy) - HALF_PIXEL, near=near, far=far)


def describe(cam: CameraModel) -> dict:
    """The rendered camera as a record block."""
    return {"convention": CONVENTION, "width": cam.width, "height": cam.height,
            "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
            "near_m": cam.near, "far_m": cam.far, "depth": "optical z (distance_to_image_plane)",
            "depth_noise": None}
