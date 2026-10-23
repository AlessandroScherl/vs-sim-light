# vs-sim copy: tests/fast/test_camera.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""`configs/camera.yaml` is the one camera.

Guards `vssim/camera.py`: fx = fy = 1406.75, cx = 960, cy = 540 from 1920x1080 at
a 42 deg vertical field of view, the mock's pinned camera agrees, and a missing
key raises rather than defaulting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vsbench.vs.camera import CameraConfig, Intrinsics, native_K
from vsbench.vs.scene import D435I


@pytest.fixture(scope="module")
def cam() -> CameraConfig:
    return CameraConfig()


def test_intrinsics_come_out_of_the_yaml(cam):
    """1920x1080 at vFoV 42 deg -> fx = fy = 1406.75, principal point centred."""
    assert (cam.width, cam.height) == (1920, 1080)
    assert cam.fov_vertical_deg == 42.0
    info = cam.camera_info()
    assert info["f_x"] == pytest.approx(1406.75, abs=5e-3)
    assert info["f_y"] == pytest.approx(1406.75, abs=5e-3)
    assert (info["c_x"], info["c_y"]) == (960.0, 540.0)
    # fx = fy is not an accident: the aperture is square and h_coeff carries the
    # pixel aspect, so both axes see the same focal length in pixels.
    assert info["f_x"] == pytest.approx(info["f_y"], rel=1e-12)
    assert info["f_y"] == pytest.approx(1080 / (2 * math.tan(math.radians(21.0))), rel=1e-12)


def test_intrinsics_object_and_K_agree(cam):
    intr = cam.intrinsics()
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (
        cam.camera_info()["f_x"], cam.camera_info()["f_y"], 960.0, 540.0)
    assert (intr.width, intr.height) == (1920, 1080)
    np.testing.assert_array_equal(intr.K, native_K(cam))
    assert Intrinsics.from_camera_info(intr.as_camera_info()) == intr


def test_the_mock_scene_camera_is_the_same_camera(cam):
    """`scene.D435I` is a second home for these numbers, pinned by hand so the mock
    is a real camera. It carries the **rounded** 1406.75 where the yaml derives
    1406.748094934653, so the focal lengths are equal to 2 dp and no further."""
    intr = cam.intrinsics()
    assert (round(intr.fx, 2), round(intr.fy, 2)) == (D435I.fx, D435I.fy)
    assert abs(intr.fx - D435I.fx) < 3e-3
    assert (intr.cx, intr.cy, intr.width, intr.height) == (
        D435I.cx, D435I.cy, D435I.width, D435I.height)


def test_from_camera_info_has_no_fallback_camera():
    """There is no default camera: a partial dict without a fallback raises."""
    with pytest.raises(KeyError, match="f_x"):
        Intrinsics.from_camera_info({"width": 640, "height": 480})
    got = Intrinsics.from_camera_info({"f_x": 500.0}, fallback=D435I)
    assert (got.fx, got.fy, got.cx) == (500.0, D435I.fy, D435I.cx)


def test_a_missing_yaml_key_raises_rather_than_defaulting(tmp_path):
    bad = tmp_path / "camera.yaml"
    bad.write_text("intrinsics:\n  resolution: {width: 1920, height: 1080}\n"
                   "  vert_aperture_mm: 20.955\n")           # no fov_vertical_deg
    with pytest.raises(KeyError):
        CameraConfig(bad)


def test_frame_windows_are_the_documented_coefficients(cam):
    assert cam.v_coeff == pytest.approx(0.767, abs=1e-3)
    assert cam.h_coeff == pytest.approx(1.364, abs=1e-3)
    v, h = cam.frame_windows_m(1.0)
    assert (v, h) == (cam.v_coeff, cam.h_coeff)
    assert cam.frame_fill("drinking_glass") == 0.50    # the per-category override
    assert cam.frame_fill("anything_else") == cam.frame_fill_default == 0.70


def test_the_usd_focal_length_follows_from_the_aperture(cam):
    """Isaac's camera prim takes a focal length in aperture units, not pixels."""
    assert cam.focal_length == pytest.approx(20.955 / 2 / math.tan(math.radians(21.0)))
    assert cam.horiz_aperture == pytest.approx(20.955 * 16 / 9)
