# vs-sim copy: tests/fast/test_scene.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The scene seam and the numpy world behind it.

Guards `vssim/scene.py`: the mock is a consistent pinhole camera, pixels round-trip
through the plane, the twist integration is the frozen optical convention, and a
closed loop on the mock converges - with a pose-based controller that exists only
to exercise `reset` / `render` / `apply` / `pose`, never to say anything about a
method.
"""

from __future__ import annotations

import numpy as np
import pytest

from vsbench.vs.scene import (
    D435I,
    MockPlaneScene,
    Pose,
    Scene,
    checker_texture,
    integrate_optical_velocity,
    pose_to_T,
    rodrigues,
    rotation_error_deg,
    rotvec_deg,
    scaled_intrinsics,
    translation_error_mm,
)


def make_scene(**kw) -> MockPlaneScene:
    """The mock at 1/8 sampling of the real camera: 240 x 135, ~5 ms a frame."""
    return MockPlaneScene(render_scale=0.125, **kw)


def offset_pose(goal: Pose, dx=0.02, dy=-0.015, dz=0.01, roll_rad=0.1) -> Pose:
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Pose(goal.position + np.array([dx, dy, dz]), goal.rotation @ rz)


# ------------------------------------------------------------- the mock world
def test_the_mock_is_a_scene_and_a_consistent_camera():
    scene = make_scene()
    assert isinstance(scene, Scene)
    intr = scene.intrinsics
    assert (intr.width, intr.height) == (240, 135)
    assert intr.fx == pytest.approx(D435I.fx * 0.125)
    assert scaled_intrinsics(D435I, 1.0) == D435I
    with pytest.raises(ValueError, match="factor must be > 0"):
        scaled_intrinsics(D435I, 0.0)
    rgb, depth, pose = scene.render()
    assert rgb.shape == (135, 240, 3) and rgb.dtype == np.uint8
    # the goal view is fronto-parallel, so every pixel that sees the plane is at d
    assert depth[depth > 0] == pytest.approx(scene.working_distance)
    assert pose.position == pytest.approx(scene.goal_pose.position)
    assert scene.n_renders == 1


def test_the_goal_is_the_benchmark_posters_goal():
    """0.8 x 0.6 m plate seen from 0.4342 m straight down: the frozen poster goal."""
    scene = make_scene()
    assert scene.working_distance == pytest.approx(0.4341815107823003)
    np.testing.assert_array_equal(scene.goal_pose.rotation, np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_array_equal(scene.goal_pose.position, [0.0, 0.0, 0.4341815107823003])


def test_pixels_round_trip_through_the_plane():
    scene = make_scene()
    px = np.array([[30.0, 20.0], [120.0, 67.5], [200.5, 100.25]])
    pts, ok = scene.backproject(px, scene.goal_pose)
    assert ok.all()
    back, valid = scene.project(pts, scene.goal_pose)
    assert valid.all() and back == pytest.approx(px, abs=1e-9)
    # and the same points seen from a displaced pose are the GT correspondences
    other = Pose(scene.goal_pose.position + np.array([0.01, 0.0, 0.0]), scene.goal_pose.rotation)
    uv, valid = scene.gt_correspondence(px, other)
    assert valid.all() and not np.allclose(uv, px)
    # a pixel whose ray misses the plate is not a correspondence
    off = np.array([[0.0, 0.0]])
    _uv, valid = scene.gt_correspondence(off, offset_pose(scene.goal_pose, dx=0.4))
    assert not valid.any()


def test_the_texture_is_deterministic_and_the_background_is_stated():
    a, b = checker_texture(), checker_texture()
    np.testing.assert_array_equal(a, b)
    assert a.shape == (512, 512, 3) and a.dtype == np.uint8
    scene = make_scene(background=7)
    scene.reset(offset_pose(scene.goal_pose, dz=0.6))          # far: the plate is small
    rgb, depth, _ = scene.render()
    assert (rgb[depth == 0] == 7).all()


# ------------------------------------------------------------- poses + twists
def test_a_pose_normalises_its_arrays_and_round_trips_a_start_entry():
    p = Pose([0.1, 0.2, 0.3], np.eye(3).tolist())
    assert p.position.dtype == np.float64 and p.rotation.shape == (3, 3)
    q = Pose.from_start(p.as_dict())
    np.testing.assert_array_equal(q.position, p.position)
    T = pose_to_T(p)
    np.testing.assert_array_equal(T[:3, 3], p.position)
    np.testing.assert_array_equal(T[:3, :3], p.rotation)


def test_the_twist_integration_is_the_optical_convention():
    """``pos += R @ v_lin * dt``; ``R <- R @ Rodrigues(w_hat, |w| dt)``."""
    R = np.diag([1.0, -1.0, -1.0])
    pose = Pose([0.1, 0.2, 0.3], R)
    out = integrate_optical_velocity(pose, [1.0, 0.0, 0.0, 0.0, 0.0, np.pi], 0.5)
    assert out.position == pytest.approx([0.6, 0.2, 0.3])       # x moves along R's x column
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert out.rotation == pytest.approx(R @ quarter_turn, abs=1e-12)   # |w| dt = pi/2
    assert pose.position == pytest.approx([0.1, 0.2, 0.3])      # pure: input untouched
    still = integrate_optical_velocity(pose, np.zeros(6), 1.0)
    np.testing.assert_array_equal(still.rotation, R)


def test_the_error_helpers():
    R = np.diag([1.0, -1.0, -1.0])
    assert rotation_error_deg(R, R) == 0.0
    assert rotation_error_deg(R, R @ rodrigues([0, 0, 1.0], np.radians(30))) == pytest.approx(30)
    assert translation_error_mm([0, 0, 0], [0.003, 0.004, 0]) == pytest.approx(5.0)
    v = rotvec_deg(rodrigues([0, 0, 1.0], np.radians(10)))
    assert v == pytest.approx([0.0, 0.0, 10.0], abs=1e-4)
    assert rotvec_deg(np.eye(3)) == [0.0, 0.0, 0.0]
    assert np.linalg.norm(rotvec_deg(np.diag([1.0, -1.0, -1.0]))) == pytest.approx(180.0)


# -------------------------------------------------------------- a closed loop
def test_a_closed_loop_on_the_mock_converges_from_a_small_offset():
    """`demo.run_episode` with the cheating pose controller: the four verbs end to end."""
    from vsbench.vs.demo import pose_servo, run_episode

    scene = make_scene()
    start = offset_pose(scene.goal_pose)
    seen = []
    rec = run_episode(scene, start, pose_servo, max_steps=200,
                      on_step=lambda i, rgb, depth, pose, twist: seen.append(rgb.shape))
    assert rec["start_translation_mm"] == pytest.approx(26.9258, abs=1e-3)
    assert rec["start_rotation_deg"] == pytest.approx(np.degrees(0.1))
    assert rec["final_translation_mm"] < 0.05
    assert rec["final_rotation_deg"] < 0.01
    assert rec["steps"] == 200 and rec["terminated_reason"] == "max_steps"
    assert seen == [(135, 240, 3)] * 200
    assert scene.n_renders == 201                         # the goal view + 200 live frames


def test_run_episode_takes_a_start_file_entry_and_stops_on_a_bad_twist():
    from vsbench.vs.demo import run_episode

    scene = make_scene()
    entry = {"start_id": "t0", **offset_pose(scene.goal_pose).as_dict()}
    rec = run_episode(scene, entry, lambda *_: np.full(6, np.nan), max_steps=5)
    assert rec["start_id"] == "t0" and rec["scene"] == "mock_plane"
    assert rec["steps"] == 0 and rec["terminated_reason"] == "non_finite_twist"
    assert rec["final_translation_mm"] == pytest.approx(rec["start_translation_mm"])
