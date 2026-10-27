# vs-sim copy: tests/fast/test_labels.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Ground truth by reprojection, against the closed-form plane.

Guards `vssim/labels.py`: the depth-based closure agrees with the mock plane's
analytic correspondence, the occlusion test works on a synthetic occluder, and
points that leave the image or the plane are invalid.
"""

from __future__ import annotations

import numpy as np
import pytest

from vsbench.vs.labels import (
    backproject,
    gt_correspondence,
    plane_membership,
    project,
    relative_pose,
    transform,
)
from vsbench.vs.scene import MockPlaneScene, Pose, pose_to_T, rodrigues


@pytest.fixture(scope="module")
def plane_pair():
    """Two views of the mock plane, with rgb + depth for each and the poses."""
    scene = MockPlaneScene(render_scale=0.125)               # 240 x 135
    goal = scene.goal_pose
    pose_a = Pose(goal.position + np.array([0.01, -0.02, 0.02]),
                  goal.rotation @ rodrigues([0.0, 0.0, 1.0], 0.2))
    pose_b = Pose(goal.position + np.array([0.06, -0.04, -0.05]),
                  goal.rotation @ rodrigues([0.3, 0.1, 1.0] / np.linalg.norm([0.3, 0.1, 1.0]),
                                            -0.35))
    scene.reset(pose_a)
    _rgb_a, depth_a, _ = scene.render()
    scene.reset(pose_b)
    _rgb_b, depth_b, _ = scene.render()
    return scene, pose_a, pose_b, depth_a, depth_b


def _K(scene: MockPlaneScene) -> np.ndarray:
    return scene.intrinsics.K


def _grid(n: int = 12) -> np.ndarray:
    us = np.linspace(40, 200, n)
    vs = np.linspace(30, 105, n)
    uu, vv = np.meshgrid(us, vs)
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


# ------------------------------------------------- the closed-form agreement
def test_depth_reprojection_equals_the_analytic_plane(plane_pair):
    """On a plane the two answers are the same answer: `labels.gt_correspondence`
    goes pixel -> depth -> world -> pixel; the mock goes through the plane with no
    depth at all. They must agree wherever the depth map is a real surface - the
    depth is nearest-sampled at integer pixels, so the grid is on integers."""
    scene, pose_a, pose_b, depth_a, depth_b = plane_pair
    pts = np.round(_grid())
    uv, valid, z_b = gt_correspondence(pts, depth_a, _K(scene), pose_to_T(pose_a),
                                       pose_to_T(pose_b), depth_b=depth_b)
    exact, ok = scene.gt_correspondence(pts, pose_b, goal_pose=pose_a)
    assert valid.mean() > 0.5
    np.testing.assert_allclose(uv[valid], exact[valid], atol=1e-6)
    assert np.all(z_b[valid] > 0)
    assert np.all(ok[valid])


def test_the_pieces_round_trip(plane_pair):
    scene, pose_a, _pose_b, depth_a, _db = plane_pair
    K = _K(scene)
    pts = np.round(_grid(4))
    X_a, ok = backproject(pts, depth_a, K)
    assert ok.all()
    uv, z = project(X_a, K)
    # the ray is built from the exact pixel; only the DEPTH is nearest-sampled
    np.testing.assert_allclose(uv, pts, atol=1e-9)
    np.testing.assert_allclose(z, X_a[:, 2], atol=0)
    T_a = pose_to_T(pose_a)
    np.testing.assert_allclose(transform(relative_pose(T_a, T_a), X_a), X_a, atol=1e-12)
    # the back-projected points lie on the plane z = 0 in the world
    world = transform(T_a, X_a)
    np.testing.assert_allclose(world[:, 2], 0.0, atol=1e-9)


# -------------------------------------------------------- the occlusion test
def test_an_occluder_in_view_b_invalidates_the_points_it_hides(plane_pair):
    """Without the depth-consistency test the `gt_*` columns over-count on the
    cluttered rooms - so it is tested with an occluder."""
    scene, pose_a, pose_b, depth_a, depth_b = plane_pair
    K, T_a, T_b = _K(scene), pose_to_T(pose_a), pose_to_T(pose_b)
    pts = np.round(_grid())
    _uv, valid_open, _z = gt_correspondence(pts, depth_a, K, T_a, T_b, depth_b=depth_b)
    occluded = depth_b.copy()
    occluded[:, :120] *= 0.5                       # a slab in front of the left half
    _uv2, valid_occ, _z2 = gt_correspondence(pts, depth_a, K, T_a, T_b, depth_b=occluded)
    assert valid_occ.sum() < valid_open.sum()
    assert np.all(valid_open[valid_occ])           # occlusion only ever removes
    # and with no depth map at all the test simply does not run
    _uv3, valid_none, _z3 = gt_correspondence(pts, depth_a, K, T_a, T_b, depth_b=None)
    assert valid_none.sum() >= valid_open.sum()


def test_points_leaving_the_image_are_invalid(plane_pair):
    scene, pose_a, _pose_b, depth_a, depth_b = plane_pair
    far = Pose(pose_a.position + np.array([0.9, 0.0, 0.0]), pose_a.rotation)
    pts = np.array([[5.0, 5.0], [235.0, 130.0], [120.0, 67.0]])
    _uv, valid, _z = gt_correspondence(pts, depth_a, _K(scene), pose_to_T(pose_a),
                                       pose_to_T(far), depth_b=depth_b)
    assert not valid.any()


def test_a_pixel_without_depth_is_invalid(plane_pair):
    """The mock writes depth 0 where a ray misses the plate; that is "no surface"."""
    scene, pose_a, pose_b, depth_a, depth_b = plane_pair
    hole = depth_a.copy()
    hole[60:70, 100:110] = 0.0
    pts = np.array([[105.0, 65.0], [180.0, 40.0]])
    _uv, valid, _z = gt_correspondence(pts, hole, _K(scene), pose_to_T(pose_a),
                                       pose_to_T(pose_b), depth_b=depth_b)
    assert list(valid) == [False, True]


def test_plane_membership_separates_object_from_ground(plane_pair):
    """Every rendered scene's ground is `z = 0`, so a pixel closer than the plane
    along its ray is on an object - no masks, no re-render."""
    scene, pose_a, _pose_b, depth_a, _db = plane_pair
    pts = np.round(_grid(6))
    on_object = plane_membership(pts, depth_a, _K(scene), pose_to_T(pose_a))
    assert not on_object.any()                     # the plane IS the ground
    lifted = depth_a * 0.5                         # everything half as far: objects
    assert plane_membership(pts, lifted, _K(scene), pose_to_T(pose_a)).all()
