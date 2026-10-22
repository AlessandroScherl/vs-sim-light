"""The analytic reference against OpenCV: projection and homography to 1e-6 px."""

import cv2
import numpy as np

from vsbench.camera import gate_camera
from vsbench.poses import sample_poses
from vsbench.reference import (
    apply_homography,
    invert,
    plane_zdepth,
    project,
    texture_homography,
    texture_to_plane,
)
from vsbench.scene import rigid, rot_x, rot_z


def _rvec_tvec(T_cw):
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    return rvec, T_cw[:3, 3].reshape(3, 1)


def test_projection_matches_opencv():
    cam = gate_camera()
    T_wp = rigid(rot_x(0.3) @ rot_z(0.7), (0.2, -0.1, 0.05))
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.3, 0.3, (50, 3))
    for T_wc in sample_poses(1, 10, T_wp):
        px, z = project(cam, T_wc, X)
        rvec, tvec = _rvec_tvec(invert(T_wc))
        ref, _ = cv2.projectPoints(X, rvec, tvec, cam.K, None)
        assert np.abs(px - ref.reshape(-1, 2)).max() < 1e-6
        assert np.all(z > 0)


def test_texture_homography_matches_opencv():
    cam = gate_camera()
    size = (0.8, 0.6)
    tw, th = 1280, 960
    T_wp = rigid(rot_x(-0.2) @ rot_z(0.4), (0.1, 0.05, -0.02))
    rng = np.random.default_rng(3)
    st = np.column_stack([rng.uniform(0, tw - 1, 200), rng.uniform(0, th - 1, 200)])
    A = texture_to_plane(size, tw, th)
    xy = apply_homography(A, st)
    X_p = np.column_stack([xy, np.zeros(len(xy))])
    X_w = X_p @ T_wp[:3, :3].T + T_wp[:3, 3]
    for T_wc in sample_poses(2, 20, T_wp):
        H = texture_homography(cam, T_wc, T_wp, size, tw, th)
        px = apply_homography(H, st)
        rvec, tvec = _rvec_tvec(invert(T_wc))
        ref, _ = cv2.projectPoints(X_w, rvec, tvec, cam.K, None)
        assert np.abs(px - ref.reshape(-1, 2)).max() < 1e-6


def test_texture_corners_land_on_the_quad():
    # texel (0, 0) centre sits half a texel inside the top-left corner of the quad
    A = texture_to_plane((0.8, 0.6), 1280, 960)
    tl = apply_homography(A, [[0.0, 0.0]])[0]
    assert np.allclose(tl, [-0.4 + 0.5 * 0.8 / 1280, 0.3 - 0.5 * 0.6 / 960])
    br = apply_homography(A, [[1279.0, 959.0]])[0]
    assert np.allclose(br, [0.4 - 0.5 * 0.8 / 1280, -0.3 + 0.5 * 0.6 / 960])


def test_plane_zdepth_is_z_not_range():
    cam = gate_camera()
    T_wp = np.eye(4)
    T_wc = rigid(np.diag([1.0, -1.0, -1.0]), (0.0, 0.0, 1.0))  # looking straight down from 1 m
    d = plane_zdepth(cam, T_wc, T_wp)
    assert np.allclose(d, 1.0)  # z-depth of a fronto-parallel plane is constant
    # a slanted view: the depth at a pixel equals the z of the ray-plane intersection
    T_wc = rigid(rot_x(np.pi - 0.5), (0.0, 0.5, 1.0))
    d = plane_zdepth(cam, T_wc, T_wp)
    rays = cam.pixel_rays()
    u, v = 100, 200
    X_c = rays[v, u] * d[v, u]
    X_w = T_wc[:3, :3] @ X_c + T_wc[:3, 3]
    assert abs(X_w[2]) < 1e-9
