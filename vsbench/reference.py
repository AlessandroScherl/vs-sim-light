"""Analytic reference: projection, plane z-depth and the poster homography. NumPy only.

Everything here is what a correct renderer must reproduce. The tests compare each
backend against these functions, and ``tests/test_reference.py`` compares these
functions against ``cv2.projectPoints`` so the reference itself is checked once.
"""

from __future__ import annotations

import numpy as np

from vsbench.camera import CameraModel


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4 x 4."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def project(cam: CameraModel, T_wc: np.ndarray, X_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (N, 2) and z-depths (N,) of world points (N, 3) seen from ``T_wc``."""
    T_cw = invert(T_wc)
    X = np.asarray(X_w, dtype=np.float64).reshape(-1, 3)
    Xc = X @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = Xc[:, 2]
    u = cam.fx * Xc[:, 0] / z + cam.cx
    v = cam.fy * Xc[:, 1] / z + cam.cy
    return np.stack([u, v], axis=1), z


def plane_zdepth(
    cam: CameraModel,
    T_wc: np.ndarray,
    T_wp: np.ndarray,
    size: tuple[float, float] | None = None,
) -> np.ndarray:
    """Per-pixel z-depth (H, W) of the plane z = 0 of frame ``T_wp``; NaN where missed.

    With ``size = (w, h)`` only hits inside the finite rectangle count. Pixels whose ray
    is parallel to the plane, hits it behind the camera or beyond ``cam.far`` are NaN.
    """
    rays_c = cam.pixel_rays()  # (H, W, 3), z = 1
    R_wc = T_wc[:3, :3]
    c_w = T_wc[:3, 3]
    d_w = rays_c @ R_wc.T
    n_w = T_wp[:3, 2]
    p0 = T_wp[:3, 3]
    denom = d_w @ n_w
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (n_w @ (p0 - c_w)) / denom  # ray parameter == z-depth because ray z = 1
    ok = np.isfinite(s) & (s > cam.near) & (s < cam.far) & (np.abs(denom) > 1e-9)
    if size is not None:
        hit_w = c_w + s[..., None] * d_w
        T_pw = invert(T_wp)
        hit_p = hit_w @ T_pw[:3, :3].T + T_pw[:3, 3]
        ok &= (np.abs(hit_p[..., 0]) <= size[0] / 2) & (np.abs(hit_p[..., 1]) <= size[1] / 2)
    out = np.where(ok, s, np.nan)
    return out.astype(np.float64)


def texture_to_plane(size: tuple[float, float], tex_w: int, tex_h: int) -> np.ndarray:
    """3 x 3 affine map from texture pixel (s, t) to plane-local (x, y) in metres.

    Integer texture indices are texel centres (as OpenCV treats image pixels). Column
    0 is at x = -w/2 + half a texel, row 0 (top of the picture) at y = +h/2 - half a
    texel, so the picture is upright when viewed from +z with x to the right.
    """
    w, h = size
    sx, sy = w / tex_w, h / tex_h
    return np.array(
        [[sx, 0.0, -w / 2 + 0.5 * sx], [0.0, -sy, h / 2 - 0.5 * sy], [0.0, 0.0, 1.0]]
    )


def plane_homography(cam: CameraModel, T_wc: np.ndarray, T_wp: np.ndarray) -> np.ndarray:
    """3 x 3 homography from plane-local (x, y) metres to image pixels."""
    T_cp = invert(T_wc) @ T_wp
    Rt = np.column_stack([T_cp[:3, 0], T_cp[:3, 1], T_cp[:3, 3]])
    return cam.K @ Rt


def texture_homography(
    cam: CameraModel,
    T_wc: np.ndarray,
    T_wp: np.ndarray,
    size: tuple[float, float],
    tex_w: int,
    tex_h: int,
) -> np.ndarray:
    """3 x 3 homography from texture pixels to image pixels (both pixel-centre indexed).

    ``cv2.warpPerspective(texture, H, (W, H))`` with this ``H`` is the reference render
    of an unlit poster.
    """
    return plane_homography(cam, T_wc, T_wp) @ texture_to_plane(size, tex_w, tex_h)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    return q[:, :2] / q[:, 2:3]
