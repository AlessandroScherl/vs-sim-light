# vs-sim copy: vssim/labels.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Ground-truth correspondence from depth and poses - the truth every evaluation reads.

Purpose
-------
Where does a pixel of view A land in view B? Back-project with A's depth, move
through the relative pose, project into B. That single closure supplies training
labels, oracle matchers and every ``gt_*`` telemetry column, so it lives in one
place and is numpy-only.

``valid`` is the load-bearing half. A correspondence is valid only when A's depth
is finite and positive, the point is in front of B, it lands inside B's image
**and**, when B's depth map is given, B's rendered depth agrees with the point's
own ``z``. That last test is the occlusion test: a point hidden behind something
in B has a rendered depth closer than its own, and without the test the ``gt_*``
columns over-count on cluttered scenes.

Fences
------
- **Depth is sampled nearest-neighbour, never interpolated.** Rendered depth is
  piecewise smooth; interpolating across an object edge invents geometry.
- **No occlusion map, no occlusion claim.** ``depth_b=None`` is legal and cheap,
  and then ``valid`` only means "in front and inside" - callers that quote a
  wrong-match share must pass ``depth_b``.

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/data/labels.py``,
itself ``genvs-sim`` ``src/refine/geometry.py`` verbatim, tolerances included).
The match-labelling half (``valid`` / ``wrong`` / ``forced`` in that project's
patch units) belongs to the learning project and is not here.
:func:`gt_correspondence_mapped` (2026-09-30, the debug log's truth per match) is new
here: the same steps with a world change between the two views.

Golden tests: ``tests/fast/test_labels.py`` - against the closed-form plane of
:class:`vsbench.vs.scene.MockPlaneScene`, plus a synthetic occluder.
"""

from __future__ import annotations

import numpy as np

OCCLUSION_ABS_TOL_M = 0.005
OCCLUSION_REL_TOL = 0.02


def backproject(points_uv, depth: np.ndarray, K) -> tuple[np.ndarray, np.ndarray]:
    """Pixels ``(N,2)`` + per-pixel z-depth -> camera-frame points, and validity."""
    pts = np.asarray(points_uv, dtype=np.float64)
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, depth.shape[0] - 1)
    z = np.asarray(depth)[v, u].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    rays = (Kinv @ np.hstack([pts, np.ones((pts.shape[0], 1))]).T).T
    return rays * z[:, None], valid


def project(X_cam, K) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points ``(N,3)`` -> pixels ``(N,2)`` and depth ``z``."""
    X = np.asarray(X_cam, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (np.asarray(K, dtype=np.float64) @ X.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, X[:, 2]


def transform(T, X) -> np.ndarray:
    """Apply a 4x4 transform to ``(N,3)`` points."""
    T = np.asarray(T, dtype=np.float64)
    return (T[:3, :3] @ np.asarray(X, dtype=np.float64).T).T + T[:3, 3]


def relative_pose(T_w_a, T_w_b) -> np.ndarray:
    """``T_b_a``: maps points from camera-A's frame into camera-B's frame."""
    return np.linalg.inv(np.asarray(T_w_b, dtype=np.float64)) @ np.asarray(T_w_a, np.float64)


def gt_correspondence(points_a, depth_a: np.ndarray, K, T_w_a, T_w_b,
                      depth_b: np.ndarray | None = None,
                      occlusion_rel_tol: float = OCCLUSION_REL_TOL,
                      occlusion_abs_tol: float = OCCLUSION_ABS_TOL_M
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel GT positions in view B for pixels of view A.

    Returns ``(points_b (N,2), valid (N,), z_b (N,))``. See the module docstring
    for what ``valid`` requires; ``depth_b`` enables the occlusion test
    ``|z_b - depth_b(u', v')| <= abs_tol + rel_tol * z_b``.
    """
    depth_a = np.asarray(depth_a)
    X_a, valid = backproject(points_a, depth_a, K)
    uv_b, z_b = project(transform(relative_pose(T_w_a, T_w_b), X_a), K)

    h, w = depth_a.shape[:2] if depth_b is None else np.asarray(depth_b).shape[:2]
    valid = valid & (z_b > 0) & np.isfinite(uv_b).all(axis=1)
    valid = valid & ((uv_b[:, 0] >= 0) & (uv_b[:, 0] <= w - 1)
                     & (uv_b[:, 1] >= 0) & (uv_b[:, 1] <= h - 1))

    if depth_b is not None:
        depth_b = np.asarray(depth_b)
        u = np.clip(np.round(np.nan_to_num(uv_b[:, 0])).astype(int), 0, w - 1)
        v = np.clip(np.round(np.nan_to_num(uv_b[:, 1])).astype(int), 0, h - 1)
        z_rendered = depth_b[v, u].astype(np.float64)
        consistent = np.abs(z_b - z_rendered) <= (occlusion_abs_tol
                                                  + occlusion_rel_tol * np.abs(z_b))
        valid = valid & np.isfinite(z_rendered) & (z_rendered > 0) & consistent
    return uv_b, valid, z_b


def gt_correspondence_mapped(points_a, depth_a: np.ndarray, K, T_w_a, T_w_b,
                             depth_b: np.ndarray | None = None, world_map=None,
                             occlusion_rel_tol: float = OCCLUSION_REL_TOL,
                             occlusion_abs_tol: float = OCCLUSION_ABS_TOL_M
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:func:`gt_correspondence` for a world that CHANGED between the two views.

    The same steps - back-project with A's depth (nearest pixel), move, project into B,
    the same ``valid`` and occlusion test - with one optional hop between the two frames:
    ``world_map(X_world) -> X_world`` says where each world point of view A is in view
    B's world (the debug log's scaled copy of an object: ``P + s (X - P)``). With
    ``world_map=None`` the returned arrays are exactly :func:`gt_correspondence`'s (a test
    pins it). Returns ``(points_b, valid, z_b, X_world)`` - ``X_world`` the points of view
    A in world coordinates BEFORE the map, so a caller can split object from floor.
    """
    depth_a = np.asarray(depth_a)
    X_a, valid = backproject(points_a, depth_a, K)
    T_w_a = np.asarray(T_w_a, dtype=np.float64)
    X_world = transform(T_w_a, X_a)
    if world_map is None:
        # the very arithmetic of gt_correspondence (one relative pose), so the arrays match
        X_b = transform(relative_pose(T_w_a, T_w_b), X_a)
    else:
        X_b = transform(np.linalg.inv(np.asarray(T_w_b, dtype=np.float64)),
                        np.asarray(world_map(X_world), dtype=np.float64))
    uv_b, z_b = project(X_b, K)

    h, w = depth_a.shape[:2] if depth_b is None else np.asarray(depth_b).shape[:2]
    valid = valid & (z_b > 0) & np.isfinite(uv_b).all(axis=1)
    valid = valid & ((uv_b[:, 0] >= 0) & (uv_b[:, 0] <= w - 1)
                     & (uv_b[:, 1] >= 0) & (uv_b[:, 1] <= h - 1))
    if depth_b is not None:
        depth_b = np.asarray(depth_b)
        u = np.clip(np.round(np.nan_to_num(uv_b[:, 0])).astype(int), 0, w - 1)
        v = np.clip(np.round(np.nan_to_num(uv_b[:, 1])).astype(int), 0, h - 1)
        z_rendered = depth_b[v, u].astype(np.float64)
        consistent = np.abs(z_b - z_rendered) <= (occlusion_abs_tol
                                                  + occlusion_rel_tol * np.abs(z_b))
        valid = valid & np.isfinite(z_rendered) & (z_rendered > 0) & consistent
    return uv_b, valid, z_b, X_world


def plane_membership(points_uv, depth: np.ndarray, K, T_w_c,
                     plane_z: float = 0.0, tol: float = 0.005) -> np.ndarray:
    """True where a pixel's rendered depth is strictly **closer** than the ground
    plane along its ray - i.e. the pixel is on an object, not on the floor.

    Every rendered scene's ground plane is ``z = plane_z`` by construction, so
    object membership needs no masks and no re-render. Pixels whose ray never
    meets the plane count as object when they carry finite depth (content above
    the horizon is not ground).
    """
    pts = np.asarray(points_uv, dtype=np.float64)
    depth = np.asarray(depth)
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, depth.shape[0] - 1)
    z_px = depth[v, u].astype(np.float64)
    has_depth = np.isfinite(z_px) & (z_px > 0)

    T = np.asarray(T_w_c, dtype=np.float64)
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    rays_cam = (Kinv @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    rays_world = rays_cam @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (plane_z - T[2, 3]) / rays_world[:, 2]
    hits_plane = np.isfinite(s) & (s > 0)
    z_plane = s * rays_cam[:, 2]
    return has_depth & np.where(hits_plane, z_px < z_plane - tol, True)


def object_mask_from_depth(depth: np.ndarray, intrinsics, pose, plane_z: float = 0.0,
                           tol: float = 0.005) -> np.ndarray:
    """Bool ``[H, W]``: True where the pixel is on an OBJECT, not on the ground plane.

    :func:`plane_membership` over the whole pixel grid of one rendered view: the
    camera's intrinsics and its optical-frame pose (``position``, ``rotation`` - the
    world-from-camera transform) place every ray, the rendered depth says whether it
    stopped short of ``z = plane_z``. This is the goal mask the ViT arms can be handed
    (``matcher.goal_mask: objects``): landmarks then come from objects only, never
    from the floor. Needs no segmentation and no second render - the goal depth is on
    record for every scene.
    """
    depth = np.asarray(depth)
    h, w = depth.shape[:2]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.column_stack([uu.ravel(), vv.ravel()])
    K = np.array([[float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                  [0.0, float(intrinsics.fy), float(intrinsics.cy)],
                  [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.asarray(pose.rotation, np.float64)
    T[:3, 3] = np.asarray(pose.position, np.float64).reshape(3)
    return plane_membership(pts, depth, K, T, plane_z=plane_z, tol=tol).reshape(h, w)



#: The object test's height threshold for the OUTLINE OVERLAP score (the instance suite,
#: 2026-09-23): 1 mm, not the 5 mm of the matchers' goal mask. Checked on the station on
#: the instance suite's preview: at 5 mm a 0.73x copy of a 12.7 cm mug loses the floor of
#: its inside (4.1 mm above the ground) and its outline overlap reads 0.73 instead of
#: 1.000; at 1-2 mm it reads 1.000. The matchers' goal mask keeps its 5 mm (unchanged).
OUTLINE_TOL_M = 0.001


def outline_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    """Intersection over union of two object outlines (bool masks of one image size):
    1.0 = the same outline, 0.0 = nothing shared, ``None`` when both are empty."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError(f"outline_iou: masks of different shapes {a.shape} vs {b.shape}")
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum()) / union
