"""Step 2 gate 7: rendered flow. The image must move as the twist predicts.

Poster and clutter scenes at 640 x 480 from a seeded start pose. For each axis alone
and for one random twist, scaled so the median predicted flow over one ``FreeCamera``
step is about 5 px: Shi-Tomasi corners of frame A, tracked to frame B with pyramidal
Lucas-Kanade, against ``exact_flow`` with the rendered depth of frame A. Corners within
5 px of a depth edge or the border, and corners predicted to leave the image, are
dropped. Pass: at least 100 tracked points, median endpoint error at most 0.3 px, and a
median cosine of at least 0.98 over points with more than 1 px of predicted flow.
"""

import numpy as np
import pytest

from vsbench.camera import gate_camera
from vsbench.gates import depth_edge_mask, flow_agreement, shi_tomasi_corners, track_lk
from vsbench.interaction import exact_flow
from vsbench.motion import FreeCamera
from vsbench.poses import sample_poses

AXES = ["vx", "vy", "vz", "wx", "wy", "wz", "random"]
SCENES = ["poster", "clutter"]
DT = 0.04
MEDIAN_FLOW_PX = 5.0
MARGIN = 5


def start_pose(scene: str) -> np.ndarray:
    seed = {"poster": 31, "clutter": 32}[scene]
    return sample_poses(seed, 1, np.eye(4), dist=(0.6, 1.0), max_slant_deg=40.0, target_jitter=0.05)[0]


def unit_twist(axis: int, seed: int) -> np.ndarray:
    if axis == 6:
        xi = np.random.default_rng(seed).normal(size=6)
        return xi / np.linalg.norm(xi)
    xi = np.zeros(6)
    xi[axis] = 1.0
    return xi


def corners_with_depth(cam, rgb, depth):
    """Corners of a frame away from depth edges and the border, with their z-depths."""
    pts = shi_tomasi_corners(rgb)
    bad = depth_edge_mask(depth, margin=MARGIN)
    iu = np.clip(np.rint(pts[:, 0]).astype(int), 0, cam.width - 1)
    iv = np.clip(np.rint(pts[:, 1]).astype(int), 0, cam.height - 1)
    keep = ~bad[iv, iu] & (depth[iv, iu] > 0)
    return pts[keep], depth[iv, iu][keep].astype(np.float64)


def rendered_flow_case(cam, renderer, T_a, axis, seed=40):
    """Render A, pick and track corners, return the agreement statistics and the arrays."""
    rgb_a, depth_a = renderer.render(T_a)
    pts, Z = corners_with_depth(cam, rgb_a, depth_a)
    xi = unit_twist(axis, seed)
    med = np.median(np.linalg.norm(exact_flow(cam, pts, Z, xi, DT), axis=1))
    xi = xi * (MEDIAN_FLOW_PX / med)
    cam_free = FreeCamera(T_a, dt=DT)
    applied = cam_free.step(xi)
    assert np.array_equal(applied, xi), "the test twist must sit inside the motion limits"
    rgb_b, _ = renderer.render(cam_free.T_wc)
    pred = exact_flow(cam, pts, Z, applied, DT)
    inside = np.all(np.isfinite(pred), axis=1)
    p_pred = pts + np.nan_to_num(pred)
    inside &= (p_pred[:, 0] >= MARGIN) & (p_pred[:, 0] <= cam.width - 1 - MARGIN)
    inside &= (p_pred[:, 1] >= MARGIN) & (p_pred[:, 1] <= cam.height - 1 - MARGIN)
    pts, pred = pts[inside], pred[inside]
    p_b, ok = track_lk(rgb_a, rgb_b, pts)
    measured = p_b[ok] - pts[ok]
    stats = flow_agreement(measured, pred[ok])
    stats["n_corners"] = int(len(pts))
    stats["twist"] = applied.tolist()
    return stats, (rgb_a, rgb_b, pts[ok], measured, pred[ok])


@pytest.mark.parametrize("axis", range(7), ids=AXES)
@pytest.mark.parametrize("scene", SCENES)
def test_rendered_flow(backend, renderer_factory, scene, axis):
    cam = gate_camera()
    r = renderer_factory(backend, scene, (640, 480))
    stats, _ = rendered_flow_case(cam, r, start_pose(scene), axis)
    print(
        f"{backend} step2 rendered flow {scene} {AXES[axis]}: {stats['n']} tracked of {stats['n_corners']} corners, "
        f"median EPE {stats['median_epe_px']:.3f} px (p90 {stats['p90_epe_px']:.3f}), "
        f"median cos {stats['median_cos']:.4f} over {stats['n_over_1px']} pts, median predicted {stats['median_pred_px']:.2f} px"
    )
    assert stats["n"] >= 100, f"only {stats['n']} tracked points"
    assert stats["median_epe_px"] <= 0.3, f"median endpoint error {stats['median_epe_px']:.3f} px"
    assert stats["median_cos"] >= 0.98, f"median cosine {stats['median_cos']:.4f}"
