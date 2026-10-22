"""Gate 3: the poster quad against cv2.warpPerspective of the texture with the analytic
homography; ORB matches between the two must sit on top of each other."""

import cv2
import numpy as np
import pytest
from conftest import RESOLUTIONS, poses_for

from vsbench.camera import gate_camera
from vsbench.gates import orb_displacements
from vsbench.poses import sample_poses
from vsbench.reference import texture_homography
from vsbench.stdscenes import POSTER_SIZE, load_poster


@pytest.mark.parametrize("res", RESOLUTIONS, ids=lambda r: f"{r[0]}x{r[1]}")
def test_homography(backend, renderer_factory, res):
    cam = gate_camera(*res)
    tex = load_poster()
    th, tw = tex.shape[:2]
    r = renderer_factory(backend, "poster", res)
    T_wp = np.eye(4)
    worst_med, fewest = 0.0, 10**9
    for T in sample_poses(21, poses_for(res), T_wp, dist=(0.5, 1.2), max_slant_deg=50.0, target_jitter=0.05):
        rgb, _ = r.render(T)
        H = texture_homography(cam, T, T_wp, POSTER_SIZE, tw, th)
        ref = cv2.warpPerspective(tex, H, (cam.width, cam.height), flags=cv2.INTER_LINEAR)
        disp = orb_displacements(rgb, ref)
        n = len(disp)
        med = float(np.median(disp)) if n else np.inf
        worst_med, fewest = max(worst_med, med), min(fewest, n)
        assert n >= 100, f"only {n} ORB matches"
        assert med <= 0.5, f"median match displacement {med:.3f} px with {n} matches"
    print(f"{backend} {res}: worst median displacement {worst_med:.3f} px, fewest matches {fewest}")
