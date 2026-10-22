"""Gate 1: 13 unlit markers, 9 on a plane and 4 off it, through the shifted-principal-point
camera. Centroids by colour threshold against the analytic projection."""

import numpy as np
import pytest
from conftest import RESOLUTIONS, poses_for

from vsbench.camera import gate_camera
from vsbench.gates import match_markers, projection_poses


@pytest.mark.parametrize("res", RESOLUTIONS, ids=lambda r: f"{r[0]}x{r[1]}")
def test_projection(backend, renderer_factory, res):
    cam = gate_camera(*res)
    r = renderer_factory(backend, "marker", res)
    errs = []
    for T in projection_poses(cam, poses_for(res), seed=11):
        rgb, _ = r.render(T)
        errs.append(match_markers(cam, T, rgb))
    errs = np.concatenate(errs)
    print(f"{backend} {res}: mean {errs.mean():.3f} px, max {errs.max():.3f} px over {len(errs)} markers")
    assert errs.mean() <= 0.5, f"mean error {errs.mean():.3f} px"
    assert errs.max() <= 1.0, f"max error {errs.max():.3f} px"
