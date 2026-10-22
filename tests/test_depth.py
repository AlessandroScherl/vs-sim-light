"""Gate 2: one plane, 0.3 to 1.5 m, slants to 60 degrees, per-pixel z-depth against the
analytic ray-plane intersection."""

import numpy as np
import pytest
from conftest import RESOLUTIONS, poses_for

from vsbench.camera import gate_camera
from vsbench.poses import sample_poses
from vsbench.reference import plane_zdepth


@pytest.mark.parametrize("res", RESOLUTIONS, ids=lambda r: f"{r[0]}x{r[1]}")
def test_depth(backend, renderer_factory, res):
    cam = gate_camera(*res)
    r = renderer_factory(backend, "plane", res)
    T_wp = np.eye(4)
    abs_err = []
    for T in sample_poses(5, poses_for(res), T_wp, dist=(0.3, 1.5), max_slant_deg=60.0, target_jitter=0.0):
        _, d = r.render(T)
        ref = plane_zdepth(cam, T, T_wp, size=(8.0, 8.0))
        valid = np.isfinite(ref) & (d > 0)
        assert valid.mean() > 0.5, "plane should cover most of the image"
        abs_err.append(np.abs(d[valid] - ref[valid]))
    e = np.concatenate(abs_err)
    med, p99 = np.median(e), np.percentile(e, 99)
    print(f"{backend} {res}: median {med * 1e3:.3f} mm, p99 {p99 * 1e3:.3f} mm over {len(e)} px")
    assert med <= 1e-3, f"median abs error {med * 1e3:.3f} mm"
    assert p99 <= 3e-3, f"99th percentile {p99 * 1e3:.3f} mm"
