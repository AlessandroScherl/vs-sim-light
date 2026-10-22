"""Step 3 gate 9: IBVS decay, plus the setup checks for ``true_point_ibvs``.

The method projects the 13 marker points with the true poses and needs no renderer, so
every test here runs the ordinary episode loop on the blank stand-in renderer. Decay: 20
seeded offsets up to 2 cm and 2 deg from the marker target pose, gain 1.0, 50 steps; the
per-step ratio fitted to the log of the feature error norm over the first 50 steps must
be within 0.002 of ``1 - gain * dt``. The three setup checks must pass on every axis.
"""

import numpy as np
import pytest
from conftest import BlankRenderer

from vsbench.camera import gate_camera
from vsbench.checks import run_checks
from vsbench.episode import EpisodeConfig, run_episode
from vsbench.interaction import normalised
from vsbench.methods import make_method
from vsbench.methods.true_point_ibvs import TruePointIBVS
from vsbench.reference import project
from vsbench.scenes import marker_target_pose, world_points
from vsbench.se3 import exp_se3

DT = 0.04
GAIN = 1.0
POINTS = world_points("marker")


class Recording(TruePointIBVS):
    """Keeps the feature error norm the method reports at every step."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.errors = []
        self.infos = []

    def reset(self):
        super().reset()
        self.errors = []
        self.infos = []

    def get_control_rate(self, current_rgb, current_depth=None):
        twist, info = super().get_control_rate(current_rgb, current_depth)
        self.errors.append(info["error_norm"])
        self.infos.append(info)
        return twist, info


def small_offset(rng) -> np.ndarray:
    """A random direction with translation up to 2 cm and rotation up to 2 deg."""
    v = rng.normal(size=3)
    w = rng.normal(size=3)
    v *= rng.uniform(0.5, 1.0) * 0.02 / np.linalg.norm(v)
    w *= rng.uniform(0.5, 1.0) * np.deg2rad(2.0) / np.linalg.norm(w)
    return np.concatenate([v, w])


def test_ibvs_decay():
    r = BlankRenderer(gate_camera())
    T_t = marker_target_pose(r.cam)
    m = Recording(POINTS, gain=GAIN)
    cfg = EpisodeConfig(dt=DT, max_steps=50, stop_tol=None)
    rng = np.random.default_rng(310)
    ratios = []
    for _ in range(20):
        log = run_episode(r, m, T_t @ exp_se3(small_offset(rng)), T_t, cfg, "marker")
        assert log.meta["status"] == "timeout" and len(log) == 50 and len(m.errors) == 50
        e = np.array(m.errors)
        assert np.all(np.isfinite(e)) and np.all(np.diff(e) < 0.0), "the feature error must fall every step"
        slope = np.polyfit(np.arange(50), np.log(e), 1)[0]
        ratios.append(float(np.exp(slope)))
        # the reported error is the norm of the normalised-coordinate feature error
        x, y = normalised(r.cam, project(r.cam, log.T_wc[0], POINTS)[0])
        xd, yd = normalised(r.cam, project(r.cam, T_t, POINTS)[0])
        assert abs(np.linalg.norm(np.concatenate([x - xd, y - yd])) - e[0]) < 1e-12
        assert all(i["inliers"].all() and not i["lost"] and i["kp_cur"].shape == (13, 2) for i in m.infos)
    ratios = np.array(ratios)
    worst = float(np.abs(ratios - (1.0 - GAIN * DT)).max())
    print(f"step3 ibvs decay true_point_ibvs: 20 offsets up to 2 cm and 2 deg, gain {GAIN}, 50 steps: fitted ratio {ratios.min():.5f} to {ratios.max():.5f} against {1.0 - GAIN * DT:.3f}, worst gap {worst:.2e}")
    assert worst < 0.002


def test_true_point_ibvs_setup_checks():
    r = BlankRenderer(gate_camera())
    T_t = marker_target_pose(r.cam)
    m = make_method("true_point_ibvs", points_w=POINTS)
    results = run_checks(r, m, T_t, EpisodeConfig(), "marker")
    for x in results:
        print(f"step3 {x.name.replace('_', ' ')} true_point_ibvs: {x.detail}" + (f"; per axis {x.per_axis}" if x.per_axis else ""))
    assert all(x.passed for x in results), [x.detail for x in results if not x.passed]
    assert results[0].numbers["v_norm"] < 1e-6 and results[0].numbers["w_norm"] < 1e-6
    assert results[1].numbers["max_drift_m"] < 1e-6 and results[1].numbers["max_drift_rad"] < 1e-6


def test_true_point_ibvs_edge_cases():
    cam = gate_camera()
    with pytest.raises(ValueError):
        TruePointIBVS(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        TruePointIBVS(POINTS, gain=np.inf)
    m = make_method("true_point_ibvs", points_w=POINTS)
    assert m.params["gain"] == 1.0 and np.array_equal(np.array(m.params["points_w"]), POINTS)
    with pytest.raises(RuntimeError):
        m.get_control_rate(None)
    m.configure(cam.K, cam.width, cam.height, DT)
    m.reset()
    with pytest.raises(RuntimeError):
        m.get_control_rate(None)
    # points behind the camera: lost, zero twist
    T_t = marker_target_pose(cam)
    T_back = T_t.copy()
    T_back[:3, 3] = (0.0, 0.0, -0.7)
    m.observe_truth(T_back, T_t)
    twist, info = m.get_control_rate(None)
    assert info["lost"] and np.all(twist == 0.0) and not info["inliers"].any()
    # the truth is consumed: a second call without a new observation raises
    with pytest.raises(RuntimeError):
        m.get_control_rate(None)
