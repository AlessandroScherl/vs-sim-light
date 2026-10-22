"""Step 3 gate 10: feature convergence, plus the setup checks for ``sift_ibvs`` and
``orb_ibvs`` on the poster scene at 640 x 480.

Feature convergence: ``sift_ibvs`` with dense depth from the ten demo starts of the brief
must get below 5 mm and 0.5 deg within 500 steps (run with those thresholds as the stop
rule and the usual 25-step hold, so the error must stay there). The ORB count is
reported, not gated. Both methods pass the three setup checks with the default stop rule.
"""

import numpy as np
import pytest

from vsbench.camera import gate_camera
from vsbench.checks import run_checks
from vsbench.episode import DEMO_STARTS, EpisodeConfig, demo_offsets, demo_start_poses, run_episode
from vsbench.methods import make_method
from vsbench.methods.feature_ibvs import FeatureIBVS
from vsbench.scenes import poster_target_pose
from vsbench.se3 import exp_se3, log_se3

FEATURE_TOL = (0.005, float(np.deg2rad(0.5)))
FEATURE_STEPS = 500


def run_demo_starts(r, method, T_t):
    cfg = EpisodeConfig(max_steps=FEATURE_STEPS, stop_tol=FEATURE_TOL)
    rows = []
    for k, T0 in enumerate(demo_start_poses(T_t)):
        log = run_episode(r, method, T0, T_t, cfg, "poster")
        rows.append({"start": k + 1, "status": log.meta["status"], "steps": len(log), "mm": log.meta["final_error_m"] * 1e3, "deg": float(np.rad2deg(log.meta["final_error_rad"])), "ms": log.meta["call_s_mean"] * 1e3})
    return rows


def summary(name, rows):
    n_ok = sum(x["status"] == "converged" for x in rows)
    return (f"{name}: {n_ok} of 10 starts below 5 mm and 0.5 deg within {FEATURE_STEPS} steps; steps {min(x['steps'] for x in rows)} to {max(x['steps'] for x in rows)}, "
            f"worst final {max(x['mm'] for x in rows):.2f} mm and {max(x['deg'] for x in rows):.3f} deg, {np.mean([x['ms'] for x in rows]):.0f} ms per call")


def test_demo_starts_are_the_brief():
    assert len(DEMO_STARTS) == 10
    xi = demo_offsets()
    assert xi.shape == (10, 6)
    assert np.allclose(xi[0], [0.05, 0, 0, 0, 0, 0]) and np.allclose(xi[9], [-0.15, -0.10, -0.20, *np.deg2rad([-6, 8, -60])])
    T_t = poster_target_pose(gate_camera())
    for T0, off in zip(demo_start_poses(T_t), xi):
        assert np.allclose(log_se3(np.linalg.inv(T_t) @ T0), off, atol=1e-12)
        assert np.allclose(T0, T_t @ exp_se3(off))


def test_feature_convergence_sift(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    rows = run_demo_starts(r, make_method("sift_ibvs"), poster_target_pose(cam))
    print(f"{backend} step3 feature convergence {summary('sift_ibvs', rows)}")
    for x in rows:
        print(f"    start {x['start']:2d} {x['status']:9s} {x['steps']:3d} steps, {x['mm']:.2f} mm, {x['deg']:.3f} deg")
    assert all(x["status"] == "converged" for x in rows), rows
    assert all(x["mm"] < 5.0 and x["deg"] < 0.5 and x["steps"] <= FEATURE_STEPS for x in rows)


def test_feature_convergence_orb_reported(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    rows = run_demo_starts(r, make_method("orb_ibvs"), poster_target_pose(cam))
    print(f"{backend} step3 feature convergence {summary('orb_ibvs', rows)} (reported, not gated)")
    for x in rows:
        print(f"    start {x['start']:2d} {x['status']:9s} {x['steps']:3d} steps, {x['mm']:.2f} mm, {x['deg']:.3f} deg")
    assert all(x["status"] != "method_error" for x in rows)


@pytest.mark.parametrize("name", ["sift_ibvs", "orb_ibvs"])
def test_feature_setup_checks(backend, renderer_factory, name):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    results = run_checks(r, make_method(name), poster_target_pose(cam), EpisodeConfig(), "poster")
    for x in results:
        print(f"{backend} step3 {x.name.replace('_', ' ')} {name}: {x.detail}" + (f"; per axis {x.per_axis}" if x.per_axis else ""))
    assert all(x.passed for x in results), [x.detail for x in results if not x.passed]


def test_feature_options(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    T_0 = demo_start_poses(T_t)[2]  # 10 cm back: depth matters most here
    rgb_t, depth_t = r.render(T_t)
    rgb_0, depth_0 = r.render(T_0)
    prior = float(np.median(depth_t[depth_t > 0]))
    # scalar depth: the prior for every point; still a restoring, converging law
    m = make_method("sift_ibvs", depth="scalar")
    assert m.wants_depth == "scalar" and m.params["depth"] == "scalar"
    log = run_episode(r, m, T_0, T_t, EpisodeConfig(max_steps=FEATURE_STEPS, stop_tol=FEATURE_TOL), "poster")
    print(f"{backend} step3 feature options sift scalar depth from start 3: {log.meta['status']} in {len(log)} steps, {log.meta['final_error_m'] * 1e3:.2f} mm")
    assert log.meta["status"] == "converged"
    # fundamental-matrix RANSAC gives a finite twist with the right sign on vz
    m = make_method("sift_ibvs", ransac="fundamental")
    m.configure(cam.K, cam.width, cam.height, 0.04)
    m.reset()
    m.set_target(rgb_t, depth_t, prior)
    twist, info = m.get_control_rate(rgb_0, depth_0)
    assert np.all(np.isfinite(twist)) and twist[2] > 0.0 and info["inliers"].sum() >= 8 and not info["lost"]
    # too few inliers: zero twist and lost
    m = make_method("sift_ibvs")
    m.configure(cam.K, cam.width, cam.height, 0.04)
    m.reset()
    m.set_target(rgb_t, depth_t, prior)
    blank = np.full_like(rgb_t, 128)
    twist, info = m.get_control_rate(blank, depth_0)
    assert info["lost"] and np.all(twist == 0.0) and np.isnan(info["error_norm"])
    # dense depth drops keypoints without depth
    twist, info = m.get_control_rate(rgb_0, np.zeros_like(depth_0))
    assert info["lost"] and not info["inliers"].any()
    # a run is repeatable
    a = m.get_control_rate(rgb_0, depth_0)[0]
    b = m.get_control_rate(rgb_0, depth_0)[0]
    assert np.array_equal(a, b)
    for bad in (dict(detector="surf"), dict(depth="none"), dict(ransac="affine"), dict(gain=np.nan), dict(min_inliers=2)):
        with pytest.raises(ValueError):
            FeatureIBVS(**bad)
    with pytest.raises(RuntimeError):
        make_method("orb_ibvs").get_control_rate(rgb_0, depth_0)
