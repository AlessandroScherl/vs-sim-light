"""Step 3 gates 3 and 4: oracle decay and oracle convergence.

The oracle commands ``gain * log_se3(inv(T_wc) @ T_wc_target)`` from the true poses, so
after each step the norm of the pose-error logarithm must be exactly ``1 - gain * dt``
times the previous one. Decay: 20 seeded starts inside the clamp, gain 1.0, 100 steps on
the rendered poster scene, every logged pose within 1e-9 of ``(1 - gain dt)^k`` times the
start. Convergence: 50 seeded starts up to 1 m and 170 deg reach 0.1 mm and 0.01 deg
within 1000 steps, every frame rendered on the poster scene.
"""

import numpy as np
import pytest

from vsbench.camera import gate_camera
from vsbench.episode import EpisodeConfig, run_episode
from vsbench.methods import make_method
from vsbench.reference import invert
from vsbench.scenes import poster_target_pose
from vsbench.se3 import exp_se3, log_se3, pose_error, random_twist

DT = 0.04
GAIN = 1.0


def error_norms(log, T_target) -> np.ndarray:
    poses = list(log.T_wc) + [log.final_pose()]
    return np.array([np.linalg.norm(log_se3(invert(T) @ T_target)) for T in poses])


def test_oracle_decay(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    rng = np.random.default_rng(300)
    cfg = EpisodeConfig(dt=DT, max_steps=100, stop_tol=None)
    m = make_method("oracle_pbvs", gain=GAIN)
    worst = 0.0
    for i in range(20):
        offset = random_twist(rng, max_angle=np.deg2rad(170.0), max_dist=1.0)
        assert np.linalg.norm(GAIN * offset[:3]) <= cfg.limits.v_max and np.linalg.norm(GAIN * offset[3:]) <= cfg.limits.w_max
        log = run_episode(r, m, T_t @ exp_se3(offset), T_t, cfg, "poster")
        assert log.meta["status"] == "timeout" and len(log) == 100
        e = error_norms(log, T_t)
        k = np.arange(len(e))
        expected = (1.0 - GAIN * DT) ** k * e[0]
        worst = max(worst, float(np.abs(e - expected).max()))
        # the commanded twist is never clamped, so applied == commanded
        assert np.array_equal(log.twist_cmd, log.twist_applied)
    print(f"{backend} step3 oracle decay: 20 starts, gain {GAIN}, 100 steps each: max |e_k - (1 - gain dt)^k e_0| {worst:.2e}")
    assert worst < 1e-9


def test_oracle_convergence(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    rng = np.random.default_rng(301)
    tol = (1e-4, np.deg2rad(0.01))
    cfg = EpisodeConfig(dt=DT, max_steps=1000, stop_tol=tol, hold_steps=1)
    m = make_method("oracle_pbvs", gain=GAIN)
    steps, e_t, e_r = [], [], []
    for i in range(50):
        offset = random_twist(rng, max_angle=np.deg2rad(170.0), max_dist=1.0)
        offset[:3] *= rng.uniform(0.2, 1.0) / max(np.linalg.norm(offset[:3]), 1e-12)  # up to 1 m
        log = run_episode(r, m, T_t @ exp_se3(offset), T_t, cfg, "poster")
        assert log.meta["status"] == "converged", log.meta
        a, b = pose_error(log.final_pose(), T_t)
        steps.append(len(log))
        e_t.append(a)
        e_r.append(b)
    steps, e_t, e_r = np.array(steps), np.array(e_t), np.array(e_r)
    print(f"{backend} step3 oracle convergence: 50 starts up to 1 m and 170 deg: all below 0.1 mm and 0.01 deg after {steps.min()} to {steps.max()} steps (median {int(np.median(steps))}); worst final {e_t.max() * 1e3:.4f} mm, {np.rad2deg(e_r.max()):.5f} deg")
    assert np.all(steps <= 1000)
    assert np.all(e_t < 1e-4) and np.all(e_r < np.deg2rad(0.01))


def test_oracle_needs_truth():
    m = make_method("oracle_pbvs")
    m.configure(gate_camera().K, 640, 480, DT)
    m.reset()
    with pytest.raises(RuntimeError):
        m.get_control_rate(np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(ValueError):
        make_method("oracle_pbvs", gain=np.nan)
