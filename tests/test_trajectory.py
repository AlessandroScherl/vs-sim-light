"""Step 2 gates 8 and 9: trajectory round trip (save, load, arrays identical) and replay
(poses re-integrated from the first pose are bit-identical to the log)."""

import json

import numpy as np
import pytest

from vsbench.motion import FreeCamera, MotionLimits
from vsbench.poses import look_at, roll
from vsbench.trajectory import Trajectory


def fly(n=200, dt=0.04, seed=50, limits=MotionLimits(v_max=0.5, w_max=1.0)):
    """A seeded flight with commands that exceed the limits half the time."""
    rng = np.random.default_rng(seed)
    T0 = roll(look_at((0.3, -0.5, 0.9), (0.0, 0.0, 0.0)), 0.3)
    cam = FreeCamera(T0, dt=dt, limits=limits)
    log = Trajectory(dt, limits, meta={"scene": "poster", "seed": seed})
    for _ in range(n):
        cmd = rng.normal(size=6) * rng.choice([0.1, 3.0])
        pose_before = cam.T_wc
        t_before = cam.t
        applied = cam.step(cmd)
        log.append(t_before, pose_before, cmd, applied)
    return cam, log


def test_round_trip(tmp_path):
    cam, log = fly()
    path = tmp_path / "flight.npz"
    log.save(path)
    back = Trajectory.load(path)
    a, b = log.arrays(), back.arrays()
    for k in a:
        assert a[k].dtype == np.float64 and b[k].dtype == np.float64
        assert np.array_equal(a[k], b[k]), k
    assert a["T_wc"].shape == (200, 4, 4) and a["twist_cmd"].shape == (200, 6)
    assert back.dt == log.dt and back.limits == log.limits and back.meta == {"scene": "poster", "seed": 50}
    # the file is one compressed npz with a JSON meta string
    with np.load(path) as f:
        assert set(f.files) == {"meta", "t", "T_wc", "twist_cmd", "twist_applied"}
        meta = json.loads(str(f["meta"]))
    assert meta["dt"] == 0.04 and meta["limits"] == {"v_max": 0.5, "w_max": 1.0}
    print(f"step2 trajectory round trip: {len(back)} rows, {sum(v.nbytes for v in a.values())} bytes of arrays, file {path.stat().st_size} bytes")


def test_replay_is_bit_identical(tmp_path):
    cam, log = fly()
    path = tmp_path / "flight.npz"
    log.save(path)
    back = Trajectory.load(path)
    poses = back.replay()
    assert poses.shape == (200, 4, 4)
    n_diff = int((poses != back.T_wc).any(axis=(1, 2)).sum())
    print(f"step2 replay: {n_diff} of {len(back)} poses differ from the log")
    assert np.array_equal(poses, back.T_wc)
    assert np.array_equal(back.final_pose(), cam.T_wc)
    # the applied twists respect the limits and differ from the commands where they had to
    v = np.linalg.norm(back.twist_applied[:, :3], axis=1)
    w = np.linalg.norm(back.twist_applied[:, 3:], axis=1)
    assert np.all(v <= 0.5 + 1e-12) and np.all(w <= 1.0 + 1e-12)
    assert np.any(back.twist_cmd != back.twist_applied)
    assert np.allclose(back.t, np.arange(200) * 0.04)


def test_path_length_and_errors():
    dt = 0.1
    T0 = np.eye(4)
    cam = FreeCamera(T0, dt=dt)
    log = Trajectory(dt)
    xi = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        pose, t = cam.T_wc, cam.t
        log.append(t, pose, xi, cam.step(xi))
    m, rad = log.path_length()
    assert abs(m - 9 * 0.02) < 1e-12 and rad < 1e-12  # 9 gaps between 10 logged poses
    cam.reset(T0)
    log = Trajectory(dt)
    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    for _ in range(10):
        pose, t = cam.T_wc, cam.t
        log.append(t, pose, xi, cam.step(xi))
    m, rad = log.path_length()
    assert m < 1e-12 and abs(rad - 9 * 0.05) < 1e-12
    assert Trajectory(dt).replay().shape == (0, 4, 4)
    assert Trajectory(dt).path_length() == (0.0, 0.0)
    with pytest.raises(ValueError):
        Trajectory(0.0)
    with pytest.raises(ValueError):
        Trajectory(dt).final_pose()
    with pytest.raises(ValueError):
        Trajectory(dt).append(0.0, np.eye(3), np.zeros(6), np.zeros(6))
