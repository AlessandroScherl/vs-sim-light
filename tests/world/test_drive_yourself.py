"""The baselines as methods in a loop of your own (``vsbench.methods.vssim_arms``), on the
MuJoCo world: the README's "Drive the camera yourself" loop.

* each wrapped baseline builds exactly the protocol cell the benchmark flies (same resolved
  settings but the device, the same derived label);
* an arm that cannot run says why (missing torch, transformers, weights);
* the sanity check at the goal: at the goal pose, with the goal picture, ``sift_ransac``
  commands (almost) no motion - the identity thresholds of vs-sim's rig check;
* the README loop itself: ``sift_ransac`` from 1 cm to the side of poster2's goal, rendered,
  commanded and applied by hand, comes closer.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")

from vsbench.methods.vssim_arms import ARM_IDS  # noqa: E402

#: the scenes whose assets this module needs (tests/conftest.py skips it without them)
NEEDS_SCENES = ("poster2",)
#: vs-sim's identity verdict (``scripts/check_baselines.py``: IDENTITY_V_MAX, IDENTITY_W_MAX)
IDENTITY_V_MAX = 1e-4                    # m/s
IDENTITY_W_MAX = float(np.radians(0.01))  # rad/s


@pytest.mark.parametrize("name", ARM_IDS)
def test_the_wrapped_baseline_is_the_protocol_cell(name):
    from vsbench.camera import gate_camera
    from vsbench.methods import make_method
    from vsbench.runtime import resolve_device
    from vsbench.vs.bench.arms import arm_label
    from vsbench.world import fly

    cam = gate_camera()
    m = make_method(name)
    try:
        m.configure(cam.K, cam.width, cam.height, 1 / 60)
        proto = fly.load_protocol()
        ref = fly.arm_config(proto, name, device="cpu")
        assert m.label == arm_label(ref) == arm_label(fly.arm_config(proto, name))
        strip = ("_source", "matcher")
        assert {k: v for k, v in m.config.items() if k not in strip} == \
            {k: v for k, v in ref.items() if k not in strip}
        assert {k: v for k, v in m.config["matcher"].items() if k != "device"} == \
            {k: v for k, v in fly.arm_config(proto, name)["matcher"].items() if k != "device"}
        assert m._arm.kind == "method" and m.privileged is False and m.wants_depth == "dense"
        if "device" in m.config["matcher"]:
            assert m.config["matcher"]["device"] == resolve_device("auto") == m.resolved_device
        if m.config["matcher"]["variant"].startswith("classical"):
            assert m._arm.matcher.ransac == "homography"      # the poster's model
    finally:
        m.close()


def test_an_arm_that_cannot_run_says_why(monkeypatch, tmp_path):
    import importlib.util

    from vsbench.methods import method_class
    from vsbench.methods import vssim_arms as va

    have = {n: method_class(n).unavailable() for n in ARM_IDS}
    assert have["sift_ransac"] is None and have["orb_ransac"] is None
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    assert "weights" in va.unavailable_reason("vitvs_dinov3")
    assert "weights" in va.unavailable_reason("artvs")
    assert va.unavailable_reason("sift_ransac") is None
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "torch" else real(name, *a))
    assert all("torch" in va.unavailable_reason(n) for n in ARM_IDS)


@pytest.fixture(scope="module")
def poster2():
    from vsbench.world.mujoco_scene import MujocoScene

    with MujocoScene("poster2") as scene:
        yield scene


def _sift(scene):
    from vsbench.methods import make_method

    m = make_method("sift_ransac")
    K = scene.intrinsics
    m.configure(K.K, K.width, K.height, dt=1 / 60)
    goal_rgb, goal_depth = scene.render_goal()
    m.set_target(goal_rgb, goal_depth)
    return m


def test_zero_twist_at_the_goal(poster2):
    """The sanity check every method should pass: at the goal, no motion."""
    m = _sift(poster2)
    try:
        poster2.reset(poster2.goal_pose)
        rgb, depth, _ = poster2.render()
        twist, info = m.get_control_rate(rgb, depth)
        assert len(info["kp_cur"]) >= 4 and not info["lost"]
        assert np.linalg.norm(twist[:3]) < IDENTITY_V_MAX, twist
        assert np.linalg.norm(twist[3:]) < IDENTITY_W_MAX, twist
    finally:
        m.close()


def test_the_readme_loop_comes_closer(poster2):
    """The README snippet, 30 steps: render, command, apply - closer than the 10 mm start."""
    from vsbench.vs.scene import Pose

    g = poster2.goal_pose
    poster2.reset(Pose(g.position + [0.01, 0.0, 0.0], g.rotation))
    m = _sift(poster2)
    try:
        for _ in range(30):
            rgb, depth, _pose = poster2.render()
            twist, _info = m.get_control_rate(rgb, depth)
            poster2.apply(twist, 1 / 60)
    finally:
        m.close()
    mm = 1000 * float(np.linalg.norm(poster2.pose.position - g.position))
    assert mm < 9.0, mm


def test_apply_is_the_documented_integration(poster2):
    """``docs/IBVS.md``: p <- p + R v dt, R <- R exp([w]x dt), in the current camera frame."""
    from vsbench.se3 import exp_so3
    from vsbench.vs.scene import Pose

    rng = np.random.default_rng(0)
    g = poster2.goal_pose
    poster2.reset(Pose(g.position.copy(), g.rotation.copy()))
    twist = rng.normal(size=6) * [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]
    dt = 1 / 60
    p0, R0 = poster2.pose.position.copy(), poster2.pose.rotation.copy()
    poster2.apply(twist, dt)
    assert np.allclose(poster2.pose.position, p0 + R0 @ twist[:3] * dt, atol=1e-15)
    assert np.allclose(poster2.pose.rotation, R0 @ exp_so3(twist[3:] * dt), atol=1e-12)
