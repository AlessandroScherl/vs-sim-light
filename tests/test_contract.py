"""Step 3 gate 1: the method contract.

Every registered method goes through ``configure``, ``reset``, ``set_target`` and
``get_control_rate`` on the poster scene at 640 x 480 and must return a finite ``(6,)``
float twist, alone or with an info dict, without modifying its inputs. The same file
checks the registry, the output normalisation, the depth form each ``wants_depth``
receives, and the three episode outcomes (converged, timeout, method_error).
"""

import numpy as np
import pytest
from conftest import BlankRenderer

from vsbench.camera import gate_camera
from vsbench.episode import EpisodeConfig, distance_prior, run_episode
from vsbench.methods import _REGISTRY as REGISTRY
from vsbench.methods import (
    INFO_KEYS,
    available_methods,
    check_method,
    make_method,
    method_class,
    normalise_output,
    register,
)
from vsbench.scenes import make_scene, poster_target_pose
from vsbench.se3 import exp_se3
from vsbench.stdscenes import MARKER_POINTS

OFFSET = np.array([0.03, -0.02, -0.05, np.deg2rad(2.0), np.deg2rad(-3.0), np.deg2rad(8.0)])

# constructor arguments a method needs for the contract gate (none for most)
CONTRACT_PARAMS: dict[str, dict] = {"true_point_ibvs": {"points_w": MARKER_POINTS}}


def start_pose(T_wc_target: np.ndarray) -> np.ndarray:
    return T_wc_target @ exp_se3(OFFSET)


@pytest.mark.parametrize("name", available_methods())
def test_contract(backend, renderer_factory, name):
    # a method whose dependencies are absent here says why (the copied vs-sim arms need torch
    # and offline weights); present, the method returns None and the gate runs in full
    why = getattr(method_class(name), "unavailable", lambda: None)()
    if why:
        pytest.skip(f"{name}: {why}")
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    m = make_method(name, **CONTRACT_PARAMS.get(name, {}))
    check_method(m)
    T_t = poster_target_pose(cam)
    T_0 = start_pose(T_t)
    rgb_t, depth_t = r.render(T_t)
    rgb_0, depth_0 = r.render(T_0)
    K = cam.K.copy()
    inputs = [K.copy(), rgb_t.copy(), depth_t.copy(), rgb_0.copy(), depth_0.copy(), T_0.copy(), T_t.copy()]
    dense = m.wants_depth == "dense"
    prior = distance_prior(depth_t)
    m.configure(K, cam.width, cam.height, 0.04)
    m.reset()
    m.set_target(rgb_t, desired_depth=depth_t if dense else None, distance_prior=prior if m.wants_depth != "none" else None)
    if m.privileged:
        m.observe_truth(T_0, T_t)
    out = m.get_control_rate(rgb_0, depth_0 if dense else None)
    twist, info = normalise_output(out)
    assert twist.shape == (6,) and twist.dtype == np.float64 and np.all(np.isfinite(twist))
    assert isinstance(info, dict)
    unknown = sorted(set(info) - set(INFO_KEYS))
    for a, b in zip(inputs, [K, rgb_t, depth_t, rgb_0, depth_0, T_0, T_t]):
        assert np.array_equal(a, b), "a method modified one of its inputs"
    kind = "twist alone" if not isinstance(out, tuple) else f"twist + info {sorted(info)}"
    print(f"{backend} step3 contract {name}: {kind}, |v| {np.linalg.norm(twist[:3]):.4f} m/s, |w| {np.linalg.norm(twist[3:]):.4f} rad/s, inputs unchanged, unknown info keys {unknown}")
    assert not unknown, f"info keys outside the contract: {unknown}"


def test_normalise_output():
    xi, info = normalise_output(np.arange(6.0))
    assert xi.shape == (6,) and xi.dtype == np.float64 and info == {}
    xi, info = normalise_output(([0, 1, 2, 3, 4, 5], {"lost": True}))
    assert np.array_equal(xi, np.arange(6.0)) and info == {"lost": True}
    xi, _ = normalise_output((np.arange(6.0).reshape(6, 1), None))
    assert xi.shape == (6,)
    xi32 = np.arange(6, dtype=np.float32)
    xi, _ = normalise_output(xi32)
    assert xi.dtype == np.float64
    xi[0] = 99.0
    assert xi32[0] == 0.0, "normalise_output must return a fresh array"
    for bad in (np.zeros(5), np.zeros((2, 6)), [0, 1, 2, 3, 4, np.nan], [0, 1, 2, 3, 4, np.inf], "abc", (np.zeros(6), {}, 1)):
        with pytest.raises(ValueError):
            normalise_output(bad)
    with pytest.raises(TypeError):
        normalise_output((np.zeros(6), [1, 2]))


def test_registry():
    assert "oracle_pbvs" in available_methods()
    with pytest.raises(KeyError):
        make_method("no_such_method")
    m = make_method("oracle_pbvs", gain=0.5)
    assert m.name == "oracle_pbvs" and m.gain == 0.5 and m.params == {"gain": 0.5}

    @register("_contract_probe")
    class Probe:
        wants_depth = "none"
        privileged = False

        def configure(self, K, width, height, dt):
            pass

        def reset(self):
            pass

        def set_target(self, desired_rgb, desired_depth=None, distance_prior=None):
            pass

        def get_control_rate(self, current_rgb, current_depth=None):
            return np.zeros(6)

    try:
        assert Probe.name == "_contract_probe"
        assert isinstance(make_method("_contract_probe"), Probe)
        with pytest.raises(ValueError):
            register("_contract_probe")(type("Other", (), {}))
        with pytest.raises(ValueError):
            register("_contract_probe2")(type("Named", (), {"name": "something_else"}))

        class Bad(Probe):
            wants_depth = "sparse"

        with pytest.raises(ValueError):
            check_method(Bad())
        with pytest.raises(TypeError):
            check_method(object())
    finally:
        REGISTRY.pop("_contract_probe", None)
        REGISTRY.pop("_contract_probe2", None)


class DepthProbe:
    """Records what depth the harness hands over, returns a zero twist."""

    name = "depth_probe"
    privileged = False

    def __init__(self, wants_depth):
        self.wants_depth = wants_depth
        self.seen = {}

    def configure(self, K, width, height, dt):
        self.seen["configure"] = (K.shape, width, height, dt)

    def reset(self):
        self.seen["reset"] = True

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None):
        self.seen["desired_rgb"] = desired_rgb.shape
        self.seen["desired_depth"] = None if desired_depth is None else desired_depth.shape
        self.seen["distance_prior"] = distance_prior

    def get_control_rate(self, current_rgb, current_depth=None):
        self.seen["current_depth"] = None if current_depth is None else current_depth.shape
        self.seen["writeable"] = current_rgb.flags.writeable
        return np.zeros(6)


@pytest.mark.parametrize("mode", ["none", "scalar", "dense"])
def test_depth_form_follows_wants_depth(mode):
    r = BlankRenderer()
    m = DepthProbe(mode)
    T = poster_target_pose(r.cam)
    log = run_episode(r, m, T, T, EpisodeConfig(max_steps=3, stop_tol=None), "poster")
    s = m.seen
    assert s["configure"] == ((3, 3), 640, 480, 0.04) and s["reset"] and s["desired_rgb"] == (480, 640, 3)
    assert s["writeable"] is False, "the harness hands out read-only images"
    if mode == "none":
        assert s["desired_depth"] is None and s["distance_prior"] is None and s["current_depth"] is None
    elif mode == "scalar":
        assert s["desired_depth"] is None and s["distance_prior"] == 1.0 and s["current_depth"] is None
    else:
        assert s["desired_depth"] == (480, 640) and s["distance_prior"] == 1.0 and s["current_depth"] == (480, 640)
    assert log.meta["status"] == "timeout" and len(log) == 3 and r.frames == 4


class Misbehaving(DepthProbe):
    def __init__(self, what):
        super().__init__("none")
        self.what = what

    def get_control_rate(self, current_rgb, current_depth=None):
        if self.what == "nan":
            return np.array([np.nan, 0, 0, 0, 0, 0])
        if self.what == "shape":
            return np.zeros(5)
        raise RuntimeError("boom")


@pytest.mark.parametrize("what", ["nan", "shape", "raise"])
def test_method_error_status(what):
    r = BlankRenderer()
    T = poster_target_pose(r.cam)
    log = run_episode(r, Misbehaving(what), T @ exp_se3(OFFSET), T, EpisodeConfig(max_steps=10), "poster")
    assert log.meta["status"] == "method_error" and len(log) == 0
    assert log.meta["error"] and log.meta["steps"] == 0


def test_converged_status_and_meta():
    r = BlankRenderer()
    T = poster_target_pose(r.cam)
    m = make_method("oracle_pbvs")
    cfg = EpisodeConfig(max_steps=100, hold_steps=5)
    log = run_episode(r, m, T @ exp_se3(0.001 * OFFSET), T, cfg, "poster")
    # starts inside stop_tol: the pose is checked before each step, so hold_steps in-tolerance
    # checks take hold_steps - 1 steps and as many frames
    assert log.meta["status"] == "converged" and len(log) == 4 and r.frames == 1 + 4
    log = run_episode(r, m, T @ exp_se3(OFFSET), T, cfg, "poster")
    assert log.meta["status"] == "converged" and 0 < len(log) < 100
    meta = log.meta
    assert meta["scene"] == "poster" and meta["method"] == "oracle_pbvs" and meta["method_params"] == {}
    assert meta["renderer"]["variant"] == "blank" and meta["camera"]["width"] == 640
    assert np.allclose(meta["T_wc_target"], T) and meta["config"]["hold_steps"] == 5
    assert meta["final_error_m"] < 0.002 and meta["final_error_rad"] < np.deg2rad(0.2)
    assert meta["call_s_mean"] > 0.0
    with pytest.raises(ValueError):
        EpisodeConfig(dt=0.0)
    with pytest.raises(ValueError):
        EpisodeConfig(stop_tol=(0.0, 1.0))
    with pytest.raises(KeyError):
        make_scene("no_such_scene")
