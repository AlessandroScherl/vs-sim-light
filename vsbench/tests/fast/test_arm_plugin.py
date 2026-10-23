# vs-sim copy: tests/fast/test_arm_plugin.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The external-arm hook (2026-10-11; `docs/controller_spec_2026-10-11.md` section 1.2 at the
workspace root): a live mask from the runner for an arm that asks, the debug log's `method`
block, `check_baselines --arm-plugin`, and the chain's `ARM_PLUGIN`.

The load-bearing test is :func:`test_an_arm_that_does_not_ask_flies_exactly_as_before`: the
truth arm flown with and without a live-mask rule offered - the per-step poses bit for bit,
the telemetry identical, the record equal key for key except the timing columns.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vsbench.vs.bench import runner
from vsbench.vs.bench.debuglog import DebugLog, read_telemetry
from vsbench.vs.labels import object_mask_from_depth
from vsbench.vs.scene import MockPlaneScene

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load("tests/fast/test_runner.py", "_plugin_test_runner")

TIMING = {"wall_time_s", "sec_per_step", "fps", "match_fps", "match_ms_med", "control_ms_med",
          "gpu_util_mean_pct", "gpu_util_max_pct", "gpu_util_samples", "peak_vram_alloc_mb",
          "vram_used_mb"}


class _CountingArm:
    """The truth arm, its ``step`` calls counted by arity - nothing else changed."""

    def __init__(self, inner):
        self._inner = inner
        self.arities: list[int] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def step(self, *args):
        self.arities.append(len(args))
        return self._inner.step(*args)


class _MaskArm(_CountingArm):
    """An arm that asks for a live mask: keeps every mask it is handed, steers as the truth arm,
    and fills ``debug["method"]`` with a few numbers (one non-finite)."""

    requires_live_mask = True
    kind = "external"

    def __init__(self, inner):
        super().__init__(inner)
        self.masks: list[np.ndarray] = []

    def step(self, rgb, depth, mask):
        self.arities.append(3)
        self.masks.append(mask.copy())
        twist, debug, matches = self._inner.step(rgb, depth)
        debug["method"] = {"n_obj": int(mask.sum()), "share": np.float64(mask.mean()),
                           "bad": float("nan"), "nested": {"v": np.array([1.5, -2.0])},
                           "flags": ["a", "b"], "ok": np.bool_(True)}
        return twist, debug, matches


def _fly(arm_wrap, *, live_mask_fn=None, debug_dir=None, max_iter=40):
    scene = tr.make_scene()
    cfg = tr.make_config(max_iter=max_iter)
    arm = arm_wrap(tr.make_arm(scene, cfg, pixel_noise=0.5))
    poses, telem = [], []
    dlog = DebugLog(debug_dir, arm_id="plugin_test") if debug_dir is not None else None
    try:
        rec = runner.run_episode(
            scene, arm, tr.make_start(scene), cfg, telemetry=telem,
            on_step=lambda i, rgb, dbg, pose: poses.append(
                (pose.position.copy(), pose.rotation.copy())),
            debug_log=dlog, live_mask_fn=live_mask_fn)
    finally:
        if dlog is not None:
            dlog.close()
    return rec, telem, poses, arm


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ------------------------------------------------------------------ the runner
def test_an_arm_that_does_not_ask_flies_exactly_as_before():
    plain = _fly(_CountingArm)
    offered = _fly(_CountingArm, live_mask_fn=lambda d, *_: d > 0)
    (rec0, tel0, poses0, arm0), (rec1, tel1, poses1, arm1) = plain, offered
    assert set(arm0.arities) == set(arm1.arities) == {2}          # never handed a mask
    assert len(poses0) == len(poses1) == rec0["steps"]
    for (p0, r0), (p1, r1) in zip(poses0, poses1, strict=True):
        assert np.array_equal(p0, p1) and np.array_equal(r0, r1)
    assert _same(tel0, tel1)
    assert set(rec0) == set(rec1) and "live_mask_rule" not in rec0
    for k in set(rec0) - TIMING:
        assert _same(rec0[k], rec1[k]), k


def test_an_arm_that_asks_gets_the_runner_mask_every_step():
    rec, _t, _p, arm = _fly(_MaskArm, max_iter=6)
    assert arm.arities == [3] * 6 and len(arm.masks) == 6
    # the default rule: the objects above the floor at 5 mm (on the mock plane: none)
    scene = tr.make_scene()
    scene.reset(scene.goal_pose)
    _rgb, depth, pose = scene.render()
    expect = object_mask_from_depth(depth, scene.intrinsics, pose)
    assert arm.masks[0].dtype == np.bool_ and arm.masks[0].shape == depth.shape
    assert not expect.any() and not arm.masks[0].any()
    assert rec["live_mask_rule"] == runner.DEFAULT_LIVE_MASK_RULE == "objects_from_depth_5mm"
    # a given rule is used and named
    rec2, _t, _p, arm2 = _fly(_MaskArm, live_mask_fn=lambda d, *_: d > 0, max_iter=4)
    assert all(m.all() for m in arm2.masks)                        # the plate fills the frame
    assert rec2["live_mask_rule"] == "custom:<lambda>"

    def floor_free(d, intr, pose):
        return d > 0
    floor_free.rule = "depth_positive"
    rec3, *_ = _fly(_MaskArm, live_mask_fn=floor_free, max_iter=2)
    assert rec3["live_mask_rule"] == "depth_positive"


@pytest.mark.parametrize("bad, err", [
    (lambda d, *_: d.astype(np.float64), TypeError),          # a depth is not a mask
    (lambda d, *_: (d > 0).astype(np.uint8), TypeError),
    (lambda d, *_: [[True]], TypeError),
    (lambda d, *_: np.ones((3, 3), bool), ValueError),        # the wrong size
])
def test_the_live_mask_tripwire_refuses_anything_but_a_bool_frame(bad, err):
    with pytest.raises(err):
        _fly(_MaskArm, live_mask_fn=bad, max_iter=2)


def test_the_default_live_mask_is_the_goal_masks_rule():
    """5 mm above the floor - `object_mask_from_depth`'s default, as the goal mask."""
    import inspect

    assert inspect.signature(object_mask_from_depth).parameters["tol"].default == 0.005
    scene = MockPlaneScene(render_scale=0.125)
    scene.reset(scene.goal_pose)
    _rgb, depth, pose = scene.render()
    assert np.array_equal(runner.default_live_mask(depth, scene.intrinsics, pose),
                          object_mask_from_depth(depth, scene.intrinsics, pose))


# --------------------------------------------------------------- the debug log
def test_the_method_block_round_trips_and_is_absent_for_other_arms(tmp_path):
    rec, *_ = _fly(_MaskArm, debug_dir=tmp_path / "ext", max_iter=5,
                   live_mask_fn=lambda d, *_: d > 0)
    steps = [e for e in read_telemetry(tmp_path / "ext" / "telemetry.jsonl.gz")
             if e["kind"] == "step"]
    assert len(steps) == rec["steps"] == 5
    m = steps[0]["method"]
    assert m["n_obj"] > 0 and m["bad"] is None and m["ok"] is True
    assert m["nested"] == {"v": [1.5, -2.0]} and m["flags"] == ["a", "b"]
    assert isinstance(m["share"], float)
    _fly(_CountingArm, debug_dir=tmp_path / "plain", max_iter=5)
    plain = [e for e in read_telemetry(tmp_path / "plain" / "telemetry.jsonl.gz")
             if e["kind"] == "step"]
    assert plain and all("method" not in e for e in plain)


# ------------------------------------------------------------ check_baselines
def _cb():
    sys.path.insert(0, str(REPO / "scripts"))
    return _load("scripts/check_baselines.py", "_plugin_check_baselines")


def _stand_in_plugin(name="standin_plugin", ids=("ext_arm",), *, label="external:standin",
                     kind="external"):
    """A stand-in plugin module: its arm is the mock's truth arm under an external name."""
    mod = types.ModuleType(name)
    mod.ARM_IDS = tuple(ids)
    mod.built = []

    def build(arm_id, *, intrinsics, scene_name, scene_spec=None):
        scene = tr.make_scene()
        cfg = tr.make_config(max_iter=5)
        arm = _MaskArm(tr.make_arm(scene, cfg))
        arm.name, arm.kind = label, kind
        mod.built.append((arm_id, scene_name))
        return arm, cfg

    def describe(arm_id):
        return {"module": name, "commit": "abc1234", "config_source": "configs/x.yaml",
                "label": label}

    mod.build, mod.describe = build, describe
    return mod


def test_check_baselines_resolves_protocol_arms_first_then_the_plugin(monkeypatch):
    cb = _cb()
    import fly

    protocol = fly.load_protocol()
    plug = _stand_in_plugin()
    monkeypatch.setitem(sys.modules, "standin_plugin", plug)
    assert cb.load_arm_plugin(None) is None
    loaded = cb.load_arm_plugin("standin_plugin")
    assert loaded is plug
    src = cb.resolve_arm_sources(protocol, ["vitvs_dinov3", "ext_arm"], loaded)
    assert src == {"vitvs_dinov3": "protocol", "ext_arm": "plugin"}
    # unknown: refused naming both lists
    with pytest.raises(SystemExit, match="the plugin standin_plugin has ext_arm"):
        cb.resolve_arm_sources(protocol, ["nope"], loaded)
    with pytest.raises(SystemExit, match="no --arm-plugin given"):
        cb.resolve_arm_sources(protocol, ["ext_arm"], None)
    # a colon or a slash in an id
    for bad in ("ext:arm", "ext/arm"):
        with pytest.raises(SystemExit, match="A-Za-z0-9_"):
            cb.resolve_arm_sources(protocol, [bad], loaded)
    # a plugin may not shadow a protocol arm, nor state a bad id, nor lack its functions
    monkeypatch.setitem(sys.modules, "shadow_plugin",
                        _stand_in_plugin("shadow_plugin", ids=("vitvs_dinov3",)))
    with pytest.raises(SystemExit, match="shadow a benchmark arm"):
        cb.resolve_arm_sources(protocol, ["vitvs_dinov3"], cb.load_arm_plugin("shadow_plugin"))
    monkeypatch.setitem(sys.modules, "colon_plugin",
                        _stand_in_plugin("colon_plugin", ids=("a:b",)))
    with pytest.raises(SystemExit, match="not \\[A-Za-z0-9_\\]"):
        cb.load_arm_plugin("colon_plugin")
    monkeypatch.setitem(sys.modules, "empty_plugin", types.ModuleType("empty_plugin"))
    with pytest.raises(SystemExit, match="lacks"):
        cb.load_arm_plugin("empty_plugin")
    with pytest.raises(SystemExit, match="not importable"):
        cb.load_arm_plugin("no_such_module_anywhere_123")


def test_a_plugin_arm_must_be_external(monkeypatch):
    cb = _cb()
    scene = tr.make_scene()
    good = _stand_in_plugin()
    arm, cfg, desc = cb.build_plugin_arm(good, "ext_arm", intrinsics=scene.intrinsics,
                                         scene_name="mock_plane")
    assert arm.name.startswith("external:") and desc["commit"] == "abc1234"
    assert good.built == [("ext_arm", "mock_plane")]
    for kw in ({"label": "vitvs_dinov3@x"}, {"kind": "method"}):
        with pytest.raises(ValueError, match="external"):
            cb.build_plugin_arm(_stand_in_plugin(**kw), "ext_arm", intrinsics=scene.intrinsics,
                                scene_name="mock_plane")


def test_the_identity_check_hands_a_live_mask_only_to_an_arm_that_asks():
    cb = _cb()
    scene = tr.make_scene()
    cfg = tr.make_config(max_iter=5)
    asks = _MaskArm(tr.make_arm(scene, cfg))
    out = cb.identity_check(scene, asks)
    assert asks.arities == [3] and out["ok"]
    _rgb, depth, _p = scene.render()
    assert np.array_equal(asks.masks[0],
                          object_mask_from_depth(depth, scene.intrinsics, scene.goal_pose))
    plain = _CountingArm(tr.make_arm(scene, cfg))
    assert cb.identity_check(scene, plain)["ok"] and plain.arities == [2]


# ------------------------------------------------------------------ the chain
def _plugin_checkout(tmp_path, name="refine-vs-x"):
    """A stand-in for the external arm's checkout: a directory carrying the module's file."""
    root = tmp_path / name
    (root / "refinevs").mkdir(parents=True)
    (root / "refinevs" / "bench_plugin.py").write_text("ARM_IDS = ()\n")
    return root


def test_the_chain_passes_the_plugin_and_its_path_and_stamps_it(tmp_path):
    ti = _load("tests/fast/test_instance_suite.py", "_plugin_test_instance_suite")
    root = _plugin_checkout(tmp_path)
    argv = tmp_path / "argv.txt"
    rc, out, log = ti._run_chain(tmp_path, pins_ok=True, script="dev_suite.sh", env_extra={
        "FAKE_ARGV": str(argv), "ARMS": "clvs_v1", "TIER": "1a",
        "ARM_PLUGIN": "refinevs.bench_plugin", "REFINEVS_ROOT": str(root),
        "PAIRS": "inst_shoe:same"})
    assert rc == 0, log
    calls = argv.read_text().splitlines()
    assert len(calls) == 1 and calls[0].endswith("--arm-plugin refinevs.bench_plugin"), calls
    assert "--arms clvs_v1 --near-steps 200 --far 0 --far-set simple5" in calls[0]
    assert (out / "chain_settings.txt").read_text().strip() == (
        "far=0 arms=clvs_v1 extra=--telemetry --clean-film plugin=refinevs.bench_plugin "
        f"root={root}")
    # without the plugin: no flag, the old stamp (earlier OUT folders keep resuming)
    other = tmp_path / "plain"
    other.mkdir()
    argv2 = other / "argv.txt"
    rc, out2, log = ti._run_chain(other, pins_ok=True, script="dev_suite.sh", env_extra={
        "FAKE_ARGV": str(argv2), "ARMS": "artvs_roi_sim_dev", "TIER": "1a",
        "PAIRS": "inst_shoe:same"})
    assert rc == 0 and "--arm-plugin" not in argv2.read_text()
    assert (out2 / "chain_settings.txt").read_text().strip() == (
        "far=0 arms=artvs_roi_sim_dev extra=--telemetry --clean-film")


@pytest.mark.parametrize("root", [None, "missing"])
def test_the_chain_refuses_a_plugin_without_its_checkout(tmp_path, root):
    """REFINEVS_ROOT has no default (the station's /workspace/refine-vs is main, without the
    plugin), and a root that does not carry the module is refused - before anything flies."""
    ti = _load("tests/fast/test_instance_suite.py", "_plugin_test_instance_suite")
    env = {"ARMS": "clvs_v1", "TIER": "1a", "ARM_PLUGIN": "refinevs.bench_plugin",
           "REFINEVS_ROOT": None if root is None else str(tmp_path / root)}
    rc, out, log = ti._run_chain(tmp_path, pins_ok=True, script="dev_suite.sh", env_extra=env)
    assert rc == 1 and "refused" in log
    assert ("needs REFINEVS_ROOT" if root is None else "is not in REFINEVS_ROOT") in log
    assert not (tmp_path / "calls.txt").exists()


def test_the_chain_puts_refine_vs_on_the_path_only_with_a_plugin(tmp_path):
    ti = _load("tests/fast/test_instance_suite.py", "_plugin_test_instance_suite")
    root = _plugin_checkout(tmp_path)
    fake = tmp_path / "py_env.sh"
    fake.write_text(ti.FAKE_PY.replace(
        '[ -n "${FAKE_ARGV:-}" ] && echo "$*" >> "$FAKE_ARGV"',
        '[ -n "${FAKE_ARGV:-}" ] && echo "PYTHONPATH=$PYTHONPATH" >> "$FAKE_ARGV"'))
    fake.chmod(0o755)
    for plugin, expect in (("refinevs.bench_plugin", f"/workspace/isaac_site:{root}"),
                           ("", None)):
        d = tmp_path / (plugin or "none")
        d.mkdir()
        argv = d / "argv.txt"
        env = {"FAKE_ARGV": str(argv), "ARMS": "clvs_v1", "TIER": "1a",
               "REFINEVS_ROOT": str(root), "PAIRS": "inst_shoe:same", "ARM_PLUGIN": plugin}
        rc, _o, log = ti._run_chain(d, pins_ok=True, script="dev_suite.sh",
                                    env_extra={**env, "VSSIM_CHAIN_PY": str(fake)})
        assert rc == 0, log
        line = argv.read_text().strip()
        assert line == ("PYTHONPATH=" + (expect or "/workspace/isaac_site")), line


@pytest.mark.parametrize("arms", ["clvs:v1", "a/b", "ok bad:id"])
def test_the_chain_refuses_a_colon_or_slash_in_an_arm_id(tmp_path, arms):
    ti = _load("tests/fast/test_instance_suite.py", "_plugin_test_instance_suite")
    rc, out, log = ti._run_chain(tmp_path, pins_ok=True, script="dev_suite.sh",
                                 env_extra={"ARMS": arms, "TIER": "1a"})
    assert rc == 1 and "is not [A-Za-z0-9_]+" in log
    assert not (tmp_path / "calls.txt").exists()


def test_the_chain_flies_the_checkout_it_lives_in(tmp_path):
    """Launched from a copy (the station's /workspace/vs-sim-plugin), the chain cds into that
    copy - not into /workspace/vs-sim."""
    import os
    import shutil
    import subprocess

    ti = _load("tests/fast/test_instance_suite.py", "_plugin_test_instance_suite")
    copy = tmp_path / "vs-sim-copy"
    (copy / "scripts" / "chains").mkdir(parents=True)
    for f in ("instance_suite.sh", "dev_suite.sh"):
        shutil.copy(REPO / "scripts" / "chains" / f, copy / "scripts" / "chains" / f)
    fake = tmp_path / "fake_py.sh"
    fake.write_text(ti.FAKE_PY.replace('[ -n "${FAKE_ARGV:-}" ] && echo "$*" >> "$FAKE_ARGV"',
                                       '[ -n "${FAKE_ARGV:-}" ] && pwd >> "$FAKE_ARGV"'))
    fake.chmod(0o755)
    out = tmp_path / "out"
    argv = tmp_path / "pwd.txt"
    env = {k: v for k, v in os.environ.items()
           if k not in ("ARMS", "EXTRA", "FAR", "BLOCK", "TIER", "PAIRS", "VSSIM_REPO",
                        "ARM_PLUGIN", "REFINEVS_ROOT")}
    env.update({"VSSIM_CHAIN_PY": str(fake), "OUT": str(out), "FAKE_CALLS": str(tmp_path / "c"),
                "FAKE_PINS_OK": "1", "FAKE_ARGV": str(argv), "ARMS": "clvs_v1", "TIER": "1a",
                "PAIRS": "inst_shoe:same"})
    rc = subprocess.run(["bash", str(copy / "scripts" / "chains" / "dev_suite.sh")], env=env,
                        capture_output=True, text=True, timeout=60).returncode
    assert rc == 0
    assert argv.read_text().strip() == str(copy)
