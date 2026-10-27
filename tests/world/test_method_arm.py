"""``MethodArm`` - a vsbench method flown by the copied benchmark loop.

* the config: the template validates under the shim, and the copied ``config.validate``
  would refuse it (why the shim exists); a broken field is refused naming it;
* refusals: a privileged method, a method holding ground truth, ``--method`` naming a protocol
  arm or a truth instrument;
* the GT-blind tripwire: the same flight twice, the second with every ground-truth read the
  loop makes poisoned - the pose ``render`` returns and ``scene.pose``, ``gt_correspondence``,
  the size-matched and the depth-true reference poses (``episode.depth_true`` on, extended
  telemetry on) - and the method's commands are the same to the last bit, while the poison
  visibly reached the record;
* the stop: ``cns_plateau`` when the method reports ``kp_cur`` / ``kp_des``, else ``cap_only``
  - loudly (a warning, a stderr line, ``stop_mode``, the label, ``stop_fallback``);
* vs-sim's plugin hook: the copied ``check_baselines`` loads this module as an
  ``--arm-plugin`` and its refusals pass (``external:`` label, kind ``external``);
* the driver: ``python -m vsbench.world.fly run --method sift_ibvs`` writes a record that
  passes the copied frozen-schema tests with ``arm_kind: external``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from vsbench.methods.feature_ibvs import SiftIBVS  # noqa: E402
from vsbench.world import method_arm as ma  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
#: the scenes whose assets this module needs (tests/conftest.py skips it without them)
NEEDS_SCENES = ("hollywood",)
#: numbers the tests measured, printed with -s (not written anywhere)
GATES: dict = {}


def _fly_smoke_helpers():
    """``assert_schema_valid`` / ``assert_provenance`` of ``test_fly_smoke.py`` (the copied
    frozen-schema checks)."""
    spec = importlib.util.spec_from_file_location("_fly_smoke_helpers", REPO / "tests" / "world"
                                                  / "test_fly_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- the worlds
def hollywood_mock(render_scale: float = 0.25):
    """The copied mock plate with the hollywood texture: GL-free, deterministic."""
    from vsbench.vs.scene import MockPlaneScene
    from vsbench.vs.scenes import load_scenes
    from vsbench.world.build import scene_spec_for

    specs, _ = load_scenes()
    spec, _meta = scene_spec_for(specs["hollywood"])
    tex = np.array(spec.objects[0].material.texture)
    return MockPlaneScene(name="hollywood_mock", texture=tex, render_scale=render_scale)


def near_start(i: int = 0) -> dict:
    from vsbench.vs import benchmark

    return benchmark.select_starts("hollywood", "near60", limit=i + 1)[i]


class RecordingSift(SiftIBVS):
    """vsbench's ``sift_ibvs``, keeping a copy of every twist it returns and how it was fed."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.twists: list[np.ndarray] = []
        self.writable: list[bool] = []
        self.truth_calls = 0

    def get_control_rate(self, current_rgb, current_depth=None):
        self.writable.append(bool(current_rgb.flags.writeable))
        out = super().get_control_rate(current_rgb, current_depth)
        self.twists.append(np.array(out[0], dtype=np.float64, copy=True))
        return out

    def observe_truth(self, T_wc, T_wc_target):     # never privileged; must never be called
        self.truth_calls += 1


class NoPoints:
    """A method that reports no points: a fixed small twist, info without kp_cur/kp_des."""

    name = "probe_no_points"
    wants_depth = "none"
    privileged = False

    def __init__(self, declare=None):
        self.params = {"declare": declare}
        if declare is not None:
            self.reports_points = declare

    def configure(self, K, width, height, dt):
        pass

    def reset(self):
        pass

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None):
        pass

    def get_control_rate(self, current_rgb, current_depth=None):
        return np.zeros(6), {"error_norm": 1.0, "lost": False}


class Poisoned:
    """A scene whose every ground-truth read is garbage; the frames are the true ones."""

    def __init__(self, inner, seed=7):
        from vsbench.vs.scene import Pose

        self.inner = inner
        self._rng = np.random.default_rng(seed)
        self._Pose = Pose
        self.name = inner.name
        self.gt_calls = 0

    def _garbage(self):
        q, _ = np.linalg.qr(self._rng.normal(size=(3, 3)))
        return self._Pose(position=self._rng.normal(size=3), rotation=q * np.sign(np.linalg.det(q)))

    intrinsics = property(lambda self: self.inner.intrinsics)
    working_distance = property(lambda self: self.inner.working_distance)
    goal_pose = property(lambda self: self.inner.goal_pose)      # the goal PICTURE needs it
    pose = property(lambda self: self._garbage())

    def reset(self, pose):
        self.inner.reset(pose)

    def apply(self, twist, dt):
        self.inner.apply(twist, dt)

    def render(self):
        rgb, depth, _true = self.inner.render()
        return rgb, depth, self._garbage()

    def gt_correspondence(self, pts_goal_px, cur_pose, goal_pose=None):
        self.gt_calls += 1
        n = len(pts_goal_px)
        return self._rng.uniform(0, 1000, size=(n, 2)), self._rng.random(n) > 0.5


def _poison_perturbation(seed=11):
    from vsbench.vs.bench.perturb import Perturbation
    from vsbench.vs.scene import Pose

    rng = np.random.default_rng(seed)

    class PoisonedReferences(Perturbation):
        def reference_pose(self):
            return Pose(position=rng.normal(size=3), rotation=np.eye(3))

        def depth_true_pose(self):
            return Pose(position=rng.normal(size=3), rotation=np.eye(3))

    return PoisonedReferences()


def _cfg(max_iter=6, **episode):
    cfg = ma.external_config("sift_ibvs", max_iter=max_iter)
    cfg["episode"].update(episode)
    ma.validate_external(cfg)
    return cfg


# ----------------------------------------------------------------- the config
def test_the_template_validates_here_and_the_copied_validator_refuses_it():
    from vsbench.vs.bench import config as cfgmod

    cfg = ma.external_config("sift_ibvs", {"gain": 0.8}, max_iter=50)
    assert cfg["_schema"] == ma.SCHEMA and cfg["matcher"]["variant"] == "external"
    assert cfg["arm"]["kind"] == "external" and cfg["stop"]["rule"] == "cns_plateau"
    assert cfg["episode"]["max_iter"] == 50 and cfg["method"] == {"name": "sift_ibvs",
                                                                  "params": {"gain": 0.8}}
    with pytest.raises(cfgmod.ConfigError, match="matcher.variant"):
        cfgmod.validate(cfg)                  # the copied enum has no `external`: hence the shim
    for path, bad, match in ((("matcher", "variant"), "classical_sift", "matcher.variant"),
                             (("arm", "kind"), "method", "arm.kind"),
                             (("seed",), None, "seed"),
                             (("stop", "rule"), "dinobot_err", "stop.rule"),
                             (("controller", "dt"), 0.0, "controller.dt"),
                             (("method", "name"), None, "method.name"),
                             (("matcher", "num_landmarks"), -1, "num_landmarks")):
        broken = json.loads(json.dumps(cfg))
        node = broken
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = bad
        with pytest.raises(ma.ExternalConfigError, match=match):
            ma.validate_external(broken)
    broken = json.loads(json.dumps(cfg))
    broken["episode"]["no_such_key"] = 1
    with pytest.raises(ValueError, match="unknown episode keys"):
        ma.validate_external(broken)
    broken = json.loads(json.dumps(cfg))
    broken["refine"] = {"x": 1}
    with pytest.raises(ma.ExternalConfigError, match="refine"):
        ma.validate_external(broken)


def test_the_label_and_the_kind_say_external():
    scene = hollywood_mock(0.125)
    arm, cfg = ma.build_method_arm("sift_ibvs", intrinsics=scene.intrinsics)
    assert arm.kind == "external" and arm.name.startswith("external:sift_ibvs@")
    assert arm.name.endswith("/cns_plateau") and arm.stop.name == "cns_plateau"
    assert abs(arm.dt - 1 / 60) < 1e-15 and arm.gain == 1.0
    other, _ = ma.build_method_arm("sift_ibvs", {"gain": 0.5}, intrinsics=scene.intrinsics)
    assert other.name != arm.name and other.gain == 0.5     # one label = one setting


# ----------------------------------------------------------------- refusals
class _Privileged(SiftIBVS):
    """A truth instrument: it says it reads the true pose."""

    privileged = True

    def observe_truth(self, T_wc, T_wc_target):
        pass


@pytest.fixture
def privileged_method():
    from vsbench.methods import _REGISTRY, register

    register("test_privileged")(_Privileged)
    yield "test_privileged"
    _REGISTRY.pop("test_privileged", None)


def test_a_privileged_or_truth_holding_method_is_refused(privileged_method):
    from vsbench.methods import make_method
    from vsbench.vs.scene import Pose

    scene = hollywood_mock(0.125)
    cfg = ma.external_config(privileged_method)
    with pytest.raises(ValueError, match="privileged"):
        ma.MethodArm(make_method(privileged_method), cfg, intrinsics=scene.intrinsics)
    m = SiftIBVS()
    m.peek = Pose(position=np.zeros(3), rotation=np.eye(3))
    with pytest.raises(AssertionError, match="ground truth"):
        ma.MethodArm(m, _cfg(), intrinsics=scene.intrinsics)
    m = SiftIBVS()
    m.world = scene                                            # an object that renders
    with pytest.raises(AssertionError, match="ground truth"):
        ma.MethodArm(m, _cfg(), intrinsics=scene.intrinsics)
    with pytest.raises(ValueError, match="names method"):
        ma.MethodArm(SiftIBVS(), ma.external_config("orb_ibvs"), intrinsics=scene.intrinsics)


def test_fly_refuses_protocol_arms_and_truth_instruments(privileged_method):
    from vsbench.world import fly

    proto = fly.load_protocol()
    fly.check_methods(proto, ["sift_ibvs", "orb_ibvs"])
    with pytest.raises(SystemExit, match="protocol arm"):
        fly.check_methods(proto, ["vitvs_dinov3"])
    with pytest.raises(SystemExit, match="privileged"):
        fly.check_methods(proto, [privileged_method])
    with pytest.raises(SystemExit, match="not a registered"):
        fly.check_methods(proto, ["no_such_method"])


# ----------------------------------------------------------------- GT-blind
def _flight(scene, perturbation, cfg):
    from vsbench.vs.bench.runner import run_episode

    method = RecordingSift()
    arm = ma.MethodArm(method, cfg, intrinsics=scene.intrinsics)
    tel: list = []
    rec = run_episode(scene, arm, near_start(0), cfg, telemetry=tel, perturbation=perturbation)
    return rec, tel, method


def test_the_method_never_sees_ground_truth():
    """Poison every truth read of the loop; the commands must not move by one bit."""
    cfg = _cfg(max_iter=8, telemetry_extended=True, depth_true=True)
    from vsbench.vs.bench.perturb import Perturbation

    clean_rec, clean_tel, clean = _flight(hollywood_mock(), Perturbation(), cfg)
    poisoned_scene = Poisoned(hollywood_mock())
    pois_rec, pois_tel, pois = _flight(poisoned_scene, _poison_perturbation(), cfg)

    assert len(clean.twists) == len(pois.twists) == 8
    same = all(np.array_equal(a, b) for a, b in zip(clean.twists, pois.twists, strict=True))
    assert same, "the method's commands changed when only ground truth was poisoned"
    assert clean.truth_calls == pois.truth_calls == 0
    assert not any(clean.writable + pois.writable), "the method was handed a writable frame"
    # ... and the poison was live: the loop read it, the record and the telemetry carry it
    assert poisoned_scene.gt_calls == 8
    assert "depth_true_pose" in pois_rec and "reference_pose" in pois_rec
    assert "depth_true_pose" not in clean_rec
    assert pois_rec["final_translation_mm"] != clean_rec["final_translation_mm"]
    assert [s["t_mm"] for s in pois_tel] != [s["t_mm"] for s in clean_tel]
    assert [s["gt_u"] for s in pois_tel] != [s["gt_u"] for s in clean_tel]
    assert [s["v"] for s in pois_tel] == [s["v"] for s in clean_tel]
    GATES["gt_blind_tripwire"] = {
        "steps": 8, "world": "copied MockPlaneScene, hollywood texture, render_scale 0.25",
        "poisoned": ["render() pose", "scene.pose", "gt_correspondence",
                     "perturbation.reference_pose", "perturbation.depth_true_pose"],
        "commands_bit_identical": bool(same), "observe_truth_calls": 0,
        "frames_read_only": True, "gt_correspondence_calls_poisoned": poisoned_scene.gt_calls,
        "final_translation_mm_clean": clean_rec["final_translation_mm"],
        "final_translation_mm_poisoned_record": pois_rec["final_translation_mm"]}


# ----------------------------------------------------------------- the stop
def test_points_give_cns_plateau(tmp_path):
    scene = hollywood_mock()
    from vsbench.vs.bench.runner import run_episode

    with warnings.catch_warnings():
        warnings.simplefilter("error", ma.StopFallbackWarning)
        arm, cfg = ma.build_method_arm("sift_ibvs", intrinsics=scene.intrinsics, max_iter=4)
        rec = run_episode(scene, arm, near_start(0), cfg)
    assert rec["stop_mode"] == "cns_plateau" and rec["stop_fallback"] is None
    assert rec["stop_rule_configured"] == "cns_plateau" and rec["arm"].endswith("/cns_plateau")
    assert rec["arm_kind"] == "external" and rec["matcher_variant"] == "external"
    assert rec["mean_n_matches"] > 20
    _fly_smoke_helpers().assert_schema_valid(rec)


@pytest.mark.parametrize("declare", [None, False])
def test_no_points_fall_back_to_cap_only_loudly(capfd, declare):
    from vsbench.vs.bench.runner import run_episode

    scene = hollywood_mock(0.125)
    cfg = ma.external_config("probe_no_points", max_iter=5)
    with pytest.warns(ma.StopFallbackWarning, match="cap_only"):
        arm = ma.MethodArm(NoPoints(declare), cfg, intrinsics=scene.intrinsics)
        rec = run_episode(scene, arm, near_start(0), cfg)
    err = capfd.readouterr().err
    assert "STOP FALLBACK" in err and "cns_plateau" in err
    assert rec["stop_mode"] == "cap_only" and rec["arm"].endswith("/cap_only")
    assert rec["stop_rule_configured"] == "cns_plateau"
    fb = rec["stop_fallback"]
    assert fb["configured"] == "cns_plateau" and fb["ran"] == "cap_only" and fb["reason"]
    assert rec["stop_step"] is None and rec["steps"] == 5
    GATES.setdefault("cap_only_fallback", {})[f"declare={declare}"] = {
        "stop_mode": rec["stop_mode"], "label": rec["arm"], "stop_fallback": fb,
        "stderr_line": [ln for ln in err.splitlines() if "STOP FALLBACK" in ln][0]}


# ----------------------------------------------------------------- the plugin hook
def test_vssim_check_baselines_takes_this_module_as_an_arm_plugin():
    """The copied ``check_baselines`` (vs-sim's external-arm hook) loads ``method_arm`` as
    ``--arm-plugin`` and its refusals pass - in a child process (the copied script puts
    ``vsbench/`` and ``vsbench/scripts`` on ``sys.path``)."""
    code = (
        "import importlib.util, json, sys\n"
        f"p = {str(REPO / 'vsbench' / 'scripts' / 'check_baselines.py')!r}\n"
        "spec = importlib.util.spec_from_file_location('_cb', p)\n"
        "cb = importlib.util.module_from_spec(spec); sys.modules['_cb'] = cb\n"
        "spec.loader.exec_module(cb)\n"
        "plug = cb.load_arm_plugin('vsbench.world.method_arm')\n"
        "proto = cb.fly.load_protocol()\n"
        "src = cb.resolve_arm_sources(proto, ['sift_ransac', 'sift_ibvs'], plug)\n"
        "from vsbench.world.camera import d435i_intrinsics\n"
        "arm, cfg, desc = cb.build_plugin_arm(plug, 'sift_ibvs', intrinsics=d435i_intrinsics(),"
        " scene_name='hollywood')\n"
        "print('OUT ' + json.dumps({'ids': list(plug.ARM_IDS), 'src': src, 'name': arm.name,"
        " 'kind': arm.kind, 'desc': desc}))\n")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("OUT ")][-1][4:])
    assert "sift_ibvs" in out["ids"] and "vitvs_dinov3" not in out["ids"]
    assert "oracle_pbvs" not in out["ids"] and "true_point_ibvs" not in out["ids"]
    assert out["src"] == {"sift_ransac": "protocol", "sift_ibvs": "plugin"}
    assert out["name"].startswith("external:") and out["kind"] == "external"


# ----------------------------------------------------------------- the driver
def test_driver_method_smoke(tmp_path):
    env = {**os.environ, "VSBENCH_STATION": "1", "CUDA_VISIBLE_DEVICES": "",
           "VSBENCH_SOFTWARE_GL": "1"}
    cmd = [sys.executable, "-m", "vsbench.world.fly", "run", "--scene", "hollywood",
           "--method", "sift_ibvs", "--start-set", "near60", "--limit", "1",
           "--max-iter", "3", "--out", str(tmp_path)]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    rec = json.loads((tmp_path / "hollywood_external_sift_ibvs.jsonl").read_text().splitlines()[0])
    assert rec["steps"] == 3 and rec["arm_kind"] == "external" and rec["first_frame_n_matches"] >= 20
    h = _fly_smoke_helpers()
    h.assert_schema_valid(rec)
    h.assert_provenance(rec)
    run_json = json.loads(next((tmp_path / "hollywood_external_sift_ibvs").glob("*/run.json"))
                          .read_text())
    assert run_json["config_schema"] == ma.SCHEMA and run_json["render_mode"] == "mujoco-llvmpipe"
    assert run_json["scene_meta"]["renderer"]["gl_renderer"].startswith("llvmpipe")
    assert rec["device"] is None and rec["station_mode"] is True
    for bad, why in ((["--method", "vitvs_dinov3"], "protocol arm"),
                     (["--method", "no_such_method"], "not a registered")):
        q = subprocess.run([sys.executable, "-m", "vsbench.world.fly", "run", "--scene",
                            "hollywood", *bad, "--start-set", "near60", "--limit", "1",
                            "--out", str(tmp_path / "refused")], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=600)
        assert q.returncode != 0 and why in q.stderr, q.stderr[-2000:]
