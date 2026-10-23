"""U6b: the copied vs-sim arms as vsbench methods (``vsbench/methods/vssim_arms.py``).

``pytest tests/world/test_vssim_arm_methods.py --backends mujoco-egl -k vssim`` runs vsbench's
own Step-3 gates on each registered arm (``vitvs_dinov3``, ``vitvs_dinov2``, ``artvs``,
``sift_ransac``, ``orb_ransac``, ``akaze_ransac``), real weights on the CPU:

* the contract gate (``tests/test_contract.py::test_contract``, called as is);
* the leak gate: the import graph of ``vsbench/methods`` (static and in a fresh interpreter,
  ``tests/test_leak.py``), and a flight of each arm through vsbench's loop in which
  ``observe_truth`` never exists to be called;
* the zero-at-target and hold-at-target checks (``vsbench.checks``);
* run as published: the arm each method builds is the protocol cell's, label for label, with
  only ``matcher.device`` stated;
* the frame's camera: the arm solves with the 640 x 480 gate camera it is handed, and the
  wrapper's twist is the copied arm's own, to the bit;
* the recorded gates, read from the committed runs (skipped until flown):
  ``python -m vsbench.demo --method sift_ransac --max-steps 300`` -> start 1 under 3 mm;
  ``python -m vsbench.check --method <arm>`` -> ``results/bridge/u6b/check_<arm>.txt``
  (zero / hold gated for all, restoring for ``sift_ransac``, reported for the others);
  ``tools/u6_equivalence.py`` -> the copied ViT-VS arm direct vs wrapped, bit for bit.

With ``VSBENCH_WRITE_GATES=1`` the gates write ``results/bridge/u6b_gate.json``.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

from vsbench.methods.vssim_arms import ARM_IDS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
U6B = REPO / "results" / "bridge" / "u6b"
GATE_JSON = REPO / "results" / "bridge" / "u6b_gate.json"
GATES: dict = {}
VIT = ("vitvs_dinov3", "vitvs_dinov2", "artvs")

needs_gl = pytest.mark.skipif(os.environ.get("VSBENCH_SOFTWARE_GL", "").strip() != "1",
                              reason="renders: source tools/cpu_env.sh (software GL)")


@pytest.fixture(scope="module", autouse=True)
def _write_gates(tmp_path_factory):
    yield
    if not GATES:
        return
    out = (GATE_JSON if os.environ.get("VSBENCH_WRITE_GATES", "").strip() == "1"
           else tmp_path_factory.mktemp("gates") / GATE_JSON.name)
    old = json.loads(out.read_text()) if out.exists() else {}
    for k, v in GATES.items():
        old[k] = {**old.get(k, {}), **v} if isinstance(v, dict) else v
    old["unit"] = "U6b - the copied vs-sim arms as vsbench methods"
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True, default=str) + "\n")
    print(f"[gates] wrote {out}")


@pytest.fixture(scope="module")
def methods():
    """One instance per arm for the module (the encoder weights load once)."""
    from vsbench.methods import make_method

    cache: dict = {}

    def get(name):
        if name not in cache:
            cache[name] = make_method(name)
        return cache[name]

    yield get
    for m in cache.values():
        m.close()


# ------------------------------------------------------------------ as published
@pytest.mark.parametrize("name", ARM_IDS)
def test_vssim_arm_is_the_protocol_cell(name):
    """The method builds the protocol cell's arm: same resolved settings but the device,
    the same derived label."""
    from vsbench.camera import gate_camera
    from vsbench.methods import make_method
    from vsbench.vs.bench.arms import arm_label
    from vsbench.world import fly

    cam = gate_camera()
    m = make_method(name)
    try:
        m.configure(cam.K, cam.width, cam.height, 0.04)
        proto = fly.load_protocol()
        ref = fly.arm_config(proto, name, device="cpu")
        assert m.label == arm_label(ref) == arm_label(fly.arm_config(proto, name))
        strip = ("_source", "matcher")
        assert {k: v for k, v in m.config.items() if k not in strip} == \
            {k: v for k, v in ref.items() if k not in strip}
        assert {k: v for k, v in m.config["matcher"].items() if k != "device"} == \
            {k: v for k, v in fly.arm_config(proto, name)["matcher"].items() if k != "device"}
        assert m._arm.kind == "method" and m.privileged is False and m.wants_depth == "dense"
        if m.config["matcher"]["variant"].startswith("classical"):
            assert m._arm.matcher.ransac == "homography"      # the poster's model
        GATES.setdefault("as_published", {})[name] = {"label": m.label,
                                                      "device": m.config["matcher"].get("device")}
    finally:
        m.close()


def test_vssim_unavailable_says_why_and_is_none_here(monkeypatch, tmp_path):
    """vsbench's contract gate skips a copied arm only when it says why; in the bridge's
    environment (weights in the offline copy) none says anything, so no check is skipped."""
    import importlib.util

    from vsbench.methods import method_class
    from vsbench.methods import vssim_arms as va

    if not os.environ.get("HF_HOME") or not os.environ.get("TORCH_HOME"):
        pytest.skip("source tools/cpu_env.sh (HF_HOME / TORCH_HOME at the offline weights)")
    assert {n: method_class(n).unavailable() for n in ARM_IDS} == {n: None for n in ARM_IDS}
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    assert "weights" in va.unavailable_reason("vitvs_dinov3")
    assert "weights" in va.unavailable_reason("artvs")
    assert va.unavailable_reason("sift_ransac") is None
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "torch" else real(name, *a))
    assert all("torch" in va.unavailable_reason(n) for n in ARM_IDS)


@needs_gl
@pytest.mark.parametrize("name", ARM_IDS)
def test_vssim_wrapper_hands_the_arm_the_frames_camera(backend, renderer_factory, methods, name):
    """The intrinsics the arm solves with are the frame's own (vsbench.check's 640 x 480 gate
    camera: fx 600, fy 620, off-centre principal point), and the wrapper's twist is the copied
    arm's own on the same frames, to the bit."""
    from vsbench.camera import gate_camera
    from vsbench.episode import distance_prior
    from vsbench.scenes import poster_target_pose
    from vsbench.se3 import exp_se3
    from vsbench.vs.baselines.control import Intrinsics
    from vsbench.vs.bench.arms import build_arm

    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    rgb_t, d_t = r.render(T_t)
    rgb_0, d_0 = r.render(T_t @ exp_se3([0.06, 0.0, 0.0, 0.0, 0.0, 0.0]))
    m = methods(name)
    m.configure(cam.K.copy(), cam.width, cam.height, 0.04)
    m.reset()
    m.set_target(rgb_t, d_t, distance_prior(d_t))
    got = m._arm.controller.intrinsics
    want = (cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height)
    assert (got.fx, got.fy, got.cx, got.cy, got.width, got.height) == want
    tw, info = m.get_control_rate(rgb_0, d_0)
    classical = m.config["matcher"]["variant"].startswith("classical")
    direct = build_arm(m.config, intrinsics=Intrinsics(*want),
                       **({"scene_spec": {"ransac": "homography"}} if classical else {}))
    try:
        direct.set_episode("vsbench", 0)
        direct.reset(rgb_t, d_t, None)
        tw_d, _dbg, _m = direct.step(rgb_0, d_0)
    finally:
        direct.close()
    assert np.array_equal(tw, tw_d.as_array()), (tw, tw_d.as_array())
    pts = info["kp_cur"]
    assert len(pts) and pts[:, 0].min() >= 0 and pts[:, 0].max() < 640
    assert pts[:, 1].min() >= 0 and pts[:, 1].max() < 480
    GATES.setdefault("frame_camera", {})[name] = {
        "intrinsics": dict(zip(("fx", "fy", "cx", "cy", "width", "height"), map(float, want))),
        "twist_6cm_right": [round(float(x), 6) for x in tw], "equals_direct_arm": True}


# ------------------------------------------------------------------ vsbench's gates
@needs_gl
@pytest.mark.parametrize("name", ARM_IDS)
def test_vssim_contract(backend, renderer_factory, name):
    from test_contract import test_contract

    test_contract(backend, renderer_factory, name)
    GATES.setdefault("contract", {})[name] = "pass"


def test_vssim_leak_import_graph():
    from test_leak import test_import_graph_static, test_import_graph_transitive

    test_import_graph_static()
    test_import_graph_transitive()
    GATES["leak_import_graph"] = "pass (static + fresh interpreter)"


@needs_gl
@pytest.mark.parametrize("name", ARM_IDS)
def test_vssim_leak_no_truth_in_a_flight(backend, renderer_factory, methods, name):
    from vsbench.camera import gate_camera
    from vsbench.episode import EpisodeConfig, run_episode
    from vsbench.scenes import poster_target_pose
    from vsbench.se3 import exp_se3

    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    m = methods(name)
    assert m.privileged is False and not hasattr(m, "observe_truth")
    T_t = poster_target_pose(cam)
    log = run_episode(r, m, T_t @ exp_se3([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), T_t,
                      EpisodeConfig(max_steps=3, stop_tol=None), "poster")
    assert log.meta["status"] == "timeout" and len(log) == 3 and log.meta["privileged"] is False
    for key, value in vars(m).items():                   # nothing the harness owns
        assert not any(hasattr(value, a) for a in ("render", "goal_pose", "observe_truth")), key
    GATES.setdefault("leak_flight", {})[name] = "pass (3 steps, never privileged)"


@needs_gl
@pytest.mark.parametrize("name", ARM_IDS)
def test_vssim_zero_and_hold_at_target(backend, renderer_factory, methods, name):
    from vsbench.camera import gate_camera
    from vsbench.checks import hold_at_target, zero_at_target
    from vsbench.scenes import poster_target_pose

    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    m = methods(name)
    T_t = poster_target_pose(cam)
    z = zero_at_target(r, m, T_t)
    h = hold_at_target(r, m, T_t)
    print(f"{backend} {name}: zero {z.detail}; hold {h.detail}")
    GATES.setdefault("zero_at_target", {})[name] = z.numbers | {"passed": z.passed}
    GATES.setdefault("hold_at_target", {})[name] = h.numbers | {"passed": h.passed}
    assert z.passed and h.passed


# ------------------------------------------------------------------ the recorded gates
def test_vssim_demo_start1_gate():
    path = U6B / "demo" / "sift_ransac" / "demo.json"
    if not path.exists():
        pytest.skip(f"{path} not run yet (see the module docstring)")
    rec = json.loads(path.read_text())
    row = rec["rows"][0]
    assert rec["method"] == "sift_ransac" and row["start"] == 1
    assert rec["config"]["max_steps"] == 300 and row["steps"] <= 300
    assert row["final_error_mm"] < 3.0
    GATES["demo_sift_ransac"] = {
        "start_1": {k: row[k] for k in ("status", "steps", "final_error_mm", "final_error_deg")},
        "all_starts": [{k: x[k] for k in ("start", "status", "steps", "final_error_mm",
                                          "final_error_deg")} for x in rec["rows"]],
        "converged": rec["converged"], "backend": rec["backend"],
        "gl_renderer": (rec.get("renderer") or {}).get("gl_renderer"),
        "gate": "start 1 under 3 mm within 300 steps"}


def parse_check(text: str) -> dict:
    """``python -m vsbench.check``'s report -> per check pass/fail, the axes, the budget."""
    out = {"budget": None, "checks": {}, "axes": {}, "all": None}
    m = re.search(r"restoring budget (\d+) steps", text)
    out["budget"] = int(m.group(1)) if m else None
    for name in ("zero_at_target", "hold_at_target", "restoring_direction"):
        m = re.search(rf"^\s+{name}\s+(pass|FAIL)\s+(.*)$", text, re.M)
        if m:
            out["checks"][name] = {"passed": m.group(1) == "pass", "detail": m.group(2).strip()}
    m = re.search(r"per axis: (.*)$", text, re.M)
    if m:
        out["axes"] = dict(re.findall(r"(\w+) (pass|FAIL)", m.group(1)))
    m = re.search(r"all: (pass|FAIL)", text)
    out["all"] = None if not m else m.group(1) == "pass"
    return out


#: The arm whose ``restoring_direction`` is gated (it is the demo gate's arm); for the other
#: five it is REPORTED: on vsbench's 640 x 480 poster from +-2 cm / +-2 deg the ORB / AKAZE
#: keypoints and the ViT token grids stall above the check's 2 mm / 0.2 deg hold (every ORB
#: first twist has the right sign), and the U6 equivalence gate shows the adapter flies the
#: copied ViT-VS arm bit for bit - method behaviour at this resolution, no threshold moved
#: (results/bridge/u6b/README.md).
RESTORING_GATED = ("sift_ransac",)


@pytest.mark.parametrize("name", ARM_IDS)
def test_vssim_check_cli_gate(name):
    """``results/bridge/u6b/check_<arm>.txt`` (the CLI's report, llvmpipe): zero and hold at
    target gated for every arm, ``restoring_direction`` for :data:`RESTORING_GATED`."""
    path = U6B / f"check_{name}.txt"
    if not path.exists() or "all:" not in path.read_text():
        pytest.skip(f"{path} not run yet")
    text = path.read_text()
    assert "llvmpipe" in text, "the check must have run on Mesa llvmpipe"
    res = parse_check(text)
    res["restoring_gated"] = name in RESTORING_GATED
    GATES.setdefault("check_cli", {})[name] = res
    assert res["checks"]["zero_at_target"]["passed"]
    assert res["checks"]["hold_at_target"]["passed"]
    if name in RESTORING_GATED:
        assert res["budget"] == 3000
        assert res["checks"]["restoring_direction"]["passed"] and res["all"] is True


def test_parse_check_reads_the_report():
    text = ("backend mujoco-egl (llvmpipe), scene poster, method x {}, restoring budget 300 steps\n"
            "setup checks for x:\n  zero_at_target       pass  |v| 0\n"
            "  hold_at_target       pass  drift 0\n"
            "  restoring_direction  FAIL  3 of 12 starts right\n"
            "    per axis: vx FAIL  vy pass  vz pass  wx pass  wy FAIL  wz pass\n  all: FAIL\n")
    r = parse_check(text)
    assert r["budget"] == 300 and r["all"] is False and r["axes"]["vx"] == "FAIL"
    assert r["checks"]["zero_at_target"]["passed"] and not r["checks"]["restoring_direction"]["passed"]
    assert np.isclose(len(r["axes"]), 6)


# ------------------------------------------------------------------ equivalence (U6)
def test_vssim_equivalence_gate():
    """The copied ViT-VS arm flown directly vs through the U6b wrapper inside the U6a
    ``MethodArm`` on the copied loop (MuJoCo D435i, hollywood near60 n_000, 100 steps):
    ``tools/u6_equivalence.py`` -> ``results/bridge/u6_equivalence.json``."""
    path = REPO / "results" / "bridge" / "u6_equivalence.json"
    if not path.exists():
        pytest.skip(f"{path} not run yet (tools/u6_equivalence.py)")
    eq = json.loads(path.read_text())
    assert eq["render_mode"] == "mujoco-llvmpipe" and "llvmpipe" in eq["gl_renderer"]
    assert eq["n_compared"] == eq["steps"] == 100
    assert eq["path_a"]["controller_intrinsics"] == eq["path_b"]["controller_intrinsics"] \
        == eq["scene_reported_intrinsics"]
    assert eq["twist_max_abs_diff"] == 0.0 and eq["steps_bit_identical"] == 100
    assert eq["path_a"]["final_translation_mm"] == eq["path_b"]["final_translation_mm"]
    assert eq["path_b"]["arm_kind"] == "external" and eq["path_b"]["inner_label"] == eq["path_a"]["label"]
    assert abs(eq["path_a"]["final_translation_mm"] - eq["m1_record"]["final_translation_mm"]) < 1e-9
    GATES["equivalence"] = {k: v for k, v in eq.items() if k != "twist_max_abs_diff_per_step"}
