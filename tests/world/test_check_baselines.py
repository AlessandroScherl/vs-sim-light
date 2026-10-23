"""U9 (docs/PORT_PLAN.md section 7): ``python -m vsbench.world.check_baselines``, the twin of
vs-sim's Isaac rig check on the MuJoCo world.

Two kinds of test:

* **the driver, fast** (no rendering, no weights): it flies the COPIED check (the reference
  module's own functions), it offers every flag of vs-sim's parser, it refuses what vs-sim
  refuses, the near verdict, the summary's merge and its refusal of mixed renderers, and one
  whole run on the copied numpy mock (a stand-in for ``MujocoScene``) with a truth arm
  through the plugin hook - check.json in vs-sim's shape plus the bridge's fields, check.md,
  records.jsonl;
* **the committed flight** (``results/bridge/u9/``; skipped until flown): the U9 matrix
  ``{sift_ransac, orb_ransac, vitvs_dinov3, artvs} x six scenes`` is complete, every scene and
  entry drew on Mesa llvmpipe with torch on 2 / 2 threads and no GPU, the records agree with
  check.json, the identity row is all green, the near verdicts are what the rule gives. The
  gate JSON ``results/bridge/u9_check.json`` is written only with ``VSBENCH_WRITE_GATES=1``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from vsbench.world import check_baselines as cb

REPO = Path(__file__).resolve().parents[2]
U9 = REPO / "results" / "bridge" / "u9"
GATE = REPO / "results" / "bridge" / "u9_check.json"


def write_gates() -> bool:
    return os.environ.get("VSBENCH_WRITE_GATES", "").strip() == "1"


# ------------------------------------------------------------------ the driver
def test_the_check_is_the_copied_one_and_the_load_leaves_sys_path_alone():
    before = list(sys.path)
    ref = cb.reference()
    assert sys.path == before
    assert Path(ref.__file__) == cb.REFERENCE
    for fn in ("identity_check", "episode_check", "near_start", "write_table",
               "resolve_arm_sources", "build_plugin_arm", "load_arm_plugin", "parse_budgets"):
        assert getattr(ref, fn).__code__.co_filename == str(cb.REFERENCE), fn
    assert ref.IDENTITY_V_MAX == 1e-4
    assert np.isclose(np.degrees(ref.IDENTITY_W_MAX), 0.01)
    assert cb.reference() is ref                       # loaded once


def test_every_flag_of_vs_sims_parser_is_offered_with_its_default():
    src = cb.REFERENCE.read_text()
    flags = re.findall(r'ap\.add_argument\("(--[a-z-]+)"', src)
    assert len(flags) >= 14
    ours = {a for act in cb.build_parser()._actions for a in act.option_strings}
    assert set(flags) <= ours, sorted(set(flags) - ours)
    d = cb.build_parser().parse_args(["--out", "x"])
    assert (d.near_steps, d.far, d.far_set, d.perturb, d.perturb_seed, d.film_every) == \
        (200, 0, "capped30", "none", 20260918, 1)
    assert d.arms is None                               # vs-sim: every protocol arm
    assert d.device == "cpu" and d.frames == "full" and not d.allow_gpu


def test_vs_sims_refusals_come_first():
    from vsbench.world import fly

    ref, protocol = cb.reference(), fly.load_protocol()

    def args(*a):
        return cb.build_parser().parse_args(["--out", "x", *a])

    with pytest.raises(SystemExit, match="needs --telemetry"):
        cb.validate_args(args("--clean-film"), ref, protocol)
    with pytest.raises(SystemExit, match="not flown"):
        cb.validate_args(args("--arms", "sift_ransac", "--budget", "artvs=5"), ref, protocol)
    with pytest.raises(SystemExit, match="unknown arm"):
        cb.validate_args(args("--arms", "no_such_arm"), ref, protocol)
    with pytest.raises(SystemExit, match="far-set"):
        cb.validate_args(args("--far-set", "nope"), ref, protocol)
    ids, plugin, src, budgets = cb.validate_args(
        args("--arms", "sift_ransac", "artvs", "--budget", "artvs=5"), ref, protocol)
    assert ids == ["sift_ransac", "artvs"] and plugin is None and budgets == {"artvs": 5}
    assert src == {"sift_ransac": "protocol", "artvs": "protocol"}


def test_the_near_verdict_reads_closer_than_it_started():
    assert cb.near_verdict({"start_mm": 7.76, "final_mm": 0.2}) == \
        {"near_ok": True, "final_over_start": round(0.2 / 7.76, 4)}
    assert cb.near_verdict({"start_mm": 7.76, "final_mm": 7.76})["near_ok"] is False
    assert cb.near_verdict({"start_mm": 7.76, "final_mm": 43.7})["near_ok"] is False
    assert cb.near_verdict({"start_mm": 7.76, "final_mm": float("nan")})["near_ok"] is False
    assert cb.near_verdict(None) == {"near_ok": False, "final_over_start": None}


def _check_doc(mode="mujoco-llvmpipe", arms=("sift_ransac",), scene="hollywood", near_mm=0.3):
    entry = {"label": "x", "render_mode": mode,
             "identity": {"ok": True, "n_matched": 24, "v_norm_m_s": 0.0, "w_norm_deg_s": 0.0},
             "near": {"start_id": "n_000", "start_mm": 7.76, "start_deg": 2.69,
                      "final_mm": near_mm, "final_deg": 0.1, "steps": 200,
                      "terminated": "max_iter"},
             "far": [], **cb.near_verdict({"start_mm": 7.76, "final_mm": near_mm})}
    return {"scenes": {scene: {"render_mode": mode, "arms": {a: dict(entry) for a in arms}}},
            "arms": list(arms), "near_steps": 200, "far": 0}


def test_the_summary_merges_runs_and_refuses_mixed_renderers(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    (a / "check.json").write_text(json.dumps(_check_doc(arms=cb.U9_ARMS)))
    (b / "check.json").write_text(json.dumps(_check_doc(arms=cb.U9_ARMS, scene="poster2",
                                                        near_mm=9.0)))
    merged = cb.load_checks([a, b])
    s = cb.summarize(merged, {})
    assert s["n_flown"] == 8 and not s["complete"] and not s["near_all_green"]
    assert s["near_fail"] == [f"poster2/{x}" for x in cb.U9_ARMS]
    assert "| hollywood | sift_ransac | OK: 24" in cb.table_md(s)
    (c / "check.json").write_text(json.dumps(_check_doc(mode="isaac-rtx", scene="poster3")))
    with pytest.raises(SystemExit, match="mixed renderers"):
        cb.load_checks([a, c])
    with pytest.raises(SystemExit, match="flown twice"):
        cb.load_checks([a, a / "check.json"])


def test_isaac_context_is_read_from_its_own_runs(tmp_path):
    near = {"scenes": {"hollywood": {"arms": {"artvs": {
        "identity": {"ok": True, "n_matched": 24, "v_norm_m_s": 0.0, "w_norm_deg_s": 0.0},
        "near": {"start_mm": 7.76, "final_mm": 1.6, "final_deg": 0.02, "steps": 200}}}}},
        "started_utc": "t1"}
    far = {"scenes": {"hollywood": {"arms": {"artvs": {"far": [
        {"start_id": "w1_000", "start_mm": 300.0, "final_mm": 2.0, "final_deg": 0.1,
         "steps": 600}]}}}}, "started_utc": "t2"}
    for run, doc in ((cb.ISAAC_NEAR_RUN, near), (cb.ISAAC_FAR_RUNS[0], far)):
        (tmp_path / run).mkdir()
        (tmp_path / run / "check.json").write_text(json.dumps(doc))
    ctx = cb.isaac_context(tmp_path)[("hollywood", "artvs")]
    assert ctx["near_ok"] and ctx["near_final_mm"] == 1.6 and ctx["far_final_mm"] == 2.0
    assert ctx["far_start_id"] == "w1_000"
    assert cb.isaac_context(tmp_path / "missing") == {}


class _MockWorld:
    """The copied numpy mock dressed as ``MujocoScene`` for the driver: the goal capture,
    its record, ``render_mode`` and a context manager. Its goal is the posters' goal, so the
    frozen hollywood start file's goal check passes on it."""

    def __init__(self, name):
        from vsbench.vs.scene import MockPlaneScene

        self._m = MockPlaneScene(render_scale=0.125, name=name)
        self.name = name
        self.render_mode = "mujoco-llvmpipe"
        self.renderer = types.SimpleNamespace(gl_renderer="llvmpipe (stand-in)")
        self.goal_frame_md5 = None
        self.goal_settle = None
        self._goal = None

    def __getattr__(self, k):
        return getattr(self._m, k)

    def render_goal(self):
        if self._goal is None:
            keep = self._m.pose
            self._m.reset(self._m.goal_pose)
            rgb, depth, _ = self._m.render()
            self._m.reset(keep)
            self._goal = (rgb, depth)
            self.goal_frame_md5 = cb._md5(rgb)
            self.goal_settle = {"diff_prev": [None]}
        return self._goal

    def goal_capture_record(self):
        return {"look": "stand-in", "look_curve_id": None, "look_curve_md5": None}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _truth_plugin():
    """A plugin arm: the copied truth arm (``PlaneTruthMatcher``) under an external name."""
    from vsbench.vs.bench import arms
    from vsbench.vs.bench import config as cfgmod
    from vsbench.vs.bench.truth import PlaneTruthMatcher

    mod = types.ModuleType("u9_truth_plugin")
    mod.ARM_IDS = ("truth_arm",)

    class Named:
        name, kind = "external:truth", "external"

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, k):
            return getattr(self._inner, k)

    def build(arm_id, *, intrinsics, scene_name, scene_spec=None):
        cfg = cfgmod.resolve({"seed": 20260902, "arm": {"kind": "oracle"},
                              "matcher": {"variant": "mutual_nn_seeded_subsample",
                                          "num_landmarks": 24, "device": "cpu"},
                              "controller": {"gain": 2.0}, "stop": {"rule": "cns_plateau"},
                              "episode": {"max_iter": 40},
                              "scoring": {"success_rule": ["predicate_a"]}})
        # the truth reads the flown world's pose (the mock the driver opened)
        matcher = PlaneTruthMatcher(mod.world._m, num_landmarks=24, seed=20260902)
        return Named(arms.build_arm(cfg, intrinsics=intrinsics, matcher=matcher)), cfg

    mod.build = build
    mod.describe = lambda arm_id: {"module": "u9_truth_plugin", "commit": "test",
                                   "config_source": "inline", "label": "external:truth"}
    return mod


def test_a_whole_check_on_the_copied_mock(tmp_path, monkeypatch):
    from vsbench.world import fly

    plugin = _truth_plugin()
    monkeypatch.setitem(sys.modules, "u9_truth_plugin", plugin)
    monkeypatch.setattr(fly, "station_guard", lambda *a, **k: {"torch_threads": 2, "stub": True})

    def open_scene(name, args, specs):
        plugin.world = _MockWorld(name)
        return plugin.world

    monkeypatch.setattr(cb, "open_scene", open_scene)
    monkeypatch.setattr(cb, "require_software_gl", lambda scene, allow_gpu=False: None)
    out = tmp_path / "run"
    assert cb.main(["--scenes", "hollywood", "--arms", "truth_arm", "--arm-plugin",
                    "u9_truth_plugin", "--near-steps", "30", "--far", "1", "--budget",
                    "truth_arm=40", "--out", str(out), "--frames", "preview"]) == 0
    doc = json.loads((out / "check.json").read_text())
    # vs-sim's header, then the bridge's block
    for k in ("scenes", "arms", "near_steps", "far", "far_set", "perturb", "perturb_seed",
              "budgets", "outline_iou", "started_utc", "arm_plugin"):
        assert k in doc, k
    br = doc["bridge"]
    assert br["vssim_commit"] == fly.vssim_commit() and br["near_rule"] == cb.NEAR_RULE
    assert br["bridge_env"]["stub"] and br["cpu_s"] > 0
    sc = doc["scenes"]["hollywood"]
    assert sc["render_mode"] == "mujoco-llvmpipe" and sc["goal_md5"]
    e = sc["arms"]["truth_arm"]
    assert e["label"] == "external:truth" and e["plugin"]["module"] == "u9_truth_plugin"
    assert e["identity"]["ok"] and e["identity"]["zero_displacement_share"] == 1.0
    nr = e["near"]
    assert nr["start_id"] == "n_000" and 3 <= nr["start_mm"] <= 8 and nr["max_iter"] == 40
    assert e["near_ok"] == (nr["final_mm"] < nr["start_mm"]) and e["near_ok"]
    assert len(e["far"]) == 1 and e["far"][0]["max_iter"] == 40
    assert e["render_mode"] == "mujoco-llvmpipe"
    recs = [json.loads(ln) for ln in (out / "records.jsonl").read_text().splitlines()]
    assert [r["check_row"] for r in recs] == ["near", "far"]
    assert recs[0]["final_translation_mm"] == nr["final_mm"]
    assert recs[1]["start_id"] == e["far"][0]["start_id"]
    assert all(r["render_mode"] == "mujoco-llvmpipe" and r["vssim_commit"] for r in recs)
    md = (out / "check.md").read_text()
    assert md.startswith("| scene | arm | identity:") and "| hollywood | truth_arm | OK: 24" in md
    assert (out / "hollywood_goal_preview.png").exists()
    assert (out / "hollywood_truth_arm_near_first_preview.png").exists()
    assert not (out / "hollywood_goal_depth.npy").exists()       # preview: md5 only
    assert len(sc["goal_depth_md5"]) == 32


# --------------------------------------------------------- the committed flight
def _flown():
    if not (U9 / "summary.json").exists():
        pytest.skip("U9 not flown yet (results/bridge/u9/summary.json)")
    summary = json.loads((U9 / "summary.json").read_text())
    runs = [Path(r["file"]).parent.name for r in summary["runs"]]
    docs = {n: json.loads((U9 / n / "check.json").read_text()) for n in runs}
    return summary, docs


def test_the_u9_matrix_is_complete_on_llvmpipe_cpu():
    summary, docs = _flown()
    assert summary["complete"] and summary["n_flown"] == len(cb.U9_ARMS) * len(cb.U9_SCENES)
    assert summary["render_mode"] == "mujoco-llvmpipe"
    for name, doc in docs.items():
        env = doc["bridge"]["bridge_env"]
        assert env["cuda_available"] is False and env["allow_gpu"] is False
        assert (env["torch_threads"], env["torch_interop_threads"]) == (2, 2)
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["LP_NUM_THREADS"] == "2"
        assert doc["bridge"]["look"] == "isaac_fit_v1" and doc["bridge"]["device"] == "cpu"
        for sname, sc in doc["scenes"].items():
            assert sc["render_mode"] == "mujoco-llvmpipe" and "llvmpipe" in sc["gl_renderer"]
            assert sc["look"] == "isaac_fit_v1" and sc["look_curve_md5"]
            for arm_id, e in sc["arms"].items():
                assert e["render_mode"] == "mujoco-llvmpipe", (sname, arm_id)
                assert "error" not in e and "build_error" not in e, (sname, arm_id)


def test_every_scene_flew_the_same_commit():
    """One driver commit for the whole matrix (``+dirty`` is allowed and documented in
    ``results/bridge/u9/README.md``: the dirt was the new, untracked test and diagnosis files,
    which no flight imports)."""
    _summary, docs = _flown()
    heads = {doc["bridge"]["vsbench_commit"].split("+")[0] for doc in docs.values()}
    assert len(heads) == 1, heads
    assert {doc["bridge"]["vssim_commit"] for doc in docs.values()} == {
        (REPO / "external" / "VSSIM_COMMIT").read_text().split()[0]}


def test_the_records_are_the_entries_of_check_json():
    summary, docs = _flown()
    for name, doc in docs.items():
        recs = [json.loads(ln) for ln in (U9 / name / "records.jsonl").read_text().splitlines()]
        for sname, sc in doc["scenes"].items():
            for arm_id, e in sc["arms"].items():
                mine = [r for r in recs if r["check_scene"] == sname and r["check_arm"] == arm_id]
                assert [r["check_row"] for r in mine] == ["near"] + ["far"] * len(e["far"])
                assert mine[0]["final_translation_mm"] == e["near"]["final_mm"]
                assert mine[0]["max_iter"] == doc["near_steps"]
                for r, f in zip(mine[1:], e["far"]):
                    assert r["start_id"] == f["start_id"]
                    assert r["final_translation_mm"] == f["final_mm"]
                for r in mine:
                    assert r["render_mode"] == "mujoco-llvmpipe" and r["look"] == "isaac_fit_v1"
                    assert r["vssim_commit"] == doc["bridge"]["vssim_commit"]


DIAGNOSE = U9 / "diagnose.json"


def near_diagnosis(row: dict, diag: dict) -> str | None:
    """Why a near row missed, from evidence only (``None`` = undiagnosed):

    * ``as_in_isaac`` - Isaac's own check (``check1``, the same near start, the same cell
      label) missed it too;
    * ``not_the_look_nor_the_world`` - Isaac landed it, but ``tools/u9_diagnose.py``
      reproduced the miss bit for bit (``isaac_fit_v1``) and the cell misses as well without
      the look (``raw``) and, on a poster, on the copied numpy mock (no renderer at all).
    """
    isaac = row.get("isaac") or {}
    if isaac.get("near_ok") is False:
        return "as_in_isaac"
    key = f"{row['scene']}/{row['arm']}"
    rep, raw, mock = (diag.get(f"{key}/{w}") for w in ("isaac_fit_v1", "raw", "mock"))
    poster = row["scene"] in ("hollywood", "poster2", "poster3")
    if (rep and rep["final_mm"] == row["near_final_mm"] and raw and not raw["closer"]
            and (not poster or (mock and not mock["closer"]))):
        return "not_the_look_nor_the_world"
    return None


def test_the_verdicts_and_the_gate():
    summary, docs = _flown()
    ref = cb.reference()
    diag = json.loads(DIAGNOSE.read_text()) if DIAGNOSE.exists() else {}
    for row in summary["rows"]:
        e = docs_entry(docs, row["scene"], row["arm"])
        assert row["near_ok"] == cb.near_verdict(e["near"])["near_ok"]
        idn = e["identity"]
        assert row["identity_ok"] == bool(idn["v_norm_m_s"] < ref.IDENTITY_V_MAX
                                          and np.deg2rad(idn["w_norm_deg_s"]) < ref.IDENTITY_W_MAX
                                          and idn["n_matched"] > 0)
    misses = {f"{r['scene']}/{r['arm']}": near_diagnosis(r, diag)
              for r in summary["rows"] if not r["near_ok"]}
    gate = {"unit": "U9 - the rig check twin (docs/PORT_PLAN.md section 7)",
            "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "matrix": {"arms": list(cb.U9_ARMS), "scenes": list(cb.U9_SCENES)},
            "identity_all_green": summary["identity_all_green"],
            "near_all_green": summary["near_all_green"],
            "identity_fail": summary["identity_fail"], "near_fail": summary["near_fail"],
            "near_fail_diagnosis": misses, "near_rule": cb.NEAR_RULE,
            "cells": {f"{r['scene']}/{r['arm']}": {
                "identity_ok": r["identity_ok"], "near_ok": r["near_ok"],
                "near_start_mm": r["near_start_mm"], "near_final_mm": r["near_final_mm"],
                "near_final_deg": r["near_final_deg"],
                "far": [(f["start_id"], f["start_mm"], f["final_mm"], f["final_deg"],
                         f["terminated"]) for f in r["far"]],
                "isaac_near_final_mm": (r["isaac"] or {}).get("near_final_mm"),
                "isaac_near_ok": (r["isaac"] or {}).get("near_ok")} for r in summary["rows"]},
            "source": "results/bridge/u9/summary.json, results/bridge/u9/diagnose.json"}
    if write_gates():
        GATE.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    # the identity row is the verdict vs-sim's script itself computes: all green
    assert summary["identity_all_green"], summary["identity_fail"]
    # the near row (the twin's reading of vs-sim's docstring) is not all green; every miss
    # must carry evidence-based diagnosis - none may be a MuJoCo-only miss left unexplained
    assert all(misses.values()), {k: v for k, v in misses.items() if v is None}
    # the classical arms come closer on every scene (as in Isaac's check)
    for r in summary["rows"]:
        if r["arm"] in ("sift_ransac", "orb_ransac"):
            assert r["near_ok"], (r["scene"], r["arm"])


def docs_entry(docs, scene, arm):
    for doc in docs.values():
        if scene in doc["scenes"] and arm in doc["scenes"][scene]["arms"]:
            return doc["scenes"][scene]["arms"][arm]
    raise KeyError((scene, arm))
