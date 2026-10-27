"""The bench driver ``python -m vsbench.world.fly``.

* ``test_driver_smoke`` - ``sift_ransac`` (weight-free) on hollywood from the first ``near60``
  start for a few steps with ``--telemetry-extended``, in a child process: the record exists,
  passes the copied frozen-schema checks, carries ``render_mode``, ``device``,
  ``vsbench_commit``, ``vssim_commit`` and the look, and the truth seam
  (``MujocoScene.gt_correspondence``) fills every ``gt_*`` column of every step.
* the runtime: station mode hides the GPU and caps the threads in a child process; the
  start-set refusal; the dirty rule of ``vsbench_commit``.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
#: the scenes whose assets this module needs (tests/conftest.py skips it without them)
NEEDS_SCENES = ("hollywood",)
COPIED_RUNNER_TEST = REPO / "vsbench" / "tests" / "fast" / "test_runner.py"
#: the valid share the truth seam must reach (posters: no occluder)
GT_VALID_SHARE_MIN = 0.99


def _copied_schema_tests():
    """The copied ``tests/fast/test_runner.py`` as a module (its frozen-schema checks)."""
    spec = importlib.util.spec_from_file_location("_vssim_copied_test_runner", COPIED_RUNNER_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assert_schema_valid(rec: dict) -> None:
    """The copied frozen-schema tests, run on a record of this driver: every measurement
    field of the frozen record present with its type, no dropped arm-suffix flag."""
    t = _copied_schema_tests()
    t.test_the_record_keeps_every_measurement_field_of_the_frozen_schema((rec, None))
    t.test_the_record_types_match_the_frozen_schema((rec, None))
    assert not (set(rec) & t.SCHEMA_DROPPED)


def assert_provenance(rec: dict) -> None:
    from vsbench.world.fly import vssim_commit

    assert rec["render_mode"].startswith("mujoco-")
    assert rec["vssim_commit"] == vssim_commit() and len(rec["vssim_commit"]) == 40
    assert rec["goal_frame_md5"] and rec["goal_render_warmup"] == 1
    assert rec["goal_render_warmup_protocol"] == 8 and rec["goal_render_warmup_reason"]
    assert rec["vsbench_commit_rule"]


def test_arm_config_overrides():
    from vsbench.world import fly

    proto = fly.load_protocol()
    cfg = fly.arm_config(proto, "vitvs_dinov3", max_iter=100, device="cpu")
    assert cfg["matcher"]["device"] == "cpu" and cfg["episode"]["max_iter"] == 100
    assert cfg["matcher"]["backbone_name"] == "dinov3_vits16"
    sift = fly.arm_config(proto, "sift_ransac", device="cpu")
    assert "device" not in sift["matcher"]            # a classical arm loads no encoder
    assert sift["episode"]["max_iter"] == 600
    assert fly.record_device(sift, {"device": "cuda"}) == "cpu"
    assert fly.record_device(cfg, {"device": "cpu"}) == "cpu"
    assert fly.record_device(None, {"device": "cpu"}) is None
    from vsbench.vs.bench.arms import arm_label

    # the driver's overrides never change a published cell's derived label
    assert arm_label(cfg) == arm_label(fly.arm_config(proto, "vitvs_dinov3"))


def test_only_the_shipped_arms_fly():
    from vsbench.world import fly

    proto = fly.load_protocol()
    ids = fly.flyable_ids(proto)
    assert {"sift_ransac", "orb_ransac", "akaze_ransac", "vitvs_dinov3", "vitvs_dinov2",
            "artvs"} <= set(ids)
    assert "cns_v1" not in ids and "dinobot" not in ids and "lightglue_ransac" not in ids
    with pytest.raises(SystemExit, match="not part of this simulator"):
        fly.arm_entry(proto, "cns_v1")
    with pytest.raises(SystemExit, match="unknown arm"):
        fly.arm_entry(proto, "no_such_arm")


def _station_env(**over) -> dict:
    return {**os.environ, "VSBENCH_STATION": "1", "CUDA_VISIBLE_DEVICES": "",
            "VSBENCH_SOFTWARE_GL": "1", **over}


def test_driver_smoke(tmp_path):
    cmd = [sys.executable, "-m", "vsbench.world.fly", "run", "--scene", "hollywood",
           "--arms", "sift_ransac", "--start-set", "near60", "--limit", "1",
           "--max-iter", "4", "--telemetry-extended", "--out", str(tmp_path)]
    p = subprocess.run(cmd, cwd=REPO, env=_station_env(), capture_output=True, text=True,
                       timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    recs = [json.loads(x) for x in (tmp_path / "hollywood_sift_ransac.jsonl").read_text().splitlines()]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["steps"] == 4 and rec["max_iter"] == 4 and rec["scene"] == "hollywood"
    assert rec["first_frame_n_matches"] >= 20
    assert rec["device"] == "cpu" and rec["station_mode"] is True
    assert rec["render_mode"] == "mujoco-llvmpipe"
    assert rec["vram_used_mb"] is None and rec["gpu_util_mean_pct"] is None   # nvidia-smi hidden
    assert_schema_valid(rec)
    assert_provenance(rec)
    run_json = next((tmp_path / "hollywood_sift_ransac").glob("*/run.json"))
    meta = json.loads(run_json.read_text())
    assert meta["render_mode"] == "mujoco-llvmpipe" and meta["vssim_commit"] == rec["vssim_commit"]
    env = meta["bridge_env"]
    assert env["torch_threads"] == 2 and env["torch_interop_threads"] == 2, env
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["station_mode"] and env["device"] == "cpu"
    assert meta["scene_meta"]["renderer"]["gl_renderer"].startswith("llvmpipe")
    from vsbench.world.look import load_look

    want = load_look("isaac_fit_v1", "hollywood")
    assert rec["look"] == "isaac_fit_v1" and rec["look_curve_id"] == "isaac_fit_v1/hollywood"
    assert rec["look_curve_md5"] == want.curve_md5
    assert rec["look_record"]["look_config_md5"] == want.config_md5
    # the truth seam: every gt_* column of every step filled and finite
    files = sorted((tmp_path / "hollywood_sift_ransac").glob("*/telemetry.jsonl.gz"))
    assert len(files) == 1, files
    with gzip.open(files[0], "rt") as f:
        steps = [json.loads(x) for x in f if x.strip()][0]["steps"]
    assert len(steps) == 4
    n_pts = n_valid = 0
    for st in steps:
        n = len(st["gu"])
        assert len(st["gt_u"]) == len(st["gt_v_px"]) == len(st["gt_ok"]) == len(st["err_patch"]) == n
        for key in ("gt_u", "gt_v_px", "err_patch"):
            assert all(v is not None and math.isfinite(float(v)) for v in st[key]), key
        n_pts += n
        n_valid += int(st["gt_v"])
    assert n_valid / n_pts >= GT_VALID_SHARE_MIN


def _child(code: str, env: dict, timeout=600):
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=timeout)


def test_station_mode_forces_cpu_and_cleans_its_stub():
    """In station mode, a shell that exported a GPU and no software GL still gets both hidden,
    two threads and ``device`` cpu; the ``nvidia-smi`` stub directory is gone when the process
    ends."""
    code = ("import json, os, shutil; from vsbench.world.fly import runtime_setup; "
            "e = runtime_setup(); e['stub'] = os.path.dirname(shutil.which('nvidia-smi')); "
            "print(json.dumps(e))")
    env = _station_env(CUDA_VISIBLE_DEVICES="0", VSBENCH_SOFTWARE_GL="0")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS",
              "TORCH_NUM_THREADS"):
        env.pop(k, None)
    p = _child(code, env)
    assert p.returncode == 0, p.stderr[-2000:]
    e = json.loads(p.stdout.strip().splitlines()[-1])
    assert e["CUDA_VISIBLE_DEVICES"] == "" and e["VSBENCH_SOFTWARE_GL"] == "1"
    assert e["cuda_available"] is False and e["allow_gpu"] is False and e["device"] == "cpu"
    assert e["torch_threads"] == 2 and e["torch_interop_threads"] == 2 and e["thread_cap"] == 2
    assert e["station_mode"] and e["nvidia_smi"] == "hidden"
    assert Path(e["stub"]).name.startswith("vsbench_nosmi_")
    assert not Path(e["stub"]).exists(), "the nvidia-smi stub directory outlived the process"


def test_default_mode_caps_nothing_and_hides_nothing():
    """Outside station mode: no thread cap, no nvidia-smi stub, CUDA left as exported (the
    child hides CUDA itself, so no GPU context can open on this machine)."""
    code = "import json; from vsbench.world.fly import runtime_setup; print(json.dumps(runtime_setup()))"
    env = {k: v for k, v in os.environ.items()
           if k not in ("VSBENCH_STATION", "VSBENCH_SOFTWARE_GL", "VSBENCH_ALLOW_GPU")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    p = _child(code, env)
    assert p.returncode == 0, p.stderr[-2000:]
    e = json.loads(p.stdout.strip().splitlines()[-1])
    assert not e["station_mode"] and e["thread_cap"] is None and e["nvidia_smi"] == "visible"
    assert e["device_requested"] == "auto" and e["device"] == "cpu"     # no CUDA visible here
    assert not e["gl_plan"]["software_gl"]


def test_only_the_protocols_start_sets(tmp_path):
    """``simple5`` is a copied start set but not a protocol set: refused, as vs-sim's fly.py."""
    p = subprocess.run([sys.executable, "-m", "vsbench.world.fly", "run", "--scene", "hollywood",
                        "--arms", "sift_ransac", "--start-set", "simple5", "--limit", "1",
                        "--max-iter", "1", "--out", str(tmp_path)], cwd=REPO,
                       env=_station_env(), capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "not in the protocol" in p.stderr, p.stderr[-2000:]
    assert not list(tmp_path.iterdir())


def test_vsbench_commit_dirty_rule():
    """Tracked changes and untracked files count; the run's own --out does not."""
    from vsbench.world.fly import dirty_paths

    status = (" M vsbench/world/fly.py\n?? runs/x/run.json\n"
              "?? notes.txt\nR  a.py -> b.py\n")
    assert dirty_paths(status) == ["vsbench/world/fly.py", "runs/x/run.json", "notes.txt", "b.py"]
    assert dirty_paths(status, REPO / "runs") == ["vsbench/world/fly.py", "notes.txt", "b.py"]
    assert dirty_paths("?? runs2/a\n", REPO / "runs") == ["runs2/a"]
