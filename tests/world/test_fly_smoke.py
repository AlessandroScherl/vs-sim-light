"""U2: the bench driver ``python -m vsbench.world.fly`` and the Milestone-1 gate.

* ``test_driver_smoke``  - ``sift_ransac`` (weight-free) on hollywood from the first
  ``near60`` start for a few steps, in a child process (the driver hides ``nvidia-smi``
  and pins threads for its own process): the record exists, passes the copied frozen-schema
  checks and carries ``render_mode``, ``vsbench_commit``, ``vssim_commit``.
* ``test_milestone1_gate`` - reads the M1 records the driver wrote under
  ``results/bridge/m1/`` (``vitvs_dinov3`` with the real DINOv3 weights on the CPU, 100
  steps, and the ``sift_ransac`` smoke) and checks the gate of docs/PORT_PLAN.md section 7
  (U2, revision 3). Both arms: the record exists, 100 steps, >= 20 matches on frame 1,
  schema-valid, the three provenance fields. ``sift_ransac`` keeps the landing gate
  (final < 5 mm and < the start's offset, ``predicate_a``). ``vitvs_dinov3`` is gated as
  "ran and was recorded" plus its error-curve statistics (median over steps 75-100, max over
  steps 50-100, from the per-step telemetry of the same start), reported, not thresholded:
  at 100 steps it is still on its way in (3.8 mm from 5.6 mm), not landed.
* ``test_telemetry_truth_seam`` - the committed ``results/bridge/m1_telemetry/`` flight
  (same start, 100 steps, ``--telemetry-extended``): every ``gt_*`` column of every step is
  finite and the truth seam (``MujocoScene.gt_correspondence``) finds ~every match valid.

Flown with (skipped until then)::

    source tools/cpu_env.sh
    $VSBENCH_PY -m vsbench.world.fly run --scene hollywood --arms vitvs_dinov3 sift_ransac \\
        --start-set near60 --limit 1 --max-iter 100 --out results/bridge/m1 --film
    $VSBENCH_PY -m vsbench.world.fly run --scene hollywood --arms sift_ransac vitvs_dinov3 \\
        --start-set near60 --limit 1 --max-iter 100 --out results/bridge/m1_telemetry \\
        --telemetry-extended

With ``VSBENCH_WRITE_GATES=1`` the two gates write ``results/bridge/u2_m1_gate.json``.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
M1 = REPO / "results" / "bridge" / "m1"
M1_TELEMETRY = REPO / "results" / "bridge" / "m1_telemetry"
GATE_JSON = REPO / "results" / "bridge" / "u2_m1_gate.json"
GATES: dict = {}
#: the valid share the truth seam must reach on the telemetry flight (posters: no occluder)
GT_VALID_SHARE_MIN = 0.99


@pytest.fixture(scope="module", autouse=True)
def _write_gates(tmp_path_factory):
    yield
    if not GATES:
        return
    out = (GATE_JSON if os.environ.get("VSBENCH_WRITE_GATES", "").strip() == "1"
           else tmp_path_factory.mktemp("gates") / GATE_JSON.name)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(GATES)
    old["unit"] = "U2 / Milestone 1 - the bench driver and the first flights (docs/PORT_PLAN.md 7)"
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    print(f"[gates] wrote {out}")
COPIED_RUNNER_TEST = REPO / "vsbench" / "tests" / "fast" / "test_runner.py"


def _copied_schema_tests():
    """The copied ``tests/fast/test_runner.py`` as a module (its frozen-schema checks)."""
    spec = importlib.util.spec_from_file_location("_vssim_copied_test_runner", COPIED_RUNNER_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assert_schema_valid(rec: dict) -> None:
    """The copied frozen-schema tests, run on a bridge record: every measurement field of
    the frozen N3.7b record present with its type, no dropped arm-suffix flag."""
    t = _copied_schema_tests()
    t.test_the_record_keeps_every_measurement_field_of_the_frozen_schema((rec, None))
    t.test_the_record_types_match_the_frozen_schema((rec, None))
    assert not (set(rec) & t.SCHEMA_DROPPED)


def assert_provenance(rec: dict, *, revision3: bool = True) -> None:
    """The bridge's provenance fields; ``revision3`` = written after the M1 review (the
    protocol's warm-up count and the dirty rule on the record)."""
    from vsbench.world.fly import vssim_commit

    assert rec["render_mode"] == "mujoco-llvmpipe"
    assert rec["vssim_commit"] == vssim_commit() == (REPO / "external" / "VSSIM_COMMIT").read_text().strip()
    assert rec["vsbench_commit"] and len(rec["vsbench_commit"].split("+")[0]) == 40
    assert rec["goal_frame_md5"] and rec["goal_render_warmup"] == 1
    if revision3:
        assert rec["goal_render_warmup_protocol"] == 8 and rec["goal_render_warmup_reason"]
        assert rec["vsbench_commit_rule"]


def _records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _telemetry(cell_dir: Path) -> list[dict]:
    """The per-episode telemetry lines (``{"meta", "steps"}``) of a cell's one run dir."""
    files = sorted(cell_dir.glob("*/telemetry.jsonl.gz"))
    assert len(files) == 1, files
    with gzip.open(files[0], "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def curve_stats(steps: list[dict]) -> dict:
    """The error curve's tail: true translation error at each step's capture pose."""
    t = {int(s["i"]): float(s["t_mm"]) for s in steps}
    tail = sorted(t[i] for i in range(75, 101))
    mid = [t[i] for i in range(50, 101)]
    n = len(tail)
    med = tail[n // 2] if n % 2 else 0.5 * (tail[n // 2 - 1] + tail[n // 2])
    return {"median_t_mm_steps_75_100": round(med, 4), "max_t_mm_steps_50_100": round(max(mid), 4),
            "min_t_mm_steps_50_100": round(min(mid), 4), "t_mm_step_1": t[1],
            "t_mm_step_100": t[100], "n_steps": len(t)}


def test_arm_config_overrides_and_pin():
    from vsbench.world import fly

    proto = fly.load_protocol()
    cfg = fly.arm_config(proto, "vitvs_dinov3", max_iter=100, device="cpu")
    assert cfg["matcher"]["device"] == "cpu" and cfg["episode"]["max_iter"] == 100
    assert cfg["matcher"]["backbone_name"] == "dinov3_vits16"
    sift = fly.arm_config(proto, "sift_ransac", device="cpu")
    assert "device" not in sift["matcher"]            # a classical arm loads no encoder
    assert sift["episode"]["max_iter"] == 600
    from vsbench.vs.bench.arms import arm_label

    # the bridge's overrides never change a published cell's derived label
    assert arm_label(cfg) == arm_label(fly.arm_config(proto, "vitvs_dinov3"))


def test_driver_smoke(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "VSBENCH_SOFTWARE_GL": "1"}
    cmd = [sys.executable, "-m", "vsbench.world.fly", "run", "--scene", "hollywood",
           "--arms", "sift_ransac", "--start-set", "near60", "--limit", "1",
           "--max-iter", "4", "--out", str(tmp_path)]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    recs = [json.loads(x) for x in (tmp_path / "hollywood_sift_ransac.jsonl").read_text().splitlines()]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["steps"] == 4 and rec["max_iter"] == 4 and rec["scene"] == "hollywood"
    assert rec["first_frame_n_matches"] >= 20
    assert rec["vram_used_mb"] is None and rec["gpu_util_mean_pct"] is None   # nvidia-smi hidden
    assert_schema_valid(rec)
    assert_provenance(rec)
    run_json = next((tmp_path / "hollywood_sift_ransac").glob("*/run.json"))
    meta = json.loads(run_json.read_text())
    assert meta["render_mode"] == "mujoco-llvmpipe" and meta["vssim_commit"] == rec["vssim_commit"]
    env = meta["bridge_env"]
    assert env["torch_threads"] == 2 and env["torch_interop_threads"] == 2, env
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["VSBENCH_SOFTWARE_GL"] == "1"
    assert meta["scene_meta"]["renderer"]["gl_renderer"].startswith("llvmpipe")
    # U4b: the record names its look (the default: isaac_fit_v1, hollywood's curve)
    from vsbench.world.look import load_look

    want = load_look("isaac_fit_v1", "hollywood")
    assert rec["look"] == "isaac_fit_v1" and rec["look_curve_id"] == "isaac_fit_v1/hollywood"
    assert rec["look_curve_md5"] == want.curve_md5
    assert rec["look_record"]["look_config_md5"] == want.config_md5
    assert rec["look_record"]["gl_renderer"].startswith("llvmpipe")


def _child(code: str, env_over: dict, timeout=600):
    env = {**os.environ, **env_over}
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=timeout)


def test_station_guard_forces_cpu_and_cleans_its_stub():
    """A shell that exported a GPU and no software GL still gets both hidden; the
    ``nvidia-smi`` stub directory is gone when the process ends."""
    code = ("import json, os, shutil; from vsbench.world.fly import station_guard; "
            "e = station_guard(); e['stub'] = os.path.dirname(shutil.which('nvidia-smi')); "
            "print(json.dumps(e))")
    p = _child(code, {"CUDA_VISIBLE_DEVICES": "0", "VSBENCH_SOFTWARE_GL": "0"})
    assert p.returncode == 0, p.stderr[-2000:]
    e = json.loads(p.stdout.strip().splitlines()[-1])
    assert e["CUDA_VISIBLE_DEVICES"] == "" and e["VSBENCH_SOFTWARE_GL"] == "1"
    assert e["cuda_available"] is False and e["allow_gpu"] is False
    assert e["torch_threads"] == 2 and e["torch_interop_threads"] == 2
    assert Path(e["stub"]).name.startswith("vsbench_nosmi_")
    assert not Path(e["stub"]).exists(), "the nvidia-smi stub directory outlived the process"


def test_require_software_gl_refuses_a_hardware_context():
    from vsbench.world.fly import require_software_gl

    def scene(mode, gl):
        return SimpleNamespace(render_mode=mode, renderer=SimpleNamespace(gl_renderer=gl),
                               close=lambda: None)

    require_software_gl(scene("mujoco-llvmpipe", "llvmpipe (LLVM 20.1.2, 256 bits)"))
    for mode, gl in (("mujoco-egl-gpu", "NVIDIA GeForce RTX 4090/PCIe/SSE2"),
                     ("mujoco-llvmpipe", "unknown"),           # an unreadable string never passes
                     ("mujoco-egl-unknown", "")):
        with pytest.raises(SystemExit, match="not Mesa llvmpipe"):
            require_software_gl(scene(mode, gl))
        require_software_gl(scene(mode, gl), allow_gpu=True)   # --allow-gpu lifts it


def test_only_the_protocols_start_sets(tmp_path):
    """``simple5`` is a copied start set but not a protocol set: refused, as vs-sim's fly.py."""
    p = subprocess.run([sys.executable, "-m", "vsbench.world.fly", "run", "--scene", "hollywood",
                        "--arms", "sift_ransac", "--start-set", "simple5", "--limit", "1",
                        "--max-iter", "1", "--out", str(tmp_path)], cwd=REPO,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "VSBENCH_SOFTWARE_GL": "1"},
                       capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "not in the protocol" in p.stderr, p.stderr[-2000:]
    assert not list(tmp_path.iterdir())


def test_vsbench_commit_dirty_rule():
    """Tracked changes and untracked files count; the run's own --out does not."""
    from vsbench.world.fly import dirty_paths

    status = (" M vsbench/world/fly.py\n?? results/bridge/m1_telemetry/x/run.json\n"
              "?? notes.txt\nR  a.py -> b.py\n")
    assert dirty_paths(status) == ["vsbench/world/fly.py", "results/bridge/m1_telemetry/x/run.json",
                                   "notes.txt", "b.py"]
    assert dirty_paths(status, REPO / "results" / "bridge" / "m1_telemetry") == [
        "vsbench/world/fly.py", "notes.txt", "b.py"]
    assert dirty_paths("?? results/bridge/m1_telemetry2/a\n",
                       REPO / "results" / "bridge" / "m1_telemetry") == ["results/bridge/m1_telemetry2/a"]


@pytest.mark.parametrize("arm", ["vitvs_dinov3", "sift_ransac"])
def test_milestone1_gate(arm):
    path = M1 / f"hollywood_{arm}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not flown yet (see the module docstring)")
    rec = _records(path)[0]
    # both arms: it ran on this world, from a committed tree, and the record is the frozen one
    assert rec["start_cell"] == "near" and rec["start_id"] == "n_000"
    assert rec["steps"] == 100 and rec["max_iter"] == 100
    assert rec["first_frame_n_matches"] >= 20
    assert_schema_valid(rec)
    assert_provenance(rec, revision3=False)          # flown at 409656d, before the review
    assert "+dirty" not in rec["vsbench_commit"], "fly M1 from a committed tree"
    row = {"start_translation_mm": rec["start_translation_mm"],
           "final_translation_mm": rec["final_translation_mm"],
           "final_rotation_deg": rec["final_rotation_deg"],
           "predicate_a": rec["success"]["predicate_a"],
           "first_frame_n_matches": rec["first_frame_n_matches"],
           "sec_per_step": rec["sec_per_step"], "vsbench_commit": rec["vsbench_commit"]}
    if arm == "sift_ransac":                          # the landing gate
        assert rec["final_translation_mm"] < 5.0
        assert rec["final_translation_mm"] < rec["start_translation_mm"]
        assert rec["success"]["predicate_a"] is True
        row["gate"] = "landing: final < 5 mm and < start, predicate_a, >= 20 frame-1 matches"
    else:                                             # ran + recorded + the curve, reported
        tel_dir = M1_TELEMETRY / f"hollywood_{arm}"
        if not tel_dir.exists():
            pytest.skip(f"{tel_dir} not flown yet (the curve statistics need its telemetry)")
        trec = _records(M1_TELEMETRY / f"hollywood_{arm}.jsonl")[0]
        ep = _telemetry(tel_dir)[0]
        assert trec["start_id"] == rec["start_id"] and trec["steps"] == 100
        stats = curve_stats(ep["steps"])
        assert stats["n_steps"] == 100 and all(math.isfinite(v) for v in stats.values())
        # the M1 flight's own film keeps every second step's error: the same curve
        film = json.loads(next((M1 / f"hollywood_{arm}").glob("*/film/n_000/frames_meta.json"))
                          .read_text())
        t_tel = {int(x["i"]): float(x["t_mm"]) for x in ep["steps"]}
        film_dev = max(abs(float(f["t_mm"]) - t_tel[int(f["step"])]) for f in film)
        row.update(curve_stats=stats, curve_source=str(tel_dir.relative_to(REPO)),
                   telemetry_flight_final_translation_mm=trec["final_translation_mm"],
                   m1_film_vs_telemetry_max_abs_dev_mm=round(film_dev, 4),
                   gate="ran and was recorded (100 steps, >= 20 frame-1 matches, frozen schema, "
                        "provenance); the curve statistics are reported, not thresholded")
        assert film_dev <= 0.01, "the telemetry flight is not the M1 flight's curve"
    GATES.setdefault("milestone1", {})[arm] = row
    print(json.dumps(row, indent=1))


def test_telemetry_truth_seam():
    """The truth seam exercised in a flight: ``gt_correspondence`` on every step."""
    cell = M1_TELEMETRY / "hollywood_sift_ransac"
    if not cell.exists():
        pytest.skip(f"{cell} not flown yet (see the module docstring)")
    rec = _records(M1_TELEMETRY / "hollywood_sift_ransac.jsonl")[0]
    assert rec["start_id"] == "n_000" and rec["steps"] == 100
    assert_schema_valid(rec)
    assert_provenance(rec)
    assert "+dirty" not in rec["vsbench_commit"], "fly the telemetry gate from a committed tree"
    ep = _telemetry(cell)[0]
    steps = ep["steps"]
    assert len(steps) == 100
    n_pts = n_valid = n_right = 0
    finite = True
    for st in steps:
        n = len(st["gu"])
        assert len(st["gt_u"]) == len(st["gt_v_px"]) == len(st["gt_ok"]) == len(st["err_patch"]) == n
        assert st["gt_v"] == sum(st["gt_ok"])
        for key in ("gt_u", "gt_v_px", "err_patch"):
            finite &= all(v is not None and math.isfinite(float(v)) for v in st[key])
        finite &= math.isfinite(float(st["gt_v"])) and math.isfinite(float(st["gt_r"]))
        n_pts += n
        n_valid += int(st["gt_v"])
        n_right += int(st["gt_r"])
    share = n_valid / n_pts
    err = [float(v) for st in steps for v in st["err_patch"]]
    row = {"n_steps": len(steps), "n_point_steps": n_pts, "gt_valid_share": round(share, 6),
           "gt_right_share": round(n_right / n_pts, 6), "all_gt_columns_finite": bool(finite),
           "err_patch_median": round(sorted(err)[len(err) // 2], 5), "err_patch_max": round(max(err), 5),
           "final_translation_mm": rec["final_translation_mm"],
           "gt_columns": sorted(k for k in steps[0] if k.startswith("gt_") or k == "err_patch"),
           "gate": f"every gt_* / err_patch value finite; gt_valid_share >= {GT_VALID_SHARE_MIN}"}
    GATES["telemetry_truth_seam"] = row
    print(json.dumps(row, indent=1))
    assert finite
    assert share >= GT_VALID_SHARE_MIN
