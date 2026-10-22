"""U2: the bench driver ``python -m vsbench.world.fly`` and the Milestone-1 gate.

* ``test_driver_smoke``  - ``sift_ransac`` (weight-free) on hollywood from the first
  ``near60`` start for a few steps, in a child process (the driver hides ``nvidia-smi``
  and pins threads for its own process): the record exists, passes the copied frozen-schema
  checks and carries ``render_mode``, ``vsbench_commit``, ``vssim_commit``.
* ``test_milestone1_gate`` - reads the M1 records the driver wrote under
  ``results/bridge/m1/`` (``vitvs_dinov3`` with the real DINOv3 weights on the CPU, 100
  steps, and the ``sift_ransac`` smoke) and checks the gate of docs/PORT_PLAN.md section 7
  (U2): final < 5 mm and < the start's offset, ``predicate_a``, >= 20 matches on frame 1,
  schema-valid, the three provenance fields. Skipped when the flight has not been flown:

      source tools/cpu_env.sh
      $VSBENCH_PY -m vsbench.world.fly run --scene hollywood --arms vitvs_dinov3 sift_ransac \\
          --start-set near60 --limit 1 --max-iter 100 --out results/bridge/m1 --film
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
M1 = REPO / "results" / "bridge" / "m1"
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


def assert_provenance(rec: dict) -> None:
    from vsbench.world.fly import vssim_commit

    assert rec["render_mode"] == "mujoco-llvmpipe"
    assert rec["vssim_commit"] == vssim_commit() == (REPO / "external" / "VSSIM_COMMIT").read_text().strip()
    assert rec["vsbench_commit"] and len(rec["vsbench_commit"].split("+")[0]) == 40
    assert rec["goal_frame_md5"] and rec["goal_render_warmup"] == 1


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


@pytest.mark.parametrize("arm", ["vitvs_dinov3", "sift_ransac"])
def test_milestone1_gate(arm):
    path = M1 / f"hollywood_{arm}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not flown yet (see the module docstring)")
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["start_cell"] == "near" and rec["start_id"] == "n_000"
    assert rec["steps"] == 100 and rec["max_iter"] == 100
    assert rec["final_translation_mm"] < 5.0
    assert rec["final_translation_mm"] < rec["start_translation_mm"]
    assert rec["success"]["predicate_a"] is True
    assert rec["first_frame_n_matches"] >= 20
    assert_schema_valid(rec)
    assert_provenance(rec)
    assert "+dirty" not in rec["vsbench_commit"], "fly M1 from a committed tree"
