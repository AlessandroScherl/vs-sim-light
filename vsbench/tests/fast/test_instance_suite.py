# vs-sim copy: tests/fast/test_instance_suite.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The instance suite's scripts (2026-09-23): the chain's flight order, the check script's
step budgets, and the simple start files `build_starts.py` writes.

The Isaac halves (placing the objects, the swap on the stage, the renders) are the
station's: `tests/slow/test_station.py` and the build report's smoke."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vsbench.vs import benchmark, starts
from vsbench.vs.scenes import load_scenes

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_chain_flies_the_core_block_first_then_genvs_objects():
    plan = _load("scripts/chains/instance_suite_plan.py", "instance_suite_plan")
    core, gen = plan.plan("core"), plan.plan("gen")
    assert plan.plan("all") == core + gen
    # 6 kinds x (same, same_small, other_small, mine1, mine2) + the mug's mine3
    # + 6 reverse scenes x 3 = 49 core pairs; GEN-VS's held-out objects after them
    assert len(core) == 6 * 5 + 1 + 6 * 3 and all(not n.startswith("gen") for _s, n in core)
    assert ("inst_mug", "mine3") in core and len(gen) == 33
    assert gen and all(n.startswith("gen") and not s.endswith("_rev") for s, n in gen)
    specs, _ = load_scenes()
    assert len(core) + len(gen) == sum(len(specs[s].live_names())
                                       for s in benchmark.INSTANCE_SCENES)
    assert core[:3] == [("inst_shoe", "same"), ("inst_shoe", "same_small"),
                        ("inst_shoe", "other_small")]


def test_the_chain_script_flies_the_ruled_cells():
    text = (REPO / "scripts" / "chains" / "instance_suite.sh").read_text()
    for token in ("dinobot_objmask vitvs_dinov3_objmask artvs_roi_scene",
                  "--budget dinobot_objmask=30", 'FAR=${FAR:-5}',
                  '--far "$FAR" --far-set simple5', "--outline-iou",
                  'swap:@$2+ground:concrete', "--film", "DONE", "LIGHTTRACK_ROOT",
                  'mkdir -p "$OUT"', 'local dir=$OUT/$1_$2', 'load_starts(scene, "simple")',
                  '"build_error"', "arm error - not DONE"):
        assert token in text, token
    assert "prealign" not in text and "pre4way" not in text      # no rotation compensation
    assert "inst_$1" not in text                                  # no doubled prefix


FAKE_PY = r"""#!/bin/bash
# a stand-in interpreter for the chain: the pin check, the plan, and check_baselines
if [ "$1" = "-" ]; then
  cat > /dev/null
  [ -n "${FAKE_PINS_OK:-}" ] && { echo "[fake] pins ok"; exit 0; }
  echo "[fake] simple start files not pinned"; exit 1
fi
case "$1" in
  scripts/chains/instance_suite_plan.py)
    [ -n "${FAKE_PLAN_ARGV:-}" ] && echo "$*" >> "$FAKE_PLAN_ARGV"
    printf 'inst_shoe same
inst_shoe same_small
inst_drill other_small
'; exit 0 ;;
  scripts/check_baselines.py)
    # Isaac's kit interpreter reads its stdin: a pair must not be able to eat the plan's next
    # line (the first station launch, 2026-09-23 21:56 UTC, skipped `inst_shoe same_small`)
    IFS= read -r _swallowed || true
    [ -n "${FAKE_ARGV:-}" ] && echo "$*" >> "$FAKE_ARGV"
    shift
    while [ $# -gt 0 ]; do
      case "$1" in --out) out=$2; shift 2 ;; --scenes) scene=$2; shift 2 ;;
                   --perturb) pert=$2; shift 2 ;; *) shift ;; esac
    done
    echo "$scene $pert" >> "$FAKE_CALLS"
    mkdir -p "$out"
    case "$pert" in
      *"${FAKE_OOM:-@@none@@}"*) echo "VkResult: ERROR_OUT_OF_DEVICE_MEMORY"
        echo '{"arms": {"a": {"label": "ok"}}}' > "$out/check.json" ;;
      *same_small*) echo '{"arms": {"a": {"build_error": "boom"}}}' > "$out/check.json" ;;
      *other_small*) exit 3 ;;
      *) echo '{"arms": {"a": {"label": "ok"}}}' > "$out/check.json" ;;
    esac
    exit 0 ;;
esac
echo "unexpected call: $*" >&2; exit 9
"""


def _run_chain(tmp_path, pins_ok: bool, env_extra: dict | None = None,
               script: str = "instance_suite.sh"):
    import os
    import subprocess

    fake = tmp_path / "fake_py.sh"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    out = tmp_path / "out"
    env = {**os.environ, "VSSIM_REPO": str(REPO), "VSSIM_CHAIN_PY": str(fake), "OUT": str(out),
           "FAKE_CALLS": str(tmp_path / "calls.txt")}
    for key in ("ARMS", "EXTRA", "FAR", "BLOCK", "TIER", "PAIRS", "ARM_PLUGIN",
                "REFINEVS_ROOT", "VSSIM_MIN_FREE_MB", "VSSIM_GPU_FREE_CMD"):                                   # never the caller's
        env.pop(key, None)
    if pins_ok:
        env["FAKE_PINS_OK"] = "1"
    env.update(env_extra or {})
    env = {k: v for k, v in env.items() if v is not None}       # None = unset
    rc = subprocess.run(["bash", str(REPO / "scripts" / "chains" / script)],
                        env=env, capture_output=True, text=True, timeout=60).returncode
    logf = out / "instance_suite.log"
    log = logf.read_text() if logf.is_file() else ""
    return rc, out, log


def test_the_chain_refuses_without_the_pinned_start_files(tmp_path):
    rc, out, log = _run_chain(tmp_path, pins_ok=False)
    assert rc == 1 and "refused" in log                   # $OUT was created for the log
    assert not (tmp_path / "calls.txt").exists()          # no pair was flown


def test_the_chain_marks_done_only_a_clean_pair_and_names_it_once(tmp_path):
    rc, out, log = _run_chain(tmp_path, pins_ok=True)
    assert rc == 0
    assert (out / "inst_shoe_same" / "DONE").is_file()               # clean
    assert not (out / "inst_shoe_same_small" / "DONE").exists()      # an arm error inside
    assert not (out / "inst_drill_other_small" / "DONE").exists()    # non-zero exit
    assert "arm error - not DONE" in log and "rc=3 - not DONE" in log
    assert not any(p.name.startswith("inst_inst_") for p in out.iterdir())
    # a re-launch skips the finished pair and flies the other two again
    _run_chain(tmp_path, pins_ok=True)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls.count("inst_shoe swap:@same+ground:concrete") == 1
    assert calls.count("inst_shoe swap:@same_small+ground:concrete") == 2


def test_a_pair_that_reads_stdin_does_not_eat_the_plan(tmp_path):
    """The chain reads its plan on fd 3 and feeds every pair /dev/null: every planned pair is
    flown exactly once even when the flight reads stdin (Isaac's kit does)."""
    rc, out, log = _run_chain(tmp_path, pins_ok=True)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == ["inst_shoe swap:@same+ground:concrete",
                     "inst_shoe swap:@same_small+ground:concrete",
                     "inst_drill swap:@other_small+ground:concrete"], calls
    assert "[chain]   start" not in log                       # no empty pair


def test_check_baselines_parses_step_budgets():
    sys.path.insert(0, str(REPO / "scripts"))
    cb = _load("scripts/check_baselines.py", "check_baselines")
    assert cb.parse_budgets(["dinobot_objmask=30"]) == {"dinobot_objmask": 30}
    assert cb.parse_budgets([]) == {}
    for bad in (["dinobot_objmask"], ["=30"], ["a=0"], ["a=x"], ["a=3", "a=4"]):
        with pytest.raises(SystemExit):
            cb.parse_budgets(bad)


def test_build_starts_writes_a_simple_file_for_any_scene(tmp_path, monkeypatch):
    """A poster needs no simulator, so the laptop can build its simple file end to end;
    the instance scenes are rooms and go through the same code inside Isaac."""
    bs = _load("scripts/build_starts.py", "build_starts")
    monkeypatch.setattr(sys, "argv", ["build_starts.py", "--kind", "simple", "--out",
                                      str(tmp_path), "--scene", "hollywood", "poster2"])
    assert bs.main() == 0
    for scene in ("hollywood", "poster2"):
        path = tmp_path / f"{scene}_starts_simple.json"
        payload = json.loads(path.read_text())
        assert payload["scene"] == scene and list(payload["cells"]) == ["simple"]
        assert payload["goal"]["distance_m"] == pytest.approx(
            benchmark.GOAL_DISTANCE_M[scene])
        assert len(starts.starts_for_cell(payload, "simple_5")) == 5
        assert path.with_suffix(".json.md5").is_file()


#: What `episode_check` returned before the debug telemetry (2026-09-30), pinned: an old
#: command must return exactly these keys and write exactly the old five frame keys.
OLD_EPISODE_KEYS = {
    "control_ms_med", "final_deg", "final_mm", "fps", "goal_mask_share", "gpu_util_max_pct",
    "gpu_util_mean_pct", "match_fps", "match_ms_med", "max_iter", "mean_n_matches", "outcome",
    "own_stop_deg", "own_stop_mm", "own_stop_rule", "own_stop_step", "peak_vram_alloc_mb",
    "perturbation", "prealign_deg", "sec_per_step", "start_deg", "start_deg_after_prealign",
    "start_id", "start_mm", "steps", "stop_deg", "stop_mm", "stop_step", "stops", "success",
    "terminated", "vram_used_mb"}


def test_check_baselines_old_path_is_unchanged(tmp_path):
    """`episode_check` without the debug keywords: the old key set, the old frame meta, the
    old files - and no telemetry anywhere."""
    pytest.importorskip("cv2")
    sys.path.insert(0, str(REPO / "scripts"))
    cb = _load("scripts/check_baselines.py", "check_baselines")
    tr = _load("tests/fast/test_runner.py", "_is_test_runner")
    scene = tr.make_scene()
    cfg = tr.make_config(max_iter=6)
    out = cb.episode_check(scene, tr.make_arm(scene, cfg), cfg, tr.make_start(scene), 6,
                           film_dir=tmp_path / "ep")
    assert set(out) == OLD_EPISODE_KEYS
    meta = json.loads((tmp_path / "ep" / "frames_meta.json").read_text())
    assert len(meta) == 6 and all(set(m) == {"file", "step", "t_mm", "r_deg", "n_matches"}
                                  for m in meta)
    assert sorted(p.name for p in (tmp_path / "ep").iterdir()) == ["episode.mp4",
                                                                   "frames_meta.json"]


def test_the_chain_flies_the_arms_and_flags_it_is_given(tmp_path):
    """2026-10-05: ARMS and EXTRA in the environment reach every check_baselines call; unset,
    the chain flies the ruled cells of 2026-09-23 with no extra flag."""
    argv = tmp_path / "argv.txt"
    _run_chain(tmp_path, pins_ok=True, env_extra={
        "FAKE_ARGV": str(argv), "ARMS": "vitvs_dinov3_objmask_ema08 dinobot_objmask",
        "EXTRA": "--telemetry --clean-film"})
    calls = argv.read_text().splitlines()
    assert len(calls) == 3
    assert all("--arms vitvs_dinov3_objmask_ema08 dinobot_objmask --near-steps" in c
               and "--outline-iou --telemetry --clean-film --perturb" in c for c in calls), calls
    default = tmp_path / "default"
    default.mkdir()
    argv2 = default / "argv.txt"
    _run_chain(default, pins_ok=True, env_extra={"FAKE_ARGV": str(argv2)})
    calls2 = argv2.read_text().splitlines()
    assert all("--arms dinobot_objmask vitvs_dinov3_objmask artvs_roi_scene --near-steps" in c
               and "--outline-iou --perturb" in c for c in calls2), calls2


def test_the_dinobot_budget_is_passed_only_when_dinobot_flies(tmp_path):
    """2026-10-06: check_baselines refuses a --budget for an arm it does not fly, so the chain
    passes DINOBot's 30-jump budget only when ARMS holds `dinobot_objmask`."""
    argv = tmp_path / "argv.txt"
    _run_chain(tmp_path, pins_ok=True, env_extra={"FAKE_ARGV": str(argv), "ARMS": "artvs_roi_sim"})
    calls = argv.read_text().splitlines()
    assert len(calls) == 3 and all("--budget" not in c for c in calls), calls
    assert all("--arms artvs_roi_sim --near-steps 200 --far 5 --far-set simple5" in c
               for c in calls), calls
    other = tmp_path / "with_dinobot"
    other.mkdir()
    argv2 = other / "argv.txt"
    _run_chain(other, pins_ok=True, env_extra={"FAKE_ARGV": str(argv2)})
    calls2 = argv2.read_text().splitlines()
    assert all("--near-steps 200 --budget dinobot_objmask=30 --far 5" in c for c in calls2), calls2


# ------------------------------------------------ the development set (owner 2026-10-10)
def test_the_dev_block_is_the_24_forward_pairs_of_the_dev_plan():
    """docs/dev_eval_plan_2026-10-10.md section 2: the six forward scenes x same / same_small /
    mine1 / other_small, in each scene's file order, a subset of the core block."""
    plan = _load("scripts/chains/instance_suite_plan.py", "instance_suite_plan")
    dev = plan.plan("dev")
    assert len(dev) == 24 == len(set(dev))
    assert set(dev) == {(s, n) for s in benchmark.DEV_SCENES for n in benchmark.DEV_LIVE_NAMES}
    assert benchmark.DEV_SCENES == ("inst_shoe", "inst_bottle", "inst_teapot", "inst_drill",
                                    "inst_mug", "inst_hammer")
    assert set(dev) <= set(plan.plan("core")) and not any(s.endswith("_rev") for s, _ in dev)
    assert dev[:4] == [("inst_shoe", "same"), ("inst_shoe", "same_small"),
                       ("inst_shoe", "other_small"), ("inst_shoe", "mine1")]
    with pytest.raises(ValueError, match="block must be one of"):
        plan.plan("devv")


def test_the_dev_starts_are_near_and_the_first_two_simple_starts():
    """Three starts per dev pair: the near sampler's first draw (`n_000`, check_baselines'
    `near_start`) and `--far 2 --far-set simple5` - a prefix of the pinned file = s_00, s_01
    on every dev scene."""
    assert benchmark.DEV_START_IDS == ("n_000", "s_00", "s_01") and benchmark.DEV_FAR == 2
    for scene in benchmark.DEV_SCENES:
        far = benchmark.select_starts(scene, "simple5", limit=benchmark.DEV_FAR)
        assert [f["start_id"] for f in far] == list(benchmark.DEV_START_IDS[1:])
    import numpy as np

    near = starts.sample_near(np.zeros(3), np.diag([1.0, -1.0, -1.0]), 0.25, n=1)[0]
    assert near["start_id"] == benchmark.DEV_START_IDS[0]


def test_the_dev_chain_flies_three_starts_with_the_debug_data(tmp_path):
    """dev_suite.sh = instance_suite.sh with BLOCK=dev, FAR=2 and EXTRA defaulting to the
    debug data; the arms are the caller's."""
    argv, plan_argv = tmp_path / "argv.txt", tmp_path / "plan.txt"
    rc, out, log = _run_chain(tmp_path, pins_ok=True, script="dev_suite.sh", env_extra={
        "FAKE_ARGV": str(argv), "FAKE_PLAN_ARGV": str(plan_argv), "ARMS": "artvs_roi_sim_dev"})
    assert rc == 0, log
    assert plan_argv.read_text().split() == ["scripts/chains/instance_suite_plan.py",
                                             "--block", "dev"]
    calls = argv.read_text().splitlines()
    assert len(calls) == 3 and all(
        "--arms artvs_roi_sim_dev --near-steps 200 --far 2 --far-set simple5" in c
        and "--outline-iou --telemetry --clean-film --perturb" in c and "--budget" not in c
        for c in calls), calls
    assert (out / "chain_settings.txt").read_text().strip() == \
        "far=2 arms=artvs_roi_sim_dev extra=--telemetry --clean-film"


def test_the_dev_chain_quick_variant_flies_the_near_start_only(tmp_path):
    argv = tmp_path / "argv.txt"
    rc, out, log = _run_chain(tmp_path, pins_ok=True, script="dev_suite.sh", env_extra={
        "FAKE_ARGV": str(argv), "ARMS": "dinobot_objmask", "TIER": "1a", "EXTRA": ""})
    assert rc == 0, log
    calls = argv.read_text().splitlines()
    assert len(calls) == 3 and all(
        "--near-steps 200 --budget dinobot_objmask=30 --far 0 --far-set simple5" in c
        and "--outline-iou --perturb" in c for c in calls), calls


@pytest.mark.parametrize("env", [{"ARMS": None}, {"OUT": ""}, {"TIER": "2"}])
def test_the_dev_chain_refuses_without_arms_out_or_a_known_tier(tmp_path, env):
    rc, out, log = _run_chain(tmp_path, pins_ok=True, script="dev_suite.sh",
                              env_extra={"ARMS": "artvs_roi_sim_dev", **env})
    assert rc == 2 and not (tmp_path / "calls.txt").exists()


def test_an_out_flown_with_other_starts_or_arms_is_refused(tmp_path):
    """A pair's DONE does not say which starts were flown: the near-only quick run and the
    full dev run must not share an OUT (the second would skip every pair)."""
    rc, out, log = _run_chain(tmp_path, pins_ok=True, script="dev_suite.sh",
                              env_extra={"ARMS": "artvs_roi_sim_dev", "TIER": "1a"})
    assert rc == 0
    n_calls = len((tmp_path / "calls.txt").read_text().splitlines())
    rc, out, log = _run_chain(tmp_path, pins_ok=True, script="dev_suite.sh",
                              env_extra={"ARMS": "artvs_roi_sim_dev"})
    assert rc == 1 and "refused" in log and "far=0" in log
    assert len((tmp_path / "calls.txt").read_text().splitlines()) == n_calls  # nothing flown
    # the same settings again: resumable as before (the clean pair is skipped)
    rc, out, log = _run_chain(tmp_path, pins_ok=True, script="dev_suite.sh",
                              env_extra={"ARMS": "artvs_roi_sim_dev", "TIER": "1a"})
    assert rc == 0 and "inst_shoe same skipped (DONE)" in log


def test_pairs_restricts_the_block_to_a_smoke(tmp_path):
    """PAIRS="scene:live ..." flies only the named pairs of the block, in the plan's order; a
    name outside the block is refused before anything flies."""
    rc, out, log = _run_chain(tmp_path, pins_ok=True, script="dev_suite.sh", env_extra={
        "ARMS": "artvs_roi_sim_dev", "PAIRS": "inst_drill:other_small inst_shoe:same"})
    assert rc == 0, log
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == ["inst_shoe swap:@same+ground:concrete",
                     "inst_drill swap:@other_small+ground:concrete"], calls
    other = tmp_path / "bad"
    other.mkdir()
    rc, out, log = _run_chain(other, pins_ok=True, script="dev_suite.sh", env_extra={
        "ARMS": "artvs_roi_sim_dev", "PAIRS": "inst_shoe:mine2"})
    assert rc == 1 and "PAIRS names 'inst_shoe:mine2'" in log
    assert not (other / "calls.txt").exists()


def test_a_pair_that_ran_out_of_gpu_memory_is_not_done(tmp_path):
    """The station's GPU is shared (2026-10-22): a render that ran out of memory leaves
    corrupted frames while the flight goes on, so its pair is never marked DONE."""
    rc, out, log = _run_chain(tmp_path, pins_ok=True, env_extra={"FAKE_OOM": "@same+"})
    assert rc == 0 and (out / "inst_shoe_same" / "check.json").is_file()
    assert not (out / "inst_shoe_same" / "DONE").exists()
    assert "inst_shoe same GPU out of memory - not DONE" in log


def test_the_chain_waits_for_free_gpu_memory_only_when_asked(tmp_path):
    rc, out, log = _run_chain(tmp_path, pins_ok=True, env_extra={
        "VSSIM_MIN_FREE_MB": "8000", "VSSIM_GPU_FREE_CMD": "echo 9000"})
    assert rc == 0 and (out / "inst_shoe_same" / "DONE").is_file() and "waiting" not in log
    (tmp_path / "b").mkdir()
    rc, out, log = _run_chain(tmp_path / "b", pins_ok=True)         # unset: never queried
    assert rc == 0 and (out / "inst_shoe_same" / "DONE").is_file()
