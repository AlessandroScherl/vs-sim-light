# vs-sim copy: tests/fast/test_benchmark.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The benchmark: six scenes x their pinned start files.

Guards `vssim/benchmark.py` and the files under `benchmark/starts/`: every pinned
file is present with its md5, its goal block is the scene's frozen goal, the four
start sets cut the populations the protocol states, and a drifted scene is refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vsbench.vs import benchmark
from vsbench.vs.camera import CameraConfig
from vsbench.vs.scene import MockPlaneScene, Pose
from vsbench.vs.scenes import EVAL_SCENES, load_scenes, poster_goal_pose

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def frozen_goals() -> dict:
    return json.loads((FIXTURES / "frozen_goals.json").read_text())["goals"]


def test_the_six_scenes_are_the_benchmark():
    assert benchmark.SCENES == EVAL_SCENES == (
        "hollywood", "poster2", "poster3", "industrial1_drill", "cluttered2", "cluttered3")
    assert set(benchmark.STARTS) == set(EVAL_SCENES) == set(benchmark.GOAL_DISTANCE_M)
    assert {"poster3", "cluttered3"} == benchmark.NEVER_SEEN


@pytest.mark.parametrize("scene", EVAL_SCENES)
def test_every_pinned_file_is_present_and_its_sidecar_agrees(scene):
    for which, (name, md5, seed) in benchmark.STARTS[scene].items():
        path = benchmark.starts_path(scene, which)
        assert path.name == name and path.is_file(), path
        assert hashlib.md5(path.read_bytes()).hexdigest() == md5, (scene, which)
        sidecar = path.with_suffix(".json.md5").read_text().split()
        assert sidecar == [md5, name]
        payload = benchmark.load_starts(scene, which)
        assert payload["scene"] == scene and payload["seed"] == seed
        assert payload["schema"] == "starts_v2"


@pytest.mark.parametrize("scene", EVAL_SCENES)
def test_the_files_goal_is_the_frozen_goal(scene, frozen_goals):
    payload = benchmark.load_starts(scene, "v2")
    g = payload["goal"]
    assert g["distance_m"] == benchmark.GOAL_DISTANCE_M[scene]
    for key in ("cam_pos", "cam_R_opt", "distance_m", "look_at"):
        assert g[key] == frozen_goals[scene][key], (scene, key)
    if "near" in benchmark.STARTS[scene]:
        near = benchmark.load_starts(scene, "near")
        assert near["goal"] == g                    # the same goal in both files


def test_the_poster_goals_are_derivable_offline_and_match_the_files():
    cam = CameraConfig()
    specs, _ = load_scenes()
    for scene in ("hollywood", "poster2", "poster3"):
        pos, R, d = poster_goal_pose(cam, specs[scene])
        g = benchmark.load_starts(scene, "v2")["goal"]
        assert d == g["distance_m"]
        np.testing.assert_array_equal(pos, g["cam_pos"])
        np.testing.assert_array_equal(R, g["cam_R_opt"])


@pytest.mark.parametrize("scene", EVAL_SCENES)
def test_the_start_sets_cut_the_stated_populations(scene):
    capped = benchmark.select_starts(scene, "capped30")
    assert len(capped) == 30
    assert all(abs(s["roll_deg"]) <= 90.0 for s in capped)
    assert all(s["cell"] == "wide100" and s["starts_seed"] for s in capped)
    wide = benchmark.select_starts(scene, "wide100")
    assert len(wide) == 100
    ids = [s["start_id"] for s in wide]
    assert [s["start_id"] for s in capped] == [i for i in ids if i in
                                               {s["start_id"] for s in capped}]
    assert len(benchmark.select_starts(scene, "wide100", limit=7)) == 7
    for s in wide[:3]:
        pose = Pose.from_start(s)
        assert pose.position.shape == (3,) and pose.rotation.shape == (3, 3)
        np.testing.assert_allclose(pose.rotation @ pose.rotation.T, np.eye(3), atol=1e-12)


def test_the_poster_only_sets_refuse_a_room():
    assert len(benchmark.select_starts("hollywood", "narrow20")) == 20
    assert len(benchmark.select_starts("poster3", "near60")) == 60
    with pytest.raises(ValueError, match="not defined for scene"):
        benchmark.select_starts("cluttered2", "near60")
    with pytest.raises(KeyError, match="unknown start set"):
        benchmark.select_starts("hollywood", "wide30")
    with pytest.raises(KeyError, match="no 'near' start file"):
        benchmark.starts_path("cluttered2", "near")


def test_check_scene_goal_accepts_the_mock_posters_goal_and_refuses_a_drift():
    """The mock plane IS the posters' goal (0.8 x 0.6 m at 0.4342 m), so the real
    hollywood start file pairs with it; a millimetre of drift is refused."""
    scene = MockPlaneScene(render_scale=0.125)
    payload = benchmark.load_starts("hollywood", "v2")
    benchmark.check_scene_goal(scene, payload)                 # no exception
    drifted = MockPlaneScene(render_scale=0.125, working_distance=0.4341815107823003 + 2e-3)
    with pytest.raises(ValueError, match="goal mismatch"):
        benchmark.check_scene_goal(drifted, payload)


def test_the_readme_lists_every_pin():
    """benchmark/README.md is the human copy of the pins; it must not fork."""
    text = (benchmark.STARTS_DIR.parent / "README.md").read_text()
    for scene, files in benchmark.STARTS.items():
        for _which, (name, md5, _seed) in files.items():
            assert name in text and md5 in text, (scene, name)


def test_the_extra_scene_is_pinned_beside_the_six_and_its_file_loads():
    """`shoe1` (2026-09-20) lives in EXTRA_* so the six stay the six; its start file is
    md5-pinned and md5-checked on the way in like theirs."""
    assert benchmark.SCENES == EVAL_SCENES and "shoe1" not in benchmark.STARTS
    assert benchmark.EXTRA_SCENES == ("shoe1", "shoe2")
    assert benchmark.load_starts("shoe2")["goal"]["distance_m"] == pytest.approx(0.215494, abs=1e-6)
    payload = benchmark.load_starts("shoe1")
    assert payload["goal"]["distance_m"] == pytest.approx(benchmark.EXTRA_GOAL_DISTANCE_M["shoe1"])
    assert len(benchmark.select_starts("shoe1", "capped30")) == 30
    assert benchmark.select_starts("shoe1", "capped30", limit=2)[0]["start_id"]
    with pytest.raises(KeyError, match="no start file pinned"):
        benchmark.load_starts("shoe9")



def test_the_instance_suite_is_registered_beside_the_six():
    """Twelve scenes, their simple start sets, and no silent fallback before the station
    has built and pinned their files."""
    specs, _ = load_scenes()
    assert benchmark.SCENES == EVAL_SCENES                   # the six stay the six
    assert len(benchmark.INSTANCE_SCENES) == 12
    assert all(s in specs for s in benchmark.INSTANCE_SCENES)
    assert not set(benchmark.INSTANCE_SCENES) & set(benchmark.STARTS)
    for name, n in (("simple5", 5), ("simple10", 10)):
        spec = benchmark.START_SETS[name]
        assert (spec["file"], spec["source"], spec["n"]) == ("simple", "simple", n)
    for scene in benchmark.INSTANCE_SCENES:
        if scene not in benchmark.INSTANCE_STARTS:           # not built on the station yet
            with pytest.raises(KeyError, match="no start file pinned"):
                benchmark.select_starts(scene, "simple5")
        else:
            assert len(benchmark.select_starts(scene, "simple5")) == 5
    with pytest.raises(KeyError, match="no 'simple' start file"):
        benchmark.select_starts("hollywood", "simple5")
