# vs-sim copy: tests/fast/test_t1_near_starts.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""T1 Stage 2's near start set (refine-vs, owner 2026-10-22): three near starts per dev scene
from the benchmark's near sampler with a new seed, beside the frozen files."""

import numpy as np
import pytest

from vsbench.vs import benchmark
from vsbench.vs import starts as startsmod

SCENES = ("inst_shoe", "inst_bottle", "inst_teapot", "inst_drill", "inst_mug", "inst_hammer")


@pytest.mark.parametrize("scene", SCENES)
def test_t1near3_is_three_near_starts_on_the_pinned_goal(scene):
    sel = benchmark.select_starts(scene, "t1near3")
    assert [s["start_id"] for s in sel] == ["t1n_000", "t1n_001", "t1n_002"]
    assert all(s["starts_seed"] == 20261022 for s in sel)
    goal = benchmark.load_starts(scene, "simple")["goal"]
    near = benchmark.load_starts(scene, "t1near")
    assert near["goal"] == goal                       # the same goal as the frozen file
    g = np.asarray(goal["cam_pos"])
    Rg = np.asarray(goal["cam_R_opt"])
    for s in sel:
        mm = 1000 * np.linalg.norm(np.asarray(s["cam_pos"]) - g)
        assert 3.0 - 1e-6 <= mm <= 8.0 + 1e-6
        R = Rg.T @ np.asarray(s["cam_R_opt"])
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert ang <= 3.0 + 1e-6


def test_the_draw_does_not_depend_on_the_id_prefix():
    a = startsmod.sample_near(np.zeros(3), np.eye(3), seed=5, n=4)
    b = startsmod.sample_near(np.zeros(3), np.eye(3), seed=5, n=4, id_prefix="x_")
    assert [s["cam_pos"] for s in a] == [s["cam_pos"] for s in b]
    assert [s["start_id"] for s in b] == ["x_000", "x_001", "x_002", "x_003"]
    assert [s["start_id"] for s in a][0] == "n_000"


def test_the_frozen_sets_are_unchanged():
    assert benchmark.select_starts("inst_mug", "simple5")[0]["start_id"] == "s_00"
    assert "t1near" not in benchmark.INSTANCE_STARTS["inst_mug_rev"]
