# vs-sim copy: tests/fast/test_films.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The film writer: composition geometry, per-step frames, a playable mp4, the hook."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench import films
from vsbench.vs.bench.arms import build_arm
from vsbench.vs.bench.runner import run_episode
from vsbench.vs.bench.truth import PlaneTruthMatcher
from vsbench.vs.scene import MockPlaneScene, Pose, rodrigues


def test_the_composition_is_goal_beside_current_under_a_caption_bar():
    goal = np.full((120, 200, 3), 60, np.uint8)
    cur = np.full((120, 200, 3), 120, np.uint8)
    frame = films.compose_match_frame(goal, cur, [[40, 30], [100, 60]], [[44, 31], [90, 70]],
                                      "caption", scale=0.5)
    assert frame.shape == (34 + 60, 200, 3)          # bar + two halves side by side
    assert (frame[:34].mean() < 40)                  # the dark caption bar
    # a green target dot on the goal half, a red matched dot on the current half
    assert tuple(frame[34 + 15, 20]) == films.GOAL_BGR
    assert tuple(frame[34 + 15, 100 + 22]) == films.CUR_BGR


def test_the_writer_keeps_step_one_and_every_nth_and_records_meta(tmp_path):
    goal = np.zeros((90, 160, 3), np.uint8)
    w = films.FilmWriter(tmp_path / "ep", goal, every=3, scale=0.5, title="arm")
    for i in range(1, 8):
        p = w.step(i, goal, {"pts_goal": [[10, 10]], "pts_cur": [[12, 11]], "n_matches": 1,
                             "feature_error_norm": 0.01}, t_mm=5.0, r_deg=0.5)
        assert (p is not None) == (i in (1, 3, 6)), i
    d = w.close()
    meta = json.loads((d / "frames_meta.json").read_text())
    assert [m["step"] for m in meta] == [1, 3, 6]
    assert meta[0]["n_matches"] == 1 and meta[0]["t_mm"] == 5.0
    # CNS's field names and a None point set are both accepted
    pg, pc = films.FilmWriter.points({"pts_desired": [[1, 2]], "pts_current": [[3, 4]]})
    assert pg.shape == (1, 2) and pc.shape == (1, 2)
    pg, pc = films.FilmWriter.points({"pts_goal": None, "pts_cur": None})
    assert pg.shape == (0, 2)


def test_three_frames_encode_to_a_video(tmp_path):
    pytest.importorskip("cv2")
    goal = np.zeros((90, 160, 3), np.uint8)
    w = films.FilmWriter(tmp_path / "ep", goal, scale=1.0)
    for i in range(1, 4):
        w.step(i, np.full((90, 160, 3), 40 * i, np.uint8), {"pts_goal": [], "pts_cur": []},
               t_mm=1.0, r_deg=0.1)
    w.close()
    out = films.encode_episode(tmp_path / "ep", scale=1.0, keep_frames=False)
    assert out is not None and out.name == "episode.mp4" and out.stat().st_size > 0
    assert not list((tmp_path / "ep").glob("step_*.jpg"))      # frames removed
    assert films.encode_episode(tmp_path / "ep") is None      # nothing left to encode


def test_the_runner_calls_the_hook_once_per_step_after_the_twist_exists():
    scene = MockPlaneScene(render_scale=0.125)
    goal = scene.goal_pose
    start = {"start_id": "t0", "cam_pos": goal.position + np.array([0.02, -0.015, 0.01]),
             "cam_R_opt": (goal.rotation @ rodrigues([0, 0, 1.0], 0.1)).tolist()}
    cfg = cfgmod.resolve({
        "seed": 20260902,
        "matcher": {"variant": "mutual_nn_seeded_subsample", "num_landmarks": 24,
                    "device": "cpu"},
        "controller": {"gain": 2.0}, "stop": {"rule": "cns_plateau"},
        "episode": {"max_iter": 12}, "scoring": {"success_rule": ["predicate_a"]},
        "arm": {"kind": "oracle"}})
    arm = build_arm(cfg, intrinsics=scene.intrinsics,
                    matcher=PlaneTruthMatcher(scene, num_landmarks=24, seed=cfg["seed"]))
    seen = []
    rec = run_episode(scene, arm, start, cfg,
                      on_step=lambda i, rgb, debug, pose: seen.append((i, rgb.shape, len(
                          debug["pts_cur"]), isinstance(pose, Pose))))
    assert rec["steps"] == 12 and len(seen) == 12
    assert seen[0][:2] == (1, (135, 240, 3)) and seen[0][3] is True and seen[-1][0] == 12
    assert 0 < seen[0][2] <= 24          # the truth matcher's visible landmarks that step


def test_films_new_captions_landing_and_final_frame(tmp_path):
    """The debug film (2026-09-30): the two-line caption with the method's name and the three
    distances, the landing border and its hold, the final frame sorted last - and the old
    call still gives the single-line bar and the five meta keys."""
    import cv2

    goal = np.full((90, 160, 3), 60, np.uint8)
    dbg = {"pts_goal": [[10, 10]], "pts_cur": [[12, 11]], "n_matches": 1,
           "feature_error_norm": 0.01}
    w = films.FilmWriter(tmp_path / "ep", goal, scale=1.0, title="ViT-VS  |  inst_shoe")
    refs = {"goal": 92.8, "size_matched": 10.1, "depth_true": None}
    w.step(1, goal, dbg, 92.8, 1.2, refs=refs, meta_extra={"kind": "step"})
    w.step(2, goal, dbg, 90.0, 1.0, refs=refs, landing=["the scored stop (cns_plateau)"])
    w.final(goal, refs=refs, r_deg=0.9, meta_extra={"kind": "final"})
    w.close()
    meta = json.loads((tmp_path / "ep" / "frames_meta.json").read_text())
    assert [m["step"] for m in meta] == [1, 2, 3] and meta[-1]["kind"] == "final"
    assert w.holds == {"step_00002.jpg": films.LANDING_HOLD_S,
                       "step_00003.jpg": films.FINAL_HOLD_S}
    one = cv2.imread(str(tmp_path / "ep" / "step_00001.jpg"))
    two = cv2.imread(str(tmp_path / "ep" / "step_00002.jpg"))
    assert one.shape[0] == 10 + 2 * 26 + 90 and two.shape[0] == 10 + 3 * 26 + 90
    # the landing border: yellow on the live half's edge (JPEG: close, not exact)
    edge = two[10 + 3 * 26 + 45, 160 + 2].astype(int)
    assert np.abs(edge - np.array(films.LANDING_BGR)).max() < 60
    frame = films.compose_match_frame(goal, goal, [], [], "c", scale=1.0,
                                      lines=["ViT-VS | step 1", "distance to goal 1.0 mm"])
    assert frame.shape[0] == 10 + 2 * 26 + 90
    # the old call: one line, five keys, no hold
    old = films.FilmWriter(tmp_path / "old", goal, scale=1.0, title="arm")
    old.step(1, goal, dbg, 5.0, 0.5)
    old.close()
    om = json.loads((tmp_path / "old" / "frames_meta.json").read_text())
    assert set(om[0]) == {"file", "step", "t_mm", "r_deg", "n_matches"} and old.holds == {}
    assert cv2.imread(str(tmp_path / "old" / "step_00001.jpg")).shape[0] == 34 + 90
    mp4 = films.encode_episode(tmp_path / "ep", holds=w.holds)
    assert mp4 is not None and mp4.stat().st_size > 0
