# vs-sim copy: tests/fast/test_runner.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The episode loop, closed on the mock scene: convergence, schema, telemetry, fences.

The mock is a textured plane seen through a scaled D435i and a matcher that answers
the correspondence question analytically for that plane, so this file measures the
**runner** - the order of operations, the record, the telemetry, the stop
bookkeeping and the GT-blind fence - and not a matcher.

The two frozen references:

- ``tests/fixtures/u7_record_schema.json`` - one real N3.7b record with its values
  replaced by their types (84 fields). Every measurement field must survive the
  port; every arm-suffix flag must be gone.
- ``tests/fixtures/u7_n38_telemetry_step.json`` - one step of an N3.8
  ``--telemetry-extended`` episode (31 keys). The extended entry must carry exactly
  those keys and the default entry exactly those minus the eight extended ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vsbench.vs.bench import arms, runner, scoring
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench.runner import EXTENDED_KEYS
from vsbench.vs.bench.truth import PlaneTruthMatcher
from vsbench.vs.scene import (
    D435I,
    MockPlaneScene,
    Pose,
    integrate_optical_velocity,
    scaled_intrinsics,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Fields of the frozen N3.7b record that are measurements or identity and must
#: survive the port unchanged in meaning.
SCHEMA_KEPT = (
    "scene", "start_id", "start_cell", "start_tier", "start_roll_deg", "starts_seed",
    "subsample_seed", "gain", "max_iter", "ema_alpha",
    "steps", "terminated_reason", "wall_time_s", "sec_per_step",
    "start_translation_mm", "start_rotation_deg",
    "final_translation_mm", "final_rotation_deg",
    "feature_error_first", "feature_error_last", "mean_n_matches", "min_n_matches",
    "stop_mode", "stop_probe", "stop_step", "stop_translation_mm", "stop_rotation_deg",
    "stop_trigger_step", "stop_cfg", "lam_scale_final", "lam_scale_min", "frz_n",
    "converged_a", "diverged", "outcome", "peak_vram_alloc_mb", "vram_used_mb",
)

#: Fields of the frozen record that this port deliberately drops. Each group has its
#: reason in `vssim/bench/runner.py` and `arms.py`; the test exists so a drop is a
#: decision rather than an oversight.
SCHEMA_DROPPED = {
    # the arm-suffix grammar, replaced by the explicit registry (arms.py)
    "method", "base_method", "refine_mode", "refine_tag", "refine_conf_tag", "refine_ema",
    "refine_iters", "refine_axw", "refine_alw", "refine_lw_tag", "refine_ur", "refine_rot",
    "refine_rotf", "refine_k", "refine_avg", "refine_ana", "refine_hw", "refine_irls",
    "refine_cz", "refine_sel", "refine_sel2", "refine_mw", "refine_fs", "refine_dopt",
    "refine_tsel", "refine_tsel2", "refine_cons", "refine_consr", "twin_label",
    "conf_gamma", "vit_ransac", "oracle_mode",
    # the D6 termination reads: reporting-only, superseded by the scored stop
    "term_step", "term_translation_mm", "term_rotation_deg", "term_consec_step",
    "term_consec_translation_mm", "term_consec_rotation_deg", "termination_cfg",
    # the `frz` stop mechanic: frozen landmarks are a matcher variant now
    "frz_step", "frz_unfreezes",
    # rotation: the mode string and the parked in-loop stack's three columns. U8
    # replaced the grammar with a `rotation:` config block, so `prealign_mode` is
    # `rotation.mode` and `rot`/`rot_detector`/`rot_cold_source` describe the parked
    # delta-theta stack that did not come across at all (the research repository, "dropped").
    "prealign_mode", "rot", "rot_detector", "rot_cold_source",
    # the rotsym roll-bucket stratification: superseded by `roll_deg` on the start
    # plus protocol v2's thirds (`vsbench.vs.bench.stats.roll_thirds`)
    "roll_bucket_deg",
}

#: Fields of the frozen record that U8 brought BACK, with their meaning intact.
#: ``prealign`` is the published 4-way's presence, and protocol v2 section 4 asks every
#: table to state it - so it is on every ViT row, ``None`` when it did not run
#: (``vsbench.vs.baselines.match.DenseMatcher.record_fields``, ``vsbench.vs.baselines.prealign`` fence 5).
SCHEMA_REVIVED = ("prealign",)

BASE_CONFIG = {
    "seed": 20260902,
    # `PlaneTruthMatcher` is a ground-truth matcher, not the variant it is injected
    # against, so every mock arm declares itself: `build_arm` refuses the mismatch
    # otherwise, and the record and the label carry the kind (review R13, C2).
    "arm": {"kind": "oracle"},
    # `device` has no default for an encoder variant (R22 correction 3); the fast tier
    # states `cpu` even where the encoder is a stub, because the LOADER refuses the
    # absence and a stub is not an excuse for an unstated setting.
    "matcher": {"variant": "mutual_nn_seeded_subsample", "num_landmarks": 24,
                "device": "cpu"},
    "controller": {"gain": 2.0},
    "stop": {"rule": "cns_plateau"},
    "episode": {"max_iter": 220},
    "scoring": {"success_rule": ["predicate_a", "paper_3cm_3deg", "cns_code_0p1d_10deg"]},
}


def make_config(**episode) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CONFIG.items()}
    cfg["episode"] = {**cfg["episode"], **episode}
    return cfgmod.resolve(cfg)


def deep(cfg: dict) -> dict:
    """A mutable copy one level deep (the config blocks are flat dicts)."""
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
            for k, v in cfg.items()}


def make_scene(**kw) -> MockPlaneScene:
    """The mock at 1/8 sampling of the real camera: 240 x 135, ~5 ms a frame."""
    return MockPlaneScene(render_scale=0.125, **kw)


def make_start(scene, *, dx=0.02, dy=-0.015, dz=0.01, roll_rad=0.1, start_id="t0") -> dict:
    g = scene.goal_pose
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return {"start_id": start_id, "cam_pos": g.position + np.array([dx, dy, dz]),
            "cam_R_opt": (g.rotation @ rz).tolist(), "cell": "mock", "tier": "mock",
            "roll_deg": float(np.degrees(roll_rad)), "starts_seed": 20260902}


def make_arm(scene, cfg, *, pixel_noise=0.0):
    matcher = PlaneTruthMatcher(scene, num_landmarks=cfg["matcher"]["num_landmarks"],
                                seed=cfg["seed"], pixel_noise=pixel_noise)
    return arms.build_arm(cfg, intrinsics=scene.intrinsics, matcher=matcher)


@pytest.fixture(scope="module")
def converged_episode():
    """One clean episode, reused by the schema and telemetry tests."""
    scene, cfg = make_scene(), make_config(max_iter=260, telemetry_extended=True)
    telem: list = []
    rec = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg,
                             telemetry=telem)
    return rec, telem


# ------------------------------------------------------------- the mock world
def test_the_mock_scene_is_a_consistent_camera():
    scene = make_scene()
    intr = scene.intrinsics
    assert (intr.width, intr.height) == (240, 135)
    assert intr.fx == pytest.approx(D435I.fx * 0.125)
    assert scaled_intrinsics(D435I, 1.0) == D435I
    rgb, depth, pose = scene.render()
    assert rgb.shape == (135, 240, 3) and rgb.dtype == np.uint8
    # the goal view is fronto-parallel, so every pixel that sees the plane is at d
    assert depth[depth > 0] == pytest.approx(scene.working_distance)
    assert pose.position == pytest.approx(scene.goal_pose.position)


def test_pixels_round_trip_through_the_plane():
    scene = make_scene()
    px = np.array([[30.0, 20.0], [120.0, 67.5], [200.5, 100.25]])
    pts, ok = scene.backproject(px, scene.goal_pose)
    assert ok.all()
    back, valid = scene.project(pts, scene.goal_pose)
    assert valid.all() and back == pytest.approx(px, abs=1e-9)
    # and the same points seen from a displaced pose are the GT correspondences
    other = Pose(scene.goal_pose.position + np.array([0.01, 0.0, 0.0]), scene.goal_pose.rotation)
    uv, valid = scene.gt_correspondence(px, other)
    assert valid.all() and not np.allclose(uv, px)


def test_the_twist_integration_is_the_old_convention():
    """``pos += R @ v_lin * dt``; ``R <- R @ Rodrigues(w_hat, |w| dt)``."""
    R = np.diag([1.0, -1.0, -1.0])
    pose = Pose([0.1, 0.2, 0.3], R)
    out = integrate_optical_velocity(pose, [1.0, 0.0, 0.0, 0.0, 0.0, np.pi], 0.5)
    assert out.position == pytest.approx([0.6, 0.2, 0.3])       # x moves along R's x column
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert out.rotation == pytest.approx(R @ quarter_turn, abs=1e-12)   # |w| dt = pi/2
    assert pose.position == pytest.approx([0.1, 0.2, 0.3])      # pure: input untouched


# -------------------------------------------------------------- closed loop
def test_an_episode_from_a_small_offset_converges(converged_episode):
    rec, telem = converged_episode
    assert rec["start_translation_mm"] == pytest.approx(26.9258, abs=1e-3)
    assert rec["final_translation_mm"] < 0.05          # mm, from 27 mm
    assert rec["final_rotation_deg"] < 0.01            # deg, from 5.73
    assert rec["converged_a"] is True and rec["outcome"] == "converged"
    assert rec["success"] == {"predicate_a": True, "paper_3cm_3deg": True,
                              "cns_code_0p1d_10deg": True}
    assert rec["terminated_reason"] == "max_iter" and rec["steps"] == 260
    assert len(telem) == 260
    assert [e["i"] for e in telem] == list(range(1, 261))
    # the published velocity-quiet stop is recorded on every arm (protocol v2 section 5)
    assert rec["stops"]["velocity_quiet"]["step"] < 260
    assert telem[0]["t_mm"] > telem[-1]["t_mm"]        # it really went somewhere


def test_the_draw_is_a_function_of_scene_start_and_seed_only():
    """Two arms at the same start see the same landmarks - what makes a read paired."""
    cfg = make_config(max_iter=3)
    recs = []
    for _ in range(2):
        scene = make_scene()
        recs.append(runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg))
    assert recs[0]["subsample_seed"] == recs[1]["subsample_seed"]
    assert recs[0]["final_translation_mm"] == recs[1]["final_translation_mm"]
    other = make_config(max_iter=3)
    other["seed"] = 7
    scene = make_scene()
    assert runner.run_episode(scene, make_arm(scene, other), make_start(scene),
                              other)["subsample_seed"] != recs[0]["subsample_seed"]


# ------------------------------------------------------------- record schema
def test_the_record_keeps_every_measurement_field_of_the_frozen_schema(converged_episode):
    rec, _ = converged_episode
    frozen = json.loads((FIXTURES / "u7_record_schema.json").read_text())["types"]
    assert len(frozen) == 84
    for field in SCHEMA_KEPT:
        assert field in frozen, f"{field} is not in the frozen schema - fix the test"
        assert field in rec, f"the port dropped the measurement field {field}"
    kept_or_dropped = set(SCHEMA_KEPT) | SCHEMA_DROPPED | set(SCHEMA_REVIVED)
    assert set(frozen) - kept_or_dropped == set(), "a frozen field is neither kept nor dropped"


def test_the_record_carries_the_live_frame_brightness_at_the_park(converged_episode):
    """The other half of the settle argument (R25 §2, R26 correction 8): a column, not
    a dump. `None` without extended telemetry, because it is measured per step."""
    rec, telem = converged_episode
    assert rec["live_rgb_mean_last50"] == pytest.approx(
        float(np.mean([e["fm"] for e in telem[-runner.LIVE_MEAN_WINDOW:]])), abs=1e-3)
    scene, cfg = make_scene(), make_config(max_iter=6, telemetry_extended=False)
    plain = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg)
    assert plain["live_rgb_mean_last50"] is None


def test_the_record_says_how_the_goal_frame_was_captured(converged_episode):
    """Protocol v2.2 + the settle finding of 2026-09-05. A mock scene captures no goal
    the way Isaac does, so it says `None` three times rather than leaving the columns
    out: "not measured" and "not applicable" read the same way in a table and only one
    of them is true. An Isaac scene fills all three from one place."""
    rec, _ = converged_episode
    for field in ("goal_render_warmup", "goal_capture_keep", "goal_settle_series",
                  "goal_frame_md5"):
        assert field in rec and rec[field] is None

    class _Scene:
        def goal_capture_record(self):
            return {"goal_render_warmup": 3, "goal_frame_md5": "abc",
                    "goal_capture_keep": "last",
                    "goal_settle_series": {"n_captures": 3, "rgb_mean": [1.0, 1.1, 1.2],
                                           "diff_prev": [None, 0.9, 0.56]}}

    fields = runner._goal_capture_fields(_Scene())
    assert fields["goal_render_warmup"] == 3 and fields["goal_frame_md5"] == "abc"
    assert fields["goal_settle_series"]["diff_prev"][-1] == 0.56
    assert runner._goal_capture_fields(object()) == runner.NO_GOAL_CAPTURE


def test_the_record_types_match_the_frozen_schema(converged_episode):
    rec, _ = converged_episode
    frozen = json.loads((FIXTURES / "u7_record_schema.json").read_text())["types"]
    equivalent = {"int": {"int"}, "float": {"float", "int"}, "bool": {"bool"},
                  "str": {"str"}, "dict": {"dict"}, "NoneType": {"NoneType"}}
    for field in SCHEMA_KEPT:
        want, got = frozen[field], type(rec[field]).__name__
        if want == "NoneType" or got == "NoneType":
            continue                          # the frozen record's Nones are absences
        assert got in equivalent[want], f"{field}: {got} is not a {want}"


def test_the_arm_suffix_flags_are_gone(converged_episode):
    rec, _ = converged_episode
    assert not (set(rec) & SCHEMA_DROPPED)
    # what replaces them: the settings themselves, on the record
    assert rec["matcher_variant"] == "mutual_nn_seeded_subsample"
    assert rec["num_landmarks"] == 24
    assert rec["solve_unrounded"] is True
    assert rec["weighting"]["trust_gamma"] == 3.0
    assert rec["checkpoint"] is None
    assert rec["working_distance_m"] == pytest.approx(0.4341815107823003)
    # the record names the matcher OBJECT, not only the variant the config declared,
    # and an arm that is not a method says so in both the record and the label
    assert rec["matcher_name"] == "plane_truth" != rec["matcher_variant"]
    assert rec["arm_kind"] == "oracle" and rec["arm"].startswith("oracle:")
    # the camera and the goal travel with the run, so telemetry re-solves offline
    assert rec["img_wh"] == [240, 135]
    assert rec["intrinsics"]["fx"] == pytest.approx(D435I.fx * 0.125)
    assert rec["goal_pos"] == [0.0, 0.0, round(0.4341815107823003, 6)]
    # the stop constants on the record are the EFFECTIVE ones, not the overrides
    assert rec["stop_cfg"]["waiting_time"] == 0.8 and rec["stop_cfg"]["dt"] == rec["dt"]


# ------------------------------------------------------------- the telemetry
def test_the_extended_telemetry_is_the_n38_keys_plus_the_live_frame_mean(converged_episode):
    """N3.8's 31 columns exactly, plus `fm` — and nothing else.

    `fm` is the live frame's mean grey level (review R26 correction 8): the goal
    capture records its brightness per capture, so the frames the loop actually steered
    on have to record theirs, or "settled live frames against an unsettled goal" stays
    a dump on a station disk. It is the one column N3.8's stored telemetry does not
    have, which is why an offline reader must tolerate its absence and why it is named
    apart from `EXTENDED_KEYS`.
    """
    _, telem = converged_episode
    frozen = json.loads((FIXTURES / "u7_n38_telemetry_step.json").read_text())
    assert frozen["n_keys"] == 31
    assert set(telem[5]) == set(frozen["step"]) | set(runner.EXTENDED_KEYS_AFTER_N38)
    assert "fm" not in frozen["step"]


def test_the_default_telemetry_is_the_extended_one_minus_the_eight_n38_columns():
    scene, cfg = make_scene(), make_config(max_iter=6, telemetry_extended=False)
    telem: list = []
    runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg, telemetry=telem)
    frozen = set(json.loads((FIXTURES / "u7_n38_telemetry_step.json").read_text())["step"])
    assert set(EXTENDED_KEYS) < frozen
    assert set(telem[3]) == frozen - set(EXTENDED_KEYS)
    assert "fm" not in telem[3], "the live-frame mean is an EXTENDED column only"


@pytest.mark.parametrize("extended,decimals", [(False, 2), (True, 4)])
def test_point_precision_is_two_decimals_by_default_and_four_extended(extended, decimals):
    """0.005 px of storage quantisation moves an offline re-solve by ~1e-2 relative."""
    scene, cfg = make_scene(), make_config(max_iter=6, telemetry_extended=extended)
    telem: list = []
    runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg, telemetry=telem)
    values = [v for e in telem for k in ("gu", "gv", "cu", "cv") for v in e[k]]
    assert values and all(v == round(v, decimals) for v in values)
    assert any(v != round(v, decimals - 1) for v in values)     # it really is finer


def test_the_telemetry_ground_truth_is_logging_only_and_can_be_switched_off():
    scene, cfg = make_scene(), make_config(max_iter=6, telemetry_extended=True)
    with_gt: list = []
    a = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg,
                           telemetry=with_gt)
    without: list = []
    scene2 = make_scene()
    b = runner.run_episode(scene2, make_arm(scene2, cfg), make_start(scene2), cfg,
                           telemetry=without, ground_truth=False)
    assert a["final_translation_mm"] == b["final_translation_mm"]     # the loop is identical
    assert {"gt_v", "gt_r", "gt_u", "gt_ok", "err_patch"} <= set(with_gt[0])
    assert not ({"gt_v", "gt_r", "gt_u", "gt_ok", "err_patch"} & set(without[0]))
    # the truth matcher is right by construction, so every match it kept is "right"
    n = len(with_gt[0]["gu"])
    assert 20 <= n <= 24                       # a few goal patches leave the start view
    assert with_gt[0]["gt_v"] == with_gt[0]["gt_r"] == n
    assert max(with_gt[0]["err_patch"]) < 1e-6


def test_the_linear_cell_indices_are_the_patch_grid(converged_episode):
    _, telem = converged_episode
    entry = telem[-1]
    assert len(entry["g_lin"]) == len(entry["c_lin"]) == len(entry["gu"])
    assert all(0 <= v < 32 * 32 for v in entry["g_lin"] + entry["c_lin"])
    # at the goal the current cells ARE the goal cells: the loop landed on them
    assert entry["g_lin"] == entry["c_lin"]


# ------------------------------------------------------------ the stop wiring
def test_the_stop_is_recorded_and_never_acted_on():
    """Protocol v2 section 5: the loop runs to the cap, so both reads exist."""
    scene, cfg = make_scene(), make_config(max_iter=220)
    rec = runner.run_episode(scene, make_arm(scene, cfg, pixel_noise=0.5),
                             make_start(scene), cfg)
    assert rec["stop_mode"] == "cns_plateau"
    assert rec["stop_probe"] is True
    assert rec["stop_step"] is not None and rec["stop_step"] < rec["steps"] == 220
    assert rec["terminated_reason"] == "max_iter"
    assert scoring.read(rec, "stop")["step"] == rec["stop_step"]
    assert scoring.read(rec, "cap")["translation_mm"] == rec["final_translation_mm"]
    # this arm parks at a biased pose, so the published quiet stop never fires here -
    # it is armed on every arm regardless, which is what "recorded for every arm" means
    assert set(rec["stops"]) == {"cns_plateau"}
    assert "velocity_quiet" in arms.episode_settings(cfg)["record_stops"]


def test_a_still_descending_error_is_not_a_plateau():
    """With a perfect matcher the error never floors inside the budget, so CNS's rule
    correctly refuses to fire - the behaviour review R12 had to restore."""
    scene, cfg = make_scene(), make_config(max_iter=220)
    rec = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg)
    assert rec["stop_step"] is None
    assert rec["final_translation_mm"] < 0.05


def test_the_lambda_schedule_scales_the_commanded_twist():
    scene = make_scene()
    cfg = cfgmod.resolve({**BASE_CONFIG, "stop": {"rule": "dec_plateau", "window": 20,
                                                  "rel_tol": 0.5, "sustain": 1},
                          "episode": {"max_iter": 120}})
    telem: list = []
    rec = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg,
                             telemetry=telem)
    assert rec["stop_trigger_step"] == rec["plateau_step"] is not None
    # C1: `stop_step` is the published quiet criterion under the decayed lambda, which
    # is what every stored `dec` record means - never the plateau trigger itself
    assert rec["stop_step"] is None or rec["stop_step"] > rec["stop_trigger_step"]
    assert rec["lam_scale_min"] < 1.0 and rec["lam_scale_final"] == rec["lam_scale_min"]
    lam = [e["lam"] for e in telem]
    assert lam[0] == pytest.approx(2.0)                       # the base gain
    assert lam[-1] == pytest.approx(2.0 * rec["lam_scale_final"], abs=1e-5)


def _scripted(arm, *, feature_errors=None, nan_twist_from=None):
    """Wrap a real arm so one step's *controller-visible* signals can be dictated.

    The mock plane is too well conditioned to diverge - a weighted least squares on
    a plane always reduces the feature error, whatever the matcher says - so the two
    early-exit paths are driven directly. The arm stays real (its controller and its
    stop rule are the ones `assert_gt_blind` walks); only the numbers the loop reads
    are scripted.
    """
    from vsbench.vs.baselines.control import Twist

    real = arm.step
    state = {"i": 0}

    def step(rgb, depth, mask=None):
        twist, debug, matches = real(rgb, depth, mask)
        state["i"] += 1
        if feature_errors is not None:
            debug["feature_error_norm"] = feature_errors[min(state["i"] - 1,
                                                             len(feature_errors) - 1)]
            twist = Twist(np.zeros(6))
        if nan_twist_from is not None and state["i"] >= nan_twist_from:
            twist = Twist(np.full(6, np.nan))
        return twist, debug, matches

    arm.step = step
    return arm


def test_the_divergence_abort_ends_the_episode_on_the_capture_pose():
    """The N0 abort: controller-visible feature error above 3x its FIRST FINITE value."""
    scene, cfg = make_scene(), make_config(max_iter=40)
    errors = [1.0] * 9 + [3.5] + [1.0] * 40          # step 10 crosses 3 x 1.0
    arm = _scripted(make_arm(scene, cfg), feature_errors=errors)
    telem: list = []
    rec = runner.run_episode(scene, arm, make_start(scene), cfg, telemetry=telem)
    assert rec["terminated_reason"] == "diverged_feature"
    assert rec["steps"] == 10
    assert rec["diverged"] is True and rec["outcome"] == "diverged"
    # the aborting step is on film, and its telemetry pose is the record's final pose
    assert len(telem) == 10 and telem[-1]["i"] == 10
    assert telem[-1]["t_mm"] == pytest.approx(rec["final_translation_mm"], abs=1e-3)
    assert rec["feature_error_first"] == 1.0 and rec["feature_error_last"] == 3.5

    off = make_config(max_iter=40, divergence_abort=False)
    scene2 = make_scene()
    kept = _scripted(make_arm(scene2, off), feature_errors=errors)
    rec2 = runner.run_episode(scene2, kept, make_start(scene2), off)
    assert rec2["steps"] == 40 and rec2["terminated_reason"] == "max_iter"


def test_a_non_finite_twist_ends_the_episode_as_aborted():
    scene, cfg = make_scene(), make_config(max_iter=30)
    arm = _scripted(make_arm(scene, cfg), nan_twist_from=4)
    telem: list = []
    rec = runner.run_episode(scene, arm, make_start(scene), cfg, telemetry=telem)
    assert rec["terminated_reason"] == "non_finite_velocity"
    assert rec["steps"] == 4 and len(telem) == 4     # the poisoned step is on film
    assert rec["outcome"] == "aborted"               # the class the old taxonomy hid


# ------------------------------------------------------------------ the fence
def test_the_control_path_cannot_see_the_scene():
    scene, cfg = make_scene(), make_config(max_iter=3)
    arm = make_arm(scene, cfg)
    runner.assert_gt_blind(arm, scene)                       # a clean arm passes
    arm.stop._sneaky_scene = scene
    with pytest.raises(AssertionError, match="GT-blind"):
        runner.assert_gt_blind(arm, scene)
    with pytest.raises(AssertionError, match="GT-blind"):
        runner.run_episode(scene, arm, make_start(scene), cfg)
    del arm.stop._sneaky_scene
    arm.controller._goal = scene.goal_pose
    with pytest.raises(AssertionError, match="GT-blind"):
        runner.assert_gt_blind(arm, scene)


def test_the_stop_rules_only_ever_see_a_stop_observation():
    """The type is the fence: no pose, no goal, no scene - only what the loop measured."""
    from dataclasses import fields

    from vsbench.vs.baselines.control import StopObservation

    # `pixel_error_mean` (2026-10-05): the controller's own points' mean pixel distance
    assert {f.name for f in fields(StopObservation)} == {
        "step", "feature_error_norm", "velocity_norm", "normalised_errors",
        "landmark_ids", "t", "pixel_error_mean"}


# ------------------------------------------------------------- the run sidecar
def test_the_runner_writes_its_own_sidecar(tmp_path):
    scene, cfg = make_scene(), make_config(max_iter=5)
    telem: list = []
    rec = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg,
                             telemetry=telem)
    with runner.open_run("mock_smoke", cfg, tmp_path, telemetry=True,
                         allow_dirty=True, allow_no_git=True) as sink:
        sink.write(rec, telem)
    written = [p.name for p in sink.path.iterdir()]
    assert set(written) == {"resolved_config.yaml", "run.json", "MD5SUMS",
                            "baseline_check.jsonl", "telemetry.jsonl.gz"}
    meta = json.loads((sink.path / "run.json").read_text())
    assert meta["seed"] == cfg["seed"] and meta["allow_dirty"] is True
    assert meta["finished_utc"] is not None
    back = [json.loads(x) for x in
            (sink.path / "baseline_check.jsonl").read_text().splitlines() if x.strip()]
    assert len(back) == 1 and back[0]["arm"] == rec["arm"]
    import gzip
    with gzip.open(sink.path / "telemetry.jsonl.gz", "rt") as fh:
        episode = json.loads(fh.readline())
    meta = episode["meta"]
    # what an offline re-solve reads: the camera, the goal and which matcher ran
    assert meta["img_wh"] == [240, 135] and meta["goal_pos"] == rec["goal_pos"]
    assert set(meta["intrinsics"]) == {"fx", "fy", "cx", "cy"}
    assert meta["matcher_name"] == "plane_truth" and meta["arm_kind"] == "oracle"
    assert meta["stop_cfg"] == rec["stop_cfg"] and len(episode["steps"]) == 5
    lines = (sink.path / "MD5SUMS").read_text().splitlines()
    assert len(lines) == 4 and all("  " in x for x in lines)


def test_a_start_payload_makes_the_runner_check_the_goal_for_drift():
    """The paired design's premise, enforced on episode 1 (U6 review R15, C2/C4).

    A 3-D scene's goal comes from the *measured* bounding box, so no offline test
    can catch an asset or layout drift; `check_goal_match` on the rig can. Handing
    `run_episode` the start file it drew from is what turns that into a check.
    """
    scene = make_scene()
    cfg = make_config(max_iter=8)
    goal = scene.goal_pose
    payload = {"scene": scene.name,
               "goal": {"cam_pos": goal.position.tolist(),
                        "cam_R_opt": goal.rotation.tolist(),
                        "distance_m": scene.working_distance}}
    rec = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg,
                             starts_payload=payload)
    assert rec["scene"] == scene.name

    drifted = {**payload, "goal": {**payload["goal"],
                                   "distance_m": scene.working_distance + 0.01}}
    with pytest.raises(ValueError, match="starts file goal mismatch"):
        runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg,
                           starts_payload=drifted)


def test_the_record_carries_gpu_load_and_frame_rate_columns(converged_episode):
    """Owner 2026-09-22: GPU utilisation (mean / max %, from nvidia-smi sampled during the
    episode; None where there is none) and the loop's and the matcher's frame rate."""
    rec, _ = converged_episode
    assert set(rec) >= {"gpu_util_mean_pct", "gpu_util_max_pct", "gpu_util_samples", "fps",
                        "match_fps"}
    assert rec["fps"] is None or rec["fps"] > 0
    assert rec["gpu_util_mean_pct"] is None or 0 <= rec["gpu_util_mean_pct"] <= 100


def test_the_absolute_ladder_rules_score_plain_thresholds():
    from vsbench.vs.bench import scoring as sc

    base = {"start_translation_mm": 300.0, "start_rotation_deg": 60.0}
    for t, r, want in ((9.9, 0.9, (True, True, True)), (15.0, 1.5, (False, True, True)),
                       (25.0, 2.5, (False, False, True)), (35.0, 3.5, (False, False, False))):
        rec = {**base, "final_translation_mm": t, "final_rotation_deg": r}
        rules = ("abs_10mm_1deg", "abs_20mm_2deg", "paper_3cm_3deg")
        got = tuple(sc.success(rec, rule, at="cap") for rule in rules)
        assert got == want, (t, r, got)



# ---------------------------------------- the instance suite's reads (2026-09-23)
class _RefPerturbation:
    """A world change that offers a second reference pose 5 mm beside the goal."""

    mode = "test_reference"

    def __init__(self, offset=(0.005, 0.0, 0.0)):
        self.offset = np.asarray(offset, float)
        self._ref = None

    def episode(self, scene_name, start_id):
        return 1

    def begin(self, scene, seed, *, goal, working_distance, intrinsics):
        self._ref = Pose(position=goal.position + self.offset, rotation=goal.rotation)
        return {"mode": self.mode}

    def live(self, rgb, step):
        return rgb

    def end(self, scene):
        return None

    def reference_pose(self):
        return self._ref


MATCHED_KEYS = {"reference_pose", "final_pose", "start_translation_mm_matched",
                "final_translation_mm_matched", "stop_translation_mm_matched", "success_matched"}


def test_a_reference_pose_adds_the_matched_reads_and_nothing_else_changes():
    from vsbench.vs.scene import translation_error_mm

    scene, cfg = make_scene(), make_config(max_iter=160)
    plain = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg)
    assert not MATCHED_KEYS & set(plain) and "outline_iou_final" not in plain
    rec = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg,
                             perturbation=_RefPerturbation())
    assert set(rec) >= MATCHED_KEYS
    assert set(rec) - MATCHED_KEYS == set(plain)             # the old fields, all still there
    ref = np.asarray(rec["reference_pose"]["position"])
    np.testing.assert_allclose(ref, scene.goal_pose.position + [0.005, 0.0, 0.0])
    fin = np.asarray(rec["final_pose"]["position"])
    assert rec["final_translation_mm_matched"] == pytest.approx(
        translation_error_mm(fin, ref))
    assert rec["final_translation_mm"] == pytest.approx(
        translation_error_mm(fin, scene.goal_pose.position))
    # the loop parks on the goal (the mock's truth matcher), so it is ~5 mm off the reference
    assert rec["final_translation_mm_matched"] == pytest.approx(5.0, abs=0.5)
    assert set(rec["success_matched"]) == set(rec["success"])
    for stop in rec["stops"].values():
        assert {"translation_mm_matched", "position", "rotation"} <= set(stop)


def test_the_outline_overlap_is_recorded_only_when_asked():
    scene = make_scene()
    cfg = make_config(max_iter=40, outline_iou=True)
    rec = runner.run_episode(scene, make_arm(scene, cfg), make_start(scene), cfg)
    assert "outline_iou_final" in rec and "outline_iou_stop" in rec
    # the mock is a plane AT the floor: no object anywhere, so both outlines are empty
    assert rec["outline_iou_final"] is None
    assert arms.episode_settings(make_config(max_iter=40))["outline_iou"] is False


def test_the_outline_overlap_helper():
    from vsbench.vs.labels import OUTLINE_TOL_M, outline_iou

    a = np.zeros((4, 6), bool)
    a[:, :3] = True
    b = np.zeros((4, 6), bool)
    b[:, 1:4] = True
    assert outline_iou(a, a) == 1.0 and outline_iou(a, ~a) == 0.0
    assert outline_iou(a, b) == pytest.approx(8 / 16)
    assert outline_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) is None
    with pytest.raises(ValueError, match="shapes"):
        outline_iou(a, np.zeros((3, 3), bool))
    assert OUTLINE_TOL_M == 0.001


def test_the_two_looser_rungs_are_plain_thresholds_too():
    from vsbench.vs.bench import scoring as sc

    base = {"start_translation_mm": 300.0, "start_rotation_deg": 60.0}
    for t, r, want in ((49.9, 4.9, (True, True)), (50.0, 4.0, (False, True)),
                       (99.9, 9.9, (False, True)), (60.0, 10.0, (False, False))):
        rec = {**base, "final_translation_mm": t, "final_rotation_deg": r}
        got = tuple(sc.success(rec, rule, at="cap")
                    for rule in ("abs_50mm_5deg", "abs_100mm_10deg"))
        assert got == want, (t, r, got)
    assert {"abs_50mm_5deg", "abs_100mm_10deg"} <= set(cfgmod.SUCCESS_RULES)
