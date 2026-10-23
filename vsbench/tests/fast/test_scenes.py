# vs-sim copy: tests/fast/test_scenes.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The six benchmark scenes, their frozen goals and their baked layouts.

Guards `vssim/scenes.py` and the Isaac-free half of `vssim/bake.py`: the six specs
load, the goal poses reproduce the frozen start files, baked layouts apply and an
id drift is refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vsbench.vs.camera import REPO_ROOT, CameraConfig
from vsbench.vs.scenes import (
    EVAL_SCENES,
    R_LOOK_DOWN,
    check_layout,
    distance_to_fill_frame,
    drop_quat,
    fronto_parallel_goal_pose,
    goal_capture_settings,
    load_layout,
    load_scenes,
    poster_goal_pose,
    sample_scatter,
    scene_view_goal_pose,
    working_distance_for_extent,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OLD_REPO = Path(__file__).resolve().parents[3] / "genvs-sim"


@pytest.fixture(scope="module")
def cam() -> CameraConfig:
    return CameraConfig()


@pytest.fixture(scope="module")
def scenes() -> dict:
    return load_scenes()[0]


@pytest.fixture(scope="module")
def frozen_goals() -> dict:
    """The `goal` block of each scene's frozen start file (fixture, read-only)."""
    return json.loads((FIXTURES / "frozen_goals.json").read_text())["goals"]


def test_the_six_scenes_load_with_their_kinds_and_ransac_models(scenes):
    for name in EVAL_SCENES:
        assert name in scenes, name
    assert [scenes[n].kind for n in EVAL_SCENES] == (
        ["poster"] * 3 + ["multi_object"] * 3)
    # planar -> homography, 3-D -> fundamental. The wrong model silently drops
    # good matches in the classical arms.
    assert [scenes[n].ransac for n in EVAL_SCENES] == (
        ["homography"] * 3 + ["fundamental"] * 3)


def test_start_spec_carries_the_frozen_v1_sampler_numbers():
    _scenes, starts = load_scenes()
    assert (starts.n, starts.seed) == (10, 20260727)
    assert (starts.translation_frac, starts.rotation_deg) == (0.18, 12.0)


# ------------------------------------------------------------- goal framing
def test_poster_goal_distance_is_derived_and_exact(cam, scenes, frozen_goals):
    """0.4342 m on all three posters, from camera.yaml and the 0.8 x 0.6 plate.

    Exact equality, not a tolerance: the goal pose is a closed-form function of
    the config, and every stored poster campaign was scored against this number.
    """
    for name in ("hollywood", "poster2", "poster3"):
        pos, R, d = poster_goal_pose(cam, scenes[name])
        assert d == frozen_goals[name]["distance_m"], name
        np.testing.assert_array_equal(pos, np.asarray(frozen_goals[name]["cam_pos"]))
        np.testing.assert_array_equal(R, np.asarray(frozen_goals[name]["cam_R_opt"]))
    assert round(frozen_goals["hollywood"]["distance_m"], 4) == 0.4342


#: What the station's measured bounding box must come back to. Each is the
#: binding (horizontal) extent implied by the frozen goal distance and the scene's
#: frame_fill - a forward prediction the offline test cannot make for itself,
#: because only Isaac measures a scene's box. Derived once, here, from the frozen
#: start files; `runner.assert_goal_matches_starts` is what checks it on the rig.
BINDING_EXTENT_M = {"industrial1_drill": 0.5131013025056193,
                    "cluttered2": 1.356152268533362,
                    "cluttered3": 1.491127451657344}


@pytest.mark.parametrize("name,rounded", [("industrial1_drill", 0.4423),
                                          ("cluttered2", 1.1690), ("cluttered3", 1.2853)])
def test_scene_view_goal_needs_a_measured_box_and_says_which(cam, scenes, frozen_goals,
                                                             name, rounded):
    """The three 3-D goals are a function of the **measured** Isaac bounding box.

    Nothing offline can derive them, so this does the honest thing instead: it
    states the horizontal extent each frozen goal implies (the binding axis at
    that scene's frame_fill), asserts the goal follows from a box of that extent,
    and hands the number to the station. Checking a box built *from* the answer
    would be a tautology; this is the prediction the station must reproduce.
    """
    g = frozen_goals[name]
    assert round(g["distance_m"], 4) == rounded
    fill = float(scenes[name].raw["goal"]["frame_fill"])
    ex = BINDING_EXTENT_M[name]
    assert ex / (fill * cam.h_coeff) == pytest.approx(g["distance_m"], rel=1e-12)
    centre_top = np.asarray(g["look_at"], float)
    lo = centre_top - np.array([ex / 2, ex / 4, 0.13])
    hi = centre_top + np.array([ex / 2, ex / 4, 0.0])
    pos, R, d = scene_view_goal_pose(cam, lo, hi, fill)
    assert d == pytest.approx(g["distance_m"], rel=1e-12)
    np.testing.assert_allclose(pos, g["cam_pos"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(R, np.asarray(g["cam_R_opt"]))


@pytest.mark.skipif(not (OLD_REPO / "src" / "eval" / "eval_scenes.py").exists(),
                    reason="the frozen ../genvs-sim checkout is not here")
def test_scene_view_goal_pose_is_the_frozen_function(cam):
    """The port itself, on boxes nobody chose: 200 random boxes at nadir and at a
    random tilt, against the frozen `eval_scenes.scene_view_goal_pose`."""
    sys.path.insert(0, str(OLD_REPO))
    try:
        from src.eval.eval_scenes import scene_view_goal_pose as old_goal
    finally:
        sys.path.remove(str(OLD_REPO))
    rng = np.random.default_rng(20260903)
    for _ in range(200):
        lo = rng.uniform(-1.0, 0.0, size=3)
        hi = lo + rng.uniform(0.05, 2.0, size=3)
        fill = float(rng.uniform(0.5, 0.95))
        elev = 90.0 if rng.random() < 0.5 else float(rng.uniform(20.0, 89.9))
        az = float(rng.uniform(-180.0, 180.0))
        a = scene_view_goal_pose(cam, lo, hi, fill, elevation_deg=elev, azimuth_deg=az)
        b = old_goal(cam, lo, hi, fill, elevation_deg=elev, azimuth_deg=az)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]


def test_a_derived_goal_passes_check_goal_match_and_a_drift_fails(cam, scenes,
                                                                  frozen_goals):
    """The real drift check is `starts.check_goal_match` (`benchmark.check_scene_goal`
    on a built scene). Here on the derivable half: the poster goal this repo derives
    is accepted, and a millimetre of asset drift is not."""
    from vsbench.vs.starts import check_goal_match
    payload = {"scene": "hollywood", "goal": frozen_goals["hollywood"]}
    pos, R, d = poster_goal_pose(cam, scenes["hollywood"])
    check_goal_match(payload, pos, R, d)                       # no exception
    with pytest.raises(ValueError, match="starts file goal mismatch"):
        check_goal_match(payload, pos + np.array([0.0, 0.0, 1e-3]), R, d)
    with pytest.raises(ValueError, match="starts file goal mismatch"):
        check_goal_match(payload, pos, R, d + 2e-3)


def test_scatter_items_places_a_scene_that_has_no_layout_yet(scenes):
    """`bake` is the only door to a new layout, and it needs the scatter: without
    it cluttered3's seven objects would all be dropped at the origin."""
    from vsbench.vs.isaac_scene import scatter_items
    spec = scenes["cluttered3"]
    sc = spec.raw["scatter"]
    offs = sample_scatter(7, seed=int(sc["seed"]), half_extent_x_m=sc["half_extent_x_m"],
                          half_extent_y_m=sc["half_extent_y_m"],
                          min_dist_m=float(sc["min_dist_m"]))
    items = scatter_items(spec)
    assert [(i["dx"], i["dy"]) for i in items] == offs
    assert [i["instance_id"] for i in items] == spec.object_ids()
    # with a baked layout the scatter is overwritten anyway, so it is not applied
    assert all("dx" not in i for i in scatter_items(spec, with_scatter=False))
    # a spec without a scatter block is untouched
    assert scatter_items(scenes["industrial1_drill"]) == [
        dict(o) for o in scenes["industrial1_drill"].raw["objects"]]


def test_every_frozen_goal_is_a_nadir_view_at_its_own_distance(frozen_goals):
    """Self-consistency of the fixture: R is exactly `R_LOOK_DOWN` and the camera
    sits `distance_m` straight above its look-at point."""
    for name, g in frozen_goals.items():
        np.testing.assert_array_equal(np.asarray(g["cam_R_opt"]), R_LOOK_DOWN)
        delta = np.asarray(g["cam_pos"]) - np.asarray(g["look_at"])
        np.testing.assert_allclose(delta[:2], 0.0, atol=1e-15)
        assert delta[2] == pytest.approx(g["distance_m"], rel=1e-15), name


def test_the_two_framing_rules_are_inverses_in_spirit(cam):
    """`fit` puts the plate inside the frame, `cover` pushes past every edge, so
    the covering distance is the smaller one."""
    fit = working_distance_for_extent(cam, 0.8, 0.6, 0.70)
    cover = distance_to_fill_frame(cam, 0.8, 0.6, 1.35)
    assert cover < fit
    assert distance_to_fill_frame(cam, 0.8, 0.6, 1.0) > cover     # less margin, farther
    with pytest.raises(ValueError, match="frame_fill must be > 0"):
        working_distance_for_extent(cam, 0.8, 0.6, 0.0)
    with pytest.raises(ValueError, match="coverage must be > 0"):
        distance_to_fill_frame(cam, 0.8, 0.6, 0.0)


def test_fronto_parallel_and_elevation_90_are_byte_identical(cam):
    pos, R = fronto_parallel_goal_pose(0.5)
    np.testing.assert_array_equal(R, R_LOOK_DOWN)
    lo, hi = np.array([-0.2, -0.15, 0.0]), np.array([0.25, 0.2, 0.12])
    a = scene_view_goal_pose(cam, lo, hi, 0.70)
    b = scene_view_goal_pose(cam, lo, hi, 0.70, elevation_deg=90.0)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], R_LOOK_DOWN)


def test_a_tilted_goal_keeps_image_right_on_world_x(cam):
    """The default azimuth is the one whose image orientation is continuous with
    the nadir view, so elevation variants stay visually comparable."""
    lo, hi = np.array([-0.2, -0.15, 0.0]), np.array([0.25, 0.2, 0.12])
    _pos, R, d = scene_view_goal_pose(cam, lo, hi, 0.70, elevation_deg=89.9)
    assert d == scene_view_goal_pose(cam, lo, hi, 0.70)[2]
    np.testing.assert_allclose(R, R_LOOK_DOWN, atol=2e-3)
    with pytest.raises(ValueError, match=r"outside \[20, 90\]"):
        scene_view_goal_pose(cam, lo, hi, 0.70, elevation_deg=10.0)


# ------------------------------------------------------------ scatter + settle
def test_scatter_is_seeded_aspect_matched_and_separated():
    kw = dict(half_extent_x_m=0.62, half_extent_y_m=0.35, min_dist_m=0.22)
    pts = sample_scatter(8, seed=20260821, **kw)
    assert pts == sample_scatter(8, seed=20260821, **kw)          # deterministic
    assert pts != sample_scatter(8, seed=20260827, **kw)          # seed matters
    arr = np.asarray(pts)
    assert len(pts) == 8
    assert np.all(np.abs(arr[:, 0]) <= 0.62) and np.all(np.abs(arr[:, 1]) <= 0.35)
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    assert d[~np.eye(8, dtype=bool)].min() >= 0.22
    with pytest.raises(ValueError, match="cannot place"):
        sample_scatter(50, seed=1, half_extent_m=0.1, min_dist_m=0.5, max_tries=200)


def test_drop_quat_is_a_unit_quaternion_with_an_exact_tilt_option():
    rng = np.random.default_rng(20260821)
    q = drop_quat(rng, tilt_deg=90.0)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    # a 90 deg tilt about a horizontal axis, composed with a yaw: the rotated
    # z axis must lie in the world xy plane (the drill lands lying down).
    w, x, y, z = q
    z_axis = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    assert z_axis[2] == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(drop_quat(np.random.default_rng(1))) == pytest.approx(1.0)


# ------------------------------------------------------------ baked layouts
@pytest.mark.parametrize("name,n", [("cluttered2", 8), ("cluttered3", 7),
                                    ("industrial1_drill", 1)])
def test_the_committed_layouts_match_their_scene_specs(scenes, name, n):
    layout = load_layout(scenes[name])
    assert layout is not None and layout["scene"] == name
    assert len(layout["objects"]) == n
    check_layout(layout, scenes[name].object_ids())            # no drift
    for obj in layout["objects"]:
        assert len(obj["pos"]) == 3 and len(obj["quat_wxyz"]) == 4
        assert np.linalg.norm(obj["quat_wxyz"]) == pytest.approx(1.0, abs=1e-6)


def test_a_layout_id_drift_is_refused(scenes):
    """PhysX settling is not bit-stable, so the frozen poses must land on exactly
    the prims they were baked for - a drift would move the goal pose."""
    layout = load_layout(scenes["cluttered2"])
    ids = scenes["cluttered2"].object_ids()
    with pytest.raises(ValueError, match="frozen layout mismatch"):
        check_layout(layout, ids[::-1])                        # reordered
    with pytest.raises(ValueError, match="frozen layout mismatch"):
        check_layout(layout, ids[:-1])                         # one dropped


def test_a_poster_scene_has_no_layout_and_a_missing_file_raises(scenes, tmp_path):
    assert load_layout(scenes["hollywood"]) is None
    spec = scenes["cluttered2"]
    with pytest.raises(FileNotFoundError, match="a re-settle may never define a goal"):
        load_layout(spec, repo_root=tmp_path)


def test_the_scenes_yaml_lives_in_this_repo():
    assert (REPO_ROOT / "configs" / "scenes.yaml").is_file()
    for name in ("cluttered2", "cluttered3", "industrial1_drill"):
        assert (REPO_ROOT / "configs" / "layouts" / f"{name}.json").is_file()


def test_every_scene_in_the_yaml_is_buildable_and_declares_its_model(scenes):
    """Every entry - benchmark or additional - names its objects by instance id and
    its geometric model; a scene that cannot be built does not belong in the file."""
    for name, spec in scenes.items():
        assert spec.ransac in ("homography", "fundamental", "none"), name
        if spec.kind == "multi_object":
            assert spec.object_ids(), f"{name} has an object without an instance_id"
            assert len(spec.object_ids()) == len(spec.raw["objects"]), name
        else:
            assert spec.kind == "poster" and spec.raw["goal"]["kind"] == "fill_frame", name


def test_the_goal_capture_settings_are_stated_in_the_yaml():
    warmup, keep = goal_capture_settings()
    assert warmup == 8 and keep == "last"
    with pytest.raises(KeyError, match="render_warmup_steps"):
        goal_capture_settings(REPO_ROOT / "configs" / "camera.yaml")


# ------------------------------------------------- object placement (2026-09-23)
def _rz(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _rx(deg):
    a = np.deg2rad(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])


def _ry(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])


def _mat(q):
    w, x, y, z = np.asarray(q, float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _old_place_objects_quat(item):
    """place_objects' orientation arithmetic before 2026-09-23, verbatim."""
    from vsbench.vs.scenes import _quat_mul, axis_angle_quat
    pre = item.get("pre_rot_x_deg")
    base_quat = (axis_angle_quat([1.0, 0.0, 0.0], np.deg2rad(float(pre)))
                 if pre else np.array([1.0, 0.0, 0.0, 0.0]))
    yaw = item.get("yaw_deg")
    if yaw:
        base_quat = _quat_mul(axis_angle_quat([0.0, 0.0, 1.0], np.deg2rad(float(yaw))), base_quat)
    return base_quat


def _old_place_extra_quat(item):
    """place_extra_objects' orientation arithmetic before 2026-09-23, verbatim."""
    from vsbench.vs.scenes import _quat_mul, axis_angle_quat
    yaw = np.deg2rad(float(item.get("yaw_deg", 0.0)))
    quat = axis_angle_quat([0.0, 0.0, 1.0], yaw)
    pre = item.get("pre_rot_x_deg")
    if pre:
        quat = _quat_mul(quat, axis_angle_quat([1.0, 0.0, 0.0], np.deg2rad(float(pre))))
    return quat


def test_item_quat_without_the_new_fields_is_the_original_arithmetic_bit_for_bit():
    """Every older scene keeps the exact quaternion it was placed with (the new path is
    only taken on request - uses_placement - and even then the non-canonical part is the
    same operations in the same order)."""
    from vsbench.vs.scenes import item_quat, uses_placement
    rng = np.random.default_rng(0)
    cases = [{}, {"pre_rot_x_deg": -90}, {"yaw_deg": 90}, {"pre_rot_x_deg": -90, "yaw_deg": 90},
             {"pre_rot_x_deg": 0, "yaw_deg": 0}, {"yaw_deg": None}]
    cases += [{"pre_rot_x_deg": float(a), "yaw_deg": float(b)}
              for a, b in rng.uniform(-180, 180, size=(20, 2))]
    for item in cases:
        assert not uses_placement(item)
        np.testing.assert_array_equal(item_quat(item), _old_place_objects_quat(item))
        # the extra-object path computed the same rotation with a different but
        # float-identical operation order (it never took a `yaw_deg: None`)
        if item.get("yaw_deg", 0.0) is not None:
            np.testing.assert_array_equal(_old_place_objects_quat(item),
                                          _old_place_extra_quat(item))


def test_canonical_quat_is_the_genvs_renderers_upright_pose():
    """Rz(yaw_pick + add) * R_orient(pick), and each pick really stands its native axis up."""
    from vsbench.vs.scenes import ORIENTATION_PICKS, canonical_quat
    native_up = {"posZ": [0, 0, 1], "negZ": [0, 0, -1], "posY": [0, 1, 0], "negY": [0, -1, 0],
                 "posX": [1, 0, 0], "negX": [-1, 0, 0]}
    for pick, (axis, deg) in ORIENTATION_PICKS.items():
        for yaw, add in ((0.0, 0.0), (270.0, 0.0), (270.0, 90.0), (90.0, 180.0)):
            R = _mat(canonical_quat({"orientation_pick": pick, "yaw_pick_deg": yaw}, add))
            Rorient = _rx(deg) if axis[0] else _ry(deg)
            np.testing.assert_allclose(R, _rz(yaw + add) @ Rorient, atol=1e-12)
            np.testing.assert_allclose(R @ np.asarray(native_up[pick], float),
                                       [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_array_equal(canonical_quat({}), [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="orientation_pick"):
        canonical_quat({"orientation_pick": "up"})


def test_item_quat_composes_canonical_then_lay_down_then_yaw():
    from vsbench.vs.scenes import item_quat
    entry = {"orientation_pick": "posZ", "yaw_pick_deg": 270.0}
    item = {"canonical": "library", "upright_yaw_add_deg": 90, "pre_rot_x_deg": 90, "yaw_deg": 0}
    np.testing.assert_allclose(_mat(item_quat(item, entry)),
                               _rx(90) @ _rz(270 + 90), atol=1e-12)
    item2 = {"canonical": "library", "pre_rot_x_deg": 90, "yaw_deg": "long_axis_x"}
    np.testing.assert_allclose(_mat(item_quat(item2, entry, yaw_deg=90.0)),
                               _rz(90) @ _rx(90) @ _rz(270), atol=1e-12)
    with pytest.raises(ValueError, match="resolved"):
        item_quat(item2, entry)                    # long_axis_x must be resolved first
    with pytest.raises(ValueError, match="asset-library entry"):
        item_quat({"canonical": "library"})


def test_long_axis_yaw_turns_a_footprint_longer_in_y_and_leaves_a_near_square_one():
    from vsbench.vs.scenes import LONG_AXIS_MARGIN, long_axis_yaw
    assert long_axis_yaw([0.10, 0.30, 0.05]) == 90.0
    assert long_axis_yaw([0.30, 0.10, 0.05]) == 0.0
    assert long_axis_yaw([0.20, 0.20 * LONG_AXIS_MARGIN, 0.1]) == 0.0   # inside the margin
    assert long_axis_yaw([0.20, 0.21, 0.1]) == 90.0


def test_the_placement_fields_are_checked():
    from vsbench.vs.scenes import check_placement_item, uses_placement
    assert uses_placement({"size_m": 0.2}) and uses_placement({"yaw_deg": "long_axis_x"})
    assert uses_placement({"canonical": "library"})
    assert not uses_placement({"yaw_deg": 90, "pre_rot_x_deg": -90, "scale": 1.0})
    for bad, msg in (({"canonical": "genvs"}, "canonical"),
                     ({"yaw_deg": "sideways"}, "long_axis_x"),
                     ({"size_m": 0.0}, "size_m"),
                     ({"upright_yaw_add_deg": 90}, "canonical: library")):
        with pytest.raises(ValueError, match=msg):
            check_placement_item({"instance_id": "x", **bad})


def test_scatter_items_of_every_older_scene_is_what_it_was(scenes):
    """The placement defaults merge only where a scene declares them: every scene that
    predates the instance suite gets its items back exactly as before."""
    from vsbench.vs.isaac_scene import scatter_items
    for name, spec in scenes.items():
        if name.startswith("inst_") or spec.kind != "multi_object":
            continue
        assert "placement" not in spec.raw, name
        items = scatter_items(spec, with_scatter=False)
        assert items == [dict(o) for o in spec.raw["objects"]], name


def test_the_instance_suite_scenes_carry_their_goal_prompt_and_live_objects(scenes):
    from vsbench.vs import benchmark
    from vsbench.vs.assets import load_library
    from vsbench.vs.bench.perturb import SAME_OBJECT_KEYS
    from vsbench.vs.isaac_scene import scatter_items
    lib = load_library()
    kinds = ("shoe", "bottle", "teapot", "drill", "mug", "hammer")
    want = tuple(f"inst_{k}{d}" for k in kinds for d in ("", "_rev"))
    assert want == benchmark.INSTANCE_SCENES
    for name in benchmark.INSTANCE_SCENES:
        spec = scenes[name]
        kind = name.split("_")[1]
        assert spec.prompt == kind and spec.ransac == "fundamental"
        assert spec.raw["goal"] == {"kind": "scene_view", "frame_fill": 0.85, "elevation_deg": 90}
        assert spec.raw["placement"]["canonical"] == "library"
        (goal,) = scatter_items(spec)
        assert goal["category"] == kind and goal["size_m"] > 0
        names = spec.live_names()
        if name.endswith("_rev"):
            assert names == ["same", "same_big", "other_big"]
        else:
            mine = [n for n in names if n.startswith("mine")]
            gen = [n for n in names if n.startswith("gen")]
            assert names == ["same", "same_small", "other_small", *mine, *gen]
            # the mug's third own pick is GEN-VS's mug TRAIN reference, not held out
            assert mine == (["mine1", "mine2", "mine3"] if kind == "mug" else ["mine1", "mine2"])
            assert gen == [f"gen{i}" for i in range(1, len(gen) + 1)] and len(gen) >= 5
        for live_name in names:
            live = spec.live_item(live_name)
            assert live["category"] == kind and live["instance_id"] in lib, (name, live_name)
            assert live["canonical"] == "library"
        same = spec.live_item("same")
        assert all(same.get(k) == goal.get(k) for k in SAME_OBJECT_KEYS), name
        other = "same_big" if name.endswith("_rev") else "same_small"
        assert spec.live_item(other)["instance_id"] == goal["instance_id"]
        cmp = (lambda a, b: a > b) if name.endswith("_rev") else (lambda a, b: a < b)
        for tag in (other, "other_big" if name.endswith("_rev") else "other_small"):
            assert cmp(spec.live_item(tag)["size_m"], goal["size_m"]), (name, tag)
        for tag in names:
            if tag.startswith(("mine", "gen")):     # the same-size level: the goal's length
                assert spec.live_item(tag)["size_m"] == goal["size_m"], (name, tag)
        with pytest.raises(KeyError, match="no live object"):
            spec.live_item("nope")
    ids = {scenes[n].live_item(t)["instance_id"] for n in benchmark.INSTANCE_SCENES
           for t in scenes[n].live_names()}
    assert "objv_bfaa6854d59f" not in ids          # broken model bounds (revision 2)
    assert scenes["inst_mug"].live_item("mine3")["instance_id"] == "025_mug"
    assert "025_mug" not in {scenes["inst_mug"].live_item(n)["instance_id"]
                             for n in scenes["inst_mug"].live_names() if n.startswith("gen")}
    # the forward and reverse scenes of a kind are the same pair, the other way round
    for kind in kinds:
        fwd, rev = scenes[f"inst_{kind}"], scenes[f"inst_{kind}_rev"]
        assert rev.object_ids() == [fwd.live_item("other_small")["instance_id"]]
        assert rev.live_item("other_big")["instance_id"] == fwd.object_ids()[0]
