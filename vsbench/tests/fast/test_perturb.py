# vs-sim copy: tests/fast/test_perturb.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Perturbations: seeded draws through a recording fake world, P4's noise model on a
frame, the runner integration on the numpy plane, and the objects-only goal mask."""

from __future__ import annotations

import numpy as np
import pytest

from vsbench.vs.bench import arms, perturb, runner
from vsbench.vs.bench import config as cfgmod
from vsbench.vs.bench.truth import PlaneTruthMatcher
from vsbench.vs.labels import object_mask_from_depth
from vsbench.vs.scene import D435I, MockPlaneScene, Pose


class _World:
    """Records the perturbation verbs an Isaac scene would take."""

    name = "fake"

    def __init__(self):
        self.calls = []
        self.lights = {"/World/DomeLight": {"intensity": 3000.0,
                                            "enable_color_temperature": None,
                                            "color_temperature": None}}

    def lighting_snapshot(self):
        self.calls.append("snapshot")
        return dict(self.lights)

    def set_lighting(self, r):
        self.calls.append(("set_lighting", r))

    def restore_lighting(self, snap):
        self.calls.append(("restore", snap))

    def put_box(self, path, centre, size_m, rotation, gray=0.5):
        self.calls.append(("put_box", path, list(centre), size_m))

    def remove_prim(self, path):
        self.calls.append(("remove_prim", path))

    def place_extra_objects(self, items, prefix="/World/Distractor"):
        self.calls.append(("place", [i["instance_id"] for i in items],
                           [i.get("pre_rot_x_deg") for i in items]))
        return [f"{prefix}_{k}" for k in range(len(items))]

    def remove_prims(self, paths):
        self.calls.append(("remove_prims", list(paths)))

    def build_poster(self, w, h, texture, prim_path="/World/Poster", uv_tiles=1.0):
        self.calls.append(("ground", prim_path, texture, round(float(uv_tiles), 3)))

    def object_pose(self, path):
        self.calls.append(("object_pose", path))
        return np.array([0.01, -0.02, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])

    def set_prim_visible(self, path, visible):
        self.calls.append(("visible", path, visible))


def _goal():
    # a camera 0.5 m above the floor looking straight down (optical z = world -z)
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return Pose(position=[0.1, 0.2, 0.5], rotation=R)


def _begin(p, world, seed=7):
    return p.begin(world, seed, goal=_goal(), working_distance=0.5, intrinsics=D435I)


def test_light_draws_inside_p2_and_restores_the_snapshot():
    w = _World()
    d = _begin(perturb.build("light"), w)
    assert 0.3 <= d["light_intensity_scale"] <= 2.0 and 2700 <= d["light_temperature_k"] <= 6500
    again = _begin(perturb.build("light"), _World())
    assert again["light_intensity_scale"] == d["light_intensity_scale"]
    perturb.build("light").end(w)                        # never begun: nothing to restore
    p = perturb.build("light")
    _begin(p, w)
    p.end(w)
    assert w.calls[-1][0] == "restore" and w.calls[0] == "snapshot"


def test_the_occluder_sits_on_the_ray_between_camera_and_scene_and_is_removed():
    w = _World()
    p = perturb.build("occluder")
    d = _begin(p, w)
    c = np.asarray(d["centre"])
    assert 0.35 * 0.5 <= d["distance_m"] <= 0.60 * 0.5
    assert 0 < c[2] < 0.5                                 # between camera and floor
    assert 0.10 <= d["image_width_frac"] <= 0.18
    assert d["size_m"] == pytest.approx(d["image_width_frac"] * 1920 * d["distance_m"] / D435I.fx)
    p.end(w)
    assert w.calls[-1] == ("remove_prim", "/World/Occluder")


def test_distractors_plan_two_to_five_objects_in_view_off_the_taken_spots():
    p = perturb.Distractors(seed=3, spec=None)
    pool = [f"ycb/{i:03d}_thing" for i in range(20)]
    items = p.plan(11, goal=_goal(), intrinsics=D435I, pool=pool)
    assert 2 <= len(items) <= 5 and len({i["instance_id"] for i in items}) == len(items)
    xy = np.array([[i["x"], i["y"]] for i in items])
    for a in range(len(xy)):
        for b in range(a + 1, len(xy)):
            assert np.hypot(*(xy[a] - xy[b])) >= 0.12
    # in view: the floor point re-projects inside the 10 % margin
    for it in items:
        u, v = it["pixel"]
        assert 192 <= u <= 1728 and 108 <= v <= 972
    assert p.plan(11, goal=_goal(), intrinsics=D435I, pool=pool) == items    # seeded
    w = _World()
    p.pool = pool
    _begin(p, w, seed=11)
    p.end(w)
    assert w.calls[0][0] == "place" and w.calls[-1][0] == "remove_prims"


def test_the_noise_model_is_seeded_per_frame_and_stays_in_range():
    params = perturb.draw_noise_params(5)
    assert 1e-4 <= params["lambda_shot"] <= 1.2e-2 and params["lambda_read"] > 0
    frame = np.full((32, 48, 3), 128, np.uint8)
    a = perturb.apply_noise(frame, params, 1)
    b = perturb.apply_noise(frame, params, 1)
    c = perturb.apply_noise(frame, params, 2)
    assert a.dtype == np.uint8 and a.shape == frame.shape
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c) and 0 < a.std() < 60


def test_a_world_without_the_verb_refuses_and_the_numpy_plane_takes_noise():
    scene = MockPlaneScene(render_scale=0.125)
    with pytest.raises(NotImplementedError, match="set_lighting"):
        perturb.build("light").begin(scene, 1, goal=scene.goal_pose, working_distance=0.4,
                                     intrinsics=scene.intrinsics)
    with pytest.raises(ValueError, match="mode"):
        perturb.build("fog")
    cfg = cfgmod.resolve({
        "seed": 20260902,
        "matcher": {"variant": "mutual_nn_seeded_subsample", "num_landmarks": 24, "device": "cpu"},
        "controller": {"gain": 2.0}, "stop": {"rule": "cns_plateau"},
        "episode": {"max_iter": 6}, "scoring": {"success_rule": ["predicate_a"]},
        "arm": {"kind": "oracle"}})
    goal = scene.goal_pose
    start = {"start_id": "t0", "cam_pos": goal.position + np.array([0.01, 0.0, 0.0]),
             "cam_R_opt": goal.rotation.tolist()}
    arm = arms.build_arm(cfg, intrinsics=scene.intrinsics,
                         matcher=PlaneTruthMatcher(scene, num_landmarks=24, seed=cfg["seed"]))
    seen = []
    rec = runner.run_episode(scene, arm, start, cfg, perturbation=perturb.build("noise", seed=1),
                             on_step=lambda i, rgb, d, p: seen.append(rgb.std()))
    assert rec["perturbation"]["mode"] == "noise" and "lambda_shot" in rec["perturbation"]
    assert rec["goal_mask_mode"] == "none" and rec["goal_mask_share"] is None
    clean = runner.run_episode(scene, arm, start, cfg, perturbation=perturb.build("none", seed=1))
    assert clean["perturbation"] == {"mode": "none", "seed": clean["perturbation"]["seed"]}
    assert rec["perturbation"]["seed"] == clean["perturbation"]["seed"]   # same draw, every arm
    assert runner.run_episode(scene, arm, start, cfg)["perturbation"]["mode"] == "none"
    assert len(seen) == 6


def test_the_objects_only_goal_mask_comes_from_depth_against_the_floor():
    goal = _goal()
    h, w = 108, 192
    small = type("I", (), {"fx": D435I.fx / 10, "fy": D435I.fy / 10, "cx": w / 2, "cy": h / 2,
                           "width": w, "height": h})()
    # the floor is 0.5 m below a camera looking straight down: depth 0.5 everywhere ...
    depth = np.full((h, w), 0.5)
    depth[40:60, 80:120] = 0.42                # ... except a box 8 cm tall
    m = object_mask_from_depth(depth, small, goal)
    assert m.shape == (h, w) and m[50, 100] and not m[10, 10]
    assert m.sum() == 20 * 40


def test_the_mask_cell_is_labelled_and_refused_where_the_matcher_ignores_it():
    cfg = cfgmod.resolve({
        "seed": 20260902,
        "matcher": {"variant": "mutual_nn_seeded_subsample", "num_landmarks": 24,
                    "device": "cpu", "goal_mask": "objects"},
        "controller": {"gain": 2.0, "weighting": {"use_trust": False, "use_margin": False,
                                                  "per_axis": False}},
        "stop": {"rule": "cns_plateau"},
        "episode": {"max_iter": 6}, "scoring": {"success_rule": ["predicate_a"]}})
    assert arms.arm_label(cfg) == ("mutual_nn_seeded_subsample@dinov3_vits16/k24/objmask/"
                                   "ur/uniform/cns_plateau")
    with pytest.raises(cfgmod.ConfigError, match="goal_mask"):
        cfgmod.validate({**cfg, "matcher": {**cfg["matcher"], "goal_mask": "floor"}})
    scene = MockPlaneScene(render_scale=0.125)
    truth = PlaneTruthMatcher(scene, num_landmarks=24, seed=1)      # ignores a goal mask
    with pytest.raises(ValueError, match="ignores reset"):
        arms.build_arm({**cfg, "arm": {"kind": "oracle"}}, intrinsics=scene.intrinsics,
                       matcher=truth)


def test_swap_instance_hides_the_goal_object_places_the_other_and_puts_it_back():
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    spec = specs["shoe1"]
    assert spec.object_ids() == ["gso_Great_Jones_Wingtip"] and spec.ransac == "fundamental"
    w = _World()
    p = perturb.build("swap:Reebok_SOMERSET_RUN", seed=3, spec=spec)
    d = _begin(p, w)
    assert d["mode"] == "swap_instance" and d["swapped"] is True
    assert d["goal_instance"] == "gso_Great_Jones_Wingtip"
    assert d["live_instance"] == "Reebok_SOMERSET_RUN"
    assert d["xy"] == [0.01, -0.02]
    assert w.calls[0] == ("object_pose", "/World/Obj_0")
    assert w.calls[1] == ("visible", "/World/Obj_0", False)
    assert w.calls[2] == ("place", ["Reebok_SOMERSET_RUN"], [-90.0])   # laid flat, like the goal
    p.end(w)
    assert w.calls[-2][0] == "remove_prims" and w.calls[-1] == ("visible", "/World/Obj_0", True)
    # the same instance: nothing changes, the record says so
    same = perturb.build("swap:gso_Great_Jones_Wingtip", spec=spec)
    assert _begin(same, _World())["swapped"] is False
    with pytest.raises(ValueError, match="instance id"):
        perturb.build("swap:", spec=spec)
    with pytest.raises(NotImplementedError, match="object_pose"):
        p2 = perturb.build("swap:x", spec=spec)
        p2.begin(MockPlaneScene(render_scale=0.125), 1, goal=_goal(), working_distance=0.4,
                 intrinsics=D435I)


def test_the_ground_swap_re_textures_the_live_floor_and_puts_the_scene_texture_back():
    """P3 for the floor (owner 2026-09-22: "assume background is different"): the goal
    picture keeps the scene's own ground, the flight sees another one."""
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    spec = specs["shoe2"]
    w = _World()
    p = perturb.build("ground:concrete", seed=1, spec=spec)
    d = _begin(p, w)
    assert d["mode"] == "ground_swap" and d["ground"] == "concrete"
    assert d["texture"] == perturb.GROUNDS["concrete"] and "cluttered2" in d["goal_texture"]
    assert w.calls[0] == ("ground", "/World/Ground", perturb.GROUNDS["concrete"], 1.0)
    p.end(w)
    assert w.calls[-1] == ("ground", "/World/Ground", d["goal_texture"], 1.0)
    with pytest.raises(ValueError, match="ground needs a texture"):
        perturb.build("ground:", spec=spec)


def test_a_plus_chains_perturbations_and_undoes_them_in_reverse():
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    spec = specs["shoe2"]
    w = _World()
    p = perturb.build("swap:Reebok_SOMERSET_RUN+ground:concrete", seed=2, spec=spec)
    d = _begin(p, w)
    assert d["mode"] == "chain" and d["parts"] == ["swap_instance", "ground_swap"]
    assert d["swap_instance"]["live_instance"] == "Reebok_SOMERSET_RUN"
    assert d["ground_swap"]["ground"] == "concrete"
    kinds = [c[0] for c in w.calls]
    assert kinds.index("place") < kinds.index("ground")          # in order
    n_before = len(w.calls)
    p.end(w)
    undone = [c[0] for c in w.calls[n_before:]]
    assert undone.index("ground") < undone.index("visible")      # undone in reverse
    assert ("visible", "/World/Obj_0", True) in w.calls          # the goal object is back



# ------------------------------------------- the named swap (instance suite, 2026-09-23)
class _BoxWorld(_World):
    """The fake world with bounding boxes: the goal object is a 0.30 x 0.10 x 0.12 m box
    centred at (0.05, -0.02) whose model origin sits elsewhere; a placed live object gets
    the box of its size_m, centred where it was put."""

    GOAL_BOX = (np.array([-0.10, -0.07, 0.0]), np.array([0.20, 0.03, 0.12]))

    def __init__(self):
        super().__init__()
        self.placed = {}
        self.extra_placements = {}

    def object_world_bbox(self, path):
        self.calls.append(("bbox", path))
        if path == "/World/Obj_0":
            return self.GOAL_BOX
        x, y, size = self.placed[path]
        half = np.array([size / 2.0, size / 6.0, 0.0])
        return (np.array([x, y, 0.0]) - half, np.array([x, y, 0.0]) + half + [0, 0, size / 3])

    def place_extra_objects(self, items, prefix="/World/Distractor"):
        paths = super().place_extra_objects(items, prefix)
        for path, it in zip(paths, items, strict=True):
            self.placed[path] = (it["x"], it["y"], it["size_m"])
            self.calls.append(("placed_at", path, it["x"], it["y"], it["size_m"],
                               it.get("canonical"), it.get("yaw_deg")))
        return paths


def test_the_named_swap_centres_the_live_object_on_the_goal_and_offers_the_size_matched_pose():
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    spec = specs["inst_shoe"]
    w = _BoxWorld()
    p = perturb.build("swap:@same_small", seed=3, spec=spec)
    d = _begin(p, w)
    small = spec.live_item("same_small")
    assert d["swapped"] is True and d["live_name"] == "same_small"
    assert d["live_instance"] == d["goal_instance"] == "Reebok_SOMERSET_RUN"
    assert d["live_size_m"] == small["size_m"] < d["goal_size_m"]
    # the live object's box centre goes on the goal object's VISIBLE centre (0.05, -0.02),
    # not on its model origin (object_pose is never asked)
    assert ("visible", "/World/Obj_0", False) in w.calls
    placed = next(c for c in w.calls if c[0] == "placed_at")
    assert placed[2:5] == (pytest.approx(0.05), pytest.approx(-0.02), small["size_m"])
    assert placed[5] == "library" and placed[6] == "long_axis_x"   # the scene's placement
    assert not any(c[0] == "object_pose" for c in w.calls)
    # the size ratio from the two boxes as placed, and the reference pose from it
    s = small["size_m"] / 0.30
    assert d["size_ratio"] == pytest.approx(s)
    ref = p.reference_pose()
    P = np.array([0.05, -0.02, 0.0])
    np.testing.assert_allclose(ref.position, P + s * (_goal().position - P))
    np.testing.assert_array_equal(ref.rotation, _goal().rotation)
    assert d["reference_position"] == pytest.approx(list(ref.position))
    p.end(w)
    assert w.calls[-2][0] == "remove_prims" and w.calls[-1] == ("visible", "/World/Obj_0", True)


def test_the_named_same_is_a_recorded_no_op_whose_reference_is_the_goal():
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    w = _BoxWorld()
    p = perturb.build("swap:@same", seed=3, spec=specs["inst_drill"])
    d = _begin(p, w)
    assert d["swapped"] is False and d["size_ratio"] == pytest.approx(1.0)
    assert not any(c[0] in ("visible", "place") for c in w.calls)
    np.testing.assert_allclose(p.reference_pose().position, _goal().position)
    p.end(w)                                        # nothing to undo
    assert not any(c[0] in ("remove_prims", "visible") for c in w.calls)


def test_a_chain_forwards_the_reference_pose_and_the_names_are_checked():
    from vsbench.vs.scenes import load_scenes

    specs, _ = load_scenes()
    spec = specs["inst_bottle_rev"]
    p = perturb.build("swap:@other_big+ground:concrete", seed=1, spec=spec)
    d = _begin(p, _BoxWorld())
    assert d["parts"] == ["swap_instance", "ground_swap"]
    assert d["swap_instance"]["size_ratio"] > 1.0          # the reverse: a bigger object
    assert p.reference_pose() is not None
    assert perturb.build("ground:concrete", spec=spec).reference_pose() is None
    assert perturb.build("swap:Reebok_SOMERSET_RUN", spec=specs["shoe1"]).reference_pose() is None
    with pytest.raises(ValueError, match="live object's name"):
        perturb.build("swap:@", spec=spec)
    with pytest.raises(KeyError, match="no live object 'nope'"):
        _begin(perturb.build("swap:@nope", spec=spec), _BoxWorld())
    with pytest.raises(NotImplementedError, match="object_world_bbox"):
        _begin(perturb.build("swap:@same_big", spec=spec), _World())


def test_the_size_matched_pose_scales_the_goal_about_the_floor_centre():
    goal = _goal()
    same = perturb.size_matched_pose(goal, [0.1, 0.2], 1.0)
    np.testing.assert_allclose(same.position, goal.position)
    half = perturb.size_matched_pose(goal, [0.1, 0.2], 0.5)
    np.testing.assert_allclose(half.position, [0.1, 0.2, 0.25])      # halfway down to P
    far = perturb.size_matched_pose(goal, [0.0, 0.0], 1.5)
    np.testing.assert_allclose(far.position, [0.15, 0.30, 0.75])
    np.testing.assert_array_equal(far.rotation, goal.rotation)
