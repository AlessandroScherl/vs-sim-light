"""U8 gates (docs/PORT_PLAN.md section 7, U8): the copied perturbation contract on ``MujocoScene``.

The copied ``vsbench/tests/fast/test_perturb.py`` proves ``vsbench.vs.bench.perturb`` against
a recording fake world (``_World``). This module runs the same contract against the real
MuJoCo world - the copied tests are not edited - and adds what only a real world can show:

1. every verb the copied modes ask for (``perturb._needs``) is there, and each copied mode
   ``begin`` / ``end`` leaves the cached goal picture's md5 unchanged **and** the world
   unedited: the goal pose drawn fresh after ``end`` has the goal's md5 (picture and depth);
2. the copied contract, mode by mode (``light``, ``occluder``, ``distractors``,
   ``ground:<name>``, ``swap:<instance_id>``, ``swap:@<name>``, ``combined``, ``a+b``), with
   the same assertions as the fake-world tests where they carry over, the call order of a
   chain through a recording proxy around the real scene, and the geometry each edit must
   produce (the occluder's face at ``distance - size / 2`` on its ray, the distractors' and
   the swapped object's box centred on the drawn floor point with the bottom on ``z = 0``,
   the named swap's box centred on the goal object's and its size ratio);
3. depth under perturbation: on a stride-4 grid at the goal pose, a pixel whose ray meets
   the same unedited surface in both worlds has **exactly** the clean world's depth, and
   every hit pixel of the edited world matches its own ray caster (median <= 1 mm, p99 <=
   3 mm, U4's gate; the ground against the analytic ``z = 0`` plane too);
4. the light: the drawn picture is the look's tone table, relit by ``s * blackbody(T)``,
   applied to the renderer's own output (depth untouched); factor 1 is the look's table;
5. the copied loop flies a chained perturbation on the world (3 steps) and writes the draw;
6. the refusals (an unknown prim, unequal light factors, a non-square tiled poster, a
   Nucleus asset; on a poster scene the copied ``ground:`` / ``swap:`` refusals).

Scenes: ``industrial1_drill`` (one object, the cheapest room), ``inst_drill`` (the named
swap; ``raw`` look - no fitted curve for the instance suite), ``hollywood`` (a poster); one
scene alive at a time. Numbers land in ``results/bridge/u8_perturb_gates.json`` only with
``VSBENCH_WRITE_GATES=1``. CPU only: ``source tools/cpu_env.sh``; skipped without software GL.

    source tools/cpu_env.sh
    VSBENCH_WRITE_GATES=1 $VSBENCH_PY -m pytest tests/world/test_perturb_world.py -q
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vsbench.glenv import software_gl_requested  # noqa: E402

pytestmark = pytest.mark.skipif(
    not software_gl_requested(),
    reason="renders: needs VSBENCH_SOFTWARE_GL=1 (source tools/cpu_env.sh) so no GPU is touched")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "bridge" / "u8_perturb_gates.json"
SEED = 20260918
VERBS = ("build_poster", "put_box", "remove_prim", "remove_prims", "place_extra_objects",
         "set_prim_visible", "object_pose", "object_world_bbox", "set_lighting",
         "lighting_snapshot", "restore_lighting")
RESULTS: dict = {}

if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))


def write_gates() -> bool:
    return os.environ.get("VSBENCH_WRITE_GATES", "").strip() == "1"


@pytest.fixture(scope="module", autouse=True)
def _write_results(tmp_path_factory):
    yield
    if not RESULTS:
        return
    out = OUT if write_gates() else tmp_path_factory.mktemp("gates") / OUT.name
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(RESULTS)
    old["unit"] = "U8 - the perturbation world verbs in MuJoCo (docs/PORT_PLAN.md section 7)"
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True, default=_jd) + "\n")
    print(f"[gates] wrote {out}")


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def _md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _open(name, look=None):
    from vsbench.world.mujoco_scene import MujocoScene

    scene = MujocoScene(name, look=look)
    gl = str(getattr(scene.renderer, "gl_renderer", "")).lower()
    if scene.render_mode != "mujoco-llvmpipe" or "llvmpipe" not in gl:
        scene.close()
        pytest.fail(f"MuJoCo opened {scene.render_mode!r} ({gl!r}), not llvmpipe - refusing")
    g, d = scene.render_goal()
    scene.goal_md5s = (_md5(g), _md5(d))
    RESULTS.setdefault("scenes", {})[name] = {
        "render_mode": scene.render_mode, "gl_renderer": scene.renderer.gl_renderer,
        "look": scene.look.name, "look_curve_id": scene.look.curve_id,
        "goal_frame_md5": scene.goal_md5s[0], "goal_depth_md5": scene.goal_md5s[1]}
    return scene


@pytest.fixture(scope="class")
def drill():
    s = _open("industrial1_drill")
    yield s
    s.close()
    del s
    gc.collect()


@pytest.fixture(scope="class")
def inst():
    s = _open("inst_drill", look="raw")
    yield s
    s.close()
    del s
    gc.collect()


@pytest.fixture(scope="class")
def poster():
    s = _open("hollywood")
    yield s
    s.close()
    del s
    gc.collect()


def _begin(p, scene, start_id="n_000"):
    seed = p.episode(scene.name, start_id)
    return p.begin(scene, seed, goal=scene.goal_pose, working_distance=scene.working_distance,
                   intrinsics=scene.intrinsics)


def _restored(scene) -> dict:
    """After ``end``: the world is the unedited one and draws the goal's bytes again."""
    scene.sync()
    rgb, depth = scene.draw_fresh(scene.goal_pose)
    cached, _ = scene.render_goal()
    out = {"unedited": scene.is_unedited(), "fresh_goal_md5": _md5(rgb) == scene.goal_md5s[0],
           "fresh_goal_depth_md5": _md5(depth) == scene.goal_md5s[1],
           "cached_goal_md5": _md5(cached) == scene.goal_md5s[0]}
    assert all(out.values()), out
    return out


def _depth(scene, clean) -> dict:
    """The depth gate of the current (edited) world against ``clean`` at the goal pose."""
    from u8_perturb import edited_geometry

    from vsbench.reference import plane_zdepth
    from vsbench.vs.scene import pose_to_T
    from vsbench.world.perturb_world import depth_gate, ray_depth

    size = float((scene.spec.raw.get("ground") or {}).get("size_m", 2.4))
    plane = plane_zdepth(scene.render_camera, pose_to_T(scene.goal_pose), np.eye(4), size=(size, size))
    g = depth_gate(clean, ray_depth(scene, scene.goal_pose), scene, edited_geometry(scene),
                   ground_plane=plane if scene.spec.kind != "poster" else None)
    assert g["n_unchanged"] > 0 and g["unchanged_max_abs_m"] == 0.0, g
    for k in ("all_vs_ray", "edited_vs_ray", "ground_vs_plane"):
        if g.get(k):
            assert g[k]["median_mm"] <= 1.0 and g[k]["p99_mm"] <= 3.0, (k, g[k])
    assert g["miss_with_depth_0"] and g["frame_finite"] and g["frame_nonneg"], g
    return g


class Recorder:
    """The real scene, with every verb call written down (the copied fake world's
    ``calls`` list, for the order assertions)."""

    def __init__(self, scene):
        self._scene = scene
        self.calls: list[str] = []

    def __getattr__(self, name):
        attr = getattr(self._scene, name)
        if name in VERBS and callable(attr):
            def rec(*a, **kw):
                self.calls.append(name)
                return attr(*a, **kw)
            return rec
        return attr


# ------------------------------------------------------------------ the drill room
class TestRoom:
    def test_every_verb_is_there(self, drill):
        from vsbench.vs.bench import perturb

        perturb._needs(drill, *VERBS)
        assert isinstance(drill.extra_placements, dict)

    def test_light_draws_inside_p2_relights_through_the_tone_table_and_restores(self, drill):
        from vsbench.vs.bench import perturb
        from vsbench.vs.scene import Pose, pose_to_T
        from vsbench.world.look import apply_lut
        from vsbench.world.perturb_world import blackbody_rgb, boot_lighting, relit_lut

        assert drill.lighting_snapshot() == boot_lighting()
        perturb.build("light").end(drill)                      # never begun: nothing to restore
        p = perturb.build("light", seed=SEED)
        d = _begin(p, drill)
        assert 0.3 <= d["light_intensity_scale"] <= 2.0 and 2700 <= d["light_temperature_k"] <= 6500
        again = perturb.build("light", seed=SEED).begin(
            _Dummy(), d["seed"], goal=drill.goal_pose, working_distance=drill.working_distance,
            intrinsics=drill.intrinsics)
        assert again["light_intensity_scale"] == d["light_intensity_scale"]      # seeded
        f = d["light_intensity_scale"] * blackbody_rgb(d["light_temperature_k"])
        np.testing.assert_allclose(drill._light_factor, f, rtol=0, atol=1e-12)
        assert drill.lighting_snapshot()["/World/SunLight"]["intensity"] == pytest.approx(
            3500.0 * d["light_intensity_scale"])
        assert not drill.dirty                                 # no recompile for the light
        pose = Pose(position=drill.goal_pose.position + np.array([0.004, -0.003, 0.002]),
                    rotation=drill.goal_pose.rotation)
        rgb, depth = drill.draw_fresh(pose)
        raw, raw_depth = drill.renderer.render(pose_to_T(pose))
        np.testing.assert_array_equal(rgb, apply_lut(raw, relit_lut(drill.look, f)))
        np.testing.assert_array_equal(depth, raw_depth)
        p.end(drill)
        assert drill._lut is None and np.all(drill._light_factor == 1.0)
        snap = drill.lighting_snapshot()["/World/DomeLight"]
        assert snap["intensity"] == 3000.0 and snap["enable_color_temperature"] is False
        clean, _ = drill.draw_fresh(pose)
        np.testing.assert_array_equal(clean, drill.look.apply(raw))
        lut1 = relit_lut(drill.look, np.ones(3) * (1 + 1e-12))
        assert np.abs(lut1.astype(int) - drill.look.lut.astype(int)).max() <= 1
        RESULTS.setdefault("room", {})["light"] = {"draw": d, "factor_rgb": f.tolist(),
                                                   "restored": _restored(drill)}

    def test_the_occluder_sits_on_its_ray_at_its_distance_and_is_removed(self, drill):
        from vsbench.vs.bench import perturb
        from vsbench.world.perturb_world import ray_depth

        clean = ray_depth(drill, drill.goal_pose)
        p = perturb.build("occluder", seed=SEED)
        d = _begin(p, drill)
        wd = drill.working_distance
        assert 0.35 * wd <= d["distance_m"] <= 0.60 * wd and 0.10 <= d["image_width_frac"] <= 0.18
        assert d["size_m"] == pytest.approx(d["image_width_frac"] * 1920 * d["distance_m"]
                                            / drill.intrinsics.fx)
        assert "/World/Occluder" in drill._prims and drill.dirty
        _rgb, depth = drill.draw_fresh(drill.goal_pose)
        u, v = (int(round(c)) for c in d["pixel"])
        face = d["distance_m"] - d["size_m"] / 2.0
        assert abs(float(depth[v, u]) - face) < 1e-3, (depth[v, u], face)
        row = depth[v] < face + 1e-3
        width = int(row.sum())
        want = drill.intrinsics.fx * d["size_m"] / face
        assert abs(width - want) <= 3, (width, want)
        g = _depth(drill, clean)
        p.end(drill)
        assert "/World/Occluder" not in drill._prims
        RESULTS.setdefault("room", {})["occluder"] = {
            "draw": d, "face_depth_err_mm": abs(float(depth[v, u]) - face) * 1e3,
            "face_width_px": width, "face_width_px_predicted": want, "depth": g,
            "compile": drill.world_log[-1], "restored": _restored(drill)}

    def test_distractors_are_ycb_objects_seated_on_their_floor_points_and_removed(self, drill):
        from vsbench.vs.bench import perturb
        from vsbench.world.perturb_world import ray_depth

        clean = ray_depth(drill, drill.goal_pose)
        p = perturb.build("distractors", seed=SEED, spec=drill.spec)
        d = _begin(p, drill)
        assert 2 <= d["n"] <= 5 and d["prim_paths"] == [f"/World/Distractor_{k}" for k in range(d["n"])]
        own = set(drill.spec.object_ids())
        for it, path in zip(d["items"], d["prim_paths"], strict=True):
            assert it["instance_id"].startswith("ycb/") and it["instance_id"] not in own
            lo, hi = drill.object_world_bbox(path)
            assert abs((lo[0] + hi[0]) / 2 - it["x"]) < 1e-9 and abs((lo[1] + hi[1]) / 2 - it["y"]) < 1e-9
            assert abs(lo[2]) < 1e-9
        g = _depth(drill, clean)
        p.end(drill)
        assert not any(k.startswith("/World/Distractor") for k in drill._prims)
        RESULTS.setdefault("room", {})["distractors"] = {
            "draw": d, "depth": g, "compile": drill.world_log[-1], "restored": _restored(drill)}

    def test_the_ground_swap_re_textures_the_floor_only(self, drill):
        from vsbench.vs.bench import perturb
        from vsbench.world.build import workspace_path

        p = perturb.build("ground:wood", seed=SEED, spec=drill.spec)
        d = _begin(p, drill)
        assert d["ground"] == "wood" and d["texture"] == perturb.GROUNDS["wood"]
        assert "industrial1" in d["goal_texture"]
        g = drill._prims["/World/Ground"]
        assert g.info["texture"] == perturb.GROUNDS["wood"]
        assert workspace_path(g.info["texture"]).exists()
        rgb, depth = drill.draw_fresh(drill.goal_pose)
        assert _md5(depth) == drill.goal_md5s[1]               # the same geometry, bit for bit
        assert _md5(rgb) != drill.goal_md5s[0]
        assert p.truth_model()["floor"] == "retextured"
        p.end(drill)
        RESULTS.setdefault("room", {})["ground"] = {"draw": d, "depth_identical": True,
                                                    "compile": drill.world_log[-1],
                                                    "restored": _restored(drill)}

    def test_swap_instance_hides_the_goal_object_places_the_other_and_puts_it_back(self, drill):
        from vsbench.vs.bench import perturb
        from vsbench.vs.scenes import load_layout
        from vsbench.world.perturb_world import ray_depth

        clean = ray_depth(drill, drill.goal_pose)
        layout = load_layout(drill.spec)
        w = Recorder(drill)
        p = perturb.build("swap:ycb/006_mustard_bottle", seed=SEED, spec=drill.spec)
        d = _begin(p, w)
        pos = np.asarray(layout["objects"][0]["pos"], float)
        assert d["swapped"] is True and d["goal_instance"] == "handal_drill_25"
        assert d["xy"] == [pos[0], pos[1]]
        assert w.calls[:3] == ["object_pose", "set_prim_visible", "place_extra_objects"]
        assert drill._prims["/World/Obj_0"].visible is False and d["prim_paths"] == ["/World/Swap_0"]
        lo, hi = drill.object_world_bbox("/World/Swap_0")
        assert abs((lo[0] + hi[0]) / 2 - pos[0]) < 1e-9 and abs(lo[2]) < 1e-9
        g = _depth(drill, clean)
        p.end(w)
        assert w.calls[-2:] == ["remove_prims", "set_prim_visible"]
        assert drill._prims["/World/Obj_0"].visible is True
        restored = _restored(drill)
        compile_ev = drill.world_log[-2]
        same = perturb.build("swap:handal_drill_25", spec=drill.spec)
        assert _begin(same, drill)["swapped"] is False and not drill.dirty     # nothing edited
        RESULTS.setdefault("room", {})["swap_instance"] = {
            "draw": d, "depth": g, "compile": compile_ev, "restored": restored}

    def test_a_plus_chain_runs_in_order_and_is_undone_in_reverse(self, drill):
        from vsbench.vs.bench import perturb

        w = Recorder(drill)
        p = perturb.build("swap:ycb/006_mustard_bottle+ground:wood", seed=SEED, spec=drill.spec)
        d = _begin(p, w)
        assert d["mode"] == "chain" and d["parts"] == ["swap_instance", "ground_swap"]
        assert w.calls.index("place_extra_objects") < w.calls.index("build_poster")
        n = len(w.calls)
        n_compiles = len(drill.world_log)
        drill.draw_fresh(drill.goal_pose)
        assert len(drill.world_log) == n_compiles + 1          # the chain's edits: one compile
        p.end(w)
        undone = w.calls[n:]
        assert undone.index("build_poster") < undone.index("set_prim_visible")
        RESULTS.setdefault("room", {})["chain"] = {"draw": d, "calls": w.calls,
                                                   "restored": _restored(drill)}

    def test_combined_begins_every_part_and_ends_clean(self, drill):
        from vsbench.vs.bench import perturb

        p = perturb.build("combined", seed=SEED, spec=drill.spec)
        d = _begin(p, drill)
        assert {"light", "distractors", "occluder", "noise"} <= set(d)
        assert drill._lut is not None and "/World/Occluder" in drill._prims
        rgb, _ = drill.draw_fresh(drill.goal_pose)
        assert p.live(rgb, 1).shape == rgb.shape
        p.end(drill)
        assert drill._lut is None
        RESULTS.setdefault("room", {})["combined"] = {"draw": d, "restored": _restored(drill)}

    def test_the_copied_loop_flies_a_chained_perturbation_on_the_world(self, drill):
        from vsbench.vs.bench import perturb
        from vsbench.vs.bench.runner import run_episode
        from vsbench.world import fly

        protocol = fly.load_protocol()
        cfg = fly.arm_config(protocol, "sift_ransac", max_iter=3, device="cpu")
        arm = fly.build(protocol, "sift_ransac", cfg, intrinsics=drill.intrinsics,
                        scene_name=drill.name, scene_spec=drill.spec)
        g = drill.goal_pose
        start = {"start_id": "u8_t0", "cam_pos": (g.position + np.array([0.004, 0.0, 0.0])).tolist(),
                 "cam_R_opt": g.rotation.tolist()}
        p = perturb.build("occluder+ground:wood", seed=SEED, spec=drill.spec)
        rec = run_episode(drill, arm, start, cfg, perturbation=p)
        assert rec["perturbation"]["mode"] == "chain" and rec["steps"] == 3
        assert rec["perturbation"]["parts"] == ["occluder", "ground_swap"]
        assert rec["goal_frame_md5"] == drill.goal_md5s[0] and rec["render_mode"] == "mujoco-llvmpipe"
        RESULTS.setdefault("room", {})["runner"] = {
            "perturbation": rec["perturbation"], "steps": rec["steps"],
            "final_translation_mm": rec["final_translation_mm"], "restored": _restored(drill)}

    def test_the_refusals(self, drill):
        from vsbench.vs.bench import perturb

        with pytest.raises(ValueError, match="not a valid prim"):
            drill.set_prim_visible("/World/Nope", False)
        with pytest.raises(ValueError, match="no object there"):
            drill.object_world_bbox("/World/Ground")
        snap = drill.lighting_snapshot()
        snap["/World/SunLight"]["intensity"] = 7000.0
        with pytest.raises(NotImplementedError, match="different factors"):
            drill.restore_lighting(snap)
        assert drill._lut is None
        with pytest.raises(NotImplementedError, match="square"):
            drill.build_poster(0.8, 0.6, perturb.GROUNDS["wood"], prim_path="/World/Extra", uv_tiles=2.0)
        with pytest.raises(FileNotFoundError, match="Nucleus"):      # D-6: not on disk
            drill.place_extra_objects([{"instance_id": "035_power_drill", "category": "drill",
                                        "x": 0.0, "y": 0.0}])
        with pytest.raises(FileNotFoundError, match="object cache"):
            drill.place_extra_objects([{"instance_id": "not_converted", "x": 0.0, "y": 0.0}],
                                      library={"not_converted": {"instance_id": "not_converted",
                                                                 "usd_path": "/tmp/none.usd"}})
        drill.remove_prims(["/World/Distractor_0"])
        RESULTS.setdefault("room", {})["refusals"] = {"restored": _restored(drill)}
        RESULTS["room"]["compiles"] = drill.world_log


class _Dummy:
    """Enough of a world for a light draw (the seeded numbers do not depend on it)."""

    def lighting_snapshot(self):
        return {}

    def set_lighting(self, r):
        pass

    def restore_lighting(self, s):
        pass


# --------------------------------------------------------- the named swap (instance suite)
class TestNamedSwap:
    def test_the_named_swap_centres_the_live_object_on_the_goal_and_offers_the_size_matched_pose(self, inst):
        from vsbench.vs.bench import perturb
        from vsbench.world.perturb_world import ray_depth

        clean = ray_depth(inst, inst.goal_pose)
        spec = inst.spec
        gmn, gmx = inst.object_world_bbox("/World/Obj_0")
        p = perturb.build("swap:@same_small", seed=SEED, spec=spec)
        d = _begin(p, inst)
        small = spec.live_item("same_small")
        assert d["swapped"] is True and d["live_size_m"] == small["size_m"] < d["goal_size_m"]
        lmn, lmx = inst.object_world_bbox("/World/Swap_0")
        c = (gmn + gmx) / 2
        assert np.allclose(((lmn + lmx) / 2)[:2], c[:2], atol=1e-9) and abs(lmn[2]) < 1e-9
        s = float(np.max(lmx - lmn) / np.max(gmx - gmn))
        assert d["size_ratio"] == pytest.approx(s) and s == pytest.approx(0.16 / 0.24, rel=1e-6)
        ref = p.reference_pose()
        P = np.array([c[0], c[1], 0.0])
        np.testing.assert_allclose(ref.position, P + s * (inst.goal_pose.position - P))
        assert d["live_placement"] == inst.extra_placements["/World/Swap_0"]
        assert d["live_placement"]["canonical"] == "library"
        assert p.truth_model()["object"] == "scaled"
        g = _depth(inst, clean)
        p.end(inst)
        RESULTS.setdefault("named_swap", {})["same_small"] = {
            "draw": d, "depth": g, "compile": inst.world_log[-1], "restored": _restored(inst)}

    def test_the_named_same_is_a_recorded_no_op(self, inst):
        from vsbench.vs.bench import perturb

        n = len(inst.world_log)
        p = perturb.build("swap:@same", seed=SEED, spec=inst.spec)
        d = _begin(p, inst)
        assert d["swapped"] is False and d["size_ratio"] == pytest.approx(1.0)
        assert not inst.dirty
        np.testing.assert_allclose(p.reference_pose().position, inst.goal_pose.position)
        p.end(inst)
        assert len(inst.world_log) == n
        RESULTS.setdefault("named_swap", {})["same"] = {"draw": d, "restored": _restored(inst)}

    def test_another_object_has_no_truth_and_a_chain_forwards_the_reference(self, inst):
        from vsbench.vs.bench import perturb

        p = perturb.build("swap:@other_small+ground:concrete", seed=SEED, spec=inst.spec)
        d = _begin(p, inst)
        assert d["parts"] == ["swap_instance", "ground_swap"]
        assert p.reference_pose() is not None and p.truth_model()["object"] == "none"
        assert p.truth_model()["floor"] == "retextured"
        p.end(inst)
        with pytest.raises(KeyError, match="no live object 'nope'"):
            _begin(perturb.build("swap:@nope", spec=inst.spec), inst)
        RESULTS.setdefault("named_swap", {})["other_small+ground"] = {
            "draw": d, "compiles": inst.world_log, "restored": _restored(inst)}


# -------------------------------------------------------------------------- a poster
class TestPoster:
    def test_a_poster_takes_light_and_occluder_and_refuses_ground_and_swap(self, poster):
        from vsbench.vs.bench import perturb
        from vsbench.world.perturb_world import ray_depth

        clean = ray_depth(poster, poster.goal_pose)
        for mode in ("light", "occluder"):
            p = perturb.build(mode, seed=SEED, spec=poster.spec)
            _begin(p, poster)
            if mode == "occluder":
                _depth(poster, clean)
            p.end(poster)
            _restored(poster)
        with pytest.raises(ValueError, match="ground"):
            _begin(perturb.build("ground:concrete", spec=poster.spec), poster)
        with pytest.raises(ValueError, match="object list"):
            _begin(perturb.build("swap:ycb/006_mustard_bottle", spec=poster.spec), poster)
        RESULTS["poster"] = {"modes": ["light", "occluder"], "refused": ["ground:", "swap:"],
                             "restored": _restored(poster), "compiles": poster.world_log}
