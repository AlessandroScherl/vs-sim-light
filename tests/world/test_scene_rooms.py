"""The three rooms in ``MujocoScene``.

Per room (``industrial1_drill``, ``cluttered2``, ``cluttered3``), on one build:

1. goal     - the goal derived from the converted objects' box (``room_goal``) passes
              ``check_scene_goal`` against every pinned start file of the room (the rooms
              have one, ``v2``) and is the goal used; residuals recorded;
2. depth    - at the goal, the check script's near start (``sample_near``, first draw) and
              the first two ``narrow`` starts, on a stride-4 pixel grid, each pixel classed
              by an independent ray caster (``mujoco.mj_multiRay`` from the camera centre on
              the compiled model: which geom the pixel's ray hits): ground pixels against the
              analytic ``z = 0`` plane (``vsbench.reference.plane_zdepth``), object pixels
              against the ray's own distance; median <= 1 mm, p99 <= 3 mm each; every pixel of the full frame finite, ``>= 0``, hit pixels in
              ``[near, far]``, and depth 0 exactly where the ray caster finds nothing;
3. objmask  - the goal mask the ``objmask`` arms get (the copied
              ``labels.object_mask_from_depth`` of the goal depth against ``z = 0``, the
              runner's call): share in (0.02, 0.6); ``vitvs_dinov3_objmask`` builds and
              asks for that mask (``goal_mask_mode == "objects"``), checked once;
4. determinism - one pose drawn twice is the same bytes; the scene rebuilt (after
              ``close``) draws the same goal bytes.

The numbers go to a pytest temp file (``-s`` prints where). The module needs the rooms'
assets (the floor textures and the converted objects) and pins pictures drawn by Mesa
llvmpipe: it runs under software GL (station mode, ``source tools/cpu_env.sh``, or
``VSBENCH_SOFTWARE_GL=1``) and is skipped otherwise. Each room build holds ~3 GB
(cluttered2's nine 4096^2 textures, uploaded to both GL contexts), one room at a time.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vsbench.glenv import software_gl_requested  # noqa: E402

pytestmark = pytest.mark.skipif(
    not software_gl_requested(),
    reason="pins llvmpipe pictures: needs software GL (station mode or VSBENCH_SOFTWARE_GL=1)")

REPO = Path(__file__).resolve().parents[2]
#: the scenes whose assets this module needs (tests/conftest.py skips it without them)
NEEDS_SCENES = ("industrial1_drill", "cluttered2", "cluttered3")
OUT = Path("u4_room_gates.json")   # written to a pytest temp dir
ROOMS = ("industrial1_drill", "cluttered2", "cluttered3")
STRIDE = 4
MASK_SHARE = (0.02, 0.6)
RESULTS: dict = {}


def write_gates() -> bool:
    return False   # the numbers go to a pytest temp file (print them with -s)


@pytest.fixture(scope="module", autouse=True)
def _write_results(tmp_path_factory):
    yield
    if not RESULTS:
        return
    out = OUT if write_gates() else tmp_path_factory.mktemp("gates") / OUT.name
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    for k, v in RESULTS.items():
        if k == "rooms":
            old.setdefault("rooms", {}).update(v)
        else:
            old[k] = v
    old["unit"] = "the three rooms in MujocoScene"
    old["written"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    print(f"[gates] wrote {out}")


def _scene(name):
    from vsbench.world.mujoco_scene import MujocoScene

    if not software_gl_requested():
        pytest.skip("MujocoScene renders: needs VSBENCH_SOFTWARE_GL=1")
    scene = MujocoScene(name)
    gl = str(getattr(scene.renderer, "gl_renderer", "")).lower()
    if scene.render_mode != "mujoco-llvmpipe" or "llvmpipe" not in gl:
        scene.close()
        pytest.fail(f"MuJoCo opened {scene.render_mode!r} ({gl!r}), not llvmpipe - refusing")
    return scene


def near_start(scene) -> dict:
    """``scripts/check_baselines.py``'s near start: the benchmark's near sampler on this
    scene's goal (3-8 mm, <= 3 deg), first draw, seed 20260902."""
    from vsbench.vs import starts as startsmod

    g = scene.goal_pose
    s = startsmod.sample_near(g.position, g.rotation, scene.working_distance, n=1)[0]
    return dict(s, starts_seed=startsmod.NEAR["seed"])


def _stats(e) -> dict:
    e = np.asarray(e, float) * 1e3
    return {"n": int(e.size), "median_mm": float(np.median(e)), "p99_mm": float(np.percentile(e, 99)),
            "max_mm": float(e.max())}


def depth_check(scene, pose, stride: int = STRIDE) -> dict:
    """The rendered depth of ``pose`` against an independent ray caster on a pixel grid."""
    import mujoco

    from vsbench.reference import plane_zdepth
    from vsbench.vs.scene import pose_to_T

    scene.reset(pose)
    _rgb, depth, _p = scene.render()
    cam = scene.render_camera
    T = pose_to_T(pose)
    m, d = scene.renderer.model, scene.renderer.data
    mujoco.mj_forward(m, d)
    ground_id = m.geom("vsbench_obj0").id
    size = scene.spec.raw["ground"]["size_m"]
    plane = plane_zdepth(cam, T, np.eye(4), size=(size, size))
    rays = cam.pixel_rays()
    vv, uu = np.mgrid[stride // 2:cam.height:stride, stride // 2:cam.width:stride]
    vv, uu = vv.ravel(), uu.ravel()
    dw = rays[vv, uu] @ T[:3, :3].T
    norm = np.linalg.norm(dw, axis=1)
    unit = np.ascontiguousarray(dw / norm[:, None])
    n = len(vv)
    gid = np.zeros(n, np.int32)
    dist = np.zeros(n, np.float64)
    mujoco.mj_multiRay(m, d, np.ascontiguousarray(T[:3, 3], dtype=np.float64), unit.ravel(), None,
                       True, -1, gid, dist, None, n, float(cam.far) * 2.0)
    z = depth[vv, uu].astype(np.float64)
    miss = (gid < 0) | (dist < 0)
    ground = ~miss & (gid == ground_id)
    objs = ~miss & ~ground
    g_err = np.abs(z[ground] - plane[vv[ground], uu[ground]])
    o_err = np.abs(z[objs] - dist[objs] / norm[objs])
    n_miss, agree_miss = int(miss.sum()), int((z[miss] == 0.0).sum())
    full = depth.astype(np.float64)
    hit = full > 0
    return {"ground_vs_plane": _stats(g_err), "objects_vs_ray": _stats(o_err) if o_err.size else None,
            "n_grid": int(n), "n_ground": int(ground.sum()), "n_object": int(objs.sum()),
            "n_ray_miss": n_miss, "ray_miss_with_depth_0": agree_miss,
            "frame_finite": bool(np.isfinite(full).all()), "frame_nonneg": bool((full >= 0).all()),
            "hit_share": float(hit.mean()),
            "hit_in_clip": bool(((full[hit] >= cam.near) & (full[hit] <= cam.far)).all()),
            "median_depth_m": float(np.median(full[hit]))}


def _md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.mark.parametrize("room", ROOMS)
def test_room_gates(room):
    from vsbench.vs import benchmark
    from vsbench.vs import starts as startsmod
    from vsbench.vs.labels import object_mask_from_depth
    from vsbench.vs.scene import Pose, Scene, rotation_error_deg, translation_error_mm

    row: dict = {}
    scene = _scene(room)
    try:
        assert isinstance(scene, Scene)
        row["compile_s"] = scene.compile_s
        row["build_s"] = scene.meta["build_s"]
        row["gl_renderer"] = scene.renderer.gl_renderer
        row["render_mode"] = scene.render_mode
        look_name = scene.look.name
        row["look"] = {"look": look_name, "curve_id": scene.look.curve_id,
                       "curve_md5": scene.look.curve_md5}
        # 1. the goal
        gc_ = scene.goal_check
        assert gc_ is not None and gc_["files"], "a room must have a pinned start file"
        for which in [f["file"] for f in gc_["files"]]:
            benchmark.check_scene_goal(scene, benchmark.load_starts(room, which))
        assert gc_["all_pass"] and gc_["used"] == "derived", gc_
        row["goal"] = {"files": gc_["files"], "used": gc_["used"],
                       "distance_m": scene.working_distance,
                       "bbox_min": gc_["derived"]["bbox_min"], "bbox_max": gc_["derived"]["bbox_max"]}
        # 2. depth
        payload = benchmark.load_starts(room, "v2")
        views = [("goal", scene.goal_pose)]
        ns = near_start(scene)
        views.append((f"near {ns['start_id']}", Pose.from_start(ns)))
        views += [(f"narrow {s['start_id']}", Pose.from_start(s))
                  for s in startsmod.starts_for_cell(payload, "narrow_2")]
        np_ = Pose.from_start(ns)
        row["near_start"] = {"start_id": ns["start_id"], "seed": ns["starts_seed"],
                             "pos_err_mm": translation_error_mm(np_.position, scene.goal_pose.position),
                             "rot_err_deg": rotation_error_deg(np_.rotation, scene.goal_pose.rotation)}
        dviews = []
        for tag, pose in views:
            r = depth_check(scene, pose)
            r["view"] = tag
            dviews.append(r)
        row["depth"] = {"views": dviews, "stride_px": STRIDE,
                        "gate": "per view: ground pixels vs the analytic z = 0 plane and object "
                                "pixels vs mujoco.mj_multiRay, median <= 1 mm and p99 <= 3 mm; the full "
                                "frame finite, >= 0, hits inside [near, far]; depth 0 where the ray "
                                "caster misses"}
        # 3. the objects' goal mask (the runner's call)
        scene.reset(scene.goal_pose)
        goal_rgb, goal_depth, _ = scene.render()
        mask = object_mask_from_depth(goal_depth, scene.intrinsics, scene.goal_pose)
        share = float(mask.mean())
        row["objmask_share"] = share
        # 4. determinism
        p = views[1][1]
        scene.reset(p)
        a, da, _ = scene.render()
        b, db, _ = scene.render()
        same = bool(np.array_equal(a, b) and np.array_equal(da, db))
        md5_goal, md5_goal_depth = _md5(goal_rgb), _md5(goal_depth)
    finally:
        scene.close()
    del scene
    gc.collect()
    scene2 = _scene(room)
    try:
        g2, d2 = scene2.render_goal()
        rebuilt_same = _md5(g2) == md5_goal and _md5(d2) == md5_goal_depth
    finally:
        scene2.close()
    del scene2
    gc.collect()
    row["determinism"] = {"redraw_same_bytes": same, "rebuild_same_goal": rebuilt_same,
                          "goal_frame_md5": md5_goal, "goal_depth_md5": md5_goal_depth}
    RESULTS.setdefault("rooms", {})[room] = row
    print(json.dumps(row, indent=1)[:4000])
    for v in dviews:
        assert v["ground_vs_plane"]["median_mm"] <= 1.0 and v["ground_vs_plane"]["p99_mm"] <= 3.0, v
        if v["objects_vs_ray"] is not None:
            assert v["objects_vs_ray"]["median_mm"] <= 1.0 and v["objects_vs_ray"]["p99_mm"] <= 3.0, v
        assert v["frame_finite"] and v["frame_nonneg"] and v["hit_in_clip"], v
        assert v["ray_miss_with_depth_0"] == v["n_ray_miss"], v
    assert dviews[0]["objects_vs_ray"] is not None and dviews[0]["hit_share"] == 1.0
    assert MASK_SHARE[0] < share < MASK_SHARE[1], share
    assert same and rebuilt_same


def test_objmask_arm_builds():
    """``vitvs_dinov3_objmask`` (the protocol's cell) builds on the rooms' intrinsics and asks
    the loop for the objects' goal mask."""
    import torch

    from vsbench.vs.scenes import load_scenes
    from vsbench.world import fly
    from vsbench.world.camera import d435i_intrinsics

    torch.set_num_threads(2)
    protocol = fly.load_protocol()
    cfg = fly.arm_config(protocol, "vitvs_dinov3_objmask", device="cpu")
    specs, _ = load_scenes()
    arm = fly.build(protocol, "vitvs_dinov3_objmask", cfg, intrinsics=d435i_intrinsics(),
                    scene_name="cluttered2", scene_spec=specs["cluttered2"])
    mode = str(getattr(arm, "goal_mask_mode", "none"))
    RESULTS["objmask_arm"] = {"arm": "vitvs_dinov3_objmask", "label": getattr(arm, "name", None),
                              "goal_mask_mode": mode, "cuda": bool(torch.cuda.is_available())}
    assert mode == "objects"
    assert not torch.cuda.is_available()
