# vs-sim copy: vssim/bake.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Freeze a settled layout, once.

Purpose
-------
PhysX settling is not bit-stable across processes or machines, so a goal pose
derived from a fresh settle would differ from the goal every stored campaign was
scored against. The rule is therefore absolute: **a re-settle may never define a
goal.** A scene that wants gravity is settled exactly once, here, and its object
poses are written to the ``layout_file`` its spec names; every later build applies
that file (:func:`vsbench.vs.scenes.check_layout` refuses an id drift).

All three benchmark rooms already carry a committed layout (``configs/layouts/``),
so this module runs only when a **new** scene is added
(``scripts/bake_layout.py``).

Fences
------
- **Idempotent.** An existing layout file is left untouched; deleting it is an
  explicit act.
- **Poses are read before ``stop()``**, which resets the xforms, and re-authored
  after; the physics APIs are then stripped again so the scene is static.
- **An escaped object reverts to its analytic pose, loudly.** A silent escape
  would bake a scene whose bounding box - and therefore whose goal distance - is
  wrong.
- **fsync before rename.** Isaac can die natively at interpreter exit; the
  artifact must be on disk before that happens.

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/data/bake.py``),
ported there from ``genvs-sim`` ``src/eval/n36_bake_layout.py`` and
``src/eval/eval_scenes.py`` (``settle_objects``, ``_ensure_ground_collider``).

Golden tests: ``tests/fast/test_scenes.py`` covers the Isaac-free half (the
layout contract and the id-drift refusal). The settle itself is station-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from vsbench.vs.camera import REPO_ROOT
from vsbench.vs.scenes import SceneSpec, drop_quat


def ensure_ground_collider(scene) -> None:                # pragma: no cover - Isaac
    """An invisible static collision slab with its top at ``z = 0``: the settle
    floor. The rendered ground quad has no collision, and a box is tunnel-proof."""
    from pxr import Gf, UsdGeom, UsdPhysics
    path = "/World/GroundCollider"
    if scene.stage.GetPrimAtPath(path).IsValid():
        return
    cube = UsdGeom.Cube.Define(scene.stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.1))
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    scene.sim.update()


def settle_objects(scene, placed: list, handles: list, cfg: dict) -> int:
    """Gravity-settle the placed objects; returns how many escaped.

    ``cfg``: ``seed``, ``steps``, ``bound_xy_m``, either ``tilt_deg`` (exact - the
    drill drops at 90 deg so it lands lying) or ``tilt_max_deg`` (a uniform cap),
    and ``drop_m`` as either a scalar (the eval bakes) or a ``(lo, hi)`` range
    drawn per object (the dataset render's 0.06-0.12 m). An object whose spec says
    ``settle: false`` is held upright and static but still gets colliders, so
    dropped objects bounce off it.

    **One drop orientation, and it is** :func:`vsbench.vs.scenes.drop_quat`. PhysX is
    not bit-stable across processes: two settles of the same spec land objects
    centimetres to a decimetre apart, which is exactly why a settled layout is
    frozen to a file once and applied from then on.
    """
    if not placed:
        return 0
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    try:
        from pxr import PhysxSchema
    except ImportError:                                   # pragma: no cover
        PhysxSchema = None

    rng = np.random.default_rng(int(cfg["seed"]))
    drop_cfg = cfg.get("drop_m", 0.06)
    drop_range = (tuple(float(v) for v in drop_cfg)
                  if isinstance(drop_cfg, (tuple, list)) else None)
    steps = int(cfg.get("steps", 240))
    bound = float(cfg.get("bound_xy_m", 0.9))
    tilt_deg = cfg.get("tilt_deg")
    tilt_max = float(cfg.get("tilt_max_deg", 15.0))

    def _collide(inner):
        for p in Usd.PrimRange(inner):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(
                    "convexHull")

    def _uncollide(inner):
        for p in Usd.PrimRange(inner):
            for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
                if p.HasAPI(api):
                    p.RemoveAPI(api)

    ensure_ground_collider(scene)
    prepped, statics = [], []
    for entry, (t_op, o_op, inner) in zip(placed, handles, strict=True):
        _collide(inner)
        if entry.get("settle") is False:
            statics.append(inner)
            continue
        t0, q0 = t_op.Get(), o_op.Get()                   # the analytic pose
        q = drop_quat(rng, tilt_deg=tilt_deg, tilt_max_deg=tilt_max)
        o_op.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
        drop = rng.uniform(*drop_range) if drop_range else float(drop_cfg)
        t_op.Set(Gf.Vec3d(float(t0[0]), float(t0[1]), float(t0[2]) + drop))
        top = scene.stage.GetPrimAtPath(entry["prim_path"])
        UsdPhysics.RigidBodyAPI.Apply(top)
        if PhysxSchema is not None:
            px = PhysxSchema.PhysxRigidBodyAPI.Apply(top)
            px.CreateLinearDampingAttr(0.8)
            px.CreateAngularDampingAttr(2.0)
        prepped.append((entry, t_op, o_op, t0, q0, inner))
    scene.sim.update()

    sc = scene.simulation_context
    sc.play()
    for _ in range(steps):
        sc.step(render=False)
    settled = [(t_op.Get(), o_op.Get()) for _e, t_op, o_op, _t0, _q0, _i in prepped]
    sc.stop()                                             # resets the xforms

    n_escaped = 0
    for (entry, t_op, o_op, t0, q0, inner), (t1, q1) in zip(prepped, settled,
                                                            strict=True):
        if (abs(t1[0]) > bound or abs(t1[1]) > bound or t1[2] < -0.02 or t1[2] > 0.5):
            n_escaped += 1
            t_op.Set(t0)
            o_op.Set(q0)
        else:
            t_op.Set(Gf.Vec3d(float(t1[0]), float(t1[1]), float(t1[2])))
            o_op.Set(q1)
        top = scene.stage.GetPrimAtPath(entry["prim_path"])
        top.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if PhysxSchema is not None:
            top.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        _uncollide(inner)
    for inner in statics:
        _uncollide(inner)
    scene.sim.update()
    return n_escaped


def write_layout(layout: dict, path: Path | str) -> Path:
    """Write a layout json atomically, fsynced before the rename."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(layout, indent=1))
    with open(tmp, "r+") as fh:
        os.fsync(fh.fileno())
    tmp.rename(out)
    return out


def bake_layout(spec: SceneSpec, scene=None,
                repo_root: Path | str = REPO_ROOT) -> Path | None:  # pragma: no cover
    """Settle ``spec`` once and freeze its object poses into its ``layout_file``.

    Returns the path written, or ``None`` when the file already existed. Needs
    Isaac (``$ISAAC_PY``) unless a scene object is injected.
    """
    if spec.kind != "multi_object":
        raise ValueError(f"{spec.name} is {spec.kind!r} - only multi_object scenes "
                         f"have layouts to bake")
    rel = spec.layout_file
    if not rel:
        raise ValueError(f"{spec.name} declares no layout_file - nothing to bake into")
    out = Path(repo_root) / rel
    if out.exists():
        print(f"[bake] {out} exists - keeping the frozen bake (delete it explicitly "
              f"to re-settle)", flush=True)
        return None
    if not spec.raw.get("settle"):
        raise ValueError(f"{spec.name} has no settle block - a bake would just copy "
                         f"analytic poses; refused")
    if scene is None:
        from vsbench.vs.isaac_scene import IsaacScene
        # `goal_render_warmup=1`: a bake captures NO goal, and the field has no default.
        scene = IsaacScene(goal_render_warmup=1)
    fresh = SceneSpec(name=spec.name, kind=spec.kind,
                      raw={k: v for k, v in spec.raw.items() if k != "layout_file"})
    built = scene.build_eval_scene(fresh, allow_settle=True)
    layout = {"scene": spec.name, **built["info"]["layout"]}
    path = write_layout(layout, out)
    print(f"[bake] baked {spec.name} -> {path} ({len(layout['objects'])} objects, "
          f"goal distance {built['distance_m']:.3f} m)", flush=True)
    return path
