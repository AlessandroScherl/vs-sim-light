# vs-sim copy: vssim/bench/perturb.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Perturbations: what changes in the world between the goal capture and the flight.

Purpose
-------
The benchmark's clean condition (P0) hands a method the goal picture of a scene and
flies it in the very same scene. A *perturbation* changes the live world **after** the
goal picture is taken and before the first live frame, so the method must reach a goal
whose picture no longer matches the world exactly - the situation a robot is in. This
module carries the taxonomy's conditions that the simulator can produce today
(``docs/perturbation_taxonomy_v1.md`` in the owner's workspace, frozen 2026-07-30):

``light``
    P2 - the dome and sun lights scaled by U[0.3, 2.0] and set to a colour temperature
    U[2700 K, 6500 K] (the taxonomy's key-light *direction* is not moved: the scene's
    lights are a dome and a distant light with no authored direction to draw around).
    Isaac only (``Scene.set_lighting`` / ``lighting_snapshot`` / ``restore_lighting``).
``distractors``
    P5 - 2 to 5 extra YCB objects, drawn from the pool of objects the scene does not
    contain, put on the floor inside the goal camera's view at least 12 cm from every
    object of the frozen layout and from each other, each with a random yaw. Isaac
    only (``Scene.place_extra_objects`` / ``remove_prims``).
``occluder``
    a neutral grey cube between the goal camera and the scene: on the ray through a
    random image point 15-35 % off the centre, at 35-60 % of the working distance from
    the goal camera, sized to cover 10-18 % of the image width from there. The goal
    picture does not contain it; the approach and the park see it. Isaac only
    (``Scene.put_box`` / ``remove_prim``).
``noise``
    P4's sensor model on every live frame: linearise (gamma 2.2), Poisson-Gaussian
    noise with ``log lambda_shot ~ U[log 1e-4, log 1.2e-2]`` and ``lambda_read`` from
    the Brooks et al. (2019) fit, re-gamma, then brightness +-0.15, contrast x[0.6, 1],
    per-channel white-balance gain +-15 %. The blur of the taxonomy is left out (a
    blurred picture would change what every matcher's resolution means; a separate
    condition if wanted). Any world, numpy only.
``combined``
    P6 - all of the above at the same severities.
``ground:<name-or-path>`` (``ground_swap``, taxonomy P3 for the floor)
    the live world's ground gets ANOTHER texture after the goal picture is taken, so a
    method cannot steer on a background it has seen (owner, 2026-09-22: "we have to
    assume background is different"). ``name`` is a key of :data:`GROUNDS` or an absolute
    path; the scene's own ground geometry (size, tiling) is kept and its texture put back
    at ``end``. Isaac only (``Scene.build_poster``).
``a+b`` (``chain``)
    any two or more of the above at once, e.g. ``swap:Reebok_SOMERSET_RUN+ground:concrete``
    - the instance-generalisation test on a floor the method has never seen.
``swap:<instance_id>`` (``swap_instance``)
    the instance-generalisation test (owner, 2026-09-20): the goal picture shows the
    scene's object (``/World/Obj_0``); for the flight that object is hidden and
    ``<instance_id>`` - another object of the same category, from the asset library -
    is put at the same floor position and yaw with the same pre-rotation and the
    category's size rule. The camera must reach the goal POSE with a different
    instance in view. Isaac only. The "same floor position" is the goal prim's
    TRANSLATE, i.e. the goal model's own origin, not its visible centre: for the shoe
    studies' wingtip the two are 0.02 mm apart, so their records stand; this path is kept
    exactly as it was.
``swap:@<name>`` (``swap_instance``, the instance suite, 2026-09-23)
    the scene's NAMED live object (``live: {<name>: {...}}`` in ``configs/scenes.yaml``,
    merged over the scene's ``placement:`` defaults - :meth:`vsbench.vs.scenes.SceneSpec.live_item`)
    replaces the goal object, oriented and sized by its own fields, with its bounding-box
    centre on the goal object's bounding-box centre. The same instance at another size
    DOES swap; ``@same`` (identical instance and placement) is a recorded no-op. The draw
    records the size ratio ``s`` (live over goal longest extent, as placed) and the
    **size-matched reference pose** - the goal pose scaled about the goal object's floor
    centre by ``s``, where a scaled copy of the goal object would give exactly the goal
    picture (:func:`size_matched_pose`); :meth:`Perturbation.reference_pose` hands it to
    the runner, which scores every landing against it beside the goal pose.
    It also offers the **depth-true pose** (owner, 2026-09-30; :func:`depth_true_pose`):
    the goal pose moved up or down so the camera is as far above the TOP of the live
    object as the goal camera was above the top of the goal object - the pose a grasp
    taught on the goal object needs - through :meth:`Perturbation.depth_true_pose`.

What happened to the object, for the debug log's truth per match (2026-09-30):
:meth:`Perturbation.truth_model` says whether a goal point of the object is still where
it was (``rigid``), moved by the size change about the floor centre (``scaled``, the
same instance at another size) or has no counterpart (``none``, another object), and
whether the floor was re-textured. It is read by :mod:`vsbench.vs.bench.debuglog` only and
never touches a draw, so every record's ``perturbation`` is what it was.

Every draw comes from ``episode_seed(scene, start_id, seed)`` - the same per
(scene, start) for every arm, so a paired read compares arms in the *same* perturbed
world - and is written into the episode record (``perturbation``). ``end()`` undoes
every world edit, so the next episode starts clean; a perturbation that cannot be
undone is refused at ``begin()`` (the lighting snapshot exists for exactly this).

Contract (:class:`Perturbation`)
--------------------------------
``begin(scene, seed, *, goal, working_distance, intrinsics) -> dict`` after the goal
capture, before the start pose is flown; returns the draw. ``live(rgb, step) -> rgb``
per live frame (identity unless the mode touches pixels). ``end(scene)`` always,
in a ``finally``. :func:`build` makes one from a mode name; :func:`for_scene` binds
the scene spec (its objects, its layout) for the modes that need it.

Golden tests: ``tests/fast/test_perturb.py`` (a recording fake scene; the noise model
on a flat frame; the runner integration on the numpy plane).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vsbench.vs.baselines.match import episode_seed

MODES = ("none", "light", "distractors", "occluder", "noise", "combined", "swap_instance",
         "ground_swap", "chain")

#: Ground textures staged on the station, by short name (P3's pool for the floor). A path
#: works too. The scenes' own grounds are `wood` (cluttered2) and `concrete` (industrial1).
GROUNDS = {
    "wood": "/workspace/assets/cluttered2/ground.png",
    "wood3": "/workspace/assets/cluttered3/ground.png",
    "concrete": "/workspace/assets/industrial1/ground.png",
    "objects1": "/workspace/assets/objects1/ground.png",
    "laminate": "/workspace/assets/polyhaven/laminate_floor_02.jpg",
    "plywood": "/workspace/assets/polyhaven/plywood.jpg",
}

#: P2's ranges (taxonomy v1).
LIGHT = {"intensity_scale": (0.3, 2.0), "temperature_k": (2700.0, 6500.0)}
#: P5's ranges.
DISTRACTORS = {"n": (2, 5), "min_gap_m": 0.12, "image_margin": 0.10, "tries": 200}
#: The occluder's ranges.
OCCLUDER = {"offset_frac": (0.15, 0.35), "distance_frac": (0.35, 0.60),
            "width_frac": (0.10, 0.18), "gray": 0.5, "prim_path": "/World/Occluder"}
#: P4's ranges (Brooks et al. 2019 "Unprocessing" fit for the read noise).
NOISE = {"log_shot": (np.log(1e-4), np.log(1.2e-2)), "read_fit": (2.18, 1.20, 0.26),
         "gamma": 2.2, "brightness": 0.15, "contrast": (0.6, 1.0), "wb_gain": 0.15}


def _rng(seed: int, salt: str) -> np.random.Generator:
    return np.random.default_rng([int(seed) % (2**31), int.from_bytes(salt.encode(), "little")])


class Perturbation:
    """The no-op: ``mode == 'none'``. Subclasses override the three verbs."""

    mode = "none"

    def __init__(self, seed: int = 0, spec: Any = None) -> None:
        self.seed = int(seed)
        self.spec = spec
        self.draw: dict = {}
        self._episode_seed = 0

    def episode(self, scene_name: str, start_id: Any) -> int:
        self._episode_seed = episode_seed(scene_name, start_id, self.seed)
        return self._episode_seed

    def begin(self, scene, seed: int, *, goal, working_distance: float, intrinsics) -> dict:
        self.draw = {"mode": self.mode, "seed": int(seed)}
        return self.draw

    def live(self, rgb: np.ndarray, step: int) -> np.ndarray:
        return rgb

    def end(self, scene) -> None:
        return None

    def record(self) -> dict:
        return dict(self.draw)

    def reference_pose(self):
        """A second pose to score the landing against (a :class:`vsbench.vs.scene.Pose`), or
        ``None``: the named instance swap offers the size-matched pose; nothing else does."""
        return None

    def depth_true_pose(self):
        """A third pose to score against (a :class:`vsbench.vs.scene.Pose`), or ``None``: the
        named instance swap offers the depth-true pose (:func:`depth_true_pose`); nothing
        else does. A pure getter, like :meth:`reference_pose`."""
        return None

    def truth_model(self) -> dict:
        """What happened to the goal's points between the goal picture and the flight.

        ``object``: ``rigid`` (the object did not move - also under light, noise,
        distractors, an occluder), ``scaled`` (the same object at another size; ``s`` and
        ``centre_xy`` say how: a point X moves to ``P + s (X - P)``, ``P = (cx, cy, 0)``)
        or ``none`` (another object: its points have no true counterpart). ``floor``:
        ``same`` or ``retextured``; ``floor_z`` the floor plane's height. Read by the
        debug log only (:mod:`vsbench.vs.bench.debuglog`); it never changes a draw.
        """
        return dict(DEFAULT_TRUTH)


#: The truth model of a world that did not change (see :meth:`Perturbation.truth_model`).
DEFAULT_TRUTH = {"object": "rigid", "floor": "same", "floor_z": 0.0}


def depth_true_pose(goal, goal_top_z: float, live_top_z: float):
    """The goal pose moved straight up or down by the change in the object's top height.

    The owner's definition (2026-09-30): the camera is as far above the TOP of the live
    object as the goal camera was above the top of the goal object, with the goal's
    sideways position and orientation - ``p = p_goal + (0, 0, live_top_z - goal_top_z)``.
    The instance suite's goals look straight down, so this is the move along the viewing
    axis; a tilted goal keeps the vertical move (sideways position kept, as worded). For
    the same object at the same size it is the goal pose. Hand example (the shoe, kids'
    size): goal z 0.368797, goal top 0.116238, live top 0.086087 -> z 0.338647.
    """
    from vsbench.vs.scene import Pose

    p = np.asarray(goal.position, np.float64).reshape(3).copy()
    p[2] += float(live_top_z) - float(goal_top_z)
    return Pose(position=p, rotation=np.asarray(goal.rotation, np.float64))


def size_matched_pose(goal, centre_xy, s: float):
    """The goal pose scaled by ``s`` about the floor point under the goal object's centre.

    ``P = (cx, cy, 0)``, ``p* = P + s (p_goal - P)``, the orientation unchanged. A copy of
    the goal object scaled by ``s`` about ``P`` (which is how a swapped object of another
    size sits: its bounding-box centre on the goal's, its bottom on the floor) looks from
    ``p*`` exactly as the goal object looks from the goal pose. For ``s = 1`` it is the
    goal pose.
    """
    from vsbench.vs.scene import Pose

    P = np.array([float(centre_xy[0]), float(centre_xy[1]), 0.0])
    p = np.asarray(goal.position, np.float64).reshape(3)
    return Pose(position=P + float(s) * (p - P), rotation=np.asarray(goal.rotation, np.float64))


#: The placement fields that make two scene items "the same object" for ``swap:@name``.
SAME_OBJECT_KEYS = ("instance_id", "size_m", "canonical", "upright_yaw_add_deg",
                    "pre_rot_x_deg", "yaw_deg")


def _same_object(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in SAME_OBJECT_KEYS)


def _needs(scene, *verbs: str) -> None:
    missing = [v for v in verbs if not callable(getattr(scene, v, None))]
    if missing:
        raise NotImplementedError(
            f"scene {getattr(scene, 'name', scene)!r} offers no {', '.join(missing)}: this "
            "perturbation needs the Isaac world (the numpy plane can only take `noise`)")


# ------------------------------------------------------------------ P2 light
class Light(Perturbation):
    mode = "light"

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        _needs(scene, "set_lighting", "lighting_snapshot", "restore_lighting")
        r = _rng(seed, "light")
        scale = float(r.uniform(*LIGHT["intensity_scale"]))
        temp = float(r.uniform(*LIGHT["temperature_k"]))
        self._snapshot = scene.lighting_snapshot()
        scene.set_lighting({"light_intensity_scale": scale, "light_temperature_k": temp})
        self.draw = {"mode": self.mode, "seed": int(seed), "light_intensity_scale": scale,
                     "light_temperature_k": temp}
        return self.draw

    def end(self, scene) -> None:
        snap = getattr(self, "_snapshot", None)
        if snap is not None:
            scene.restore_lighting(snap)
            self._snapshot = None


# ------------------------------------------------------------ the geometry
def ray_to_plane(u: float, v: float, intrinsics, pose, plane_z: float = 0.0):
    """World point where the ray through pixel ``(u, v)`` of ``pose`` meets ``z = plane_z``
    (``None`` when it never does)."""
    d_cam = np.array([(u - intrinsics.cx) / intrinsics.fx,
                      (v - intrinsics.cy) / intrinsics.fy, 1.0])
    R = np.asarray(pose.rotation, np.float64)
    p = np.asarray(pose.position, np.float64).reshape(3)
    d = R @ d_cam
    if abs(d[2]) < 1e-9:
        return None
    s = (plane_z - p[2]) / d[2]
    if s <= 0:
        return None
    return p + s * d


def _layout_xy(spec) -> list[tuple[float, float]]:
    """The frozen layout's object positions (x, y), or nothing for a poster."""
    if spec is None:
        return []
    from vsbench.vs.scenes import load_layout

    layout = load_layout(spec)
    if not layout:
        return []
    return [(float(o["pos"][0]), float(o["pos"][1])) for o in layout["objects"]]


# -------------------------------------------------------------- P5 distractors
class Distractors(Perturbation):
    mode = "distractors"

    def __init__(self, seed: int = 0, spec: Any = None, pool: list[str] | None = None) -> None:
        super().__init__(seed, spec)
        self.pool = pool            # None: the YCB library minus the scene's own objects
        self._paths: list[str] = []

    def plan(self, seed: int, *, goal, intrinsics, pool: list[str]) -> list[dict]:
        """The seeded placement, Isaac-free: ``[{instance_id, x, y, yaw_deg}]``."""
        r = _rng(seed, "distractors")
        n = int(r.integers(DISTRACTORS["n"][0], DISTRACTORS["n"][1] + 1))
        picks = [str(x) for x in r.choice(np.asarray(pool, dtype=object), size=min(n, len(pool)),
                                          replace=False)]
        taken = list(_layout_xy(self.spec))
        m = DISTRACTORS["image_margin"]
        w, h = intrinsics.width, intrinsics.height
        items = []
        for iid in picks:
            for _ in range(DISTRACTORS["tries"]):
                u = float(r.uniform(m * w, (1 - m) * w))
                v = float(r.uniform(m * h, (1 - m) * h))
                hit = ray_to_plane(u, v, intrinsics, goal)
                if hit is None:
                    continue
                gap = DISTRACTORS["min_gap_m"]
                if all(np.hypot(hit[0] - x, hit[1] - y) >= gap for x, y in taken):
                    taken.append((float(hit[0]), float(hit[1])))
                    items.append({"instance_id": iid, "x": float(hit[0]), "y": float(hit[1]),
                                  "yaw_deg": float(r.uniform(0.0, 360.0)), "pixel": [u, v]})
                    break
        return items

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        _needs(scene, "place_extra_objects", "remove_prims")
        pool = self.pool
        if pool is None:
            from vsbench.vs.assets import load_ycb_library

            own = set(self.spec.object_ids()) if self.spec is not None else set()
            pool = sorted(i for i in load_ycb_library() if i not in own)
        items = self.plan(seed, goal=goal, intrinsics=intrinsics, pool=pool)
        self._paths = scene.place_extra_objects(items)
        self.draw = {"mode": self.mode, "seed": int(seed), "n": len(items),
                     "items": [{k: v for k, v in it.items()} for it in items],
                     "prim_paths": list(self._paths)}
        return self.draw

    def end(self, scene) -> None:
        if self._paths:
            scene.remove_prims(self._paths)
            self._paths = []


# ----------------------------------------------------------------- occluder
class Occluder(Perturbation):
    mode = "occluder"

    def plan(self, seed: int, *, goal, working_distance: float, intrinsics) -> dict:
        r = _rng(seed, "occluder")
        w, h = intrinsics.width, intrinsics.height
        off = float(r.uniform(*OCCLUDER["offset_frac"])) * w
        ang = float(r.uniform(0.0, 2 * np.pi))
        u, v = intrinsics.cx + off * np.cos(ang), intrinsics.cy + off * np.sin(ang) * (h / w)
        dist = float(r.uniform(*OCCLUDER["distance_frac"])) * float(working_distance)
        width_px = float(r.uniform(*OCCLUDER["width_frac"])) * w
        size_m = width_px * dist / float(intrinsics.fx)
        d_cam = np.array([(u - intrinsics.cx) / intrinsics.fx,
                          (v - intrinsics.cy) / intrinsics.fy, 1.0])
        R = np.asarray(goal.rotation, np.float64)
        centre = np.asarray(goal.position, np.float64).reshape(3) + R @ (d_cam * dist)
        return {"centre": [float(c) for c in centre], "size_m": float(size_m),
                "pixel": [float(u), float(v)], "distance_m": dist,
                "image_width_frac": width_px / w, "rotation": R.tolist()}

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        _needs(scene, "put_box", "remove_prim")
        plan = self.plan(seed, goal=goal, working_distance=working_distance, intrinsics=intrinsics)
        scene.put_box(OCCLUDER["prim_path"], plan["centre"], plan["size_m"],
                      np.asarray(plan["rotation"]), gray=OCCLUDER["gray"])
        self._placed = True
        self.draw = {"mode": self.mode, "seed": int(seed), "prim_path": OCCLUDER["prim_path"],
                     **{k: v for k, v in plan.items() if k != "rotation"}}
        return self.draw

    def end(self, scene) -> None:
        if getattr(self, "_placed", False):
            scene.remove_prim(OCCLUDER["prim_path"])
            self._placed = False


# -------------------------------------------------------------------- P4 noise
def draw_noise_params(seed: int) -> dict:
    r = _rng(seed, "noise")
    log_shot = float(r.uniform(*NOISE["log_shot"]))
    a, b, sd = NOISE["read_fit"]
    log_read = a * log_shot + b + float(r.normal(0.0, sd))
    return {"lambda_shot": float(np.exp(log_shot)), "lambda_read": float(np.exp(log_read)),
            "brightness": float(r.uniform(-NOISE["brightness"], NOISE["brightness"])),
            "contrast": float(r.uniform(*NOISE["contrast"])),
            "wb_gain": [float(g) for g in r.uniform(1 - NOISE["wb_gain"], 1 + NOISE["wb_gain"], 3)]}


def apply_noise(rgb: np.ndarray, params: dict, seed: int) -> np.ndarray:
    """P4 on one frame, seeded: linearise, shot + read noise, re-gamma, brightness,
    contrast, white balance. uint8 in, uint8 out."""
    r = np.random.default_rng(int(seed) % (2**32))
    g = float(NOISE["gamma"])
    x = (np.asarray(rgb, np.float64)[..., :3] / 255.0) ** g
    var = params["lambda_shot"] * x + params["lambda_read"]
    x = x + r.normal(0.0, 1.0, x.shape) * np.sqrt(np.maximum(var, 0.0))
    x = np.clip(x, 0.0, 1.0) ** (1.0 / g)
    x = (x - 0.5) * params["contrast"] + 0.5 + params["brightness"]
    x = x * np.asarray(params["wb_gain"], np.float64).reshape(1, 1, 3)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


class Noise(Perturbation):
    mode = "noise"

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        self._params = draw_noise_params(seed)
        self._seed = int(seed)
        self.draw = {"mode": self.mode, "seed": int(seed), **self._params}
        return self.draw

    def live(self, rgb: np.ndarray, step: int) -> np.ndarray:
        return apply_noise(rgb, self._params, self._seed * 1009 + int(step))


# ------------------------------------------------------------------ P6 combined
class Combined(Perturbation):
    mode = "combined"

    def __init__(self, seed: int = 0, spec: Any = None) -> None:
        super().__init__(seed, spec)
        self.parts = [Light(seed, spec), Distractors(seed, spec), Occluder(seed, spec),
                      Noise(seed, spec)]

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        self.draw = {"mode": self.mode, "seed": int(seed)}
        started = []
        try:
            for part in self.parts:
                self.draw[part.mode] = part.begin(scene, seed, goal=goal,
                                                  working_distance=working_distance,
                                                  intrinsics=intrinsics)
                started.append(part)
        except Exception:
            for part in reversed(started):
                part.end(scene)
            raise
        return self.draw

    def live(self, rgb, step):
        for part in self.parts:
            rgb = part.live(rgb, step)
        return rgb

    def end(self, scene) -> None:
        for part in reversed(self.parts):
            part.end(scene)


# --------------------------------------------------------- swap the instance
class SwapInstance(Perturbation):
    """Hide the scene's object, show ``live_instance`` in its place for the flight.

    ``swap:<instance_id>`` is the original path (model-origin placement, the category's
    size rule); ``swap:@<name>`` places the scene's named live object at the goal
    object's visible centre and offers the size-matched reference pose.
    """

    mode = "swap_instance"
    GOAL_PRIM = "/World/Obj_0"

    def __init__(self, seed: int = 0, spec: Any = None, live_instance: str = "") -> None:
        super().__init__(seed, spec)
        if not live_instance:
            raise ValueError("swap_instance needs the live instance id: `swap:<instance_id>`"
                             " or a named live object: `swap:@<name>`")
        live_instance = str(live_instance)
        self.live_name = live_instance[1:] if live_instance.startswith("@") else None
        if self.live_name == "":
            raise ValueError("`swap:@` needs the live object's name, e.g. `swap:@same_small`")
        self.live_instance = live_instance
        self._paths: list[str] = []
        self._hidden = False
        self._reference = None
        self._depth_true = None
        self._truth: dict | None = None

    def reference_pose(self):
        return self._reference

    def depth_true_pose(self):
        return self._depth_true

    def truth_model(self) -> dict:
        return dict(self._truth) if self._truth is not None else dict(DEFAULT_TRUTH)

    def _begin_named(self, scene, seed, goal) -> dict:
        from vsbench.vs.scenes import placed_item

        _needs(scene, "object_world_bbox", "set_prim_visible", "place_extra_objects",
               "remove_prims")
        if self.spec is None or not self.spec.object_ids():
            raise ValueError("swap:@<name> needs the scene spec with its object list")
        goal_item = placed_item(self.spec, (self.spec.raw.get("objects") or [])[0])
        live = self.spec.live_item(self.live_name)
        gmn, gmx = (np.asarray(v, np.float64) for v in scene.object_world_bbox(self.GOAL_PRIM))
        centre = (gmn + gmx) / 2.0
        swapped = not _same_object(goal_item, live)
        placement = None
        if swapped:
            scene.set_prim_visible(self.GOAL_PRIM, False)
            self._hidden = True
            new = {**live, "x": float(centre[0]), "y": float(centre[1])}
            self._paths = scene.place_extra_objects([new], prefix="/World/Swap")
            lmn, lmx = (np.asarray(v, np.float64)
                        for v in scene.object_world_bbox(self._paths[0]))
            placement = (getattr(scene, "extra_placements", None) or {}).get(self._paths[0])
        else:
            lmn, lmx = gmn, gmx
        s = float(np.max(lmx - lmn) / np.max(gmx - gmn))
        self._reference = size_matched_pose(goal, centre[:2], s)
        # the debug log's two offers (2026-09-30), kept OFF the draw so the record's
        # `perturbation` is bit-identical: the depth-true pose from the two tops, and what
        # happened to the object - the same instance at another size is `scaled` about the
        # floor centre (how the copy was placed: box centre on the goal's, bottom on the floor)
        self._depth_true = depth_true_pose(goal, gmx[2], lmx[2])
        same_but_size = all(goal_item.get(k) == live.get(k)
                            for k in SAME_OBJECT_KEYS if k != "size_m")
        if not swapped:
            self._truth = dict(DEFAULT_TRUTH)
        elif same_but_size:
            self._truth = {**DEFAULT_TRUTH, "object": "scaled", "s": s,
                           "centre_xy": [float(centre[0]), float(centre[1])]}
        else:
            self._truth = {**DEFAULT_TRUTH, "object": "none"}
        self.draw = {
            "mode": self.mode, "seed": int(seed), "live_name": self.live_name,
            "goal_instance": str(goal_item["instance_id"]),
            "live_instance": str(live["instance_id"]),
            "goal_size_m": goal_item.get("size_m"), "live_size_m": live.get("size_m"),
            "swapped": swapped, "size_ratio": s,
            "goal_bbox": [[float(v) for v in gmn], [float(v) for v in gmx]],
            "live_bbox": [[float(v) for v in lmn], [float(v) for v in lmx]],
            "floor_centre_xy": [float(centre[0]), float(centre[1])],
            "reference_position": [float(v) for v in self._reference.position],
            "live_placement": placement, "prim_paths": list(self._paths)}
        return self.draw

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        self._reference, self._depth_true, self._truth = None, None, None
        if self.live_name is not None:
            return self._begin_named(scene, seed, goal)
        _needs(scene, "object_pose", "set_prim_visible", "place_extra_objects", "remove_prims")
        if self.spec is None or not self.spec.object_ids():
            raise ValueError("swap_instance needs the scene spec with its object list")
        item = dict((self.spec.raw.get("objects") or [])[0])
        goal_instance = str(item["instance_id"])
        if self.live_instance == goal_instance:
            self.draw = {"mode": self.mode, "seed": int(seed), "goal_instance": goal_instance,
                         "live_instance": goal_instance, "swapped": False}
            return self.draw
        pos, _quat = scene.object_pose(self.GOAL_PRIM)
        scene.set_prim_visible(self.GOAL_PRIM, False)
        self._hidden = True
        new = {"instance_id": self.live_instance, "category": item.get("category", "shoe"),
               "x": float(pos[0]), "y": float(pos[1]), "yaw_deg": float(item.get("yaw_deg", 0.0))}
        if item.get("pre_rot_x_deg"):
            new["pre_rot_x_deg"] = float(item["pre_rot_x_deg"])
        self._paths = scene.place_extra_objects([new], prefix="/World/Swap")
        self._truth = {**DEFAULT_TRUTH, "object": "none"}       # another object: no truth
        self.draw = {"mode": self.mode, "seed": int(seed), "goal_instance": goal_instance,
                     "live_instance": self.live_instance, "swapped": True,
                     "xy": [new["x"], new["y"]], "prim_paths": list(self._paths)}
        return self.draw

    def end(self, scene) -> None:
        if self._paths:
            scene.remove_prims(self._paths)
            self._paths = []
        if self._hidden:
            scene.set_prim_visible(self.GOAL_PRIM, True)
            self._hidden = False


# ------------------------------------------------------------- P3: the floor
class GroundSwap(Perturbation):
    """Re-texture the live world's ground; the goal picture keeps the scene's own."""

    mode = "ground_swap"
    PRIM = "/World/Ground"

    def __init__(self, seed: int = 0, spec: Any = None, texture: str = "") -> None:
        super().__init__(seed, spec)
        if not texture:
            raise ValueError("ground needs a texture: `ground:<name|path>`, "
                             f"names {sorted(GROUNDS)}")
        self.texture = GROUNDS.get(texture, texture)
        self.texture_name = str(texture)
        self._orig = None

    def _ground(self) -> dict:
        g = (self.spec.raw.get("ground") if self.spec is not None else None) or {}
        if not g.get("texture"):
            raise ValueError("ground swap needs the scene spec's `ground` block")
        return g

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        _needs(scene, "build_poster")
        g = self._ground()
        size = float(g.get("size_m", 2.4))
        tiles = size / float(g.get("tile_period_m", size))
        self._orig = str(g["texture"])
        scene.build_poster(size, size, self.texture, prim_path=self.PRIM, uv_tiles=tiles)
        self.draw = {"mode": self.mode, "seed": int(seed), "ground": self.texture_name,
                     "texture": self.texture, "goal_texture": self._orig}
        return self.draw

    def truth_model(self) -> dict:
        """The floor shows another picture; the object is untouched."""
        return {**DEFAULT_TRUTH, "floor": "retextured"}

    def end(self, scene) -> None:
        if self._orig is None:
            return
        g = self._ground()
        size = float(g.get("size_m", 2.4))
        scene.build_poster(size, size, self._orig, prim_path=self.PRIM,
                           uv_tiles=size / float(g.get("tile_period_m", size)))
        self._orig = None


# ------------------------------------------------------------------ `a+b`
class Chain(Perturbation):
    """Several perturbations at once, in order; undone in reverse."""

    mode = "chain"

    def __init__(self, parts: list) -> None:
        super().__init__(seed=0, spec=None)
        self.parts = list(parts)

    def episode(self, scene_name: str, start_id: Any) -> int:
        seeds = [p.episode(scene_name, start_id) for p in self.parts]
        self._episode_seed = seeds[0] if seeds else 0
        return self._episode_seed

    def begin(self, scene, seed, *, goal, working_distance, intrinsics) -> dict:
        self.draw = {"mode": self.mode, "seed": int(seed),
                     "parts": [p.mode for p in self.parts]}
        started = []
        try:
            for part in self.parts:
                self.draw[part.mode] = part.begin(scene, seed, goal=goal,
                                                  working_distance=working_distance,
                                                  intrinsics=intrinsics)
                started.append(part)
        except Exception:
            for part in reversed(started):
                part.end(scene)
            raise
        return self.draw

    def live(self, rgb, step):
        for part in self.parts:
            rgb = part.live(rgb, step)
        return rgb

    def end(self, scene) -> None:
        for part in reversed(self.parts):
            part.end(scene)

    def reference_pose(self):
        """The first part's reference pose - at most one part (the named swap) has one."""
        for part in self.parts:
            ref = part.reference_pose()
            if ref is not None:
                return ref
        return None

    def depth_true_pose(self):
        """The first part's depth-true pose - at most one part (the named swap) has one."""
        for part in self.parts:
            ref = part.depth_true_pose()
            if ref is not None:
                return ref
        return None

    def truth_model(self) -> dict:
        """The parts merged: the object from the part that changed it, the floor
        re-textured when any part re-textured it."""
        out = dict(DEFAULT_TRUTH)
        for part in self.parts:
            tm = part.truth_model()
            if tm.get("object", "rigid") != "rigid":
                out.update({k: v for k, v in tm.items() if k not in ("floor", "floor_z")})
            if tm.get("floor") == "retextured":
                out["floor"] = "retextured"
        return out


BUILDERS = {"none": Perturbation, "light": Light, "distractors": Distractors,
            "occluder": Occluder, "noise": Noise, "combined": Combined}


def build(mode: str, *, seed: int = 0, spec: Any = None) -> Perturbation:
    """A perturbation from its mode name; ``spec`` is the scene's :class:`SceneSpec`
    (needed by ``distractors`` and ``combined`` for the pool and the layout, and by
    ``swap:<instance_id>`` for the object to replace)."""
    if "+" in mode:
        return Chain([build(m, seed=seed, spec=spec) for m in mode.split("+") if m])
    if mode.startswith("swap:"):
        return SwapInstance(seed=seed, spec=spec, live_instance=mode.split(":", 1)[1])
    if mode.startswith("ground:"):
        return GroundSwap(seed=seed, spec=spec, texture=mode.split(":", 1)[1])
    if mode not in BUILDERS:
        raise ValueError(f"perturbation mode must be one of {MODES}, swap:<instance_id>, "
                         f"swap:@<live name>, ground:<name|path> or a `+` chain of those, "
                         f"got {mode!r}")
    return BUILDERS[mode](seed=seed, spec=spec)
