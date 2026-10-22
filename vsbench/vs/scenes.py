# vs-sim copy: vssim/scenes.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The scenes: specs, goal poses, scatter, baked layouts.

Purpose
-------
Six benchmark scenes - three posters and three 3-D rooms - and the two things that
must be identical across methods and across sessions are the **goal pose** and the
**object layout**. Both are derived here, in pure numpy, so they can be checked
without booting Isaac and pinned by a test against the frozen start files.

The six (``configs/scenes.yaml``; the yaml also carries older probe scenes, which
:data:`EVAL_SCENES` deliberately does not list):

======================= ======== ================================= ==============
scene                   kind     goal rule                         distance (m)
======================= ======== ================================= ==============
``hollywood``           poster   ``fill_frame`` coverage 1.35       0.4341815108
``poster2``             poster   ``fill_frame`` coverage 1.35       0.4341815108
``poster3``             poster   ``fill_frame`` coverage 1.35       0.4341815108
``industrial1_drill``   clutter  ``scene_view`` frame_fill 0.85     0.4422820342
``cluttered2``          clutter  ``scene_view`` frame_fill 0.85     1.1689734192
``cluttered3``          clutter  ``scene_view`` frame_fill 0.85     1.2853190576
======================= ======== ================================= ==============

The poster distances are a closed-form function of ``camera.yaml`` and the plate
size, so :func:`distance_to_fill_frame` reproduces them offline. The three
``scene_view`` distances are a function of the **measured** scene bounding box,
which only Isaac can produce - the test therefore drives
:func:`scene_view_goal_pose` from the box implied by each frozen start file and
asserts the pose comes back exactly.

Beside the six, the file carries the **instance suite** (2026-09-23): twelve scenes
``inst_<kind>`` / ``inst_<kind>_rev`` with one object each, a ``placement:`` block (the
kind's orientation, stated once), a ``prompt:`` word and named ``live:`` objects. The
placement fields (:func:`item_quat`, :func:`placed_item`, :func:`long_axis_yaw`) are pure
here and applied on the stage by :meth:`vsbench.vs.isaac_scene.IsaacScene.place_objects`.

Fences
------
- **A re-settle may never define a goal.** PhysX settling is not bit-stable
  across processes, so a scene that declares a ``layout_file`` must load it and
  refuse an id drift (:func:`check_layout`); baking is a separate, idempotent act
  (:mod:`vsbench.vs.bake`).
- **Distances are derived, never tabulated.** The numbers in the table above are
  test expectations, not constants the code reads.
- **The scatter rectangle is aspect-matched** to the 16:9 goal frame (distinct
  ``half_extent_x_m`` / ``half_extent_y_m``), because image-right is world +X
  under :func:`scene_view_goal_pose`.

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/data/scenes.py``),
ported there from ``genvs-sim`` ``src/eval/eval_scenes.py`` and proven equal to
it on 200 random boxes (a test here repeats that check when the old checkout is
beside this one).

Golden tests: ``tests/fast/test_scenes.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from vsbench.vs.camera import REPO_ROOT, CameraConfig

DEFAULT_SCENES_YAML = REPO_ROOT / "configs" / "scenes.yaml"

#: "looking straight down" - the identity USD orientation of this rig
#: (:data:`vsbench.vs.isaac_scene.R_OPT2USD_BASE`); image-right is world +X.
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

#: The six benchmark scenes. Order is the reporting order.
EVAL_SCENES = ("hollywood", "poster2", "poster3",
               "industrial1_drill", "cluttered2", "cluttered3")


# ------------------------------------------------------------ config loading
@dataclass
class StartSpec:
    """The `starts:` block of the scenes yaml - the historical narrow sampler's numbers
    (:func:`vsbench.vs.starts.sample_narrow` owns the sampler itself)."""

    n: int = 10
    seed: int = 20260727
    translation_frac: float = 0.18
    rotation_deg: float = 12.0


@dataclass
class SceneSpec:
    name: str
    kind: str                       # 'poster' | 'multi_object'
    raw: dict = field(default_factory=dict)

    @property
    def ransac(self) -> str:
        """Geometric model the classical matchers use: planar -> homography,
        3-D clutter -> fundamental. The wrong one silently drops good matches.

        **No default.** It used to fall back to ``"none"``, which turned a scene that
        forgot to declare a model into a without-RANSAC run wearing the with-RANSAC
        label. Every scene in ``scenes.yaml`` declares it; a new one must too.
        """
        if "ransac" not in self.raw:
            raise ValueError(
                f"scene {self.name!r} declares no `ransac` model - it is a per-scene "
                "property with no safe default (homography for a planar target, "
                "fundamental for a 3-D scene, none to run without RANSAC deliberately)"
            )
        return str(self.raw["ransac"])

    @property
    def layout_file(self) -> str | None:
        return self.raw.get("layout_file")

    def object_ids(self) -> list[str]:
        return [str(o["instance_id"]) for o in (self.raw.get("objects") or [])
                if o.get("instance_id")]

    @property
    def prompt(self) -> str | None:
        """The word that names the scene's object for a language-prompted detector
        (ART-VS's detector mode, ``matcher.prompt: scene``); ``None`` when not stated."""
        p = self.raw.get("prompt")
        return None if p in (None, "") else str(p)

    def live_names(self) -> list[str]:
        """The named live objects of an instance-suite scene (``live:``), in file order."""
        return [str(k) for k in (self.raw.get("live") or {})]

    def live_item(self, name: str) -> dict:
        """One named live object, merged over the scene's ``placement`` defaults and
        the goal object's category - :func:`placed_item`."""
        live = self.raw.get("live") or {}
        if name not in live:
            raise KeyError(f"scene {self.name!r} has no live object {name!r}; "
                           f"it has {self.live_names()}")
        objects = self.raw.get("objects") or []
        category = objects[0].get("category") if objects else None
        return placed_item(self, {**({"category": category} if category else {}),
                                  **dict(live[name])})


def load_scenes(path: Path | str | None = None) -> tuple[dict[str, SceneSpec], StartSpec]:
    """Read the scene config -> ``({name: SceneSpec}, StartSpec)``."""
    doc = yaml.safe_load(Path(path or DEFAULT_SCENES_YAML).read_text())
    scenes = {name: SceneSpec(name=name, kind=body["kind"], raw=body)
              for name, body in (doc.get("scenes") or {}).items()}
    s = doc.get("starts") or {}
    return scenes, StartSpec(n=int(s.get("n", 10)), seed=int(s.get("seed", 20260727)),
                             translation_frac=float(s.get("translation_frac", 0.18)),
                             rotation_deg=float(s.get("rotation_deg", 12.0)))


# --------------------------------------------------------------- goal framing
def working_distance_for_extent(cam: CameraConfig, width_m: float, height_m: float,
                                frame_fill: float | None = None) -> float:
    """Smallest distance at which a plate **fits** the frame (``camera.yaml``)::

        d = max(w / (fill * h_coeff), h / (fill * v_coeff))
    """
    fill = float(frame_fill if frame_fill is not None else cam.frame_fill_default)
    if fill <= 0.0:
        raise ValueError(f"frame_fill must be > 0, got {fill}")
    return max(width_m / (fill * cam.h_coeff), height_m / (fill * cam.v_coeff))


def distance_to_fill_frame(cam: CameraConfig, width_m: float, height_m: float,
                           coverage: float = 1.35) -> float:
    """Distance at which a plate **covers** the frame - no backdrop visible::

        d = min(w / h_coeff, h / v_coeff) / coverage

    ``coverage > 1`` is margin, so a perturbed start still sees only poster.
    Visible backdrop supplies near-identical featureless patches, which
    nearest-neighbour matching pairs arbitrarily.
    """
    if coverage <= 0:
        raise ValueError(f"coverage must be > 0, got {coverage}")
    return min(width_m / cam.h_coeff, height_m / cam.v_coeff) / coverage


def fronto_parallel_goal_pose(distance_m: float, centre=None) -> tuple[np.ndarray, np.ndarray]:
    """Top-down goal pose for a plane in the world XY plane -> ``(cam_pos, R_opt)``."""
    c = np.zeros(3) if centre is None else np.asarray(centre, float)
    return c + np.array([0.0, 0.0, float(distance_m)]), R_LOOK_DOWN.copy()


def scene_view_goal_pose(cam: CameraConfig, bbox_min, bbox_max,
                         frame_fill: float | None = None,
                         elevation_deg: float = 90.0,
                         azimuth_deg: float = -90.0
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """Goal pose framing a whole measured scene bounding box.

    The camera sits ``d`` from ``centre_top`` (the bbox centre lifted to the bbox
    top plane), so at 90 deg elevation the footprint fits at the nearest visible
    plane. ``elevation_deg == 90`` returns :data:`R_LOOK_DOWN` **exactly** (the
    historical byte-identical path); a tilted view keeps the same distance along
    the viewing ray and picks the one azimuth whose image orientation is
    continuous with the nadir view (image-right stays world +X).

    Returns ``(cam_pos, R_opt, distance_m)``.
    """
    lo = np.asarray(bbox_min, float)
    hi = np.asarray(bbox_max, float)
    extent = hi - lo
    d = working_distance_for_extent(cam, float(extent[0]), float(extent[1]), frame_fill)
    centre = (lo + hi) * 0.5
    centre_top = np.array([centre[0], centre[1], float(hi[2])])
    e = float(elevation_deg)
    if not 20.0 <= e <= 90.0:
        raise ValueError(f"elevation_deg {e} outside [20, 90]")
    if e == 90.0:
        return centre_top + np.array([0.0, 0.0, d]), R_LOOK_DOWN.copy(), d
    er, az = np.radians(e), np.radians(float(azimuth_deg))
    cam_pos = centre_top + d * np.array([np.cos(az) * np.cos(er),
                                         np.sin(az) * np.cos(er), np.sin(er)])
    z = centre_top - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, -1.0]), z)
    x = x / np.linalg.norm(x)
    return cam_pos, np.column_stack((x, np.cross(z, x), z)), d


def poster_goal_pose(cam: CameraConfig, spec: SceneSpec) -> tuple[np.ndarray, np.ndarray, float]:
    """The goal pose of a ``poster`` scene - fully derivable offline."""
    if spec.kind != "poster":
        raise ValueError(f"{spec.name} is {spec.kind!r}, not a poster")
    goal = spec.raw.get("goal") or {}
    if goal.get("kind") != "fill_frame":
        raise ValueError(f"{spec.name}: unsupported poster goal rule {goal.get('kind')!r}")
    d = distance_to_fill_frame(cam, float(spec.raw["width_m"]), float(spec.raw["height_m"]),
                               float(goal.get("coverage", 1.35)))
    pos, R = fronto_parallel_goal_pose(d)
    return pos, R, d


# ------------------------------------------------------------ scatter + settle
def axis_angle_quat(axis, angle_rad: float) -> np.ndarray:
    """``(w, x, y, z)`` unit quaternion for a rotation about ``axis``."""
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    h = float(angle_rad) / 2.0
    return np.concatenate([[np.cos(h)], np.sin(h) * ax])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# ------------------------------------------------- object placement (2026-09-23)
#: ``orientation_pick`` (asset library) -> the one rotation that maps that native axis to
#: world +Z, as ``(axis, degrees)``. Verbatim from the GEN-VS renderer
#: (``gen-vs`` / ``genvs-sim`` ``src/data_gen/grasp_resolver.py::ORIENTATION_TO_ROT``,
#: "ported VERBATIM from render_canonical.py"), whose upright canonical pose is
#: ``Rz(yaw_pick) * R_orient(orientation_pick)``.
ORIENTATION_PICKS = {
    "posZ": ((1.0, 0.0, 0.0), 0.0), "negZ": ((1.0, 0.0, 0.0), 180.0),
    "posY": ((1.0, 0.0, 0.0), 90.0), "negY": ((1.0, 0.0, 0.0), -90.0),
    "posX": ((0.0, 1.0, 0.0), -90.0), "negX": ((0.0, 1.0, 0.0), 90.0),
}

#: ``yaw_deg: long_axis_x`` - turn the placed object by 90 deg about the vertical when,
#: after its canonical pose and its lay-down, its footprint is longer along world Y than
#: along world X by more than :data:`LONG_AXIS_MARGIN` (image-right is world +X under
#: :func:`scene_view_goal_pose`, so the object then lies along the 16:9 frame's width).
#: The margin keeps a near-square footprint from flipping on a rounding difference;
#: near-square objects (the drills) state their yaw instead.
LONG_AXIS_X = "long_axis_x"
LONG_AXIS_MARGIN = 1.02

#: The item fields that switch an object onto the placement path below. An item with
#: none of them is placed by the original arithmetic, bit for bit (tests pin it).
PLACEMENT_KEYS = ("canonical", "upright_yaw_add_deg", "size_m")
CANONICAL_MODES = ("library",)


def uses_placement(item: dict) -> bool:
    """True when an item asks for the placement path (a canonical pose, a stated size or
    the long-axis yaw) - False for every object of the scenes that predate it."""
    return (any(k in item for k in PLACEMENT_KEYS)
            or str(item.get("yaw_deg")) == LONG_AXIS_X)


def check_placement_item(item: dict) -> None:
    """Refuse an item whose placement fields cannot mean what they say."""
    canon = item.get("canonical")
    if canon is not None and canon not in CANONICAL_MODES:
        raise ValueError(f"object {item.get('instance_id')!r}: canonical is {canon!r}, "
                         f"not one of {CANONICAL_MODES}")
    yaw = item.get("yaw_deg")
    if isinstance(yaw, str) and yaw != LONG_AXIS_X:
        raise ValueError(f"object {item.get('instance_id')!r}: yaw_deg {yaw!r} is neither "
                         f"a number nor {LONG_AXIS_X!r}")
    if "size_m" in item and not float(item["size_m"]) > 0:
        raise ValueError(f"object {item.get('instance_id')!r}: size_m must be > 0, "
                         f"got {item['size_m']!r}")
    if "upright_yaw_add_deg" in item and canon != "library":
        raise ValueError(f"object {item.get('instance_id')!r}: upright_yaw_add_deg turns the "
                         "library's upright canonical pose, so it needs canonical: library")


def placed_item(spec: SceneSpec, item: dict) -> dict:
    """``item`` over the scene's ``placement:`` defaults (the instance suite states the
    orientation of a kind once, per scene), checked. A scene without ``placement``
    returns the item unchanged."""
    defaults = dict(spec.raw.get("placement") or {})
    out = {**defaults, **dict(item)}
    if defaults or uses_placement(out):
        check_placement_item(out)
    return out


def canonical_quat(entry: dict, upright_yaw_add_deg: float = 0.0) -> np.ndarray:
    """The library's UPRIGHT canonical pose, ``Rz(yaw_pick + add) * R_orient(pick)``.

    ``entry`` is the asset-library entry (``orientation_pick``, ``yaw_pick_deg``); a
    missing pick leaves that factor out. ``upright_yaw_add_deg`` turns the upright
    object about the vertical before anything else happens to it - which side of a
    lying object faces up is decided here (a drill's side profile, a bottle's label).
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    pick = entry.get("orientation_pick")
    if pick:
        if pick not in ORIENTATION_PICKS:
            raise ValueError(f"unknown orientation_pick {pick!r}")
        axis, deg = ORIENTATION_PICKS[pick]
        q = axis_angle_quat(axis, np.deg2rad(deg))
    yaw = float(entry.get("yaw_pick_deg") or 0.0) + float(upright_yaw_add_deg)
    if yaw:
        q = _quat_mul(axis_angle_quat([0.0, 0.0, 1.0], np.deg2rad(yaw)), q)
    return q


def item_quat(item: dict, entry: dict | None = None, yaw_deg: float | None = None) -> np.ndarray:
    """The orientation an item is placed with: ``Rz(yaw) * Rx(pre_rot_x) * C``.

    ``C`` is :func:`canonical_quat` when the item says ``canonical: library`` and the
    identity otherwise; ``pre_rot_x_deg`` is the lay-down about world X; the in-plane
    ``yaw`` comes last (``yaw_deg`` overrides the item's own - the caller resolves
    ``long_axis_x``). With ``C`` the identity this is the arithmetic
    :meth:`vsbench.vs.isaac_scene.IsaacScene.place_objects` always used, operation for
    operation.
    """
    if item.get("canonical") == "library":
        if entry is None:
            raise ValueError(f"object {item.get('instance_id')!r}: canonical: library needs "
                             "its asset-library entry")
        q = canonical_quat(entry, float(item.get("upright_yaw_add_deg") or 0.0))
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    pre = item.get("pre_rot_x_deg")
    if pre:
        rx = axis_angle_quat([1.0, 0.0, 0.0], np.deg2rad(float(pre)))
        q = rx if item.get("canonical") != "library" else _quat_mul(rx, q)
    yaw = item.get("yaw_deg") if yaw_deg is None else yaw_deg
    if isinstance(yaw, str):
        raise ValueError(f"yaw_deg {yaw!r} must be resolved to a number first "
                         "(long_axis_yaw)")
    if yaw:
        q = _quat_mul(axis_angle_quat([0.0, 0.0, 1.0], np.deg2rad(float(yaw))), q)
    return q


def long_axis_yaw(extent) -> float:
    """``yaw_deg: long_axis_x`` resolved from the footprint after canonical + lay-down:
    90 when the object is longer along world Y than along X (by :data:`LONG_AXIS_MARGIN`),
    else 0."""
    ext = np.asarray(extent, float).reshape(-1)
    return 90.0 if float(ext[1]) > LONG_AXIS_MARGIN * float(ext[0]) else 0.0


def drop_quat(rng: np.random.Generator, tilt_deg: float | None = None,
              tilt_max_deg: float = 15.0) -> np.ndarray:
    """Pre-drop orientation for gravity settling: random yaw plus a tilt that is
    either **exact** (the drill drops on its side at 90 deg) or uniform."""
    yaw = axis_angle_quat([0.0, 0.0, 1.0], rng.uniform(-np.pi, np.pi))
    ang = (np.deg2rad(float(tilt_deg)) if tilt_deg is not None
           else np.deg2rad(rng.uniform(0.0, float(tilt_max_deg))))
    d = rng.uniform(-np.pi, np.pi)
    q = _quat_mul(axis_angle_quat([np.cos(d), np.sin(d), 0.0], ang), yaw)
    return q / np.linalg.norm(q)


def sample_scatter(n: int, seed: int, half_extent_m: float | None = None,
                   min_dist_m: float = 0.20, max_tries: int = 4000,
                   half_extent_x_m: float | None = None,
                   half_extent_y_m: float | None = None) -> list[tuple[float, float]]:
    """Seeded ``(dx, dy)`` scatter, uniform in a rectangle with a minimum pairwise
    centre distance (deterministic rejection sampling).

    The rectangle is aspect-matched to the 16:9 goal frame, so pass distinct
    x/y half extents; ``half_extent_m`` fills both.
    """
    hx = float(half_extent_x_m if half_extent_x_m is not None else half_extent_m)
    hy = float(half_extent_y_m if half_extent_y_m is not None else half_extent_m)
    rng = np.random.default_rng(int(seed))
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < int(n):
        if tries >= max_tries:
            raise ValueError(f"sample_scatter: cannot place {n} points with "
                             f"min_dist={min_dist_m} in +/-({hx}, {hy}) after "
                             f"{max_tries} tries")
        tries += 1
        p = np.array([rng.uniform(-hx, hx), rng.uniform(-hy, hy)])
        if all(float(np.hypot(p[0] - q[0], p[1] - q[1])) >= float(min_dist_m) for q in pts):
            pts.append((float(p[0]), float(p[1])))
    return pts


# -------------------------------------------------------------- baked layouts
def load_layout(spec: SceneSpec, repo_root: Path | str = REPO_ROOT) -> dict | None:
    """The scene's frozen layout, or ``None`` when it declares no ``layout_file``."""
    rel = spec.layout_file
    if not rel:
        return None
    path = Path(repo_root) / rel
    if not path.exists():
        raise FileNotFoundError(
            f"{spec.name} declares layout_file {rel!r} but it does not exist - "
            f"bake it once with vsbench.vs.bake; a re-settle may never define a goal")
    return json.loads(path.read_text())


def check_layout(layout: dict, placed_ids) -> None:
    """Refuse an id drift between a frozen layout and the scene being built.

    The layout's objects must be exactly the placed objects, in order. A drift means the
    frozen poses would be authored onto the wrong prims, and the goal pose that
    every stored campaign was scored against would move.
    """
    ids = [str(o["instance_id"]) for o in layout.get("objects", [])]
    want = [str(i) for i in placed_ids]
    if ids != want:
        raise ValueError(f"frozen layout mismatch: layout has {ids}, scene spec has {want}")


# ------------------------------------------------------- goal capture settings
def goal_capture_settings(path: Path | str | None = None) -> tuple[int, str]:
    """``(goal_render_warmup, goal_capture_keep)`` from the scenes yaml's ``scene:`` block.

    How many whole goal captures :class:`vsbench.vs.isaac_scene.IsaacScene` takes before
    the goal frame is kept, and which of them is kept (``last`` | ``first``). The
    count has **no default in code**: the first RTX capture after a pose change still
    carries render history, and the benchmark's settle experiment found the third
    capture still differing from the second by ~0.6 grey levels, so the yaml says 8.
    A caller that renders no goal passes ``goal_render_warmup=1`` itself.
    """
    doc = yaml.safe_load(Path(path or DEFAULT_SCENES_YAML).read_text())
    block = doc.get("scene") or {}
    if "render_warmup_steps" not in block:
        raise KeyError("scenes yaml has no `scene.render_warmup_steps` - the goal "
                       "capture count has no default; state it")
    return int(block["render_warmup_steps"]), str(block.get("goal_capture_keep", "last"))
