"""``MujocoScene``: vs-sim's ``Scene`` contract on the MuJoCo renderer (PORT_PLAN U1).

The copied loop (``vsbench.vs.bench.runner.run_episode``) and the copied baselines fly on
this world unchanged: four verbs (``reset``, ``render``, ``apply``) and four facts
(``name``, ``intrinsics``, ``goal_pose``, ``working_distance``, ``pose``), plus the extras
the loop reads with ``getattr`` (``gt_correspondence``, ``goal_capture_record``).

What it mirrors from the reference world ``vsbench.vs.isaac_scene.IsaacScene``:

* ``intrinsics`` reports ``CameraConfig().intrinsics()`` (principal point ``(960, 540)``)
  while the frames are drawn with Isaac's half-pixel geometry (``vsbench.world.camera``);
* ``apply`` is the copied ``integrate_optical_velocity`` with Isaac's SVD
  re-orthogonalisation every ``reortho_every`` (300) steps, the counter zeroed by ``reset``;
* the goal frame is captured once and served from ``render()`` whenever the camera sits
  exactly at the goal pose; ``goal_capture_record()`` says how (one capture: MuJoCo is
  frame-deterministic, so Isaac's warm-up series has nothing to settle);
* ``gt_correspondence`` is the copied ``vsbench.vs.labels.gt_correspondence`` on the cached
  goal depth and the last rendered depth, with the reported ``K`` - Isaac's call, verbatim.

What it adds: ``render_mode`` (``mujoco-llvmpipe`` on the CPU, ``mujoco-<gl>-gpu``
otherwise), carried into every record through ``goal_capture_record`` so no MuJoCo number
can be read as an RTX one (D-9), ``meta`` (how the scene was built), and the **look**
(U4b, ``vsbench.world.look``): ``isaac_fit_v1`` by default - the fitted light, the scene-fitted
8192 shadow map and the scene's tone curve on every RGB frame (the goal frame included, depth
never) - or ``look="raw"``, the U1/U4 picture bit for bit, for the gates that compare against
the copied mock. ``goal_capture_record`` puts the look's name, curve id and md5 on every
record.

The rooms (U4): ``vsbench.world.build.build_scene`` places the converted objects by the
frozen layout; the goal is ``IsaacScene.build_eval_scene``'s - the copied
``scene_view_goal_pose`` on the box Isaac measured (``vsbench.world.assets.scene_bbox(placed,
"usd")``: Isaac's ``BBoxCache`` rule on the cached extents) - and is then checked against
every pinned start file of the scene (``check_goal_match``'s tolerances). On a miss the
owner's open decision D-4 applies, default (a): the frozen file's goal wins and the residual
is recorded (``goal_check`` on ``meta``); ``goal_rule="measured"`` keeps the derived goal
(b), ``"refuse"`` raises (c).
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

from vsbench.world import build as buildmod
from vsbench.world.camera import d435i_camera_model, d435i_intrinsics, describe
from vsbench.world.perturb_world import WorldVerbs

#: Isaac's ``reortho_every`` default (``IsaacScene.__init__``).
REORTHO_EVERY = 300
#: D-4: what a room does when its derived goal misses a pinned start file's goal
GOAL_RULES = ("frozen_on_miss", "measured", "refuse")
#: Why one goal capture where ``configs/scenes.yaml`` says 8 (stated on every record).
WARMUP_REASON = ("1 capture, not the protocol's 8: the 8 settle RTX render history; MuJoCo "
                 "frames are deterministic (two draws of one pose are the same bytes)")


def render_mode_from(gl: str, gl_renderer: str | None) -> str:
    """``mujoco-llvmpipe`` for Mesa's software rasteriser, ``mujoco-<gl>-gpu`` otherwise."""
    name = (gl_renderer or "").lower()
    if "llvmpipe" in name or "softpipe" in name or "swrast" in name:
        return "mujoco-llvmpipe"
    if not name or name == "unknown":
        from vsbench.glenv import software_gl_requested

        return "mujoco-llvmpipe" if software_gl_requested() else f"mujoco-{gl}-unknown"
    return f"mujoco-{gl}-gpu"


def _pinned_files(name: str) -> list[str]:
    from vsbench.vs import benchmark

    try:
        pins = benchmark._pins(name)
    except KeyError:
        return []
    return sorted(pins, key=lambda w: (w != "v2", w))


def room_goal(cam, spec, placed, goal_rule: str = "frozen_on_miss"):
    """A room's goal: ``build_eval_scene``'s ``scene_view_goal_pose`` on Isaac's box of the
    placed objects, checked against every pinned start file of the scene (D-4 on a miss).

    Returns ``(position, R, distance, goal_check)``; ``goal_check`` records the derived goal,
    the residual against each file, the tolerance and which goal is used.
    """
    from vsbench.vs import benchmark
    from vsbench.vs import starts as startsmod
    from vsbench.vs.scenes import scene_view_goal_pose
    from vsbench.world.assets import scene_bbox

    lo, hi = scene_bbox(placed, "usd")
    g = spec.raw.get("goal") or {}
    pos, R, d = scene_view_goal_pose(cam, lo, hi, frame_fill=g.get("frame_fill"),
                                     elevation_deg=float(g.get("elevation_deg", 90.0)),
                                     azimuth_deg=float(g.get("azimuth_deg", -90.0)))
    pos, R, d = np.asarray(pos, float), np.asarray(R, float), float(d)
    files, miss = [], None
    for which in _pinned_files(spec.name):
        payload = benchmark.load_starts(spec.name, which)      # md5-pinned
        fg = payload["goal"]
        row = {"file": which, "distance_m": float(fg["distance_m"]),
               "dpos_mm": float(np.linalg.norm(np.asarray(fg["cam_pos"], float) - pos) * 1e3),
               "drot_deg": float(startsmod.rotation_geodesic_deg(np.asarray(fg["cam_R_opt"], float), R)),
               "ddist_mm": abs(float(fg["distance_m"]) - d) * 1e3}
        try:
            startsmod.check_goal_match(payload, pos, R, d)
            row["pass"], row["refusal"] = True, None
        except ValueError as e:
            row["pass"], row["refusal"] = False, str(e)
            miss = miss or payload
        files.append(row)
    check = {"derived": {"cam_pos": pos.tolist(), "cam_R_opt": R.tolist(), "distance_m": d,
                         "bbox_min": [float(v) for v in lo], "bbox_max": [float(v) for v in hi],
                         "rule": "scene_view_goal_pose(cam, scene_bbox(placed, 'usd'), goal block)"},
             "files": files, "tolerance": {"pos_m": 5e-4, "rot_deg": 0.05, "dist_m": 1e-3},
             "goal_rule": goal_rule, "all_pass": miss is None, "used": "derived"}
    if miss is not None:
        if goal_rule == "refuse":
            raise ValueError(f"{spec.name}: the derived goal misses a pinned start file "
                             f"(D-4 (c)): {[f for f in files if not f['pass']]}")
        if goal_rule == "frozen_on_miss":
            fg = miss["goal"]
            pos, R, d = (np.asarray(fg["cam_pos"], float), np.asarray(fg["cam_R_opt"], float),
                         float(fg["distance_m"]))
            check["used"] = "frozen"
    check["d4"] = ("D-4 (a) default: the frozen goal wins on a miss, the residual is recorded"
                   if goal_rule == "frozen_on_miss" else f"D-4 rule {goal_rule!r}")
    return pos, R, d, check


class MujocoScene(WorldVerbs):
    """One vs-sim benchmark scene, rendered by MuJoCo, behind ``vsbench.vs.scene.Scene``.

    ``MujocoScene("hollywood")`` reads the copied ``configs/scenes.yaml``, builds the plate,
    the backdrop and the light, compiles the MuJoCo model at the D435i's 1920 x 1080, and
    derives the goal with the copied ``poster_goal_pose``. ``MujocoScene("cluttered2")``
    builds the ground and the converted objects (loaded from the cache for this scene only)
    and derives the goal from their box (:func:`room_goal`). Hold it as a context manager or
    call :meth:`close`: it owns a GL context.

    The perturbation verbs (``build_poster``, ``put_box``, ``remove_prim(s)``,
    ``place_extra_objects``, ``set_prim_visible``, ``object_pose``, ``object_world_bbox``,
    ``set_lighting`` / ``lighting_snapshot`` / ``restore_lighting``; PORT_PLAN U8) come from
    :class:`vsbench.world.perturb_world.WorldVerbs`: an edit recompiles the model at the
    next frame, the look stays the unedited scene's, the goal picture stays the unedited one.
    """

    def __init__(self, name: str, *, gl: str = "egl", samples: int = 4, specs: dict | None = None,
                 reortho_every: int = REORTHO_EVERY, verbose: bool = False,
                 goal_rule: str = "frozen_on_miss", cache=None, look=None) -> None:
        import time

        from vsbench.render.base import make_renderer
        from vsbench.vs.camera import CameraConfig
        from vsbench.vs.scene import Pose
        from vsbench.vs.scenes import load_scenes, poster_goal_pose
        from vsbench.world.look import resolve_look

        if goal_rule not in GOAL_RULES:
            raise ValueError(f"goal_rule {goal_rule!r} not in {GOAL_RULES}")
        if specs is None:
            specs, _ = load_scenes()
        if name not in specs:
            raise KeyError(f"{name!r} is not in configs/scenes.yaml")
        self.spec = specs[name]
        self.name = name
        self.cam = CameraConfig()
        self._intr = d435i_intrinsics()
        self.render_camera = d435i_camera_model(self._intr)
        kw = {} if self.spec.kind == "poster" else {"cache": cache}
        built = buildmod.build_scene(self.spec, **kw)
        meta = built.meta
        self.placed = built.placed
        self.look = resolve_look(look, name)
        self.built_spec = built.scene_spec
        self.scene_spec = self.look.scene_spec(built.scene_spec)
        self.renderer_kwargs = self.look.renderer_kwargs(self.scene_spec)
        t_compile = time.monotonic()
        self.renderer = make_renderer(f"mujoco-{gl}", self.scene_spec, self.render_camera,
                                      samples=int(samples), **self.renderer_kwargs)
        self.compile_s = round(time.monotonic() - t_compile, 3)
        self._gl, self._samples, self._cache = gl, int(samples), cache
        st = self.renderer.model.stat           # pinned for every edited world (perturb_world)
        self._stat = (float(st.extent), tuple(float(c) for c in st.center))
        self.render_mode = render_mode_from(gl, getattr(self.renderer, "gl_renderer", None))
        self.goal_check: dict | None = None
        if self.spec.kind == "poster":
            pos, R, d = poster_goal_pose(self.cam, self.spec)
        else:
            pos, R, d, self.goal_check = room_goal(self.cam, self.spec, self.placed, goal_rule)
        self._goal = Pose(position=pos, rotation=R)
        self._distance = float(d)
        self._pose = self._goal
        self.reortho_every = int(reortho_every)
        self._reortho_counter = 0
        self._goal_frame: tuple[np.ndarray, np.ndarray] | None = None
        self._last = None
        self.goal_settle: dict | None = None
        self.goal_frame_md5: str | None = None
        self.n_renders = 0
        self._verbose = bool(verbose)
        self._init_world(built)
        self.look_record = self.look.record(self.scene_spec, self.renderer_kwargs)
        self.look_record["gl_renderer"] = getattr(self.renderer, "gl_renderer", None)
        self.meta = {**meta, "render_mode": self.render_mode, "samples": int(samples),
                     "look": self.look_record,
                     "render_camera": describe(self.render_camera),
                     "renderer": getattr(self.renderer, "info", dict)(),
                     "goal_distance_m": self._distance, "compile_s": self.compile_s}
        if self.goal_check is not None:
            self.meta["goal_check"] = self.goal_check

    # ------------------------------------------------------------ the facts
    @property
    def intrinsics(self):
        return self._intr

    @property
    def working_distance(self) -> float:
        return self._distance

    @property
    def goal_pose(self):
        return self._goal

    @property
    def pose(self):
        return self._pose

    # ------------------------------------------------------------ the verbs
    def reset(self, pose) -> None:
        from vsbench.vs.scene import Pose

        self._pose = pose if isinstance(pose, Pose) else Pose(*pose)
        self._reortho_counter = 0

    def apply(self, twist, dt: float) -> None:
        from vsbench.vs.isaac_scene import reorthogonalize
        from vsbench.vs.scene import Pose, integrate_optical_velocity

        self._pose = integrate_optical_velocity(self._pose, np.asarray(twist), dt)
        self._reortho_counter += 1
        if self._reortho_counter >= self.reortho_every:
            self._pose = Pose(position=self._pose.position,
                              rotation=reorthogonalize(self._pose.rotation))
            self._reortho_counter = 0

    def render(self):
        """``(rgb, depth, pose)`` at the camera; the cached goal frame at the goal pose."""
        if self._at_goal():
            rgb, depth = (a.copy() for a in self.render_goal())   # the cache stays read-only
        else:
            rgb, depth = self._draw(self._pose)
        self._last = (self._pose, rgb, depth)
        return rgb, depth, self._pose

    # ------------------------------------------------------------ the goal
    def _at_goal(self) -> bool:
        return (np.array_equal(self._pose.position, self._goal.position)
                and np.array_equal(self._pose.rotation, self._goal.rotation))

    def _draw(self, pose) -> tuple[np.ndarray, np.ndarray]:
        from vsbench.vs.scene import pose_to_T

        self.sync()                                          # an edited world recompiles here
        self.n_renders += 1
        rgb, depth = self.renderer.render(pose_to_T(pose))
        rgb = self._tone(np.ascontiguousarray(rgb))          # the tone curve: RGB only
        return np.ascontiguousarray(rgb), np.ascontiguousarray(depth, dtype=np.float32)

    def draw_fresh(self, pose) -> tuple[np.ndarray, np.ndarray]:
        """The CURRENT world drawn at ``pose``, never from the goal cache, the camera left
        where it is - for diagnostics (e.g. the perturbed world seen from the goal pose)."""
        return self._draw(pose)

    def render_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """The goal view, drawn once per scene and cached (the camera does not move)."""
        if self._goal_frame is None:
            rgb, depth = self._draw(self._goal)
            rgb.setflags(write=False)
            depth.setflags(write=False)
            self._goal_frame = (rgb, depth)
            md5 = hashlib.md5(rgb.tobytes()).hexdigest()
            self.goal_frame_md5 = md5
            self.goal_settle = {"n_captures": 1, "keep": "last", "kept_index": 0,
                                "rgb_mean": [round(float(rgb.astype(np.float64).mean()), 6)],
                                "diff_prev": [None], "rgb_md5": [md5],
                                "depth_md5": [hashlib.md5(depth.tobytes()).hexdigest()],
                                "capture_retries": [0], "render_mode": self.render_mode,
                                "look": self.look.name}
            if self._verbose:
                print(f"[mujoco_scene] {self.name}: goal captured once, md5 {md5}", flush=True)
        return self._goal_frame

    def goal_capture_record(self) -> dict:
        """What a record says about the goal frame (``IsaacScene``'s four fields) plus
        ``render_mode`` and the look (name, curve id, curve md5, the whole look record), so
        every record flown on this world names its renderer and its look."""
        from vsbench.vs.scenes import goal_capture_settings

        protocol_warmup, _keep = goal_capture_settings()
        return {"goal_render_warmup": 1,
                "goal_render_warmup_protocol": int(protocol_warmup),
                "goal_render_warmup_reason": WARMUP_REASON,
                "goal_capture_keep": "last",
                "goal_settle_series": self.goal_settle, "goal_frame_md5": self.goal_frame_md5,
                "render_mode": self.render_mode, "look": self.look.name,
                "look_curve_id": self.look.curve_id, "look_curve_md5": self.look.curve_md5,
                "look_record": self.look_record}

    # ------------------------------------------------------------ the truth
    def gt_correspondence(self, pts_goal_px, cur_pose, goal_pose=None):
        """Where each goal pixel lands in ``cur_pose``'s image, occlusion-tested - for
        records and telemetry only, never for a controller (``IsaacScene``'s rule)."""
        from vsbench.vs.labels import gt_correspondence
        from vsbench.vs.scene import pose_to_T

        goal = self._goal if goal_pose is None else goal_pose
        if self._last is None:
            raise RuntimeError("gt_correspondence needs a rendered current frame first")
        last_pose, _rgb, depth_c = self._last
        if not (np.allclose(last_pose.position, cur_pose.position)
                and np.allclose(last_pose.rotation, cur_pose.rotation)):
            raise ValueError("gt_correspondence was asked about a pose that is not the last "
                             "rendered frame, so no depth map can occlusion-test it; render first")
        _rgb_g, depth_g = self.render_goal()
        uv, valid, _z = gt_correspondence(pts_goal_px, depth_g, self._intr.K, pose_to_T(goal),
                                          pose_to_T(cur_pose), depth_b=depth_c)
        return uv, valid

    # ------------------------------------------------------------ lifetime
    def close(self) -> None:
        r, self.renderer = getattr(self, "renderer", None), None
        if r is not None:
            r.close()

    def __enter__(self) -> MujocoScene:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        sys.stdout.flush()
        sys.stderr.flush()
        self.close()
        return False
