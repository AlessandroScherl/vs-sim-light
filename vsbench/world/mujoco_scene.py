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
can be read as an RTX one (D-9), and ``meta`` (how the scene was built).

Posters only in U1; the rooms are U4 (``vsbench.world.build.scene_spec_for`` refuses them).
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

from vsbench.world import build as buildmod
from vsbench.world.camera import d435i_camera_model, d435i_intrinsics, describe

#: Isaac's ``reortho_every`` default (``IsaacScene.__init__``).
REORTHO_EVERY = 300


def render_mode_from(gl: str, gl_renderer: str | None) -> str:
    """``mujoco-llvmpipe`` for Mesa's software rasteriser, ``mujoco-<gl>-gpu`` otherwise."""
    name = (gl_renderer or "").lower()
    if "llvmpipe" in name or "softpipe" in name or "swrast" in name:
        return "mujoco-llvmpipe"
    if not name or name == "unknown":
        from vsbench.glenv import software_gl_requested

        return "mujoco-llvmpipe" if software_gl_requested() else f"mujoco-{gl}-unknown"
    return f"mujoco-{gl}-gpu"


class MujocoScene:
    """One vs-sim benchmark scene, rendered by MuJoCo, behind ``vsbench.vs.scene.Scene``.

    ``MujocoScene("hollywood")`` reads the copied ``configs/scenes.yaml``, builds the plate,
    the backdrop and the light, compiles the MuJoCo model at the D435i's 1920 x 1080, and
    derives the goal with the copied ``poster_goal_pose``. Hold it as a context manager or
    call :meth:`close`: it owns a GL context.
    """

    def __init__(self, name: str, *, gl: str = "egl", samples: int = 4, specs: dict | None = None,
                 reortho_every: int = REORTHO_EVERY, verbose: bool = False) -> None:
        from vsbench.render.base import make_renderer
        from vsbench.vs.camera import CameraConfig
        from vsbench.vs.scene import Pose
        from vsbench.vs.scenes import load_scenes, poster_goal_pose

        if specs is None:
            specs, _ = load_scenes()
        if name not in specs:
            raise KeyError(f"{name!r} is not in configs/scenes.yaml")
        self.spec = specs[name]
        self.name = name
        self.cam = CameraConfig()
        self._intr = d435i_intrinsics()
        self.render_camera = d435i_camera_model(self._intr)
        scene_spec, meta = buildmod.scene_spec_for(self.spec)
        self.scene_spec = scene_spec
        self.renderer = make_renderer(f"mujoco-{gl}", scene_spec, self.render_camera,
                                      samples=int(samples))
        self.render_mode = render_mode_from(gl, getattr(self.renderer, "gl_renderer", None))
        pos, R, d = poster_goal_pose(self.cam, self.spec)
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
        self.meta = {**meta, "render_mode": self.render_mode, "samples": int(samples),
                     "render_camera": describe(self.render_camera),
                     "renderer": getattr(self.renderer, "info", dict)(),
                     "goal_distance_m": self._distance}

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

        self.n_renders += 1
        rgb, depth = self.renderer.render(pose_to_T(pose))
        return np.ascontiguousarray(rgb), np.ascontiguousarray(depth, dtype=np.float32)

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
                                "capture_retries": [0], "render_mode": self.render_mode}
            if self._verbose:
                print(f"[mujoco_scene] {self.name}: goal captured once, md5 {md5}", flush=True)
        return self._goal_frame

    def goal_capture_record(self) -> dict:
        """What a record says about the goal frame (``IsaacScene``'s four fields) plus
        ``render_mode``, so every record flown on this world names its renderer."""
        return {"goal_render_warmup": 1, "goal_capture_keep": "last",
                "goal_settle_series": self.goal_settle, "goal_frame_md5": self.goal_frame_md5,
                "render_mode": self.render_mode}

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
