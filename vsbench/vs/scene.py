# vs-sim copy: vssim/scene.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The scene seam: what a servoing loop needs from a world, plus a world that needs no Isaac.

Purpose
-------
:class:`Scene` is the whole contract between a servoing loop and a simulator: place
the camera, render a frame, integrate a twist, and say what the intrinsics, the goal
pose and the working distance are. Isaac Sim lives behind it
(:class:`vsbench.vs.isaac_scene.IsaacScene`); :class:`MockPlaneScene` is a complete
implementation in numpy - a textured plane rendered through the *real* pinhole
model - so a whole loop (matcher, control law, stop rules, records) can be
exercised on a laptop, in seconds, with no GPU and no simulator.

The convention every pose and twist here uses
--------------------------------------------
A :class:`Pose` is the camera's **optical** frame in world coordinates: the columns
of ``rotation`` are image-right, image-down and the viewing direction. A twist is
``(v_x, v_y, v_z, w_x, w_y, w_z)`` in that same optical frame, and
:func:`integrate_optical_velocity` is the one integration rule
(``pos += R @ v_lin * dt``; ``R <- R @ Rodrigues(w_hat, |w| dt)``).

Fences
------
- **The mock is a camera model, not a renderer.** It has no shading, no
  occlusion and one plane. It is enough to close a servo loop and it must never
  be used to make a claim about a method's accuracy.
- **Ground truth flows one way.** :meth:`Scene.goal_pose`, :attr:`Scene.pose` and a
  world's ``gt_correspondence`` exist for records and evaluation. A controller that
  reads them is not visual servoing; keep them out of the control path.

Provenance
----------
Extracted from the Refine-VS research repository (``refinevs/eval/scene.py``),
whose integration rule and pose convention were ported verbatim from ``genvs-sim``
``src/eval/free_flying_scene.py``. The mock's truth *matcher* stayed behind - it
speaks that project's match format - but the mock's closed-form correspondence
(:meth:`MockPlaneScene.gt_correspondence`) is here. New here (2026-09-30):
:attr:`MockPlaneScene.live_scale`, an opt-in scaling of the rendered plate for the debug
log's scaled-truth test (off by default; the geometry verbs keep the unscaled plate).

Golden tests: ``tests/fast/test_scene.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from vsbench.vs.camera import Intrinsics

#: The fielded camera (``configs/camera.yaml``): D435i, 1920 x 1080, 42 deg vertical
#: field of view, so ``fx = fy = 1080 / (2 tan 21 deg) = 1406.75`` and the principal
#: point is the image centre. ``tests/fast/test_camera.py`` asserts this agrees with
#: :class:`vsbench.vs.camera.CameraConfig`.
D435I = Intrinsics(fx=1406.75, fy=1406.75, cx=960.0, cy=540.0, width=1920, height=1080)


def scaled_intrinsics(intr: Intrinsics, factor: float) -> Intrinsics:
    """The same camera at a coarser sampling - every quantity scales together.

    A 1920 x 1080 numpy render costs ~2 M rays per step, which the fast tier will
    not pay 300 times. Scaling ``(fx, fy, cx, cy, width, height)`` by one factor
    keeps the *model* exact (same field of view, same principal point in
    normalised coordinates) and only coarsens the sampling, so a mock episode is
    a real pinhole episode at a smaller image.
    """
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    return Intrinsics(fx=intr.fx * factor, fy=intr.fy * factor,
                      cx=intr.cx * factor, cy=intr.cy * factor,
                      width=max(1, int(round(intr.width * factor))),
                      height=max(1, int(round(intr.height * factor))))


# --------------------------------------------------------------------- poses

@dataclass(frozen=True)
class Pose:
    """A camera pose in the **optical** frame: position in metres, rotation as ``R``.

    The columns of ``rotation`` are the camera's x (image right), y (image down)
    and z (viewing direction) axes in world coordinates - the ``cam_R_opt`` of
    every frozen start file.
    """

    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.position, dtype=np.float64).reshape(3)
        r = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "rotation", r)

    @classmethod
    def from_start(cls, start: dict) -> Pose:
        """Build from a start-file entry (``cam_pos`` + ``cam_R_opt``)."""
        return cls(position=start["cam_pos"], rotation=start["cam_R_opt"])

    def as_dict(self) -> dict:
        return {"cam_pos": [float(v) for v in self.position],
                "cam_R_opt": self.rotation.tolist()}


def pose_to_T(pose: Pose) -> np.ndarray:
    """``Pose`` -> the 4x4 camera-to-world matrix :mod:`vsbench.vs.labels` wants.

    One home: the truth every ground-truth column is computed from is not a thing
    to keep two versions of.
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(pose.rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(pose.position, dtype=np.float64).reshape(3)
    return T


def rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotation matrix for ``angle_rad`` about a unit ``axis`` (numpy only)."""
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def integrate_optical_velocity(pose: Pose, twist: np.ndarray, dt: float) -> Pose:
    """One optical-frame 6-DoF integration step.

    ``pos += R @ v_lin * dt`` and ``R <- R @ Rodrigues(w_hat, |w| dt)``: the twist is
    ``(v_x, v_y, v_z, w_x, w_y, w_z)`` in the camera's **optical** frame (x right,
    y down, z forward), which is the convention an IBVS law commands in. Pure - the
    input pose is not mutated.
    """
    v = np.asarray(twist, dtype=np.float64).reshape(-1)
    lin, ang = v[:3], v[3:6]
    new_pos = pose.position + pose.rotation @ lin * float(dt)
    ang_norm = float(np.linalg.norm(ang))
    R = pose.rotation
    if ang_norm > 1e-9:
        R = R @ rodrigues(ang / ang_norm, ang_norm * float(dt))
    return Pose(position=new_pos, rotation=R)


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotations, degrees."""
    rel = np.asarray(R_a, np.float64).T @ np.asarray(R_b, np.float64)
    c = (np.trace(rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def translation_error_mm(p_a, p_b) -> float:
    """Distance in millimetres."""
    return float(np.linalg.norm(np.asarray(p_a, np.float64)
                                - np.asarray(p_b, np.float64)) * 1000.0)


def rotvec_deg(R_rel: np.ndarray) -> list[float]:
    """Axis-angle of a relative rotation in degrees.

    The angular twin of the camera-frame goal direction (a telemetry column), with
    the near-pi diagonal fallback.
    """
    R = np.asarray(R_rel, np.float64)
    c = (np.trace(R) - 1.0) / 2.0
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if ang < 1e-9:
        return [0.0, 0.0, 0.0]
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = float(np.linalg.norm(ax))
    if n < 1e-12:                       # ang ~ pi: the diagonal form
        ax = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        n = float(np.linalg.norm(ax)) or 1.0
    v = ax / n * np.degrees(ang)
    return [round(float(x), 4) for x in v]


# ------------------------------------------------------------------ protocol

@runtime_checkable
class Scene(Protocol):
    """Everything an episode needs from a world - and nothing else.

    Four verbs and four facts. An implementation may add more (both worlds here add
    :meth:`gt_correspondence`, the mock adds :meth:`project` and :meth:`backproject`);
    an episode loop needs only what is below.
    """

    #: scene name as it appears in every record and in ``episode_seed``
    name: str

    @property
    def intrinsics(self) -> Intrinsics:
        """Full-image pinhole intrinsics of the rendered frames."""
        ...

    @property
    def working_distance(self) -> float:
        """Goal-view distance to the target, metres (success gates are stated in units of it)."""
        ...

    @property
    def goal_pose(self) -> Pose:
        """The goal camera pose. Read for the record and the telemetry only."""
        ...

    @property
    def pose(self) -> Pose:
        """Where the camera is now - so a record's final pose costs no extra frame."""
        ...

    def reset(self, pose: Pose) -> None:
        """Place the camera at ``pose`` (the old ``set_camera_pose_optical``)."""
        ...

    def render(self) -> tuple[np.ndarray, np.ndarray, Pose]:
        """``(rgb uint8 [H, W, 3], depth float [H, W] metres, pose)`` at the camera."""
        ...

    def apply(self, twist, dt: float) -> None:
        """Integrate one commanded twist for ``dt`` seconds."""
        ...


# ----------------------------------------------------------------- the mock

def checker_texture(size: int = 512, *, seed: int = 20260903) -> np.ndarray:
    """A deterministic high-frequency texture: checkerboard x smooth noise.

    Matchable at any scale (the checker gives structure, the noise breaks its
    periodicity so mutual nearest neighbours are not ambiguous) and free: no
    asset, no download, identical in both environments.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    checker = (((xx // 16) + (yy // 16)) % 2).astype(np.float64)
    coarse = rng.random((size // 16 + 1, size // 16 + 1))
    noise = np.repeat(np.repeat(coarse, 16, axis=0), 16, axis=1)[:size, :size]
    grey = 40.0 + 120.0 * checker + 90.0 * noise
    rgb = np.stack([grey, np.roll(grey, 7, axis=0), np.roll(grey, 13, axis=1)], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class MockPlaneScene:
    """A textured plane at ``z = 0``, seen through a real pinhole camera.

    The world is one plane spanning ``width_m x height_m`` about the origin with
    :func:`checker_texture` on it; the goal camera looks straight down at it from
    ``working_distance`` with ``R = diag(1, -1, -1)`` - the goal pose of every
    frozen poster start file (0.8 x 0.6 m at 0.4342 m: the benchmark's posters).
    Rendering is an exact inverse warp: each pixel's ray
    is intersected with the plane and the texture is sampled bilinearly, so the
    depth map is the true per-pixel camera-frame ``z`` and the correspondence
    between two views is available in closed form. Pass the real poster texture
    (``texture=``) and this IS the benchmark's poster scene without Isaac: same
    plate, same goal, same pixel geometry - a tone curve and the shading apart.

    Args:
        intrinsics: the camera. Defaults to :data:`D435I` at ``render_scale``.
        render_scale: coarser sampling of the same camera (see
            :func:`scaled_intrinsics`); ``0.125`` renders 240 x 135 and is what
            the fast tests use.
    """

    def __init__(self, *, name: str = "mock_plane", width_m: float = 0.8,
                 height_m: float = 0.6, working_distance: float = 0.4341815107823003,
                 intrinsics: Intrinsics | None = None, render_scale: float = 1.0,
                 texture: np.ndarray | None = None, background: int = 0) -> None:
        self.name = name
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self._distance = float(working_distance)
        base = D435I if intrinsics is None else intrinsics
        self._intr = base if render_scale == 1.0 else scaled_intrinsics(base, render_scale)
        self.texture = checker_texture() if texture is None else np.asarray(texture, np.uint8)
        self.background = int(background)
        self._goal = Pose(position=[0.0, 0.0, self._distance],
                          rotation=np.diag([1.0, -1.0, -1.0]))
        self._pose = self._goal
        self.n_renders = 0
        #: ``(s, cx, cy)`` or ``None``: the plate scaled by ``s`` about ``(cx, cy, 0)`` in
        #: the RENDERED picture only (the debug log's scaled-truth test, 2026-09-30).
        #: ``gt_correspondence`` / ``backproject`` keep describing the unscaled plate, so a
        #: truth matcher is then wrong away from the centre - which is what the test wants.
        #: ``None`` (the default) is the arithmetic every other test pins.
        self.live_scale: tuple[float, float, float] | None = None

    # ------------------------------------------------------------- the facts
    @property
    def intrinsics(self) -> Intrinsics:
        return self._intr

    @property
    def working_distance(self) -> float:
        return self._distance

    @property
    def goal_pose(self) -> Pose:
        return self._goal

    @property
    def pose(self) -> Pose:
        return self._pose

    # ------------------------------------------------------------- the verbs
    def reset(self, pose: Pose) -> None:
        self._pose = pose if isinstance(pose, Pose) else Pose(*pose)

    def apply(self, twist, dt: float) -> None:
        self._pose = integrate_optical_velocity(self._pose, np.asarray(twist), dt)

    def render(self) -> tuple[np.ndarray, np.ndarray, Pose]:
        """Inverse-warp the plane into the current view; returns ``(rgb, depth, pose)``."""
        self.n_renders += 1
        intr = self._intr
        vv, uu = np.mgrid[0:intr.height, 0:intr.width].astype(np.float64)
        rays = np.stack([(uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy,
                         np.ones_like(uu)], axis=-1)
        world = rays @ self._pose.rotation.T                     # (H, W, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -self._pose.position[2] / world[..., 2]
        hit = np.isfinite(t) & (t > 0)
        t = np.where(hit, t, 0.0)
        pts = self._pose.position + t[..., None] * world
        rgb, inside = self._sample(pts[..., 0], pts[..., 1])
        ok = hit & inside
        rgb[~ok] = self.background
        depth = np.where(ok, t, 0.0)
        return rgb, depth, self._pose

    # -------------------------------------------------------------- geometry
    def _sample(self, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear texture lookup at plane coordinates; second value is the in-plane mask."""
        if self.live_scale is not None:
            # the plate scaled by s about (cx, cy): the point shown at x is the texture's
            # point c + (x - c) / s
            sc, cx, cy = (float(v) for v in self.live_scale)
            x_m = cx + (x_m - cx) / sc
            y_m = cy + (y_m - cy) / sc
        th, tw = self.texture.shape[:2]
        s = (x_m / self.width_m + 0.5) * (tw - 1)
        # Texture row 0 sits at world +Y (the plate's top edge, image-up in the goal
        # view): the USD convention `IsaacScene.build_poster` authors (uv (0, 1) at
        # (-hx, +hy)), so a picture rendered here and by Isaac read the same way up.
        # Measured against Isaac's hollywood goal render (2026-09-17): pixel-aligned
        # to the pixel, gradient correlation 0.81, mean residual 15 grey levels after
        # a tone fit - the residual is lighting, not geometry.
        r = (0.5 - y_m / self.height_m) * (th - 1)
        inside = (s >= 0) & (s <= tw - 1) & (r >= 0) & (r <= th - 1)
        s0 = np.clip(np.floor(s), 0, tw - 2).astype(np.int64)
        r0 = np.clip(np.floor(r), 0, th - 2).astype(np.int64)
        fs = np.clip(s - s0, 0.0, 1.0)[..., None]
        fr = np.clip(r - r0, 0.0, 1.0)[..., None]
        tex = self.texture.astype(np.float64)
        top = tex[r0, s0] * (1 - fs) + tex[r0, s0 + 1] * fs
        bot = tex[r0 + 1, s0] * (1 - fs) + tex[r0 + 1, s0 + 1] * fs
        return (top * (1 - fr) + bot * fr).astype(np.uint8), inside

    def backproject(self, pixels: np.ndarray, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """Pixels -> the world points they see on the plane; second value is validity."""
        px = np.asarray(pixels, np.float64).reshape(-1, 2)
        intr = self._intr
        rays = np.stack([(px[:, 0] - intr.cx) / intr.fx,
                         (px[:, 1] - intr.cy) / intr.fy, np.ones(len(px))], axis=-1)
        world = rays @ pose.rotation.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -pose.position[2] / world[:, 2]
        valid = np.isfinite(t) & (t > 0)
        pts = pose.position + np.where(valid, t, 0.0)[:, None] * world
        in_plane = (np.abs(pts[:, 0]) <= self.width_m / 2.0) & (
            np.abs(pts[:, 1]) <= self.height_m / 2.0)
        return pts, valid & in_plane

    def project(self, points_world: np.ndarray, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """World points -> pixels in ``pose``'s image; second value is validity."""
        p = np.asarray(points_world, np.float64).reshape(-1, 3)
        cam = (p - pose.position) @ pose.rotation
        intr = self._intr
        with np.errstate(divide="ignore", invalid="ignore"):
            u = intr.fx * cam[:, 0] / cam[:, 2] + intr.cx
            v = intr.fy * cam[:, 1] / cam[:, 2] + intr.cy
        uv = np.stack([u, v], axis=-1)
        valid = (cam[:, 2] > 1e-6) & np.isfinite(u) & np.isfinite(v) & (
            u >= 0) & (u <= intr.width - 1) & (v >= 0) & (v <= intr.height - 1)
        return uv, valid

    def gt_correspondence(self, pts_goal_px: np.ndarray, cur_pose: Pose,
                          goal_pose: Pose | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Where each goal pixel lands in ``cur_pose``'s image (exact, on the plane).

        The ground-truth seam: :class:`vsbench.vs.isaac_scene.IsaacScene` answers the same
        question by reprojecting the goal depth map, this one answers it in closed
        form. **For evaluation and logging only** - a controller must never see it.
        """
        goal = self._goal if goal_pose is None else goal_pose
        pts, ok_back = self.backproject(pts_goal_px, goal)
        uv, ok_fwd = self.project(pts, cur_pose)
        return uv, ok_back & ok_fwd


