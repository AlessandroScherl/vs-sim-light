"""The servo loop: ``run_episode`` renders, asks the method for a twist, moves the camera
and logs, until the true pose error has stayed below ``stop_tol`` for ``hold_steps``:
the pose is checked at the start of every step, and the episode is ``converged`` once
``hold_steps`` consecutive checks were inside the tolerance (an episode that starts at the
target takes ``hold_steps - 1`` steps).

The harness owns the truth. A method receives images, depth only in the form its
``wants_depth`` asked for, the intrinsics and the control period; only a privileged
method (the two test instruments) gets the poses, through ``observe_truth``. Stopping on
the true pose is a test-only rule that Step 7 replaces with a real stop policy.

Row ``k`` of the returned ``EpisodeLog`` (a ``Trajectory`` with more columns) holds the
pose the camera was at, the twist the method commanded and the twist ``FreeCamera``
applied, the true translation and rotation error at that pose, the method's own
``error_norm``, the seconds its call took, a 64-bit hash of the RGB frame it saw, its
``lost`` flag and up to 500 of its matches (``uint16`` pixel coordinates times 32 with an
inlier flag; ``float16`` would be too coarse above 256 px). ``final_pose()`` is where the
camera ended. The meta string records scene, camera, renderer variant, method name and
parameters, start and target pose, the hash of the desired frame, git commit and the
outcome: ``status`` is ``converged``, ``timeout`` or ``method_error`` (an exception or a
non-finite twist, whose message is kept). Everything goes into one compressed ``.npz``
that ``python -m vsbench.replay`` re-renders and turns into a video.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vsbench.camera import CameraModel
from vsbench.methods.base import VSMethod, check_method, normalise_output
from vsbench.motion import FreeCamera, MotionLimits, check_pose
from vsbench.se3 import exp_se3, pose_error
from vsbench.trajectory import ARRAYS as TRAJECTORY_ARRAYS
from vsbench.trajectory import Trajectory

STATUSES = ("converged", "timeout", "method_error")
FORMAT = "vsbench.episode/1"
MATCH_SCALE = 32.0  # uint16 pixel coordinates times 32: 1/32 px steps up to 2047 px
MAX_MATCHES = 500  # per step
STEP_ARRAYS = ("err_t", "err_r", "error_norm", "call_s", "frame_hash", "lost", "match_count")
MATCH_ARRAYS = ("match_px", "match_inlier")


def frame_hash(rgb: np.ndarray) -> int:
    """64-bit BLAKE2b of the raw RGB bytes, as a Python int (stored as uint64)."""
    h = hashlib.blake2b(np.ascontiguousarray(rgb).tobytes(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def encode_matches(info: dict, max_matches: int = MAX_MATCHES) -> tuple[np.ndarray, np.ndarray]:
    """``(px (M, 4) uint16, inlier (M,) bool)`` from an info dict's ``kp_cur``, ``kp_des``
    and ``inliers``; columns are ``u_cur, v_cur, u_des, v_des`` times ``MATCH_SCALE``.
    More than ``max_matches`` matches are thinned evenly so inliers and outliers keep their
    mix. Missing or inconsistent keys give no matches."""
    empty = (np.zeros((0, 4), dtype=np.uint16), np.zeros(0, dtype=bool))
    kc, kd = info.get("kp_cur"), info.get("kp_des")
    if kc is None or kd is None:
        return empty
    a = np.asarray(kc, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(kd, dtype=np.float64).reshape(-1, 2)
    if len(a) != len(b) or len(a) == 0:
        return empty
    inl = info.get("inliers")
    inl = np.ones(len(a), dtype=bool) if inl is None else np.asarray(inl).reshape(-1).astype(bool)
    if len(inl) != len(a):
        inl = np.ones(len(a), dtype=bool)
    if len(a) > max_matches:
        idx = np.unique(np.rint(np.linspace(0, len(a) - 1, max_matches)).astype(int))
        a, b, inl = a[idx], b[idx], inl[idx]
    px = np.concatenate([a, b], axis=1) * MATCH_SCALE
    px = np.where(np.isfinite(px), px, 0.0)
    return np.clip(np.rint(px), 0, np.iinfo(np.uint16).max).astype(np.uint16), inl


def decode_matches(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(kp_cur (M, 2), kp_des (M, 2))`` float pixels from the stored ``uint16`` rows."""
    f = np.asarray(px, dtype=np.float64).reshape(-1, 4) / MATCH_SCALE
    return f[:, :2], f[:, 2:]


class EpisodeLog(Trajectory):
    """A ``Trajectory`` plus, per step, the true errors, the method's error norm, the call
    time, the frame hash, the lost flag and the stored matches. One compressed ``.npz``."""

    def __init__(self, dt: float, limits: MotionLimits = MotionLimits(), meta: dict | None = None):
        super().__init__(dt, limits, meta)
        self._err_t: list[float] = []
        self._err_r: list[float] = []
        self._enorm: list[float] = []
        self._call_s: list[float] = []
        self._hash: list[int] = []
        self._lost: list[bool] = []
        self._mcount: list[int] = []
        self._mpx: list[np.ndarray] = []
        self._minl: list[np.ndarray] = []

    # -- building ----------------------------------------------------------------------
    def append_step(self, t, T_wc, twist_cmd, twist_applied, err_t: float, err_r: float, call_s: float, frame_hash: int, info: dict | None = None) -> None:
        info = info or {}
        self.append(t, T_wc, twist_cmd, twist_applied)
        self._err_t.append(float(err_t))
        self._err_r.append(float(err_r))
        en = info.get("error_norm")
        self._enorm.append(float(en) if en is not None else float("nan"))
        self._call_s.append(float(call_s))
        self._hash.append(int(frame_hash))
        self._lost.append(bool(info.get("lost", False)))
        px, inl = encode_matches(info)
        self._mcount.append(len(px))
        self._mpx.append(px)
        self._minl.append(inl)

    # -- arrays ------------------------------------------------------------------------
    @property
    def err_t(self) -> np.ndarray:
        return np.array(self._err_t, dtype=np.float64)

    @property
    def err_r(self) -> np.ndarray:
        return np.array(self._err_r, dtype=np.float64)

    @property
    def error_norm(self) -> np.ndarray:
        return np.array(self._enorm, dtype=np.float64)

    @property
    def call_s(self) -> np.ndarray:
        return np.array(self._call_s, dtype=np.float64)

    @property
    def frame_hash(self) -> np.ndarray:
        return np.array(self._hash, dtype=np.uint64)

    @property
    def lost(self) -> np.ndarray:
        return np.array(self._lost, dtype=bool)

    @property
    def match_count(self) -> np.ndarray:
        return np.array(self._mcount, dtype=np.int32)

    @property
    def match_px(self) -> np.ndarray:
        return np.concatenate(self._mpx, axis=0) if self._mpx else np.zeros((0, 4), dtype=np.uint16)

    @property
    def match_inlier(self) -> np.ndarray:
        return np.concatenate(self._minl) if self._minl else np.zeros(0, dtype=bool)

    def arrays(self) -> dict[str, np.ndarray]:
        out = super().arrays()
        for k in STEP_ARRAYS + MATCH_ARRAYS:
            out[k] = getattr(self, k)
        return out

    def matches(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(kp_cur (M, 2), kp_des (M, 2), inliers (M,))`` stored for step ``k``, in pixels."""
        kp_cur, kp_des = decode_matches(self._mpx[k])
        return kp_cur, kp_des, self._minl[k].copy()

    # -- file --------------------------------------------------------------------------
    def save(self, path) -> None:
        meta = {**self.meta, "dt": self.dt, "limits": asdict(self.limits), "format": FORMAT}
        np.savez_compressed(Path(path), meta=json.dumps(meta), **self.arrays())

    @classmethod
    def load(cls, path) -> EpisodeLog:
        with np.load(Path(path)) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path} is not an episode log (format {meta.get('format')!r})")
            arr = {k: np.array(f[k]) for k in TRAJECTORY_ARRAYS + STEP_ARRAYS + MATCH_ARRAYS}
        lim = meta.pop("limits", None)
        dt = meta.pop("dt")
        meta.pop("format", None)
        out = cls(dt, MotionLimits(**lim) if lim else MotionLimits(), meta)
        starts = np.concatenate([[0], np.cumsum(arr["match_count"])])
        for k in range(len(arr["t"])):
            out.append(arr["t"][k], arr["T_wc"][k], arr["twist_cmd"][k], arr["twist_applied"][k])
            out._err_t.append(float(arr["err_t"][k]))
            out._err_r.append(float(arr["err_r"][k]))
            out._enorm.append(float(arr["error_norm"][k]))
            out._call_s.append(float(arr["call_s"][k]))
            out._hash.append(int(arr["frame_hash"][k]))
            out._lost.append(bool(arr["lost"][k]))
            a, b = int(starts[k]), int(starts[k + 1])
            out._mcount.append(b - a)
            out._mpx.append(arr["match_px"][a:b].astype(np.uint16))
            out._minl.append(arr["match_inlier"][a:b].astype(bool))
        return out

# The ten demo starts of the Step 3 brief: (translation in cm, rotation vector in degrees),
# offsets in the target camera frame; T_wc0 = T_wc_target @ exp_se3(offset). Do not change.
DEMO_STARTS: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((5, 0, 0), (0, 0, 0)),
    ((0, 5, 0), (0, 0, 0)),
    ((0, 0, -10), (0, 0, 0)),
    ((0, 0, 0), (0, 0, 15)),
    ((0, 0, 0), (0, 8, 0)),
    ((0, 0, 0), (-8, 0, 0)),
    ((8, -5, -10), (0, 0, 20)),
    ((-10, 6, 8), (5, 5, -25)),
    ((12, 8, -15), (6, -8, 40)),
    ((-15, -10, -20), (-6, 8, -60)),
)


def demo_offsets() -> np.ndarray:
    """The ten demo offsets as twists ``(10, 6)`` in metres and radians."""
    return np.array([[*(np.array(t) / 100.0), *np.deg2rad(r)] for t, r in DEMO_STARTS])


def demo_start_poses(T_wc_target: np.ndarray) -> list[np.ndarray]:
    """``T_wc_target @ exp_se3(offset)`` for each of the ten demo offsets."""
    T = np.asarray(T_wc_target, dtype=np.float64)
    return [T @ exp_se3(xi) for xi in demo_offsets()]


@dataclass(frozen=True)
class EpisodeConfig:
    dt: float = 0.04
    max_steps: int = 3000
    limits: MotionLimits = MotionLimits()
    stop_tol: tuple[float, float] | None = (0.002, float(np.deg2rad(0.2)))  # metres, radians
    hold_steps: int = 25

    def __post_init__(self) -> None:
        if not self.dt > 0.0:
            raise ValueError("dt must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.hold_steps < 1:
            raise ValueError("hold_steps must be at least 1")
        if self.stop_tol is not None:
            tol = tuple(float(x) for x in self.stop_tol)
            if len(tol) != 2 or not all(x > 0.0 for x in tol):
                raise ValueError("stop_tol must be (metres, radians), both positive, or None")
            object.__setattr__(self, "stop_tol", tol)

    def to_meta(self) -> dict:
        d = asdict(self)
        d["limits"] = asdict(self.limits)
        d["stop_tol"] = list(self.stop_tol) if self.stop_tol is not None else None
        return d


def distance_prior(depth: np.ndarray) -> float | None:
    """Median valid (positive) depth of a depth image, or ``None`` when nothing was hit."""
    d = np.asarray(depth, dtype=np.float64)
    valid = d[d > 0.0]
    return float(np.median(valid)) if valid.size else None


def git_commit(root: Path | None = None) -> str | None:
    root = root or Path(__file__).resolve().parents[1]
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def _read_only(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.setflags(write=False)
    return v


def _camera_of(renderer) -> CameraModel:
    cam = getattr(renderer, "cam", None)
    if not isinstance(cam, CameraModel):
        raise TypeError("run_episode needs a renderer with a CameraModel in its 'cam' attribute (every vsbench backend has one)")
    return cam


def _renderer_meta(renderer) -> dict:
    info = getattr(renderer, "info", None)
    try:
        d = dict(info()) if callable(info) else {}
    except Exception:
        d = {}
    d.setdefault("variant", getattr(renderer, "name", type(renderer).__name__))
    return d


def run_episode(renderer, method: VSMethod, T_wc0: np.ndarray, T_wc_target: np.ndarray, cfg: EpisodeConfig = EpisodeConfig(), scene_name: str = "poster") -> EpisodeLog:
    """Close the loop from ``T_wc0`` towards ``T_wc_target`` and return the log.

    Setup: render the target view, then ``configure``, ``reset`` and ``set_target``.
    Each step: check the stop rule on the true pose, render, ``observe_truth`` if the
    method is privileged, ``get_control_rate`` (timed), ``FreeCamera.step``, log.
    """
    check_method(method)
    cam = _camera_of(renderer)
    T_wc0 = check_pose(T_wc0)
    T_wc_target = _read_only(check_pose(T_wc_target))
    dense = method.wants_depth == "dense"

    rgb_t, depth_t = renderer.render(T_wc_target)
    prior = distance_prior(depth_t)
    method.configure(cam.K.copy(), cam.width, cam.height, cfg.dt)
    method.reset()
    method.set_target(
        _read_only(rgb_t),
        desired_depth=_read_only(depth_t) if dense else None,
        distance_prior=prior if method.wants_depth != "none" else None,
    )

    free = FreeCamera(T_wc0, dt=cfg.dt, limits=cfg.limits)
    meta = {
        "format_episode": "vsbench.episode/1",
        "scene": scene_name,
        "camera": {"K": cam.K.tolist(), "width": cam.width, "height": cam.height, "near": cam.near, "far": cam.far},
        "renderer": _renderer_meta(renderer),
        "method": method.name,
        "method_params": dict(getattr(method, "params", {}) or {}),
        "wants_depth": method.wants_depth,
        "privileged": bool(method.privileged),
        "distance_prior": prior,
        "T_wc_target": T_wc_target.tolist(),
        "T_wc0": T_wc0.tolist(),
        "target_hash": frame_hash(rgb_t),
        "config": cfg.to_meta(),
        "git_commit": git_commit(),
    }
    log = EpisodeLog(cfg.dt, cfg.limits, meta=meta)
    call_s: list[float] = []
    status = "timeout"
    error_msg = None
    streak = 0
    for _ in range(cfg.max_steps):
        e_t, e_r = pose_error(free.T_wc, T_wc_target)
        if cfg.stop_tol is not None and e_t < cfg.stop_tol[0] and e_r < cfg.stop_tol[1]:
            streak += 1
            if streak >= cfg.hold_steps:
                status = "converged"
                break
        else:
            streak = 0
        rgb, depth = renderer.render(free.T_wc)
        if method.privileged:
            method.observe_truth(free.T_wc, T_wc_target)
        t0 = time.perf_counter()
        try:
            out = method.get_control_rate(_read_only(rgb), _read_only(depth) if dense else None)
            twist, info = normalise_output(out)
        except Exception as e:  # the method's fault, recorded, never a crash of the loop
            call_s.append(time.perf_counter() - t0)
            status = "method_error"
            error_msg = f"{type(e).__name__}: {e}"
            break
        call_s.append(time.perf_counter() - t0)
        t, pose = free.t, free.T_wc
        applied = free.step(twist)
        log.append_step(t, pose, twist, applied, e_t, e_r, call_s[-1], frame_hash(rgb), info)
    e_t, e_r = pose_error(free.T_wc, T_wc_target)
    log.meta.update(
        {
            "status": status,
            "error": error_msg,
            "steps": len(log),
            "final_error_m": e_t,
            "final_error_rad": e_r,
            "call_s_mean": float(np.mean(call_s)) if call_s else None,
            "call_s_max": float(np.max(call_s)) if call_s else None,
        }
    )
    return log
