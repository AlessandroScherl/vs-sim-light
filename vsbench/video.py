"""The debug video of an episode, built from the log and the re-rendered frames.

960 x 720, every second logged frame, H.264 through ``imageio-ffmpeg`` with OpenCV
``mp4v`` as the fallback. Four panels:

* top left: the current image with the stored matches, inliers green, outliers red;
* top right: the desired image with the same matches, and a line from every n-th match
  across the two images; a method that reports no keypoints gets the absolute difference
  image instead;
* bottom left: a 3D inset drawn with OpenCV from a fixed third-person view: the scene
  footprint, the target frustum (blue), the current frustum (orange), the path so far, one
  arrow for the applied linear velocity (green) and one for the angular velocity (magenta);
* bottom right: translation error, rotation error and the twist norms over time with a
  cursor at the current step.

A header line carries step, time, true errors, twist norms, inlier count, status and
whether the re-rendered frame matched the logged hash. ``match_sheet`` writes a
full-resolution still of the matching at a few steps for a human to look at.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vsbench.camera import CameraModel
from vsbench.episode import EpisodeLog
from vsbench.poses import look_at
from vsbench.reference import project
from vsbench.replay import ReplayResult, camera_of, iter_frames, logged_backend, render_target
from vsbench.stdscenes import MARKER_POINTS, POSTER_SIZE

VIDEO_SIZE = (960, 720)  # width, height
PANEL = (480, 360)
EVERY = 2  # every second logged frame
LINE_EVERY = 8  # one match line per this many stored matches
GREEN = (40, 220, 40)
RED = (230, 50, 50)
BLUE = (70, 130, 255)
ORANGE = (255, 150, 30)
MAGENTA = (230, 60, 230)
WHITE = (240, 240, 240)
GREY = (120, 120, 120)
DARK = (28, 28, 32)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# -- top row ----------------------------------------------------------------------------
def draw_matches(rgb_cur: np.ndarray, rgb_des: np.ndarray, kp_cur: np.ndarray, kp_des: np.ndarray, inliers: np.ndarray, size=PANEL, line_every: int = LINE_EVERY) -> np.ndarray:
    """Current | desired side by side at ``size`` each, dots on every match and a line from
    every ``line_every``-th one; inliers green, outliers red."""
    w, h = size
    sx, sy = w / rgb_cur.shape[1], h / rgb_cur.shape[0]
    a = cv2.resize(np.ascontiguousarray(rgb_cur), (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(np.ascontiguousarray(rgb_des), (w, h), interpolation=cv2.INTER_AREA)
    out = np.concatenate([a, b], axis=1)
    if len(kp_cur):
        pc = np.rint(kp_cur * [sx, sy]).astype(int)
        pd = np.rint(kp_des * [sx, sy]).astype(int) + [w, 0]
        order = np.argsort(~inliers)  # outliers drawn last, so they stay visible
        step = line_every if len(kp_cur) > 5 * line_every else 1  # few matches: every line
        for j, i in enumerate(order):
            col = GREEN if inliers[i] else RED
            if not inliers[i] or j % step == 0:
                cv2.line(out, tuple(pc[i]), tuple(pd[i]), col, 1, cv2.LINE_AA)
        for i in range(len(pc)):
            col = GREEN if inliers[i] else RED
            cv2.circle(out, tuple(pc[i]), 2, col, -1, cv2.LINE_AA)
            cv2.circle(out, tuple(pd[i]), 2, col, -1, cv2.LINE_AA)
    cv2.line(out, (w, 0), (w, h), WHITE, 1)
    return out


def draw_difference(rgb_cur: np.ndarray, rgb_des: np.ndarray, size=PANEL) -> np.ndarray:
    """Current | absolute difference to the desired image, for methods without keypoints."""
    w, h = size
    a = cv2.resize(np.ascontiguousarray(rgb_cur), (w, h), interpolation=cv2.INTER_AREA)
    diff = np.abs(rgb_cur.astype(np.int16) - rgb_des.astype(np.int16)).astype(np.uint8)
    b = cv2.resize(diff, (w, h), interpolation=cv2.INTER_AREA)
    out = np.concatenate([a, b], axis=1)
    cv2.line(out, (w, 0), (w, h), WHITE, 1)
    return out


# -- 3D inset -----------------------------------------------------------------------------
def inset_camera(size=PANEL) -> CameraModel:
    return CameraModel.from_focal(size[0], size[1], fx=1.1 * size[0])


VIEW_DIRECTION = np.array([1.3, -1.5, 1.0]) / np.linalg.norm([1.3, -1.5, 1.0])  # beside, behind, above


def inset_view(log: EpisodeLog, T_wc_target: np.ndarray) -> np.ndarray:
    """One fixed third-person viewpoint for the whole episode: it looks at the centroid of
    the scene footprint, the target camera and every logged camera position, from far
    enough along ``VIEW_DIRECTION`` that all of them fit."""
    pts = [T_wc_target[:3, 3][None]] + [line for line in scene_footprint(log.meta.get("scene", "poster"))]
    if len(log):
        pts.append(log.T_wc[:, :3, 3])
    P = np.concatenate(pts, axis=0)
    mid = P.mean(axis=0)
    radius = max(float(np.linalg.norm(P - mid, axis=1).max()), 0.3)
    eye = mid + 2.6 * radius * VIEW_DIRECTION
    return look_at(eye, mid, up=(0.0, 0.0, 1.0))


def frustum_points(cam: CameraModel, T_wc: np.ndarray, depth: float) -> np.ndarray:
    """Camera origin plus the four image corners at ``depth`` metres, in world coordinates (5, 3)."""
    corners = np.array([[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]], dtype=np.float64)
    X = np.stack([(corners[:, 0] - cam.cx) / cam.fx * depth, (corners[:, 1] - cam.cy) / cam.fy * depth, np.full(4, depth)], axis=1)
    Xw = X @ T_wc[:3, :3].T + T_wc[:3, 3]
    return np.concatenate([T_wc[:3, 3][None], Xw], axis=0)


def _px(view_cam: CameraModel, T_view: np.ndarray, X_w: np.ndarray) -> np.ndarray | None:
    p, z = project(view_cam, T_view, X_w)
    if np.any(z <= 0.05) or not np.all(np.isfinite(p)):
        return None
    return np.rint(p).astype(int)


def _polyline(img, pts, colour, thickness=1, closed=False):
    if pts is None or len(pts) < 2:
        return
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], closed, colour, thickness, cv2.LINE_AA)


def draw_frustum(img, view_cam, T_view, cam, T_wc, colour, depth=0.15, thickness=1):
    P = _px(view_cam, T_view, frustum_points(cam, T_wc, depth))
    if P is None:
        return
    for i in range(1, 5):
        cv2.line(img, tuple(P[0]), tuple(P[i]), colour, thickness, cv2.LINE_AA)
    _polyline(img, P[1:], colour, thickness, closed=True)
    cv2.line(img, tuple(P[1]), tuple(P[2]), colour, thickness + 1, cv2.LINE_AA)  # top edge thicker: image-up


def scene_footprint(scene_name: str) -> list[np.ndarray]:
    """World polylines that show where the scene is: the poster rectangle or the marker grid."""
    if scene_name == "marker":
        g = MARKER_POINTS[:9]
        return [np.array([g[0], g[2], g[8], g[6], g[0]])] + [np.array([p, p]) for p in MARKER_POINTS[9:]]
    w, h = POSTER_SIZE
    return [np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0], [-w / 2, -h / 2, 0]])]


def draw_inset(log: EpisodeLog, k: int, cam: CameraModel, T_wc_target: np.ndarray, size=PANEL) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), DARK, dtype=np.uint8)
    view_cam = inset_camera(size)
    T_view = inset_view(log, T_wc_target)
    for line in scene_footprint(log.meta.get("scene", "poster")):
        P = _px(view_cam, T_view, line)
        if P is not None:
            if len(line) == 2:
                cv2.circle(img, tuple(P[0]), 2, WHITE, -1, cv2.LINE_AA)
            else:
                _polyline(img, P, WHITE, 1)
    # world axes at the origin
    ax = _px(view_cam, T_view, np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np.float64))
    if ax is not None:
        for i, col in zip(range(1, 4), ((220, 80, 80), (80, 200, 80), (90, 120, 240))):
            cv2.line(img, tuple(ax[0]), tuple(ax[i]), col, 1, cv2.LINE_AA)
    poses = log.T_wc
    path = poses[: k + 1, :3, 3]
    P = _px(view_cam, T_view, path)
    _polyline(img, P, GREY, 1)
    draw_frustum(img, view_cam, T_view, cam, T_wc_target, BLUE)
    T_k = poses[k]
    draw_frustum(img, view_cam, T_view, cam, T_k, ORANGE, thickness=2)
    xi = log.twist_applied[k]
    R, c = T_k[:3, :3], T_k[:3, 3]
    for vec, col, scale in ((xi[:3], GREEN, 1.0), (xi[3:], MAGENTA, 0.25)):
        if np.linalg.norm(vec) > 1e-9:
            tip = c + R @ vec * scale
            A = _px(view_cam, T_view, np.stack([c, tip]))
            if A is not None:
                cv2.arrowedLine(img, tuple(A[0]), tuple(A[1]), col, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(img, "3D view, fixed third-person camera", (8, 16), FONT, 0.4, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "target", (8, 34), FONT, 0.4, BLUE, 1, cv2.LINE_AA)
    cv2.putText(img, "current", (70, 34), FONT, 0.4, ORANGE, 1, cv2.LINE_AA)
    cv2.putText(img, "v (1 s)", (140, 34), FONT, 0.4, GREEN, 1, cv2.LINE_AA)
    cv2.putText(img, "w (0.25 s)", (200, 34), FONT, 0.4, MAGENTA, 1, cv2.LINE_AA)
    cv2.putText(img, "path", (285, 34), FONT, 0.4, GREY, 1, cv2.LINE_AA)
    return img


# -- plots --------------------------------------------------------------------------------
def _plot(img, x0, y0, w, h, t, series, colours, labels, title, k):
    """One small line chart with autoscaled y, a cursor at index k and a value readout."""
    cv2.rectangle(img, (x0, y0), (x0 + w - 1, y0 + h - 1), (60, 60, 66), 1)
    ymax = max(1e-12, max(float(np.nanmax(s)) for s in series))
    n = len(t)
    xs = x0 + 4 + (w - 8) * (np.arange(n) / max(n - 1, 1))
    for s, col in zip(series, colours):
        ys = y0 + h - 4 - (h - 22) * np.clip(np.nan_to_num(s) / ymax, 0.0, 1.0)
        pts = np.stack([xs, ys], axis=1).astype(int)
        _polyline(img, pts, col, 1)
    cx = int(xs[min(k, n - 1)])
    cv2.line(img, (cx, y0 + 16), (cx, y0 + h - 4), WHITE, 1)
    cv2.putText(img, title, (x0 + 6, y0 + 12), FONT, 0.38, WHITE, 1, cv2.LINE_AA)
    x = x0 + w - 6
    for s, col, lab in reversed(list(zip(series, colours, labels))):
        txt = f"{lab} {float(s[min(k, n - 1)]):.3g}"
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.38, 1)
        x -= tw + 10
        cv2.putText(img, txt, (x, y0 + 12), FONT, 0.38, col, 1, cv2.LINE_AA)


def draw_plots(log: EpisodeLog, k: int, size=PANEL) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), DARK, dtype=np.uint8)
    t = log.t
    n = len(t)
    if n == 0:
        return img
    ph = h // 3
    _plot(img, 0, 0, w, ph, t, [log.err_t * 1e3], [ORANGE], ["mm"], "translation error, mm", k)
    _plot(img, 0, ph, w, ph, t, [np.rad2deg(log.err_r)], [BLUE], ["deg"], "rotation error, deg", k)
    v = np.linalg.norm(log.twist_applied[:, :3], axis=1)
    om = np.linalg.norm(log.twist_applied[:, 3:], axis=1)
    _plot(img, 0, 2 * ph, w, h - 2 * ph, t, [v, om], [GREEN, MAGENTA], ["|v| m/s", "|w| rad/s"], "applied twist norms", k)
    cv2.putText(img, f"t = {t[min(k, n - 1)]:.2f} s of {t[-1] + log.dt:.2f} s", (w - 150, h - 8), FONT, 0.38, GREY, 1, cv2.LINE_AA)
    return img


# -- one frame ----------------------------------------------------------------------------
def compose_frame(log: EpisodeLog, k: int, rgb_cur: np.ndarray, rgb_des: np.ndarray, cam: CameraModel, T_wc_target: np.ndarray, identical: bool = True) -> np.ndarray:
    kp_cur, kp_des, inl = log.matches(k)
    if len(kp_cur):
        top = draw_matches(rgb_cur, rgb_des, kp_cur, kp_des, inl)
        right_title = "desired image, matches: inliers green, outliers red"
    else:
        top = draw_difference(rgb_cur, rgb_des)
        right_title = "|current - desired|"
    bottom = np.concatenate([draw_inset(log, k, cam, T_wc_target), draw_plots(log, k)], axis=1)
    frame = np.concatenate([top, bottom], axis=0)
    W, H = VIDEO_SIZE
    assert frame.shape == (H, W, 3), frame.shape
    m = log.meta
    e_t, e_r = log.err_t[k] * 1e3, np.rad2deg(log.err_r[k])
    xi = log.twist_applied[k]
    n_in = int(inl.sum()) if len(inl) else 0
    head = f"{m.get('method', '?')}  step {k}/{len(log)}  err {e_t:.1f} mm {e_r:.2f} deg  |v| {np.linalg.norm(xi[:3]):.3f} m/s  |w| {np.linalg.norm(xi[3:]):.3f} rad/s"
    if len(kp_cur):
        head += f"  inliers {n_in} of {len(inl)} stored matches"
    if log.lost[k]:
        head += "  LOST"
    tail = f"{m.get('scene', '?')} on {m.get('renderer', {}).get('variant', '?')}  status {m.get('status', '?')}" + ("" if identical else "  REPLAY NOT IDENTICAL")
    cv2.rectangle(frame, (0, 0), (W, 18), (0, 0, 0), -1)
    cv2.putText(frame, head, (6, 13), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "current image", (6, PANEL[1] - 6), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, right_title, (PANEL[0] + 6, PANEL[1] - 6), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(tail, FONT, 0.42, 1)
    cv2.putText(frame, tail, (W - tw - 6, 13), FONT, 0.42, RED if not identical else GREY, 1, cv2.LINE_AA)
    return frame


# -- encoders -----------------------------------------------------------------------------
class _FfmpegWriter:
    def __init__(self, path: Path, fps: float, size=VIDEO_SIZE):
        import imageio_ffmpeg

        self._gen = imageio_ffmpeg.write_frames(str(path), size=size, fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", output_params=["-crf", "26", "-preset", "medium"], macro_block_size=8)
        self._gen.send(None)
        self.name = "libx264 via imageio-ffmpeg"

    def write(self, rgb: np.ndarray) -> None:
        self._gen.send(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> None:
        self._gen.close()


class _OpenCVWriter:
    def __init__(self, path: Path, fps: float, size=VIDEO_SIZE):
        self._w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise RuntimeError("OpenCV could not open an mp4v writer")
        self.name = "mp4v via OpenCV"

    def write(self, rgb: np.ndarray) -> None:
        self._w.write(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._w.release()


def open_writer(path, fps: float, size=VIDEO_SIZE):
    """H.264 through imageio-ffmpeg when it imports, OpenCV mp4v otherwise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _FfmpegWriter(path, fps, size)
    except Exception:
        return _OpenCVWriter(path, fps, size)


# -- the video ----------------------------------------------------------------------------
def write_video(log: EpisodeLog, renderer, path, every: int = EVERY) -> ReplayResult:
    """Re-render the episode, write every ``every``-th frame as a 960 x 720 panel video at
    real-time speed, and return the replay check over all frames."""
    if len(log) == 0:
        raise ValueError("empty episode log")
    cam = camera_of(log)
    T_t = np.array(log.meta["T_wc_target"])
    rgb_des, _, target_ok = render_target(log, renderer)
    res = ReplayResult(len(log), getattr(renderer, "name", "?"), logged_backend(log), target_ok)
    fps = 1.0 / (log.dt * every)
    w = open_writer(path, fps)
    try:
        for k, rgb, _, ok in iter_frames(log, renderer):
            if not ok:
                res.mismatches.append(k)
            if k % every == 0:
                w.write(compose_frame(log, k, rgb, rgb_des, cam, T_t, identical=ok and target_ok))
    finally:
        w.close()
    res.encoder = w.name  # type: ignore[attr-defined]
    return res


def match_sheet(log: EpisodeLog, renderer, path, steps: tuple[int, ...] | None = None) -> None:
    """A still: current | desired with all stored matches at a few steps, full resolution,
    one row per step, so a human can judge the matching at a glance."""
    cam = camera_of(log)
    if steps is None:
        n = len(log)
        steps = tuple(sorted({0, n // 4, n // 2, n - 1}))
    rgb_des, _, _ = render_target(log, renderer)
    rows = []
    for k in steps:
        rgb, _ = renderer.render(log.T_wc[k])
        kp_cur, kp_des, inl = log.matches(k)
        if len(kp_cur):
            row = draw_matches(rgb, rgb_des, kp_cur, kp_des, inl, size=(cam.width, cam.height), line_every=4)
            txt = f"step {k}: {int(inl.sum())} inliers of {len(inl)} stored matches, err {log.err_t[k] * 1e3:.1f} mm {np.rad2deg(log.err_r[k]):.2f} deg"
        else:
            row = draw_difference(rgb, rgb_des, size=(cam.width, cam.height))
            txt = f"step {k}: no keypoints, |current - desired| shown, err {log.err_t[k] * 1e3:.1f} mm {np.rad2deg(log.err_r[k]):.2f} deg"
        cv2.rectangle(row, (0, 0), (row.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(row, txt, (8, 16), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(row, "current", (8, row.shape[0] - 8), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(row, "desired", (cam.width + 8, row.shape[0] - 8), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
