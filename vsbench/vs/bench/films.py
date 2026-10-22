# vs-sim copy: vssim/bench/films.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The match films: one frame per control step, encoded to a browser-playable video.

Purpose
-------
"See the failures with the matching process": the goal view with its target points
on the left, the live view with the matched points and an arrow from each match to
where it should be on the right, a caption with the step, the match count, the
feature error and the true pose error on top. One JPEG per step during the episode
(:class:`FilmWriter`, driven from outside the loop - nothing here is on a control
path), then :func:`encode_episode` turns the directory into ``episode.mp4``.

Fences
------
- **H.264 via imageio-ffmpeg, never cv2's ``mp4v``.** cv2's writer produces files a
  browser cannot play; ``imageio-ffmpeg`` ships a static ffmpeg with libx264. It is a
  visualisation-only dependency: absent, the encoder falls back to cv2 and says so.
- **Frames are BGR** inside, because cv2 is; :class:`FilmWriter` takes the scene's
  RGB and converts once at the boundary.
- **The films are additive**: :class:`FilmWriter` is handed a copy of what the loop
  already computed, after the twist exists.
- **The debug additions are optional (2026-09-30)**: driven by
  :class:`vsbench.vs.bench.debuglog.DebugLog`, a frame gets a caption of several lines (the
  method's name, the distance to the three references), a yellow border and a hold on
  the frame where a stop fired (the "landing"), and one more frame after the last move;
  :class:`CleanFilmWriter` writes the un-drawn half-size live picture for the offline
  viewer. Called without the new keywords, every function here does what it did.
- **In a debug film the video frames and ``frames_meta.json`` do not line up 1:1**: a held
  frame is shown for several video frames. Each held entry carries ``hold_s`` (its time on
  screen), so the video frame of an entry is the sum of the earlier entries' frames
  (``round(hold_s * FPS)`` for a held one, 1 otherwise). The clean film
  (``clean.mp4`` + ``clean_meta.json``) has no holds: one video frame per entry.

Provenance
----------
The composition and the writer of the Refine-VS research repository
(``refinevs/eval/films.py``), ported there from ``genvs-sim``'s
``baseline_check.compose_match_frame`` / ``n2_film_assemble.py``; the study's
success/failure pick and its HTML page stayed behind.

Golden tests: ``tests/fast/test_films.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

#: Playback rate of an assembled episode video.
FPS = 10

#: The composition's colours, BGR: goal targets green, matched points red, the arrow
#: from each match to its target orange.
GOAL_BGR = (80, 200, 80)
CUR_BGR = (0, 0, 230)
ARROW_BGR = (60, 160, 255)
#: The landing border (a stop fired on this frame), BGR yellow.
LANDING_BGR = (0, 230, 255)
#: How long a landing frame and the final frame are held in the film, seconds.
LANDING_HOLD_S = 1.0
FINAL_HOLD_S = 2.0


def ffmpeg_bin() -> str | None:
    """A real H.264 encoder, or ``None``."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - optional, viz only

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def compose_match_frame(goal_bgr, cur_bgr, pts_desired, pts_current,
                        caption: str, scale: float = 0.5, *, lines: list[str] | None = None,
                        border_bgr: tuple | None = None) -> np.ndarray:
    """One match-visualisation frame: goal | current, points, arrows, a caption bar.

    ``lines`` (debug films): a caption of several lines instead of ``caption``, 26 px a
    line; ``border_bgr``: a 6-px frame around the live half (the landing mark). Without
    them: exactly the single-line bar of old."""
    import cv2  # noqa: PLC0415

    def _half(img):
        return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA)

    g, c = _half(goal_bgr), _half(cur_bgr)
    pd = np.asarray(pts_desired, float).reshape(-1, 2) * scale
    pc = np.asarray(pts_current, float).reshape(-1, 2) * scale
    for xd, yd in pd.astype(int):
        cv2.circle(g, (int(xd), int(yd)), 4, GOAL_BGR, -1)
    for (xc, yc), (xd, yd) in zip(pc.astype(int), pd.astype(int), strict=False):
        cv2.arrowedLine(c, (int(xc), int(yc)), (int(xd), int(yd)), ARROW_BGR, 1, tipLength=0.18)
        cv2.circle(c, (int(xc), int(yc)), 4, CUR_BGR, -1)
        cv2.circle(c, (int(xd), int(yd)), 3, GOAL_BGR, 1)
    if border_bgr is not None:
        cv2.rectangle(c, (2, 2), (c.shape[1] - 3, c.shape[0] - 3), border_bgr, 6)
    canvas = np.concatenate([g, c], axis=1)
    if lines is None:
        bar = np.zeros((34, canvas.shape[1], 3), np.uint8)
        cv2.putText(bar, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
    else:
        bar = np.zeros((10 + 26 * len(lines), canvas.shape[1], 3), np.uint8)
        for k, text in enumerate(lines):
            colour = LANDING_BGR if text.startswith("LANDING") else (235, 235, 235)
            cv2.putText(bar, text, (8, 26 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colour, 1, cv2.LINE_AA)
    return np.concatenate([bar, canvas], axis=0)


class FilmWriter:
    """One JPEG per step plus ``frames_meta.json``, in one episode directory.

    Constructed with the episode's goal frame (RGB) and a directory; :meth:`step` is
    called once per control step with the live frame, the step's debug dict and the
    true pose error (scoring-side, caption-only). ``every`` keeps step 1 and every
    n-th step after it. ``title`` is prepended to every caption (the arm's name).
    """

    def __init__(self, out_dir, goal_rgb: np.ndarray, *, every: int = 1,
                 scale: float = 0.5, quality: int = 85, title: str = "") -> None:
        import cv2  # noqa: PLC0415

        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.goal_bgr = cv2.cvtColor(
            np.ascontiguousarray(np.asarray(goal_rgb, np.uint8)[..., :3]), cv2.COLOR_RGB2BGR)
        self.every = max(1, int(every))
        self.scale = float(scale)
        self.quality = int(quality)
        self.title = str(title)
        self.meta: list[dict] = []
        #: per-file display time in seconds (landing frames, the final frame); files
        #: not named here show for one frame at :data:`FPS` (debug films only)
        self.holds: dict[str, float] = {}
        self._landed: list[str] = []
        self._last_step = 0

    @staticmethod
    def points(debug: dict) -> tuple[np.ndarray, np.ndarray]:
        """``(goal-side desired, current matched)`` full-image pixels from a debug dict
        (a bench arm's ``pts_goal`` / ``pts_cur`` or CNS's ``pts_desired`` / ``pts_current``)."""
        zero = np.zeros((0, 2))
        pg = debug.get("pts_goal", debug.get("pts_desired", zero))
        pc = debug.get("pts_cur", debug.get("pts_current", zero))
        if pg is None or pc is None:
            return zero.copy(), zero.copy()
        return np.asarray(pg, float).reshape(-1, 2), np.asarray(pc, float).reshape(-1, 2)

    @staticmethod
    def _mm(x) -> str:
        return "-" if x is None else f"{float(x):.1f} mm"

    def _refs_line(self, refs: dict, r_deg: float) -> str:
        return (f"distance to goal {self._mm(refs.get('goal'))} | size-matched "
                f"{self._mm(refs.get('size_matched'))} | depth-true "
                f"{self._mm(refs.get('depth_true'))} | angle {float(r_deg):.1f} deg")

    def step(self, step_i: int, rgb: np.ndarray, debug: dict,
             t_mm: float, r_deg: float, *, refs: dict | None = None,
             landing: list[str] | None = None, meta_extra: dict | None = None) -> Path | None:
        """Write one frame; returns its path, or ``None`` when this step is skipped.

        The debug film's keywords (all optional): ``refs`` the distances to the three
        references (``goal`` / ``size_matched`` / ``depth_true``, mm or ``None``) - a
        two-line caption; ``landing`` the plain-word names of the stops that fired on
        this step - a yellow border, a third caption line, a hold, and the frame is
        written even when ``every`` would skip it; ``meta_extra`` merged into the frame's
        ``frames_meta.json`` entry."""
        import cv2  # noqa: PLC0415

        if step_i != 1 and step_i % self.every and not landing:
            return None
        pg, pc = self.points(debug)
        n = int(debug.get("n_matched", debug.get("n_matches", len(pc))))
        fe = float(debug.get("feature_error_norm", np.nan))
        fe_txt = "   -  " if not np.isfinite(fe) else f"{fe:6.4f}"
        cap = (f"{self.title}  step {step_i:4d}  matches {n:3d}  |e| {fe_txt}  "
               f"pos {t_mm:7.1f} mm  rot {r_deg:5.1f} deg   "
               f"goal (green=targets) | live (red=matched, arrows=to target)")
        lines = border = None
        name = f"step_{step_i:05d}.jpg"
        if refs is not None:
            if landing:
                self._landed.append(f"landed @ {step_i} ({', '.join(landing)})")
            fe_plain = "-" if not np.isfinite(fe) else f"{fe:.4g}"
            first = (f"{self.title}  |  step {step_i}  |  matches {n}  |  method error "
                     f"{fe_plain}")
            if self._landed:
                first += "  |  " + "; ".join(self._landed)
            lines = [first, self._refs_line(refs, r_deg)]
            if landing:
                lines.append("LANDING: " + "; ".join(landing))
                border = LANDING_BGR
                self.holds[name] = LANDING_HOLD_S
            if meta_extra is None:
                meta_extra = {}
            if landing:
                meta_extra = {**meta_extra, "hold_s": LANDING_HOLD_S}
        cur = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., :3]),
                           cv2.COLOR_RGB2BGR)
        frame = compose_match_frame(self.goal_bgr, cur, pg, pc, cap, scale=self.scale,
                                    lines=lines, border_bgr=border)
        cv2.imwrite(str(self.dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        entry = {"file": name, "step": int(step_i), "t_mm": round(float(t_mm), 2),
                 "r_deg": round(float(r_deg), 2), "n_matches": n}
        if meta_extra:
            entry.update(meta_extra)
        self.meta.append(entry)
        self._last_step = int(step_i)
        return self.dir / name

    def final(self, rgb: np.ndarray, *, refs: dict, r_deg: float,
              after_step: int | None = None, what: str = "after the last move (final pose)",
              meta_extra: dict | None = None) -> Path:
        """One more frame after the last move (the record's final pose): the live picture
        with no points drawn, sorted last, held :data:`FINAL_HOLD_S` s. It is numbered
        ``after_step + 1`` (the last control step + 1, as the debug viewer looks it up);
        without ``after_step``, the last written frame + 1. ``what`` is its caption (the
        recorder says when no move followed the last step)."""
        import cv2  # noqa: PLC0415

        i = (int(after_step) if after_step is not None else self._last_step) + 1
        name = f"step_{i:05d}.jpg"
        lines = [f"{self.title}  |  {what}"
                 + ("  |  " + "; ".join(self._landed) if self._landed else ""),
                 self._refs_line(refs, r_deg)]
        cur = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., :3]),
                           cv2.COLOR_RGB2BGR)
        frame = compose_match_frame(self.goal_bgr, cur, [], [], "", scale=self.scale,
                                    lines=lines)
        cv2.imwrite(str(self.dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self.holds[name] = FINAL_HOLD_S
        t_goal = refs.get("goal")
        entry = {"file": name, "step": int(i),
                 "t_mm": None if t_goal is None else round(float(t_goal), 2),
                 "r_deg": round(float(r_deg), 2), "n_matches": 0, "hold_s": FINAL_HOLD_S}
        if meta_extra:
            entry.update(meta_extra)
        self.meta.append(entry)
        return self.dir / name

    def close(self) -> Path:
        (self.dir / "frames_meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
        return self.dir


def encode_episode(ep_dir, scale: float = 1.0, stride: int = 1, fps: int = FPS,
                   keep_frames: bool = False, *, holds: dict[str, float] | None = None,
                   pattern: str = "step_*.jpg", out_name: str = "episode.mp4",
                   crf: int = 27) -> Path | None:
    """``step_*.jpg`` in ``ep_dir`` -> ``episode.mp4``; ``None`` when there is nothing.

    ffmpeg (libx264, yuv420p, ``+faststart``) when :func:`ffmpeg_bin` finds one,
    otherwise cv2 with an ``avc1``/``mp4v`` fourcc - a fallback, not a substitute.
    ``keep_frames=False`` deletes the JPEGs once the video exists. The debug keywords:
    ``holds`` (file name -> seconds on screen; the others one frame), ``pattern`` /
    ``out_name`` / ``crf`` (the clean film: ``clean_*.jpg`` -> ``clean.mp4`` at crf 20).
    """
    import cv2  # noqa: PLC0415

    ep_dir = Path(ep_dir)
    frames = sorted(ep_dir.glob(pattern))[::max(1, int(stride))]
    if not frames:
        return None
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    out = ep_dir / out_name
    done = None
    ff = ffmpeg_bin()
    if ff:
        lst = ep_dir / "_frames.txt"
        lst.write_text("".join(f"file '{f.name}'\nduration {(holds or {}).get(f.name, 1.0 / fps)}\n"
                               for f in frames)
                       # the concat demuxer drops the LAST entry's duration: name it again
                       # so a held final frame is really held (debug films only)
                       + (f"file '{frames[-1].name}'\n" if holds else ""))
        r = subprocess.run(
            [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-vf", f"scale={ow}:{oh}", "-r", str(int(fps)),   # one frame per step
             "-c:v", "libx264", "-preset", "veryfast",
             "-crf", str(int(crf)), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            capture_output=True, cwd=str(ep_dir), check=False)
        lst.unlink(missing_ok=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            done = out
        else:
            print(f"[film] ffmpeg failed for {ep_dir.name}, cv2 fallback: "
                  f"{r.stderr[-200:]!r}", flush=True)
    if done is None:
        for fourcc in ("avc1", "mp4v"):
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps, (ow, oh))
            if vw.isOpened():
                break
        else:
            return None
        for f in frames:
            img = cv2.imread(str(f))
            if img is None or img.shape[:2] != (h, w):
                continue
            # a held frame is repeated for its hold (debug films only)
            reps = max(1, int(round((holds or {}).get(f.name, 1.0 / fps) * fps)))
            for _ in range(reps):
                vw.write(img if (ow, oh) == (w, h)
                         else cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA))
        vw.release()
        done = out if out.exists() and out.stat().st_size > 0 else None
    if done is not None and not keep_frames:
        for f in ep_dir.glob(pattern):
            f.unlink()
    return done


class CleanFilmWriter:
    """The clean live film: every step's live picture, un-drawn, at half size.

    For the offline viewer (``scripts/debug_view.py``), which draws its own panels on
    it. ``step(i, rgb)`` writes ``clean_{i:05d}.jpg``, ``final(rgb, after_step)`` one more
    frame after the last move (numbered ``after_step + 1``, as the viewer looks it up),
    ``close()`` writes ``clean_meta.json`` (``[{"frame", "step", "kind"}]``) and encodes
    ``clean.mp4`` (H.264, crf 20), the JPEGs deleted after. The size is rounded to even
    pixels (960 x 540 for the D435i picture). Leftover ``clean_*.jpg`` of a crashed earlier
    run in the same directory are deleted first, so they cannot end up in this film.
    """

    def __init__(self, out_dir, *, every: int = 1, scale: float = 0.5, quality: int = 92) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every))
        self.scale = float(scale)
        self.quality = int(quality)
        self.meta: list[dict] = []
        self._last = 0
        for f in self.dir.glob("clean_*.jpg"):
            f.unlink()

    def _write(self, i: int, rgb: np.ndarray, kind: str) -> None:
        import cv2  # noqa: PLC0415

        img = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., :3]),
                           cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        size = (max(2, int(round(w * self.scale / 2.0)) * 2),
                max(2, int(round(h * self.scale / 2.0)) * 2))
        cv2.imwrite(str(self.dir / f"clean_{i:05d}.jpg"),
                    cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self.meta.append({"frame": len(self.meta), "step": int(i), "kind": kind})
        self._last = int(i)

    def step(self, step_i: int, rgb: np.ndarray) -> None:
        if step_i != 1 and step_i % self.every:
            return
        self._write(int(step_i), rgb, "step")

    def final(self, rgb: np.ndarray, after_step: int | None = None) -> None:
        self._write((int(after_step) if after_step is not None else self._last) + 1, rgb,
                    "final")

    def close(self) -> Path | None:
        (self.dir / "clean_meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
        return encode_episode(self.dir, scale=1.0, keep_frames=False, pattern="clean_*.jpg",
                              out_name="clean.mp4", crf=20)
