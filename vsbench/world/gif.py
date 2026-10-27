"""GIFs of a flight: the shared encoder and the animated debug view (``--debug-gif``).

:func:`write_gif` writes frames as one looping GIF with one shared palette (the README's
demo and the debug view). :func:`debug_gif` animates the copied debug view
(``vsbench.vs.bench.debugview``: ``Episode``, ``draw_timeline_base``, ``compose``, drawn
unchanged) over one episode directory written by the copied ``DebugLog`` - every
``every``-th logged step (chosen so the GIF has at most ``max_frames`` frames, stated in the
returned dict), each stop's step always, and the final frame; stops held 1 s, the final
frame 2 s, as the copied ``render_episode`` holds them in its MP4. Nothing here is on a
control path.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def write_gif(frames, path, *, fps: float, hold_s: float = 1.5, colors: int = 128,
              durations_ms=None) -> Path:
    """RGB frames (H, W, 3 uint8) as one looping GIF: one palette from a sample of the frames;
    ``durations_ms`` (one per frame) or ``1/fps`` with the first and last frame held
    ``hold_s``."""
    from PIL import Image

    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
    if not frames:
        raise ValueError("no frames")
    sample = frames[:: max(1, len(frames) // 12)]
    strip = Image.fromarray(np.concatenate(sample, axis=0))
    pal = strip.quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT)
    ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    if durations_ms is None:
        dur = [int(round(1000 / float(fps)))] * len(ims)
        dur[0] = dur[-1] = int(1000 * hold_s)
    else:
        dur = [int(d) for d in durations_ms]
        if len(dur) != len(ims):
            raise ValueError(f"{len(dur)} durations for {len(ims)} frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=dur, loop=0, optimize=True)
    return path


def _resize(img, width: int):
    import cv2

    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (int(width), int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def debug_gif(ep_dir, out_path, *, every: int | None = None, max_frames: int = 90,
              width: int = 800, fps: float = 8.0, colors: int = 128,
              goal_mask=None) -> dict:
    """``out_path``: the copied debug view of one episode directory, animated.

    ``goal_mask`` (bool [H, W], optional) replaces the mask the debug log drew from the goal
    depth (a file-mask flight: the copied recorder only knows the depth outline). Returns what
    was drawn: the steps, ``every``, the frame count, the size in bytes.
    """
    import cv2

    from vsbench.vs.bench import debugview as dv
    from vsbench.vs.bench.films import FINAL_HOLD_S, LANDING_HOLD_S

    ep = dv.Episode(ep_dir)
    try:
        if goal_mask is not None:
            m = np.asarray(goal_mask, dtype=np.uint8) * 255
            ep.mask = cv2.resize(m, (dv.PANEL_W, dv.PANEL_H),
                                 interpolation=cv2.INTER_NEAREST) > 127
        n = len(ep.steps)
        if every is None:
            every = max(1, math.ceil(n / max(1, max_frames - 1)))
        stops = {m[0] for m in ep.markers}
        entries = [("step", s) for k, s in enumerate(ep.steps)
                   if k % every == 0 or int(s["i"]) in stops or k == n - 1]
        if ep.final is not None:
            entries.append(("final", ep.final))
        base = dv.draw_timeline_base(ep)
        frames, dur, steps = [], [], []
        for kind, entry in entries:
            img = dv.compose(ep, entry, kind=kind, timeline_base=base)
            frames.append(cv2.cvtColor(_resize(img, width), cv2.COLOR_BGR2RGB))
            if kind == "final":
                dur.append(int(1000 * FINAL_HOLD_S))
                steps.append("final")
            else:
                held = int(entry["i"]) in stops
                dur.append(int(1000 * LANDING_HOLD_S) if held else int(round(1000 / fps)))
                steps.append(int(entry["i"]))
    finally:
        ep.close()
    out = write_gif(frames, out_path, fps=fps, colors=colors, durations_ms=dur)
    return {"path": str(out), "every": int(every), "frames": len(frames), "logged_steps": n,
            "steps": steps, "width": int(width), "fps": float(fps), "colors": int(colors),
            "bytes": out.stat().st_size}
