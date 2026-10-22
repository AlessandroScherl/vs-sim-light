"""``python -m vsbench.replay <log.npz> [--video out.mp4] [--backend <variant>]``.

Rebuilds the scene and the camera from the log's meta, re-renders the desired view and
every logged pose, and compares the frame hashes with the ones recorded during the
episode. The method is never run: twists, errors and matches come from the log. A
mismatch, for example after switching from ``osmesa`` to ``glfw``, is reported per frame,
and with ``--video`` the video is still written and marked as not identical. The exit
code is 0 only when every frame is identical.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vsbench.camera import CameraModel
from vsbench.episode import EpisodeLog, frame_hash
from vsbench.render.base import make_renderer
from vsbench.scenes import make_scene


@dataclass
class ReplayResult:
    steps: int
    backend: str
    logged_backend: str
    target_identical: bool
    mismatches: list[int] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return self.target_identical and not self.mismatches

    def summary(self) -> str:
        if self.identical:
            return f"replay identical: {self.steps} frames plus the desired view on {self.backend}"
        parts = []
        if not self.target_identical:
            parts.append("desired view differs")
        if self.mismatches:
            parts.append(f"{len(self.mismatches)} of {self.steps} frames differ (first at step {self.mismatches[0]})")
        return f"replay NOT identical on {self.backend} (logged on {self.logged_backend}): " + ", ".join(parts)


def camera_of(log: EpisodeLog) -> CameraModel:
    c = log.meta["camera"]
    return CameraModel(np.array(c["K"]), int(c["width"]), int(c["height"]), float(c["near"]), float(c["far"]))


def logged_backend(log: EpisodeLog) -> str:
    return str(log.meta.get("renderer", {}).get("variant", "mujoco-egl"))


def build_renderer(log: EpisodeLog, backend: str | None = None):
    """The scene and camera of the log on ``backend`` (default: the logged variant)."""
    return make_renderer(backend or logged_backend(log), make_scene(log.meta["scene"]), camera_of(log))


def render_target(log: EpisodeLog, renderer) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(rgb, depth, identical)`` of the desired view."""
    rgb, depth = renderer.render(np.array(log.meta["T_wc_target"]))
    return rgb, depth, frame_hash(rgb) == int(log.meta["target_hash"])


def iter_frames(log: EpisodeLog, renderer) -> Iterator[tuple[int, np.ndarray, np.ndarray, bool]]:
    """Yield ``(k, rgb, depth, identical)`` for every logged pose, in order."""
    hashes = log.frame_hash
    for k, T in enumerate(log.T_wc):
        rgb, depth = renderer.render(T)
        yield k, rgb, depth, frame_hash(rgb) == int(hashes[k])


def replay_check(log: EpisodeLog, renderer, on_frame=None) -> ReplayResult:
    """Re-render everything and compare hashes; ``on_frame(k, rgb, depth, identical)`` sees each frame."""
    _, _, target_ok = render_target(log, renderer)
    res = ReplayResult(len(log), getattr(renderer, "name", "?"), logged_backend(log), target_ok)
    for k, rgb, depth, ok in iter_frames(log, renderer):
        if not ok:
            res.mismatches.append(k)
        if on_frame is not None:
            on_frame(k, rgb, depth, ok)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("log", help="an episode .npz written by run_episode")
    ap.add_argument("--video", default=None, help="write the debug video here (mp4)")
    ap.add_argument("--backend", default=None, help="renderer variant (default: the one in the log)")
    a = ap.parse_args(argv)
    log = EpisodeLog.load(a.log)
    m = log.meta
    print(f"{Path(a.log).name}: scene {m['scene']}, method {m['method']}, {len(log)} steps, status {m.get('status')}, logged on {logged_backend(log)}")
    r = build_renderer(log, a.backend)
    try:
        if a.video:
            from vsbench.video import write_video

            res = write_video(log, r, a.video)
            print(f"video: {a.video}")
        else:
            res = replay_check(log, r)
    finally:
        r.close()
    print(res.summary())
    return 0 if res.identical else 1


if __name__ == "__main__":
    sys.exit(main())
