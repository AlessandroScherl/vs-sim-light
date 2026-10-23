#!/usr/bin/env python
"""Draw the front page's pictures (``docs/media/``), on the CPU, in the default look.

    source tools/cpu_env.sh
    PYTHONPATH=. $VSBENCH_PY tools/readme_media.py scenes   # scenes.png: the six goal views
    PYTHONPATH=. $VSBENCH_PY tools/readme_media.py demo     # demo.gif, debug_view.jpg, demo.json
    PYTHONPATH=. $VSBENCH_PY tools/readme_media.py gif      # re-encode demo.gif from saved frames

``scenes`` renders each scene's goal picture in its own process (a room holds up to ~3.6 GB)
and lays the six out in a 3 x 2 contact sheet. ``demo`` flies one published baseline
(``sift_ransac``) on ``cluttered2`` from a start a few centimetres off the goal, through the
copied benchmark loop, with two listeners: a GIF hook (the copied ``compose_match_frame``:
goal | live camera, the matched points and an arrow from each to its target, plus a short
caption with the true distance to the goal) and the copied ``DebugLog``, whose episode
directory the copied debug view (``vsbench.vs.bench.debugview.render_still``) draws as
``debug_view.jpg``. ``demo.json`` keeps the flight's numbers and commits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MEDIA = REPO / "docs" / "media"
SCENES = ("hollywood", "poster2", "poster3", "industrial1_drill", "cluttered2", "cluttered3")
TILE_W, TILE_H = 480, 270


# ------------------------------------------------------------------------ scenes
def cmd_tile(args) -> int:
    """One scene's goal picture, downsized, into ``args.tile`` (own process)."""
    from vsbench.world import fly

    fly.station_guard(2)
    import cv2

    from vsbench.vs.scenes import load_scenes
    from vsbench.world.mujoco_scene import MujocoScene

    specs, _ = load_scenes()
    with MujocoScene(args.one, specs=specs) as scene:
        fly.require_software_gl(scene)
        rgb, _depth = scene.render_goal()
        print(f"[media] {args.one}: look {scene.look.name}, {scene.render_mode}", flush=True)
    small = cv2.resize(np.ascontiguousarray(rgb), (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.tile, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    return 0


def cmd_scenes(args) -> int:
    from PIL import Image, ImageDraw

    label_h, pad = 30, 6
    sheet = Image.new("RGB", (3 * TILE_W + 4 * pad, 2 * (TILE_H + label_h) + 3 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    with tempfile.TemporaryDirectory() as tmp:
        for k, name in enumerate(SCENES):
            tile = Path(tmp) / f"{name}.png"
            subprocess.run([sys.executable, __file__, "tile", "--one", name, "--tile", str(tile)],
                           check=True)
            x = pad + (k % 3) * (TILE_W + pad)
            y = pad + (k // 3) * (TILE_H + label_h + pad)
            with Image.open(tile) as im:
                sheet.paste(im.convert("RGB"), (x, y))
            draw.text((x + 4, y + TILE_H + 6), name, fill=(20, 20, 20), font_size=20)
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "scenes.png"
    sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
    print(f"[media] {out} {out.stat().st_size / 1e6:.2f} MB", flush=True)
    return 0


# -------------------------------------------------------------------------- demo
def demo_start(scene, offset_mm, axis, angle_deg) -> dict:
    """A start ``offset_mm`` (goal camera frame: x right, y down, z forward) off the goal,
    turned by ``angle_deg`` about ``axis`` (goal camera frame), in a start file's shape."""
    from vsbench.vs import starts as startsmod

    g = scene.goal_pose
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R_rel = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    base = startsmod.sample_near(g.position, g.rotation, scene.working_distance, n=1,
                                 id_prefix="readme_")[0]
    pos = g.position + g.rotation @ (np.asarray(offset_mm, float) / 1000.0)
    return dict(base, start_id="readme_demo", cell="readme", tier="readme",
                cam_pos=[float(v) for v in pos], cam_R_opt=(g.rotation @ R_rel).tolist())


class GifHook:
    """``on_step``: one GIF frame every ``every`` steps - goal | live, matches, a caption."""

    def __init__(self, goal_rgb, *, title: str, every: int, width: int, goal_pose) -> None:
        import cv2

        self.goal_bgr = cv2.cvtColor(np.ascontiguousarray(goal_rgb), cv2.COLOR_RGB2BGR)
        self.title, self.every, self.goal = title, int(every), goal_pose
        self.scale = width / 2 / goal_rgb.shape[1]
        self.frames: list[np.ndarray] = []
        self.rows: list[dict] = []

    def frame(self, rgb, debug, t_mm, r_deg, step_i, *, points=True) -> np.ndarray:
        import cv2

        from vsbench.vs.bench.films import FilmWriter, compose_match_frame

        pg, pc = FilmWriter.points(debug) if points else (np.zeros((0, 2)), np.zeros((0, 2)))
        cur = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        img = compose_match_frame(self.goal_bgr, cur, pg, pc, "", scale=self.scale)[34:]
        img = img.copy()
        w = img.shape[1] // 2
        for x0, text in ((0, "goal view"), (w, "live camera")):
            cv2.rectangle(img, (x0, 0), (x0 + 104, 22), (0, 0, 0), -1)
            cv2.putText(img, text, (x0 + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        bar = np.full((30, img.shape[1], 3), 24, np.uint8)
        cv2.putText(bar, f"{self.title}   step {step_i:3d}   distance to goal {t_mm:5.1f} mm, "
                         f"{r_deg:4.1f} deg", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
        return cv2.cvtColor(np.concatenate([bar, img], axis=0), cv2.COLOR_BGR2RGB)

    def __call__(self, step_i, rgb, debug, pose):
        from vsbench.vs.scene import rotation_error_deg, translation_error_mm

        t = translation_error_mm(pose.position, self.goal.position)
        r = rotation_error_deg(pose.rotation, self.goal.rotation)
        self.rows.append({"step": int(step_i), "t_mm": round(float(t), 3), "r_deg": round(float(r), 3)})
        if step_i == 1 or step_i % self.every == 0:
            self.frames.append(self.frame(rgb, debug, t, r, step_i))



def write_gif(frames, path: Path, *, fps: int, hold_s: float, colors: int) -> None:
    """The frames as one looping GIF: one shared palette, the first and last frame held."""
    from PIL import Image

    strip = Image.fromarray(np.concatenate(frames[:: max(1, len(frames) // 12)], axis=0))
    pal = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    dur = [int(1000 / fps)] * len(ims)
    dur[0] = dur[-1] = int(1000 * hold_s)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=dur, loop=0,
                optimize=True)


def cmd_demo(args) -> int:
    from vsbench.world import fly

    env = fly.station_guard(2)
    from vsbench.vs import benchmark
    from vsbench.vs.bench import debugview
    from vsbench.vs.bench.debuglog import DebugLog
    from vsbench.vs.bench.runner import run_episode
    from vsbench.vs.scene import Pose, rotation_error_deg, translation_error_mm
    from vsbench.vs.scenes import load_scenes
    from vsbench.world.mujoco_scene import MujocoScene

    protocol = fly.load_protocol()
    specs, _ = load_scenes()
    MEDIA.mkdir(parents=True, exist_ok=True)
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with MujocoScene(args.scene, specs=specs) as scene:
        fly.require_software_gl(scene)
        benchmark.check_scene_goal(scene, benchmark.load_starts(args.scene, "v2"))
        goal_rgb, _ = scene.render_goal()
        start = demo_start(scene, args.offset_mm, args.axis, args.angle_deg)
        sp = Pose.from_start(start)
        t0 = translation_error_mm(sp.position, scene.goal_pose.position)
        r0 = rotation_error_deg(sp.rotation, scene.goal_pose.rotation)
        print(f"[media] {args.scene} look {scene.look.name}: start {t0:.1f} mm / {r0:.2f} deg, "
              f"goal {scene.working_distance:.3f} m", flush=True)
        cfg = fly.arm_config(protocol, args.arm, max_iter=args.max_iter, device="cpu")
        arm = fly.build(protocol, args.arm, cfg, intrinsics=scene.intrinsics,
                        scene_name=args.scene, scene_spec=specs[args.scene])
        hook = GifHook(goal_rgb, title=f"{args.arm} on {args.scene}", every=args.every,
                       width=args.width, goal_pose=scene.goal_pose)
        ep_dir = work / "debug_episode"
        with DebugLog(ep_dir, arm_id=args.arm, clean=True) as dlog:
            rec = run_episode(scene, arm, start, cfg, on_step=hook, debug_log=dlog)
        # the final pose, after the last move, without points
        rgb, _d, pose = scene.render()
        hook.frames.append(hook.frame(rgb, {}, rec["final_translation_mm"],
                                      rec["final_rotation_deg"], rec["steps"], points=False))
    gif = MEDIA / "demo.gif"
    np.savez_compressed(work / "gif_frames.npz", frames=np.stack(hook.frames))
    write_gif(hook.frames, gif, fps=args.fps, hold_s=1.5, colors=args.colors)
    still = debugview.render_still(ep_dir, which=args.debug_step, out_dir=MEDIA,
                                   name="debug_view_full.png")
    import cv2

    img = cv2.imread(str(still))
    cv2.imwrite(str(MEDIA / "debug_view.jpg"), cv2.resize(img, (1200, 750), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    still.unlink()
    keep = ("start_translation_mm", "start_rotation_deg", "final_translation_mm",
            "final_rotation_deg", "steps", "terminated_reason", "stop_step", "stop_translation_mm",
            "sec_per_step", "mean_n_matches", "look", "look_curve_id", "render_mode", "success")
    out = {"what": "the flight behind docs/media/demo.gif and debug_view.jpg (tools/readme_media.py demo)",
           "scene": args.scene, "arm": args.arm, "start": {k: start[k] for k in ("start_id", "cam_pos", "cam_R_opt")},
           "offset_mm": args.offset_mm, "axis": args.axis, "angle_deg": args.angle_deg,
           "max_iter": args.max_iter, "gif_every": args.every, "gif_fps": args.fps,
           "gif_colors": args.colors, "gif_frames": len(hook.frames),
           "debug_step": args.debug_step, "vsbench_commit": fly.vsbench_commit(),
           "vssim_commit": fly.vssim_commit(), "bridge_env": env,
           "record": {k: rec.get(k) for k in keep}, "curve": hook.rows}
    (MEDIA / "demo.json").write_text(json.dumps(out, indent=1) + "\n")
    for p in (gif, MEDIA / "debug_view.jpg"):
        print(f"[media] {p} {p.stat().st_size / 1e6:.2f} MB", flush=True)
    print(f"[media] {rec['start_translation_mm']:.1f} -> {rec['final_translation_mm']:.2f} mm, "
          f"{rec['steps']} steps, {rec['terminated_reason']}, {len(hook.frames)} GIF frames", flush=True)
    return 0


def cmd_gif(args) -> int:
    """Re-encode demo.gif from the saved frames (no flight)."""
    frames = list(np.load(Path(args.work) / "gif_frames.npz")["frames"])
    gif = MEDIA / "demo.gif"
    write_gif(frames, gif, fps=args.fps, hold_s=1.5, colors=args.colors)
    print(f"[media] {gif} {gif.stat().st_size / 1e6:.2f} MB, {len(frames)} frames", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenes")
    s.set_defaults(fn=cmd_scenes)
    t = sub.add_parser("tile")
    t.add_argument("--one", required=True, choices=SCENES)
    t.add_argument("--tile", required=True)
    t.set_defaults(fn=cmd_tile)
    d = sub.add_parser("demo")
    d.add_argument("--scene", default="cluttered2")
    d.add_argument("--arm", default="sift_ransac")
    d.add_argument("--offset-mm", type=float, nargs=3, default=[50.0, -40.0, -90.0])
    d.add_argument("--axis", type=float, nargs=3, default=[0.2, -0.1, 1.0])
    d.add_argument("--angle-deg", type=float, default=15.0)
    d.add_argument("--max-iter", type=int, default=160)
    d.add_argument("--every", type=int, default=3, help="one GIF frame every N steps")
    d.add_argument("--fps", type=int, default=12)
    d.add_argument("--colors", type=int, default=96)
    d.add_argument("--width", type=int, default=640)
    d.add_argument("--debug-step", default="40", help="the debug view's step (or landing/final)")
    d.add_argument("--work", default=str(REPO / "results" / "_tmp" / "readme_media"))
    d.set_defaults(fn=cmd_demo)
    g = sub.add_parser("gif", help="re-encode demo.gif from the last demo's saved frames")
    g.add_argument("--fps", type=int, default=12)
    g.add_argument("--colors", type=int, default=96)
    g.add_argument("--work", default=str(REPO / "results" / "_tmp" / "readme_media"))
    g.set_defaults(fn=cmd_gif)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
