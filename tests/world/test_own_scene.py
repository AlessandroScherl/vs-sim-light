"""Your own scene (README, "Your own scene"): a plain OBJ converted with
``tools/convert_objects.py obj`` and a scenes file of your own, built without rendering."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _cube(tmp: Path) -> Path:
    from PIL import Image

    Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        tmp / "cube.png")
    (tmp / "cube.mtl").write_text("newmtl skin\nKd 1 1 1\nmap_Kd cube.png\n")
    h = 0.05
    V = [(x, y, z) for z in (-h, h) for y in (-h, h) for x in (-h, h)]
    faces = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    lines = ["mtllib cube.mtl", "usemtl skin"] + ["v %f %f %f" % v for v in V]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    lines += ["f " + " ".join(f"{i}/{k + 1}" for k, i in enumerate(f)) for f in faces]
    (tmp / "cube.obj").write_text("\n".join(lines) + "\n")
    return tmp / "cube.obj"


def test_a_plain_obj_and_a_scenes_file_of_your_own(tmp_path, monkeypatch):
    from vsbench.vs.scenes import load_scenes
    from vsbench.world import assets as A
    from vsbench.world.build import build_scene

    cache = tmp_path / "cache"
    p = subprocess.run([sys.executable, str(REPO / "tools" / "convert_objects.py"), "--cache",
                        str(cache), "obj", "my/cube", str(_cube(tmp_path))],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert (A.object_dir("my/cube", cache) / "meta.json").exists()
    monkeypatch.setenv("VSBENCH_OBJECT_CACHE", str(cache))
    (tmp_path / "my_scenes.yaml").write_text(f"""
scenes:
  my_poster:
    kind: poster
    width_m: 0.8
    height_m: 0.6
    thickness_m: 0.01
    texture: {tmp_path / 'cube.png'}
    backdrop: {{size_m: 4.0, gray: 0.5}}
    goal: {{kind: fill_frame, coverage: 1.35}}
    ransac: homography
  my_room:
    kind: multi_object
    ground: {{texture: {tmp_path / 'cube.png'}, size_m: 2.4, tile_period_m: 2.4}}
    objects:
      - {{category: box, instance_id: my/cube, dx: 0.0, dy: 0.0}}
      - {{category: box, instance_id: my/cube, dx: 0.2, dy: 0.1}}
    goal: {{kind: scene_view, frame_fill: 0.85, elevation_deg: 90}}
    ransac: fundamental
""")
    specs, _ = load_scenes(tmp_path / "my_scenes.yaml")
    assert build_scene(specs["my_poster"]).scene_spec.objects
    room = build_scene(specs["my_room"])
    assert len(room.placed) == 2
    for placed, (dx, dy) in zip(room.placed, [(0.0, 0.0), (0.2, 0.1)], strict=True):
        assert placed.scale == 1.0                         # a '/' in the id: native metric size
        assert np.allclose(placed.pos, [dx, dy, 0.05], atol=1e-6)    # seated on z = 0
    from vsbench.world.look import resolve_look

    assert resolve_look(None, "my_room").name == "raw"     # no fitted look for your own scene
