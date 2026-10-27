# Assets and your own scenes

The pictures and objects are not in the repository (third-party artwork and scans). vsbench
looks for them under `$VSBENCH_WORKSPACE_ROOT` (default `~/vsbench_data`).

**Posters.** Each needs one picture:

```
$VSBENCH_WORKSPACE_ROOT/assets/hollywood/resized.png   ART-VS's poster (its catkin_ws/ibvs/models/viso/materials/textures/resized.png)
$VSBENCH_WORKSPACE_ROOT/assets/poster2/poster2.png     the COCO val2017 image 000000000724, saved as PNG
$VSBENCH_WORKSPACE_ROOT/assets/poster3/poster3.jpg     Bruegel, "The Tower of Babel" (Vienna), Wikimedia Commons, 3840 x 2810
```

**Rooms.** The three rooms need their floor pictures and their objects converted into the
object cache (`$VSBENCH_WORKSPACE_ROOT/objects`). These files are available on request from
the authors - or build your own scene. With the files from us, also point two variables at
them: `VSSIM_YCB_ROOT` (the YCB scans, used by `cluttered2` / `cluttered3` and `swap:ycb/...`)
and `VSSIM_ASSET_ROOT` (the curated object library, used by `swap:` and the distractors).

**Your own scene.** Write a scenes file in the format of
[`vsbench/configs/scenes.yaml`](vsbench/configs/scenes.yaml), with your own pictures:

```yaml
scenes:
  my_poster:                     # a picture on a 0.8 x 0.6 m plate, seen straight on
    kind: poster
    width_m: 0.8
    height_m: 0.6
    thickness_m: 0.01
    texture: /path/to/picture.png
    backdrop: {size_m: 4.0, gray: 0.5}
    goal: {kind: fill_frame, coverage: 1.35}
    ransac: homography
  my_room:                       # objects on a floor, seen from straight above
    kind: multi_object
    ground: {texture: /path/to/floor.png, size_m: 2.4, tile_period_m: 2.4}
    objects:
      - {category: mug, instance_id: my/mug, dx: 0.0, dy: 0.0}    # metres from the centre
    goal: {kind: scene_view, frame_fill: 0.85, elevation_deg: 90}
    ransac: fundamental
```

Convert each object once: `python tools/convert_objects.py obj my/mug path/to/mug.obj` takes a
Wavefront OBJ (metres, z up; its `.mtl` and `map_Kd` texture beside it) or a USD file
(`pip install -e ".[usd]"`) into the object cache; an id with a `/` keeps the object's own size.
Then load the file and drive the camera yourself, as in the README:

```python
from vsbench.vs.scenes import load_scenes
specs, _ = load_scenes("my_scenes.yaml")
scene = MujocoScene("my_room", specs=specs)
```

The benchmark command (`python -m vsbench fly`) flies the six scenes above, whose start poses
are frozen.
