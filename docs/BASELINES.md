# Baselines, modes and flags

Everything you can fly and every flag the commands take. The method names come from
[`vsbench/configs/protocol.yaml`](../vsbench/configs/protocol.yaml) (each name is a *cell*: an
arm file in [`vsbench/configs/arms/`](../vsbench/configs/arms/) plus the cell's changes); the
flag tables are generated from the argparse definitions (`python tools/cli_flags.py --write`;
`tests/test_baselines_doc.py` fails when the two drift apart).

`python -m vsbench list` prints the same names.

## What every baseline shares

Unless a name below says otherwise: the IBVS law at gain 2.0, 60 steps per second, 24 matched
points, no weighting, sub-pixel points, a cap of 600 steps (`--max-iter` changes it), the flight
aborted when the feature error grows past 3x its start, scored at CNS's plateau stop
(`cns_plateau`) with the published velocity-quiet stop and `dec_plateau` recorded beside it.
Every record carries seven success columns: `predicate_a` (0.1 x working distance or 10 mm, and
1 deg), `paper_3cm_3deg`, `cns_code_0p1d_10deg`, `abs_10mm_1deg`, `abs_20mm_2deg`,
`abs_50mm_5deg`, `abs_100mm_10deg`.

## The names you can fly

### ViT-VS ([arXiv 2503.04545](https://arxiv.org/abs/2503.04545))

| name | what it is |
| --- | --- |
| `vitvs_dinov3` | ViT-VS on DINOv3 ViT-S/16 (512 px input, 32 x 32 patches): mutual nearest-neighbour patch matches, no RANSAC. Needs the gated DINOv3 weights. |
| `vitvs_dinov2` | ViT-VS exactly as it ships: DINOv2 ViT-S/14 from torch.hub, block-11 tokens log-binned to 3456-d descriptors. |
| `vitvs_dinov3_objmask` | `vitvs_dinov3` with its points taken only on the goal picture's objects (the floor masked out from the goal depth). Rooms and instance scenes only. |
| `vitvs_dinov3_objmask_ema08` | `vitvs_dinov3_objmask` plus the published simulation smoothing: an exponential moving average of the twist, 0.8 on the new value. |
| `vitvs_dinov3_pre4way` | `vitvs_dinov3` with the published 4-way roll search (0/90/180/270 deg) done once before the loop. |
| `vitvs_dinov3_pre4way_objmask` | both of the above: the 4-way roll search and the objects-only mask. |

### ART-VS ([art-vs.github.io](https://art-vs.github.io/))

| name | what it is |
| --- | --- |
| `artvs` | ViT-VS plus the adaptive switch: DINOv2 ViT-S/14 at 224 px on the whole picture, then, once the error falls below 0.20 of its first value (the switch latches), 36 tiles (6 x 6 over 1344 px, each read at 224). The published 4-way roll is OFF in this cell. |
| `artvs_pre4way` | `artvs` with the published 4-way roll search done once before the loop. |
| `artvs_objmask` | `artvs` with the objects-only goal mask (handed to its coarse stage). Rooms and instance scenes only. |
| `artvs_pre4way_objmask` | both: the 4-way roll search and the objects-only mask. |

### Classical (SIFT / ORB / AKAZE into the same IBVS law)

Each detector in two columns, which are two different methods and never pooled: with RANSAC
(3 px; the scene's own model - a homography on the posters, a fundamental matrix in the rooms)
and without. The best 24 matches by descriptor distance.

| name | what it is |
| --- | --- |
| `sift_ransac` | SIFT + RANSAC. |
| `sift_noransac` | SIFT, no RANSAC. |
| `orb_ransac` | ORB + RANSAC. |
| `orb_noransac` | ORB, no RANSAC. |
| `akaze_ransac` | AKAZE + RANSAC. |
| `akaze_noransac` | AKAZE, no RANSAC. |
| `sift_ransac_objmask` | `sift_ransac` with the objects-only goal mask. Rooms and instance scenes only. |

`--mask objects` on any baseline flies its `_objmask` cell where the protocol has one, else the
same baseline with `matcher.goal_mask: objects` (the label says `objmask`).

### Registered methods (`@register`, flown through `MethodArm`)

| name | what it is |
| --- | --- |
| `sift_ibvs` | vsbench's own feature IBVS (`vsbench/methods/feature_ibvs.py`) with SIFT: ratio test, RANSAC, depth at each keypoint, the point IBVS law. The template for your own method. |
| `orb_ibvs` | the same with ORB. |

Their constructor takes `--param key=value`: `gain` (1.0), `depth` (`dense` \| `scalar`),
`ransac` (`homography` \| `fundamental`), `ransac_px` (3.0), `ratio` (0.75 SIFT, 0.8 ORB),
`max_features` (2000), `min_inliers` (8), `seed` (0). Your own method joins this list with
`--plugin`. (The six baselines are registered as methods too, for your own loop -
`make_method("sift_ransac")` - but `fly` always flies them as the copied arm.)

The protocol also names cells whose methods are not part of this simulator (CNS, DINOBot,
LightGlue, RoMa, ViSP, ART-VS's object mode with LangSAM): `fly` refuses them by name.

## Scenes

| scenes | start sets | how to fly |
| --- | --- | --- |
| `hollywood`, `poster2`, `poster3` (posters) | `capped30`, `wide100`, `narrow20`, `near60` | `python -m vsbench fly` |
| `industrial1_drill`, `cluttered2`, `cluttered3` (rooms) | `capped30`, `wide100` | `python -m vsbench fly` |
| `shoe1`, `shoe2` (one shoe) | `capped30`, `wide100` | `python -m vsbench fly` |
| `inst_shoe`, `inst_bottle`, `inst_teapot`, `inst_drill`, `inst_mug`, `inst_hammer` (one object, the big one in the goal picture) | `simple5`, `simple10`, `t1near3` | `python -m vsbench.world.check_baselines --far-set ...` |
| the same with `_rev` (the small one in the goal picture) | `simple5`, `simple10` | `python -m vsbench.world.check_baselines --far-set ...` |

The start sets: `capped30` the first 30 of the frozen wide starts with roll up to 90 deg (method
screening, the default); `wide100` all 100 (report in thirds by rotation); `narrow20` 20 close
starts (~78 mm / 5.8 deg, posters); `near60` 60 starts 3-8 mm off (posters); `simple5` /
`simple10` the instance scenes' first 5 / all 10 starts (roll under 60 deg); `t1near3` three
starts 3-8 mm off. `fly` takes only the protocol's four sets; the instance sets fly through
`check_baselines`, e.g.

```bash
python -m vsbench.world.check_baselines --scenes inst_hammer --arms vitvs_dinov3_objmask \
    --near-steps 200 --far 5 --far-set simple5 --perturb swap:@mine1 --out runs/check/hammer
```

`configs/scenes.yaml` holds a few more scenes without a start file here (`objects1`,
`bottle1`, `mug1`, ...); they render, but no frozen starts exist to fly them from.

## `--perturb`: change the live world after the goal picture

The goal picture is always the unchanged scene; the change applies to the flight only. Every
draw is seeded per scene and start (`--perturb-seed`), so every method flies the same world.

| mode | what changes |
| --- | --- |
| `none` | nothing (default). |
| `light` | the lights scaled by 0.3-2.0 and set to a colour temperature of 2700-6500 K. |
| `distractors` | 2-5 extra objects on the floor in view, at least 12 cm from everything. |
| `occluder` | a grey box between the camera and the scene, covering 10-18 % of the picture width. |
| `noise` | a camera noise model on every live frame (shot and read noise, brightness, contrast, white balance). |
| `combined` | all four of the above at once. |
| `ground:<name>` | another floor texture: `wood`, `wood3`, `concrete`, `objects1`, `laminate`, `plywood`, or a path. Rooms and instance scenes. |
| `swap:<instance_id>` | the scene's first object replaced by another object of the library at the same place and turn, sized by its category's rule. |
| `swap:@<name>` | the scene's object replaced by one of its named live objects (the instance scenes, below). |
| `a+b` | several at once, e.g. `swap:@gen1+ground:concrete`. |

The instance scenes' named live objects (`live:` in `configs/scenes.yaml`):

| name | in `inst_<kind>` | in `inst_<kind>_rev` |
| --- | --- | --- |
| `same` | the goal object itself (a recorded no-op, the check) | the same |
| `same_small` / `same_big` | the goal object at the other size | the goal object at the big size |
| `other_small` / `other_big` | another object of the kind at its real small size | another object at the big size |
| `mine1`, `mine2` (mug: `mine3`) | other objects of the kind, scaled to the goal object's size | - |
| `gen1` ... `gen5` / `gen6` | more objects of the kind at the goal's size (GEN-VS's held-out set) | - |

With `swap:@<name>` every record is scored twice: against the goal pose and against the
**size-matched pose** (`*_matched` fields, `success_matched`) - the goal pose scaled about the
object's floor centre by the size ratio, the pose from which a scaled copy of the goal object
would look exactly like the goal picture. For a live object of the goal's size it is the goal
pose.

## `--mask` and `--look`

`--mask none` (default) \| `objects` (keep only matches whose goal point lies on the goal
picture's objects, from the goal depth; rooms and instance scenes) \| `<file.png>` (your own
mask on the goal picture, grey > 127 = keep). `--look isaac_fit_v1` (default: a light, shadow
map and tone curve fitted to Isaac Sim's renders of the same scenes) \| `raw` (MuJoCo's plain
shading).

## Flags

### `python -m vsbench fly`

Its own three flags; every other flag goes to `python -m vsbench.world.fly run` (next table).

<!-- flags:vsbench-fly -->
| flag | values | default | what it does |
| --- | --- | --- | --- |
| `--method` | METHOD ... | required | baseline names (vitvs_dinov3, artvs, sift_ransac, ...) and/or registered method names |
| `--plugin` | PLUGIN (repeatable) | - | a .py file or module that registers your method; repeatable |
| `--out` | OUT | runs | output directory (default: runs) |
<!-- /flags:vsbench-fly -->

### `python -m vsbench list`

<!-- flags:vsbench-list -->
| flag | values | default | what it does |
| --- | --- | --- | --- |
| `--plugin` | PLUGIN (repeatable) | - | a .py file or module that registers your method; repeatable |
<!-- /flags:vsbench-list -->

### `python -m vsbench.world.fly run` (also every other flag of `vsbench fly`)

Through `python -m vsbench fly`, give every name with `--method`; it sends the baselines to
`--arms` and sets `--out` itself.

<!-- flags:fly-run -->
| flag | values | default | what it does |
| --- | --- | --- | --- |
| `--scene` | SCENE | required | the scene to fly: hollywood, poster2, poster3, industrial1_drill, cluttered2 or cluttered3 (the scenes with the protocol's start sets) |
| `--arms` | ARMS ... | - | protocol arm ids |
| `--method` | METHOD ... | - | registered vsbench methods, flown through MethodArm |
| `--param` | PARAM (repeatable) | - | --method constructor argument key=value, repeatable |
| `--start-set` | START_SET | capped30 | the protocol's frozen start set: capped30 (default), wide100, narrow20 or near60 (the last two on the posters only) |
| `--limit` | N | - | fly only the first N starts of the set |
| `--start-ids` | START_IDS ... | - | fly only these start ids of the set (e.g. w3_095) |
| `--max-iter` | N | - | override episode.max_iter |
| `--device` | auto \| cuda \| cpu | auto | torch device of the encoder arms (ViT-VS, ART-VS): auto (default; CUDA when available, the CPU in station mode), cuda or cpu |
| `--out` | OUT | required | output directory (`python -m vsbench fly` defaults it to runs) |
| `--telemetry` | (switch) |  | the copied per-step telemetry (telemetry.jsonl.gz in each run dir) |
| `--telemetry-extended` | (switch) |  | episode.telemetry_extended: the per-match truth columns (gt_u, gt_v_px, gt_ok, err_patch, ...); implies --telemetry |
| `--film` | (switch) |  | the copied FilmWriter's match film |
| `--film-every` | N | 2 | --film: keep every N-th step (default 2) |
| `--film-scale` | X | 0.25 | --film: frame size relative to 1920 x 1080 (default 0.25) |
| `--mask` | MASK | none | filter the matches with a mask on the goal picture: none (default), objects (the goal's objects, from the goal depth; the copied objmask cell) or a .png file (grey > 127 = keep; vsbench.world.masks) |
| `--debug-gif` | (switch) |  | the copied debug view animated per episode: film/<start>/debug.gif (implies --telemetry) |
| `--debug-gif-every` | N | - | one debug frame every N steps (default: at most --debug-gif-frames) |
| `--debug-gif-frames` | N | 90 | --debug-gif: at most this many frames (default 90) |
| `--samples` | N | 4 | MuJoCo multisampling (colour only) |
| `--look` | isaac_fit_v1 \| raw | - | the picture: isaac_fit_v1 (default) or raw (vsbench.world.look) |
| `--perturb` | PERTURB | none | none \| light \| distractors \| occluder \| noise \| combined \| ground:<name> \| swap:<id> \| swap:@<name> \| a+b (vsbench.vs.bench.perturb), applied after every episode's goal capture |
| `--perturb-seed` | N | 20260918 | seed of the perturbation draws (default 20260918; one draw per scene and start, the same for every method) |
| `--threads` | N | - | cap torch / BLAS / Mesa threads (default: none; 2 in station mode) |
| `--allow-gpu` | (switch) |  | station mode only: lift its GPU refusals |
<!-- /flags:fly-run -->

### `python -m vsbench.world.fly summary`

Per cell: how many flights reached each success rule, at the stop and at the cap.

<!-- flags:fly-summary -->
| flag | values | default | what it does |
| --- | --- | --- | --- |
| `--out` | OUT | required | the --out directory of one or more runs (reads its *.jsonl) |
| `--json` | JSON | - | also write the table to this .json file |
<!-- /flags:fly-summary -->

Before `run` or `summary`:

<!-- flags:fly-global -->
| flag | values | default | what it does |
| --- | --- | --- | --- |
| `--protocol` | PROTOCOL | - | default: vsbench/configs/protocol.yaml |
<!-- /flags:fly-global -->

### `python -m vsbench.world.check_baselines`

Per scene and baseline: an identity check (the camera at the goal must not move), a near flight
and `--far N` far flights.

<!-- flags:check -->
| flag | values | default | what it does |
| --- | --- | --- | --- |
| `--scenes` | SCENES ... | hollywood poster2 poster3 industrial1_drill cluttered2 cluttered3 | the scenes to check, one after the other (default: the six benchmark scenes) |
| `--arms` | ARMS ... | - | protocol arm ids (default: sift_ransac orb_ransac vitvs_dinov3 artvs) |
| `--near-steps` | N | 200 | steps of the near check (default 200) |
| `--far` | N | 0 | fly the first N far starts to the cap |
| `--far-set` | FAR_SET | capped30 | which start set the far starts come from (default capped30; wide100 has every roll; simple5, simple10, t1near3 on the inst_* scenes) |
| `--out` | OUT | required | output directory (check.json, records.jsonl, frames, films) |
| `--film` | (switch) |  | a match video per episode: <out>/films/<scene>_<arm>_<start>/episode.mp4 |
| `--film-every` | N | 1 | keep every n-th step |
| `--perturb` | PERTURB | none | none \| light \| distractors \| occluder \| noise \| combined \| swap:<id> \| swap:@<live name> \| ground:<name>, or a `+` chain - applied after the goal capture of every near/far episode |
| `--perturb-seed` | N | 20260918 | seed of the perturbation draws (default 20260918) |
| `--budget` | ARM=STEPS (repeatable) | - | one arm's own step budget for its near AND far starts (repeatable) |
| `--outline-iou` | (switch) |  | record the object-outline overlap with the goal picture at every recorded stop and at the end (episode.outline_iou) |
| `--telemetry` | (switch) |  | the copied debug log per episode (<out>/films/<scene>_<arm>_<start>/) |
| `--clean-film` | (switch) |  | a clean half-size live film per episode; needs --telemetry |
| `--arm-plugin` | MODULE | - | an importable module stating ARM_IDS, build() and describe() (vsbench.world.method_arm flies the registered vsbench methods) |
| `--mask` | MASK | none | none \| objects \| <file.png>: filter every arm's matches with a goal mask (vsbench.world.masks; as fly run --mask) |
| `--look` | isaac_fit_v1 \| raw | - | the picture (vsbench.world.look; default isaac_fit_v1) |
| `--device` | auto \| cuda \| cpu | auto | torch device of the encoder arms: auto (default; CUDA when available, the CPU in station mode), cuda or cpu |
| `--samples` | N | 4 | MuJoCo multisampling (colour only) |
| `--frames` | full \| preview \| none | full | full: vs-sim's goal frame, goal depth and first live frame; preview: quarter-size PNGs, depth md5 only; none |
| `--threads` | N | - | cap torch / BLAS / Mesa threads (default: none; 2 in station mode) |
| `--allow-gpu` | (switch) |  | station mode only: lift its GPU refusals |
<!-- /flags:check -->

### `python -m vsbench.world.check_baselines summary`

<!-- flags:check-summary -->
| flag | values | default | what it does |
| --- | --- | --- | --- |
| `--runs` | RUNS ... | required | check.json files or their directories |
| `--out` | OUT | required | where summary.json and summary_table.md go |
<!-- /flags:check-summary -->
