# PORT_PLAN — vs-sim's capabilities, objects, depth camera and the ViT-VS / ART-VS baselines on the MuJoCo simulator

*Planner: Fable, 2026-10-22, for the owner's ask of the same day ("port the main capabilities
and baselines art-vs and vit-vs from our current isaac sim setup to there too ... the objects
... the depth cameras ... without stopping or hindering the current eval run"). Builders: Opus,
one unit at a time (§7). Reviewer: Fable. Every fact below was read from code or checked with a
read-only command on the station on 2026-10-22; nothing was guessed. The plan commits nobody to
anything in `~/runpod_dev/vs-sim`: that repository is frozen while the T1 Stage 2 evaluation flies.*

---

## 0. Scope, in one paragraph

**Source** — `~/runpod_dev/vs-sim` (package `vssim`): the Isaac Sim 5.1 benchmark. Its contract
is `vssim.scene.Scene` — four verbs (`reset(pose)`, `render() -> (rgb, depth, pose)`,
`apply(twist, dt)`) and four facts (`intrinsics`, `goal_pose`, `working_distance`, `pose`).
Everything that matters for this port is *behind* that contract and is pure Python:
the published baselines (`vssim.baselines`: ViT-VS, ART-VS, CNS v1, DINOBot, classical), the
episode loop and records (`vssim.bench`), the six scenes' geometry (`configs/scenes.yaml`,
`configs/layouts/*.json`, `vssim.scenes`), the frozen starts (`benchmark/starts/*.json`,
md5-pinned, `vssim.benchmark`), the camera (`configs/camera.yaml`, `vssim.camera`). The
package core imports numpy and PyYAML only; Isaac is imported inside methods; torch lives in
`vssim.baselines` / `vssim.bench` only (`tests/fast/test_package.py` asserts both).

**Target** — `~/genvs_ws/vs_sim_light` (package `vsbench`, local git `main` @ `39522f3`): a
MuJoCo renderer behind a `Renderer` protocol (`render(T_wc) -> (rgb, depth)`), a free-flying
SE(3) camera, a servo loop with a method registry (`VSMethod`, `@register`), episode logs and
replay videos, and the gates that prove each piece against analytic math. Steps 1–3 are done
and green (`results/STEP3_REPORT.md`: 13/13 gates, `sift_ibvs` 10/10 starts under 1 mm on the
poster).

**What "port" means here** — make the MuJoCo world *a second world behind vs-sim's contract*,
so the published baselines, the bench, the six scenes and their frozen starts fly on it
unchanged; and make the converted object library and the D435i depth camera available to both
the bench and vsbench's own student-facing loop. Not in scope: any claim that a MuJoCo number
replaces an Isaac number (§6), any edit to `vs-sim` (§8, D-2), anything that touches the GPU.

---

## 1. Facts verified on the station (what the builders will find)

| what | where / value | how checked |
| --- | --- | --- |
| Live eval | `docker` container `isaac` (up 6 days); locks `~/genvs_ws/night_{flight,gpu,isaac}.lock`; GPU 15.6 / 24.6 GB used; load 22 on 16 threads (AMD Ryzen 7 5800X, 62 GB RAM) | `docker ps`, `nvidia-smi`, `uptime` |
| Host Python | `/usr/bin/python3` 3.12.3 — **no** torch, pxr, mujoco, transformers anywhere on the host (`find / -name torch -path '*site-packages*'` → nothing). The only torch is inside the Isaac container (off limits). | imports, find |
| vsbench venv | `~/genvs_ws/vs_sim_light/.venv` (py 3.12.3): mujoco 3.13.0, numpy 2.5.3, opencv-python-headless 5.0.0, Pillow, imageio-ffmpeg. No torch/transformers/usd-core/trimesh. 2.0 GB. | imports |
| Software GL | Mesa EGL vendor json `/usr/share/glvnd/egl_vendor.d/50_mesa.json` present; **no libOSMesa** (`ldconfig -p` has none). So `VSBENCH_SOFTWARE_GL=1` + `mujoco-egl` (llvmpipe) is the CPU path, not `mujoco-osmesa`. The bake-off ran exactly this on this box: EGL/llvmpipe and OSMesa were bit-identical. | ls, ldconfig |
| Weights (never download) | DINOv3 ViT-S/16: `~/genvs_ws/cache/hf/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/` (165 MB, `model.safetensors` + configs) → `HF_HOME=~/genvs_ws/cache/hf`. DINOv2 ViT-S/14: `~/genvs_ws/torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth` (85 MB) + code `~/genvs_ws/torch_home/hub/facebookresearch_dinov2_main/` → `TORCH_HOME=~/genvs_ws/torch_home`. Also there: DINO v1 ViT-S/8 (83 MB, DINOBot), RoMa, LightGlue, DINOv3-B. The host-side `~/.cache/huggingface` and `~/.cache/torch` are **empty**. | find, du |
| Object USDs | 323 library entries (`vs-sim/assets/asset_library_v2.json`): 223 under `~/genvs_ws/genvs-sim/experiments/ingest/usd/<cat>/<id>/<id>.usd` (18 GB tree), 94 under `~/genvs_ws/genvs-sim/assets/gso_usd/<cat>/<id>/<id>.usd` (2.6 GB), 6 Nucleus (`/Isaac/Props/YCB/Axis_Aligned/*.usd`, **not on disk**). **0 of the 317 local files are missing.** Every file is binary `PXR-USDC`, with `metersPerUnit`/`upAxis` metadata, a `UsdPreviewSurface` + `UsdUVTexture` material and one texture in `textures/*.png|jpg` beside it (805 png + 300 jpg in the trees). Sources: gso 105, objaverse 103, oo3d 56, housecat6d 39, handal 14, isaac_native 6. Categories: shoe 103, bottle 45, teapot 38, mug 35, hammer 31, drill 25, tv_remote 12, toy_car 10, drinking_glass 6, wine_bottle 6, tube 4, stapler 4, can 4. | python over the json, `head -c 8`, `strings` |
| YCB scans (rooms) | `~/genvs_ws/assets/ycb/<id>/` (89 dirs, 2.0 GB): the Isaac USD (`usd/<id>.usd`, **OmniPBR MDL** material) **and** the original `google_16k/textured.obj` + `textured.mtl` + `texture_map.png` — a plain textured OBJ MuJoCo can load directly. `ycb_library.json` has 82 entries. | ls |
| Scene textures | `~/genvs_ws/assets/{hollywood/resized.png 1500×1125 RGBA, poster2/poster2.png 375×500, poster3/poster3.jpg 3840×2810, industrial1/ground.png 1024², cluttered2/ground.png 4096², cluttered3/ground.png 4096²}` (the yaml's `/workspace/...` is `~/genvs_ws/...` on the host) | PIL |
| Camera | `configs/camera.yaml`: 1920×1080, 42° vertical FoV, square 20.955 mm aperture → `fx = fy = 1406.75`, principal point `(960, 540)`; `vssim.scene.D435I` pins the same. Isaac camera: bare `UsdGeom.Camera`, clipping 0.001–1000 m. Depth annotator: **`distance_to_image_plane`** (= camera-frame z, metres — the same quantity a z-buffer gives). No depth noise model anywhere in vs-sim (the `noise` perturbation is RGB-only). | camera.py, isaac_scene.py:704-729 |
| Isaac lighting | dome light intensity 3000, colour (0.2, 0.2, 0.2); distant light 3500, angle 0.53°, rotateXYZ (−30, 0, 30). | isaac_scene.py:704-710 |
| Poster geometry | quad `[-0.4, 0.4] × [-0.3, 0.3]` at `z = 0`, texture row 0 at world +Y; backdrop 4 × 4 m grey 0.5 at `z = −0.02`; goal `(0, 0, 0.4341815108)`, `R = diag(1, −1, −1)`. | isaac_scene.py:816-897, scenes.py |
| Room geometry | ground: 2.4 m textured quad at `z = 0`, single span; objects: wrapper `/World/Obj_i` = `Translate(pos) · Orient(quat) · Scale(s)` applied to the asset's native coordinates (`_place_pose`), `pos`/`quat_wxyz` frozen in `configs/layouts/<scene>.json`; `s = 1` for `ycb/...` ids, `target_for(category)` longest-extent rule for curated ids (drill → 0.25 m); goal = `scene_view_goal_pose(cam, bbox_min, bbox_max, frame_fill 0.85)` from the **measured** bbox of everything placed. Frozen start files carry the resulting goal: `industrial1_drill` d = 0.4422820342; `cluttered2` d = 1.1689734192, look_at `(0.0469, 0.1515, 0.2424)`; `cluttered3` d = 1.2853190576. `check_scene_goal` tolerance: 5e-4 m / 0.05° / 1e-3 m on distance. | isaac_scene.py:900-1125, layouts, starts files |
| What an arm needs | `Arm.reset(goal_rgb, goal_depth, goal_mask)` then `Arm.step(rgb, depth) -> (twist, debug, matches)`; depth is read at the current pixels by truncation (`depth_at`), invalid (≤0, >10 m, NaN, inf) → 0.5 m. Intrinsics go in at `build_arm(cfg, intrinsics=scene.intrinsics, ...)`. No arm ever sees a pose. | arms.py:250-480 |
| The loop | `vssim.bench.runner.run_episode(scene, arm, start, config, *, starts_payload=..., telemetry=..., on_step=..., perturbation=..., debug_log=...)` → record dict. It calls only the four verbs + four facts, plus optional extras (`gt_correspondence`, `goal_capture_record`) through `getattr`. Integration: `integrate_optical_velocity` (`pos += R v dt`, `R ← R·Rodrigues(ŵ, |w| dt)`), dt = 1/60, gain 2.0, cap 600. | runner.py:494-612 |
| Where the drivers hard-wire Isaac | `scripts/fly.py:149`, `scripts/check_baselines.py:438`, `examples/fly_benchmark.py:30` all `from vssim.isaac_scene import IsaacScene`. The library never does. | grep |
| vsbench ↔ vssim conventions | identical optical frame (x right, y down, z forward); vsbench `T_wc` = `vssim.scene.pose_to_T(Pose)`. **Different** integration (`T @ exp_se3(twist·dt)` vs the two-step rule above) and **different pixel-centre convention** (§4). vsbench `Plane` texture row 0 at local +y = vssim mock's convention. | se3.py, motion.py, scene.py |

---

## 2. Architecture — options and one recommendation

### The options

**(a) Depend on `vssim` as a library; add a MuJoCo `Scene` that satisfies its contract.**
vsbench gains one module that builds a MuJoCo world from vs-sim's own scene specs, layouts and
camera, and exposes `reset/render/apply + intrinsics/goal_pose/working_distance/pose`. The
baselines, `build_arm`, `run_episode`, the records, the scoring, the perturbation framework,
the frozen starts and the md5 pins are **imported, not copied**. One home for every piece of
published-baseline code; the parity fixtures and the fast tier travel with it.

**(b) Copy the baseline code into `vsbench`.** Six to eight thousand lines
(`features.py` 1132, `match.py` 1283, `tiling.py` 477, `control.py` 1465, `prealign.py` 455,
`arms.py` 700, `config.py` 335, plus the fixtures) re-homed under a different method
interface. Two homes for a "run as published" method is exactly what `vs-sim/CLAUDE.md` rule 7
and the workspace's one-home rule forbid; every future fix to a baseline would have to be made
twice, and the "as published" claim would rest on which copy was flown.

**(c) Hybrid: (a) for the benchmark side, plus a thin adapter so a vs-sim arm also appears as
a `vsbench` method.** `vsbench`'s registry gets `vitvs_dinov3`, `artvs`, ... as
`VSMethod` classes that *wrap* a `vssim.bench.arms.Arm` (configure → intrinsics, `set_target`
→ `arm.reset`, `get_control_rate` → `arm.step`). No baseline code is copied; only the glue.

### Recommendation: **(c), built as (a) first.**

1. The one-home rule and the "as published" parity decide against (b) outright.
2. (a) costs almost nothing: `vssim`'s core is numpy + PyYAML, Isaac is lazy, and `run_episode`
   touches a scene only through the contract. The MuJoCo scene is **one new class** plus a
   scene builder; nothing in vs-sim needs to change for it to fly.
3. The adapter in (c) is the student-facing half the owner asked for ("light weight simulator
   with same capabilities"): `python -m vsbench.demo --method vitvs_dinov3` on a laptop. It is
   ~150 lines and comes after the benchmark side is proven (unit U6).
4. **Where the new class lives**: in `vsbench` (`vsbench/vssim_bridge/`), *for now*. The
   contract-respecting place for a second world is `vssim` itself (its CLAUDE.md: "Add to a
   world freely"), next to `MockPlaneScene` and `IsaacScene` — but nobody edits vs-sim while
   the eval flies, and whether a MuJoCo world belongs in the benchmark repo at all is the
   owner's call (D-2 in §8). Building it as a bridge module with no `vsbench`-internal
   imports except the renderer makes a later move a file move.

### How `vssim` gets into the vsbench environment without touching `~/runpod_dev/vs-sim`

- `git clone ~/runpod_dev/vs-sim ~/genvs_ws/vs_sim_light/external/vs-sim` — a clone from a
  local path only **reads** the source repository. Check out the commit that is HEAD today
  and write it to `external/VSSIM_COMMIT`; `external/` is git-ignored in vsbench, the pin is
  committed.
- `pip install --no-deps -e external/vs-sim` into `.venv` (editable so the pin can be moved
  with `git -C external/vs-sim checkout <sha>`; the egg-info lands in *our* clone). `--no-deps`
  because vs-sim's `baselines` extra names `opencv-python`, which would be installed beside
  the venv's `opencv-python-headless` and shadow it; the deps are installed by hand (U0).
- Never `pip install -e ~/runpod_dev/vs-sim`: an editable install writes `vssim.egg-info/`
  into the read-only checkout.

### What the bench driver needs that vs-sim does not offer today

`scripts/fly.py` has no `--world` switch; it constructs `IsaacScene` directly. So the bridge
ships its own small driver (`python -m vsbench.vssim_bridge.fly ...`, U2) that does what
`fly.py run` does — load the arm config, `build_arm`, iterate `benchmark.select_starts`,
`run_episode`, write jsonl — on the MuJoCo scene. A `--world {isaac,mujoco,mock}` factory in
`fly.py` / `check_baselines.py` is the clean end state and is listed as a vs-sim change for
the owner to approve after the eval (D-2).

---

## 3. Capability inventory — what vs-sim does, and what happens to each

Legend: **now** = in this port's units (§7); **later** = after the first milestone, needs
either a decision or a scene verb we add; **n/a** = Isaac-specific or not meaningful in MuJoCo.

| capability | where in vs-sim | MuJoCo | why / how |
| --- | --- | --- | --- |
| The `Scene` contract (4 verbs, 4 facts) | `vssim/scene.py` | **now** (U1) | `MujocoScene` wraps `vsbench.render.mujoco_backend.MujocoRenderer`; `apply` uses `vssim.scene.integrate_optical_velocity` (the benchmark's rule, not vsbench's `exp_se3`). |
| D435i camera (intrinsics, 1920×1080, depth in metres) | `configs/camera.yaml`, `vssim/camera.py` | **now** (U1) | `CameraConfig().intrinsics()` → `CameraModel`; §4 for the half-pixel convention. |
| Goal frame captured once, served from `render()` at the goal pose; `goal_capture_record()` | `isaac_scene.py:455-560` | **now** (U1) | MuJoCo is frame-deterministic (bake-off gate), so one capture is exact; record `n_captures: 1`, `render_mode: "mujoco-llvmpipe"` / `"mujoco-egl-gpu"` so no MuJoCo record can be mistaken for an RTX one. |
| `gt_correspondence(goal_px, cur_pose)` (occlusion-tested truth for telemetry) | `isaac_scene.py:659`, `vssim/labels.py` | **now** (U1) | pure numpy on the cached goal depth + last depth — same code, same call. |
| Three poster scenes (plate, backdrop, goal, frozen starts) | `scenes.yaml`, `scenes.py::poster_goal_pose` | **now** (U1) | exact geometry; `check_scene_goal` must pass by construction. |
| Three rooms (ground, baked layouts, bbox-derived goal) | `scenes.yaml`, `layouts/*.json`, `isaac_scene.py::place_objects` | **now** (U3, U4) | needs the converted meshes; goal re-derived from *our* bbox and checked against the frozen file (D-4 says what happens if it drifts). |
| The frozen starts and start sets (`capped30`, `wide100`, `narrow20`, `near60`, instance `simple5/10`, `t1near*`), md5 pins | `vssim/benchmark.py`, `benchmark/starts/` | **now** (import) | nothing to port: poses are poses. |
| The published baselines: ViT-VS (DINOv3 / published DINOv2 chain), ART-VS (36-tile latching switch), the 4-way pre-align, classical SIFT/ORB/AKAZE ± RANSAC | `vssim/baselines/*`, `configs/arms/*.yaml` | **now** (U2, U5) on CPU | `matcher.device: cpu` override; the fast-tier fixtures prove the code in the new env before any flight. |
| CNS v1, DINOBot, LightGlue, RoMa, ViSP arms | same | **later** | need `$CNS_ROOT` + torch_geometric / scikit-learn / ViSP bindings; nothing MuJoCo-specific — they fly the day their deps are in the venv. Not in the owner's ask. |
| ART-VS object mode `artvs_roi_sim` (GroundingDINO, LightTrack, LangSAM) | `vssim/baselines/objroi.py` | **later** | heavy detector weights, `$LIGHTTRACK_ROOT`; and it needs the instance-suite scenes. Not in the ask. |
| `objmask` arms (goal mask from goal depth vs the `z = 0` ground) | `vssim/labels.py::object_mask_from_depth`, `runner.py` | **now** (free with U4) | our ground is also at `z = 0`; the runner computes it. |
| The episode loop, telemetry, divergence abort, three success rules, stop rules recorded, records + run sidecar | `vssim/bench/runner.py`, `scoring.py`, `run.py` | **now** (import) | `open_run` refuses a dirty tree: the driver passes `allow_dirty=True` and writes the vsbench commit + the `VSSIM_COMMIT` pin into the sidecar. |
| Films (`FilmWriter` via `on_step`), clean film, debug view | `vssim/bench/films.py`, `debugview.py`, `debuglog.py` | **now** (import; smoke in U9) | pure numpy/OpenCV; `ffmpeg` binary: use `imageio_ffmpeg.get_ffmpeg_exe()` if `shutil.which("ffmpeg")` is empty. |
| Perturbation `noise` (RGB Poisson–Gaussian + brightness/contrast/WB) | `perturb.py::apply_noise` | **now** (import) | numpy on the live frame; world-agnostic. |
| Perturbation `ground:<name>` (floor re-texture) | needs `scene.build_poster(...)` | **later** (U8) | MuJoCo textures are baked at `MjSpec.compile()`; a re-texture = rebuild the model (~1 s for a room). Implement as a scene verb that recompiles; fine per episode. |
| Perturbation `occluder` (grey cube), `distractors` (2–5 extra YCB), `swap:` (instance swap) | needs `put_box`, `place_extra_objects`, `remove_prims`, `set_prim_visible`, `object_world_bbox`, `object_pose` | **later** (U8) | same recompile path; `swap` also needs the converted library (U7). All of them are scene verbs the bridge can add without touching vs-sim. |
| Perturbation `light` (dome + sun scaled, colour temperature) | `set_lighting`, `lighting_snapshot`, `restore_lighting` | **later** (U8) | MuJoCo has `model.light_diffuse/ambient` live (no recompile); the mapping from Isaac's intensities to MuJoCo's is a tone fit, not physics — state it on the record. |
| Scatter + gravity settle to **bake** a new layout | `vssim/bake.py` (PhysX) | **n/a** | the rule "a re-settle may never define a goal" means we only *load* layouts. MuJoCo could settle, but it would be a new layout, i.e. a new scene — out of scope. |
| Semantic/instance segmentation annotator | `isaac_scene.py:1263`, `1471` | **later** | MuJoCo's segmentation render (`mjRND_SEGMENT`, already used for the depth pass) gives geom ids for free; only needed by the object-mode arms. |
| Goal-capture warm-up (8 RTX captures, keep last) | `scenes.yaml::scene`, `_capture_goal` | **n/a** | RTX history artefact; MuJoCo is deterministic (recorded as `n_captures: 1`). |
| RTX non-determinism bookkeeping (`goal_frame_md5`, settle series) | `isaac_scene.py` | **now** (trivial) | md5 of the one capture. |
| Depth noise / holes model | — | **n/a in vs-sim** | vs-sim has none (depth is true metres). Adding a D435i noise model would be new science (D-5). |
| Scale jitter, `clear_scene_prims`, Nucleus assets | `isaac_scene.py` | **n/a** / later | jitter is a dataset-generation leftover; the 6 Nucleus objects are not on disk (D-6). |
| GPU rendering of the same MuJoCo scene | `mujoco-egl` without `VSBENCH_SOFTWARE_GL` | **later, never during the eval** | bake-off: GPU and CPU MuJoCo renders agree to 44.6 dB; 1140 fps vs 100 fps at 640×480. |

---

## 4. The depth camera: D435i intrinsics into MuJoCo

**Intrinsics.** `vssim.camera.CameraConfig().intrinsics()` → `Intrinsics(fx=fy=1406.748,
cx=960, cy=540, 1920×1080)`. The MuJoCo backend already takes pixel intrinsics
(`c.focal_pixel = (fx, fy)`, `c.principal_pixel = (W/2 − cx − 0.5, H/2 − cy − 0.5)`,
`c.sensor_size` only sets the ratio) and renders off-screen at `offwidth ≥ W`, `offheight ≥ H`
(1920×1080 is within llvmpipe's limits; the bake-off rendered 1440×1080).

**The half-pixel convention — decide it, pin it, test it.** Three conventions meet here:

| | ray through pixel column `u` (integer index) | principal point |
| --- | --- | --- |
| Isaac (USD camera, aperture centred on the window) | `((u + 0.5) − 960) / fx` | geometric image centre |
| `vssim.scene.MockPlaneScene` | `(u − 960) / fx` | pixel 960's centre — **half a pixel right/down of Isaac** |
| `vsbench.CameraModel` (OpenCV: integer index = pixel centre) | `(u − cx) / fx` | whatever `cx` says; centred image = `cx = 959.5` |

Recommendation: the MuJoCo camera is built with **`cx = 959.5, cy = 539.5`** (Isaac's
geometry — the benchmark), while `MujocoScene.intrinsics` reports the frozen `D435I`
`(960, 540)` that every baseline, record and start file was made with, exactly as Isaac does.
The mock's half-pixel offset against Isaac was already measured and accepted ("pixel-aligned
to the pixel", 2026-09-17); at 0.43 m half a pixel is 0.15 mm, far under any success rule.
The gate below makes the choice visible rather than silent. (Alternative: `cx = 960`, matching
the mock, if the owner prefers mock parity over Isaac parity — D-3.)

**Depth.** MuJoCo's reversed-Z buffer (`mjDEPTH_ZEROFAR`) linearised to camera-frame z is
the same quantity as Isaac's `distance_to_image_plane`. vsbench already reads depth from a
second, non-multisampled render (multisampling moved the depth sample off the pixel centre by
up to 3.9 mm on a 60° plane — documented in `mujoco_backend.py`). Nothing hit → `0.0`, which
`vssim.baselines.control.clean_depth` maps to the 0.5 m fallback, the same branch Isaac's
background (inf / > 10 m) takes. Clipping: keep vsbench's gate-tested `near = 0.05, far = 10`
(scenes are flown at 0.43–1.29 m, wide starts up to ~0.5 m further; Isaac's 0.001–1000 is
irrelevant to geometry). Noise: none, as in vs-sim (D-5).

**Gates (U1), all CPU, all against analytic truth that vsbench already has:**

1. *Projection* — 13 unlit marker spheres (`stdscenes.marker_scene`) through the D435i camera
   at 1920×1080: detected centroids vs `vsbench.reference.project` with `cx = 959.5`:
   median ≤ 0.15 px, max ≤ 0.5 px (the existing `test_projection` rule, at the new resolution).
2. *Depth* — the plane gate (`tests/test_depth.py`) at 1920×1080 from the three poster goal
   poses and six `narrow` starts: median |z − analytic| ≤ 1 mm, p99 ≤ 3 mm (bake-off rule);
   report the numbers at 0.43 m and 1.29 m.
3. *Convention* — render the hollywood plate with `MockPlaneScene(texture=hollywood,
   render_scale=1)` and with `MujocoScene("hollywood")` from the goal pose and three `near60`
   starts; `cv2.phaseCorrelate` on the grey images must return the predicted (0.5, 0.5) px
   shift within 0.1 px at every pose, and the green-channel gradient correlation ≥ 0.8 (the
   mock-vs-Isaac figure was 0.81).
4. *Contract* — `isinstance(scene, vssim.scene.Scene)`; `vssim.benchmark.check_scene_goal`
   passes on all three posters (goal pose from `poster_goal_pose`, by construction).

---

## 5. The 323 objects: USD → MuJoCo

**Two input kinds, two readers, one cache format.**

- *YCB (the two cluttered rooms)*: skip the USD (OmniPBR/MDL, which usd-core cannot resolve
  anyway) and read `google_16k/textured.obj` + `texture_map.png`. A 60-line numpy OBJ reader
  (`v`, `vt`, `f a/b` with distinct position/uv indices → split vertices per `(v, vt)` pair);
  no `trimesh` dependency. **Risk to test**: the Isaac USD was made from this OBJ by
  `scripts/convert_ycb.py` (not on the host; the container's copy is off limits), so its axes
  may differ from the OBJ's. The bbox-goal gate below catches a mismatch; the fallback is the
  usd-core path on the YCB USD (its mesh is readable even though its material is MDL; the
  texture is `usd/textures/texture_map.png`).
- *The 317 curated objects*: `usd-core` (pip, ~35 MB wheel, pure CPU, no Isaac; verify the
  cp312 wheel exists with `pip index versions usd-core` before U3 — 24.11+ ships it). Open the
  stage, walk `UsdGeom.Mesh` prims, take `points`, `faceVertexCounts/Indices` (fan-triangulate
  n-gons), the `st` primvar (`faceVarying` or `vertex` interpolation → split to one uv per
  corner), apply `ComputeLocalToWorldTransform(0)` **of the asset layer only**, and
  **ignore** `metersPerUnit` / `upAxis`: USD references do not auto-scale or re-orient a
  referenced layer and Isaac did not either (the library note on `SM_Mug_*` says as much), so
  Isaac saw exactly the layer's composed points. Texture: follow `UsdUVTexture.inputs:file`;
  if unresolved, the single file under `textures/`. UV origin: USD and OBJ are bottom-left,
  vsbench `Mesh.uvs` wants `v = 0` at the top row → `v' = 1 − v`.
- *What is lost*: PBR parameters (roughness/metallic), MDL materials, normals maps, vertex
  colours; MuJoCo gets diffuse colour + one RGB texture, flat shading. Scale/orientation
  conventions are **not** lost: they are reproduced by re-running vs-sim's own rules
  (`vssim.assets.target_for`, `vssim.scenes.item_quat/canonical_quat/long_axis_yaw`) on the
  bbox we measure from the converted points.

**Cache layout** (outside the repo, like `/workspace` is for Isaac; D-7):
```
~/genvs_ws/vsbench_assets/
  objects/<instance_id>/mesh.npz        float32 V (N,3) in the asset's native frame & units,
                                        float32 UV (N,2) top-left origin, int32 F (M,3)
  objects/<instance_id>/texture.png     RGB, longest side capped at 4096 (llvmpipe memory)
  objects/<instance_id>/meta.json       source path + md5 + mtime, native bbox min/max,
                                        n_vertices, n_faces, texture source + size, converter
                                        version, usd-core version
  weights/                              a COPY of the two weight folders (250 MB) so the shared
                                        caches are never opened for writing (HF lock files)
  INDEX.json                            instance_id -> status (ok / failed:<reason>)
```
Disk: usdc averages ~20 MB per object; a float32 `npz` (zip-compressed) is ~0.3× of it,
textures copied ≈ 5–10 MB each. Estimate **4–6 GB for all 317**, hard cap 10 GB (`INDEX.json`
records the running total; the converter refuses to exceed the cap). Only the ~10 objects the
six scenes need are converted in U3 (< 200 MB); the full library is U7, run under `nice -n 19`
with `OMP_NUM_THREADS=1`, one process, in the background (~0.5–3 s per object → under an hour).

**The CPU-only parity check for a converted object** (no Isaac render is available on the
host without reading `vssim_out`, which is off limits; and none is needed):

- *The bbox-goal gate (U3)*: for each room, place the converted objects by the frozen layout
  (`T(pos)·R(quat)·S(s)`), compute the world bbox of all placed points, call
  `vssim.scenes.scene_view_goal_pose(cam, bbox_min, bbox_max, frame_fill=0.85)` and compare
  to the frozen start file's `goal` block with `vssim.starts.check_goal_match`'s own
  tolerances (5e-4 m position / 0.05° / 1e-3 m distance). This tests mesh units, axes, the
  scale rule and the layout semantics at once: `industrial1_drill` must give 0.4422820342 m,
  `cluttered2` 1.1689734192 m with look_at `(0.0469, 0.1515, 0.2424)` (the 0.2424 is the
  upright pitcher's top), `cluttered3` 1.2853190576 m. Isaac measured its bbox with USD's
  `ComputeWorldBound`, which may use authored `extent`s a hair looser than the points; if the
  gate misses by ≤ 2 mm, report it and apply D-4.
- *Silhouette check against Isaac (later, when the GPU is free)*: `vs-sim/scripts/render_goal.py`
  under `$ISAAC_PY` writes a goal PNG + depth per scene; the bridge gets a
  `tools/compare_goal.py` that overlays the MuJoCo goal render on it and reports the IoU of the
  depth-derived object outline (`vssim.labels.object_mask_from_depth`) and the per-object
  centroid offset in px. Pass bar: outline IoU ≥ 0.95, centroid ≤ 2 px. This is the only step
  in the plan that needs Isaac, and it is explicitly *after* the eval.

---

## 6. The six benchmark scenes in MuJoCo, and what the comparison means

**Rebuild rules (U1 posters, U4 rooms)** — `vsbench/vssim_bridge/build.py::scene_spec_for(name)`
reads `vssim.scenes.load_scenes()` and returns a `vsbench.scene.SceneSpec`:

- poster: `Plane(I, (0.8, 0.6), Material(texture=<png as RGB>, unlit=False))` at `z = 0`,
  backdrop `Plane(z = −0.02, (4, 4), grey 0.5)`; texture paths rewritten `/workspace/...` →
  `~/genvs_ws/...` (a `VSBENCH_WORKSPACE_ROOT` setting, default `~/genvs_ws`). Goal from
  `poster_goal_pose` — no measurement involved.
- room: `Plane(I, (2.4, 2.4), ground texture)` (single span: `uv_tiles = size/tile_period = 1`
  for all three), one `Mesh` per layout entry from the cache, pose `T(pos)·R(quat)`, vertices
  pre-scaled by `s`. Goal from the measured bbox (§5 gate), then `check_scene_goal`.
- lighting: ambient 0.2 (the dome's grey) + one directional `Light` whose direction is the
  USD distant light's `−Z` axis rotated by `Rx(−30°)·Ry(0)·Rz(30°)` (compute, don't guess:
  `vssim.scenes.axis_angle_quat` and friends are available), `cast_shadow=True`, intensity
  fitted once so the hollywood goal render's mean grey matches the mock's tone fit — stated in
  `meta`, never claimed as photometric parity.

**Starts** — `vssim.benchmark.select_starts(scene, set)` → `Pose.from_start(s)` → `scene.reset`.
Identical poses, identical draw seeds (`episode_seed(scene, start_id, seed)`), identical arms.

**Is "same starts, different renderer" meaningful?** Yes, for exactly these uses:
- a **fast development loop**: a new controller or stop rule exercised against the real
  baselines, the real loop, the real records and the real success rules, on a laptop, in
  minutes, without a 60 GB simulator; convention bugs (sign, frame, pixel order, depth
  truncation) show up here just as in Isaac;
- **regression of the baseline code**: the fast tier plus a nightly MuJoCo flight of
  `capped30 × {vitvs_dinov3, artvs}` pins the published methods' behaviour on a deterministic
  renderer (RTX is not frame-deterministic, so Isaac cannot do this);
- **teaching and external users** (the stated purpose of VS-Bench).

Not meaningful for: replacing an Isaac number. Different renderer → different pixels →
different matches → different landings; shading, anti-aliasing, texture filtering and the lack
of PBR all move the ViT features. Every MuJoCo record therefore carries `render_mode:
mujoco-*` and the bridge's summary prints a banner; `vssim.bench.stats` must never pool the
two (a one-line check on `render_mode` in the bridge's summary refuses a mixed input). The
instance-suite and T1 pictures stay Isaac's.

---

## 7. Units of work (ordered; `∥` = can run in parallel with the previous unit)

Every unit runs under the station rules, scripted once in **U0** so nobody retypes them:

```
# tools/cpu_env.sh  — source it in every shell that builds or tests
export CUDA_VISIBLE_DEVICES=""            # no GPU, ever
export VSBENCH_SOFTWARE_GL=1              # Mesa llvmpipe for MuJoCo (mujoco-egl; no libOSMesa here)
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCH_NUM_THREADS=2                # the bridge calls torch.set_num_threads(2) too
export HF_HOME=~/genvs_ws/vsbench_assets/weights/hf  HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1
export TORCH_HOME=~/genvs_ws/vsbench_assets/weights/torch
export VSBENCH_WORKSPACE_ROOT=~/genvs_ws   # where /workspace/... textures and USDs really are
alias py='nice -n 19 ~/genvs_ws/vs_sim_light/.venv/bin/python'
```
Forbidden for every unit: the `isaac` container, `~/genvs_ws/night_*.lock`, `/workspace/vssim_out`,
`~/genvs_ws/vssim_out`, any write under `~/runpod_dev/vs-sim`, `~/runpod_dev/refine-vs*`,
`~/runpod_dev/vs-sim-*`, `~/genvs_ws/refine-vs*`, `~/genvs_ws/cache`, `~/genvs_ws/torch_home`.
Every long command: `nice -n 19`, ≤ 2 threads, and a `timeout`. Expected CPU budgets below are
for this box at load ~20; they are estimates to be replaced by measured numbers in each gate's
report.

### Milestone 1 — "the depth camera, the poster, ViT-VS flying one start on CPU"

**U0 — Environment and pins** (no code in `vsbench/` except `tools/`)
- Inputs: the venv, the vs-sim checkout, the two weight folders.
- Do: `tools/cpu_env.sh` (above); `git clone ~/runpod_dev/vs-sim external/vs-sim`, pin
  `external/VSSIM_COMMIT`; `pip install --no-deps -e external/vs-sim`; install **CPU** wheels
  only: `torch`, `torchvision` from `https://download.pytorch.org/whl/cpu` (~750 MB on disk;
  a CPU wheel cannot touch the GPU), `transformers>=4.57`, `PyYAML`, `scikit-learn`,
  `usd-core` (verify the cp312 wheel first), `pytest`; copy the two weight folders (250 MB)
  to `~/genvs_ws/vsbench_assets/weights/{hf,torch}` with `cp -r` (read-only use of the
  sources); add `external/`, `~/genvs_ws/vsbench_assets` to `.gitignore`/docs.
- Gate: `py -c "import vssim, torch, transformers, mujoco; print(torch.cuda.is_available())"`
  prints `False`; `nvidia-smi` memory unchanged before/after; vs-sim's fast tier runs in the
  venv: `py -m pytest external/vs-sim/tests/fast -q -x` **green or skipped-with-reason** (it
  flies arms on the numpy mock with stub backbones and checks every parity fixture; last known
  632 passed / 15 skipped in 47 s on the station — expect ~2–4 min at nice 19). Any failure
  caused by cv2 5.0 or numpy 2.5 is reported, not patched in vs-sim.
- Cost: ~10 min wall, mostly pip. Disk: ~1.5 GB venv growth + 0.25 GB weights.

**U1 — `MujocoScene` for the posters + the D435i gates** (`vsbench/vssim_bridge/{__init__,scene,build,camera}.py`, `tests/bridge/test_scene_posters.py`)
- Inputs: U0; `vssim.scenes`, `vssim.camera`, `vsbench.render.mujoco_backend`.
- Do: `camera.d435i_camera_model() -> CameraModel` (cx 959.5 / cy 539.5, near 0.05, far 10)
  and `d435i_intrinsics() -> vssim Intrinsics`; `build.scene_spec_for(spec)` for `kind:
  poster`; `scene.MujocoScene(name, *, gl="egl", samples=4)` implementing the contract with
  `integrate_optical_velocity`, the one-shot goal capture, `goal_capture_record()`,
  `render_mode`, `gt_correspondence` (via `vssim.labels.gt_correspondence` on the cached goal
  depth and the last depth, as `IsaacScene` does), `goal_frame_md5`, a `close()` and context
  manager. Pose ↔ `T_wc` through `vssim.scene.pose_to_T`.
- Gate: the four gates of §4 as pytest tests, plus `isinstance(scene, Scene)` and
  `check_scene_goal` on hollywood / poster2 / poster3. The test writes
  `results/bridge/u1_camera_gates.json` with the measured numbers.
- Cost: a 1920×1080 llvmpipe render with the extra depth pass ≈ 0.15–0.3 s; the tests
  render < 100 frames → ~1 min.

**U2 — The bench driver and the first flight** (`vsbench/vssim_bridge/fly.py`, `tests/bridge/test_fly_smoke.py`)
- Inputs: U1; `vssim.bench.{config,arms,runner,run,scoring}`, `vssim.benchmark`,
  `external/vs-sim/configs/arms/*.yaml`.
- Do: `python -m vsbench.vssim_bridge.fly --scene hollywood --arm vitvs_dinov3 --start-set
  near60 --limit 1 --max-steps 100 --out ~/genvs_ws/vsbench_out/<run>`: `config.load(arm
  yaml)` + overrides `{matcher: {device: cpu}, episode: {max_iter: N}}`, `build_arm(cfg,
  intrinsics=scene.intrinsics, scene_spec=...)` (the classical arms need `scene_spec` for
  their RANSAC model), `benchmark.load_starts/select_starts`, `runner.run_episode(scene, arm,
  start, cfg, starts_payload=payload, telemetry=...)`, one jsonl record per start with
  `render_mode`, `vsbench_commit`, `vssim_commit` added; `summary` sub-command reusing
  `vssim.bench.scoring.landing_stats`. `torch.set_num_threads(2)` at start-up; refuse to run
  if `torch.cuda.is_available()`.
- Gate (**the M1 gate**): ViT-VS DINOv3 (real weights, CPU) on hollywood from the first
  `near60` start (3–8 mm away) for 100 steps: the record exists, `final_translation_mm` <
  the start's `pos_err_m·1000` (it moved toward the goal) and < 5 mm, `predicate_a` success
  under the frozen schema; `n_matches` ≥ 20 on frame 1 (the goal picture matched). Also the
  classical `sift_ransac` on the same start (no weights) as a weight-free smoke. The test
  stores the record under `results/bridge/`.
- Cost: DINOv3-S/16 at 512 on 2 threads ≈ 0.4–0.8 s/step + render ≈ 0.3 s → **100 steps ≈ 2 min**.

### Milestone 2 — the rooms, the objects, ART-VS, parity

**U3 ∥ U2 — Object converter + the bbox-goal gate** (`vsbench/vssim_bridge/assets.py`, `tools/convert_objects.py`, `tests/bridge/test_assets.py`)
- Inputs: the USD trees, the YCB OBJs, `vssim.assets` (library, `target_for`,
  `resolve_usd_path` with `VSSIM_ASSET_ROOT=~/genvs_ws/genvs-sim`, `VSSIM_YCB_ROOT=~/genvs_ws/assets/ycb`).
- Do: the two readers of §5, the cache format, `load_object(instance_id) -> (V, UV, F, texture,
  meta)`, `placed_mesh(entry, item, layout_entry) -> vsbench.scene.Mesh` applying the scale
  rule (`"/" in id → 1.0`, else `target_for(category)` on the measured bbox under the item's
  quaternion; `size_m` when stated), and `scene_bbox(meshes)`. Convert only the objects of the
  six scenes (drill `handal_drill_25` + the 8 YCB ids) plus three library objects of different
  sources (one gso, one objaverse, one housecat) as reader coverage.
- Gate: the bbox-goal gate of §5 on `industrial1_drill`, `cluttered2`, `cluttered3` at
  `check_goal_match` tolerances (report the residuals in mm even when green); every converted
  mesh has finite vertices, consistent face indices, UVs in [0, 1], a texture; a contact sheet
  `results/bridge/u3_objects.png` (each object rendered once by MuJoCo from the library's
  upright canonical pose, 256 px tiles) for the owner to eyeball.
- Cost: minutes; disk < 200 MB.

**U4 — The three rooms in `MujocoScene`** (extends `build.py`, `tests/bridge/test_scene_rooms.py`)
- Inputs: U1, U3.
- Do: `scene_spec_for` for `kind: multi_object` (ground, layout meshes, lighting), the
  bbox-derived goal, `check_scene_goal`; `object_mask_from_depth` works (ground at `z = 0`).
- Gate: `check_scene_goal` passes on all three rooms (or D-4 applies and the record says so);
  depth gate on the ground plane at the room goal distances; `vitvs_dinov3_objmask` builds and
  the goal mask share is in (0.02, 0.6); a goal render per room saved to `results/bridge/`.
- Cost: a room compile ≈ 1–3 s (the 4096² ground texture dominates); renders as U1.

**U5 — Baseline parity: mock vs MuJoCo, ViT-VS and ART-VS** (`tools/parity_mock_vs_mujoco.py`, `results/bridge/u5_parity.md`)
- Inputs: U2; `MockPlaneScene(texture=<hollywood png>, render_scale=1.0)` (the vs-sim mock
  *is* the poster scene without shading).
- Do: fly `vitvs_dinov3`, `vitvs_dinov2` and `artvs` (device cpu) on hollywood from the same
  three `near60` starts and two `narrow20` starts, 150 steps, on both worlds; same seeds, same
  arm objects rebuilt per world. Compare per start: landing (mm, deg) at step 150, the step at
  which `cns_plateau` fired, the first-frame match count, and the per-step commanded-twist
  norm curves (correlation).
- Gate: landings agree within 3 mm / 0.3° on every start and ART-VS's coarse→tiled switch
  fires in both worlds (the latch is on the record as `artvs_switched`, from
  `TiledMatcher.record_fields`; the step it fired at comes from the telemetry);
  the vs-sim fast-tier fixtures (U0) remain the proof that the *code* is the published one —
  this unit proves the *world* does not break it. Write the numbers down; a miss is a finding
  about the renderer, not something to tune away.
- Cost: ART-VS's tiled phase is 36 crops at 224 px per step ≈ 1.5 s/step on 2 threads →
  150 steps ≈ 4–5 min per start; whole unit ≈ 1 h wall at nice 19. Run it in the background
  with `timeout 2h`.

**U6 ∥ U5 — vs-sim arms as `vsbench` methods** (`vsbench/methods/vssim_arms.py`, `tests/bridge/test_vssim_arm_method.py`)
- Inputs: U2.
- Do: `@register("vitvs_dinov3")`, `@register("artvs")`, `@register("vitvs_dinov2")`,
  `@register("sift_ransac")` classes wrapping `build_arm` from the arm yaml (parameters:
  `device`, `num_landmarks`, overrides as `key=value`); `wants_depth = "dense"`,
  `privileged = False`; `configure(K, w, h, dt)` → `Intrinsics` + the arm's `controller.dt`
  override; `set_target` → `arm.reset(rgb, depth)`; `get_control_rate` → `arm.step`, returning
  `kp_cur/kp_des` from `matches.to_rows()` and `error_norm` from `debug["feature_error_norm"]`.
  The import-graph rule of `vsbench/methods/` (no loop, renderer or scene imports) holds:
  the module imports `vssim.bench.arms` and `vssim.bench.config` only, lazily. Document that
  vsbench's own loop integrates with `exp_se3` at dt 0.04 by default — a different world rule
  from the benchmark's; `python -m vsbench.demo --method vitvs_dinov3 --param device=cpu`
  is a demo, the benchmark driver is U2.
- Gate: vsbench's contract, leak and zero-at-target gates pass for the new methods (run with
  `--backends mujoco-egl`, `-k vssim`); the demo's start 1 converges under 3 mm within 300 steps
  with `sift_ransac`; the ViT methods run the three setup checks (`python -m vsbench.check`).
- Cost: ~10 min.

### Milestone 3 — the whole library, perturbations, the rig check

**U7 ∥ anything — Convert the full library** (`tools/convert_objects.py --all`)
- Gate: `INDEX.json` with ≥ 310 of 317 `ok` (the failures named, with reasons); total size
  under the 10 GB cap; a 323-tile contact sheet; a spot check of 12 random objects' native
  extents against `vssim.assets.RESCUE_TARGETS_M` after scaling (longest extent = target ± 1 mm).
- Cost: single process, `nice -n 19`, `OMP_NUM_THREADS=1`, ~1 h; disk 4–6 GB.

**U8 — Perturbations that need world verbs** (`scene.py`: `build_poster`, `put_box`, `remove_prim(s)`, `place_extra_objects`, `set_prim_visible`, `object_world_bbox`, `object_pose`, `set_lighting`/`lighting_snapshot`/`restore_lighting`)
- Do: each verb edits a `SceneSpec` copy and recompiles the MuJoCo model (the renderer is
  rebuilt; the camera pose is kept); `perturb._needs(scene, *verbs)` then admits
  `ground:`, `occluder`, `distractors`, `light`, `combined`, `swap:@name` (the last needs U7
  and the instance-suite scenes, which `scene_spec_for` supports once U7 exists).
- Gate: `vssim.bench.perturb` fast tests' recording-fake contract run against `MujocoScene`
  (begin/end leaves the scene byte-identical: goal render md5 equal before and after); one
  episode per mode on `cluttered2` with `sift_ransac` writes its draw to the record.
- Cost: recompile ≈ 1–3 s per episode; ~20 min for the gate.

**U9 — The rig check twin** (`vsbench/vssim_bridge/check_baselines.py`)
- Do: what `vs-sim/scripts/check_baselines.py` does, on the MuJoCo world: per scene × arm,
  the arm gets its own goal picture as the live frame (must re-find its points and command
  ~zero), flies a 3–8 mm start for 200 steps, optionally the first N `capped30` starts;
  writes `check.json` + `check.md` + goal frames; films through `FilmWriter`.
- Gate: `check.md` all-green for `{sift_ransac, orb_ransac, vitvs_dinov3, artvs}` × six scenes
  at the zero-motion and the near-start rows; the `capped30` row is *reported* (CPU cost).
- Cost: zero-motion + near rows ≈ 6 scenes × 4 arms × (1 + 200 steps) ≈ 2–3 h at nice 19 →
  background with a `timeout`.

**U10 — Docs** (`README.md` § "The vs-sim bridge", `docs/BRIDGE.md`, the `results/bridge/*.md` reports)
- The bridge's README section states: what it is, what it is not (§6), the environment
  (`tools/cpu_env.sh`), the cache, the pin, and the three decisions the owner took. The
  owner-facing HTML view is the parent agent's call (workspace rule 7).

---

## 8. Risks and open owner decisions

Each as: what it is about · options · cost · one recommendation.

**D-1 Architecture.** Depend on `vssim` (pinned clone + `pip -e`, no copies) vs copy the
baselines into `vsbench`. Cost of depending: a pin to maintain, two repos to read; cost of
copying: two homes for "as published" code, forbidden by the one-home rule. **Recommend:
depend (§2, option (c) built as (a) first).**

**D-2 Where the MuJoCo world lives, and the vs-sim changes to approve later.** (i) Keep
`MujocoScene` in `vsbench/vssim_bridge/` for good, or (ii) move it into `vssim` as a third world
(`vssim/mujoco_scene.py`, with MuJoCo as an optional extra) after the eval, where `fly.py
--world mujoco` and `check_baselines.py --world mujoco` then fly it. Three vs-sim edits would
be wanted for (ii): the `--world` factory in the three drivers, a `mujoco` optional extra in
`pyproject.toml`, and `opencv-python` → `opencv-python-headless` (or either) in the
`baselines` extra. Cost: (ii) is one file move + ~40 lines; it makes the benchmark repo carry
a second renderer, which is a scope statement about vs-sim. **Recommend: build as (i) now;
decide (ii) when the eval is over, leaning yes — a benchmark that flies without Isaac is more
useful to the people vs-sim was carved out for.** Nobody touches vs-sim before then.

**D-3 Half-pixel convention of the MuJoCo camera.** Match Isaac (`cx = 959.5`) or the numpy
mock (`cx = 960`) in pixel-centre terms; `scene.intrinsics` reports the frozen (960, 540)
either way. Cost: none; 0.15 mm at the poster distance. **Recommend: Isaac's**, pinned by the
phase-correlation gate so it is visible (§4).

**D-4 When our room bbox disagrees with Isaac's by more than the goal tolerance.** Options:
(a) the frozen start file's goal pose wins (the benchmark's truth; the record notes the bbox
residual), (b) our measured goal wins (then the frozen starts are no longer the same offsets
from the goal), (c) refuse to fly the scene. Cost: (a) keeps every start comparable;
(b) breaks pairing silently. **Recommend (a)**, with the residual printed in `check.md`.

**D-5 Depth noise model.** vs-sim's depth is true metres; there is nothing to port. Adding a
D435i-like noise model (quantisation, holes at edges, ±2 % range error) would be a new
perturbation condition in both worlds, i.e. a science decision. **Recommend: not now**; the
bridge keeps the hook (`Perturbation.live` already touches the live frame; a `depth_noise`
mode would touch depth the same way).

**D-6 The six Nucleus objects** (`/Isaac/Props/YCB/Axis_Aligned/{035_power_drill, 025_mug,
...}.usd`) are not on the host disk. Options: skip them (none of the six benchmark scenes uses
one), or fetch them later from the Isaac assets pack. **Recommend: skip; mark `nucleus` in
`INDEX.json`.**

**D-7 Disk and cache location.** `~/genvs_ws/vsbench_assets/` (same volume as everything
else, 766 GB free), 4–6 GB for the library + 1.5 GB for the venv growth + 0.25 GB weights
copy, hard cap 10 GB on the object cache. **Recommend: approve.**

**D-8 Installing CPU torch into the vsbench venv.** ~750 MB; a CPU wheel has no CUDA runtime,
so it cannot take GPU memory even by accident; plus `CUDA_VISIBLE_DEVICES=""` as belt and
braces. Alternative: a second venv. **Recommend: the one venv, CPU wheels.**

**D-9 Record labelling.** Every MuJoCo record carries `render_mode: mujoco-llvmpipe|
mujoco-egl-gpu`, `vsbench_commit`, `vssim_commit`; the bridge's `summary` refuses a mixed
`render_mode` input. **Recommend: approve as a standing rule — MuJoCo numbers are never
pooled with Isaac numbers in any table.**

**Risks (technical, with the mitigation in the plan)**
- *Shared CPU*: load is ~22 on 16 threads from the eval and other users' ROS/grasp processes.
  Everything here is `nice -n 19` with ≤ 2 threads, so it yields; wall times will be 1.5–3× the
  estimates above and every long unit runs in the background with a `timeout`.
- *llvmpipe memory*: a 1920×1080 context ×2 (colour + depth) + a 4096² texture + meshes ≈
  0.3–0.6 GB RSS per scene process; fine (43 GB available), but never more than two bridge
  processes at once.
- *Per-step cost of ART-VS on CPU* (~1.5 s): the gates use 100–200 steps and near/narrow
  starts; a 600-step `capped30` campaign is hours and belongs on the GPU after the eval.
- *Library versions*: the venv's OpenCV is 5.0 and numpy 2.5; vs-sim's fast tier was last green
  on cv2 4.x / numpy 1.26–2.x. U0 runs the tier first; a red test is reported to the owner as
  a vs-sim finding, never patched in the clone.
- *`usd-core` wheel for Python 3.12*: verified at U3 start (`pip index versions usd-core`); if
  absent, a 3.11 `uv venv` just for the converter (the cache format is interpreter-agnostic).
- *YCB OBJ vs Isaac's YCB USD axes*: caught by the bbox-goal gate; fallback is usd-core on
  the YCB USD mesh.
- *Fixture tolerance on CPU*: the DINOv3 token fixture is 1e-5 on CPU and the last digit
  follows the thread count (features.py docstring); run the fast tier with the same
  `OMP_NUM_THREADS=2` every time and report, do not loosen.
- *HF lock files*: `transformers.from_pretrained(local_files_only=True)` may still create
  `.locks/` — hence the weights copy in `vsbench_assets/weights`, never the shared cache.

---

## 9. In plain words

**What the owner asked.** We have a heavy simulator (Isaac Sim: needs the big GPU, a 60 GB
install, and right now it is busy for days with the T1 Stage 2 evaluation) and a light one
(MuJoCo: runs on a laptop CPU, no install pain). The question was whether the light one can
do the same things: the same test scenes, the same camera, the same objects, and the same
competitor methods (ViT-VS and ART-VS) — without disturbing the evaluation that is running.

**The answer is yes, and the trick is to not copy anything.** vs-sim was built so that a
"world" is just four actions (place the camera, take a picture, move by a small velocity,
tell me the facts). The competitor methods, the scoring, the start poses and the scene
descriptions never talk to Isaac directly — they talk to those four actions. So we write one
new "world" class for MuJoCo that answers those four actions, and everything else runs
unchanged. The published methods stay in their one home (vs-sim); the light simulator just
imports them. That is option (c) in §2.

**The camera.** The real camera is an Intel RealSense D435i. Its numbers (1920×1080 pixels,
a 42° vertical view, focal length 1406.75 px) are in one file in vs-sim and go straight into
MuJoCo. MuJoCo gives true distance per pixel (a "depth image") from its drawing buffer, the
same kind of depth Isaac gives. One tiny subtlety — whether "the centre of the picture" means
the middle of a pixel or the edge between two pixels (half a pixel, 0.15 mm at our distance)
— is decided (we follow Isaac), written down, and tested, rather than left to chance.

**The objects.** The 323 objects exist on this machine as USD files (Isaac's format) with a
picture texture each; the eight YCB kitchen objects also exist as plain OBJ meshes. A small
converter reads them on the CPU (a tool called `usd-core`, no Isaac needed), writes a compact
cache (about 5 GB), and we re-apply vs-sim's own sizing and orientation rules. The proof that
the conversion is right needs no Isaac render: Isaac's goal camera for each room was placed
from the measured box around the objects, and that number is frozen in the start files; if our
converted objects give the same box to within half a millimetre, the meshes, units, axes and
sizes are right.

**What a MuJoCo result means.** The pictures differ from Isaac's (simpler lighting, no fancy
materials), so a MuJoCo landing error is **not** an Isaac landing error and will never be put
in the same table. What it *is* good for: trying ideas fast on a laptop against the real
competitors and the real scoring, catching bugs, teaching, and checking that the competitor
code still behaves the same after a change (MuJoCo always draws the same picture twice;
Isaac does not).

**How the evaluation stays safe.** Every step here runs with the GPU hidden
(`CUDA_VISIBLE_DEVICES=""`), MuJoCo drawing on the CPU, PyTorch's CPU-only build, lowest CPU
priority (`nice 19`) and at most two threads; it never opens the Isaac container, the lock
files, the result folders or the vs-sim checkout for writing; it downloads no model weights
(the DINO weights are already on disk and get copied once). The first milestone is small: the
camera, the three poster scenes, and ViT-VS flying one 5-mm start on the CPU in about two
minutes.

**Decisions for the owner** (details in §8): approve the "depend, don't copy" design (D-1);
decide later whether the MuJoCo world moves into vs-sim once the evaluation is over (D-2);
accept Isaac's pixel-centre convention (D-3) and "the frozen goal wins" if a room's box
disagrees by a hair (D-4); no depth noise for now (D-5); skip the six objects that live only on
NVIDIA's asset server (D-6); the 10 GB disk budget (D-7); CPU-only PyTorch in the existing
environment (D-8); and the standing rule that MuJoCo numbers are labelled and never mixed with
Isaac numbers (D-9).
