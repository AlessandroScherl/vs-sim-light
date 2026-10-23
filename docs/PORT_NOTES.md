# PORT_NOTES - developer notes for vs_sim_light (vsbench)

> Developer notes: how the simulator was built and checked (steps 1-3, the vs-sim port units
> U0-U6, the station rules); moved off the front page on 2026-10-23. Users start at `README.md`.

## 0. History in one paragraph

VS-Bench, a lightweight visual-servoing simulator and benchmark. Step 1: one `Renderer`
interface with three backends (MuJoCo, PyBullet, Open3D), the gates that check each one
against analytic reference math, the measurements behind the choice, and
`bakeoff_report.md`; MuJoCo won. Step 2: the motion layer, a free-flying camera that
integrates a clamped 6-DoF velocity command on SE(3), the point interaction matrix, a
trajectory log, and the gates that prove the image moves exactly as the twist predicts.
Step 3: the servo loop, one method interface with a registry, three basic controllers,
the setup checks, an episode log that replays into a debug video, and a demo.
Metrics beyond the true pose error and perturbations come later.

## 1. The renderer (step 1)

### Install and first render

```
pip install -e ".[all,dev]"          # or one of [mujoco] [pybullet] [open3d]
python -m vsbench.smoke              # first backend that works here; writes smoke.png
VSBENCH_ALLOW_GPU=1 python -m vsbench.smoke --backend mujoco-glfw     # a laptop with a display (opt-in)
```

On a Linux machine without a display or GPU, MuJoCo needs one of: the `libosmesa6`
package (`MUJOCO_GL=osmesa`, `--backend mujoco-osmesa`) or Mesa's EGL (`libegl1` and
`libgl1-mesa-dri`, usually present) with `--backend mujoco-egl`.
PyBullet's TinyRenderer and Open3D's CPU mode need nothing else. **Software GL is the
default** (since 2026-10-23, `vsbench/glenv.py`): Mesa's llvmpipe is forced for every
backend and the GPU-capable variants (`mujoco-glfw`, `pybullet-egl`, `open3d-gpu`) are
refused, so a machine with a GPU never touches it; `VSBENCH_ALLOW_GPU=1` is the one opt-in
(`VSBENCH_SOFTWARE_GL=1`, still set by `tools/cpu_env.sh`, is now the default made explicit).

### Result of the bake-off (station, 2026-09-18)

`results/bakeoff_report.md`: MuJoCo is the pick. MuJoCo (EGL/llvmpipe, OSMesa, and on the
RTX 4090) and PyBullet's TinyRenderer pass all four gates; Open3D fails cross-process
determinism on the CPU (an intermittent artefact on one mesh under llvmpipe) and returns
an unusable depth image on the GPU. PyBullet has no wheels for Python 3.12, macOS or
Windows and renders the clutter scene at 11 fps; MuJoCo renders it at about 100 fps at
640 x 480 in one CPU process (380 fps without anti-aliasing, 1140 fps on the GPU). Open3D
has the sharpest slanted poster but 11 fps on the clutter scene on the CPU. GPU and CPU
MuJoCo renders agree to 44.6 dB; EGL and OSMesa are bit-identical.

### The interface

```python
import numpy as np
from vsbench import CameraModel, SceneSpec, Plane, Material, Light
from vsbench.render import make_renderer
from vsbench.stdscenes import load_poster, POSTER_SIZE
from vsbench.poses import look_at

cam = CameraModel.from_focal(640, 480, fx=600.0)            # K in pixels, OpenCV convention
scene = SceneSpec((Plane(np.eye(4), POSTER_SIZE, Material(texture=load_poster())),),
                  lights=(Light(direction=(0.2, 0.1, -1.0)),), ambient=0.3)
r = make_renderer("mujoco-egl", scene, cam)                 # or mujoco-osmesa, mujoco-glfw,
                                                            # pybullet-tiny, open3d-cpu, ...
T_wc = look_at(eye=(0.3, -0.5, 0.9), target=(0.0, 0.0, 0.0))  # camera-to-world, optical frame
rgb, depth = r.render(T_wc)                                 # uint8 (H, W, 3); float32 (H, W) metres
r.set_light((0.0, 0.5, -1.0), intensity=0.7)
r.close()
```

Conventions: `T_wc` is camera-to-world with x right, y down, z forward; integer pixel
`(u, v)` is the pixel centre; depth is z-depth in metres, 0 where nothing was hit; one
renderer per process. Every backend converts to its own frame internally.

## 2. The camera motion (step 2)

```python
from vsbench import FreeCamera, Trajectory
from vsbench.interaction import predicted_flow

cam_free = FreeCamera(T_wc, dt=0.04)                        # 25 Hz, limits 2 m/s and 4 rad/s
log = Trajectory(cam_free.dt, cam_free.limits)
for k in range(25):                                         # one second: pan right at 10 deg/s
    twist = [0.0, 0.0, 0.0, 0.0, 0.1745, 0.0]               # [vx, vy, vz, wx, wy, wz], optical frame
    t, pose = cam_free.t, cam_free.T_wc
    applied = cam_free.step(twist)                          # T_wc <- T_wc @ exp_se3(twist * dt), clamped
    log.append(t, pose, twist, applied)
    rgb, depth = r.render(cam_free.T_wc)                    # the image content moves left
flow = predicted_flow(cam, [[320.0, 240.0]], depth[240, 320], applied, cam_free.dt)  # px per step
log.save("flight.npz")                                      # Trajectory.load(...).replay() is bit-identical
```

The twist is the camera's body velocity in its own optical frame, linear part first, in
m/s and rad/s. Positive `vx` slides the camera right (the image moves left), `vy` down,
`vz` forward (the image expands); positive `wx` tilts up, `wy` pans right, `wz` rolls
clockwise seen from behind. `results/step2_axes.png` shows each axis for one second with
the predicted and the tracked flow, and `results/STEP2_REPORT.md` lists the nine motion
gates with their numbers.

## 3. The Step-3 test loop and demo (step 3)

`vsbench.episode.run_episode` is vsbench's own Step-3 loop: a **test instrument** for the
Step 1-3 gates (replay, frame hashes, video), which stops on the true pose. It is **not the
benchmark loop**: that is the copied vs-sim loop `vsbench.vs.bench.runner.run_episode`, flown
by `python -m vsbench.world.fly` (below; `docs/PORT_PLAN.md` section 2.5; whether to retire
the Step-3 loop later is the owner's open decision D-10).

A method sees images, gets depth only in the form it asked for and the intrinsics, and
returns a camera-frame twist; `run_episode` renders, moves the camera with `FreeCamera`
and logs everything into one `.npz` that replays into a video.

```
VSBENCH_SOFTWARE_GL=1 python -m vsbench.demo --method sift_ibvs       # the ten demo starts, logs, one video
python -m vsbench.demo --method orb_ibvs --param gain=0.8              # constructor arguments as key=value
python -m vsbench.demo --method true_point_ibvs                        # marker scene, the privileged instrument
python -m vsbench.demo --method sift_ibvs --jobs 4                     # ten starts over 4 processes (same logs, bit for bit)
VSBENCH_SOFTWARE_GL=1 python -m vsbench.check --method sift_ibvs      # the three setup checks, pass/fail per axis
python -m vsbench.replay results/demo/sift_ibvs/start_10.npz --video out.mp4   # re-render, compare hashes, video
```

The demo prints status, steps and final error per start, saves `results/demo/<method>/start_NN.npz`,
and writes `start_10.mp4` and `start_10_matches.png` for the largest offset. The video
shows the current and desired image with the matches (inliers green, outliers red), a 3D
view with the target and current frustum, the path and the velocity arrows, and the error
and twist curves; the match sheet is the same matching at four steps in full resolution.
Built-in methods: `oracle_pbvs` and `true_point_ibvs` (privileged test instruments that see
the true poses), `sift_ibvs` and `orb_ibvs` (SIFT or ORB, ratio test, RANSAC on a
homography, depth at the inlier keypoints, the point IBVS law with `interaction_matrix`).

```python
from vsbench.camera import gate_camera
from vsbench.episode import EpisodeConfig, EpisodeLog, run_episode
from vsbench.methods import make_method
from vsbench.render import make_renderer
from vsbench.scenes import make_scene, poster_target_pose
from vsbench.se3 import exp_se3

cam = gate_camera()
r = make_renderer("mujoco-egl", make_scene("poster"), cam)
T_target = poster_target_pose(cam)                              # straight at the poster, 70 % of the width
T_start = T_target @ exp_se3([0.05, 0.0, -0.1, 0.0, 0.0, 0.26])   # 5 cm right, 10 cm back, 15 deg roll
log = run_episode(r, make_method("sift_ibvs"), T_start, T_target, EpisodeConfig(max_steps=500), "poster")
print(log.meta["status"], len(log), "steps", log.err_t[-1] * 1e3, "mm")   # converged 128 steps 0.82 mm
log.save("episode.npz")                                          # EpisodeLog.load(...) gives it back
```

### Write your own method (for the Step-3 loop)

One class with `name`, `wants_depth` (`none`, `scalar` or `dense`), `privileged = False`
and the four calls, plus one `@register` line. This one servos two axes from the global
image shift and passes the zero-at-target and hold-at-target checks; put it in any module
you import before `make_method("shift_ibvs")`:

```python
import cv2
import numpy as np
from vsbench.methods import register

@register("shift_ibvs")
class ShiftIBVS:
    wants_depth = "scalar"          # one number: the distance to the target view
    privileged = False

    def __init__(self, gain=1.0):
        self.gain = gain
        self.params = {"gain": gain}                  # recorded in the episode meta

    def configure(self, K, width, height, dt):
        self.fx, self.fy = K[0, 0], K[1, 1]

    def reset(self):
        self.desired = None

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None):
        self.desired = cv2.cvtColor(np.asarray(desired_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
        self.Z = distance_prior

    def get_control_rate(self, current_rgb, current_depth=None):
        gray = cv2.cvtColor(np.asarray(current_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(gray, self.desired)   # px shift current -> desired
        twist = np.zeros(6)
        twist[0] = -self.gain * dx / self.fx * self.Z   # content must move by -dx, so slide the other way
        twist[1] = -self.gain * dy / self.fy * self.Z
        return twist, {"error_norm": float(np.hypot(dx, dy)), "lost": response < 0.05}
```

The twist is `[vx, vy, vz, wx, wy, wz]` in the optical frame (m/s, rad/s), the images are
read-only `uint8 (H, W, 3)`, depth is `float32 (H, W)` metres with 0 where nothing was hit.
The optional info dict may carry `kp_cur`, `kp_des`, `inliers`, `error_norm`, `observer_s`,
`controller_s` and `lost`; the video and the log use them. For a full feature pipeline,
subclass `FeatureIBVS` in `vsbench/methods/feature_ibvs.py` and override one of `_describe`,
`_match`, `_ransac`, `_depths` or `_law`. Code under `vsbench/methods/` must not import the
episode loop, a renderer or a scene (a test checks the import graph), and only the two
privileged instruments ever see a pose.

## 4. Gates and measurements (steps 1-3)

```
VSBENCH_SOFTWARE_GL=1 pytest --backends mujoco-egl        # every gate (steps 1 to 3), one variant, about 8 min
VSBENCH_SOFTWARE_GL=1 tools/run_all.sh                    # step 1: gates, measurements, bake-off report
VSBENCH_SOFTWARE_GL=1 tools/run_step2.sh                  # step 2: axis sheet, nine gates, report
VSBENCH_SOFTWARE_GL=1 tools/run_step3.sh                  # step 3: demo per method, thirteen gates, report
```

Step 1: `tools/run_gates.py` runs each gate file in its own pytest process per variant
and writes `results/gates_<variant>.json`; `tools/bench_speed.py`, `bench_parallel.py`,
`quality_sheet.py`, `cross_variant.py`, `lighting_check.py` and `install_friction.sh`
write the other `results/*.json`; `tools/bakeoff_report.py` turns them into
`results/bakeoff_report.md`. The brief is in `docs/STEP1_RENDERER_BAKEOFF.md`.

Step 2: `tools/fly_axes.py` writes `results/step2_axes.png` and `step2_axes.json`;
`tools/step2_report.py` runs the nine motion gates (six need no renderer, the rendered
flow gate needs MuJoCo), times `FreeCamera.step` and writes `results/STEP2_REPORT.md`.
The brief is in `docs/STEP2_CAMERA_MOTION.md`.

Step 3: `python -m vsbench.demo` for each built-in method writes `results/demo/<method>/`
(ten logs, `demo.json`, one video, one match sheet; `--jobs N` spreads the starts over N
processes, which pays off with a GPU renderer, not on a CPU-only laptop where one episode
already uses every core; `tools/bench_episodes.py` measures it); `tools/step3_report.py` runs the
thirteen servo-loop gates (contract, leak, oracle decay and convergence, the three setup
checks, negative control, IBVS decay, feature convergence, log round trip, replay, video)
and writes `results/STEP3_REPORT.md` with every measured number, the demo tables and the
time per `get_control_rate` call. The brief is in `docs/STEP3_SERVO_LOOP.md`.

## 5. The vs-sim bridge: environment and pin (port unit U0)

The port of vs-sim's scenes, camera, objects and published baselines onto this simulator is
planned in `docs/PORT_PLAN.md`. Owner decision 2026-10-22: the vs-sim code is **copied** into
this repository, which stands alone and stays separate from vs-sim for good. A local clone of
vs-sim sits in `external/vs-sim` (git-ignored), checked out at the commit in
`external/VSSIM_COMMIT` (committed): it is the pinned **source** of that copy and of a drift
check, not a runtime dependency, and it is **not** pip-installed in the venv (its fast tier is
run with `PYTHONPATH=external/vs-sim`).
Weights are a copy, outside the repository, so the shared caches are never opened:

```
source tools/cpu_env.sh                                   # no GPU, llvmpipe, 2 threads, nice 19, offline weights
git clone ~/runpod_dev/vs-sim external/vs-sim             # a local clone only reads the source
git -C external/vs-sim checkout "$(cat external/VSSIM_COMMIT)"
$VSBENCH_PY -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch torchvision
$VSBENCH_PY -m pip install --only-binary=:all: -c tools/constraints.txt "transformers>=4.57" "PyYAML>=6" "scikit-learn>=1.0" usd-core "pytest>=8" opencv-python-headless
# weights (~250 MB), copied once from the station's caches:
#   ~/genvs_ws/vsbench_assets/weights/hf/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/   (HF_HOME)
#   ~/genvs_ws/vsbench_assets/weights/torch/hub/{checkpoints/dinov2_vits14_pretrain.pth,
#                                                facebookresearch_dinov2_main/, trusted_list}  (TORCH_HOME)
PYTHONPATH=external/vs-sim env -u VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT \
  $VSBENCH_PY -m pytest external/vs-sim/tests/fast -q   # the clone's fast tier in this venv
```

`~/genvs_ws/vsbench_assets/` (outside the repository) is the bridge's cache: `weights/` now,
the converted objects later. Torch is the CPU build (`+cpu`); `torch.cuda.is_available()` is
`False` by construction. OpenCV is pinned to `opencv-python-headless==4.11.0.86`, the Isaac
flight interpreter's version (`tools/constraints.txt`, `pyproject.toml`; parent decision
2026-10-22): OpenCV 5.0 has no `AKAZE_create` and vs-sim's classical fixture is pinned to
4.10 / 4.11, so under 5.0 six AKAZE tests failed and nine fixture tests skipped; under 4.11
the clone's fast tier is 789 passed / 13 skipped of 802 (the 13 need the frozen genvs-sim or
ART-VS checkouts, DINO v1 or LightGlue, none of which this venv has). The two asset-root
variables must be unset for that tier (three `test_assets.py` tests assert the defaults).
The measured record is `results/bridge/u0_env.json`; the per-test split, which the copy must
reproduce, is `results/bridge/u0_clone_fast_tier.txt`.

## 6. The copy of vs-sim (port unit U-copy)

vs-sim's published baselines, bench, scenes, frozen starts, camera and fast tier are
**copied** into `vsbench/` by a script, in vs-sim's own layout (`vssim/` is `vsbench/vs/`, and
`configs/`, `benchmark/`, `assets/asset_library_v2.json`, `tests/{conftest.py,fast,fixtures}`,
`scripts/` sit beside it), so `import vsbench.vs.bench.runner` is vs-sim's
`vssim.bench.runner`. **Nothing in those trees is edited by hand**: a baseline is run as
published, and a vs-sim fix comes in by moving the pin. What was copied, the five rewrite
rules, the three patches and why, what is flown here and what is copied but not wired:
`vsbench/vs/PROVENANCE.md`.

```
source tools/cpu_env.sh
$VSBENCH_PY tools/vssim_copy.py                         # re-copy at external/VSSIM_COMMIT
$VSBENCH_PY tools/vssim_drift.py                        # exit 0 = the copy is the pin's copy
env -u VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT $VSBENCH_PY -m pytest vsbench/tests/fast -q   # 789 passed, 13 skipped
$VSBENCH_PY -m pytest tests/vs -q                       # drift + empty-glob guards (own process)
.venv/bin/ruff check --config tools/ruff_vssim_copy.toml vsbench/vs vsbench/tests vsbench/scripts
```

## 7. The MuJoCo world and the bench driver (port units U1, U2, U4)

`vsbench.world.mujoco_scene.MujocoScene` is vs-sim's `Scene` contract on this renderer (the
three posters and the three rooms; the D435i camera with Isaac's half-pixel principal point,
`PORT_PLAN` §4), and `python -m vsbench.world.fly` is vs-sim's `scripts/fly.py run` on it: the
copied arms, loop, records and success rules, unchanged.

The rooms (U4: `industrial1_drill`, `cluttered2`, `cluttered3`) are built from the converted
object cache (`~/genvs_ws/vsbench_assets/objects`, U3/U7) by the frozen layouts: the ground is
the room's texture once across 2.4 m at `z = 0`, each object `T(pos) R(quat)` with vertices
pre-scaled by vs-sim's scale rule, the lights the posters' stand-in. The goal is re-derived from
the objects' box (Isaac's `BBoxCache` rule) and checked against the frozen start file; on a
miss the frozen goal wins and the residual is recorded (open decision D-4, `goal_rule`). A room
holds up to ~3.6 GB of memory (cluttered2: nine 4096² textures, uploaded to both GL contexts)
and takes ~10 s to build; its goal depth equals Isaac's own to micrometres, its raw picture does
not (no PBR, no tone mapping; `results/bridge/u4_rooms_vs_isaac.png`) - U4b's Isaac look, the
default since, closes most of that (next section). The protocol has no near
start set for the rooms, so `fly` (protocol sets only) flies them from `capped30` / `wide100`;
the near smoke uses `check_baselines`' near start through `tools/u4_rooms.py smoke`.

```
VSBENCH_WRITE_GATES=1 $VSBENCH_PY -m pytest tests/world/test_scene_rooms.py -q   # the U4 gates
PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py measure    # cost, memory, previews (one process per room)
PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py isaac      # side by side with Isaac's goal pictures
PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py smoke --arms sift_ransac   # check_baselines' near start
```

```
source tools/cpu_env.sh
$VSBENCH_PY -m vsbench.world.fly run --scene hollywood --arms vitvs_dinov3 sift_ransac \
    --start-set near60 --limit 1 --max-iter 100 --out results/bridge/m1 --film
$VSBENCH_PY -m vsbench.world.fly run --scene hollywood --arms sift_ransac \
    --start-set near60 --limit 1 --max-iter 100 --out <dir> --telemetry-extended   # + per-match truth
$VSBENCH_PY -m vsbench.world.fly summary --out results/bridge/m1
$VSBENCH_PY -m pytest tests/world -q                    # VSBENCH_WRITE_GATES=1 rewrites the committed gate JSONs
```

The driver never touches the GPU: it forces `CUDA_VISIBLE_DEVICES=""` and
`VSBENCH_SOFTWARE_GL=1`, runs torch on 2 intra- and 2 inter-op threads, hides `nvidia-smi`
(the loop would otherwise record the Isaac evaluation's GPU as this run's), and refuses to fly
unless the built scene is on Mesa llvmpipe (`--allow-gpu`, off by default, lifts that; never
while the Isaac evaluation runs). It accepts only the protocol's start sets. Every record
carries `render_mode`, `vsbench_commit` (`+dirty` by the rule in `vsbench_commit_rule`) and
`vssim_commit`, so a MuJoCo number is never pooled with an Isaac one. The world tests that
render skip unless software GL is requested.

## 8. Isaac's look (port unit U4b)

Owner decision 2026-10-23: the MuJoCo pictures should look like Isaac's. Since U4b every
`MujocoScene` (and so every flight) draws in the **`isaac_fit_v1`** look by default; the old
picture stays available as **`raw`** (`MujocoScene(name, look="raw")`, `fly run --look raw`),
used by the gates that compare against the copied mock (U1's half-pixel gate). Only the colour
picture changes: depth, poses, goals and starts are the same bytes in both looks.

`isaac_fit_v1` (`vsbench/world/look.py`, numbers in `vsbench/world/configs/look_isaac_fit_v1.json`):
- **the light**: the fill (ambient) share is 0.15, measured from Isaac's own shadows and
  shaded sides (was 0.2), and the light is raised per scene by 1.175 (rooms, poster3), 1.065
  (hollywood) or 1.0 (poster2) - as far as it goes before MuJoCo saturates the sun's term or
  clips more than 0.1 % of the goal picture (Isaac's brightest pixel is ~250, it never clips);
- **the shadows**: an 8192 shadow map over a square fitted to the scene (0.41 mm per texel in
  a room, was 1.8 mm): smooth edges instead of stair steps. Isaac's sun is only 0.53° wide, so
  its shadow edges are sharp too (1-2 px); sampling the sun with several lights would cost
  2-3x per frame for little, and is not done;
- **the tone curve**: one per scene and channel, fitted to Isaac's goal picture (exposure,
  gamma, a floor and a Reinhard tone map; monotone; stored as a 256-entry table with its md5,
  the curve id `isaac_fit_v1/<scene>`), applied to the RGB only.

Every record says which look it was drawn with (`look`, `look_curve_id`, `look_curve_md5`,
`look_record` with the light, the shadow map and the config's md5). The config names the
Isaac pictures it was fitted on as `$VSBENCH_ISAAC_GOALS/...` (default
`~/genvs_ws/vssim_out/check1`, resolved by `vsbench.world.look.resolve_isaac_path`); records
made before 2026-10-23 name the md5 of the same file with that root written out
(`8d010bcd…`, `config_md5_expanded`; the curves did not change). The fit and its
measurement (`tools/u4b_look.py`, record `results/bridge/u4b_look.json`, gates
`tests/world/test_look.py`):

| scene | mean grey raw → look (Isaac), goal | held-out near view | histogram distance raw → look, goal / held-out |
| --- | --- | --- | --- |
| hollywood | 91 → 182 (184) | 89 → 180 (181) | 92.8 → 5.7 / 91.9 → 5.6 |
| poster2 | 97 → 190 (191) | 97 → 190 (191) | 93.9 → 4.6 / 94.9 → 4.8 |
| poster3 | 84 → 181 (182) | 83 → 180 (181) | 98.1 → 2.7 / 98.4 → 2.5 |
| industrial1_drill | 83 → 185 (185) | 84 → 185 (185) | 102.3 → 2.7 / 101.8 → 2.5 |
| cluttered2 | 35 → 127 (132) | 35 → 127 (132) | 97.6 → 8.9 / 96.8 → 8.8 |
| cluttered3 | 126 → 216 (217) | 125 → 216 (217) | 91.6 → 2.5 / 91.5 → 2.3 |

(The histogram distance is the mean per-channel earth mover's distance in grey levels; Isaac
against its own re-render scores 0.01-0.2. The held-out view is the check's near start, 3-8 mm
from the goal, so it shows nearly the same content: it guards against over-fitting pixels,
not against a different view.) Cost on llvmpipe at load ~24: +0.1-0.2 s per 1920 x 1080 frame
(the curve 15-30 ms, the rest the larger shadow map), +200 MB memory. The U4 smoke
(`sift_ransac`, 100 steps, near start `n_000`) re-flown in both looks: `industrial1_drill`
16.39 mm (raw) → 0.56 mm (look), Isaac 0.17 mm at 200 steps; `cluttered2` 0.31 → 1.20 mm
(Isaac 0.66); `cluttered3` 0.57 → 0.51 mm (Isaac 2.19) - context only, never pooled.

What still differs: no specular highlights, no transparency, no ambient occlusion (MuJoCo's
diffuse shading); Isaac's dark tones carry a sun-lit floor (its glossy ground reflecting the
sky), which the curve applies everywhere, so shadows stay lighter than Isaac's on dark ground
(cluttered2: 66 vs 44 grey inside the shadow; drill 61 vs 54, cluttered3 99 vs 98); Isaac's
picture is slightly softer (its denoiser) and cluttered2's ground keeps a larger per-pixel
difference (19 grey levels). `results/bridge/u4b_look_vs_isaac.png` (raw | look | Isaac) and
`results/bridge/u4b_drill_shadow_crop.png` show it.

```
source tools/cpu_env.sh
PYTHONPATH=. $VSBENCH_PY tools/u4b_look.py components && PYTHONPATH=. $VSBENCH_PY tools/u4b_look.py fit
PYTHONPATH=. $VSBENCH_PY tools/u4b_look.py measure && PYTHONPATH=. $VSBENCH_PY tools/u4b_look.py shadows
PYTHONPATH=. $VSBENCH_PY tools/u4_rooms.py smoke --look isaac_fit_v1 --out results/bridge/u4b/isaac_fit_v1 \
    --record results/bridge/u4b_look.json --key smoke_isaac_fit_v1
VSBENCH_WRITE_GATES=1 $VSBENCH_PY -m pytest tests/world/test_look.py -q
```

## 9. Methods on the benchmark loop, baselines in the demo (port units U6a, U6b)

Two thin adapters, no baseline code in either (`docs/PORT_PLAN.md` section 2.5).

**A vsbench method on the benchmark loop** (`vsbench/world/method_arm.py`, U6a). `MethodArm`
subclasses the copied `Arm` and overrides `step`: the copied loop flies any registered,
non-privileged vsbench method with the benchmark's scene, starts, period (1/60 s), records,
success rules, telemetry and films. Its config is the template
`vsbench/world/configs/external_method.yaml`; the record says `arm_kind: external` and the
label `external:<method>@<params hash>/<stop>`, so a method cell can never be read as a
published baseline. The scored stop is `cns_plateau` when the method reports `kp_cur` /
`kp_des`, otherwise the cell falls back to `cap_only` **loudly** (a warning, a stderr line,
`stop_mode: cap_only`, the label, and `stop_fallback` on the record). The method gets frames,
depth in the form it asked for and the intrinsics, nothing else; a privileged method is
refused, and a test poisons every truth read of the loop and shows the commands do not move.

```
source tools/cpu_env.sh
$VSBENCH_PY -m vsbench.world.fly run --scene hollywood --method sift_ibvs \
    --start-set near60 --limit 1 --out results/bridge/u6a --telemetry      # --param gain=0.8 ...
```

**The copied arms in the demo** (`vsbench/methods/vssim_arms.py`, U6b). `vitvs_dinov3`,
`vitvs_dinov2`, `artvs`, `sift_ransac`, `orb_ransac` and `akaze_ransac` are registered vsbench
methods: each builds the protocol cell of the same name with the copied `build_arm` (only
`matcher.device`, default `cpu`, is stated; weights from `$HF_HOME` / `$TORCH_HOME`), so

```
source tools/cpu_env.sh
$VSBENCH_PY -m vsbench.demo --method vitvs_dinov3          # the ten demo starts, a video
$VSBENCH_PY -m vsbench.check --method sift_ransac          # the three setup checks (--max-steps N)
```

The demo's world rule differs from the benchmark's: vsbench's Step-3 loop integrates with
`exp_se3` at its own `dt` (0.04 s) on the 640 x 480 gate camera and stops on the true pose;
the benchmark integrates at the arm's 1/60 s with the copied two-step rule on the D435i and
records its stops. Numbers from the two loops are never pooled. A copied arm cannot be flown
through `MethodArm` (`--method vitvs_dinov3` is refused: fly it with `--arms`).

## 10. The front page: one command, the example, the pictures

`README.md` is for outside users (owner, 2026-10-23: "the end-product should be just a
simulator with some scenes, a .gif demo how it looks like, the debug visualizations and how to
use it really simple"); everything that was on it before moved here unchanged.

- `python -m vsbench fly --scene S --method NAME... [--plugin FILE_OR_MODULE] [flags]`
  (`vsbench/__main__.py`) is a thin router over `python -m vsbench.world.fly run`: names that
  are protocol cells go to `--arms` (the copied baseline as published), any other name to
  `--method` (`MethodArm`); `--plugin` imports a `.py` file or module first, so its
  `@register` lines run; every other flag passes through; `--out` defaults to `runs`.
  `python -m vsbench list [--plugin ...]` prints what can be flown. The two old entries are
  unchanged. Test: `tests/world/test_unified_fly.py` (routing, pass-through, the plugin, and
  the README command flown for three steps under software GL).
- `examples/my_method.py` (`my_ibvs`): the README's method, ORB matches + homography RANSAC +
  the point IBVS law written out; from `poster2` `near60` `n_000` it lands 5.58 -> 0.45 mm in
  150 steps (2026-10-23, `isaac_fit_v1`, llvmpipe, ~0.5 s/step at load ~30). It reports
  `kp_cur` / `kp_des`, so its cell scores the `cns_plateau` stop (no `cap_only` fallback).
- `tools/readme_media.py` draws `docs/media/`: `scenes` (the six goal views, default look, one
  process per scene) and `demo` (`sift_ransac` on `cluttered2` from a start 110.5 mm / 15 deg off
  the goal, built by `demo_start` - not a protocol start; 0.99 mm after 160 steps; the GIF from the copied
  `compose_match_frame`, the debug view from the copied `DebugLog` + `debugview.render_still`;
  the flight's numbers and commits in `docs/media/demo.json`); `gif` re-encodes the GIF from the
  saved frames without flying.
- Weights for outside users (read from the code, not tried off the station): the DINOv3 cell
  loads with `local_files_only=True` on every call (`vsbench/vs/baselines/features.py`), so it
  never downloads - the gated `facebook/dinov3-vits16-pretrain-lvd1689m` must be in the HF
  cache first; the DINOv2 cells (`vitvs_dinov2`, `artvs`) call `torch.hub.load(...,
  source="github", skip_validation=True)`, which fetches the hub repo and the checkpoint on
  first use when they are not cached (`HF_HUB_OFFLINE` does not stop torch.hub).

## 11. Layout

```
vsbench/camera.py       CameraModel (K, size, near, far) and its OpenGL projection
vsbench/scene.py        SceneSpec: Plane, Box, Sphere, Mesh, Light, Material (plain data)
vsbench/reference.py    analytic projection, plane z-depth, poster homography (NumPy only)
vsbench/render/         base.py (Renderer protocol, make_renderer), one module per backend
vsbench/stdscenes.py    marker board, plane, poster, clutter and lighting scenes
vsbench/se3.py          hat, vee, exp/log on SO(3) and SE(3), pose_error, orthonormalise (NumPy only)
vsbench/motion.py       MotionLimits, clamp_twist, FreeCamera (NumPy only)
vsbench/interaction.py  interaction_matrix, predicted_flow, exact_flow (NumPy only)
vsbench/trajectory.py   Trajectory: append, save, load, replay, path_length (NumPy only)
vsbench/gates.py        marker detection, ORB matching, corner tracking shared by tests and tools
vsbench/smoke.py        python -m vsbench.smoke
vsbench/scenes.py       make_scene(name), the poster and marker target poses, the marker points
vsbench/methods/        base.py (VSMethod protocol, info keys, normalise_output), the registry,
                        oracle_pbvs.py, true_point_ibvs.py (privileged), feature_ibvs.py (sift_ibvs, orb_ibvs),
                        vssim_arms.py (the copied vs-sim arms as methods, U6b)
vsbench/episode.py      EpisodeConfig, run_episode, EpisodeLog (a Trajectory with errors, hashes, matches), the ten demo
                        starts - the Step-3 test loop, not the benchmark loop (that is vsbench/vs/bench/runner.py)
vsbench/checks.py       zero_at_target, hold_at_target, restoring_direction; vsbench/check.py: python -m vsbench.check
vsbench/replay.py       python -m vsbench.replay <log> [--video out.mp4]
vsbench/video.py        the four-panel debug video and the match sheet (imageio-ffmpeg, OpenCV fallback)
vsbench/demo.py         python -m vsbench.demo
vsbench/__main__.py     python -m vsbench fly / list: the one entry of the front page (routes to world/fly.py)
vsbench/world/          the MuJoCo world behind vs-sim's Scene contract (mujoco_scene, build, camera, assets),
                        look.py + configs/look_isaac_fit_v1.json (the Isaac look, U4b),
                        the bench driver fly.py, method_arm.py + configs/external_method.yaml (U6a)
tests/                  step 1: test_reference, test_projection, test_depth, test_homography, test_determinism
                        step 2: test_se3, test_motion, test_interaction, test_flow_rendered, test_trajectory
                        step 3: test_contract, test_leak, test_oracle, test_checks, test_true_point_ibvs,
                        test_feature_ibvs, test_episode_log, test_replay, test_video
tools/                  measurements and the reports (run_all.sh, run_step2.sh, run_step3.sh);
                        cpu_env.sh (the station rules for the vs-sim bridge, source it first);
                        readme_media.py (the front page's pictures in docs/media/)
examples/my_method.py   the README's method (my_ibvs)
docs/media/             demo.gif, scenes.png, debug_view.jpg, demo.json (the front page)
external/               the pinned vs-sim clone, the copy's source (git-ignored), and VSSIM_COMMIT (the pin, committed)
vsbench/vs/ ...          the vs-sim copy: vs/ (= vssim/), configs/, benchmark/, assets/, tests/, scripts/,
                        experiments/ - generated by tools/vssim_copy.py, never edited (vsbench/vs/PROVENANCE.md)
assets/poster.png       CC0 photograph (see assets/README.md)
results/                JSON records, contact sheets, bakeoff_report.md, step2_axes.png, STEP2_REPORT.md,
                        STEP3_REPORT.md, demo/<method>/ (one video and one match sheet per method; logs are not committed)
```

MIT licence. Dependencies come through pip; nothing is vendored.

---

## U8 — perturbations on the MuJoCo world (2026-10-23)

**What you can do.** Every perturbation mode of the copied `vsbench.vs.bench.perturb` now
runs on `MujocoScene`:

```
source tools/cpu_env.sh
$VSBENCH_PY -m vsbench.world.fly run --scene cluttered2 --arms sift_ransac --start-set capped30 \
    --limit 1 --max-iter 100 --perturb occluder --out results/my_run       # vs-sim's flag
#   --perturb light | distractors | occluder | noise | combined | ground:<name|path>
#             | swap:<instance_id> | swap:@<live name> | a+b   (--perturb-seed, default 20260918)
```

| mode | what the world does | cost on cluttered2 (CPU, load ~25) |
| --- | --- | --- |
| `noise` | nothing (the copied per-frame sensor model on the picture) | 0 |
| `light` | relights the picture through the scene's tone curve (below) | 0 (no recompile) |
| `ground:<name>` | the floor gets another texture (`perturb.GROUNDS` names or a path; `/workspace/...` → `$VSBENCH_WORKSPACE_ROOT`) | one recompile, 6.3–6.8 s |
| `occluder` | a grey cube between the goal camera and the scene | one recompile, 7.6 s |
| `distractors` | 2–5 extra YCB objects from the converted cache, on the floor in view | one recompile, 8.8 s |
| `swap:<id>` | `/World/Obj_0` hidden, `<id>` put at its model origin | one recompile, 6.8–8.2 s |
| `swap:@<name>` | instance suite: the named live object, box centre on the goal object's | one recompile (inst_drill 1.4–4.1 s) |
| `combined`, `a+b` | the parts, in order; undone in reverse | one recompile for all parts (7.7 s) |

Plus ~0.6 s to free the old model; the restore after `end` costs one more compile (7–8 s),
deferred to the next frame drawn - in a campaign it merges with the next episode's `begin`,
so an episode's `world_compiles` can list the previous episode's restore (`unedited: true`).

Per-mode numbers (begin / end, compile seconds, RSS): `results/bridge/u8_perturb.json`.

**How it works.** `vsbench/world/perturb_world.py` gives `MujocoScene` the verbs Isaac's scene
has (`build_poster`, `put_box`, `remove_prim(s)`, `place_extra_objects`, `set_prim_visible`,
`object_pose`, `object_world_bbox`, `set_lighting` / `lighting_snapshot` /
`restore_lighting`, `extra_placements`), under Isaac's prim paths (`/World/Ground`,
`/World/Obj_<i>` in layout order, `/World/Occluder`, `/World/Distractor_<k>`,
`/World/Swap_<k>`). MuJoCo bakes geometry and textures into the compiled model, so an edit
marks the world dirty and the next frame recompiles it — the old model and its GL contexts
are freed first (never two rooms in one process), and several edits (a chain's parts, one
episode's `end` plus the next one's `begin`) cost one compile. A recompile keeps the camera,
the look's renderer settings and MuJoCo's model statistics (`stat.extent` / `stat.center`)
of the unedited scene, so a pixel that sees the same unedited surface keeps its depth bit
for bit; when the world is back to the unedited scene it is compiled exactly as at start-up,
so the goal redrawn after `end` has the goal's md5 (gated, every mode).

**The goal picture is the unperturbed one** (as in vs-sim): the runner captures it before
the first `begin` and serves it from the cache at the goal pose. `MujocoScene` adds one
guard: a verb that edits the world captures the goal first if nobody has.

**The look under a perturbation (builder decision, recorded as D-11 in PORT_PLAN §8).** The
scene's look (`isaac_fit_v1`: fill share, light gain, shadow map and its fitted square,
tone curve) is fitted once on the unperturbed goal and **stays fixed**: an occluder, a
distractor, another floor or another object is drawn through the same renderer response,
as Isaac's fixed renderer would draw it. Nothing is re-fitted per perturbation. The cost,
measured (`results/bridge/u8_perturb.json`, `ground_mean_rgb_at_goal`): the curve is only
constrained by the colours of its own scene's goal picture, so a floor of very different
colour is extrapolated. The same two floors under each other's curve (mean RGB of the
ground at the goal): the industrial concrete under its own scene's curve (197, 190, 184),
under cluttered2's (191, 168, 166) - about 20 levels pinker; cluttered2's wood under its own
curve (160, 116, 102), under the drill scene's (152, 91, 53). Preview of every mode at the
goal pose: `results/bridge/u8_modes.png`. Scenes without a fitted curve (the instance suite) fly in
the `raw` look, as before.

**The light.** vs-sim's `light` scales the dome and the sun by the same `s` and gives both
the same colour temperature `T`, so every pixel's linear radiance changes by
`f_c = s * blackbody_c(T)` (USD's `UsdLux.BlackbodyTemperatureAsRgb`, luminance 1; Isaac's RTX
may use its own table). MuJoCo clamps each light's diffuse term at 1 and the fitted look
already drives the sun to ~1, so a brighter light cannot be drawn in the model; it is applied
inside the scene's tone curve instead: `y = 255 srgb(w / (1 + w))`, `w = f_c (floor + e^k
x^gamma)` (the `raw` look: `y = clip(f_c x)`), a 256-entry table, depth untouched. A light
state in which the dome and the sun differ in factor is refused.

**Refused, loudly:** an unknown prim (`ValueError`, as Isaac); a Nucleus asset (D-6, not on
disk) or an object missing from the converted cache (`FileNotFoundError` naming
`tools/convert_objects.py`); a non-square tiled poster; unequal light factors. On a poster
scene the copied modes themselves refuse `ground:` (no ground block) and `swap:` (no object
list). Nothing else is refused: every copied mode is feasible.

**The distractor pool** is the copied rule (the YCB library minus the scene's own objects,
82 − 8 ids); U8 converted the 74 YCB scans the cache did not hold (`tools/convert_objects.py
convert <ids>`, 74/74 ok, 4096² textures, 249 s, cache now 2.9 GB).

**Memory.** cluttered2 holds 3.5 GB after start-up; a perturbed world 3.5–4.3 GB (each
distractor adds a 4096² texture); the peak was 4.85 GB in the measure run and 5.55 GB in the
smoke flights (`combined`, with the arm and the frames). After each `end` the process sits at
3.54–3.80 GB (the allocator keeps ~0.2–0.3 GB of freed pages; no steady growth over the 10
modes - `rss_after_end` per mode in the measure record). Budget ~6 GB per cluttered2 process
when flying perturbations, never two rooms in one process.

**Smoke** (`results/bridge/u8/`, recorded, not thresholded; `sift_ransac`, 100 steps, CPU):
cluttered2 from 7.76 mm - none 1.20 mm (U4b's number), light 0.52, occluder 0.17,
distractors 0.73, `ground:concrete` 4.13, `ground:laminate` 2.98 (divergence abort),
noise 18.41, combined 0.44, `swap:ycb/006_mustard_bottle` 1.83, `+ground:concrete` 8.64
(divergence abort); inst_drill from 6.45 mm - none 0.23, `@same` 0.23, `@same_small` 3.77 mm
from the size-matched pose, `@other_small` lost (no SIFT matches on another drill).

**Tests and tools.** `tests/world/test_perturb_world.py` (the copied `test_perturb` contract
on the real world: industrial1_drill, inst_drill, hollywood; gate JSON
`results/bridge/u8_perturb_gates.json`), `tools/u8_perturb.py measure | smoke | summary`.

---

## U5 — the baselines in the copied mock and in MuJoCo (2026-10-23)

**What it is.** The same arm, start, landmark draw and loop, flown in two worlds: the copied
numpy `MockPlaneScene` (an exact inverse warp of the poster texture, no light) and
`MujocoScene(look="raw")`; plus a third, `mujoco960`, the MuJoCo scene drawn with the mock's
principal point (960, 540) instead of Isaac's (959.5, 539.5), which separates the half pixel
from the tone/filtering differences.

```
source tools/cpu_env.sh
PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py fly --world mock --arms vitvs_dinov3   # or mujoco, mujoco960
PYTHONPATH=. $VSBENCH_PY tools/u5_parity.py compare      # results/bridge/u5/parity.{md,json}
VSBENCH_WRITE_GATES=1 $VSBENCH_PY -m pytest tests/world/test_parity.py   # results/bridge/u5_parity.json
```

Defaults: hollywood, `near60` n_000–n_002 + `narrow20` start_00–01, 150 steps. Each record
carries a `u5` block (world, principal point drawn / reported, GL renderer, final pose, CPU
seconds, ART-VS's switch step, a compact per-step curve). No telemetry file is kept (~100 kB a
start); the curve is enough for the comparison.

**Result.** `sift_ransac` lands the same in all three worlds to ≤ 0.1 mm; ART-VS to ≤ 0.8 mm.
ViT-VS (both backbones) misses the 3 mm / 0.3° landing gate on one start (n_001), DINOv2 only
by its wobble, DINOv3 by settling on a different floor; ART-VS's switch does not fire on n_002
in either world. The half-pixel control shows these are not a renderer bias: half a pixel
alone moves ViT-VS's landing by up to 4 mm. Read a single ViT-VS landing in MuJoCo as one
draw from a ±2–4 mm spread, not as a number to compare to the mock (or to Isaac) start by
start; compare curve medians over many starts.

**The gate, re-stated at the final review** (old and new side by side in
`results/bridge/u5/parity.md`, the old verdicts kept as `original_gate` in the JSON): the
landing gate stays for SIFT and ART-VS; ViT-VS is gated on the curve medians (steps 101–150
within 3 mm / 0.5° per start, and their median spread no larger than the half-pixel
control's); ART-VS's switch on "the worlds agree whether and when (±5 steps)". DINOv2 passes,
DINOv3 misses by rotation on n_001 (0.53° > 0.5°; spread 0.162° vs 0.125°), the switch passes.
U5: accepted with findings.

**Costs (shared station, nice 19, 2 threads, load 19–31).** ART-VS 7–10 s/step once tiled
(36 DINOv2 tiles), ~0.7–1.3 s/step before; ViT-VS 0.85–1.8 s/step; SIFT 0.9–1.7 s/step; the mock
world's 1920×1080 numpy warp ≈ 0.7 s a frame. The full U5 set: ~26 000 CPU-s, 2 h 43 min on
two processes.

## U9 — the rig check twin (2026-10-23)

**What you can do.** vs-sim's Isaac rig check, on the MuJoCo world:

```
source tools/cpu_env.sh
$VSBENCH_PY -m vsbench.world.check_baselines --scenes hollywood cluttered2 \
    --arms sift_ransac vitvs_dinov3 artvs --near-steps 200 --far 1 --out results/my_check
$VSBENCH_PY -m vsbench.world.check_baselines summary --runs results/my_check --out results/my_check
#   vs-sim's flags: --far N --far-set --film --film-every --perturb --perturb-seed --budget ARM=N
#   --outline-iou --telemetry --clean-film --arm-plugin MODULE (vsbench.world.method_arm flies
#   the registered vsbench methods); the bridge's: --look --device (cpu) --samples
#   --frames full|preview|none --threads --allow-gpu (never during the Isaac evaluation)
```

Per scene x arm: **identity** (the camera at the goal, the goal picture as the live frame:
vs-sim's `ok`), **near** (the near sampler's first draw on the scene's goal, `n_000`, 7.76 mm /
2.69 deg on every scene - the start Isaac's check flies), **far** (`--far N` starts of
`--far-set`, to the cap). Writes vs-sim's `check.json` / `check.md`, the goal frame and the first
live frame, plus `records.jsonl` (the full copied record of every episode) and a `bridge` block.

**How it works.** No third copy of the check: `vsbench.world.check_baselines.reference()` loads
the copied `vsbench/scripts/check_baselines.py` as a module (its `sys.path` additions undone
after the load) and flies its `identity_check`, `episode_check`, `near_start`, `write_table`,
`parse_budgets` and plugin hook; the protocol helpers, station guard and provenance are
`vsbench.world.fly`'s. The only new logic is `main`'s scene loop on `MujocoScene` (one scene
alive at a time, built, goal-checked against the far set's file, flown, closed) and the
capture of each record (the reference module's `run_episode` is wrapped while the check runs;
the file is not edited). A vs-sim fix to the check arrives with the next pin move. Needs a
source checkout (the packaging excludes `vsbench/scripts`, as it excludes `tools/`).

**The near verdict (builder default, PORT_PLAN D-12).** vs-sim's script prints the near
landing but computes no verdict; its docstring says "it must land closer than it started, by a
lot". The twin writes `near_ok = final < start` (the sentence read literally) and reports
`final_over_start`; "by a lot" is not quantified there, so it is not thresholded here.

**Result** (`results/bridge/u9/README.md`, gate `results/bridge/u9_check.json`): identity
24 / 24 green; near 15 / 24 closer - SIFT and ORB on all six scenes, ViT-VS only on hollywood,
ART-VS on poster3 and cluttered3. Eight of the nine misses are misses in Isaac's own check from
the same start (the method, not the world); the ninth (ART-VS on hollywood, Isaac 1.6 mm, here
20.1 mm) misses as well without the look and on the copied numpy mock: ART-VS never leaves its
coarse phase from that start outside Isaac (the reverse happens on poster3). `tools/u9_diagnose.py`
flies one cell in `isaac_fit_v1` / `raw` / the mock with per-step telemetry for such questions.

**Cost.** 11.2 CPU-hours for the 24 cells with one `capped30` start each (3 h 03 min wall on two
processes at load ~24); a scene process 45-72 min; peak RSS 2.2-2.4 GB (posters, drill room),
4.4-4.8 GB (cluttered2 / cluttered3). Without `--far` a scene is ~15-25 min.

**Records.** Every scene block and entry carries `render_mode` (`mujoco-llvmpipe`) and the GL
renderer string; `check.json`'s `bridge` block the commits, the station env (torch 2 / 2, CUDA
off, `LP_NUM_THREADS=2`), the look and CPU seconds. `summary` refuses to merge mixed renderers
and puts Isaac's numbers (`$VSBENCH_ISAAC_CHECKS`, default `~/genvs_ws/vssim_out`: `check1` for
identity / near, `check2` / `check3` for the far row; read only) beside each cell - context,
never pooled (D-9).
