# vsbench

VS-Bench, a lightweight visual-servoing simulator and benchmark. Step 1: one `Renderer`
interface with three backends (MuJoCo, PyBullet, Open3D), the gates that check each one
against analytic reference math, the measurements behind the choice, and
`bakeoff_report.md`; MuJoCo won. Step 2: the motion layer, a free-flying camera that
integrates a clamped 6-DoF velocity command on SE(3), the point interaction matrix, a
trajectory log, and the gates that prove the image moves exactly as the twist predicts.
Step 3: the servo loop, one method interface with a registry, three basic controllers,
the setup checks, an episode log that replays into a debug video, and a demo.
Metrics beyond the true pose error and perturbations come later.

## Install and first render

```
pip install -e ".[all,dev]"          # or one of [mujoco] [pybullet] [open3d]
python -m vsbench.smoke              # first backend that works here; writes smoke.png
python -m vsbench.smoke --backend mujoco-glfw     # a laptop with a display
```

On a Linux machine without a display or GPU, MuJoCo needs one of: the `libosmesa6`
package (`MUJOCO_GL=osmesa`, `--backend mujoco-osmesa`) or Mesa's EGL (`libegl1` and
`libgl1-mesa-dri`, usually present) with `VSBENCH_SOFTWARE_GL=1 --backend mujoco-egl`.
PyBullet's TinyRenderer and Open3D's CPU mode need nothing else. `VSBENCH_SOFTWARE_GL=1`
forces Mesa's llvmpipe for every backend, so a machine with a GPU never touches it.

## Result of the bake-off (station, 2026-09-18)

`results/bakeoff_report.md`: MuJoCo is the pick. MuJoCo (EGL/llvmpipe, OSMesa, and on the
RTX 4090) and PyBullet's TinyRenderer pass all four gates; Open3D fails cross-process
determinism on the CPU (an intermittent artefact on one mesh under llvmpipe) and returns
an unusable depth image on the GPU. PyBullet has no wheels for Python 3.12, macOS or
Windows and renders the clutter scene at 11 fps; MuJoCo renders it at about 100 fps at
640 x 480 in one CPU process (380 fps without anti-aliasing, 1140 fps on the GPU). Open3D
has the sharpest slanted poster but 11 fps on the clutter scene on the CPU. GPU and CPU
MuJoCo renders agree to 44.6 dB; EGL and OSMesa are bit-identical.

## The interface

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

## Moving the camera (step 2)

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

## Closing the loop (step 3)

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

### Write your own method

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

## Gates and measurements

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

## The vs-sim bridge: environment and pin (port unit U0)

The port of vs-sim's scenes, camera, objects and published baselines onto this simulator is
planned in `docs/PORT_PLAN.md`. Owner decision 2026-10-22: the vs-sim code is **copied** into
this repository, which stands alone and stays separate from vs-sim for good. A local clone of
vs-sim sits in `external/vs-sim` (git-ignored), checked out at the commit in
`external/VSSIM_COMMIT` (committed): it is the pinned **source** of that copy and of a drift
check, not a runtime dependency, and it is **not** pip-installed in the venv (its fast tier is
run from the clone's own directory, where `python -m pytest` finds the `vssim` package).
Weights are a copy, outside the repository, so the shared caches are never opened:

```
source tools/cpu_env.sh                                   # no GPU, llvmpipe, 2 threads, nice 19, offline weights
git clone ~/runpod_dev/vs-sim external/vs-sim             # a local clone only reads the source
git -C external/vs-sim checkout "$(cat external/VSSIM_COMMIT)"
$VSBENCH_PY -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch torchvision
$VSBENCH_PY -m pip install --only-binary=:all: "transformers>=4.57" "PyYAML>=6" "scikit-learn>=1.0" usd-core "pytest>=8"
# weights (~250 MB), copied once from the station's caches:
#   ~/genvs_ws/vsbench_assets/weights/hf/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/   (HF_HOME)
#   ~/genvs_ws/vsbench_assets/weights/torch/hub/{checkpoints/dinov2_vits14_pretrain.pth,
#                                                facebookresearch_dinov2_main/, trusted_list}  (TORCH_HOME)
cd external/vs-sim && $VSBENCH_PY -m pytest tests/fast -q   # vs-sim's fast tier in this venv (from the clone)
```

`~/genvs_ws/vsbench_assets/` (outside the repository) is the bridge's cache: `weights/` now,
the converted objects later. Torch is the CPU build (`+cpu`); `torch.cuda.is_available()` is
`False` by construction. Known drift: OpenCV 5.0 has no `AKAZE_create`, so vs-sim's AKAZE
arms cannot be built in this venv, and its classical fixture (pinned to OpenCV 4.10 / 4.11)
skips (with OpenCV 4.11, the flight interpreter's version, those tests pass); everything else
in the fast tier passes. The measured record is `results/bridge/u0_env.json`.

## Layout

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
                        oracle_pbvs.py, true_point_ibvs.py (privileged), feature_ibvs.py (sift_ibvs, orb_ibvs)
vsbench/episode.py      EpisodeConfig, run_episode, EpisodeLog (a Trajectory with errors, hashes, matches), the ten demo starts
vsbench/checks.py       zero_at_target, hold_at_target, restoring_direction; vsbench/check.py: python -m vsbench.check
vsbench/replay.py       python -m vsbench.replay <log> [--video out.mp4]
vsbench/video.py        the four-panel debug video and the match sheet (imageio-ffmpeg, OpenCV fallback)
vsbench/demo.py         python -m vsbench.demo
tests/                  step 1: test_reference, test_projection, test_depth, test_homography, test_determinism
                        step 2: test_se3, test_motion, test_interaction, test_flow_rendered, test_trajectory
                        step 3: test_contract, test_leak, test_oracle, test_checks, test_true_point_ibvs,
                        test_feature_ibvs, test_episode_log, test_replay, test_video
tools/                  measurements and the reports (run_all.sh, run_step2.sh, run_step3.sh);
                        cpu_env.sh (the station rules for the vs-sim bridge, source it first)
external/               the pinned vs-sim clone, the copy's source (git-ignored), and VSSIM_COMMIT (the pin, committed)
assets/poster.png       CC0 photograph (see assets/README.md)
results/                JSON records, contact sheets, bakeoff_report.md, step2_axes.png, STEP2_REPORT.md,
                        STEP3_REPORT.md, demo/<method>/ (one video and one match sheet per method; logs are not committed)
```

MIT licence. Dependencies come through pip; nothing is vendored.
