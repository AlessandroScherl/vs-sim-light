# VS-Bench Step 3 brief: servo loop

Close the loop for the first time: a method sees images and returns a twist, the camera moves, and everything is logged and replayable. Build the method interface, three basic controllers, the setup checks, the replay-based debug video and a one-command demo. Every later baseline and every student method plugs into what is built here.

Context: VS-Bench is a lightweight visual-servoing simulator and benchmark for students and researchers. Step 1 delivered the MuJoCo renderer. Step 2 delivered SE(3) math, `FreeCamera`, the point interaction matrix and the trajectory log. This step adds the controller side and the debugging tools.

## Ground rules

Only the simplest controllers are built here, but the interface they use is the one every later method will use, so it has to be right.

- Read first: `README`, `results/STEP2_REPORT.md`, the two earlier briefs in `docs/`, `vsbench/se3.py`, `motion.py`, `interaction.py`, `trajectory.py`, `render/base.py`, and wherever the poster, marker and clutter scenes are defined.
- Reuse, do not duplicate: `FreeCamera` for motion, `interaction_matrix` for every IBVS law, `Trajectory` as the base of the episode log, `log_se3` for pose error.
- If the three scenes live in test code, move them unchanged into `vsbench/scenes.py` behind `make_scene(name)`.
- The harness owns the truth. A method receives images, optional depth and intrinsics, nothing else. Code under `vsbench/methods/` must not import the episode loop, a renderer or a scene.
- Out of scope: perturbations, noisy depth, SSIM or keypoint stop rules, metrics beyond true pose error, pose samplers, parallel runs, external baselines.
- One new dependency is allowed: `imageio-ffmpeg` for video. Ask before adding anything else.
- Install rules as before. Everything runs on Linux without GPU or display, and all Step 1 and 2 gates stay green.

## What to build

One method protocol with a registry, three built-in controllers, and the tooling around them.

```
vsbench/
  methods/
    base.py             VSMethod protocol, info keys, output normalisation
    __init__.py         make_method(name, **params), register decorator
    oracle_pbvs.py      privileged
    true_point_ibvs.py  privileged
    feature_ibvs.py     sift_ibvs, orb_ibvs
  episode.py            EpisodeConfig, run_episode, EpisodeLog
  checks.py             zero_at_target, hold_at_target, restoring_direction
  replay.py             python -m vsbench.replay
  video.py              panels, 3D inset, encoder
  demo.py               python -m vsbench.demo
  scenes.py             make_scene(name), if not already present
tests/                  test_contract, test_leak, test_oracle, test_checks,
                        test_true_point_ibvs, test_feature_ibvs,
                        test_episode_log, test_replay, test_video
tools/                  step3_report.py, run_step3.sh
```

```python
class VSMethod(Protocol):
    name: str
    wants_depth: Literal["none", "scalar", "dense"]
    privileged: bool          # True only for the two test instruments

    def configure(self, K: np.ndarray, width: int, height: int, dt: float) -> None: ...
    def reset(self) -> None: ...
    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None) -> None: ...
    def get_control_rate(self, current_rgb, current_depth=None): ...
        # returns twist (6,), or (twist, info)
    def observe_truth(self, T_wc, T_wc_target) -> None: ...   # called only when privileged
```

The optional `info` dict may hold `kp_cur` (M, 2), `kp_des` (M, 2), `inliers` (M,) bool, `error_norm`, `observer_s`, `controller_s` and `lost`. Every key is optional.

| Name | Privileged | Law | Purpose |
| --- | --- | --- | --- |
| `oracle_pbvs` | Yes | `twist = gain * log_se3(inv(T_wc) @ T_wc_target)` | Proves the loop. The pose error must shrink by exactly `1 - gain*dt` per step. |
| `true_point_ibvs` | Yes | Projects known 3D points with the true poses. `twist = -gain * pinv(L) @ (s - s_target)`, with `L` stacked from `interaction_matrix` and true depth. | Proves the IBVS law and its signs without image processing. Needs no renderer. |
| `sift_ibvs`, `orb_ibvs` | No | Describe the target once. Per step: detect, match with a ratio test, reject outliers with RANSAC, read depth at the inlier keypoints, then the same IBVS law. | The basic real controller, and the template for student methods. |

- Feature depth: `dense` reads the current depth at each keypoint and drops keypoints without depth. `scalar` uses the distance prior for all. It is a constructor argument, default `dense`.
- RANSAC model: homography with a 3 px threshold by default, fundamental matrix selectable for non-planar scenes.
- With fewer than 8 inliers the method returns a zero twist and sets `lost`.
- Gains default to 1.0 and are constructor arguments.
- A student adds a method with one class and one `@register("my_method")` line.

## Loop, log, replay and video

The loop is one function, and everything it produces goes into one `.npz` that can be turned into a video later.

```python
@dataclass(frozen=True)
class EpisodeConfig:
    dt: float = 0.04
    max_steps: int = 3000
    limits: MotionLimits = MotionLimits()
    stop_tol: tuple[float, float] | None = (0.002, np.deg2rad(0.2))   # metres, radians
    hold_steps: int = 25

def run_episode(renderer, method, T_wc0, T_wc_target,
                cfg: EpisodeConfig, scene_name: str) -> EpisodeLog: ...
```

- **Setup.** Render the target pose, then call `configure`, `reset` and `set_target`. The distance prior is the median valid depth of the target view. A method gets depth only in the form it asked for.
- **Each step.** Render, call `observe_truth` if privileged, call `get_control_rate` and time it, pass the twist to `FreeCamera.step`, log.
- **End.** Status is `converged` when the true pose error stays below `stop_tol` for `hold_steps`, `timeout` at `max_steps`, or `method_error` on an exception or a non-finite twist. Stopping on true pose is a test-only rule that Step 7 replaces with a real stop policy.
- **Log.** `EpisodeLog` extends `Trajectory` in the same file: true translation and rotation error, the method's `error_norm`, time per call, a 64-bit hash of every RGB frame, and up to 500 matches per step. Matches are `uint16` pixel coordinates times 32, with an inlier flag and a count per step, because `float16` is too coarse above 256 px. The meta string holds scene name, camera, renderer variant, method name and parameters, target pose and git commit.
- **Replay.** `python -m vsbench.replay <log> [--video out.mp4]` rebuilds the scene, re-renders every logged pose and compares frame hashes. It never runs the method. A mismatch, for example after switching from `osmesa` to `glfw`, is reported, and the video is still written and marked as not identical.
- **Video.** 960 x 720, every second frame, H.264 through `imageio-ffmpeg`, OpenCV `mp4v` as fallback.
  - Top row: current and desired image side by side with match lines, inliers green and outliers red. Without keypoints, the absolute difference image instead.
  - Bottom left: 3D inset with the target frustum, the current frustum, the path so far, one arrow for the applied linear velocity and one for the angular velocity. Drawn with OpenCV from a fixed third-person view.
  - Bottom right: translation error, rotation error and twist norms over time.
- **Check.** `python -m vsbench.check --method <name>` runs the three setup checks on the poster scene and prints pass or fail per axis.
- **Demo.** `python -m vsbench.demo [--method sift_ibvs] [--scene poster]` runs the ten demo offsets, prints status, final error and steps per start, and saves the logs under `results/demo/<method>/` plus one video.

## Work packages

Seven packages in strict order. The loop is proven with the oracle before any image processing exists, and the checks are proven able to fail before they are trusted.

1. **Interface, loop and oracle.** `methods/base.py`, the registry, `oracle_pbvs.py`, `episode.py` with a plain `Trajectory` log. *Done when* the contract, leak, oracle decay and oracle convergence gates pass.
2. **Setup checks.** `checks.py` and its command. *Done when* the oracle passes all three checks, and a test-only oracle with one flipped axis fails them on exactly that axis.
3. **True-point IBVS.** `true_point_ibvs.py` on the 13 marker points, no renderer. *Done when* it passes the setup checks and the IBVS decay gate.
4. **Feature IBVS.** `feature_ibvs.py` with SIFT and ORB. *Done when* both pass the setup checks and SIFT passes the feature convergence gate on the poster scene.
5. **Episode log and replay.** `EpisodeLog`, `replay.py`. *Done when* the log round-trip and replay gates pass.
6. **Debug video.** `video.py`. *Done when* the video gate passes and one video per method has been written for a human to look at.
7. **Demo, README and report.** `demo.py`, README sections for running the demo and writing your own method, and `tools/run_step3.sh` producing `results/STEP3_REPORT.md`. *Done when* a fresh clone regenerates the report and the demo output with one command.

## Gates and deliverables

Thirteen gates, all pass or fail, at 640 x 480 with seeded inputs and clean images. Pose error means the translation norm and the rotation angle between current and target pose.

| Gate | Setup | Pass |
| --- | --- | --- |
| Contract | Every registered method through `configure`, `reset`, `set_target`, `get_control_rate` on the poster scene | Finite `(6,)` float twist, alone or with an info dict. Inputs are not modified. |
| Leak | A spy method with `privileged = False`; an import-graph test over `vsbench/methods/` | `observe_truth` is never called. No import of the episode loop, a renderer or a scene. |
| Oracle decay | 20 random starts inside the clamp, gain 1.0 | Norm of `log_se3` of the pose error equals `(1 - gain*dt)^k` times its start value within 1e-9 over 100 steps |
| Oracle convergence | 50 random starts up to 1 m and 170 deg | Below 0.1 mm and 0.01 deg within 1000 steps |
| Zero at target | Every method at the exact target pose | Linear and angular twist norms below 1e-6 |
| Hold at target | 50 steps from the target pose | Pose moves less than 1e-6 m and 1e-6 rad |
| Restoring direction | Plus and minus 2 cm, then plus and minus 2 deg, on each axis: 12 starts per method | On the first step, the twist component on that axis has the sign of the same component of `log_se3(inv(T_wc) @ T_wc_target)`. The episode converges. |
| Negative control | Test-only oracle with the sign of one axis flipped | The checks fail on that axis and only there |
| IBVS decay | `true_point_ibvs`, offsets up to 2 cm and 2 deg, gain 1.0 | Fitted per-step ratio of the feature error norm over the first 50 steps within 0.002 of `1 - gain*dt` |
| Feature convergence | `sift_ibvs`, poster scene, the ten offsets below, dense depth | All ten below 5 mm and 0.5 deg within 500 steps. The ORB count is reported, not gated. |
| Log round trip | Save, then load | Arrays and meta identical |
| Replay | Re-render a logged episode in the same GL variant | All frame hashes identical |
| Video | A 300-step episode | Decodes, 150 frames, 960 x 720, under 5 MB |

The target pose looks straight at the poster centre from the distance where the poster spans about 70 % of the image width. Each start is `T_wc0 = T_wc_target @ exp_se3(offset)`, with the offset given in the target camera frame. If a start fails, report it with its video. Do not change the list.

| Start | Translation, cm | Rotation, deg |
| --- | --- | --- |
| 1 | (5, 0, 0) | (0, 0, 0) |
| 2 | (0, 5, 0) | (0, 0, 0) |
| 3 | (0, 0, -10) | (0, 0, 0) |
| 4 | (0, 0, 0) | (0, 0, 15) |
| 5 | (0, 0, 0) | (0, 8, 0) |
| 6 | (0, 0, 0) | (-8, 0, 0) |
| 7 | (8, -5, -10) | (0, 0, 20) |
| 8 | (-10, 6, 8) | (5, 5, -25) |
| 9 | (12, 8, -15) | (6, -8, 40) |
| 10 | (-15, -10, -20) | (-6, 8, -60) |

Deliverables:

- The modules and test files above, green in CI together with all Step 1 and 2 gates.
- `results/STEP3_REPORT.md` with every gate's measured numbers, the demo table for SIFT and ORB, and time per `get_control_rate` call.
- One demo video per method under `results/demo/`.
- README sections: run the demo, run the checks, write your own method in about 20 lines.
