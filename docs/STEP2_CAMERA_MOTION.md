# VS-Bench Step 2 brief: camera motion

Add the motion layer on top of the MuJoCo renderer: integrate a camera-frame twist on SE(3) at a fixed step, clamp it, log the trajectory, and prove with tests that the image moves exactly as the twist predicts. The controller in Step 3 and the motion blur in Step 4 reuse these modules unchanged.

Context: VS-Bench is a lightweight visual-servoing simulator and benchmark for students and researchers. A free-flying camera is moved by a 6-DoF velocity command, and the simulator returns RGB and depth at each pose. Step 1 delivered the renderer and chose MuJoCo. This step covers motion only.

## Ground rules

The renderer is settled, so this step adds motion on top of it and changes nothing underneath.

- Read first: `README`, `bakeoff_report.md`, `vsbench/camera.py`, `vsbench/reference.py`, `vsbench/render/base.py` and `vsbench/render/mujoco_backend.py`. Reuse the projection functions in `reference.py`, and extend existing pose helpers instead of adding a second set.
- MuJoCo is the renderer. Do not change the `Renderer` interface or the backend. If a Step 2 gate exposes a backend bug, stop and report it.
- The motion code is pure NumPy. `se3.py`, `motion.py`, `interaction.py` and `trajectory.py` never import MuJoCo or a renderer. SciPy is allowed in tests only, as an oracle.
- No physics. `mj_step` is never called. Simulated time is `k * dt` and belongs to the motion module.
- Out of scope: controllers, the method interface, stop rules, video, perturbations, motion models beyond the clamp, parallel runs.
- Install rules as in Step 1: pip only, Python 3.10 to 3.12, Linux without GPU or display must pass. All Step 1 gates stay green.

## What to build

Four small modules: SE(3) math, a free camera that integrates a clamped twist, the point interaction matrix, and a trajectory log.

```
vsbench/
  se3.py           hat, vee, exp_so3, log_so3, exp_se3, log_se3,
                   pose_error, orthonormalise
  motion.py        MotionLimits, clamp_twist, FreeCamera
  interaction.py   interaction_matrix, predicted_flow, exact_flow
  trajectory.py    Trajectory: append, save, load, replay, path_length
tests/             test_se3, test_motion, test_interaction,
                   test_flow_rendered, test_trajectory
tools/             fly_axes.py
```

```python
@dataclass(frozen=True)
class MotionLimits:
    v_max: float = 2.0       # m/s, loose on purpose, tightened in Step 7
    w_max: float = 4.0       # rad/s

class FreeCamera:
    def __init__(self, T_wc0: np.ndarray, dt: float = 0.04,
                 limits: MotionLimits = MotionLimits()): ...
    T_wc: np.ndarray         # current 4x4 camera-to-world pose
    k: int                   # step counter
    t: float                 # simulated seconds, computed as k * dt
    def step(self, twist: np.ndarray) -> np.ndarray: ...   # returns the applied twist
    def reset(self, T_wc0: np.ndarray) -> None: ...

class Trajectory:
    def append(self, t, T_wc, twist_cmd, twist_applied) -> None: ...
    def save(self, path) -> None: ...      # one compressed .npz, float64, plus a JSON meta string
    @classmethod
    def load(cls, path) -> "Trajectory": ...
    def replay(self) -> np.ndarray: ...    # re-integrates the applied twists from the first pose
    def path_length(self) -> tuple[float, float]: ...   # metres, radians
```

Conventions:

- The twist is `[vx, vy, vz, wx, wy, wz]`, linear part first, in m/s and rad/s. It is the body twist of the camera, expressed in the optical frame: x right, y down, z forward.
- The update is `T_wc[k+1] = T_wc[k] @ exp_se3(twist * dt)`. Right-multiplication, because the twist is in the camera frame.
- `exp_se3` is the closed form, with a series expansion below 1e-6 rad. The rotation is re-orthonormalised by SVD after every step.
- The clamp scales the linear and angular parts separately and keeps their direction. A NaN or infinite twist raises `ValueError`. `step` is the only place where a command becomes motion, so later motion models slot in there.
- The log is the start of the Step 3 episode log, which adds errors and matches to the same file.

Interaction matrix for a point at normalised coordinates `x = (u - cx) / fx`, `y = (v - cy) / fy` and z-depth `Z`:

```
L = [[-1/Z,    0,  x/Z,      x*y,  -(1 + x*x),   y],
     [   0, -1/Z,  y/Z,  1 + y*y,        -x*y,  -x]]

pixel flow over a short time tau = diag(fx, fy) @ L @ twist * tau
```

`exact_flow` does not linearise: it back-projects the pixel with its depth, moves the point into the new camera frame with the inverse of `exp_se3(twist * tau)`, and projects again.

What each positive axis must do, for the tests and for checking the axis sheet by eye:

| Positive | Camera | Image content |
| --- | --- | --- |
| `vx` | Slides right | Moves left |
| `vy` | Slides down | Moves up |
| `vz` | Moves forward | Expands from the centre |
| `wx` | Tilts up | Moves down |
| `wy` | Pans right | Moves left |
| `wz` | Rolls clockwise, seen from behind | Rotates counter-clockwise |

## Work packages

Six packages in strict order. A package starts only when the previous done-when holds, and each ends with a commit.

1. **SE(3) math.** `se3.py` and `test_se3`, with `scipy.linalg.expm` and `logm` as the oracle. *Done when* the round-trip and orthonormality gates pass.
2. **Free camera.** `motion.py` and `test_motion`. *Done when* the composition, direction and clamp gates pass.
3. **Interaction matrix.** `interaction.py` and `test_interaction`, no renderer involved. *Done when* the analytic flow gate passes for each axis and for one random twist.
4. **Rendered flow.** `test_flow_rendered` with the MuJoCo backend on the poster and clutter scenes. *Done when* the rendered flow gate passes on Linux without GPU.
5. **Trajectory log.** `trajectory.py` and `test_trajectory`. *Done when* the round-trip and replay gates pass.
6. **Axis sheet and report.** `tools/fly_axes.py` writes `results/step2_axes.png`: one row per axis at 0.1 m/s or 10 deg/s, with the start frame, the frame after 1 s, and flow arrows, predicted in green and measured in red. `STEP2_REPORT.md` lists every gate with its measured numbers and the time per `FreeCamera.step`. The README gains a ten-line example that moves the camera and renders. *Done when* a fresh clone regenerates the sheet and the report with one command.

## Gates and deliverables

Nine gates, all pass or fail, at 640 x 480 with seeded inputs. The first six need no renderer.

| Gate | Setup | Pass |
| --- | --- | --- |
| Exp and log round trip | 1000 random twists with rotation below pi, plus rotation angles of 1e-9, 1e-6 and 1e-3 rad. Compared with the SciPy oracle. | Max absolute error below 1e-10, no NaN |
| Orthonormality | 10,000 random clamped steps | Norm of `R.T @ R - I` below 1e-12, determinant within 1e-12 of 1 |
| Composition | A constant random twist: 1000 steps of `dt` against one exponential over `1000 * dt` | Difference below 1e-9 m and 1e-9 rad |
| Direction | From a tilted start pose, each axis alone for 1 s | Translation equals speed times the matching column of `R` within 1e-12. Rotation is a right-hand turn about the matching camera axis. Signs agree with the axis table. |
| Clamp | Twists above the limits, NaN, infinity | Direction kept and norm equal to the limit; NaN and infinity raise |
| Analytic flow | 200 random pixels at depths of 0.3 to 2 m. Each axis alone and one random twist, scaled to a median flow of about 1 px. `predicted_flow` against `exact_flow`. | Difference below 0.01 px, and it shrinks about fourfold when `tau` is halved |
| Rendered flow | Poster and clutter scenes. Each axis alone and one random twist, scaled to a median predicted flow of about 5 px. Shi-Tomasi corners tracked from frame A to frame B with pyramidal Lucas-Kanade. Prediction is `exact_flow` with the rendered depth of frame A. Points within 5 px of a depth edge or the border are dropped. | At least 100 tracked points. Median endpoint error at most 0.3 px. Median cosine between measured and predicted flow at least 0.98, over points with more than 1 px of predicted flow. |
| Trajectory round trip | Save, then load | Arrays identical |
| Replay | Re-integrate the applied twists from the first pose | Poses bit-identical to the log |

If the clutter scene gives fewer than 100 tracked points, add a textured floor under it. Do not lower the count.

Deliverables:

- The four modules and five test files, green in CI together with all Step 1 gates.
- `results/step2_axes.png` and `STEP2_REPORT.md`.
- The README example for students.
