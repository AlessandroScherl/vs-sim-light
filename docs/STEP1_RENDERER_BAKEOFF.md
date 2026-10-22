# VS-Bench Step 1 brief: renderer bake-off

Build one `Renderer` interface with three backends (MuJoCo, PyBullet, Open3D), run the same correctness gates and measurements on each, and generate a report that picks the VS-Bench renderer. Nothing is throwaway: the interface, the winning backend and the gate tests are Step 1 of the main plan.

Context: VS-Bench is a lightweight visual-servoing simulator and benchmark. A free-flying camera is moved by a 6-DoF velocity command, and the simulator returns RGB and depth at each pose. This step covers rendering only.

## Ground rules

The users are students on ordinary laptops plus one headless lab server, so install and first use must be trivial. The target: `pip install -e .`, then `python -m vsbench.smoke` shows a rendered image and a frame rate within a minute.

- pip only. No Docker, no ROS, no conda requirement, no GPU requirement. At most one documented system package per OS.
- Python 3.10 to 3.12, developed on 3.11. Linux without GPU and without display must work. macOS and Windows laptops should work.
- Rendering only. No physics, controller, sampling, metrics or perturbations in this step.
- MIT licence. Dependencies come through pip; no third-party source is copied into the repo.
- Every backend is optional at import time. A missing package skips that backend's tests and never breaks the others.
- The API hints below come from memory. Check each against the library's current docs before relying on it.

## What to build

One small package: plain-data camera and scene types, analytic reference math, one `Renderer` protocol, and three backends behind a factory.

```
vsbench/
  camera.py            CameraModel
  scene.py             SceneSpec: Plane, Box, Sphere, Mesh, Light
  reference.py         analytic projection, plane depth, homography (NumPy only)
  render/
    base.py            Renderer protocol, make_renderer(name, scene, cam)
    mujoco_backend.py
    pybullet_backend.py
    open3d_backend.py
  smoke.py             one image plus fps, for students
tests/                 test_reference, test_projection, test_depth,
                       test_homography, test_determinism
tools/                 bench_speed.py, quality_sheet.py, bakeoff_report.py
assets/                poster.png (CC0), generated marker board
```

```python
@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray          # 3x3, pixels; fx != fy and off-centre cx, cy must work
    width: int
    height: int
    near: float = 0.05
    far: float = 10.0

class Renderer(Protocol):
    name: str              # "mujoco-osmesa", "mujoco-egl", "pybullet-tiny", ...
    def __init__(self, scene: SceneSpec, cam: CameraModel): ...
    def render(self, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
        # rgb uint8 (H, W, 3); depth float32 (H, W), z-depth in metres, 0 where nothing is hit
    def set_light(self, direction, intensity: float, color=(1, 1, 1)) -> None: ...
    def close(self) -> None: ...
```

Conventions:

- `T_wc` is a 4 x 4 camera-to-world pose. The optical frame is x right, y down, z forward, as in OpenCV. Each backend converts internally, and no caller ever sees an OpenGL frame.
- Depth is z-depth, not ray length. A backend that returns ray length or a normalised buffer converts it inside.
- `SceneSpec` is plain data, and each backend builds its native scene from it. A poster is a textured quad with a size in metres, never a tiled ground plane.
- One renderer per process. Construction may be slow; `render` must not reload assets.

Backend hints (verify against current docs):

| Backend | Setup | Camera and depth | Watch for |
| --- | --- | --- | --- |
| MuJoCo | Model built from `SceneSpec` with `mujoco.MjSpec` or an MJCF string. `MUJOCO_GL` set to `osmesa`, `egl` or `glfw` before import. | Camera on a mocap body: set `mocap_pos` and `mocap_quat`, call `mj_forward`. Intrinsics through the camera's pixel-unit focal and principal-point attributes. `mujoco.Renderer` with `enable_depth_rendering()`. | Offscreen buffer size under `<visual><global>` must cover 1440 x 1080. Anti-aliasing through `offsamples`. Poster as a mesh quad or thin box so the texture does not tile. OSMesa needs the Mesa system package on Linux. |
| PyBullet | `p.connect(p.DIRECT)`, TinyRenderer by default. GPU variant: `loadPlugin` of `eglRenderer`, then `ER_BULLET_HARDWARE_OPENGL`. | Projection matrix built from `K`, not from a field of view. Depth is `far*near / (far - (far - near)*zbuf)`. | Check `p.isNumpyEnabled()`, since some wheels return tuples. Poster as an OBJ quad with UVs plus `loadTexture`. |
| Open3D | `EGL_PLATFORM=surfaceless` before import gives CPU rendering on Linux. `OffscreenRenderer(w, h)`. | `setup_camera(K, T_cw, w, h)`. `render_to_depth_image(z_in_view_space=True)`. | Turn post-processing off so colours are raw. CPU mode is Linux only. |

## Work packages

Six packages in strict order. A package starts only when the previous done-when holds, and each ends with a commit.

1. **Skeleton and reference math.** Package, `pyproject.toml`, pytest, CI on Linux without GPU. `CameraModel`, `SceneSpec`, `reference.py`. *Done when* `test_reference` passes: the analytic homography agrees with `cv2.projectPoints` to 1e-6 px.
2. **MuJoCo backend.** The expected winner goes first. *Done when* all four gates pass with `MUJOCO_GL=osmesa` on a machine without GPU.
3. **PyBullet backend.** TinyRenderer is required, the EGL variant only if a GPU is present. *Done when* the same gates pass, or each failure is documented with an image.
4. **Open3D backend.** Half-day time-box. *Done when* the gates pass, or the time-box ends and the blocker is written down. A blocker is a valid result, because it says something about ease of use.
5. **Measurements.** Run the tools in the next section for every backend variant that passed its gates. *Done when* the raw numbers are in `results/*.json`.
6. **Report and smoke test.** `bakeoff_report.md` generated from the JSON, plus `python -m vsbench.smoke --backend <name>`. *Done when* a fresh clone reproduces the report with one command.

## Gates and measurements

Gates are pass or fail and identical for every backend. Measurements are numbers for the report. Tests are parametrised over backend names, run at 640 x 480 and once at 1440 x 1080, and use seeded poses.

| Gate | Setup | Pass |
| --- | --- | --- |
| Projection | 13 unlit coloured markers at known 3D points, 9 on a plane and 4 off it. 20 poses. `K` with fx different from fy and the principal point shifted by (+17, -11) px. Centroids by colour threshold, compared with `reference.py`. | Mean error at most 0.5 px, max at most 1.0 px |
| Depth | One plane, 20 poses, 0.3 to 1.5 m, slants up to 60 deg. Compared with analytic z-depth per pixel. | Median absolute error at most 1 mm, 99th percentile at most 3 mm |
| Homography | Poster quad, 20 poses. Reference is `cv2.warpPerspective` of the texture with the analytic homography. ORB matches between render and reference. | Median match displacement at most 0.5 px, at least 100 matches |
| Determinism | Same pose twice in one process, and once in each of two fresh processes | Arrays identical |

The shifted principal point matters: APIs that only take a field of view ignore it without any error.

| Measurement | How | Output |
| --- | --- | --- |
| Speed | Median ms per frame over 200 frames after 20 warm-up frames, RGB plus depth. Poster scene and a light clutter scene of 10 primitives and 3 meshes, about 50k triangles. Both resolutions. | ms per frame, fps |
| Parallel scaling | 1, 2, 4 and 8 processes, one renderer each | Total and per-process fps |
| Poster quality | Poster at 1.5 m with a 70 deg slant, and at 3x minification. Reference is a 4x supersampled warp reduced with `INTER_AREA`. | Contact sheet PNG; ORB and SIFT inlier counts against the reference |
| Cross-variant difference | Same poses on the CPU and GPU variants of one backend | PSNR and max absolute pixel difference |
| Lighting control | Move the light, change intensity, add a second light | Yes or no per capability, one image pair |
| Install friction | Fresh venv. Exact commands, system packages, time to first image, errors. | One log per OS actually tested; the others marked untested |

Every JSON record stores CPU model, core count, OS, Python version, package versions, backend variant and GPU if any.

## Decision rule and deliverables

The report applies five criteria in order and the first difference decides; a tie goes to MuJoCo. It states the pick and the numbers behind it, and the final call stays with the project owner.

1. Passes all four gates on Linux with no GPU and no display. Otherwise the backend is out.
2. Installs with pip plus at most one system package, and has official wheels for Linux, macOS and Windows on Python 3.10 to 3.12.
3. Reaches at least 25 fps at 640 x 480 in one CPU-only process, which is real time for the 25 Hz loop. Faster wins.
4. Better poster quality: higher inlier counts against the reference, then the contact sheet by eye.
5. Better maintained: date of the last release and Python versions covered.

If MuJoCo wins but OSMesa is awkward on laptops, laptops use `glfw` and servers use `osmesa` or `egl`. The cross-variant difference then tells us whether published runs must pin one variant.

Deliverables:

- The repo as laid out, with gates green in CI for the winning backend.
- `results/*.json`, `bakeoff_report.md` and the contact sheets.
- `python -m vsbench.smoke` for students.
- A short README: three-line install and first render.
