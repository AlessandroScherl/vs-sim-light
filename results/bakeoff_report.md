# VS-Bench step 1: renderer bake-off report

Generated from `results/*.json`. Machine: AMD Ryzen 7 5800X 8-Core Processor, 16 threads, Linux-6.14.0-37-generic-x86_64-with-glibc2.39, Python 3.12.3. GPU present: NVIDIA GeForce RTX 4090, used only by the variants labelled gpu (mujoco-egl-gpu, mujoco-egl-gpu-noaa, open3d-gpu, pybullet-egl); every other row ran with `VSBENCH_SOFTWARE_GL=1` (Mesa llvmpipe). Load average recorded before the speed runs: 1.2 to 13.5 (each run follows the previous tool in the chain; the CPU rows also shared the machine with another job using about three cores).

## 1. Gates (Linux, no display; rows labelled gpu are informational)

| variant | GL renderer | projection | depth | homography | determinism | all |
|---|---|---|---|---|---|---|
| mujoco-egl-gpu | NVIDIA GeForce RTX 4090/PCIe/SSE2 | passed<br>mean 0.038 px, max 0.122 px over 260 markers | passed<br>median 0.000 mm, p99 0.014 mm over 6129087 px | passed<br>worst median displacement 0.000 px, fewest matches 1958 | passed | pass |
| mujoco-egl | llvmpipe (LLVM 20.1.2, 256 bits) | passed<br>mean 0.038 px, max 0.122 px over 260 markers | passed<br>median 0.000 mm, p99 0.001 mm over 6129087 px | passed<br>worst median displacement 0.000 px, fewest matches 1986 | passed | pass |
| mujoco-osmesa | llvmpipe (LLVM 19.1.7, 256 bits) | passed<br>mean 0.038 px, max 0.122 px over 260 markers | passed<br>median 0.000 mm, p99 0.001 mm over 6129087 px | passed<br>worst median displacement 0.000 px, fewest matches 1986 | passed | pass |
| open3d-cpu | Filament headless (Vulkan on llvmpipe per Open3D's log) | passed<br>mean 0.038 px, max 0.129 px over 260 markers | passed<br>median 0.000 mm, p99 0.001 mm over 6129087 px | passed<br>worst median displacement 0.000 px, fewest matches 2024 | failed | FAIL |
| open3d-gpu | Filament (GPU) | passed<br>mean 0.038 px, max 0.129 px over 260 markers | failed | passed<br>worst median displacement 0.000 px, fewest matches 1950 | failed | FAIL |
| pybullet-egl | - | unavailable: RuntimeError: eglRendererPlugin is not available in this pybullet build | | | | out |
| pybullet-tiny | TinyRenderer (software) | passed<br>mean 0.065 px, max 0.195 px over 260 markers | passed<br>median 0.001 mm, p99 0.007 mm over 6129087 px | passed<br>worst median displacement 0.000 px, fewest matches 2057 | passed | pass |

Thresholds: projection mean <= 0.5 px and max <= 1.0 px; depth median <= 1 mm and 99th percentile <= 3 mm; homography median displacement <= 0.5 px with >= 100 ORB matches; determinism identical arrays.

## 2. Install

| library | release | wheels (PyPI) | system package | criterion 2 |
|---|---|---|---|---|
| mujoco | 3.13.0 (2026-09-09) | Linux, macOS, Windows: cp310-cp315 | Linux: libosmesa6 (OSMesa) or none with EGL/Mesa; macOS, Windows: none | yes |
| pybullet | 3.2.7 (2025-01-30) | Linux only, cp36-cp311; no wheel for 3.12; no macOS or Windows wheels (source build needs a C++ compiler) | none, but a compiler on 3.12 and on every macOS/Windows machine | no |
| open3d | 0.20.0 (2026-09-16) | Linux, macOS, Windows: cp310-cp314 | Linux: none beyond Mesa EGL (libegl1, libgl1-mesa-dri, usually present) | yes |

Fresh-venv logs (`results/friction/`):

- `friction_mujoco-egl_py3.10.log`: pip done in 13 s; smoke exit code 0; first image after 14 s from venv creation
- `friction_mujoco-egl_py3.11.log`: pip done in 14 s; smoke exit code 0; first image after 14 s from venv creation
- `friction_mujoco-egl_py3.12.log`: pip done in 10 s; smoke exit code 0; first image after 11 s from venv creation
- `friction_mujoco-osmesa_py3.12.log`: pip done in 10 s; smoke exit code 0; first image after 11 s from venv creation
- `friction_open3d-cpu_py3.11.log`: pip done in 71 s; smoke exit code 0; first image after 74 s from venv creation
- `friction_open3d-cpu_py3.12.log`: pip done in 26 s; smoke exit code 0; first image after 29 s from venv creation
- `friction_pybullet-tiny_py3.11.log`: pip done in 22 s; smoke exit code 0; first image after 25 s from venv creation
- `friction_pybullet-tiny_py3.12.log`: pip done in 65 s; smoke exit code 0; first image after 69 s from venv creation

macOS and Windows: untested (no such machine in this run).

## 3. Speed (one CPU process, RGB plus depth, median of 200 frames)

| variant | scene | resolution | triangles | median ms | p90 ms | fps | build s |
|---|---|---|---|---|---|---|---|
| mujoco-egl-gpu-noaa | poster_lit | 640x480 | 2 | 0.84 | 0.86 | 1184.7 | 0.2 |
| mujoco-egl-gpu-noaa | poster_lit | 1440x1080 | 2 | 9.39 | 9.59 | 106.5 | 0.2 |
| mujoco-egl-gpu-noaa | clutter | 640x480 | 49808 | 0.76 | 0.77 | 1321.1 | 0.2 |
| mujoco-egl-gpu-noaa | clutter | 1440x1080 | 49808 | 8.80 | 8.92 | 113.7 | 0.2 |
| mujoco-egl-gpu | poster_lit | 640x480 | 2 | 0.91 | 0.92 | 1100.6 | 0.2 |
| mujoco-egl-gpu | poster_lit | 1440x1080 | 2 | 6.07 | 6.21 | 164.6 | 0.2 |
| mujoco-egl-gpu | clutter | 640x480 | 49808 | 0.88 | 0.89 | 1140.2 | 0.2 |
| mujoco-egl-gpu | clutter | 1440x1080 | 49808 | 5.62 | 5.74 | 178.1 | 0.2 |
| mujoco-egl-noaa | poster_lit | 640x480 | 2 | 1.14 | 1.18 | 874.9 | 0.2 |
| mujoco-egl-noaa | poster_lit | 1440x1080 | 2 | 12.26 | 12.53 | 81.6 | 0.2 |
| mujoco-egl-noaa | clutter | 640x480 | 49808 | 2.63 | 3.05 | 380.8 | 0.2 |
| mujoco-egl-noaa | clutter | 1440x1080 | 49808 | 14.74 | 22.92 | 67.9 | 0.2 |
| mujoco-egl | poster_lit | 640x480 | 2 | 2.20 | 2.38 | 455.5 | 0.2 |
| mujoco-egl | poster_lit | 1440x1080 | 2 | 18.50 | 18.73 | 54.0 | 0.3 |
| mujoco-egl | clutter | 640x480 | 49808 | 10.14 | 12.99 | 98.6 | 0.3 |
| mujoco-egl | clutter | 1440x1080 | 49808 | 37.61 | 45.39 | 26.6 | 0.3 |
| mujoco-osmesa | poster_lit | 640x480 | 2 | 2.11 | 2.25 | 474.8 | 0.2 |
| mujoco-osmesa | poster_lit | 1440x1080 | 2 | 18.21 | 18.54 | 54.9 | 0.3 |
| mujoco-osmesa | clutter | 640x480 | 49808 | 9.75 | 12.44 | 102.6 | 0.3 |
| mujoco-osmesa | clutter | 1440x1080 | 49808 | 38.10 | 46.28 | 26.2 | 0.3 |
| open3d-cpu | poster_lit | 640x480 | 2 | 32.49 | 34.59 | 30.8 | 0.6 |
| open3d-cpu | poster_lit | 1440x1080 | 2 | 85.36 | 89.28 | 11.7 | 0.6 |
| open3d-cpu | clutter | 640x480 | 49808 | 91.50 | 120.75 | 10.9 | 0.6 |
| open3d-cpu | clutter | 1440x1080 | 49808 | 182.95 | 234.60 | 5.5 | 0.6 |
| open3d-gpu | poster_lit | 640x480 | 2 | 27.96 | 32.44 | 35.8 | 0.7 |
| open3d-gpu | poster_lit | 1440x1080 | 2 | 62.95 | 63.84 | 15.9 | 0.7 |
| open3d-gpu | clutter | 640x480 | 49808 | 27.88 | 29.66 | 35.9 | 0.8 |
| open3d-gpu | clutter | 1440x1080 | 49808 | 63.78 | 65.72 | 15.7 | 0.8 |
| pybullet-tiny | poster_lit | 640x480 | 2 | 26.21 | 27.80 | 38.2 | 0.3 |
| pybullet-tiny | poster_lit | 1440x1080 | 2 | 110.99 | 116.20 | 9.0 | 0.3 |
| pybullet-tiny | clutter | 640x480 | 49808 | 92.64 | 96.54 | 10.8 | 0.3 |
| pybullet-tiny | clutter | 1440x1080 | 49808 | 374.65 | 394.63 | 2.7 | 0.3 |

Parallel scaling (poster scene, 640 x 480, one renderer per process):

| variant | processes | total fps | per-process fps |
|---|---|---|---|
| mujoco-egl-gpu | 1 | 1084.7 | 1084.7 |
| mujoco-egl-gpu | 2 | 1223.1 | 611.6 |
| mujoco-egl-gpu | 4 | 1292.8 | 323.2 |
| mujoco-egl-gpu | 8 | 926.0 | 115.8 |
| mujoco-egl | 1 | 450.3 | 450.3 |
| mujoco-egl | 2 | 397.6 | 198.8 |
| mujoco-egl | 4 | 422.7 | 105.7 |
| mujoco-egl | 8 | 404.1 | 50.5 |
| mujoco-osmesa | 1 | 483.0 | 483.0 |
| mujoco-osmesa | 2 | 415.9 | 208.0 |
| mujoco-osmesa | 4 | 437.1 | 109.3 |
| mujoco-osmesa | 8 | 411.8 | 51.5 |
| open3d-cpu | 1 | 32.3 | 32.3 |
| open3d-cpu | 2 | 58.8 | 29.4 |
| open3d-cpu | 4 | 95.0 | 23.8 |
| open3d-cpu | 8 | 115.0 | 14.4 |
| open3d-gpu | 1 | 37.6 | 37.6 |
| open3d-gpu | 2 | 74.6 | 37.3 |
| open3d-gpu | 4 | 110.6 | 27.6 |
| open3d-gpu | 8 | 87.6 | 10.9 |
| pybullet-tiny | 1 | 40.1 | 40.1 |
| pybullet-tiny | 2 | 77.8 | 38.9 |
| pybullet-tiny | 4 | 149.3 | 37.3 |
| pybullet-tiny | 8 | 255.7 | 32.0 |

## 4. Poster quality (against a 4x supersampled warp)

| variant | case | ORB inliers / matches | SIFT inliers / matches | PSNR dB (frame / poster) | brightness gain | sheet |
|---|---|---|---|---|---|---|
| mujoco-egl-gpu | slant70_1.5m | 679 / 737 | 96 / 120 | 36.2 / 26.3 | 0.97 | `quality_mujoco-egl-gpu.png` |
| mujoco-egl-gpu | minify3x | 2426 / 2498 | 679 / 718 | 35.7 / 32.3 | 0.99 | `quality_mujoco-egl-gpu.png` |
| mujoco-egl | slant70_1.5m | 694 / 757 | 98 / 122 | 36.4 / 26.5 | 0.96 | `quality_mujoco-egl.png` |
| mujoco-egl | minify3x | 2437 / 2498 | 690 / 731 | 35.8 / 32.4 | 0.99 | `quality_mujoco-egl.png` |
| mujoco-osmesa | slant70_1.5m | 694 / 757 | 98 / 122 | 36.4 / 26.5 | 0.96 | `quality_mujoco-osmesa.png` |
| mujoco-osmesa | minify3x | 2437 / 2498 | 690 / 731 | 35.8 / 32.4 | 0.99 | `quality_mujoco-osmesa.png` |
| open3d-cpu | slant70_1.5m | 1290 / 1318 | 172 / 184 | 43.1 / 33.2 | 0.99 | `quality_open3d-cpu.png` |
| open3d-cpu | minify3x | 2444 / 2506 | 684 / 717 | 36.0 / 32.7 | 0.99 | `quality_open3d-cpu.png` |
| open3d-gpu | slant70_1.5m | 1177 / 1203 | 169 / 181 | 41.0 / 31.0 | 0.99 | `quality_open3d-gpu.png` |
| open3d-gpu | minify3x | 2189 / 2287 | 607 / 661 | 32.5 / 29.2 | 0.98 | `quality_open3d-gpu.png` |
| pybullet-tiny | slant70_1.5m | 588 / 797 | 57 / 94 | 0.5 / 20.3 | 0.99 | `quality_pybullet-tiny.png` |
| pybullet-tiny | minify3x | 1502 / 1785 | 353 / 459 | 2.7 / 19.9 | 0.98 | `quality_pybullet-tiny.png` |

Cross-variant difference (same poses, two variants of one backend; the EGL variant here runs Mesa 25.2's llvmpipe, the OSMesa variant conda-forge Mesa 24's):

| a | b | scene | PSNR min dB | max abs pixel diff | depth max diff m | depth p99 diff m (both hit) | hit masks disagree | identical |
|---|---|---|---|---|---|---|---|---|
| mujoco-egl-gpu | mujoco-egl-llvmpipe | poster | 44.6 | 65 | 1.2e+00 | 2.6e-06 | 0.001 % | False |
| mujoco-egl-gpu | mujoco-egl-llvmpipe | clutter | 44.6 | 35 | 2.6e+00 | 2.1e-05 | 0.000 % | False |
| mujoco-egl | mujoco-osmesa | poster | 361.2 | 0 | 0.0e+00 | 0.0e+00 | 0.000 % | True |
| mujoco-egl | mujoco-osmesa | clutter | 361.2 | 0 | 0.0e+00 | 0.0e+00 | 0.000 % | True |
| open3d-gpu | open3d-cpu | poster | 40.6 | 57 | 1.5e+00 | 1.4e+00 | 80.383 % | False |
| open3d-gpu | open3d-cpu | clutter | 28.5 | 246 | 7.6e+00 | 1.2e+00 | 73.434 % | False |

Lighting control:

| variant | move light | change intensity | second light | image |
|---|---|---|---|---|
| mujoco-egl-gpu | yes | yes | yes | `lighting_mujoco-egl-gpu.png` |
| mujoco-egl | yes | yes | yes | `lighting_mujoco-egl.png` |
| mujoco-osmesa | yes | yes | yes | `lighting_mujoco-osmesa.png` |
| open3d-cpu | yes | yes | no | `lighting_open3d-cpu.png` |
| open3d-gpu | yes | yes | no | `lighting_open3d-gpu.png` |
| pybullet-tiny | yes | yes | no | `lighting_pybullet-tiny.png` |

## Findings per backend (what the gates and tools turned up)

- **MuJoCo.** Intrinsics go in as `focal_pixel`, `principal_pixel`, `sensor_size`, `resolution`; the principal point is an offset from the image centre, positive left and up, which the shifted-principal-point gate confirmed. A multisampled depth buffer resolves off pixel centre (99th percentile 3.9 mm on a 60 degree plane, on llvmpipe and on the RTX 4090 alike); `mujoco.Renderer`'s remedy, a segmentation pass, only helps on llvmpipe, because NVIDIA keeps the sample positions. Depth therefore comes from a second render into a context whose offscreen buffer has `offsamples = 0`, which is exact on both. Near and far are `vis.map.znear/zfar` times `stat.extent`. User-mesh texcoords have v = 0 at the top row. Unlit colours are exact (0.5 -> 128).
- **PyBullet.** TinyRenderer rasterises half a pixel off OpenGL's centres: a (+0.50, -0.50) px centroid offset in the projection gate, corrected by shifting the principal point handed to the projection matrix. The 3.2.7 wheel exists for Linux up to Python 3.11; on 3.12 pip builds it from source (about a minute, needs a compiler) and the build has no NumPy support unless `--no-build-isolation` is used, so `getCameraImage` returns tuples. No macOS or Windows wheels. The pip build has no `eglRendererPlugin`, so the GPU variant `pybullet-egl` cannot be loaded at all. Lighting is one global light (no second light), `unlit` is the ambient term alone, the clear colour is white and cannot be set in DIRECT mode (`configureDebugVisualizer(rgbBackground=...)` has no effect), and textures are sampled without mipmaps, which shows as aliasing on the minified poster.
- **Open3D.** `Camera.set_projection(K, near, far, W, H)` takes a full intrinsic matrix; add half a pixel to the principal point. With post-processing off the output is `0.5 * srgb_decode(colour)`; the backend pre-encodes colours and albedo textures so the raw texture comes back (fit 0.985, 33 dB against the reference warp). A second `OffscreenRenderer` of the same size in one process renders differently and destroying one while another is alive aborts the process, so instances are kept until exit and the gate files run one per process. Cross-process determinism failed in 5 of 10 runs on the clutter scene: 268 to 396 white pixels on the 14k-triangle mesh at the image corner differ between an in-process render and a fresh process; the same mesh alone renders identically, and the mesh has no degenerate faces or NaN normals. Filament ran on Vulkan/llvmpipe (lavapipe) for the CPU variant and on Vulkan/NVIDIA for `open3d-gpu`. On the GPU the depth image is unusable in this version: `render_to_depth_image(z_in_view_space=True)` returned a valid plane depth for under 1 % to 30 % of the pixels, and two renders of one pose in one process differed in thousands of depth pixels by up to 0.9 m, while RGB passed projection and homography. Filament keeps one directional light (the sun): a second directional light is accepted and ignored, a point or spot light does add. Lit scenes are darker than MuJoCo's at the same nominal intensity until the sun is scaled (230000 lux for intensity 1).

## 5. Decision

Criterion 1 (all four gates on Linux, no GPU, no display): mujoco passes (mujoco-egl, mujoco-osmesa); open3d fails (open3d-cpu); pybullet passes (pybullet-tiny).
Criterion 2 (pip plus at most one system package, official wheels for Linux, macOS and Windows on 3.10-3.12): mujoco; out: pybullet.
Criterion 3 (>= 25 fps at 640 x 480 on the clutter scene, one CPU process, best variant): mujoco 102.6 fps; passing: mujoco.

**Pick: mujoco** (the only library through criteria 1-3): mujoco-egl 98.6 fps, mujoco-osmesa 102.6 fps at 640 x 480 on the clutter scene in one CPU process.

Variants: laptops use `mujoco-glfw` (a window system, no extra package), headless Linux uses `mujoco-osmesa` (the `libosmesa6` package) or `mujoco-egl` (Mesa EGL, `VSBENCH_SOFTWARE_GL=1` for llvmpipe). The EGL and OSMesa variants rendered identical pixels in this run, so a published CPU run need not pin one of them.

CPU against GPU: the same poses on the RTX 4090 and on llvmpipe agree to 44.6 dB with pixel differences up to 65 of 255 at anti-aliased edges and texture minification; depth agrees to 0.02 mm (99th percentile) where both hit, with hit masks disagreeing on 0.001 % of pixels. A run is reproducible bit for bit within the CPU class; results published from a GPU and reproduced on a CPU (or the reverse) differ at the level above, so a published run should state which class it used.

The final call stays with the project owner.
