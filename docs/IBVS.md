# Image-based visual servoing (IBVS) in this simulator

A short, self-contained introduction for students: what IBVS is, how the included
baselines find their image points, and exactly what the simulator does with the velocity
your method returns. Every statement about the simulator names the function that does it.
For the full theory read F. Chaumette and S. Hutchinson, ["Visual servo control, Part I: Basic
approaches"](https://doi.org/10.1109/MRA.2006.250573), IEEE Robotics & Automation Magazine, 2006.

## 1. The idea

A camera should move until the picture it sees matches a **goal picture** taken from the goal
pose. IBVS never estimates the 3-D pose. It works on **image features**: points that can be
found in both pictures. Let `s*` be where the points are in the goal picture and `s` where the
same points are in the live picture. The error is

    e = s - s*

and the controller looks for the camera velocity that makes `e` shrink.

## 2. The camera and its optical frame

Everything is expressed in the camera's **optical frame** (the OpenCV convention):

- **x** points right in the image, **y** points down, **z** points forward (out of the lens,
  towards the scene);
- lengths are in **metres**, angles in **radians**, time in **seconds**;
- a pixel `(u, v)` is column `u`, row `v`; the intrinsics `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
  are in pixels.

The simulated camera is an Intel RealSense D435i colour camera: 1920 x 1080 pixels,
`fx = fy = 1406.748` px, 42 degrees vertical field of view (`vsbench/configs/camera.yaml`). It
gives a colour picture (`uint8`, H x W x 3) and a depth picture (`float32`, H x W, the z
distance in metres, 0 where nothing was hit). `MujocoScene.intrinsics` reports the principal
point as `(960, 540)`; the picture is drawn with Isaac Sim's half-pixel convention
(`vsbench/world/camera.py`), which the baselines were built for.

A pixel becomes a **normalised image coordinate** by

    x = (u - cx) / fx        y = (v - cy) / fy

(`normalised_coords` in `vsbench/vs/baselines/control.py`). A 3-D point at depth `Z` in the
camera frame projects to `(x, y) = (X / Z, Y / Z)`.

## 3. The interaction matrix

The camera velocity, the **twist**, is six numbers:

    v = [vx, vy, vz, wx, wy, wz]

`vx, vy, vz` (m/s) move the camera along its own x, y, z axes; `wx, wy, wz` (rad/s) rotate it
about them. A positive `vz` moves towards the scene; a positive `wz` rolls the camera
clockwise as seen from behind it (x turns towards y).

How fast does a point `(x, y)` at depth `Z` move in the image when the camera moves with `v`?
The answer is linear in `v`: `d(x, y)/dt = L v`, with the **point interaction matrix**

    L = [ -1/Z     0    x/Z    x*y     -(1 + x^2)    y ]
        [   0    -1/Z   y/Z   1 + y^2     -x*y      -x ]

(`interaction_matrix` in `vsbench/vs/baselines/control.py`; `vsbench/interaction.py` has the
same matrix, one 2 x 6 block per point, for your own code). One point gives two rows; `N` points are stacked into a
`2N x 6` matrix. Six unknowns need at least three points; the baselines need at least
`min_matches` of them, otherwise they command a zero twist.

**Why depth matters.** The first three columns (translation) are divided by `Z`; the last
three (rotation) are not. Moving the camera sideways moves a near point a lot and a far point
little, while a rotation moves both the same. With the wrong `Z` the controller confuses
translation with rotation and over- or under-shoots. The baselines read `Z` for each point
from the live depth picture at that pixel, and use 0.5 m where the depth is missing
(`depth_at` in `vsbench/vs/bench/arms.py`).

## 4. The control law

Ask the error to decay exponentially, `de/dt = -lambda e`. With `de/dt = L v` the least-squares
answer is

    v = -lambda * pinv(L) @ e

where `pinv` is the Moore-Penrose pseudo-inverse. All four included baselines use exactly this
law with **gain `lambda = 2.0`**, a plain (undamped) pseudo-inverse, no weighting, no smoothing
and **no clamping of the velocity** (`weighted_ls_solve` in `vsbench/vs/baselines/control.py`;
`controller:` block of each file in `vsbench/configs/arms/`). The gain is inside the law: the
simulator itself never scales a twist.

## 5. Where the points come from

The control law only needs pairs of pixels (goal, live). The baselines differ in how they
find them:

- **SIFT / ORB / AKAZE** (`sift_ransac`, `orb_ransac`, `akaze_ransac`;
  `vsbench/vs/baselines/classical.py`): detect keypoints in both pictures, match their
  descriptors with Lowe's ratio test (0.7), keep the best 24 matches by descriptor distance,
  then reject outliers with **RANSAC** (a homography on the planar posters, the fundamental
  matrix in the 3-D rooms, 3 px threshold). If RANSAC cannot fit a model, the matches are kept
  unfiltered. The `_noransac` variants skip RANSAC.
- **ViT-VS** (`vitvs_dinov3`, `vitvs_dinov2`; `vsbench/vs/baselines/match.py`,
  `features.py`): both pictures go through a pretrained vision transformer (DINOv3 ViT-S/16
  or DINOv2 ViT-S/14), which gives one descriptor per image patch. Patches that are each
  other's nearest neighbour in the two pictures (mutual nearest neighbours, cosine
  similarity) are matches; a seeded draw keeps 24 of them, and the patch centres, mapped back
  to full-image pixels, are the points. No RANSAC.
- **ART-VS** (`artvs`; `vsbench/vs/baselines/tiling.py`): ViT-VS with DINOv2 at 224 px while
  the camera is far; once the image error has fallen to 20 % of its start it switches, for
  good, to a 6 x 6 grid of tiles matched tile by tile at higher resolution.

Your own method can do anything (`examples/my_method.py` uses ORB matches and the same law,
with its own gain); it only has to return a twist.

## 6. How the simulator applies a twist

The twist is the camera's velocity **in its current optical frame**. One step of length `dt`
(`MujocoScene.apply` in `vsbench/world/mujoco_scene.py`, which calls
`integrate_optical_velocity` in `vsbench/vs/scene.py`) updates the camera-to-world pose
`(p, R)` as

    p <- p + R @ v_lin * dt
    R <- R @ exp([w]x * dt)

with `v_lin = (vx, vy, vz)`, `w = (wx, wy, wz)` and `exp([w]x dt)` the rotation by angle
`|w| dt` about the axis `w / |w|` (Rodrigues' formula). Both use the pose at the start of the
step (first-order integration; the twist is constant during the step). Every 300 steps the
rotation is re-orthogonalised (an SVD) to remove rounding drift. Nothing else happens to the
twist: no clamping, no smoothing, no gain.

In the benchmark loop `dt = 1/60 s` (the arm's `controller.dt`; for your own method,
`vsbench/world/configs/external_method.yaml`). In a loop of your own you pass `dt` to `apply`.

## 7. One step of the benchmark loop

`run_episode` in `vsbench/vs/bench/runner.py`, per step:

1. **render**: `scene.render()` draws the live colour and depth pictures at the current pose;
2. **method**: the method gets the live pictures (never the pose) and returns a twist and its
   matched points;
3. **stop rule**: the scored stop rule (below) sees the step's point errors;
4. **apply**: `scene.apply(twist, dt)` moves the camera.

The episode ends at the step budget (`max_iter`, 600 by default), earlier only if the twist is
not finite or the image error grows above 3 times its first value (the divergence abort,
`DivergenceGuard`; on for every included baseline and for your own method, which reports the
error as `error_norm`).

## 8. When to stop, and what counts as success

The **stop rule** is CNS's `cns_plateau` (`CnsPlateau` in `vsbench/vs/baselines/control.py`):
once the mean point error (normalised units) is below 0.01 and has stopped improving for 0.8 s,
the rule fires. The loop keeps running to the budget and **records** both readings: the pose
error where the rule fired (`stop_translation_mm`, `stop_rotation_deg`, `stop_step`) and at the
end (`final_translation_mm`, `final_rotation_deg`).

The record's `success` block applies several rules to the end pose
(`vsbench/vs/bench/scoring.py`), for example:

- `abs_10mm_1deg`: closer than 10 mm and 1 degree to the goal pose (also 20 mm / 2 deg,
  50 mm / 5 deg, 100 mm / 10 deg);
- `paper_3cm_3deg`: closer than 30 mm and 3 degrees;
- `predicate_a` (ART-VS's rule): closer than 10 mm and 1 degree, or both errors shrunk below
  10 % of where they started.

The distances are measured from the true poses, which the simulator knows and your method
never sees.
