# U4 control-parity fixture — provenance

`u4_telemetry_hollywood_w2_057.npz` was produced **once**, on 2026-09-02, from a
record the frozen old code wrote on the station in August — not by `refinevs`.
Machine-readable copy of the same facts:
`u4_telemetry_hollywood_w2_057.json` (and inside the `.npz` itself, key `meta`).

## The exact command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate refinevs
cd ~/Desktop/runpod_dev/refine-vs
python tests/fixtures/u4_make_control_fixture.py --episode w2_057
```

`u4_make_control_fixture.py` never imports `refinevs`.

## What kind of fixture this is

The U1 and U2 fixtures were made by *running* the old code here. This one could
not be: the control law's inputs are what a matcher produced on a rendered frame,
and re-rendering needs Isaac and the station. It does not have to be — the old
harness already wrote every input and the output beside it. So the fixture is an
**extract of recorded field telemetry**: one converged episode, exactly the arrays
the solve consumed and the twist it produced.

| | |
| --- | --- |
| record | `../../../docs/results/2026-08-21_n2_admission/gates/hollywood_wide/telemetry.jsonl.gz` (md5 `44f1d4a6ad918c35ce9eebb534a204c7`), committed in the workspace docs repo |
| writer | `genvs-sim@legacy-pre-cleanup` `src/eval/baseline_check.py::record_step_telemetry` (L1296-1440); the old repo's worktree is clean at `2b11ab8`, one docs-only commit past the tag |
| campaign | N2 admission gate, hollywood wide-100 starts, 2026-08-21 |
| arm | `head_w_g3_axw@v11` — trust³ least-squares weights, per-axis (`_axw`), **rounded** points (`_ur` was not yet fielded), no margin weighting |
| episode | start `w2_057`: 338.9 mm at step 1 down to 1.3 mm, ran to the 600-step cap (`max_iter`), final 1.651 mm |
| stored | every **2nd** step → 300 steps × 24 landmarks, the whole trajectory rather than one phase |
| size | 163 KB (cap is 5 MB) · md5 `0a97b4b8462f6b9201abf078d3cd49b5` |

**One regeneration, 2026-09-03** (review R12 correction 8). The `meta` block gained
two facts the parity tests lean on and could not otherwise check — `stop_mode` (none:
no lambda schedule was armed, so `v_recorded` is the control law's own twist and not
a scaled one) and `lam_in_record` (false, verified over the record's own steps). **No
array changed**: all ten compare bit-identical to the first build (max |diff| = 0.0),
which is asserted before the swap. Only the md5 moved, `075d3f2af86b5ac7fce20078c0e7bd2f`
→ `0a97b4b8462f6b9201abf078d3cd49b5`; the earlier number is the one review R12
independently reproduced.

## Arrays

`step_index` (300), `pts_goal` / `pts_cur` (300, 24, 2 — full-image pixels,
column-row), `conf` (the head's per-match trust), `w_recorded` (the least-squares
weight the old controller used, column 0), `depth` (300, 24 — see below),
`v_recorded` (300, 6 — the commanded twist), `feature_error_recorded`, `t_mm`
(true translation error, **scoring only** — no test feeds it to a control path),
`z_med_recorded`.

## The one reconstructed input: per-match depth

The record stores only the *median* depth `z_med`, and the interaction matrix
needs one Z per point. hollywood is a planar poster, so the depth is available in
closed form, and that is what makes this fixture possible at all. The
reconstruction is the reviewer's (`R7`, `scratchpad/review/r7_repro.py`, which
reached p99 relative error 1.1e-4 over 15,064 steps of these files):

```
R_goal = diag(1, -1, -1), goal_pos = (0, 0, 0.434182)     eval_scenes.py L194-206
R_cur  = R_goal @ Rodrigues(rot_to_goal_deg)^T            record_step_telemetry
p_cur  = goal_pos - R_cur @ d_cam_m                       same
m      = ((u - cx)/f, (v - cy)/f, 1)                      poster = world plane z = 0
Z      = -p_cur.z / (m @ R_cur^T).z
```

Intrinsics from `genvs-sim/configs/camera.yaml` via `free_flying_scene.camera_info`:
fx = fy = 1080 / (2 tan 21°) = **1406.748094934653**, principal point (960, 540).

**Checked, not assumed**: the reconstruction's median matches the recorded
`z_med` to **2.7e-4 m** across all 300 steps, and the test asserts it. This is why
`w2_057` and not the lowest-final-error episode `w1_108`: on `w1_108` the camera
spends its first ~300 steps looking past the poster, the depth map there is the
room and not the plane, and the reconstruction is off by up to 0.4 m — which
would have made a "parity failure" out of a fixture defect. Episodes were screened
on that agreement before one was picked.

## Measured parity (2026-09-02, conda env `refinevs`, py3.11 / torch 2.14.0+cpu)

**The parity statement is bit-identity, not a tolerance.** Given the same rows, the
new `weighted_ls_solve` and the frozen `genvs-sim@legacy-pre-cleanup`
`ibvs_core.solve_velocity` return the same twist to **0.0** on all 300 steps —
`tests/fast/test_control.py::test_the_solve_is_bit_identical_to_the_old_one` loads
the old function by path and measures it (skipped where `../genvs-sim` is absent).
Everything below is therefore a statement about the *record*, not about the port.

Relative error `||v_new − v_recorded|| / ||v_recorded||` over the 300 steps:

| what the new code is given | p50 | p99 | max | share < 1e-3 |
| --- | --- | --- | --- | --- |
| the rows + the **recorded** weights → `weighted_ls_solve` | 1.26e-05 | 2.06e-04 | 2.38e-04 | **1.000** |
| the rows + the recorded **trust** → `IBVSController.step` (weights recomputed) | 8.32e-05 | 6.94e-04 | 1.94e-03 | **0.997** |

The bar in `MIGRATION_PLAN.md` is 1e-3 relative on ≥ 99 % of steps; both rows
clear it.

**Why the record cannot be reproduced exactly** (corrected 2026-09-03, review R12
— the first version of this section blamed the wrong thing). It is **not** pixel
quantisation. This arm ran the *rounded* path, so `pts_goal` and `pts_cur` are
exact integers and their 2-dp storage is lossless; a test asserts the integrality,
and a ±0.005 px perturbation moves the solve by 8.8e-4 (p50) — seventy times the
observed p50 of 1.3e-5, which is how one can tell they are not quantised. The
residual is two things:

1. **the depth**, on the far steps. The record stores only `z_med`, so the fixture
   reconstructs Z from the poster plane while the controller sampled the depth map
   (truncating to the pixel). The reconstruction agrees with `z_med` to 2.7e-4 m,
   and the solve moves **9.8e-4 relative per millimetre of Z** (p50, measured), so
   a quarter-millimetre is exactly the 1.7–2.4e-4 seen on steps 1–11, where the
   camera is 339 mm out and the plane's depth spread across the frame is largest;
2. **the 7-dp storage of the twist itself**, on the terminal steps. At the
   episode's smallest commanded twist (‖v‖ = 7.7e-4 m/s) the rounding grid alone is
   a **9.1e-5** relative floor.

The recomputed-weight row adds a third: 4-dp trust cubed is 1.5e-4 of weight.

**What would not fix it.** An N3.8 `--telemetry-extended` record — 4-dp pixels and
per-match depth `zv` — does *not* tighten this, which the first version of this
document predicted it would. Measured on the local N3.8 smoke record (R12 §3.3):
re-solving with the recorded `w`, `w2` and `zv` gives p50 1.05e-4, p99 9.4e-4,
99.3 % < 1e-3 — looser than this fixture, because `w` is stored at 5 dp and `conf`
and `mg` at 4 dp while the smallest weights are ~1e-8. Record quantisation is the
floor either way. The reason to build a second fixture from such a record is not
precision but **coverage**: `w2_057` has no margin weighting, so nothing here pins
the fielded `_mw` term. On the N3.8 smoke record that formula is confirmed —
`w = conf³·(0.01 + mg)` to 8e-5 and `w2 = w·(16/9)²` to 2.1e-5 over 8,000 steps of
`head_w_g3_axw_mw_ur@v2cm` — so a second fixture should carry that path with the
honest bar (recorded `w`, `w2`: ≥ 99 % of steps < 1e-3), and bit-identity to the
old code as the parity statement.

## What the fixture pins that a synthetic test cannot

- the **interaction matrix and the pseudo-inverse** against 300 real, ill-conditioned,
  24-landmark configurations, including the terminal phase where the twist is
  ~1e-5 m/s and cancellation dominates;
- the **weight formula**: `w == clip(trust, 0, ∞)³` holds to 1.5e-4 absolute,
  while γ = 1, 2 or 4 miss by 0.10–0.38 — so the test tells the fielded `_g3`
  apart from every neighbour it could have been;
- the **per-axis factor**: only `w_y = w_x · (W/H)²` reproduces the recorded twist;
- the **raw feature-error norm** the stop rules and the divergence guard read,
  to the record's own 6 decimals (max deviation 5.0e-07).

## Regenerating

A fixture is produced once and then frozen; regenerating one means the port
changed a number, and that is a decision to be stated in the commit message and
in `../../RESULTS.md` (`tests/fixtures/README.md`). `--episode` and `--stride`
exist for that decision, not for convenience.
