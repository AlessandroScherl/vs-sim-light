# The benchmark: six scenes, frozen starts

"Test scene" means two things frozen together: a **scene** (`configs/scenes.yaml`
plus, for the rooms, its baked layout in `configs/layouts/`, which together fix the
goal pose) and the **start poses** flown from it (the files here). `vssim.benchmark`
pairs them and pins the files; nothing reads a start file without checking its md5.

## The six scenes

| scene | kind | what it is | goal distance (m) | never seen before its first campaign |
| --- | --- | --- | --- | --- |
| `hollywood` | poster | the ART-VS Hollywood poster, 0.8 × 0.6 m, grey backdrop | 0.4341815108 | no |
| `poster2` | poster | same plate, a held-out COCO photo | 0.4341815108 | no |
| `poster3` | poster | same plate, Bruegel's *Tower of Babel* (public domain, 3840 × 2810) | 0.4341815108 | **yes** |
| `industrial1_drill` | room | one HANDAL drill lying on its side on a non-repeating concrete floor | 0.4422820342 | no |
| `cluttered2` | room | eight YCB scans (metric, native scale) settled on a wood-table floor; the pitcher held upright | 1.1689734192 | no |
| `cluttered3` | room | seven YCB scans (the pool minus the pitcher), a fresh seeded scatter + settle on a laminate floor | 1.2853190576 | **yes** |

The goal of every scene is a **top-down view** (`R = diag(1, −1, −1)`: image-right
= world +X). Posters: the plate *covers* the frame with a 1.35 margin. Rooms: the
measured bounding box of everything placed fills 85 % of the limiting frame
dimension, seen from straight above. The poster distances are closed-form from
`configs/camera.yaml`; the room distances come from the bounding box Isaac measures,
which is why `vssim.benchmark.check_scene_goal` must run on the station before the
first episode — a drifted asset or layout shows up there and nowhere else.

## The start files

| file | md5 | seed | cells | built |
| --- | --- | --- | --- | --- |
| `hollywood_starts_v2.json` | `9a242541b0d6d32f2e399bfc971e368d` | 20260820 | wide 500 (+ wide100 / wide20 / wideroll10), narrow 40 | 2026-08-20 |
| `poster2_starts_v2.json` | `12087f69a22beb7c4615f96455a0da34` | 20260820 | same | 2026-08-20 |
| `poster3_starts_v2.json` | `a9538e0ee2bf6f4c8ca42dbbe378ccbf` | 20260827 | same | 2026-08-27 |
| `industrial1_drill_starts_v2.json` | `b8e07e2dac718332be91ffca2df167ea` | 20260820 | same | 2026-08-20 |
| `cluttered2_starts_v2.json` | `e258fabd8fe5b0d5fc623312fa10fd63` | 20260820 | same | 2026-08-20 |
| `cluttered3_starts_v2.json` | `c7b8edd1fe702e8fde7bcbf269ec0e68` | 20260827 | same | 2026-08-27 |
| `hollywood_starts_near.json` | `7e768de50e0e39346def7097115dfc63` | 20260902 | near 60 | 2026-09-02 |
| `poster2_starts_near.json` | `8652872908ff6af14c27bc15c713f737` | 20260902 | near 60 | 2026-09-02 |
| `poster3_starts_near.json` | `6452dfe278f7f97e56371fbbd5ef0aa7` | 20260902 | near 60 | 2026-09-02 |

Every file carries its own `goal` block (`cam_pos`, `cam_R_opt`, `distance_m`,
`look_at`), each start's pose (`cam_pos`, `cam_R_opt` — the optical frame: columns
are image-right, image-down, viewing direction) and its initial error
(`pos_err_m`, `rot_err_deg`, and `roll_deg` for the wide cell). A `.md5` sidecar
sits beside each file; `tests/fast/test_benchmark.py` checks both.

**The cells.** *wide* (500): the DMLVS / ART-VS lineage distribution — positions
uniform in a 1.2 × 1.2 × 0.3 m box in the goal camera frame, each camera looking at
a point on one of four circles (radius 0.08 × 1..4 m) around the goal's fixation
point, with a roll linspace to ±179.5° per circle (the published ±120° widened so
near-upside-down starts are inside the primary distribution; the `|roll| ≤ 120`
subset is the published-parity population). *narrow* (40): ±18 % of the working
distance per axis, up to 12° about a random axis — mean 78 mm / 5.8° on the
posters. *near* (60): 3–8 mm and up to 3° — the terminal phase.

## The start sets (`vssim.benchmark.START_SETS`)

| set | cut from | n | scenes | use |
| --- | --- | --- | --- | --- |
| `capped30` | wide100, first 30 **by stored order** with \|roll\| ≤ 90° | 30 | all six | method screening — every stored campaign flew exactly this population |
| `wide100` | wide100 | 100 | all six | the headline band; report in thirds by initial rotation (`vssim.starts.roll_thirds`: 33 / 33 / 34) |
| `narrow20` | narrow, the first 20 | 20 | posters | precision band |
| `near60` | near | 60 | posters | terminal-phase probes |

```python
from vssim import benchmark
starts = benchmark.select_starts("cluttered2", "capped30")     # 30 dicts, campaign order
payload = benchmark.load_starts("cluttered2")                  # the whole pinned file
benchmark.check_scene_goal(scene, payload)                     # on a built IsaacScene
```

## Regenerating a file

`scripts/build_starts.py` rebuilds any file from its seed (posters offline; rooms
under `$ISAAC_PY`). A rebuild reproduces every id, cell, count and share exactly and
every rotation matrix to ~4e-16 — the residual is the BLAS behind two matrix
products, not the code — so a rebuilt file's **md5 differs from the pin** while its
content does not. `tests/fast/test_starts.py` is that comparison for hollywood.

## Heavy data (not in git)

| what | where the yaml / library expects it | env var |
| --- | --- | --- |
| poster textures, ground textures | `/workspace/assets/<scene>/…` (absolute paths in `configs/scenes.yaml`) | — (edit the yaml to relocate) |
| curated object USDs (`handal_drill_25`, the probe scenes' objects) | `<root>/experiments/ingest/usd/…`, `<root>/assets/gso_usd/…` | `VSSIM_ASSET_ROOT` (default: `assets/usd/` in this repo, git-ignored; on the station `/workspace/genvs-sim`) |
| YCB scans (`ycb/…` ids) + `ycb_library.json` | `/workspace/assets/ycb` | `VSSIM_YCB_ROOT` |

On the 4090 station everything is on the container volume at those paths.

## Where these came from

The scenes, layouts, samplers and files are those of evaluation protocol v2 of the
Refine-VS project (2026-09-02; `docs/eval_protocol_v2_2026-09-02.md` in that
workspace, sections 2–3), carried over unchanged: the files here are byte-identical
to the pinned originals (`docs/results/2026-08-21_n2_admission/`,
`docs/results/2026-08-27_n36_finals/out/s5/`,
`docs/results/2026-09-02_n38_terminal_probe/starts/`).
