# U8 parity fixture — `u8_rotation_parity.json`

*Written 2026-09-03, on review R19 item 7. Maker: `u8_make_rotation_fixture.py`.
Reference: `../genvs-sim` @ tag `legacy-pre-cleanup` (= commit `905d42b`), the vendored
ViT-VS matcher inside it, and the fielded DINOv3 backbone.*

## Why it exists

Every U8 parity test used to import the frozen old repository live and **skip** when it
was not on disk. `../MIGRATION_PLAN.md` §1 asks each unit for a fixture *stored under
`tests/fixtures/`*, for exactly the machine this repository is heading to: a fresh
clone, or the station container mounting only `refine-vs`. A test whose only outcome
there is `skip` proves nothing. The live-import tests are kept beside these — they are
strictly stronger where the old repo exists — but the stored ones are what a machine
without it can run, and they are a real pass, not a skip.

## What is in it (59 KB)

| key | what | produced by |
| --- | --- | --- |
| `unwarp_points` | 64 random points (`default_rng(0)`, uniform −50…2000) mapped at 7 angles (0, 7.5, 30, 90, 180, −137.25, 359.9), in **both** call shapes: the plain 1920×1080 index map and the `raw_wh` squashed round trip | `rot_aware.unwarp_points` (`src/control/matchers/rot_aware.py:73`) |
| `rotate_hypothesis` | md5 of the frame bytes at 0, 90, 180, 270 and 17°, plus the md5 of the source frame so a changed fixture image fails loudly rather than silently re-basing | `prealign.rotate_hypothesis` at Pillow's published **NEAREST** |
| `prealign_pick` | the frozen `RollPrealigner.pick` record (picked angle, per-angle similarity, selection counts, margin) on the synthetic goal rolled 90°, `seed=2026`, `tie_order: legacy` | `prealign.RollPrealigner` driven through the **vendored** `find_correspondences_batch` — the published arithmetic, not a re-implementation |
| `bank_counts_u1_fixture` | the 12-bank mutual-match vector on `u1_frame_mug1_goal.jpg` at a 50° and a 60° native roll, and the valid-patch count per bank | the real DINOv3 chain, CPU |

The synthetic goal is byte-identical to `tests/fast/test_rotation.py::_synthetic_goal`
(a 5×5-pixel random mosaic at 960×540, `default_rng(8)`); the maker asserts its md5
before writing, and the test asserts it before reading.

## The numbers a reader wants without opening the file

- **`prealign_pick`**: picked **90°**, 48 pairs selected at every hypothesis, margin
  **0.002355**. The margin is small because the stub encoder used for this fixture is a
  cell-statistics descriptor, not DINOv3 — what is pinned here is the *search*, not the
  backbone.
- **bank counts, 60° roll**: `[250, 276, 612, 249, 235, 181, 225, 179, 189, 125, 153,
  163]` for banks 0…330 — bank 60 wins with **612 against 250** for the unrotated bank
  the fielded chain would use, and 612 is exactly its valid-patch count (the current
  frame *is* that bank).
- **bank counts, 50° roll** (off the bank grid, a 10° residual): `[288, 337, 380, 214,
  214, 187, 207, 205, 180, 134, 149, 174]` — the nearest bank still wins, **380 against
  337** at 30° and **288** at 0°.
- **valid patches per bank**: `[1024, 760, 612, 512, 612, 760, 1024, 760, 612, 512,
  612, 760]` — 0° and 180° keep the full grid (Pillow's lossless copy and transpose),
  90° and 270° lose half of a 16:9 frame.

## Reproducing it

```bash
cd refine-vs && HF_HUB_OFFLINE=1 python tests/fixtures/u8_make_rotation_fixture.py
```

Needs `../genvs-sim` checked out and the DINOv3 weights cached. Regenerating it is a
**decision**, not a fix: if a stored number moves, the finding is that something in the
chain moved, and it belongs in `RESULTS.md` and in a review before the file changes.

## Environments this was checked in

Written under `refinevs` (py3.11.16, torch 2.14.0+cpu, Pillow 12.3.0) and read back
green under both `refinevs` and the laptop `bot` env (py3.10.20, torch 2.11.0+cpu) —
the bank counts are identical in the two, which is the point of storing them.
