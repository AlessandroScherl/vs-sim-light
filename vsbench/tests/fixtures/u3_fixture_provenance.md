# U3 parity fixtures — provenance

Three fixtures, all produced **once**, on 2026-09-03, by the frozen old code
(`../../../genvs-sim` @ tag `legacy-pre-cleanup` = commit `905d42b`), not by
`refinevs`. Machine-readable copy of the same facts: `u3_fixture_provenance.json`,
and every fixture repeats its own provenance in its header keys. The `.json`
files are written on one compact line each - they are data, this file is what a
human reads.

## The exact command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate refinevs
cd ~/Desktop/runpod_dev/refine-vs
HF_HUB_OFFLINE=1 python3 tests/fixtures/u3_make_baseline_fixtures.py
```

Run at `genvs-sim` HEAD `2b11ab8` = tag `legacy-pre-cleanup` (`905d42b`) plus one
doc-only commit (the `CLAUDE.md` / `README.md` superseded headers, +4 lines). Same
code, two hashes: the table below cites the tag, `u3_fixture_provenance.json` cites
HEAD.

`u3_make_baseline_fixtures.py` never imports `refinevs`. Inputs are the same
(goal, current) pair U1 and U2 use — `u1_frame_mug1_goal.jpg` and
`u2_frame_mug1_roll010.jpg`, 1920×1080 — and the same episode identity
`("mug1", 7)` at seed 0, so every U1–U3 fixture is one scene at one start.

| file | size | md5 | old code it ran |
| --- | --- | --- | --- |
| `u3_classical_mug1.json` | 19.7 KB | `1133410909669c116183e981a06585ed` | `src/control/matchers/classical.py::ClassicalMatcher` × 3 detectors × 3 RANSAC models |
| `u3_vitvs_dinov2.json` | 2.7 KB | `dff725f67ffd5770a4bc446c20efa3f2` | `src/control/matchers/vit_dense.py::ViTDenseMatcher(backbone="dinov2")` |
| `u3_artvs_dinov3.json` | 6.3 KB | `e408eb1d219296c1656da18efcb6d8d9` | `src/control/matchers/vit_tiled.py::ARTVSMatcher(backbone="dinov3")` |

## 1. Classical — `u3_classical_mug1.json`

Built as `baseline_check.build_controller` builds a classical arm: the scene's
RANSAC model, `num_pairs` = the ViT path's `--K` (24). Nine cells, each with the
keypoint counts on both images, the pre-RANSAC count, the inlier count, whether a
model was fitted, and every matched pixel coordinate and confidence.

**Cross-environment**: the two OpenCV versions this repository runs — 4.11.0
(`refinevs`) and 4.10.0 (`bot`) — produce **identical** keypoint counts, match
counts, inlier counts and coordinates (max |Δ| 0.0 on all nine cells, measured
2026-09-03). The old dev box pinned 4.12.0.88.

**`cv2.setRNGSeed(0)` before each cell is load-bearing** (corrected by review
R16). The first version of this note said four repeats were identical with and
without it, so the seeding was belt and braces. The *observation* holds for the
nine cells; the *mechanism* was wrong. SIFT's FLANN matcher builds a randomised
KD-tree forest from `cv::theRNG()`, so its ratio-test set moves with the process
RNG — 101 vs 103 surviving matches over nine seedings on this pair, three distinct
sets. Only the top-24 by descriptor distance happened to be stable, which is luck,
not a property. Detection is deterministic; SIFT matching is not. ORB and AKAZE
match by brute force and RANSAC's estimator seeds its own generator per call
(identical inlier sets across seeds), so both really are unaffected.
`ClassicalMatcher.set_episode` now seeds OpenCV per episode and per frame, so a
fielded classical arm is a function of `(scene, start, seed)` like the dense one.

**Counts, for the record** (goal keypoints / current keypoints / raw / inliers):

| detector | keypoints | none | homography | fundamental |
| --- | --- | --- | --- | --- |
| SIFT | 153 / 194 | 24 (no model) | 15 inliers | 22 inliers |
| ORB | 922 / 956 | 24 (no model) | 24 inliers | 24 inliers |
| AKAZE | 1004 / 1039 | 24 (no model) | 23 inliers | 24 inliers |

## 2. The published ViT-VS (DINOv2) arm — `u3_vitvs_dinov2.json`

**U1 recorded that the DINOv2 weights are not cached. That is true of the HF copy
`facebook/dinov2-small` and false of the published torch.hub one**:
`~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` (88 MB) has been on
this laptop since 2026-05-07. Nothing was downloaded to make this fixture, and
`PublishedDinoV2Backbone` loads with `skip_validation=True` against that cache.

Recorded: the descriptor tensor's shape `[1, 1, 1369, 3456]`, its
double-precision sum and absolute maximum (two grid-wide invariants that do not
depend on which cells the draw picked), the reported geometry (37 × 37, input
518, patch 14) and the 24 drawn landmarks with their similarities.

**The recorded sum is machine-dependent and is pinned loosely on purpose** (rel
1e-5). It is a float32 reduction over 4.7 M values whose summation order follows
the torch thread count: measured against this fixture on the laptop, rel 0 at 1
thread, 8.9e-7 at 4, 1.6e-6 at 10 (the default here). A tighter pin would fail on
a box with a different core count while the descriptors were bit-identical. The
landmark check is the exact one — no reduction order can move an argmax.

**The port reproduces the descriptors bit-for-bit (max |Δ| 0.0) and the drawn
landmark pixels exactly**; only `conf` differs, by 1.1e-6, because
`mutual_nn_match` takes the cosine with a matmul where the vendored
`chunk_cosine_sim` used `torch.einsum` — a different float32 reduction order,
which changes no argmax and therefore no landmark.

**Getting there needed one real fix.** `torchvision`'s `ToTensor` returns a
*contiguous* tensor; a `permute` does not, and the patch-embedding convolution
takes a different kernel path for a non-contiguous input. Values identical to the
last bit, descriptors **1.9e-4** apart, and enough mutual-nearest-neighbour
argmaxes flipped that matched landmarks moved by up to **778 px**.
`PublishedDinoV2Backbone.preprocess` therefore calls `.contiguous()`, and
`tests/fast/test_features_dinov2.py::test_the_model_input_must_be_contiguous`
fails if it is ever removed as redundant. On the DINOv3 chain the same layout
change is worth 7.3e-7 (and would move `refinevs.features.FrozenBackbone`
*closer* to U1's fixture, 8.77e-7 → 7.60e-7); it is ~250× larger here because
these descriptors concatenate nine neighbourhoods of un-normalized block-11
outputs (scale ≈ 11) instead of one L2-normalized token. **U1's
`FrozenBackbone.preprocess` now does it too** (adopted on review R16, which
measured that the old DINOv3 chain's own input was contiguous NCHW): the token
effect is ~7.5e-7 and it moves *towards* the fixture, 8.78e-7 → 7.6e-7 measured
here (7.93e-7 in the review's run — the last digit follows the torch thread
count), with an identical 612-pair mutual pool on this pair either way. The U1
fixture is unchanged.

## 3. ART-VS — `u3_artvs_dinov3.json`

One coarse frame and one tiled frame (forced with the vendored
`switch_iteration=1`), plus the whole tile grid and the `TilingConfig` fields.

**On DINOv3, deliberately**, not on ART-VS's own `backbone="dinov2"` default:
what this fixture pins is U3's arithmetic — the tile grid, the tile-local →
global mapping with its `(y, x)` flip, the per-tile budget and the latch — and a
patch-**16** backbone is the case `TilingConfig.patch_size = 14` broke. Running
it on DINOv2 would put 14 on both sides of the fence and prove nothing.

Reproduction: coarse points **bit-identical** (max |Δ| 0.0), tiled points 2.3e-13
(the pixel round trip through `Matches.from_pixel_pairs`), confidences 1.3e-6 on
both — the U1 token gap (the old chain's fused rescale+normalize), seen through a
cosine, not a difference in this unit's arithmetic.

**A tiled frame does not carry `num_landmarks` landmarks.** The per-tile budget is
the published `max(1, round(K / n_tiles))` applied to every tile, so at 36 tiles
K = 24 → 36 rows, K = 48 → 36, K = 96 → 108, while the record's `num_landmarks`
still says K. Verbatim old behaviour, stated here because a support column read
off `num_landmarks` would be wrong — read `n_matches`, as on a CNS row.

**One inconsistency in the old config is repaired, and it moved no number.**
`vit_tiled.py:99` overwrote `TilingConfig.patch_size` *after* construction, so
the old object kept `patches_per_vit_input = 224 // 14 = 16` and
`effective_patches = 96` — DINOv2's numbers on a patch-16 backbone. Nothing read
them (`transform_patch_to_global` uses `patch_size`, `slice_resolution` and
`vit_input_size` only), which is why the points still reproduce exactly. The
fixture records the stale pair (16, 96) and
`tests/fast/test_tiling.py::test_the_tiling_geometry_matches_the_old_config`
states the difference against the port's consistent (14, 84) rather than hiding
it. `TilingConfig` now takes `patch_size` as a required argument, so the
hardcode cannot come back.

## The caveat that travels with all of this

**No runbook has invoked `vitvs` or `artvs` since 2026-07-27** (U1's carried
note; `../../../INVENTORY.md` on `vit_tiled.py` — last live use
`docs/BASELINE_RESULTS_2026-07-27.md`). These fixtures prove the port equals the
old code. They do **not** make any published-baseline number current: the arms
must be re-run on the station before their numbers are quoted.

Regenerating a fixture is a decision, not a fix: it means the port changed a
number (`README.md` in this directory).
