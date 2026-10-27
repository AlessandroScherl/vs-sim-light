# U2 parity fixture — provenance

`u2_matches_mug1.npz` was produced **once**, on 2026-09-02, by the frozen old
code, not by `refinevs`. Machine-readable copy of the same facts:
`u2_matches_mug1.json` (and inside the `.npz` itself).

## The exact command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate refinevs
cd ~/Desktop/runpod_dev/refine-vs
HF_HUB_OFFLINE=1 PYTHONPATH=~/Desktop/runpod_dev/genvs-sim \
  python tests/fixtures/u2_make_match_fixture.py --device cpu
```

`u2_make_match_fixture.py` never imports `refinevs`. It builds the old
`src/control/matchers/vit_dense.py::ViTDenseMatcher` exactly as
`src/eval/baseline_check.py` builds the **fielded dense arm**:

```python
ViTDenseMatcher(backbone="dinov3", num_pairs=24, device="cpu",
                ransac="none", emit_margins=True)          # baseline_check.py:1966-1996
matcher.set_subsample_seed(episode_subsample_seed("mug1", 7, 0))   # :2662-2670
matcher.reset(goal_rgb); matcher.match(current_rgb)
```

Two old-code internals are captured by monkeypatching, so the recorded numbers
are the vendored function's own and not a transcription of its arithmetic:

- `torch.Tensor.sort` is wrapped to keep `cyclical_dists_norm` (the array the
  vendored `find_correspondences_batch` sorts, hence `cyclic_all`) and to record
  **both** the non-stable order the old code takes and the stable order;
- `torch.randperm` is replaced by `arange` **for the pool pass only**, so the
  vendored `filtered_points` come back unpermuted — i.e. in the vendored sorted
  order. The draw passes use the real `randperm` under the real seed.

## The pair

| | |
| --- | --- |
| goal | `u1_frame_mug1_goal.jpg` — **the U1 fixture's image**, so the goal tokens are not duplicated: the parity test reads them from `u1_dinov3_tokens_mug1_goal.npz` |
| current | `u2_frame_mug1_roll010.jpg` — a real, non-goal 1920×1080 Isaac render of the **same scene**: the WP-C rotation bench's 10° camera-roll frame, copied byte-for-byte from `docs/results/2026-08-16_rotsym_wpc_preview/bench_frames/mug1/roll010.jpg`. A 10° roll is a genuine viewpoint change with a healthy mutual pool (612 of 1024 patches survive); no synthetic homography was needed. |
| episode | scene `mug1`, start `7`, base seed `0` → `episode_subsample_seed = 493031399` |

## What was pinned

| what | value |
| --- | --- |
| old repo | `../../../genvs-sim`, read-only, clean worktree at HEAD `2b11ab8`. The tag `legacy-pre-cleanup` is `905d42b`; `2b11ab8` is one commit later and touches only `the repository rules`/`README.md`, so every ported source file is the tagged one. |
| weights | `facebook/dinov3-vits16-pretrain-lvd1689m` from `~/.cache/huggingface` — gated, never downloaded (`HF_HUB_OFFLINE=1`) |
| env | conda `refinevs`: python 3.11.16, torch 2.14.0+cpu, transformers 4.57.1, numpy 1.26.4 |
| device | `cpu` |

## Files and md5s

| file | md5 | size | what |
| --- | --- | --- | --- |
| `u2_frame_mug1_roll010.jpg` | `002cf1f4ed22cd58965c6bd901760f1d` | 201 KB | the current-view frame |
| `u2_matches_mug1.npz` | `7f9bd86ff9d27f31567def14c28944f5` | 1.51 MB | everything below, plus the old deploy tokens of the current frame and the metadata table above |

Decoded-pixel md5 of the current frame: `cur_image_pixels_md5` in the npz, so a
re-encoded JPEG cannot pass unnoticed.

## What the fixture holds

| arrays | what |
| --- | --- |
| `cur_tokens` | float32 `(32, 32, 384)` — the **old** deploy tokens of the current frame; the goal's are U1's |
| `pool_{goal_lin,cur_lin,sim,margin,cyclic}` | the 612-pair mutual pool **in the vendored order** |
| `cyclic_all` | the cyclic-consistency score of every one of the 1024 goal patches |
| `sorted_order_{vendored,stable}` | the two sort orders of `cyclic_all` (see the finding below) |
| `sig_*` | `s1`, `second`, `margin`, `ratio`, `n_lookalikes`, `entropy_t05`, `entropy_t1` for all 612 pairs, from the old `src/eval/n2b_significance.py::row_significance` (which rounds to 5 decimals) |
| `draw24_*`, `draw48_*` | the seeded draws: patch indices, similarity, margin, and the full-image pixels the old `_patch_to_full_image` produced |
| `fs2_*`, `fs3_*` | the `pin_first_frame` (`_fs`) variant's frames 2 and 3 — the pinned 24 tracked one-way over the current frame, then over the goal frame |

## What the fixture proves, and how close

Measured 2026-09-02 in **both** environments (`refinevs` py3.11/torch 2.14.0+cpu
and `bot` py3.10/torch 2.11.0+cpu), `tests/fast/test_match.py`. The parity tests
pass `tie_order="legacy"` explicitly; the repository default is `"stable"` (see
the tie-order section below). Note that "bit-identical" means **given identical
tokens**: a fresh run of the frozen old chain is itself only ~3e-7 from these
stored tokens (BLAS thread-count effects), which is why the fixture stores the
tokens rather than the images alone.

| quantity | bar | measured |
| --- | --- | --- |
| pool membership and order (612 pairs) | exact | **exact** |
| pool cyclic scores / similarities / margins, and `cyclic_all` | ≤ 1e-6 | **0.0** |
| the 24 and the 48 drawn pairs | exact | **exact** |
| the `_fs` tracker's frames 2 and 3 | exact / ≤ 1e-6 | **exact / 0.0** |
| `sig_*` against `row_significance` | ≤ 5e-6 (its own `round(x, 5)`) | **≤ 5.0e-6**, and equal after rounding |
| full-image pixels of the draw (`to_rows`) | ≤ 1e-6 | **0.0** |
| the same chain on **new** U1 tokens (8.8e-7 goal / 2.3e-6 current from these) | anything that moves must be a near tie (gap < 1e-5) | **0 of 612 pool pairs, 0 of 24 drawn pairs** move; similarities within 5.4e-7, margins within 6.0e-7 |

Three behaviours restored after review R11 were re-checked directly against the
frozen old matcher on this pair (not through the fixture), 2026-09-03: a pixel
`reset(goal_mask=…)` covering the left half of the frame gives the **same drawn
24** and pixel difference **0.0**; an all-False mask gives **0 matches** in both;
and two frames that match nothing leave both matchers' draw counters at **2**, so
frame seeds stay aligned across arms.

## The tie order: the finding this fixture forced, and its real consequences

*(This section is the single home of the record. `refinevs/match.py` fence 4,
`the repository rules` and `README.md` carry one-line pointers here.)*

**What is true.** At the fielded `threshold=1.0` every survivor of the cyclic
filter scores exactly 1.0: a perfect round trip gives `-||delta|| = 0`, the shift
makes it `max_dist`, and the scale gives `max_dist / (max_dist + 1e-8)` — on this
frame `max_dist ≈ 6` cells, so `1 − 1.7e-9`, which float32 (spacing 6e-8 below 1)
rounds **to exactly 1.0**. That rounding is also why the vendored threshold
carries a `− 1e-6` epsilon. Measured here: 612 entries exactly 1.0, next value
0.8333. So the vendored `sort(descending=True)` — `stable=False` — is ordering a
block of equal keys, PyTorch does not define that order, and it is that order
`randperm` indexes: **the tie order decides which 24 patches the loop steers on.**

**What is NOT true** (corrected 2026-09-03, review R11): an earlier version of
this note claimed torch 2.11.0+cpu and 2.14.0+cpu return different orders on the
all-tied array the vendored `max_dist < 1e-6` branch produces. They do not. On
this laptop the two builds agree on every array measured — the fixture's
`cyclic_all`, `ones(1024)` as 1-D and as `[1, 1024]`, `ones(1369/4096/5000)`, and
the all-tied branch through `mutual_nn_match(goal, goal)` — and both draw the
same legacy 24. The non-contractuality is real but its evidence is (a) PyTorch's
own documentation for `stable=False`, and (b) the CUDA sort kernel, which orders
ties differently from the CPU one.

**What each setting is worth.**

| | `tie_order="stable"` (**the default**, owner ruling 2026-09-03) | `tie_order="legacy"` |
| --- | --- | --- |
| order of the 612 ties | ascending goal-patch index | whatever the sort kernel returns |
| same on any torch build / device | yes | **no** — the sort runs where the tokens live, so legacy on the station ≠ legacy on this laptop |
| reproduces this fixture | no (a different, equally random 24 — 0 of 24 shared) | **yes**, which is what the U2 parity tests pass it for |
| pool membership, sims, margins | identical | identical |
| paired reads (all arms share a start's draw) | unaffected | unaffected |

**What can and cannot be replayed from the stored campaigns.**

- **Not N3.7b / N3.7d, for a reason that precedes the tie order**: those records
  (`baseline_check` jsonl, per-episode fields) never stored the drawn landmark
  indices at all. No port, under any tie order, on any device, can replay those
  episodes step by step — the information does not exist.
- **The old station draws themselves** came from the CUDA sort plus CUDA
  `randperm` under `torch.cuda.manual_seed`; `refinevs.match` uses a CPU
  generator and sorts on the tokens' device, so it reproduces them on no device.
  That is a deliberate design choice (device-independent draws), not an accident.
- **N3.8 extended telemetry (`--telemetry-extended`) does store `g_lin` and
  `c_lin` per step**, so an N3.8 episode **can** be replayed on its recorded
  landmark sets: feed the recorded goal patches through `Pool.take(...)` /
  `DenseMatcher.pin_landmarks(...)` instead of drawing, and compare velocities.
  That is the replay the plan can promise, and it is what the U4 control-re-solve
  fixture should use.
- What a re-fly of a stored campaign *can* match: pool membership up to near-ties
  below the token-noise floor, pool size, `cyclic_all`, sims and margins to
  ~1e-6, and the arm-level statistics **within draw-to-draw variability** — which
  is not small: the old repo measured the same baseline arm at 5.33 vs 6.22 mm
  median under two draws (`vit_dense.py:237`). "Reproduces the published medians"
  therefore means "within the paired-bootstrap CI of a re-draw", not equality.

**Near ties on this fixture** (why the new-vs-old-token test asserts what it
does): the smallest best-vs-second gap is **5.007e-6** over goal rows (2 rows
below 1e-5) and **7.689e-6** over current columns (1 column), against token
differences of 8.8e-7 (goal) and 2.3e-6 (current). Zero pairs actually flip on
CPU, but a flip is possible in principle, so
`test_new_tokens_only_move_near_ties_in_the_pool` asserts that anything that
moves is a near tie rather than asserting a hard zero.

Regenerating this fixture is a decision, not a fix: it means the port changed a
number. Say so in the commit message and in `../../RESULTS.md`.
