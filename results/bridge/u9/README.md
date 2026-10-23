# U9 - the rig check twin on MuJoCo (2026-10-23)

`python -m vsbench.world.check_baselines` (port unit U9, `docs/PORT_PLAN.md` section 7) flew vs-sim's
rig check - the copied `scripts/check_baselines.py`, loaded as a library - on the MuJoCo world:
`{sift_ransac, orb_ransac, vitvs_dinov3, artvs}` x the six benchmark scenes, the `isaac_fit_v1`
look, CPU only (Mesa llvmpipe, torch 2 / 2 threads, no GPU, `nice -n 19`), one scene per
process, two processes at a time.

```
source tools/cpu_env.sh
$VSBENCH_PY -m vsbench.world.check_baselines --scenes <scene> \
    --arms sift_ransac orb_ransac vitvs_dinov3 artvs --near-steps 200 --far 1 \
    --out <dir>/<scene> --frames preview                    # one process per scene
$VSBENCH_PY -m vsbench.world.check_baselines summary --runs results/bridge/u9/<scene> ... \
    --out results/bridge/u9                                 # + Isaac's own check, context only
PYTHONPATH=. $VSBENCH_PY tools/u9_diagnose.py --scene hollywood --arm artvs --worlds isaac_fit_v1 raw mock
VSBENCH_WRITE_GATES=1 $VSBENCH_PY -m pytest tests/world/test_check_baselines.py -q   # results/bridge/u9_check.json
```

The rows: **identity** - the camera at the goal, the goal picture as the live frame; vs-sim's
verdict `ok` (|v| < 1e-4 m/s, |w| < 0.01 deg/s, at least one match). **near** - the benchmark's
near sampler on the scene's goal, first draw (`n_000`, 7.76 mm / 2.69 deg on every scene, the
same start Isaac's check flew), 200 steps; vs-sim's script computes no verdict for it, its
docstring says "it must land closer than it started": the twin's `near_ok` is exactly that
(final < start; "by a lot" is not quantified there and not thresholded here). **capped30** - the
first start of the scene's `capped30` set, flown to the cell's cap (600 steps): recorded, never
thresholded (vs-sim's check gates nothing on it either).

Isaac's numbers are its own `check_baselines` runs on the station (read only):
`~/genvs_ws/vssim_out/check1` (2026-09-17, the same 12-arm protocol labels, same near start,
200 steps, CUDA) for identity and near; `check2` / `check3` (2026-09-17, first `capped30`
start, hollywood and cluttered2 only) for the far row. Context only, never pooled (D-9).

## Result

| scene | arm | identity (n / \|v\| m/s / \|w\| deg/s) | near start -> final mm, deg (MuJoCo) | near | Isaac near final mm, deg [closer] | capped30 start -> final mm (MuJoCo) | Isaac capped30 start -> final mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hollywood | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.009, 2.69 -> 0.004 | closer | 0.171, 0.018 [yes] | w1_034: 479 -> 0.0 | w1_034: 479 -> 0.2 |
| hollywood | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.592, 2.69 -> 0.096 | closer | 0.427, 0.070 [yes] | w1_034: 479 -> 1.0 | - |
| hollywood | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 5.755, 2.69 -> 0.486 | closer | 3.962, 0.510 [yes] | w1_034: 479 -> 6.0 | w1_034: 479 -> 9.8 |
| hollywood | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 20.111, 2.69 -> 2.542 | NOT closer | 1.615, 0.021 [yes] | w1_034: 479 -> 3.5 | w1_034: 479 -> 2.9 |
| poster2 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.029, 2.69 -> 0.006 | closer | 0.146, 0.022 [yes] | w1_034: 479 -> 0.0 | - |
| poster2 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.661, 2.69 -> 0.121 | closer | 0.415, 0.039 [yes] | w1_034: 479 -> 1.1 | - |
| poster2 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 19.560, 2.69 -> 2.713 | NOT closer | 14.890, 2.105 [no] | w1_034: 479 -> 8.5 | - |
| poster2 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 9.515, 2.69 -> 2.436 | NOT closer | 17.155, 3.282 [no] | w1_034: 479 -> 6.1 | - |
| poster3 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.159, 2.69 -> 0.022 | closer | 0.165, 0.023 [yes] | w1_076: 293 -> 0.0 | - |
| poster3 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.363, 2.69 -> 0.037 | closer | 0.210, 0.010 [yes] | w1_076: 293 -> 4.0 | - |
| poster3 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 8.971, 2.69 -> 0.727 | NOT closer | 8.839, 0.917 [no] | w1_076: 293 -> 15.5 | - |
| poster3 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 2.158, 2.69 -> 0.204 | closer | 24.053, 2.769 [no] | w1_076: 293 -> 2.3 | - |
| industrial1_drill | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.157, 2.69 -> 0.022 | closer | 0.175, 0.019 [yes] | w1_034: 479 -> 2982.0 | - |
| industrial1_drill | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.401, 2.69 -> 0.033 | closer | 1.011, 0.107 [yes] | w1_034: 479 -> 11397.2 | - |
| industrial1_drill | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 25.051, 2.69 -> 2.883 | NOT closer | 19.802, 2.261 [no] | w1_034: 479 -> 415.3 | - |
| industrial1_drill | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 35.351, 2.69 -> 4.041 | NOT closer | 29.710, 3.106 [no] | w1_034: 479 -> 97.8 | - |
| cluttered2 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.121, 2.69 -> 0.007 | closer | 0.662, 0.026 [yes] | w1_034: 479 -> 0.5 | w1_034: 479 -> 20815.1 |
| cluttered2 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.649, 2.69 -> 0.056 | closer | 1.701, 0.094 [yes] | w1_034: 479 -> 2.0 | - |
| cluttered2 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 57.352, 2.69 -> 2.492 | NOT closer | 43.754, 1.939 [no] | w1_034: 479 -> 1052.4 | w1_034: 479 -> 1100.6 |
| cluttered2 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 39.235, 2.69 -> 2.532 | NOT closer | 99.233, 4.809 [no] | w1_034: 479 -> 3144.9 | w1_034: 479 -> 2078.1 |
| cluttered3 | sift_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 0.225, 2.69 -> 0.007 | closer | 2.186, 0.082 [yes] | w1_076: 293 -> 0.4 | - |
| cluttered3 | orb_ransac | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 2.401, 2.69 -> 0.122 | closer | 1.814, 0.056 [yes] | w1_076: 293 -> 3.3 | - |
| cluttered3 | vitvs_dinov3 | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 42.899, 2.69 -> 1.895 | NOT closer | 47.670, 2.061 [no] | w1_076: 293 -> 41.1 | - |
| cluttered3 | artvs | OK: 24 / 0.0e+00 / 0.0e+00 | 7.76 -> 7.506, 2.69 -> 0.218 | closer | 2.234, 0.196 [yes] | w1_076: 293 -> 2546.1 | - |

- **Identity: 24 / 24 green** (every arm re-finds its 24 points with zero displacement and
  commands exactly zero motion; Isaac's check1: the same 24 / 24).
- **Near: 15 / 24 closer.** The classical arms (SIFT, ORB) come closer on all six scenes
  (0.009-2.40 mm). ViT-VS DINOv3 comes closer on hollywood only (5.76 mm); ART-VS on hollywood
  no, on poster3 (2.16 mm) and cluttered3 (7.51 mm) yes.
- **The nine misses, diagnosed** (`results/bridge/u9_check.json`, `near_fail_diagnosis`;
  `diagnose.json`):
  - **eight are the method as in Isaac** - Isaac's own check missed the same cell from the same
    start, by similar amounts: ViT-VS poster2 19.6 mm (Isaac 14.9), poster3 8.97 (8.84),
    drill 25.1 (19.8), cluttered2 57.4 (43.8), cluttered3 42.9 (47.7); ART-VS poster2 9.5 (17.2;
    both end on the divergence abort, at step 15 here and 16 in Isaac), drill 35.4 (29.7),
    cluttered2 39.2 (99.2). Not the look either: with the Isaac look off (`raw`) ViT-VS on
    poster2 lands 28.8 mm, on the drill 23.5, ART-VS on the drill 15.9; on the copied numpy mock
    (no renderer at all) ViT-VS on poster2 lands 20.2 mm.
  - **one is not the look and not the world: hollywood / ART-VS** (20.1 mm here, 1.6 mm in
    Isaac). Re-flown it reproduces 20.111 mm to the last digit; it misses without the look
    (`raw`, 13.8 mm) and on the copied mock with the hollywood texture (18.3 mm). In all three
    non-Isaac worlds ART-VS never leaves its coarse phase (24 matches throughout, the switch to
    tiles never fires) and the coarse phase walks away from the goal; Isaac's record shows the
    switch fired (35.9 matches on average) and the tiled phase landed it. The reverse happens on
    poster3 (2.16 mm here, 24.1 mm in Isaac). So ART-VS from this near start is a coin flip on
    pixel detail - whether the coarse error falls under the switch threshold - which the U5
    parity flights saw too (`near60/n_002` never switches on the mock nor on MuJoCo). One
    suspect is not tested: Isaac ran DINOv2 on CUDA, the twin on the CPU (the copied fast tier
    holds CPU tokens to 1e-5; the GPU stays off limits while the evaluation runs).
- **capped30, recorded**: SIFT lands 0.0-0.5 mm on five scenes and is lost on the drill
  (479 -> 2982 mm); ORB 1.0-4.0 mm on
  five, lost on the drill; ViT-VS 6.0 / 8.5 / 15.5 / 41.1 mm on hollywood / poster2 / poster3 /
  cluttered3, lost in the drill room (415 mm) and cluttered2 (1052 mm; Isaac 1101); ART-VS 3.5 /
  6.1 / 2.3 mm on the posters (Isaac hollywood 2.9), lost in the rooms (97.8, 3145, 2546 mm;
  Isaac cluttered2 2078). Isaac's cluttered2 SIFT went to 20.8 m where ours landed 0.5 mm.

## Cost

CPU (user + system, `/usr/bin/time`): hollywood 6977 s, poster2 6642, poster3 8519,
industrial1_drill 7379, cluttered2 5400, cluttered3 5228 - **40 145 s = 11.2 CPU-hours**;
wall 3 h 03 min on two processes (11:57-15:00, load ~23-25 from the other users). Peak RSS
2.2-2.4 GB on the posters and the drill room, 4.4-4.8 GB on cluttered2 / cluttered3. The
diagnosis flights: ~1 200 CPU-s; the film smoke 367 CPU-s. Per step (near rows, CPU): SIFT
0.74-1.77 s, ORB 0.25-0.58, ViT-VS 0.66-1.14, ART-VS coarse 0.32-0.63, tiled ~3.6 s.

## Files

- `<scene>/check.json`, `<scene>/check.md` - vs-sim's report (the copied `write_table`), one per
  scene process, plus the bridge block (`vsbench_commit`, `vssim_commit`, the station env, the
  look, CPU seconds) and `render_mode` on every scene and entry;
- `<scene>/records.jsonl` - every flown episode's full copied record (`check_row` near / far);
- `<scene>/run.log` - the progress lines, the table, `/usr/bin/time`;
- `<scene>/*_preview.png` - the goal picture and the first live frame of the near start
  (quarter size; `--frames preview`, the goal depth's md5 in check.json);
- `summary.json`, `summary_table.md` - the merged matrix with Isaac's numbers beside each cell;
- `diagnose.json` - the diagnosis flights (`tools/u9_diagnose.py`);
- `film_smoke/` - the copied `FilmWriter` on this world: hollywood / SIFT, the near row with
  `--film --film-every 4` (`films/.../episode.mp4`, 187 KB); it lands where the matrix's flight
  did (0.009 mm), the film does not change the flight.

**Provenance.** Every scene flew vsbench `e4b2f0c` (the driver's commit) and vs-sim
`7ac261d7` (the copy's pin). cluttered2 and industrial1_drill read clean; the other four read
`e4b2f0c+dirty` - the dirt is the then-untracked new files `tests/world/test_check_baselines.py`
(and `tools/u9_diagnose.py` for the diagnosis flights), which no flight imports; both are in
the U9 commits.

## In plain words

We asked each of the four baselines three questions on the new light simulator, the same way
the Isaac rig check asks them. First: put the camera exactly where it should be and show the
method its own goal picture - does it stay still? All four did, on all six scenes. Second:
start 8 mm away - does it end closer? The two classical methods always did, to well under a
millimetre most of the time. The two learned ones (ViT-VS, ART-VS) mostly did not - but in
Isaac they mostly did not either, from the very same start, by similar amounts: that is how
these methods behave close to the goal, not a fault of the new simulator. One case differs
(ART-VS on hollywood): Isaac landed it, the light simulator did not; it also fails on a plain
picture-warping stand-in with no renderer at all, so it is not the new simulator's picture -
ART-VS's switch from its rough to its fine mode is a coin flip here, and it fell the other way
(on poster3 it fell the other way round). Third: a far start - recorded for the record.
