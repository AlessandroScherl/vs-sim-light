# PROVENANCE — the copy of vs-sim inside vsbench

**Nothing under `vsbench/vs/`, `vsbench/configs/`, `vsbench/benchmark/`, `vsbench/assets/`,
`vsbench/tests/` or `vsbench/scripts/` is edited by hand** (this page and `COPY_MANIFEST.json`
excepted). A baseline is run as published; a knob that changes a published method is a
different arm config with a different label, never an edit. A vs-sim fix comes in by moving
the pin (below), never by an edit here.

Owner decisions of 2026-10-22: **D-1 copy** (vs_sim_light stands alone; vs-sim is not a
runtime dependency) and **D-2 separate for good** (the MuJoCo world stays here).
Plan: `docs/PORT_PLAN.md` §2; layout **B, the mirror** (parent decision 2026-10-22, after the
first attempt with the plan's `vsbench/vs/data/` layout needed ~18 patched files).

## What was copied, from where

- **Source:** vs-sim (`~/runpod_dev/vs-sim`, read through the local clone
  `external/vs-sim`) at commit **`7ac261d7384861b1d1ad0cd645c6e51b5046b650`**
  (`external/VSSIM_COMMIT`; 2026-10-22 16:47 +0200, "Chain: wait for free GPU memory ...").
  Every blob was read with `git show <pin>:<path>`, never from a working tree.
- **Tool:** `tools/vssim_copy.py` (writes `COPY_MANIFEST.json`: per file the source path, blob
  sha, mode, the rules that fired, the patch id, the sha256 of what was written).
- **Layout B:** `vsbench/` plays the part of vs-sim's repository root, so every
  `Path(__file__).parents[k]` in the copied code lands where it lands in vs-sim:

  | vs-sim | here | how |
  | --- | --- | --- |
  | `vssim/` (11 modules, `baselines/` 13, `bench/` 13) | `vsbench/vs/` | rewrite rules + provenance line |
  | `configs/` (camera, scenes, protocol, 13 arm configs, 3 layouts) | `vsbench/configs/` | byte-identical |
  | `benchmark/` (README + 29 start files + 29 md5 sidecars) | `vsbench/benchmark/` | byte-identical |
  | `assets/asset_library_v2.json` | `vsbench/assets/asset_library_v2.json` | byte-identical |
  | `tests/conftest.py`, `tests/fast/` (37 files) | `vsbench/tests/conftest.py`, `vsbench/tests/fast/` | rewrite rules + provenance line |
  | `tests/fixtures/` (56 files, incl. the MIT-vendored `reference/`) | `vsbench/tests/fixtures/` | byte-identical (the reference sources are md5-pinned to their releases) |
  | `scripts/` (8 drivers + `chains/`, 5 files) | `vsbench/scripts/` | rewrite rules (`.sh`: R1, R2 only) + provenance line; **reference only** |
  | `experiments/README.md`, `.gitkeep` | `vsbench/experiments/` | byte-identical (a `test_run.py` test watches git see a tracked file there) |

  225 files: 88 rewritten, 3 of them patched, 137 byte-identical.
- **Left behind:** `tests/slow/` (station / Isaac checks), `examples/`, `docs/`, `DESIGN.md`,
  `README.md`, `CLAUDE.md`, `pyproject.toml`, `.gitignore` (link, don't copy — the owner's
  one-home rule). `test_package.py::test_nothing_refers_to_the_research_repository` therefore
  walks no `examples/` here (vs-sim has two files there).

## The rewrite rules (`tools/vssim_copy.py::RULES`, in this order)

| id | pattern | becomes |
| --- | --- | --- |
| R1 | `\bfrom vssim import\b` | `from vsbench.vs import` |
| R2 | `\bfrom vssim\.` | `from vsbench.vs.` |
| R3a | `\bimport vssim\.` | `import vsbench.vs.` |
| R3b | `\bimport vssim\b` (not dotted) | `import vsbench.vs` |
| R4 | `\bvssim\.(?=[A-Za-z_])` | `vsbench.vs.` (docstrings, dotted module strings) |

R3b binds `vsbench`, so it agrees with R4 (the plan's draft `from vsbench import vs as vssim`
bound `vssim` while R4 spelled its uses `vsbench.vs.`, unbound). Hyphenated / underscored
names (`vssim-config-1`, `vssim_out`, `vssim_version`, `VSSIM_*`) never match: they are schema
names and record data. `.sh` files get R1 and R2 only (the chains' Python heredocs).
Line 1 of every rewritten file (line 2 after a shebang) is
`# vs-sim copy: <src> @ <sha> (tools/vssim_copy.py); edits: rewrite rules [+ patch <id>]  # noqa: E501`.

## The patches (`tools/vssim_copy_patches.json`; cap 6, three used)

| id | file | pairs | why |
| --- | --- | --- | --- |
| P1 | `vsbench/tests/fast/test_package.py` | 3 | the hygiene test names the package by its import name: `CORE_ALLOWED` gets `vsbench`; the fresh-interpreter no-torch check imports `vsbench.vs`; `hasattr(vsbench.vs, name)` (the one bare use of the module object) |
| P2 | `vsbench/vs/bench/run.py` | 2 | `__import__("vsbench.vs", fromlist=["__version__"]).__version__` (there is no `vssim`; the record key stays `vssim_version`); `git -c color.status=false -c status.relativePaths=true -c status.branch=false -c status.showStash=false status --short` instead of `status --porcelain` so `dirty_paths` reports paths relative to `REPO_ROOT = vsbench/` (porcelain is relative to the repository top, and the `experiments/` filter never matched in the nested copy); the four `-c` settings pin the output against a user's git config (`relativePaths=false` would bring top-relative paths back, `branch` / `showStash` would add lines the parser reads as paths) |
| P3 | `vsbench/tests/fast/test_debuglog.py` | 1 | the throwaway checkout the debug chain is driven in symlinks `vs`, not `vssim` |

The plan's `camera.py` data-root patch is not needed in layout B: vs-sim's own
`REPO_ROOT = Path(__file__).resolve().parents[1]` is `vsbench/`.

## Run provenance (what a record says)

`vsbench.vs.bench.run.start_run` writes `run.json` with `git_commit` = **this repository's**
commit (it runs `git -C vsbench/`), `git_dirty`, `git_dirty_paths` (relative to `vsbench/`;
files elsewhere in this repository appear as `../path`), `allow_dirty`, and `vssim_version`
(the copy's `__version__`). It refuses a dirty tree unless `allow_dirty=True`; a driver here may
pass it, and the record then says `allow_dirty: true` with the dirty paths. The bridge's
drivers (U2) add `vssim_commit` (from `COPY_MANIFEST.json`) and `render_mode`, so a number is
always traceable to the exact published-baseline code that produced it.

## What is flown here, what is copied but not wired

- **Flown on the MuJoCo world (from U2 on):** ViT-VS (DINOv3 ViT-S/16, published DINOv2
  ViT-S/14 chain), ART-VS (36-tile latching switch), the 4-way pre-align, the classical
  SIFT / ORB / AKAZE arms ± RANSAC (OpenCV pinned to 4.11.0.86, the flight interpreter's), the
  bench (loop, records, scoring, stats, perturbation `noise`, films, debug log), the scenes,
  the frozen starts, the camera.
- **Copied, not wired** (their dependencies are not in this venv; their tests skip or use
  stubs exactly as in vs-sim): CNS v1 (`$CNS_ROOT`, torch_geometric), DINOBot (DINO v1 ViT-S/8),
  LightGlue, RoMa, ViSP IBVS, ART-VS object mode (`objroi`: GroundingDINO / LangSAM /
  LightTrack). `isaac_scene.py` and `bake.py` are the **reference** world: importable without
  Isaac, `IsaacScene(...)` needs Isaac Sim. `vsbench/scripts/` are vs-sim's Isaac drivers,
  present only so vs-sim's tests of them run; nothing here runs them.

## Proving and keeping faithfulness

```
source tools/cpu_env.sh
env -u VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT $VSBENCH_PY -m pytest vsbench/tests/fast -q   # the copied tier
$VSBENCH_PY -m pytest tests/vs -q                       # the drift check + the glob guards
$VSBENCH_PY tools/vssim_drift.py                        # exit 0 = the copy is the pin's copy
.venv/bin/ruff check --config tools/ruff_vssim_copy.toml vsbench/vs vsbench/tests vsbench/scripts
```

- The copied tier must give the pristine clone's split test for test
  (`results/bridge/u0_clone_fast_tier.txt`, at the pin and OpenCV 4.11: 789 passed, 13 skipped).
  The two asset-root variables must be unset (three `test_assets.py` tests assert the defaults).
  Run it in its own pytest process: vsbench's `tests/conftest.py` and the copy's
  `vsbench/tests/conftest.py` are both imported as the module `conftest`.
- The copy's `conftest.py` puts `vsbench/` on `sys.path` (as vs-sim's puts its root), so inside
  that test process vsbench's own modules are also importable top-level (`scene`, `camera`, ...);
  nothing in the tier imports them that way.
- `tests/vs/test_drift.py` (hand-written, outside the roots): byte equality with the pin through
  the rules and patches, no file under the roots outside the manifest, every entry at the pin,
  ≤ 6 patches each with a reason, and **no glob loop of the copied tier can pass on an empty
  directory** (`test_control.py`'s and `test_arms.py`'s arm-config loops, `test_package.py`'s
  walks, the start files: each directory, as the copied test resolves it, holds as many files
  as at the pin).
- Ruff: vs-sim's rule set (`E, F, I, UP, B, SIM`, line 100) in `tools/ruff_vssim_copy.toml`;
  the copied code is never reordered or rewrapped, so the lines the rewrite pushed past 100
  columns (and the two imports isort would now wrap) are per-file ignores there, listed.
  vsbench's own ruff run excludes the copied trees (`pyproject.toml`).

## Pulling a vs-sim fix in (deliberately, whole pin only)

1. `git -C external/vs-sim fetch` (reads the local vs-sim checkout or GitHub; never writes it).
2. `tools/vssim_drift.py --upstream <newsha>` — per copied file, whether upstream changed it, with the diff.
3. `tools/vssim_copy.py --from <newsha> --set-pin` — refuses if any copied file was edited by
   hand, a stray file sits under the roots, or a file it did not write is in the way; it only
   ever deletes files the previous manifest lists. A patch whose `old` text moved makes it stop:
   re-review the patch.
4. Re-run the copied tier (and the clone's tier at the new pin for the new split), the drift
   test and ruff; commit with a message naming the vs-sim commits pulled.

## Upstream findings to report when vs-sim unfreezes

vs-sim is frozen while the T1 Stage 2 evaluation flies; these are written down here, never
patched in the copy (a fix comes in by moving the pin).

- **The film encoder doubles a relative path** (`vssim/bench/films.py::encode_episode`, copied
  as `vsbench/vs/bench/films.py`). It runs ffmpeg with `cwd=str(ep_dir)` but passes
  `-i str(ep_dir / "_frames.txt")` and the output `str(ep_dir / out_name)` as given; when
  `ep_dir` is relative, ffmpeg resolves them inside `ep_dir` again (`<ep_dir>/<ep_dir>/...`),
  fails, and the encoder falls back to cv2's writer (`[film] ffmpeg failed ..., cv2
  fallback`). vs-sim's own drivers pass absolute run directories, so it never shows there.
  Fix upstream: `ep_dir = Path(ep_dir).resolve()` at the top of `encode_episode` (or pass
  `lst.name` / `out.name` to ffmpeg). Here `vsbench.world.fly` resolves `--out` to an
  absolute path before any film is written (found in U2, 2026-10-22).
