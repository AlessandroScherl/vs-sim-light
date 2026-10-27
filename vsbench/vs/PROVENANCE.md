# Provenance of the vs-sim copy

The benchmark loop, the scoring, the scenes, the frozen start poses and the published
baselines in this package were **copied from vs-sim**, the Isaac Sim visual-servoing
benchmark, at commit **`7ac261d7384861b1d1ad0cd645c6e51b5046b650`** (2026-10-22). vs-sim is
not a dependency: everything needed is here.

## What was copied (172 files)

| vs-sim | here |
| --- | --- |
| `vssim/` (the scene contract, camera, scenes, starts, benchmark, labels, assets, the Isaac scene's Isaac-free helpers) | `vsbench/vs/` |
| `vssim/baselines/` (ViT-VS and ART-VS matching, the SIFT / ORB / AKAZE matcher, the IBVS control law and stop rules, the 4-way pre-alignment) | `vsbench/vs/baselines/` |
| `vssim/bench/` (the loop, records, scoring, statistics, perturbations, films, debug log and view) | `vsbench/vs/bench/` |
| `configs/` (camera, scenes, protocol, the arm configs of the kept baselines, room layouts) | `vsbench/configs/` |
| `benchmark/` (the frozen start files and their md5 sidecars) | `vsbench/benchmark/` |
| `assets/asset_library_v2.json` | `vsbench/assets/` |
| `tests/fast/`, `tests/fixtures/` (the copied test tier; vendored reference sources with their licences) | `vsbench/tests/` |
| `scripts/check_baselines.py`, `scripts/fly.py` (the rig check `vsbench.world.check_baselines` runs) | `vsbench/scripts/` |

`COPY_MANIFEST.json` lists every copied file with its source path, source blob and the sha256
of the bytes written here.

## How they were copied, and that they are unmodified

Each file was read from the vs-sim commit above (`git show <commit>:<path>`), never from a
working tree. Data files are byte-identical (apart from `P4` in four fixture notes). Python files had their import name rewritten,
`vssim` to `vsbench.vs` (five regular expressions, listed in the manifest), and got a one-line
provenance comment at the top; three files carry a small documented patch (`P1`-`P3` in the
manifest: the import name in a hygiene test and in the run-provenance code, a symlink name in
a test), and thirteen carry the comment- and documentation-only `P4` (the name of the source
repository's internal rules file and local home paths neutralised; no code line changed).
Nothing else was changed: a baseline runs exactly as published, and a different
setting is a different arm config, never an edit.

**The retained files are unmodified since the copy.** `tests/test_copy_faithful.py` checks
every file against its sha256 in the manifest, and that no other file sits under the copied
directories. A copied test whose subject was left out is deselected, not edited
(`tests/copied_deselect.txt`; `tools/test_copied.sh` runs the copied tier).

## What was left out

- Methods not wired into this simulator: CNS v1, DINOBot, LightGlue, RoMa, ViSP IBVS and
  ART-VS's object mode (`objroi`), with their arm configs, tests and fixtures. The copied
  `configs/protocol.yaml` still names their cells; `python -m vsbench fly` refuses them.
- Isaac Sim itself: the layout baker (`bake.py`), the Isaac drivers and run chains of
  `scripts/`, the slow test tier. `vsbench/vs/isaac_scene.py` stays for its Isaac-free
  helpers (the object scatter, the re-orthogonalisation); its `IsaacScene` class needs Isaac Sim.
- vs-sim's documentation, examples and packaging.

The vendored CNS stop-policy reference (`tests/fixtures/reference/cns_stop_policy.py`, MIT)
stays: `cns_plateau`, CNS's stop rule, is the rule every kept baseline is scored with.

## What is not from vs-sim

The MuJoCo world (`vsbench/world/`), the renderer (`vsbench/render/`), the method interface
(`vsbench/methods/`), the command line (`vsbench/__main__.py`) and the runtime
(`vsbench/runtime.py`, `vsbench/glenv.py`) belong to this repository. Every record carries
`vssim_commit` (the commit above) and `render_mode`, so a number from this simulator is never
mistaken for an Isaac Sim number.
