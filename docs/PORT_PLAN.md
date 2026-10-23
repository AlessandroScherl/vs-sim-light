# PORT_PLAN — vs-sim's capabilities, objects, depth camera and the ViT-VS / ART-VS baselines on the MuJoCo simulator

*Planner: Fable, 2026-10-22, for the owner's ask of the same day ("port the main capabilities
and baselines art-vs and vit-vs from our current isaac sim setup to there too ... the objects
... the depth cameras ... without stopping or hindering the current eval run"). Builders: Opus,
one unit at a time (§7). Reviewer: Fable. Every fact below was read from code or checked with a
read-only command on the station on 2026-10-22; nothing was guessed. Nothing here touches
`~/runpod_dev/vs-sim`: that repository is frozen while the T1 Stage 2 evaluation flies.*

*Revision 2 (same day, after the owner's rulings): **D-1 = COPY** — vs_sim_light stands alone;
the baseline, bench, scene and start code is copied into `vsbench`, not imported from a library.
**D-2 = the MuJoCo world stays here for good** and never moves into vs-sim. §2 and §7 were
rewritten for that; D-3 … D-9 stay open with the recommendations as working defaults.*

*Revision 3 (2026-10-23, after the Fable review of Milestone 1, verdict PASS WITH
CORRECTIONS): §2.1–2.4 now describe the copy **as built** (layout B, the mirror: `vsbench/`
plays vs-sim's repository root, the copied fast tier is `vsbench/tests/fast`, `tests/vs/` holds
the drift test only, patches P1/P2/P3, the plan's `camera.py` patch dropped); §4 gate 3 is
ECC-gated with a calibration row, phaseCorrelate reported plus two null tests, the
marker-centroid signed mean as the primary evidence; §7 records the OpenCV 4.11.0.86 pin and
the 789 / 13 split (U0), the ViT-VS curve statistics (U2) and the station rules as enforced
(`LP_NUM_THREADS=2`, `torch.set_num_interop_threads(2)`, the hard GPU refusal); §8 D-3 now
carries measured evidence, and two risks were added (the pre-U1 llvmpipe thread gap, a
renderer path that bypassed software GL).*

---

## 0. Scope, in one paragraph

**Source** — `~/runpod_dev/vs-sim` (package `vssim`, HEAD today `7ac261d7`): the Isaac Sim 5.1
benchmark. Its contract is `vssim.scene.Scene` — four verbs (`reset(pose)`, `render() ->
(rgb, depth, pose)`, `apply(twist, dt)`) and four facts (`intrinsics`, `goal_pose`,
`working_distance`, `pose`). Everything that matters for this port is *behind* that contract
and is pure Python: the published baselines (`vssim.baselines`: ViT-VS, ART-VS, CNS v1,
DINOBot, classical), the episode loop and records (`vssim.bench`), the six scenes' geometry
(`configs/scenes.yaml`, `configs/layouts/*.json`, `vssim.scenes`), the frozen starts
(`benchmark/starts/*.json`, md5-pinned, `vssim.benchmark`), the camera (`configs/camera.yaml`,
`vssim.camera`). The package core imports numpy and PyYAML only; Isaac is imported inside
methods; torch lives in `vssim.baselines` / `vssim.bench` only (`tests/fast/test_package.py`
asserts both).

**Target** — `~/genvs_ws/vs_sim_light` (package `vsbench`, branch `port/vssim`): a MuJoCo
renderer behind a `Renderer` protocol (`render(T_wc) -> (rgb, depth)`), a free-flying SE(3)
camera, a servo loop with a method registry (`VSMethod`, `@register`), episode logs and replay
videos, and the gates that prove each piece against analytic math. Steps 1–3 are done and
green (`results/STEP3_REPORT.md`: 13/13 gates, `sift_ibvs` 10/10 starts under 1 mm).

**What "port" means here (owner, D-1/D-2)** — vs_sim_light becomes a **stand-alone**
simulator + benchmark: a faithful copy of vs-sim's baselines, bench, scenes and starts lives
inside `vsbench`, a MuJoCo world satisfies the copied `Scene` contract, the 323 objects are
converted into a MuJoCo-readable cache, and the D435i depth camera is reproduced. A drift tool
keeps the copy honest against a pinned vs-sim commit so that a later vs-sim fix is pulled in
deliberately. Not in scope: any claim that a MuJoCo number replaces an Isaac number (§6), any
edit to `vs-sim`, anything that touches the GPU.

---

## 1. Facts verified on the station (what the builders will find)

| what | where / value | how checked |
| --- | --- | --- |
| Live eval | `docker` container `isaac` (up 6 days); locks `~/genvs_ws/night_{flight,gpu,isaac}.lock`; GPU 15.6 / 24.6 GB used; load 22 on 16 threads (AMD Ryzen 7 5800X, 62 GB RAM) | `docker ps`, `nvidia-smi`, `uptime` |
| Host Python | `/usr/bin/python3` 3.12.3 — **no** torch, pxr, mujoco, transformers anywhere on the host. The only torch is inside the Isaac container (off limits). | imports, find |
| vsbench venv | `~/genvs_ws/vs_sim_light/.venv` (py 3.12.3): mujoco 3.13.0, numpy 2.5.3, opencv-python-headless 5.0.0, Pillow, imageio-ffmpeg. U0 is adding CPU torch/torchvision, transformers, PyYAML, scikit-learn, usd-core, pytest. | imports |
| Software GL | Mesa EGL vendor json `/usr/share/glvnd/egl_vendor.d/50_mesa.json` present; **no libOSMesa**. So `VSBENCH_SOFTWARE_GL=1` + `mujoco-egl` (llvmpipe) is the CPU path. The bake-off ran exactly this on this box; EGL/llvmpipe and OSMesa were bit-identical. | ls, ldconfig |
| Weights (never download) | DINOv3 ViT-S/16: `~/genvs_ws/cache/hf/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/` (165 MB). DINOv2 ViT-S/14: `~/genvs_ws/torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth` (85 MB) + code `~/genvs_ws/torch_home/hub/facebookresearch_dinov2_main/`. Also there: DINO v1 ViT-S/8 (83 MB), RoMa, LightGlue, DINOv3-B. U0 copies the two needed folders to `~/genvs_ws/vsbench_assets/weights/{hf,torch}` so the shared caches are never opened for writing. | find, du |
| Object USDs | 323 library entries (`vs-sim/assets/asset_library_v2.json`): 223 under `~/genvs_ws/genvs-sim/experiments/ingest/usd/<cat>/<id>/<id>.usd` (18 GB tree), 94 under `~/genvs_ws/genvs-sim/assets/gso_usd/<cat>/<id>/<id>.usd` (2.6 GB), 6 Nucleus (`/Isaac/Props/YCB/Axis_Aligned/*.usd`, **not on disk**). **0 of the 317 local files are missing.** Every file is binary `PXR-USDC`, with `metersPerUnit`/`upAxis` metadata, a `UsdPreviewSurface` + `UsdUVTexture` material and one texture in `textures/*.png|jpg` beside it. Sources: gso 105, objaverse 103, oo3d 56, housecat6d 39, handal 14, isaac_native 6. Categories: shoe 103, bottle 45, teapot 38, mug 35, hammer 31, drill 25, tv_remote 12, toy_car 10, drinking_glass 6, wine_bottle 6, tube 4, stapler 4, can 4. | python over the json, `head -c 8`, `strings` |
| YCB scans (rooms) | `~/genvs_ws/assets/ycb/<id>/` (89 dirs, 2.0 GB): the Isaac USD (`usd/<id>.usd`, **OmniPBR MDL** material) **and** the original `google_16k/textured.obj` + `textured.mtl` + `texture_map.png`. | ls |
| Scene textures | `~/genvs_ws/assets/{hollywood/resized.png 1500×1125 RGBA, poster2/poster2.png 375×500, poster3/poster3.jpg 3840×2810, industrial1/ground.png 1024², cluttered2/ground.png 4096², cluttered3/ground.png 4096²}` (the yaml's `/workspace/...` is `~/genvs_ws/...` on the host) | PIL |
| Camera | `configs/camera.yaml`: 1920×1080, 42° vertical FoV, square 20.955 mm aperture → `fx = fy = 1406.75`, principal point `(960, 540)`; `vssim.scene.D435I` pins the same. Isaac camera: bare `UsdGeom.Camera`, clipping 0.001–1000 m. Depth annotator: **`distance_to_image_plane`** (= camera-frame z, metres). No depth noise model anywhere in vs-sim (the `noise` perturbation is RGB-only). | camera.py, isaac_scene.py:704-729 |
| Isaac lighting | dome light intensity 3000, colour (0.2, 0.2, 0.2); distant light 3500, angle 0.53°, rotateXYZ (−30, 0, 30). | isaac_scene.py:704-710 |
| Poster geometry | quad `[-0.4, 0.4] × [-0.3, 0.3]` at `z = 0`, texture row 0 at world +Y; backdrop 4 × 4 m grey 0.5 at `z = −0.02`; goal `(0, 0, 0.4341815108)`, `R = diag(1, −1, −1)`. | isaac_scene.py:816-897, scenes.py |
| Room geometry | ground: 2.4 m textured quad at `z = 0`, single span; objects: wrapper `/World/Obj_i` = `Translate(pos) · Orient(quat) · Scale(s)` applied to the asset's native coordinates, `pos`/`quat_wxyz` frozen in `configs/layouts/<scene>.json`; `s = 1` for `ycb/...` ids, `target_for(category)` longest-extent rule for curated ids (drill → 0.25 m); goal = `scene_view_goal_pose(cam, bbox_min, bbox_max, frame_fill 0.85)` from the **measured** bbox. Frozen start files carry the resulting goal: `industrial1_drill` d = 0.4422820342; `cluttered2` d = 1.1689734192, look_at `(0.0469, 0.1515, 0.2424)`; `cluttered3` d = 1.2853190576. `check_scene_goal` tolerance: 5e-4 m / 0.05° / 1e-3 m. | isaac_scene.py:900-1125, layouts, starts |
| What an arm needs | `Arm.reset(goal_rgb, goal_depth, goal_mask)` then `Arm.step(rgb, depth) -> (twist, debug, matches)`; depth read at the current pixels by truncation (`depth_at`), invalid → 0.5 m. Intrinsics go in at `build_arm(cfg, intrinsics=..., ...)`. No arm ever sees a pose. | arms.py:250-480 |
| The loop | `vssim.bench.runner.run_episode(scene, arm, start, config, *, starts_payload, telemetry, on_step, perturbation, debug_log, live_mask_fn)` → record dict; touches a scene only through the four verbs + four facts plus optional `getattr` extras. Integration `integrate_optical_velocity`, dt = 1/60, gain 2.0, cap 600. vs-sim already has an **external-arm hook** (`check_baselines --arm-plugin`, `tests/fast/test_arm_plugin.py`, 2026-10-11): an outside module may build an `Arm` and the runner flies it. | runner.py, test_arm_plugin.py |
| Isaac-only vs Isaac-free in `isaac_scene.py` | the module imports without Isaac (lazy imports) and holds Isaac-free functions the copy needs: `scatter_items` (used by `tests/fast/test_scenes.py` ×3 and by any scene without a baked layout), `render_mode_of`, `optical_R_to_usd_quat`. Nothing else in `vssim/` imports it except docstrings. | grep |
| Sizes of what is copied | `vssim/` 17.8 k lines in `baselines/` + `bench/` plus ~3 k in the core; `configs/` 140 KB; `benchmark/starts/` 3.3 MB; `tests/fixtures/` 4.4 MB (incl. the MIT-vendored `reference/` CNS stop policy and ART-VS controller the control tests run beside the port); `tests/fast/` 38 files. | du, ls |
| Path roots inside the copied code | `vssim/camera.py: REPO_ROOT = Path(__file__).resolve().parents[1]` is the one data root (`configs/`, `benchmark/starts/`, `assets/asset_library_v2.json` hang off it; used by `assets.py`, `benchmark.py`, `scenes.py`, `bench/run.py`, `bake.py`). Tests: `FIXTURES = parents[1]/"fixtures"` (test_match, test_features, …) and `REPO = parents[2]` (test_protocol, test_debuglog, test_arm_plugin). Package-name strings outside imports: `test_package.py` `CORE_ALLOWED = {"numpy","yaml","vssim"}`, `test_classical.py` `importorskip("vssim.scenes")`, `test_cns.py` `__import__("vssim.baselines.match")`, `test_match.py` `match="vssim.baselines.cns.build_cns_arm"`, `test_debuglog.py:1069` `("vssim","configs","benchmark")`. Schema strings `vssim-config-1`, `vssim-protocol-1`, `vssim-debug-telemetry/1`, `vssim_out` are **data, not names** and must not change. | grep |
| vsbench ↔ vssim conventions | identical optical frame; vsbench `T_wc` = `vssim.scene.pose_to_T(Pose)`. **Different** integration (`T @ exp_se3(twist·dt)` vs the two-step rule) and **different pixel-centre convention** (§4). vsbench `Plane` texture row 0 at local +y = vssim mock's convention. | se3.py, motion.py, scene.py |

---

## 2. Architecture (owner rulings D-1 = copy, D-2 = separate for good)

### 2.1 Where the copy lives: layout B, the mirror (as built in U-copy)

`vsbench/` plays the part of vs-sim's **repository root**: `vssim/` becomes `vsbench/vs/` and
vs-sim's repository-level trees sit beside it inside `vsbench/`, so every
`Path(__file__).resolve().parents[k]` in the copied code lands on its counterpart (parent
decision 2026-10-22, after a first attempt with the `vsbench/vs/data/` layout of revision 2
needed ~18 patched files).

```
vsbench/vs/                      ← the copy of vssim/ (11 modules, baselines/ 13, bench/ 13):
                                   rewrite rules + provenance line; isaac_scene.py and bake.py
                                   are the REFERENCE world (importable without Isaac)
  COPY_MANIFEST.json             ← generated: per file src path, blob sha, mode, rules fired,
                                   patch id, sha256 written
  PROVENANCE.md                  ← hand-written: what, from which commit, what is wired
vsbench/configs/                 ← byte-identical (camera, scenes, protocol, 13 arms, 3 layouts)
vsbench/benchmark/               ← byte-identical (README + 29 start files + 29 md5 sidecars)
vsbench/assets/asset_library_v2.json   ← byte-identical
vsbench/tests/conftest.py, vsbench/tests/fast/ (37 files)   ← THE COPIED FAST TIER (rules)
vsbench/tests/fixtures/ (56 files, incl. the MIT-vendored reference/)   ← byte-identical
vsbench/scripts/ (8 drivers + chains/)   ← reference only (R1/R2 on .sh); nothing runs them
vsbench/experiments/{README.md,.gitkeep}  ← byte-identical (a test watches git see a tracked file)
vsbench/world/                   ← NEW code (hand-written): mujoco_scene.py build.py camera.py
                                   assets.py fly.py (+ method_arm.py check_baselines.py later)
vsbench/methods/vssim_arms.py    ← NEW (U6b): the copied arms as vsbench methods
tests/vs/test_drift.py           ← hand-written: the drift test and the empty-glob guards ONLY
tests/world/                     ← hand-written: the world's gates (U1, U2, U3)
tools/vssim_copy.py tools/vssim_drift.py tools/vssim_copy_patches.json
tools/ruff_vssim_copy.toml tools/convert_objects.py tools/cpu_env.sh
external/vs-sim  external/VSSIM_COMMIT   (U0; git-ignored clone + committed pin — the SOURCE
                                          of the copy and the drift reference, never imported)
```

225 files: 88 rewritten (3 of them patched), 137 byte-identical. vs-sim's `scene.py`,
`camera.py`, `scenes.py`, `assets.py` live at `vsbench/vs/…` and do not collide with
vsbench's own `vsbench/{scene,camera,scenes}.py`. The fence "nothing hand-written in the six
copied trees" is checked by the drift test (every file under them is in the manifest). One
caveat of the mirror: the copied `vsbench/tests/conftest.py` puts `vsbench/` on `sys.path` (as
vs-sim's puts its root), and it and vsbench's `tests/conftest.py` are both the module
`conftest`, so the copied tier runs in **its own pytest process**.

### 2.2 What is copied, what is left behind

| copied (verbatim modulo §2.3) | left behind | why |
| --- | --- | --- |
| `vssim/*.py` (11 modules incl. `isaac_scene.py`, `bake.py`), `vssim/baselines/*`, `vssim/bench/*` | — | the code that is flown (and the reference world) |
| `scripts/` (8 drivers + `chains/`) → `vsbench/scripts/`, **reference only** | — | they hard-wire `IsaacScene` and the station and nothing here runs them, but vs-sim's fast tier tests some of them, so they come along for those tests; the bridge writes its own drivers (U2, U9) |
| `configs/` whole, `benchmark/` whole, `assets/asset_library_v2.json`, `experiments/{README.md,.gitkeep}` | `examples/`, the rest of `experiments/`, `docs/`, `DESIGN.md`, `README.md`, `CLAUDE.md`, `pyproject.toml`, `.gitignore` | repo-level material; the owner's one-home rule says link, don't copy docs (`experiments/README.md` comes because a `test_run.py` test watches git see a tracked file there) |
| `tests/conftest.py`, `tests/fast/*` (37 files), `tests/fixtures/*` (56 files, 4.4 MB, incl. `reference/`) | `tests/slow/*` | station/Isaac-only checks |

Instance-suite scenes (`inst_*`) and their starts come along inside `scenes.yaml` and
`benchmark/starts/` for free; flying them needs the converted library (U7).

### 2.3 The rewrite: mechanical rules + a short, reviewed patch list

`tools/vssim_copy.py --from <sha>` reads every source blob with `git -C external/vs-sim show
<sha>:<path>` (never the working tree), applies the rules below, prepends one provenance line,
writes the file and the manifest entry. The **same function** is what `tools/vssim_drift.py`
applies to the source before diffing, so the two can never disagree.

Rules as built (`tools/vssim_copy.py::RULES`; regexes, in this order, on `.py` files; `.sh`
files get R1 and R2 only; yaml/json/md/npz are copied byte-identical):
1. `\bfrom vssim import\b` → `from vsbench.vs import`
2. `\bfrom vssim\.` → `from vsbench.vs.`
3. `\bimport vssim\.` → `import vsbench.vs.`  and bare `\bimport vssim\b` → `import vsbench.vs`
   (R3b; revision 2's `from vsbench import vs as vssim` bound `vssim` while rule 4 spelled its
   uses `vsbench.vs.`, unbound — the built rule binds `vsbench` and agrees with rule 4)
4. `\bvssim\.(?=[A-Za-z_])` → `vsbench.vs.` everywhere else — docstring references, dotted
   module strings (`importorskip("vssim.scenes")`, `__import__("vssim.baselines.match")`,
   `match="vssim.baselines.cns..."`) — so that error messages and tests keep naming modules
   that exist. Hyphenated / underscored identifiers (`vssim-config-1`, `vssim_out`) do **not**
   match and stay as they are (they are schema names and record data).
5. The provenance line: `# vs-sim copy: <src path> @ <sha> (tools/vssim_copy.py); edits: rewrite rules [+ patch <id>]  # noqa: E501` as line 1 of every rewritten file (line 2 after a shebang; the drift tool strips it). For yaml/json the manifest and `vsbench/vs/PROVENANCE.md` carry the list instead (the md5 sidecars of the start files must stay bit-exact).

Patches as built (`tools/vssim_copy_patches.json`: dest file → exact `old → new` string
pairs, each with a one-line reason; cap **six**, three used):
- **P1** `vsbench/tests/fast/test_package.py` (3 pairs): `CORE_ALLOWED = {"numpy", "yaml", "vssim"}` → `{"numpy", "yaml", "vsbench"}`; the fresh-interpreter no-torch check imports `vsbench.vs`; `hasattr(vsbench.vs, name)` (the one bare use of the module object). The test keeps asserting that `import vsbench.vs` pulls no torch in a fresh interpreter.
- **P2** `vsbench/vs/bench/run.py` (2 pairs): `"vssim_version": __import__("vsbench.vs", fromlist=["__version__"]).__version__` (there is no module `vssim`; the record key stays `vssim_version`), and `dirty_paths` runs `_git("-c", "color.status=false", "-c", "status.relativePaths=true", "-c", "status.branch=false", "-c", "status.showStash=false", "status", "--short", strip=False)` instead of `status --porcelain`: porcelain paths are relative to the repository top (`vsbench/x`), `--short` prints them relative to `git -C REPO_ROOT` (= `vsbench/`), so the copied `experiments/` filter works in the nested copy; the four `-c` settings pin that output against a user's git config (revision 3).
- **P3** `vsbench/tests/fast/test_debuglog.py` (1 pair): the throwaway checkout the debug chain is driven in symlinks `("vs", "configs", "benchmark")`, not `("vssim", …)`.
- **Dropped:** revision 2's `vsbench/vs/camera.py` data-root patch (in layout B vs-sim's own `REPO_ROOT = parents[1]` is `vsbench/`) and the `test_protocol` / `test_arm_plugin` `REPO` patches (their `parents[2]` is `vsbench/` too). `git_head()` reports **vsbench's** commit (it runs `git -C vsbench/`); the driver adds `vssim_commit` from the manifest (U2).

Anything else that fails after the rules is a finding to report, not a seventh patch: the
builder stops and asks.

### 2.4 Proving and keeping faithfulness

1. **The copied fast tier is the proof.** `env -u VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT
   pytest vsbench/tests/fast -q`, in its own pytest process, must pass with the same
   pass/skip split the pristine clone gives in the same venv (U0 measures the clone's split
   with `PYTHONPATH=external/vs-sim`; U-copy matches it, skip-for-skip: **789 passed, 13
   skipped** at the pin and OpenCV 4.11.0.86). The tier carries
   the parity fixtures that pin "as published": DINOv3 tokens (`u1_*`, 1e-5 on CPU), the
   612-pair mutual pool and the exact 24 drawn landmarks (`u2_*`), ART-VS tiling and the
   classical arms (`u3_*`), the control law and stop rules against the vendored CNS and ART-VS
   reference files (`u4_*`, `fixtures/reference/`), the record schema (`u7_*`), the 4-way
   search (`u8_*`), every `configs/arms/*.yaml` building with its derived label, the md5 pins
   of every start file, the goal poses of all six scenes.
2. **`tools/vssim_drift.py`** (also `tests/vs/test_drift.py`, the only file in `tests/vs/`,
   run as `pytest tests/vs`): for every manifest entry, `git show <pin>:<src>` → rules →
   patches → compare to the dest minus its provenance line; byte-identical or the test fails
   naming the file and the first differing line. Also checks that no file under the six
   copied trees (`vsbench/{vs,configs,benchmark,assets,tests,scripts}`, `vsbench/experiments`)
   is absent from the manifest (nothing hand-written hides there), that
   `external/VSSIM_COMMIT` equals every manifest entry's commit, ≤ 6 patches each with a
   reason, and that no glob loop of the copied tier can pass on an empty directory.
3. **Pulling a vs-sim fix in deliberately** (documented in `PROVENANCE.md`):
   `git -C external/vs-sim fetch` (reads GitHub or the local checkout), `tools/vssim_drift.py
   --upstream <newsha>` prints, per copied file, whether upstream changed it and the diff;
   the builder re-runs `tools/vssim_copy.py --from <newsha>`, re-runs `pytest tests/vs`, updates
   `VSSIM_COMMIT`, and the commit message names the vs-sim commits pulled. Partial pulls are
   refused by design (one commit pin for the whole copy), so the copy is always *a* vs-sim.
4. **Record provenance**: every record written by the bridge carries `render_mode`,
   `vsbench_commit` and `vssim_commit` (from the manifest), so a number can always be traced
   to the exact published-baseline code that produced it. `vsbench_commit` gets `+dirty` by
   the copied `dirty_paths` rule (tracked changes and untracked, not-ignored files) minus the
   run's own `--out`; the record states that rule (`vsbench_commit_rule`), because the copied
   `run.json` `git_dirty_paths` cannot drop an `--out` outside `vsbench/` and so lists the
   first arm's fresh output as dirty on the second arm's sidecar (revision 3).
5. **Fence in words** (`vsbench/vs/PROVENANCE.md`, repeated in README): "Nothing under
   `vsbench/vs/`, `vsbench/configs/`, `vsbench/benchmark/`, `vsbench/assets/`,
   `vsbench/tests/` or `vsbench/scripts/` is edited by hand. A baseline is run as published;
   a knob that changes a published method is a different arm config with a different label,
   never an edit."

### 2.5 One scored loop, two adapters (how the copy meets vsbench's registry)

**Recommendation: one benchmark loop — the copied `vsbench.vs.bench.runner.run_episode`.**
It is the published protocol (dt 1/60, gain 2.0, cap 600, the stop *recorded* and never acted
on, the divergence abort, three success rules, the frozen record schema, telemetry, films,
debug log, perturbations, GT-blind tripwire). vsbench's Step-3 `vsbench.episode.run_episode`
stops on the true pose (a test-only rule its own brief says Step 7 would replace) and logs a
different format; it is **kept as the test instrument for the Step 1–3 gates** (replay, frame
hashes, video) and marked in the README as "not the benchmark loop". Whether to retire it
later is a repo-structure decision for the owner (D-10).

Two thin adapters, no baseline code in either:
- **`vsbench/world/method_arm.py::MethodArm`** — a `vsbench.methods.VSMethod` flown by the
  benchmark loop. It subclasses the copied `Arm` and overrides `step(rgb, depth, mask=None)`
  to call `method.get_control_rate(rgb, depth)`, returning a `Twist`, a `debug` dict with
  every key `_fly`/`step_telemetry` read (`feature_error_norm` from `info["error_norm"]`,
  `step_ms`, `n_matches`, `depth_values`, `normalised_errors` when `kp_cur/kp_des` are given)
  and a `Matches` built from `kp_cur/kp_des` (so films, telemetry and the `cns_plateau` stop
  work when the method reports points; a method that reports none gets `stop.rule: cap_only`).
  `matcher` is a `MethodFrontEnd(CorrespondenceMatcher)` (`set_episode`, `reset`,
  `record_fields` → `method.params`), `controller` a `MethodController` exposing `dt`, `gain`,
  `reset()`; `kind = "method"`, `matcher_name = method.name`. Its config comes from a
  vsbench-owned template `vsbench/world/configs/external_method.yaml` that states the five
  required fields (`matcher.variant: external` is **not** in the copied enum, so the shim
  validates the config itself and the record's `arm_kind`/label carry `external:`; the copied
  `config.validate` is not edited). This is the direction vs-sim's own external-arm hook
  already anticipates. Gate: vsbench's `sift_ibvs` flown by the benchmark loop on hollywood
  from a `near60` start lands under 5 mm and its record passes the frozen schema test.
- **`vsbench/methods/vssim_arms.py`** — the copied arms (`vitvs_dinov3`, `vitvs_dinov2`,
  `artvs`, `sift_ransac`) registered as `VSMethod`s for the Step-3 demo/video/check tooling
  (`configure` → `Intrinsics`, `set_target` → `arm.reset`, `get_control_rate` → `arm.step` +
  `kp_cur/kp_des/error_norm`). The methods-package import rule (no loop/renderer/scene) holds:
  it imports `vsbench.vs.bench.arms` and `.config` lazily. README states that the demo's
  `exp_se3` integration at dt 0.04 is a different world rule from the benchmark's.

### 2.6 The bench drivers

`python -m vsbench.world.fly` (U2) does what `vs-sim/scripts/fly.py run` does on the MuJoCo
world: `vsbench.vs.bench.config.load(arm yaml)` + overrides, `build_arm(cfg,
intrinsics=scene.intrinsics, scene_spec=...)`, `benchmark.select_starts`, `run_episode(...,
starts_payload=payload)`, one jsonl record per start (+ `render_mode`, `vsbench_commit`,
`vssim_commit`), `summary` via the copied `scoring.landing_stats`; `--method <vsbench name>`
flies a `MethodArm`. `python -m vsbench.world.check_baselines` (U9) is the twin of
`scripts/check_baselines.py`.

---

## 3. Capability inventory — what vs-sim does, and what happens to each

Legend: **now** = in this port's units (§7); **later** = after the first milestone, needs a
decision or a scene verb we add; **n/a** = Isaac-specific or not meaningful in MuJoCo.

| capability | where in the copy (`vsbench/vs/…`) | MuJoCo | why / how |
| --- | --- | --- | --- |
| The `Scene` contract (4 verbs, 4 facts) | `scene.py` | **now** (U1) | `MujocoScene` wraps `vsbench.render.mujoco_backend.MujocoRenderer`; `apply` uses the copied `integrate_optical_velocity` (the benchmark's rule, not vsbench's `exp_se3`). |
| D435i camera (intrinsics, 1920×1080, depth in metres) | `data/configs/camera.yaml`, `camera.py` | **now** (U1) | `CameraConfig().intrinsics()` → `CameraModel`; §4 for the half-pixel convention. |
| Goal frame captured once, served from `render()` at the goal pose; `goal_capture_record()` | (Isaac: `isaac_scene.py:455-560`) | **now** (U1) | MuJoCo is frame-deterministic (bake-off gate), so one capture is exact; record `n_captures: 1`, `render_mode: "mujoco-llvmpipe"` / `"mujoco-egl-gpu"` so no MuJoCo record can be mistaken for an RTX one. |
| `gt_correspondence(goal_px, cur_pose)` (occlusion-tested truth for telemetry) | `labels.py` | **now** (U1) | pure numpy on the cached goal depth + last depth — same code, same call. |
| Three poster scenes (plate, backdrop, goal, frozen starts) | `data/configs/scenes.yaml`, `scenes.py::poster_goal_pose` | **now** (U1) | exact geometry; `check_scene_goal` passes by construction. |
| Three rooms (ground, baked layouts, bbox-derived goal) | `scenes.yaml`, `data/configs/layouts/*.json`, `isaac_scene.py::scatter_items` | **now** (U3, U4) | needs the converted meshes; goal re-derived from *our* bbox and checked against the frozen file (D-4). |
| Frozen starts and start sets (`capped30`, `wide100`, `narrow20`, `near60`, `simple5/10`, `t1near*`), md5 pins | `benchmark.py`, `data/benchmark/starts/` | **now** (copy) | poses are poses. |
| ViT-VS (DINOv3 / published DINOv2 chain), ART-VS (36-tile latching switch), 4-way pre-align, classical SIFT/ORB/AKAZE ± RANSAC | `baselines/{features,match,tiling,prealign,classical,control}.py`, `data/configs/arms/*.yaml` | **now** (U2, U5) on CPU | `matcher.device: cpu` override; the copied fixtures prove the code before any flight. |
| CNS v1, DINOBot, LightGlue, RoMa, ViSP arms | `baselines/{cns,dinobot,lightglue,roma,visp_ibvs}.py` | **copied, not wired** | need `$CNS_ROOT` + torch_geometric / ViSP bindings / more weights; nothing MuJoCo-specific — they fly the day their deps are in the venv. Not in the ask. |
| ART-VS object mode `artvs_roi_sim` (GroundingDINO, LightTrack, LangSAM) | `baselines/objroi.py` | **copied, not wired** | heavy detector weights, `$LIGHTTRACK_ROOT`; needs the instance-suite scenes. Not in the ask. |
| `objmask` arms (goal mask from goal depth vs the `z = 0` ground) | `labels.py::object_mask_from_depth`, `bench/runner.py` | **now** (free with U4) | our ground is also at `z = 0`. |
| The episode loop, telemetry, divergence abort, three success rules, stop rules recorded, records + run sidecar | `bench/{runner,scoring,run}.py` | **now** (copy) | `open_run` refuses a dirty tree: the driver passes `allow_dirty=True` and writes both commits into the sidecar. |
| Films (`FilmWriter` via `on_step`), clean film, debug view, debug log | `bench/{films,debugview,debuglog}.py` | **now** (copy; smoke in U9) | `ffmpeg` binary: `imageio_ffmpeg.get_ffmpeg_exe()` if `shutil.which("ffmpeg")` is empty. |
| Perturbation `noise` (RGB Poisson–Gaussian + brightness/contrast/WB) | `bench/perturb.py::apply_noise` | **now** (copy) | numpy on the live frame; world-agnostic. |
| Perturbation `ground:<name>` | needs `scene.build_poster(...)` | **now** (U8, built) | MuJoCo textures are baked at `MjSpec.compile()`; a re-texture = rebuild the model (cluttered2 ~7 s, measured). `vsbench/world/perturb_world.py`. |
| Perturbation `occluder`, `distractors`, `swap:<id>`, `swap:@<name>`, `combined`, `a+b` | needs `put_box`, `place_extra_objects`, `remove_prim(s)`, `set_prim_visible`, `object_world_bbox`, `object_pose` | **now** (U8, built) | same recompile path (cluttered2 7–9 s); extra objects from the converted cache by the copied placement rules (the 74 YCB distractor scans converted in U8). The copied `perturb.py` is not edited. |
| Perturbation `light` | `set_lighting`, `lighting_snapshot`, `restore_lighting` | **now** (U8, built) | not a model edit (MuJoCo clamps each light at 1, the fitted look drives the sun to ~1): the factor `s * blackbody(T)` scales the radiance inside the scene's tone curve; depth untouched. |
| Scatter + gravity settle to **bake** a new layout | `bake.py` (PhysX) | **n/a** | "a re-settle may never define a goal"; we only *load* layouts. |
| Segmentation annotator | (Isaac) | **later** | MuJoCo's segmentation pass (already used for depth) gives geom ids; only the object-mode arms need it. |
| Goal-capture warm-up (8 RTX captures) | `scenes.yaml::scene` | **n/a** | RTX history artefact; recorded as `n_captures: 1`. |
| Depth noise / holes model | — | **n/a in vs-sim** | none exists; adding one is new science (D-5). |
| Scale jitter, `clear_scene_prims`, Nucleus assets | (Isaac) | **n/a** / later | jitter is a dataset leftover; the 6 Nucleus objects are not on disk (D-6). |
| GPU rendering of the same MuJoCo scene | `mujoco-egl` without `VSBENCH_SOFTWARE_GL` | **later, never during the eval** | bake-off: GPU and CPU renders agree to 44.6 dB; 1140 vs 100 fps at 640×480. |

---

## 4. The depth camera: D435i intrinsics into MuJoCo

**Intrinsics.** `vsbench.vs.camera.CameraConfig().intrinsics()` → `Intrinsics(fx=fy=1406.748,
cx=960, cy=540, 1920×1080)`. The MuJoCo backend already takes pixel intrinsics
(`c.focal_pixel = (fx, fy)`, `c.principal_pixel = (W/2 − cx − 0.5, H/2 − cy − 0.5)`) and
renders off-screen at `offwidth ≥ W`, `offheight ≥ H` (the bake-off rendered 1440×1080 on
llvmpipe).

**The half-pixel convention — decide it, pin it, test it.** Three conventions meet here:

| | ray through pixel column `u` (integer index) | principal point |
| --- | --- | --- |
| Isaac (USD camera, aperture centred on the window) | `((u + 0.5) − 960) / fx` | geometric image centre |
| `vssim.scene.MockPlaneScene` | `(u − 960) / fx` | pixel 960's centre — half a pixel right/down of Isaac |
| `vsbench.CameraModel` (OpenCV: integer index = pixel centre) | `(u − cx) / fx` | whatever `cx` says; centred image = `cx = 959.5` |

Working default (D-3): the MuJoCo camera is built with **`cx = 959.5, cy = 539.5`** (Isaac's
geometry — the benchmark), while `MujocoScene.intrinsics` reports the frozen `D435I`
`(960, 540)` that every baseline, record and start file was made with, exactly as Isaac does.
At 0.43 m half a pixel is 0.15 mm. The gate below makes the choice visible.

**Depth.** MuJoCo's reversed-Z buffer (`mjDEPTH_ZEROFAR`) linearised to camera-frame z is the
same quantity as Isaac's `distance_to_image_plane`. vsbench already reads depth from a second,
non-multisampled render (multisampling moved the sample off the pixel centre by up to 3.9 mm on
a 60° plane). Nothing hit → `0.0`, which the copied `clean_depth` maps to the 0.5 m fallback,
the branch Isaac's background (inf / > 10 m) takes. Clipping: vsbench's gate-tested `near =
0.05, far = 10` (scenes fly at 0.43–1.29 m, wide starts ~0.5 m further). Noise: none (D-5).

**Gates (U1), all CPU, against analytic truth vsbench already has:**
1. *Projection* — 13 unlit marker spheres (`stdscenes.marker_scene`) through the D435i camera
   at 1920×1080: centroids vs `vsbench.reference.project` with `cx = 959.5`: median ≤ 0.15 px,
   max ≤ 0.5 px.
2. *Depth* — the plane gate (`tests/test_depth.py`) at 1920×1080 from the three poster goal
   poses and six `narrow` starts: median |z − analytic| ≤ 1 mm, p99 ≤ 3 mm; report at 0.43 m
   and 1.29 m.
3. *Convention* (revision 3: as built) — render the hollywood plate with the copied
   `MockPlaneScene(texture=hollywood, render_scale=1)` and with `MujocoScene("hollywood")` from
   the goal pose and three `near60` starts. **Gated by ECC** (`cv2.findTransformECC`,
   translation): the MuJoCo picture sits at the predicted (−0.5, −0.5) px from the mock's,
   ± 0.1 px per axis and view; ECC's own bias is **calibrated** on the mock's goal picture
   moved by known exact shifts (−0.5,−0.5), (−0.3,−0.3), (−0.5,0), (−1.5,−1.5): ≤ 0.1 px
   (measured ≤ 0.056 px). `cv2.phaseCorrelate` is **reported, not gated**: an exact half
   pixel is its degenerate case (the same calibration shift of −0.5 reads −0.17), so the
   revision-2 wording "phaseCorrelate returns (0.5, 0.5) within 0.1 px" could not be met by
   any renderer. Two phaseCorrelate **null tests** are gated at 0.1 px: MuJoCo moved back by
   the predicted half pixel (residual ~0), and MuJoCo rendered with the mock's principal point
   (960, 540) (~0). Green-channel gradient correlation ≥ 0.8 (mock-vs-Isaac was 0.81). The
   **primary evidence** for the convention is gate 1's signed mean of the marker centroids
   against the exact sphere-image centroid: (−0.002, +0.003) px with `cx = 959.5`, against
   (−0.50, −0.50) px if the principal point were (960, 540).
4. *Contract* — `isinstance(scene, vsbench.vs.scene.Scene)`; `check_scene_goal` on all three
   posters.

---

## 5. The 323 objects: USD → MuJoCo

**Two input kinds, two readers, one cache format.**

- *YCB (the two cluttered rooms)*: skip the USD (OmniPBR/MDL) and read
  `google_16k/textured.obj` + `texture_map.png` with a 60-line numpy OBJ reader (`v`, `vt`,
  `f a/b` → split vertices per `(v, vt)` pair); no `trimesh`. **Risk to test**: the Isaac USD
  was made from this OBJ by a converter not on the host, so axes may differ; the bbox-goal gate
  catches it; fallback = the usd-core path on the YCB USD mesh (texture `usd/textures/texture_map.png`).
- *The 317 curated objects*: `usd-core` (U0 installs it; verify the cp312 wheel). Open the
  stage, walk `UsdGeom.Mesh` prims, take `points`, `faceVertexCounts/Indices`
  (fan-triangulate n-gons), the `st` primvar (`faceVarying` or `vertex` → one uv per corner),
  apply `ComputeLocalToWorldTransform(0)` **of the asset layer only**, and **ignore**
  `metersPerUnit` / `upAxis`: USD references do not auto-scale or re-orient a referenced layer
  and Isaac did not either, so Isaac saw exactly the layer's composed points. Texture: follow
  `UsdUVTexture.inputs:file`; if unresolved, the single file under `textures/`. UV origin: USD
  and OBJ are bottom-left, vsbench `Mesh.uvs` wants `v = 0` at the top row → `v' = 1 − v`.
- *What is lost*: PBR parameters, MDL materials, normal maps, vertex colours; MuJoCo gets
  diffuse colour + one RGB texture, flat shading. Scale and orientation conventions are **not**
  lost: they are reproduced by re-running the copied rules (`vsbench.vs.assets.target_for`,
  `vsbench.vs.scenes.item_quat/canonical_quat/long_axis_yaw`) on the bbox measured from the
  converted points.

**Cache layout** (outside the repo; D-7):
```
~/genvs_ws/vsbench_assets/
  objects/<instance_id>/mesh.npz        float32 V (N,3) native frame & units, float32 UV (N,2)
                                        top-left origin, int32 F (M,3)
  objects/<instance_id>/texture.png     RGB, longest side capped at 4096
  objects/<instance_id>/meta.json       source path + md5 + mtime, native bbox, counts, texture
                                        source + size, converter + usd-core versions
  weights/{hf,torch}                    U0's copy of the two weight folders (250 MB)
  INDEX.json                            instance_id -> ok / failed:<reason> / nucleus; running total
```
Disk: usdc averages ~20 MB; a compressed float32 `npz` ≈ 0.3× of it, textures 5–10 MB each →
**4–6 GB for all 317**, hard cap 10 GB (the converter refuses to exceed it). U3 converts only
the ~10 objects the six scenes need (< 200 MB); U7 does the rest in the background.

**The CPU-only parity check (no Isaac render needed)** — *the bbox-goal gate (U3)*: place the
converted objects by the frozen layout (`T(pos)·R(quat)·S(s)`), compute the world bbox, call
`scene_view_goal_pose(cam, bbox_min, bbox_max, frame_fill=0.85)` and compare to the frozen
start file's `goal` block with `check_goal_match`'s tolerances (5e-4 m / 0.05° / 1e-3 m):
`industrial1_drill` 0.4422820342 m, `cluttered2` 1.1689734192 m with look_at `(0.0469, 0.1515,
0.2424)` (the upright pitcher's top), `cluttered3` 1.2853190576 m. This tests units, axes, the
scale rule and the layout semantics at once. Isaac's bbox came from USD `ComputeWorldBound`
(possibly authored `extent`s a hair looser than the points): a miss ≤ 2 mm is reported and
D-4 applies. *Silhouette check against Isaac* (later, GPU free): `vs-sim/scripts/render_goal.py`
under `$ISAAC_PY` → `tools/compare_goal.py` overlays the MuJoCo goal render; outline IoU ≥
0.95, centroid ≤ 2 px. The only Isaac-dependent step, explicitly after the eval.

---

## 6. The six benchmark scenes in MuJoCo, and what the comparison means

**Rebuild rules (U1 posters, U4 rooms)** — `vsbench/world/build.py::scene_spec_for(name)`
reads `vsbench.vs.scenes.load_scenes()` and returns a `vsbench.scene.SceneSpec`:
- poster: `Plane(I, (0.8, 0.6), Material(texture=<png as RGB>))` at `z = 0`, backdrop
  `Plane(z = −0.02, (4, 4), grey 0.5)`; texture paths rewritten `/workspace/...` →
  `$VSBENCH_WORKSPACE_ROOT` (default `~/genvs_ws`). Goal from `poster_goal_pose`.
- room: `Plane(I, (2.4, 2.4), ground texture)` (single span), one `Mesh` per layout entry from
  the cache, pose `T(pos)·R(quat)`, vertices pre-scaled by `s`. Goal from the measured bbox
  (§5 gate), then `check_scene_goal`.
- lighting: ambient 0.2 + one directional `Light` whose direction is the USD distant light's
  `−Z` rotated by `Rx(−30°)·Ry(0)·Rz(30°)` (computed, not guessed), `cast_shadow=True`, intensity
  fitted once so the hollywood goal render's mean grey matches the mock's tone fit — stated in
  `meta`, never claimed as photometric parity. **Since U4b this is the `raw` look**; flights
  draw in the `isaac_fit_v1` look by default (§7 U4b: a fitted fill share and light gain, a
  scene-fitted 8192 shadow map, a per-scene tone curve fitted to Isaac's goal pictures).

**Starts** — `benchmark.select_starts(scene, set)` → `Pose.from_start(s)` → `scene.reset`.
Identical poses, identical draw seeds, identical arms.

**Is "same starts, different renderer" meaningful?** Yes, for: a **fast development loop**
(a new controller against the real baselines, loop, records and success rules, on a laptop, in
minutes — convention bugs show up here as in Isaac); **regression of the baseline code** (the
fast tier plus a nightly MuJoCo `capped30 × {vitvs_dinov3, artvs}` on a deterministic renderer,
which RTX cannot be); **teaching and external users**. Not meaningful for replacing an Isaac
number: different pixels → different matches → different landings. Every MuJoCo record carries
`render_mode: mujoco-*`; the bridge's `summary` refuses a mixed `render_mode` input (D-9).

---

## 7. Units of work (ordered; `∥` = can run in parallel with the previous unit)

Every unit runs under the station rules, scripted once in **U0** (`tools/cpu_env.sh`):
`CUDA_VISIBLE_DEVICES=""`, `VSBENCH_SOFTWARE_GL=1`,
`MUJOCO_GL=egl`, `OMP/MKL/OPENBLAS_NUM_THREADS=2`, `LP_NUM_THREADS=2` (Mesa llvmpipe's
rasteriser threads; added in U1 — before that llvmpipe spawned one thread per CPU, §8),
`torch.set_num_threads(2)` and `torch.set_num_interop_threads(2)` (the bridge driver, before
any torch op, so records read 2 / 2),
`HF_HOME=~/genvs_ws/vsbench_assets/weights/hf`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`TORCH_HOME=~/genvs_ws/vsbench_assets/weights/torch`, `VSBENCH_WORKSPACE_ROOT=~/genvs_ws`,
`VSSIM_ASSET_ROOT=~/genvs_ws/genvs-sim`, `VSSIM_YCB_ROOT=~/genvs_ws/assets/ycb`, every command
`nice -n 19` with a `timeout`. Forbidden: the `isaac` container, `~/genvs_ws/night_*.lock`,
`/workspace/vssim_out`, `~/genvs_ws/vssim_out`, any write under `~/runpod_dev/vs-sim`,
`~/runpod_dev/refine-vs*`, `~/runpod_dev/vs-sim-*`, `~/genvs_ws/refine-vs*`,
`~/genvs_ws/cache`, `~/genvs_ws/torch_home`. CPU budgets are estimates at load ~20, to be
replaced by measured numbers in each gate's report.

**Enforced, not only scripted (revision 3):** `python -m vsbench.world.fly` sets
`CUDA_VISIBLE_DEVICES=""` and `VSBENCH_SOFTWARE_GL=1` by force (not `setdefault`), refuses to
run when torch sees a GPU, and after building the scene refuses unless `render_mode ==
"mujoco-llvmpipe"` and the context's GL renderer string names llvmpipe/softpipe/swrast;
`--allow-gpu` (off by default, never during the eval) lifts both. Once software GL is
requested, `vsbench.glenv.force_software_gl` overwrites the EGL vendor / device variables
instead of `setdefault`, and `MujocoRenderer` honours `VSBENCH_SOFTWARE_GL` itself (not only
through `make_renderer`). Every world test that renders skips unless software GL is
requested, so `pytest tests/world` without `cpu_env.sh` cannot open a GPU context. Gate tests
write their committed JSON only with `VSBENCH_WRITE_GATES=1` (else a pytest temp file).

### Milestone 1 — "a faithful copy, the depth camera, the poster, ViT-VS flying one start on CPU"

**U0 — Environment, pin, weights copy** *(done)*
- Gate: `import torch` prints `cuda available: False`; `nvidia-smi` memory unchanged;
  `external/vs-sim` at the pinned commit (`external/VSSIM_COMMIT`), **not** pip-installed;
  the pristine clone's fast tier run once via `PYTHONPATH=external/vs-sim env -u
  VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT py -m pytest external/vs-sim/tests/fast -q` and its
  pass/skip split written to `results/bridge/u0_clone_fast_tier.txt` (this is the number
  U-copy must match). Any red test caused by library versions is reported, never patched.
- As built: under OpenCV 5.0 six AKAZE tests failed (5.0 has no `AKAZE_create`) and nine
  classical-fixture tests skipped (pinned to 4.10 / 4.11), so OpenCV is **pinned to
  `opencv-python-headless==4.11.0.86`**, the Isaac flight interpreter's version
  (`tools/constraints.txt`, `pyproject.toml`; parent decision 2026-10-22). At the pin the
  clone's tier is **789 passed, 13 skipped** of 802 (the 13 need the frozen genvs-sim or
  ART-VS checkouts, DINO v1 or LightGlue); the two asset-root variables must be unset (three
  `test_assets.py` tests assert the defaults).
- Cost: ~10 min; disk ~1.5 GB venv + 0.25 GB weights.

**U-copy — The faithful copy and the drift tool** (`tools/vssim_copy.py`, `tools/vssim_drift.py`, `tools/vssim_copy_patches.json`, `vsbench/vs/**`, `tests/vs/**`, `vsbench/vs/{COPY_MANIFEST.json,PROVENANCE.md}`, `tests/vs/test_drift.py`; `pyproject.toml` package-data **after U0 has finished with that file**)
- Inputs: U0's pin. Source blobs read with `git show`, never the clone's working tree.
- Do: §2.1–2.4 exactly: copy list of §2.2, rules of §2.3, ≤ 6 patches, provenance line,
  manifest; `[tool.setuptools.package-data] "vsbench.vs" = ["data/**/*"]`; the drift tool and
  its test; `PROVENANCE.md` naming what is flown here and what is copied-not-wired.
- Gate (as built, layout B): `env -u VSSIM_ASSET_ROOT -u VSSIM_YCB_ROOT py -m pytest
  vsbench/tests/fast -q` (own process) passes with the **same pass/skip split as U0's clone
  run**, skip-for-skip (789 / 13; `results/bridge/ucopy_fast_tier.txt`);
  `tools/vssim_drift.py` exits 0 and `py -m pytest tests/vs -q` (the drift test) is green;
  copied `test_package.py` proves `import vsbench.vs` pulls no torch; `ruff check --config
  tools/ruff_vssim_copy.toml vsbench/vs vsbench/tests vsbench/scripts` with vs-sim's own
  rules (`E,F,I,UP,B,SIM`, line 100) is clean (the lines the rewrite pushed past 100 columns
  are listed per-file ignores); no file under the copied trees outside the manifest;
  vsbench's own gates (`pytest tests --backends mujoco-egl --ignore=tests/vs
  --ignore=tests/world`) still green.
- Cost: ~5 min of scripting runtime; the tier ≈ 2–4 min at nice 19. Disk ~9 MB.

**U1 — `MujocoScene` for the posters + the D435i gates** (`vsbench/world/{__init__,mujoco_scene,build,camera}.py`, `tests/world/test_scene_posters.py`)
- Inputs: U-copy; `vsbench.vs.{scenes,camera,scene,labels,benchmark}`, `vsbench.render.mujoco_backend`.
- Do: `camera.d435i_camera_model() -> CameraModel` (cx 959.5 / cy 539.5, near 0.05, far 10)
  and `d435i_intrinsics() -> vsbench.vs.camera.Intrinsics`; `build.scene_spec_for(spec)` for
  `kind: poster`; `MujocoScene(name, *, gl="egl", samples=4)` implementing the contract with
  the copied `integrate_optical_velocity`, one-shot goal capture, `goal_capture_record()`,
  `render_mode`, `gt_correspondence` via `vsbench.vs.labels.gt_correspondence` on the cached
  goal depth + last depth, `goal_frame_md5`, `close()`, context manager. Pose ↔ `T_wc` via
  the copied `pose_to_T`.
- Gate: the four gates of §4 as tests writing `results/bridge/u1_camera_gates.json`.
- Cost: a 1920×1080 llvmpipe render + depth pass ≈ 0.15–0.3 s; < 100 frames → ~1 min.

**U2 — The bench driver and the first flight** (`vsbench/world/fly.py`, `tests/world/test_fly_smoke.py`)
- Inputs: U1; `vsbench.vs.bench.{config,arms,runner,run,scoring}`, `vsbench.vs.benchmark`,
  `vsbench/configs/arms/*.yaml`.
- Do: §2.6. `torch.set_num_threads(2)` and `torch.set_num_interop_threads(2)` at start-up;
  refuse to run if `torch.cuda.is_available()` or the scene is not on llvmpipe (§7 station
  rules); only the protocol's start sets (as vs-sim's `fly.py`).
- Gate (**the M1 gate**, revision 3): on hollywood from the first `near60` start
  (`n_000`, 5.58 mm / 2.98° away), 100 steps, real weights on the CPU:
  - **ViT-VS DINOv3: "ran and was recorded"** — the record exists with 100 steps, ≥ 20
    matches on frame 1, passes the copied frozen-schema tests and carries the provenance
    fields — **plus its error-curve statistics, reported**: the median true translation error
    over steps 75–100 and the maximum over steps 50–100, from the per-step telemetry of the
    same start (`results/bridge/m1_telemetry/`, which reproduces the M1 flight's curve). Its
    landing is reported, not gated: at 100 steps ViT-VS is still on its way in (3.78 mm at the
    cap), so revision 2's "< 5 mm, `predicate_a`" read a landing that had not happened.
  - **`sift_ransac` keeps the landing gate**: `final_translation_mm` < the start's offset and
    < 5 mm, `predicate_a`, ≥ 20 matches on frame 1, schema-valid.
  - **The truth seam in a flight**: `sift_ransac` flown again with `--telemetry-extended`:
    every `gt_*` / `err_patch` column finite on every step, valid share ≥ 0.99.
  Records under `results/bridge/m1/` and `results/bridge/m1_telemetry/`; numbers in
  `results/bridge/u2_m1_gate.json`.
- Measured (2026-10-23): `sift_ransac` 5.577 → 0.176 mm / 0.105°; truth seam valid share
  1.000 over 2393 point-steps, every `gt_*` finite, `err_patch` median 0.0008 / max 0.083
  patch. ViT-VS 5.577 → 3.778 mm / 0.518° at the cap; **median 4.61 mm over steps 75–100,
  max 5.70 mm over steps 50–100** (above its start: the error went back out before it came
  in), 3.73 mm at step 100. The telemetry flight (vsbench `04521b1`) reproduced both M1
  landings bit for bit (M1 flown at `409656d`); its curve matches the M1 film's every-second
  step to 0.005 mm.
- Cost: DINOv3-S/16 at 512 on 2 threads ≈ 0.4–0.8 s/step + render ≈ 0.3 s → **100 steps ≈ 2 min**.

### Milestone 2 — the rooms, the objects, ART-VS, parity, one loop

**U3 ∥ U2 — Object converter + the bbox-goal gate** (`vsbench/world/assets.py`, `tools/convert_objects.py`, `tests/world/test_assets.py`)
- Inputs: the USD trees, the YCB OBJs, `vsbench.vs.assets` (library, `target_for`, `resolve_usd_path`).
- Do: the two readers of §5, the cache format, `load_object(id)`, `placed_mesh(entry, item,
  layout_entry) -> vsbench.scene.Mesh` applying the scale rule, `scene_bbox(meshes)`. Convert
  the objects of the six scenes (`handal_drill_25` + 8 YCB ids) plus one gso, one objaverse,
  one housecat object as reader coverage.
- Gate: the bbox-goal gate of §5 on the three rooms at `check_goal_match` tolerances (residuals
  reported in mm even when green); finite vertices, consistent faces, UVs in [0, 1], a texture
  per mesh; a contact sheet `results/bridge/u3_objects.png` (each object rendered once from the
  library's upright canonical pose).
- Cost: minutes; disk < 200 MB.

**U4 — The three rooms in `MujocoScene`** (extends `build.py`; `tests/world/test_scene_rooms.py`)
- Gate: `check_scene_goal` passes on all three rooms (or D-4 applies and the record says so);
  depth gate on the ground plane at the room goal distances; `vitvs_dinov3_objmask` builds and
  the goal mask share is in (0.02, 0.6); a goal render per room under `results/bridge/`.
- Cost: a room compile ≈ 1–3 s (the 4096² ground texture dominates).

**U4b ∥ U5 — Match Isaac's look** (`vsbench/world/look.py`, `vsbench/world/configs/look_isaac_fit_v1.json`, `tools/u4b_look.py`, `tests/world/test_look.py`) *(done, 2026-10-23)*
- Owner decision 2026-10-23: "fit one brightness/tone curve per scene to the Isaac goal
  pictures already on disk, raise the light, and use a finer, softer shadow map". Why: the
  geometry already matched (goal ~1e-11 mm, depth µm, mask IoU ≥ 0.99) but the raw picture was
  far darker and flatter (cluttered2's ground mean grey 32 vs Isaac's 131), its shadows jagged
  (1.8 mm shadow-map texels), and `sift_ransac` on `industrial1_drill` landed 16.4 mm in MuJoCo
  vs 0.17 mm in Isaac from the same near start.
- Built: `MujocoScene(name, look=...)`, default **`isaac_fit_v1`**, **`raw`** = the U1/U4
  picture bit for bit (gated: hollywood's and the drill's goal md5s). The look = (1) a fill
  share 0.15 (was 0.2) chosen on the rooms' dim pixels (shadows, turned-away faces) and a
  per-scene light gain - the largest with ≤ 0.1 % of the goal picture clipped and the sun's
  term ≤ 1 (MuJoCo saturates the diffuse term at 1, measured; Isaac never clips): 1.175 on the
  rooms and poster3, 1.065 hollywood, 1.0 poster2; (2) an 8192 shadow map over a square fitted
  to the scene's geometry seen from the sun (`vis.map.shadowclip`; 0.41 mm per texel in a
  room) - the backend gained `shadowsize` / `shadowclip_m`, and the depth context (never
  shadowed) a 64 px map, so memory grows by ~200 MB, not 456; (3) per scene and channel a
  monotone curve `255·srgb(v/(1+v))`, `v = floor + e^k·x^γ`, fitted by least squares on binned
  medians of Isaac's goal picture against the real look-lit MuJoCo goal, applied as a 256-entry
  table (md5 in the config, with the Isaac source md5s) to the RGB only. Every record carries
  `look`, `look_curve_id`, `look_curve_md5`, `look_record` (through `goal_capture_record`).
  `fly run --look`, `tools/u4_rooms.py smoke --look`. U1's mock-comparison gate draws `raw`.
- Measured (`results/bridge/u4b_look.json`; Isaac = `vssim_out/check1`, read only): mean grey
  raw → look vs Isaac, goal (fitted) / held-out near view: hollywood 91 → 182 vs 184 / 89 →
  180 vs 181; poster2 97 → 190 vs 191 / 97 → 190 vs 191; poster3 84 → 181 vs 182 / 83 → 180 vs
  181; drill 83 → 185 vs 185 / 84 → 185 vs 185; cluttered2 35 → 127 vs 132 / 35 → 127 vs 132;
  cluttered3 126 → 216 vs 217 / 125 → 216 vs 217. Histogram distance (mean per-channel EMD,
  grey levels) 92–102 → 2.3–8.9 on both views; Isaac against its own re-render 0.01–0.2 (the
  floor). The held-out view is only 3–8 mm from the goal (nearly the same content). Depth
  identical in both looks (every scene, goal and near). Cost at load ~24 on 2 llvmpipe threads:
  +0.1–0.2 s per frame (curve 15–30 ms; the rest the larger shadow map — 4096 over the same
  square measured ≈ raw's cost at 0.82 mm per texel, the fallback if frame time matters more),
  +200 MB. Sampling the 0.53° sun with 2 / 4 lights: +0.1 / +0.5–0.7 s per frame for a penumbra
  of 1–2 px, not used (Isaac's edges are 1–2 px sharp too).
- Smoke (recorded, not thresholded; `results/bridge/u4b/`): `sift_ransac`, 100 steps, `n_000`:
  the raw re-fly reproduces U4 to the last digit; in the look `industrial1_drill` 16.39 → 0.56 mm
  (Isaac 0.17 mm at 200 steps), `cluttered2` 0.31 → 1.20 mm (Isaac 0.66), `cluttered3` 0.57 →
  0.51 mm (Isaac 2.19). Context only, never pooled (D-9).
- Still different: specular highlights, transparency and ambient occlusion (MuJoCo shades
  diffuse only); shadows stay lighter than Isaac's on dark ground (Isaac's dark-tone floor is a
  sun-lit reflection that vanishes in shadow, the curve lifts it everywhere: cluttered2 66 vs 44
  grey inside the shadow, drill 61 vs 54, cluttered3 99 vs 98); Isaac's picture is a little
  softer (denoiser); cluttered2's ground keeps a 19-level per-pixel difference. A new scene
  (no Isaac goal picture on disk) has no curve: it flies with `--look raw` until one is fitted.

**U5 — Baseline parity: mock vs MuJoCo, ViT-VS and ART-VS** (`tools/u5_parity.py`, `results/bridge/u5/parity.md`, `results/bridge/u5_parity.json`) *(accepted with findings, 2026-10-23)*
- Do: fly `vitvs_dinov3`, `vitvs_dinov2`, `artvs` (cpu) on hollywood from three `near60` and
  two `narrow20` starts, 150 steps, on the copied `MockPlaneScene(texture=hollywood,
  render_scale=1)` and on `MujocoScene` (in the `raw` look since U4b: the mock shows the
  texture's own tone); same seeds. Compare landing (mm, deg) at step 150,
  the `cns_plateau` step, first-frame match count, twist-norm curve correlation.
- Gate: landings within 3 mm / 0.3° on every start; ART-VS's coarse→tiled switch fires in both
  worlds (`artvs_switched` on the record; the step from the telemetry). A miss is a finding
  about the renderer, written down, never tuned away.
- Cost: ART-VS tiled phase ≈ 1.5 s/step → 150 steps ≈ 4–5 min per start; ≈ 1 h in the
  background with `timeout 2h`.
- As built (2026-10-23): `tools/u5_parity.py fly|compare`, `tests/world/test_parity.py`,
  records `results/bridge/u5/<world>/`, table `results/bridge/u5/parity.{md,json}`, gate
  `results/bridge/u5_parity.json`. Starts `near60` n_000–n_002 and `narrow20` start_00–01, 150
  steps, `sift_ransac` added as the control. Three worlds: `mock` (the copied
  `MockPlaneScene` named `hollywood`, so the landmark draw seeds pair), `mujoco` (raw look,
  Isaac's principal point) and `mujoco960` (the same scene drawn at the mock's (960, 540), U1's
  control renderer; flown for the ViT arms and SIFT, **not ART-VS** — stopped after 8 min to
  spare the shared CPU at ~8 s/step). ART-VS's switch step is read from the arm's
  `TiledMatcher.tiled` latch on `on_step`. Every MuJoCo record: GL `llvmpipe (LLVM 20.1.2)`.
- Measured, mock vs mujoco (the gate): **`sift_ransac` PASS** (landings 0.03–0.58 mm in all
  three worlds, |Δ| ≤ 0.06 mm / 0.02°); **`artvs` landing PASS** (|Δ| ≤ 0.79 mm / 0.08°);
  **`vitvs_dinov2` FAIL on n_001** (1.53 vs 5.33 mm, Δ 3.80 mm / 0.44° — the wobble: the medians
  of steps 101–150 agree, 1.44 vs 3.30 mm and 0.33 vs 0.53°; MuJoCo's curve wanders 2.57–5.33 mm
  and the cap caught its top); **`vitvs_dinov3` FAIL on n_001** (4.28 vs 8.16 mm, Δ 3.89 mm /
  0.65° — not only the wobble: steps 76–150 span 2.87–5.01 mm in the mock and 5.78–8.25 mm in
  MuJoCo, rotation medians 0.21 vs 0.73°); the other four starts of both pass. **ART-VS switch:
  the plan's gate FAILS on n_002 in both worlds** (it never latches: the coarse stage drifts
  out, 5.99 → 8.88 / 9.67 mm, so the error proxy never falls below 0.20×); on the other four
  starts it fires in both (steps 2–17 vs 3–17), so the worlds agree start by start.
- What explains it: the pure half pixel (`mujoco960` vs `mujoco`, same tone and filtering)
  moves the ViT-VS landings as much as the whole mock-vs-MuJoCo difference (DINOv3: median
  2.62 mm, max 4.03 mm; on n_001 it is the half pixel that separates them — `mock` vs
  `mujoco960` is 0.15 mm there; DINOv2: max 1.42 mm), and on n_002 the half pixel alone moves
  DINOv3 by 3.58 mm where tone + half pixel moved it 0.64 mm. So neither difference is a bias
  that explains the misses: ViT-VS's token-quantised floor (one DINOv3 token = 60 × 34 px of the
  1920 × 1080 picture; ±2–4 mm) jumps on sub-pixel picture changes. SIFT (sub-pixel keypoints, RANSAC homography) sees neither
  difference (≤ 0.1 mm); ART-VS's tiled phase (36 tiles of a 1344 square, one token = 20 × 11 px)
  sits in between (≤ 0.8 mm). Nothing tuned: the gate test names the three misses as findings
  (`KNOWN_LANDING_MISSES`, `KNOWN_SWITCH_MISSES`) and fails on any other.
- Gate, re-stated (final review, 2026-10-23; from the committed records, no new flight). Old
  and new side by side; the old verdicts stay in the gate JSON (`original_gate`) and in
  `results/bridge/u5/parity.json` (`verdict.original_gate`):

  | gate | original rule | original verdict | re-stated rule | re-stated verdict |
  | --- | --- | --- | --- | --- |
  | landing | every arm, every start: landings within 3 mm / 0.3° at the cap | SIFT, ART-VS pass; both ViT-VS FAIL on n_001 | SIFT and ART-VS: unchanged; ViT-VS: reported, gated by the curve rule | SIFT, ART-VS pass |
  | ViT-VS curve | (none) | — | per start: curve medians over steps 101–150 within 3 mm / 0.5°, AND their spread (median over the five starts) no larger than between `mujoco960` and `mujoco` (the half-pixel control) | **DINOv2 pass** (spread 1.06 vs 1.19 mm, 0.127 vs 0.130°); **DINOv3 misses on rotation**: n_001 0.526° > 0.5 (translation 2.64 mm, within), rotation spread 0.162 vs 0.125° (translation 1.36 vs 1.82 mm, within) |
  | ART-VS switch | fires on every start in both worlds | FAIL on n_002 (in both) | the two worlds agree on whether it fires and, where both fire, on when (±5 steps) | pass (agree on all five; \|Δ step\| 0–3) |

  Spread = the median over the starts, because the per-start bound already covers the worst
  start; the max over the starts is reported, not gated (DINOv2's translation max 2.22 vs
  1.69 mm and DINOv3's rotation max 0.53 vs 0.37° exceed the control's). **Outcome: accepted
  with findings** — DINOv3's n_001 rotation is the same token-quantised-floor finding as its
  landing miss; `tests/world/test_parity.py` names every miss of either rule
  (`KNOWN_LANDING_MISSES`, `KNOWN_CURVE_MISSES`, `KNOWN_SWITCH_MISSES`) and fails on any other.
- CPU: 26 000 CPU-s (mock 11 300, mujoco 10 400, mujoco960 4 300, + the stopped ART-VS job),
  2 h 43 min wall on two nice-19 processes at load 19–31; ART-VS 7–10 s/step in the tiled phase
  (not 1.5), ViT-VS 0.85–1.8 s/step, SIFT 0.9–1.7 s/step (the mock's 1920×1080 numpy warp
  ≈ 0.7 s a frame). poster2 / poster3 not flown (a gate failed; the owner's call).

**U6a ∥ U5 — `MethodArm`: vsbench methods fly the benchmark loop** (`vsbench/world/method_arm.py`, `vsbench/world/configs/external_method.yaml`, `tests/world/test_method_arm.py`)
- Do: §2.5 first adapter; `fly.py --method sift_ibvs`.
- Gate: `sift_ibvs` (vsbench's own, Step 3) flown by the copied loop on hollywood from a
  `near60` start lands < 5 mm; its record passes the copied schema test (`u7_record_schema`
  key set with `arm_kind: external`); `assert_gt_blind` passes; with `kp_cur/kp_des` reported
  the `cns_plateau` stop is recorded, without them the config falls back to `cap_only` loudly.
- Cost: ~10 min.

**U6b ∥ U6a — The copied arms as vsbench methods** (`vsbench/methods/vssim_arms.py`, `tests/world/test_vssim_arm_methods.py`)
- Do: §2.5 second adapter.
- Gate: vsbench's contract, leak and zero-at-target gates pass for `vitvs_dinov3`, `artvs`,
  `sift_ransac` (`--backends mujoco-egl -k vssim`); the demo's start 1 converges under 3 mm in
  300 steps with `sift_ransac`; the ViT methods pass `python -m vsbench.check`.
- Cost: ~10 min.

### Milestone 3 — the whole library, perturbations, the rig check

**U7 ∥ anything — Convert the full library** (`tools/convert_objects.py --all`)
- Gate: `INDEX.json` ≥ 310 of 317 `ok` (failures named); under the 10 GB cap; a 323-tile
  contact sheet; 12 random objects' scaled longest extents equal `RESCUE_TARGETS_M` ± 1 mm.
- Cost: one process, `nice -n 19`, `OMP_NUM_THREADS=1`, ~1 h; disk 4–6 GB.

**U8 — Perturbations that need world verbs** (`mujoco_scene.py`: `build_poster`, `put_box`, `remove_prim(s)`, `place_extra_objects`, `set_prim_visible`, `object_world_bbox`, `object_pose`, `set_lighting`/`lighting_snapshot`/`restore_lighting`) *(done, 2026-10-23)*
- Do: each verb edits a `SceneSpec` copy and recompiles (camera pose kept); the copied
  `perturb._needs` then admits `ground:`, `occluder`, `distractors`, `light`, `combined`,
  `swap:@name` (needs U7 + instance-suite scenes via `scene_spec_for`).
- Gate: the copied `test_perturb` contract against `MujocoScene` (begin/end leaves the goal
  render md5 unchanged); one episode per mode on `cluttered2` with `sift_ransac` writes its
  draw to the record.
- Cost: recompile 1–3 s per episode; ~20 min.
- Built (`vsbench/world/perturb_world.py`, mixed into `MujocoScene`; `vsbench.world.fly run
  --perturb` / `--perturb-seed`, vs-sim's flags and default seed 20260918;
  `tests/world/test_perturb_world.py`; `tools/u8_perturb.py measure | smoke | summary`; user
  notes in `docs/PORT_NOTES.md`): the world is a table of prims under Isaac's paths
  (`/World/Ground`, `/World/Obj_<i>` in layout order, `/World/Poster`, `/World/Backdrop`, and
  the verbs' `/World/Occluder`, `/World/Distractor_<k>`, `/World/Swap_<k>`); a verb edits the
  table, the next frame recompiles (the old model and GL contexts freed first; several edits
  - a chain, one episode's `end` plus the next `begin` - one compile). A recompile keeps the
  camera, the look's renderer settings and MuJoCo's `stat.extent` / `stat.center` of the
  unedited scene (new backend kwargs `stat_extent` / `stat_center`: the clipping planes and the
  shadow clip are stored relative to the extent); back at the unedited table it compiles as
  at start-up. Extra objects: `vsbench.world.assets.placed_object` with the copied rules on
  the cache; the distractor pool (YCB library minus the scene's own, 74 scans) converted in
  U8 (`tools/convert_objects.py convert`, 74/74 ok, 249 s, cache 2.9 GB). The light: a relight
  of the radiance inside the tone curve (below). Every copied mode is feasible; refused
  loudly: unknown prims, Nucleus assets (D-6), objects missing from the cache, non-square
  tiled posters, light states whose dome and sun factors differ (not a pure rescale).
- The goal frame: as vs-sim - captured once before the first `begin`, served from the cache at
  the goal pose, so a perturbed flight steers to the unperturbed goal picture; a verb that
  edits the world captures the goal first if nobody has (no goal can be drawn perturbed).
- The look under a perturbation (D-11, builder default): **fixed** - the scene's
  `isaac_fit_v1` light, shadow map (its square fitted to the *unedited* scene) and tone curve
  stay as fitted on the unperturbed goal; an edited world is drawn through the same response.
  The `light` mode: Isaac scales dome and sun by the same `s` with the same temperature `T`,
  so every pixel's linear radiance changes by `f_c = s * blackbody_c(T)` (USD's
  `UsdLux.BlackbodyTemperatureAsRgb`); applied as `y = 255 srgb(w/(1+w))`, `w = f_c (floor +
  e^k x^gamma)` (raw: `y = clip(f_c x)`), a 256-entry table; `f = 1` is the scene's table bit
  for bit (gated). Known cost: the curve is constrained only by its own scene's colours - the
  same floors under each other's curve (ground mean RGB at the goal): industrial concrete
  under its own curve (197, 190, 184), under cluttered2's (191, 168, 166), i.e. ~20 levels
  pinker; cluttered2's wood under its own (160, 116, 102), under the drill's (152, 91, 53).
- Gates (`results/bridge/u8_perturb_gates.json`, 14 tests, ~70 s): every verb present; the
  copied contract mode by mode on industrial1_drill (light, occluder, distractors, ground,
  swap, chain order through a recording proxy, combined, the copied loop flying
  `occluder+ground:wood` 3 steps), inst_drill (`swap:@same_small` centred on the goal box,
  size ratio 0.16 / 0.24, the size-matched pose; `@same` a recorded no-op without a compile;
  `@other_small+ground:concrete`) and hollywood (light, occluder; `ground:` and `swap:`
  refused by the copied code); after every mode the cached goal md5 unchanged **and** the goal
  redrawn from the restored world = the goal md5 (picture and depth); the occluder's face at
  `distance - size/2` on its ray (< 1 mm) and its image width within 3 px; the light picture =
  the relit table on the renderer's own output, depth identical.
- Depth under perturbation (gated in the tests, measured on all 17 scene x mode pairs of
  `results/bridge/u8_perturb.json`, stride-4 grid at the goal and the smoke start): every pixel
  whose ray meets the same unedited surface in both worlds (one grid cell of margin around
  edited geometry) has **exactly** the clean depth (max |difference| 0 m, 105 k-130 k pixels
  per view; without the margin 1.000000 of them identical - a single silhouette pixel of a
  drill-room distractor's foot differed by 2.1 mm in one test run, which is why the margin
  exists); every hit pixel against its own ray caster median <= 1e-4 mm, p99 <= 4e-4 mm (the
  occluder, distractors, swapped objects included; max 10.2 mm on one silhouette pixel of
  `@other_small`); the ground against the analytic `z = 0` plane median 4e-5 mm.
- Cost (cluttered2, load ~25, 2 llvmpipe threads; `results/bridge/u8_perturb.json`):
  recompile 6.3–8.8 s per edit (ground 6.3–6.8, occluder 7.6, swap 6.8–8.2, distractors 8.8,
  combined 7.7) plus ~0.6 s to free the old model; the restore 7.1–8.0 s, deferred to the next
  frame drawn, so in a flight it merges with the next episode's `begin`; `light` and `noise`
  0 s. RSS: 3.50 GB after start-up, 3.5–4.3 GB after a perturbed compile, peak 4.85 GB
  (measure) / 5.55 GB (in the smoke flights, `combined`); after each `end` 3.54–3.80 GB (the
  allocator keeps ~0.2–0.3 GB; no steady growth over the 10 modes). inst_drill 0.7–4.1 s
  (other objects' 4096² textures), industrial1_drill 0.6–1.6 s.
- Smoke (recorded, not thresholded; `results/bridge/u8/`, `smoke_summary.json`; `sift_ransac`,
  100 steps, CPU, the draw on every record): cluttered2 from the U4b smoke start (`n_000`,
  7.76 mm): none 1.20 mm (= U4b), light 0.52, occluder 0.17, distractors 0.73,
  `ground:concrete` 4.13, `ground:laminate` 2.98 (divergence abort at step 39), noise 18.41,
  combined 0.44, `swap:ycb/006_mustard_bottle` 1.83, `swap:...+ground:concrete` 8.64
  (divergence abort at step 79); inst_drill (`raw`, `t1near3` `t1n_000`, 6.45 mm): none 0.23,
  `@same` 0.23 (no-op, bit-identical), `@same_small` 139.6 mm from the goal / **3.77 mm from
  the size-matched pose**, `@other_small` 738 mm (0 matches on frame 1: SIFT finds nothing on
  another drill), `@other_small+ground:concrete` 345 mm. Context only, never pooled (D-9).

**U9 — The rig check twin** (`vsbench/world/check_baselines.py`)
- Do: per scene × arm, the arm gets its own goal picture as the live frame (must re-find its
  points, ~zero motion), flies a 3–8 mm start 200 steps, optionally the first N `capped30`;
  `check.json` + `check.md` + goal frames; films via `FilmWriter`.
- Gate: all-green for `{sift_ransac, orb_ransac, vitvs_dinov3, artvs}` × six scenes on the
  zero-motion and near rows; the `capped30` row reported.
- Cost: ≈ 2–3 h at nice 19 → background with `timeout`.
- Built (2026-10-23): `python -m vsbench.world.check_baselines` flies the copied
  `vsbench/scripts/check_baselines.py` as a library (`reference()`: its `identity_check`,
  `episode_check`, `near_start`, `write_table`, thresholds and plugin hook - no third copy) on
  `MujocoScene`, one scene alive at a time, with `vsbench.world.fly`'s protocol helpers, station
  guard and provenance; vs-sim's flags plus `--look`, `--device`, `--samples`, `--frames`;
  `records.jsonl` (each episode's full copied record), `render_mode` on every scene and entry, a
  `bridge` block; `summary` merges per-scene runs, refuses mixed renderers, puts Isaac's own
  check beside each cell. `tests/world/test_check_baselines.py` (fast driver tests on the copied
  mock + the committed flight; gate `results/bridge/u9_check.json`), `tools/u9_diagnose.py`.
- Measured (`results/bridge/u9/README.md`; near start `n_000` 7.76 mm / 2.69°, 200 steps; one
  `capped30` start to the cap): **identity 24 / 24 green**. **Near 15 / 24 closer** under D-12
  (a): SIFT / ORB all six (0.009–2.40 mm); ViT-VS hollywood only; ART-VS poster3, cluttered3.
  Of the nine misses, eight miss in Isaac's own `check1` from the same start (ViT-VS poster2,
  poster3, drill, cluttered2, cluttered3; ART-VS poster2, drill, cluttered2; similar
  magnitudes); hollywood / ART-VS (Isaac 1.6 mm, here 20.1) misses also without the look and on
  the copied mock - ART-VS never leaves its coarse phase from that start outside Isaac (poster3
  the reverse). So the near gate as written is **not all green, and every miss is the method,
  not the world** (`near_fail_diagnosis`). Cost 11.2 CPU-h, 3 h wall on two processes.
- Gate, re-stated (final review, 2026-10-23). Old: "all-green on the zero-motion and near
  rows". New: the identity row all green (24 / 24, pass) and, for the near row, `final <
  start` stays the **recorded verdict** (`near_ok`, with `final_over_start` reported per cell,
  no threshold; D-12 (a)) while the **port gate** is "every near miss is shared with Isaac's
  check from the same start or diagnosed outside the world": **9 / 9, pass**
  (`results/bridge/u9_check.json`: `port_gate`, `port_gate_explained`, `port_gate_pass`;
  `near_all_green` stays `false`, as recorded).
- Still open (after the Isaac evaluation): re-fly the hollywood / ART-VS near cell with
  `--allow-gpu` (CUDA DINOv2, as Isaac ran it) to close the one untested suspect, CPU vs CUDA
  tokens; never while the evaluation runs.

**U10 — Docs** (`README.md`, `docs/PORT_NOTES.md`, `vsbench/vs/PROVENANCE.md`, the
`results/bridge/*.md` reports) *(done, 2026-10-23)*: what it is, what it is not (§6), the
environment, the cache, the pin and the drift procedure, the owner's decisions. The developer
notes are `docs/PORT_NOTES.md` (the plan once named a `docs/BRIDGE.md`; it was never made); the
README is the short user-facing page. The owner-facing HTML view is the parent agent's call
(workspace rule 7).

**Standing procedure (not a unit) — pulling a vs-sim fix**: §2.4 item 3; always a whole-pin
move, always `pytest tests/vs` green, always named in the commit message.

---

## 8. Decisions and risks

Decided by the owner (2026-10-22):
- **D-1 COPY.** vs_sim_light stands alone; `vsbench/vs/` is a faithful, drift-checked copy of
  vs-sim's baselines, bench, scenes and starts (§2.1–2.4). `external/vs-sim` is kept only as
  the pinned source and drift reference, never imported or pip-installed.
- **D-2 SEPARATE FOR GOOD.** The MuJoCo world lives in `vsbench/world/` and never moves into
  vs-sim; no vs-sim change is requested, now or later.

Open — the recommendation is the working default until the owner rules:
- **D-3 Half-pixel convention.** Isaac's `cx = 959.5` (default) vs the mock's 960;
  `scene.intrinsics` reports (960, 540) either way. Cost: none; 0.15 mm at the poster distance.
  *Measured (U1, `results/bridge/u1_camera_gates.json`):* with the default the 78 marker
  centroids sit (−0.002, +0.003) px from the exact sphere-image centroids (median 0.016 px,
  max 0.040 px); with (960, 540) they would sit (−0.50, −0.50) px. Against the mock, ECC reads
  the MuJoCo picture at (−0.45 … −0.51, −0.53 … −0.56) px over four views (ECC's calibrated
  bias ≤ 0.056 px), the phaseCorrelate residual after the predicted half pixel is ≤ 0.062 px
  and MuJoCo drawn at (960, 540) matches the mock to ≤ 0.070 px. So the default is exactly
  Isaac's geometry and exactly half a pixel from the mock's; the choice remains the owner's.
- **D-4 When our room bbox misses the frozen goal tolerance.** (a) the frozen file's goal wins,
  residual recorded (default); (b) our measured goal wins (breaks start pairing); (c) refuse.
- **D-5 Depth noise model.** vs-sim has none; adding one is a science decision. Default: not now.
- **D-6 The six Nucleus-only objects.** Default: skip, marked `nucleus` in `INDEX.json`.
- **D-7 Disk and cache location.** `~/genvs_ws/vsbench_assets/`, 4–6 GB + 10 GB cap. Default: yes.
- **D-8 CPU-only torch in the existing venv** (~750 MB; cannot touch the GPU). Default: yes.
- **D-9 Record labelling.** Every MuJoCo record carries `render_mode: mujoco-*`,
  `vsbench_commit`, `vssim_commit`; MuJoCo and Isaac numbers are never pooled. Default: yes.
- **D-11 (new, U8) The look under a perturbation.** (a) **fixed** (default, built): the
  scene's fitted light, shadow map and tone curve stay as fitted on the unperturbed goal, so a
  perturbation changes the picture as it would through Isaac's fixed renderer; cost: the curve
  extrapolates outside its scene's colours (a re-textured floor ~20 grey levels off its own
  scene's curve, measured). (b) re-fit per perturbation: impossible without an Isaac picture
  of the perturbed world. (c) `raw` for perturbed flights: loses the U4b match everywhere.
- **D-12 (new, U9) The near verdict of the rig check.** vs-sim's script computes none (its
  docstring: "it must land closer than it started, by a lot"). (a) **final < start** (default,
  built; the ratio reported): 15 / 24 on MuJoCo, the eight Isaac-shared misses included;
  (b) a factor for "by a lot" (e.g. final < start / 2): stricter, a number the owner would set;
  (c) no verdict, report only (exactly vs-sim's script). Under (a) and (b) the U9 gate's "near
  all green" cannot hold for ViT-VS / ART-VS, because it does not hold in Isaac either.
  Final review (2026-10-23): (a) stays the recorded verdict, `final_over_start` reported with no
  threshold; the port is gated instead on "every near miss is shared with Isaac's check from
  the same start or diagnosed outside the world" (9 / 9). The hollywood / ART-VS cell should be
  re-flown on the GPU (`--allow-gpu`) after the Isaac evaluation, to close the CPU-vs-CUDA
  suspect.
- **D-10 (new) The Step-3 loop.** Keep `vsbench.episode.run_episode` as the Step 1–3 gate
  instrument beside the one benchmark loop (default), or retire it once `MethodArm` (U6a) flies
  every vsbench method through the benchmark loop. Cost of keeping: two loops in one repo, one
  clearly labelled "not the benchmark"; cost of retiring: rewrite the replay/hash/video gates.

Risks (technical, with the mitigation in the plan):
- *Shared CPU*: load ~22 on 16 threads; everything `nice -n 19`, ≤ 2 threads; wall times
  1.5–3× the estimates; long units in the background with `timeout`.
- *The pre-U1 llvmpipe thread gap* (found in U1, revision 3): until U1 `tools/cpu_env.sh` did
  not set `LP_NUM_THREADS`, so every MuJoCo render on llvmpipe before it (e.g. vsbench's own
  gates in U-copy) ran Mesa's rasteriser with one thread per CPU (16) — at nice 19, but above
  the ≤ 2-thread rule. Mitigation: `cpu_env.sh` exports `LP_NUM_THREADS=2`
  since U1 and the bridge driver sets it too (`setdefault`) before any GL load; a shell that
  renders without `cpu_env.sh` still has the gap (vsbench's Step 1–3 tools do not set it).
- *A renderer path that bypassed software GL* (found in the M1 review pass, revision 3):
  `VSBENCH_SOFTWARE_GL` was honoured only by `make_renderer`; `tools/convert_objects.py
  sheet` builds `MujocoRenderer` directly, and without `MUJOCO_EGL_DEVICE_ID` MuJoCo's EGL
  takes the first device that initialises — device 0, the GPU (the Mesa software device is
  index 1 on the station). U3's contact sheet (`results/bridge/u3_objects.png`, 12 small
  tiles) was therefore most likely drawn on the GPU; the record does not say. Fixed on `port/vssim`:
  `MujocoRenderer` now forces software GL itself when it is requested. Any branch cut before
  the fix (e.g. `port/u7-library`) still has the bypass until it merges `port/vssim`.
- *The third GPU route: the gate harness's backend selection* (found 2026-10-23, fixed at the
  start of U4b): `pytest tests/world` without `--backends` parametrised vsbench's gates
  (`tests/conftest.py::pytest_generate_tests`, e.g. `test_vssim_arm_methods.py`) over every
  installed variant, and Open3D 0.20's Filament engine — `open3d-cpu` included — opens a
  Vulkan context at start-up that selected the RTX 4090 ("Selected physical device 'NVIDIA
  GeForce RTX 4090'", brief, nothing left on the GPU). Fixed: under `VSBENCH_SOFTWARE_GL=1`
  the default set is `vsbench.render.base.SOFTWARE_VARIANTS` only (`mujoco-egl`,
  `mujoco-osmesa`, `pybullet-tiny`, `open3d-cpu`); naming any other variant is a usage error,
  and `make_renderer` / `probe_backend` refuse it too (so `tools/run_gates.py`'s
  all-backends default and `vsbench.smoke` are covered), unless `VSBENCH_ALLOW_GPU=1`;
  `force_software_gl` (and `tools/cpu_env.sh`) pin the Vulkan loader to Mesa's lavapipe
  (`VK_DRIVER_FILES` / `VK_ICD_FILENAMES`; a non-existent file when lavapipe is missing), so
  `open3d-cpu` now logs "Selected physical device 'llvmpipe'" and maps no NVIDIA library.
  Test: `tests/test_gpu_guard.py` (no rendering; a child process enumerates Vulkan devices
  and finds only a CPU one).
- *The last GPU route: software GL was opt-in* (final review, 2026-10-23): every guard above
  keyed on `VSBENCH_SOFTWARE_GL=1`, so a shell that had not sourced `tools/cpu_env.sh` and
  built a world or a renderer directly (`MujocoScene`, `make_renderer`, the Step 1–3 entry
  points) got glvnd's first EGL device, the GPU. Fixed: **software GL is the default**
  (`vsbench.glenv.software_gl_requested` is true unless `VSBENCH_ALLOW_GPU=1`;
  `VSBENCH_SOFTWARE_GL=0` alone no longer opens the GPU); `MujocoScene.__init__`,
  `MujocoRenderer`, `make_renderer`, `vsbench.{demo,check,smoke,render_once,replay}`,
  `tools/{run_gates,bench_*,quality_sheet,lighting_check,fly_axes}.py` and
  `tools/convert_objects.py sheet` call `ensure_software_gl()` first, and the GPU-capable
  variants (`mujoco-glfw`, `pybullet-egl`, `open3d-gpu`) are refused without the opt-in.
  `tools/run_gates.py` defaults to the variants this process may open. Test:
  `tests/test_gpu_guard.py::test_the_default_is_software_gl_in_a_clean_process` (a child with
  none of the switches set builds `MujocoScene("poster2")` and a `mujoco-egl` renderer on
  llvmpipe; it checks the decisions before any context exists, and the parent lays a Mesa
  safety net in case the guard were broken). The Step 1–3 chains
  (`tools/step_chains/`) now say `VSBENCH_ALLOW_GPU=1` on each GPU row.
- *Faithfulness erosion*: the only defence is mechanical — the manifest, the ≤ 6 patch list,
  the drift test in the fast tier, and the fence "nothing hand-written under `vsbench/vs/`".
  A seventh patch is a question for the owner, not a commit.
- *Library versions*: cv2 5.0 / numpy 2.5 vs vs-sim's last green (cv2 4.x / numpy 1.26–2.x);
  U0's clone run is the baseline; a red test is a vs-sim finding reported upstream, never
  patched in the copy. As built: OpenCV pinned to 4.11.0.86 (U0), numpy 2.5 green.
- *llvmpipe memory*: 1920×1080 ×2 contexts + a 4096² texture ≈ 0.3–0.6 GB RSS per scene
  process; never more than two bridge processes at once.
- *ART-VS on CPU* (~1.5 s/step): gates use 100–200 steps and near/narrow starts; a 600-step
  `capped30` campaign belongs on the GPU after the eval.
- *`usd-core` cp312 wheel*: verified in U0; fallback a 3.11 `uv venv` just for the converter
  (the cache format is interpreter-agnostic).
- *YCB OBJ vs Isaac's YCB USD axes*: caught by the bbox-goal gate; fallback usd-core on the USD mesh.
- *Fixture tolerance on CPU*: DINOv3 tokens 1e-5, last digit follows the thread count; run the
  tier with the same `OMP_NUM_THREADS=2` every time; report, do not loosen.
- *HF lock files*: `from_pretrained(local_files_only=True)` may create `.locks/`; hence the
  weights copy in `vsbench_assets/weights`, never the shared cache.

---

## 9. In plain words

**What the owner asked.** We have a heavy simulator (Isaac Sim: needs the big graphics card, a
60 GB install, and it is busy for days with the running evaluation) and a light one (MuJoCo:
runs on a laptop CPU). Can the light one do the same things — the same test scenes, the same
camera, the same objects, the same competitor methods ViT-VS and ART-VS — without disturbing
the evaluation? Yes.

**The owner's two rulings.** The light simulator must **stand on its own**: the competitor
methods, the scoring loop, the scene descriptions and the frozen starting positions are
**copied** into it (not borrowed from vs-sim at run time), and the MuJoCo world stays in the
light simulator for good. The risk of a copy is that it quietly drifts from the original and
the "run as published" promise dies. So the copy is made by a script (never by hand), every
copied file carries a line saying which vs-sim version it came from, the original's own test
suite and comparison files (the "fixtures" that pin the published behaviour down to the exact
24 matched points) come along and must pass, and a **drift check** runs with the tests: it
re-derives every copied file from the pinned vs-sim version and refuses any difference. When
vs-sim fixes something, we move the pin and re-copy on purpose, in one commit that names the fix.

**How the pieces meet.** vs-sim was built so that a "world" is four actions (place the camera,
take a picture, move by a small velocity, tell me the facts). We write one new MuJoCo world
that answers those four actions; the copied competitor methods and scoring loop run on it
unchanged. The scoring loop is the one benchmark loop; the light simulator's own simple
test-loop from Step 3 stays only as a test instrument. Two small adapters let a student's method
fly the benchmark loop, and let the copied competitors appear in the student demo.

**The camera and depth.** The real camera is an Intel RealSense D435i; its numbers (1920×1080,
42° vertical view, focal length 1406.75 px) come from the copied camera file. MuJoCo gives a true
distance per pixel from its drawing buffer, the same kind of depth Isaac gives. One tiny
subtlety — whether "the centre of the picture" is the middle of a pixel or the edge between two
(half a pixel, 0.15 mm) — is decided (we follow Isaac), written down and tested.

**The objects.** The 323 objects exist on this machine as Isaac-format files with a picture
texture each; the eight YCB kitchen objects also exist as plain meshes. A converter reads them on
the CPU, writes a compact cache (about 5 GB), and we re-apply the copied sizing and orientation
rules. The proof needs no Isaac render: Isaac's goal camera for each room was placed from the
measured box around the objects and that number is frozen in the start files; if our converted
objects give the same box to within half a millimetre, the meshes, units, axes and sizes are right.

**What a MuJoCo result means.** The pictures differ from Isaac's (simpler lighting, no fancy
materials), so a MuJoCo landing error is **not** an Isaac landing error and is never put in the
same table. Since U4b (owner decision 2026-10-23) the MuJoCo pictures are made to *look* like
Isaac's: a little more light, shadows drawn with four times finer detail, and for each scene a
brightness curve copied from Isaac's own goal picture. The average brightness now lands within
1-5 grey levels of Isaac's on all six scenes (it was 90-100 levels darker), also on a second
view the curve never saw. What a curve cannot copy stays different: shiny highlights, see-through
parts, soft corner darkening, and shadows on the dark wooden floor that are still too light.
The old picture stays available ("raw") for the tests that compare against the copied mock. It is for trying ideas fast against the real competitors and the real scoring,
catching bugs, teaching, and checking that the competitor code still behaves the same after a
change.

**Disturbances (U8).** vs-sim can change the world after the goal picture is taken - dim
or tint the light, put a grey box in front of the camera, scatter extra objects, lay another
floor, swap the object for another one - so a method must reach a goal whose picture no
longer matches exactly. All of these now work in the light simulator too, with vs-sim's own
code deciding what changes and where. MuJoCo has to rebuild its scene for each change (about
7-9 seconds for the biggest room); everything the change does not touch stays exactly as it
was, down to the depth of every pixel, and the scene comes back bit for bit afterwards. The
"Isaac look" of each scene stays as it was fitted on the clean scene: a new floor is seen
through the same "camera" - which makes a floor whose colours the scene never had come out a
little off (about 20 grey levels).

**Do the competitors behave the same in the simple copied world and in MuJoCo? (U5)** We flew
each competitor from the same five starting positions in both. The classical SIFT method lands
in the same place to a tenth of a millimetre, ART-VS to under a millimetre. ViT-VS lands within
3 mm on four starts of five but not on the fifth (4–5 mm apart). The test that tells why: the
same MuJoCo picture moved by half a pixel (nothing else changed) moves ViT-VS's landing by up
to 4 mm on its own. So ViT-VS does not see a "different world" - it reacts strongly to any tiny
change of the picture, because it matches whole squares of the picture (each about 60 pixels wide)
and its resting point jitters by a few millimetres. ART-VS's switch to its fine mode happened at the same
moment in both worlds, except on one start where it never happened in either world (the
camera drifted away first). The gate was not loosened silently: at the final review it was
re-stated for ViT-VS (compare where the curve settles, not the single last step, and allow
the same jitter the half-pixel test shows) and for ART-VS's switch (the two worlds agree on
whether and when it happens). DINOv2 then passes; DINOv3 still misses on one start by a
fraction of a degree. The old verdicts stay written beside the new ones; U5 is "accepted
with findings".

**How the evaluation stays safe.** Everything runs with the graphics card hidden, MuJoCo drawing
on the CPU, PyTorch's CPU-only build, lowest CPU priority and at most two threads; nothing opens
the Isaac container, the lock files, the result folders or the vs-sim checkout for writing; no
model weights are downloaded. The first milestone is small: the faithful copy with its tests
green, the camera, the three poster scenes, and ViT-VS flying one 5-mm start on the CPU in
about two minutes.

**Still open for the owner** (§8): the pixel-centre convention (D-3), "the frozen goal wins"
(D-4), no depth noise for now (D-5), skipping the six objects that live only on NVIDIA's server
(D-6), the 10 GB disk budget (D-7), CPU-only PyTorch in the existing environment (D-8), the
never-pool rule (D-9), whether the old Step-3 loop stays as a test instrument (D-10), and
whether a perturbed world keeps its scene's fitted look (D-11; built that way: an occluder,
another floor or another object is seen through the same "camera" as the clean scene).
