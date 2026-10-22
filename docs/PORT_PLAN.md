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

### 2.1 Where the copy lives: `vsbench/vs/`, mirroring `vssim/` file for file

```
vsbench/vs/                      ← the copy; NOTHING hand-written lives here (drift-clean)
  __init__.py scene.py camera.py scenes.py assets.py labels.py starts.py benchmark.py demo.py
  isaac_scene.py                 ← copied verbatim as the REFERENCE world (importable without
                                   Isaac; `IsaacScene(...)` raises ImportError here) — needed
                                   for `scatter_items` and the Isaac-free maths tests
  bake.py                        ← copied, reference only (PhysX settle; never run here)
  baselines/  (all 14 modules)   ← ViT-VS, ART-VS, prealign, classical, control = FLOWN HERE;
                                   cns, dinobot, objroi, lightglue, roma, visp_ibvs = copied,
                                   NOT WIRED (deps absent; tests skip with the reason)
  bench/      (all 12 modules)   ← the loop, arms, config, scoring, stats, perturb, films,
                                   debuglog, debugview, truth, io, run
  data/                          ← the one data root (REPO_ROOT, see 2.3)
    configs/{camera.yaml, scenes.yaml, protocol.yaml, arms/*.yaml, layouts/*.json}
    benchmark/{README.md, starts/*.json + *.md5}
    assets/asset_library_v2.json
  COPY_MANIFEST.json             ← dest -> {src path, src blob sha, vs-sim commit, rewrite
                                   rules applied, per-file patch id}
  PROVENANCE.md                  ← what was copied from which commit, what was left behind
                                   and why, what is wired and what is not
tests/vs/                        ← the copied fast tier
  conftest.py  fast/test_*.py  fixtures/  (vs-sim's tests/{conftest.py,fast,fixtures})
vsbench/world/                   ← NEW code (hand-written): the MuJoCo world and the glue
  mujoco_scene.py build.py camera.py assets.py fly.py method_arm.py check_baselines.py
vsbench/methods/vssim_arms.py    ← NEW: the copied arms as vsbench methods (demo/video side)
tools/vssim_copy.py tools/vssim_drift.py tools/vssim_copy_patches.json tools/convert_objects.py
external/vs-sim  external/VSSIM_COMMIT   (U0; git-ignored clone + committed pin — the SOURCE
                                          of the copy and the drift reference, never imported)
```

Why a subpackage and not flattening into `vsbench/`: `vssim` has its own `scene.py`,
`camera.py`, `scenes.py` and `assets.py`, which collide with vsbench's; a subpackage keeps the
copied tree a one-to-one mirror of `vssim/` (so the drift diff is a path map), keeps vsbench's
Step 1–3 code untouched, and keeps the fence "nothing hand-written under `vsbench/vs/`"
checkable by one glob.

### 2.2 What is copied, what is left behind

| copied (verbatim modulo §2.3) | left behind | why |
| --- | --- | --- |
| `vssim/*.py` (11 modules incl. `isaac_scene.py`, `bake.py`), `vssim/baselines/*`, `vssim/bench/*` | `scripts/` (`fly.py`, `check_baselines.py`, `build_starts.py`, `bake_layout.py`, `render_goal.py`, `debug_view.py`, `dev_compare.py`, `smoke_arms_plane.py`, `chains/`) | drivers that hard-wire `IsaacScene` and the station; the bridge writes its own two drivers (U2, U9) |
| `configs/` whole, `benchmark/` whole, `assets/asset_library_v2.json` | `examples/`, `experiments/`, `docs/`, `DESIGN.md`, `README.md`, `CLAUDE.md`, `pyproject.toml` | repo-level material; the owner's one-home rule says link, don't copy docs |
| `tests/conftest.py`, `tests/fast/*` (38 files), `tests/fixtures/*` (4.4 MB, incl. `reference/`) | `tests/slow/*` | station/Isaac-only checks |

Instance-suite scenes (`inst_*`) and their starts come along inside `scenes.yaml` and
`benchmark/starts/` for free; flying them needs the converted library (U7).

### 2.3 The rewrite: mechanical rules + a short, reviewed patch list

`tools/vssim_copy.py --from <sha>` reads every source blob with `git -C external/vs-sim show
<sha>:<path>` (never the working tree), applies the rules below, prepends one provenance line,
writes the file and the manifest entry. The **same function** is what `tools/vssim_drift.py`
applies to the source before diffing, so the two can never disagree.

Rules (regexes, in this order, on `.py` files; yaml/json/md/npz are copied byte-identical):
1. `\bfrom vssim import\b` → `from vsbench.vs import`
2. `\bfrom vssim\.` → `from vsbench.vs.`
3. `\bimport vssim\.` → `import vsbench.vs.`  and bare `\bimport vssim\b` → `from vsbench import vs as vssim`
4. `\bvssim\.(?=[A-Za-z_])` → `vsbench.vs.` everywhere else — docstring references, dotted
   module strings (`importorskip("vssim.scenes")`, `__import__("vssim.baselines.match")`,
   `match="vssim.baselines.cns..."`) — so that error messages and tests keep naming modules
   that exist. Hyphenated / underscored identifiers (`vssim-config-1`, `vssim_out`) do **not**
   match and stay as they are (they are schema names and record data).
5. The provenance line: `# vs-sim copy: <src path> @ <sha> (tools/vssim_copy.py); edits: rewrite rules [+ patch <id>]` as line 1 of every `.py` (before the docstring; the drift tool strips it). For yaml/json a sibling `PROVENANCE.md` carries the list instead (the md5 sidecars of the start files must stay bit-exact).

Patches (`tools/vssim_copy_patches.json`: dest file → exact `old → new` string pairs, each
with a one-line reason; expected to be **six or fewer**):
- `vsbench/vs/camera.py`: `REPO_ROOT = Path(__file__).resolve().parents[1]` → `REPO_ROOT = Path(__file__).resolve().parent / "data"` (the one data root; every other module derives from it, so this is the only data-path edit).
- `tests/vs/fast/test_package.py`: `CORE_ALLOWED = {"numpy", "yaml", "vssim"}` → `{"numpy", "yaml", "vsbench"}`; the Isaac-only set and the `startswith` checks follow rule 4. The test keeps asserting that `import vsbench.vs` pulls no torch in a fresh interpreter — which also pins that `import vsbench` itself stays numpy-only.
- `tests/vs/fast/{test_protocol,test_debuglog,test_arm_plugin}.py`: `REPO = Path(__file__).resolve().parents[2]` → a `DATA = Path(vsbench.vs.__file__).parent / "data"` for `configs/…` reads and `TESTS = Path(__file__).resolve().parents[1]` for `tests/fast/…` loads; `test_debuglog.py:1069`'s tuple `("vssim", "configs", "benchmark")` → the matching paths. The builder lists every `REPO /` use in those three files in the patch reasons.
- `vsbench/vs/bench/run.py`: no patch — `git_head()` now reports **vsbench's** commit (it runs git in the data dir, which is inside this repository), and the driver adds `vssim_commit` from the manifest (U2).

Anything else that fails after the rules is a finding to report, not a seventh patch: the
builder stops and asks.

### 2.4 Proving and keeping faithfulness

1. **The copied fast tier is the proof.** `pytest tests/vs -q` must pass with the same
   pass/skip split the pristine clone gives in the same venv (U0 measures the clone's split
   with `PYTHONPATH=external/vs-sim`; U-copy must match it, skip-for-skip). The tier carries
   the parity fixtures that pin "as published": DINOv3 tokens (`u1_*`, 1e-5 on CPU), the
   612-pair mutual pool and the exact 24 drawn landmarks (`u2_*`), ART-VS tiling and the
   classical arms (`u3_*`), the control law and stop rules against the vendored CNS and ART-VS
   reference files (`u4_*`, `fixtures/reference/`), the record schema (`u7_*`), the 4-way
   search (`u8_*`), every `configs/arms/*.yaml` building with its derived label, the md5 pins
   of every start file, the goal poses of all six scenes.
2. **`tools/vssim_drift.py`** (also `tests/vs/test_drift.py`, in the fast tier): for every
   manifest entry, `git show <pin>:<src>` → rules → patches → compare to the dest minus its
   provenance line; byte-identical or the test fails naming the file and the first differing
   line. Also checks that no file under `vsbench/vs/` or `tests/vs/` is absent from the
   manifest (nothing hand-written hides there), and that `external/VSSIM_COMMIT` equals every
   manifest entry's commit.
3. **Pulling a vs-sim fix in deliberately** (documented in `PROVENANCE.md`):
   `git -C external/vs-sim fetch` (reads GitHub or the local checkout), `tools/vssim_drift.py
   --upstream <newsha>` prints, per copied file, whether upstream changed it and the diff;
   the builder re-runs `tools/vssim_copy.py --from <newsha>`, re-runs `pytest tests/vs`, updates
   `VSSIM_COMMIT`, and the commit message names the vs-sim commits pulled. Partial pulls are
   refused by design (one commit pin for the whole copy), so the copy is always *a* vs-sim.
4. **Record provenance**: every record written by the bridge carries `render_mode`,
   `vsbench_commit` and `vssim_commit` (from the manifest), so a number can always be traced
   to the exact published-baseline code that produced it.
5. **Fence in words** (`vsbench/vs/PROVENANCE.md`, repeated in README): "Nothing under
   `vsbench/vs/` is edited by hand. A baseline is run as published; a knob that changes a
   published method is a different arm config with a different label, never an edit."

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
| Perturbation `ground:<name>` | needs `scene.build_poster(...)` | **later** (U8) | MuJoCo textures are baked at `MjSpec.compile()`; a re-texture = rebuild the model (~1–3 s). A scene verb that recompiles; fine per episode. |
| Perturbation `occluder`, `distractors`, `swap:` | needs `put_box`, `place_extra_objects`, `remove_prim(s)`, `set_prim_visible`, `object_world_bbox`, `object_pose` | **later** (U8) | same recompile path; `swap` also needs the converted library (U7). All scene verbs of the new world; the copied `perturb.py` is not edited. |
| Perturbation `light` | `set_lighting`, `lighting_snapshot`, `restore_lighting` | **later** (U8) | MuJoCo `model.light_diffuse/ambient` live; the Isaac-to-MuJoCo intensity mapping is a tone fit — stated on the record. |
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
3. *Convention* — render the hollywood plate with the copied `MockPlaneScene(texture=hollywood,
   render_scale=1)` and with `MujocoScene("hollywood")` from the goal pose and three `near60`
   starts; `cv2.phaseCorrelate` must return the predicted (0.5, 0.5) px shift within 0.1 px at
   every pose; green-channel gradient correlation ≥ 0.8 (mock-vs-Isaac was 0.81).
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
  `meta`, never claimed as photometric parity.

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

Every unit runs under the station rules, scripted once in **U0** (`tools/cpu_env.sh`, being
written now; its content is U0's): `CUDA_VISIBLE_DEVICES=""`, `VSBENCH_SOFTWARE_GL=1`,
`MUJOCO_GL=egl`, `OMP/MKL/OPENBLAS_NUM_THREADS=2`, `torch.set_num_threads(2)`,
`HF_HOME=~/genvs_ws/vsbench_assets/weights/hf`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`TORCH_HOME=~/genvs_ws/vsbench_assets/weights/torch`, `VSBENCH_WORKSPACE_ROOT=~/genvs_ws`,
`VSSIM_ASSET_ROOT=~/genvs_ws/genvs-sim`, `VSSIM_YCB_ROOT=~/genvs_ws/assets/ycb`, every command
`nice -n 19` with a `timeout`. Forbidden: the `isaac` container, `~/genvs_ws/night_*.lock`,
`/workspace/vssim_out`, `~/genvs_ws/vssim_out`, any write under `~/runpod_dev/vs-sim`,
`~/runpod_dev/refine-vs*`, `~/runpod_dev/vs-sim-*`, `~/genvs_ws/refine-vs*`,
`~/genvs_ws/cache`, `~/genvs_ws/torch_home`. CPU budgets are estimates at load ~20, to be
replaced by measured numbers in each gate's report.

### Milestone 1 — "a faithful copy, the depth camera, the poster, ViT-VS flying one start on CPU"

**U0 — Environment, pin, weights copy** *(in progress; unchanged)*
- Gate: `import torch` prints `cuda available: False`; `nvidia-smi` memory unchanged;
  `external/vs-sim` at the pinned commit (`external/VSSIM_COMMIT`), **not** pip-installed;
  the pristine clone's fast tier run once via `PYTHONPATH=external/vs-sim py -m pytest
  external/vs-sim/tests/fast -q` and its pass/skip split written to
  `results/bridge/u0_clone_fast_tier.txt` (this is the number U-copy must match). Any red test
  caused by cv2 5.0 / numpy 2.5 is reported, never patched.
- Cost: ~10 min; disk ~1.5 GB venv + 0.25 GB weights.

**U-copy — The faithful copy and the drift tool** (`tools/vssim_copy.py`, `tools/vssim_drift.py`, `tools/vssim_copy_patches.json`, `vsbench/vs/**`, `tests/vs/**`, `vsbench/vs/{COPY_MANIFEST.json,PROVENANCE.md}`, `tests/vs/test_drift.py`; `pyproject.toml` package-data **after U0 has finished with that file**)
- Inputs: U0's pin. Source blobs read with `git show`, never the clone's working tree.
- Do: §2.1–2.4 exactly: copy list of §2.2, rules of §2.3, ≤ 6 patches, provenance line,
  manifest; `[tool.setuptools.package-data] "vsbench.vs" = ["data/**/*"]`; the drift tool and
  its test; `PROVENANCE.md` naming what is flown here and what is copied-not-wired.
- Gate: `py -m pytest tests/vs -q` passes with the **same pass/skip split as U0's clone run**
  (skip-for-skip); `tools/vssim_drift.py` exits 0 and `tests/vs/test_drift.py` is in the
  tier; copied `test_package.py` proves `import vsbench.vs` pulls no torch; `ruff check
  vsbench/vs tests/vs` with vs-sim's own ruff rules (`E,F,I,UP,B,SIM`, line 100) is clean
  modulo the provenance line; `git status` shows no file under `vsbench/vs/` outside the
  manifest; vsbench's own gates (`pytest tests --backends mujoco-egl -x -k "not vs"`) still
  green.
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
  `vsbench/vs/data/configs/arms/*.yaml`.
- Do: §2.6. `torch.set_num_threads(2)` at start-up; refuse to run if `torch.cuda.is_available()`.
- Gate (**the M1 gate**): ViT-VS DINOv3 (real weights, CPU) on hollywood from the first
  `near60` start (3–8 mm away), 100 steps: the record exists, `final_translation_mm` < the
  start's offset and < 5 mm, `predicate_a` success under the frozen schema, `n_matches` ≥ 20
  on frame 1; plus `sift_ransac` on the same start as a weight-free smoke. Records under
  `results/bridge/`.
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

**U5 — Baseline parity: mock vs MuJoCo, ViT-VS and ART-VS** (`tools/parity_mock_vs_mujoco.py`, `results/bridge/u5_parity.md`)
- Do: fly `vitvs_dinov3`, `vitvs_dinov2`, `artvs` (cpu) on hollywood from three `near60` and
  two `narrow20` starts, 150 steps, on the copied `MockPlaneScene(texture=hollywood,
  render_scale=1)` and on `MujocoScene`; same seeds. Compare landing (mm, deg) at step 150,
  the `cns_plateau` step, first-frame match count, twist-norm curve correlation.
- Gate: landings within 3 mm / 0.3° on every start; ART-VS's coarse→tiled switch fires in both
  worlds (`artvs_switched` on the record; the step from the telemetry). A miss is a finding
  about the renderer, written down, never tuned away.
- Cost: ART-VS tiled phase ≈ 1.5 s/step → 150 steps ≈ 4–5 min per start; ≈ 1 h in the
  background with `timeout 2h`.

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

**U8 — Perturbations that need world verbs** (`mujoco_scene.py`: `build_poster`, `put_box`, `remove_prim(s)`, `place_extra_objects`, `set_prim_visible`, `object_world_bbox`, `object_pose`, `set_lighting`/`lighting_snapshot`/`restore_lighting`)
- Do: each verb edits a `SceneSpec` copy and recompiles (camera pose kept); the copied
  `perturb._needs` then admits `ground:`, `occluder`, `distractors`, `light`, `combined`,
  `swap:@name` (needs U7 + instance-suite scenes via `scene_spec_for`).
- Gate: the copied `test_perturb` contract against `MujocoScene` (begin/end leaves the goal
  render md5 unchanged); one episode per mode on `cluttered2` with `sift_ransac` writes its
  draw to the record.
- Cost: recompile 1–3 s per episode; ~20 min.

**U9 — The rig check twin** (`vsbench/world/check_baselines.py`)
- Do: per scene × arm, the arm gets its own goal picture as the live frame (must re-find its
  points, ~zero motion), flies a 3–8 mm start 200 steps, optionally the first N `capped30`;
  `check.json` + `check.md` + goal frames; films via `FilmWriter`.
- Gate: all-green for `{sift_ransac, orb_ransac, vitvs_dinov3, artvs}` × six scenes on the
  zero-motion and near rows; the `capped30` row reported.
- Cost: ≈ 2–3 h at nice 19 → background with `timeout`.

**U10 — Docs** (`README.md` § "The benchmark and the baselines (copied from vs-sim)",
`docs/BRIDGE.md`, `vsbench/vs/PROVENANCE.md`, the `results/bridge/*.md` reports): what it is,
what it is not (§6), the environment, the cache, the pin and the drift procedure, the owner's
decisions. The owner-facing HTML view is the parent agent's call (workspace rule 7).

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
- **D-4 When our room bbox misses the frozen goal tolerance.** (a) the frozen file's goal wins,
  residual recorded (default); (b) our measured goal wins (breaks start pairing); (c) refuse.
- **D-5 Depth noise model.** vs-sim has none; adding one is a science decision. Default: not now.
- **D-6 The six Nucleus-only objects.** Default: skip, marked `nucleus` in `INDEX.json`.
- **D-7 Disk and cache location.** `~/genvs_ws/vsbench_assets/`, 4–6 GB + 10 GB cap. Default: yes.
- **D-8 CPU-only torch in the existing venv** (~750 MB; cannot touch the GPU). Default: yes.
- **D-9 Record labelling.** Every MuJoCo record carries `render_mode: mujoco-*`,
  `vsbench_commit`, `vssim_commit`; MuJoCo and Isaac numbers are never pooled. Default: yes.
- **D-10 (new) The Step-3 loop.** Keep `vsbench.episode.run_episode` as the Step 1–3 gate
  instrument beside the one benchmark loop (default), or retire it once `MethodArm` (U6a) flies
  every vsbench method through the benchmark loop. Cost of keeping: two loops in one repo, one
  clearly labelled "not the benchmark"; cost of retiring: rewrite the replay/hash/video gates.

Risks (technical, with the mitigation in the plan):
- *Shared CPU*: load ~22 on 16 threads; everything `nice -n 19`, ≤ 2 threads; wall times
  1.5–3× the estimates; long units in the background with `timeout`.
- *Faithfulness erosion*: the only defence is mechanical — the manifest, the ≤ 6 patch list,
  the drift test in the fast tier, and the fence "nothing hand-written under `vsbench/vs/`".
  A seventh patch is a question for the owner, not a commit.
- *Library versions*: cv2 5.0 / numpy 2.5 vs vs-sim's last green (cv2 4.x / numpy 1.26–2.x);
  U0's clone run is the baseline; a red test is a vs-sim finding reported upstream, never
  patched in the copy.
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
same table. It is for trying ideas fast against the real competitors and the real scoring,
catching bugs, teaching, and checking that the competitor code still behaves the same after a
change.

**How the evaluation stays safe.** Everything runs with the graphics card hidden, MuJoCo drawing
on the CPU, PyTorch's CPU-only build, lowest CPU priority and at most two threads; nothing opens
the Isaac container, the lock files, the result folders or the vs-sim checkout for writing; no
model weights are downloaded. The first milestone is small: the faithful copy with its tests
green, the camera, the three poster scenes, and ViT-VS flying one 5-mm start on the CPU in
about two minutes.

**Still open for the owner** (§8): the pixel-centre convention (D-3), "the frozen goal wins"
(D-4), no depth noise for now (D-5), skipping the six objects that live only on NVIDIA's server
(D-6), the 10 GB disk budget (D-7), CPU-only PyTorch in the existing environment (D-8), the
never-pool rule (D-9), and whether the old Step-3 loop stays as a test instrument (D-10).
