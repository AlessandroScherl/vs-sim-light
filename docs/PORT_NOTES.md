# PORT_NOTES — notes for users of the MuJoCo port, per unit

Notes that would otherwise have gone into `README.md` (owner request 2026-10-23: the README
is being rewritten into a short user guide; per-unit notes live here). The plan is
`docs/PORT_PLAN.md`; the copy's provenance is `vsbench/vs/PROVENANCE.md`.

---

## U8 — perturbations on the MuJoCo world (2026-10-23)

**What you can do.** Every perturbation mode of the copied `vsbench.vs.bench.perturb` now
runs on `MujocoScene`:

```
source tools/cpu_env.sh
$VSBENCH_PY -m vsbench.world.fly run --scene cluttered2 --arms sift_ransac --start-set capped30 \
    --limit 1 --max-iter 100 --perturb occluder --out results/my_run       # vs-sim's flag
#   --perturb light | distractors | occluder | noise | combined | ground:<name|path>
#             | swap:<instance_id> | swap:@<live name> | a+b   (--perturb-seed, default 20260918)
```

| mode | what the world does | cost on cluttered2 (CPU, load ~25) |
| --- | --- | --- |
| `noise` | nothing (the copied per-frame sensor model on the picture) | 0 |
| `light` | relights the picture through the scene's tone curve (below) | 0 (no recompile) |
| `ground:<name>` | the floor gets another texture (`perturb.GROUNDS` names or a path; `/workspace/...` → `$VSBENCH_WORKSPACE_ROOT`) | one recompile, ~7 s |
| `occluder` | a grey cube between the goal camera and the scene | one recompile, ~8 s |
| `distractors` | 2–5 extra YCB objects from the converted cache, on the floor in view | one recompile, ~9 s |
| `swap:<id>` | `/World/Obj_0` hidden, `<id>` put at its model origin | one recompile, ~8 s |
| `swap:@<name>` | instance suite: the named live object, box centre on the goal object's | one recompile (inst_drill ~1–5 s) |
| `combined`, `a+b` | the parts, in order; undone in reverse | one recompile for all parts |

Per-mode numbers (begin / end, compile seconds, RSS): `results/bridge/u8_perturb.json`.

**How it works.** `vsbench/world/perturb_world.py` gives `MujocoScene` the verbs Isaac's scene
has (`build_poster`, `put_box`, `remove_prim(s)`, `place_extra_objects`, `set_prim_visible`,
`object_pose`, `object_world_bbox`, `set_lighting` / `lighting_snapshot` /
`restore_lighting`, `extra_placements`), under Isaac's prim paths (`/World/Ground`,
`/World/Obj_<i>` in layout order, `/World/Occluder`, `/World/Distractor_<k>`,
`/World/Swap_<k>`). MuJoCo bakes geometry and textures into the compiled model, so an edit
marks the world dirty and the next frame recompiles it — the old model and its GL contexts
are freed first (never two rooms in one process), and several edits (a chain's parts, one
episode's `end` plus the next one's `begin`) cost one compile. A recompile keeps the camera,
the look's renderer settings and MuJoCo's model statistics (`stat.extent` / `stat.center`)
of the unedited scene, so a pixel that sees the same unedited surface keeps its depth bit
for bit; when the world is back to the unedited scene it is compiled exactly as at start-up,
so the goal redrawn after `end` has the goal's md5 (gated, every mode).

**The goal picture is the unperturbed one** (as in vs-sim): the runner captures it before
the first `begin` and serves it from the cache at the goal pose. `MujocoScene` adds one
guard: a verb that edits the world captures the goal first if nobody has.

**The look under a perturbation (builder decision, recorded as D-11 in PORT_PLAN §8).** The
scene's look (`isaac_fit_v1`: fill share, light gain, shadow map and its fitted square,
tone curve) is fitted once on the unperturbed goal and **stays fixed**: an occluder, a
distractor, another floor or another object is drawn through the same renderer response,
as Isaac's fixed renderer would draw it. Nothing is re-fitted per perturbation. The cost,
measured (`results/bridge/u8_perturb.json`, `ground_mean_rgb_at_goal`): the curve is only
constrained by the colours of its own scene's goal picture, so a floor of very different
colour is extrapolated — cluttered2's curve (fitted on dark wood) shows the industrial
concrete brighter and pinker than the drill scene's own curve shows the same concrete (see
the numbers in the U8 section of PORT_PLAN §7 and the preview sheet
`results/bridge/u8_modes.png`). Scenes without a fitted curve (the instance suite) fly in
the `raw` look, as before.

**The light.** vs-sim's `light` scales the dome and the sun by the same `s` and gives both
the same colour temperature `T`, so every pixel's linear radiance changes by
`f_c = s * blackbody_c(T)` (USD's `UsdLux.BlackbodyTemperatureAsRgb`, luminance 1; Isaac's RTX
may use its own table). MuJoCo clamps each light's diffuse term at 1 and the fitted look
already drives the sun to ~1, so a brighter light cannot be drawn in the model; it is applied
inside the scene's tone curve instead: `y = 255 srgb(w / (1 + w))`, `w = f_c (floor + e^k
x^gamma)` (the `raw` look: `y = clip(f_c x)`), a 256-entry table, depth untouched. A light
state in which the dome and the sun differ in factor is refused.

**Refused, loudly:** an unknown prim (`ValueError`, as Isaac); a Nucleus asset (D-6, not on
disk) or an object missing from the converted cache (`FileNotFoundError` naming
`tools/convert_objects.py`); a non-square tiled poster; unequal light factors. On a poster
scene the copied modes themselves refuse `ground:` (no ground block) and `swap:` (no object
list). Nothing else is refused: every copied mode is feasible.

**The distractor pool** is the copied rule (the YCB library minus the scene's own objects,
82 − 8 ids); U8 converted the 74 YCB scans the cache did not hold (`tools/convert_objects.py
convert <ids>`, 74/74 ok, 4096² textures, 249 s, cache now 2.9 GB).

**Memory.** cluttered2 holds ~3.5 GB; a perturbed compile peaks at ~4.3 GB (`combined`:
distractors add 4096² textures). After an `end` the process stays ~0.2–0.3 GB above its
start-up RSS (the allocator keeps freed pages); repeated episodes do not grow it further
(`rss_after_end` per mode in the measure record).

**Tests and tools.** `tests/world/test_perturb_world.py` (the copied `test_perturb` contract
on the real world: industrial1_drill, inst_drill, hollywood; gate JSON
`results/bridge/u8_perturb_gates.json`), `tools/u8_perturb.py measure | smoke`.
