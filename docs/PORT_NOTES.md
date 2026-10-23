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
| `ground:<name>` | the floor gets another texture (`perturb.GROUNDS` names or a path; `/workspace/...` → `$VSBENCH_WORKSPACE_ROOT`) | one recompile, 6.3–6.8 s |
| `occluder` | a grey cube between the goal camera and the scene | one recompile, 7.6 s |
| `distractors` | 2–5 extra YCB objects from the converted cache, on the floor in view | one recompile, 8.8 s |
| `swap:<id>` | `/World/Obj_0` hidden, `<id>` put at its model origin | one recompile, 6.8–8.2 s |
| `swap:@<name>` | instance suite: the named live object, box centre on the goal object's | one recompile (inst_drill 1.4–4.1 s) |
| `combined`, `a+b` | the parts, in order; undone in reverse | one recompile for all parts (7.7 s) |

Plus ~0.6 s to free the old model; the restore after `end` costs one more compile (7–8 s),
deferred to the next frame drawn - in a campaign it merges with the next episode's `begin`,
so an episode's `world_compiles` can list the previous episode's restore (`unedited: true`).

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
colour is extrapolated. The same two floors under each other's curve (mean RGB of the
ground at the goal): the industrial concrete under its own scene's curve (197, 190, 184),
under cluttered2's (191, 168, 166) - about 20 levels pinker; cluttered2's wood under its own
curve (160, 116, 102), under the drill scene's (152, 91, 53). Preview of every mode at the
goal pose: `results/bridge/u8_modes.png`. Scenes without a fitted curve (the instance suite) fly in
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

**Memory.** cluttered2 holds 3.5 GB after start-up; a perturbed world 3.5–4.3 GB (each
distractor adds a 4096² texture); the peak was 4.85 GB in the measure run and 5.55 GB in the
smoke flights (`combined`, with the arm and the frames). After each `end` the process sits at
3.54–3.80 GB (the allocator keeps ~0.2–0.3 GB of freed pages; no steady growth over the 10
modes - `rss_after_end` per mode in the measure record). Budget ~6 GB per cluttered2 process
when flying perturbations, never two rooms in one process.

**Smoke** (`results/bridge/u8/`, recorded, not thresholded; `sift_ransac`, 100 steps, CPU):
cluttered2 from 7.76 mm - none 1.20 mm (U4b's number), light 0.52, occluder 0.17,
distractors 0.73, `ground:concrete` 4.13, `ground:laminate` 2.98 (divergence abort),
noise 18.41, combined 0.44, `swap:ycb/006_mustard_bottle` 1.83, `+ground:concrete` 8.64
(divergence abort); inst_drill from 6.45 mm - none 0.23, `@same` 0.23, `@same_small` 3.77 mm
from the size-matched pose, `@other_small` lost (no SIFT matches on another drill).

**Tests and tools.** `tests/world/test_perturb_world.py` (the copied `test_perturb` contract
on the real world: industrial1_drill, inst_drill, hollywood; gate JSON
`results/bridge/u8_perturb_gates.json`), `tools/u8_perturb.py measure | smoke | summary`.
