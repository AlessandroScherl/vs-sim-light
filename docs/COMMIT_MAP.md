# Commit map of the 2026-10-23 history rewrite

On 2026-10-23 the history was rewritten (owner decision) to erase two unrelated PDFs and an unreferenced video (`docs/supplementary.mp4`). Every commit ID changed. Records and gate files written before the rewrite cite the OLD IDs (e.g. `vsbench_commit`, `fitted_at_vsbench_commit`); this table translates them. Commits that only removed those files became empty and were dropped.

| old | new | subject |
|---|---|---|
| `c920d2747c8b` | `1dab3a4d9440` | U6b: the vitvs_dinov2 setup check report (CPU, llvmpipe; zero and hold pass, gated; restor |
| `4f75df1fd95d` | `2dfc3f8daf94` | Three comparison sheets as JPEG (u4_rooms_vs_isaac, u4b_look_vs_isaac, u8_modes: 11 MB ->  |
| `7336f89f5a4b` | `b20722644115` | U4b held out at a far pose, report only: Isaac's first capped30 live frame (check3 films,  |
| `c1f91a8ace22` | `ca61582a0b5c` | PORT_PLAN: U10 names docs/PORT_NOTES.md (not docs/BRIDGE.md), marked done |
| `f4bcfa6f77e0` | `7527bde9546a` | Look config: the Isaac root as $VSBENCH_ISAAC_GOALS (resolved where read; md5s kept, no re |
| `d9abf89f801e` | `b2699903b4fb` | U9 / D-12: final < start stays the recorded near verdict (final_over_start reported, no th |
| `73d5f914dd5f` | `5ccde7d63890` | U5 gate re-stated from the committed records (accepted with findings): landing gate kept f |
| `e4fb2c6c226b` | `7a4ff72ab70e` | pyproject: PyYAML and torch in dependencies (the copied runner imports torch at load), tra |
| `9b36e863df87` | `999434f00b71` | fly --allow-gpu sets VSBENCH_ALLOW_GPU=1 by default, so its GL behaviour is unchanged unde |
| `d2f174fb55c7` | `577f05faa8ed` | The last GPU route closed: software GL is the default (MujocoScene, MujocoRenderer, make_r |
| `ec5661ed2b43` | `1a1b9a2558a5` | Root clutter: chain logs removed, Step 1-3 chains moved to tools/step_chains (repo-relativ |
| `d6040a12af22` | `72120762c7b6` | Merge U9 (the check_baselines twin on MuJoCo: identity 24/24; near 15/24 closer, every mis |
| `3aae06d891b2` | `a0284aa01ce0` | U9 docs: PORT_NOTES (the twin, its flags, the near verdict, result, cost), PORT_PLAN (U9 a |
| `05c7362c5cff` | `07be86fc232c` | U9 flown: the rig check twin, 4 arms x 6 scenes on MuJoCo (CPU, llvmpipe) |
| `c4bc94ee5a93` | `d2fbc7c7542d` | U9 tests and the diagnosis tool |
| `1369f725e416` | `f8b323d981c5` | U5 docs: PORT_PLAN (U5 as built, measured, what explains the differences, cost; plain word |
| `5124e3b4b85a` | `b50ca80971cc` | U5 results: hollywood, 4 arms x 5 starts x 150 steps in the copied mock, MuJoCo raw and Mu |
| `fb0d05608ff3` | `ad460c13f599` | U5: per arm and world pair, the spread of the landing and curve differences in the compari |
| `5cfb39ba05ba` | `62026218d2c3` | U5: the switch verdict says where it did not fire and whether the worlds agree; the gate t |
| `e4b2f0c85820` | `defe71072f2a` | U9: python -m vsbench.world.check_baselines, the twin of vs-sim's rig check on MuJoCo |
| `06c041871d1a` | `9d637dd32fd6` | README: perturbations and the object library are in (U8) |
| `4ea65cdc7349` | `80306fdbc014` | Merge U8 (every vs-sim perturbation mode on the MuJoCo world: light, occluder, distractors |
| `5677cccecc6a` | `0c76d728bce9` | U8 docs (2/2): PORT_PLAN (capability rows now built; U8 as built, measured, smoke; D-11 th |
| `f68fe1e8d1ca` | `196bf50fb346` | U8 results: the gates (14/14: the copied perturb contract on industrial1_drill, inst_drill |
| `8eea92b50cdd` | `1d39ce26f7c0` | U8 docs (1/2): docs/PORT_NOTES.md (the U8 section: modes, costs, the look under a perturba |
| `ba2b078cdf25` | `a54ab03d00d3` | Merge the user-facing README (short guide, demo GIF, scenes, debug views, one fly command, |
| `a8f438f517b9` | `36fc90b83c02` | Front page for outside users: README rewritten (what it is, the GIF, quick start, plug in  |
| `5707f213826a` | `504d70c44df0` | One command for baselines and your own method: python -m vsbench fly (routes protocol cell |
| `90bc336cc66f` | `b8e5c92f203b` | U8: the perturbation world verbs on MujocoScene (build_poster, put_box, remove_prim(s), pl |
| `e8f10f01eac5` | `d6b8ed0249ff` | U5: the parity tool (mock / MuJoCo raw / MuJoCo at the mock's principal point, same starts |
| `50cc96ecfa42` | `7057e3bc8da4` | U4b gates green: the look gate JSON (config pinned, raw = U4 bit for bit, depth identical, |
| `aceae8d04dd5` | `642ac8473422` | U4b docs: README (Isaac's look), PROVENANCE (the look on every record; the copy untouched) |
| `8562b033b332` | `05c808ccbf41` | U4b smoke (2/2): the U4 smoke in the isaac_fit_v1 look, and the table beside raw and Isaac |
| `16f1f13384ba` | `6b35dda5641c` | U4b smoke (1/2): the U4 smoke re-flown in the raw look reproduces U4 to the last digit (si |
| `818027de86e1` | `5626cb80ed51` | U4b results: raw vs isaac_fit_v1 vs Isaac on the six scenes (goal fitted, near view held o |
| `7076ead002fd` | `e4337114cfb0` | U4b: the fitted look config isaac_fit_v1 (fill share 0.15, gains, six per-scene curves, Is |
| `7f43ba2842f9` | `3ed2f5ac669c` | U4b: the Isaac look (isaac_fit_v1 default, raw for the mock gates) - code and gates |
| `2a8e3546a7a2` | `3a7fd7516e1b` | Close the third GPU route: the gate harness picks software variants only under VSBENCH_SOF |
| `2df1a9fa89fe` | `7f9c452eff36` | Merge U6a/U6b (MethodArm: vsbench methods on the copied loop; the copied arms as vsbench m |
| `b8a8d1b03468` | `9979d32942f2` | U6 results (2/2): vsbench.check reports for akaze_ransac, artvs, vitvs_dinov3 (restoring r |
| `7a56b35dd2fd` | `5bb0eaabfc85` | U6 results (1/2): the U6a flight, the equivalence gate, the demo gate, two classical check |
| `41dc4f61ae6b` | `269944d2bc3b` | U4 gates green on the three rooms; costs, previews, the Isaac side by side |
| `84b6f587b500` | `66c2e5c8c6df` | U4 smoke: ViT-VS DINOv3 on cluttered2, 100 steps from n_000: 7.76 -> 65.12 mm / 2.97 deg ( |
| `7212ea92f726` | `8b975abcd398` | U6: parent rulings - contract gate skips a copied arm only when it says why (no torch / we |
| `7c030d21c003` | `54a5a372f027` | U4 smoke: sift_ransac, 100 steps, check_baselines' near start n_000 (7.76 mm / 2.69 deg) o |
| `034c9d4d1ccc` | `45d5ea74f83c` | U6a/U6b docs: README (the Step-3 loop is not the benchmark loop; MethodArm and the copied  |
| `73b96f071320` | `24eb1f027756` | U6a + U6b: MethodArm (a vsbench method on the copied benchmark loop) and the copied arms a |
| `adf655ceb0a5` | `c75cb8591375` | U4: the three rooms in MujocoScene (ground, converted objects by the frozen layout, bbox-d |
| `58cc70f6ddaa` | `3f304efc92c2` | Merge U7 (the whole object library converted: 317/317 on disk ok, 6 Nucleus-only skipped,  |
| `9b428df2b868` | `70b0aaab1b90` | U7: the full library converted (317/317 on-disk curated ok, 6 nucleus skipped), 2.51 GB; s |
| `abfc28ce5212` | `b7655490d610` | Merge branch 'port/vssim' into port/u7-library |
| `ae6181b4ef7c` | `dbc8f269068c` | U7: convert --all (the whole library, resumable, disk guard), the extent spot check, sheet |
| `585408f27d59` | `709099a74073` | U2 / M1 review: the truth seam in a flight, and ViT-VS's error curve |
| `04521b1cdc2c` | `46c465ae40b1` | PORT_PLAN revision 3 (after the M1 review) and the README's world/driver section |
| `65c603522c80` | `c2e76007bb47` | M1 review: hard GPU refusal, honest gates, one dirty rule, the warm-up on the record |
| `bb324cc1e7c1` | `e2eeb2fee625` | M1 review: harden patch P2 (git status --short pinned against git config); upstream film-e |
| `7f63d316a2f1` | `0d915abf0113` | U2 / Milestone 1: ViT-VS (DINOv3) and SIFT+RANSAC fly hollywood on the MuJoCo world (CPU) |
| `409656dbcb99` | `6809eed3c41b` | U2: the bench driver python -m vsbench.world.fly (run, summary) |
| `b8217c7d01fc` | `e54b49e86d5e` | U1: MujocoScene for the posters + the D435i camera gates |
| `cbf539caacb2` | `b006bb344d3c` | Merge U3 (object converter, bbox-goal gate) into port/vssim |
| `42965664ce95` | `0b0b67958fe4` | U-copy: the faithful copy of vs-sim in vs-sim's own layout (B), green |
| `ae0708feaaf5` | `23900fd75397` | U-copy (part 1, gate NOT met): the copy tool, the drift tool and their test - stopped for  |
| `1b3c2a3f349c` | `2c447adf1910` | U3: object converter (USD/OBJ -> MuJoCo cache) and the bbox-goal gate |
| `f0458baec831` | `23a2b1ce17ee` | Pin OpenCV to 4.11.0.86 (the flight interpreter's) and re-measure the clone's fast tier |
| `7e0bd2fc0f5d` | `0af5a4b7dffd` | U0 (plan rev. 2): the clone's per-test fast-tier split, asset roots in cpu_env.sh |
| `36648e515a4b` | `97693ea819ef` | U0: CPU-only environment and the vs-sim pin (tools/cpu_env.sh, external/VSSIM_COMMIT) |
| `d43ee6decb3b` | `38bf2ac3ee3c` | PORT_PLAN revision 2: the owner's rulings (copy vs-sim into vsbench/vs, separate for good) |
| `3e96ed2c7de2` | `02ff167d2461` | PORT_PLAN: the plan for porting vs-sim's world, D435i depth camera, objects and the ViT-VS |
| `39522f33c4c0` | `097c3860b44d` | Snapshot of VS-Bench as found on the station (steps 1-3, 2026-09-19), before the port from |
| `351f23695347` | (dropped: empty after the rewrite) | Remove two unrelated PDFs (a bachelor-thesis concept and a second document) from docs/ (ow |
| `b5df482e24a9` | (dropped: empty after the rewrite) | Remove two unrelated PDFs (a bachelor-thesis concept and a second document) from docs/ (ow |
