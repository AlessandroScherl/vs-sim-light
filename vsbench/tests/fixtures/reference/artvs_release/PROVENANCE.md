# The ART-VS release as a test reference - provenance

These files are the **published ART-VS code** that `vssim.baselines.objroi.RoiTiledMatcherSim`
(matcher variant `artvs_roi_sim`), the `dinov3_vits16_artvs256` backbone and the controller
knobs of 2026-10-05 are tested against (`tests/fast/test_objroi_sim.py`,
`tests/slow/test_artvs_release_parity.py`). They are third-party code: `pyproject.toml`
excludes `tests/fixtures/reference/` from ruff and `tests/fast/test_package.py` does not walk
it; nothing here may be reformatted or "fixed". If the port disagrees with them, the port
changes.

**Source**: the ART-VS release, `https://github.com/AlessandroScherl/ART-VS` as cloned in the
owner's workspace (`../ART-VS`, read-only), git **`9b1208caf552b7fbe270b4797ee8e2e51916845b`**
("Initial public release of ART-VS (IROS 2026)"). MIT, "Copyright (c) 2024-2026 Alessandro
Scherl and the ART-VS authors" - `LICENSE` here is the release's file (md5
`2cae802ed078c53f1d0940218dc65208`). Path shorthand: `R/` =
`catkin_ws/ibvs/src/visual_servoing/`.

The build spec is `docs/results/2026-10-05_artvs_release_check/sim_spec.md` (workspace root);
the owner's rulings of 2026-10-05: the release's **simulation** program in object mode,
rotation compensation off, the bench's shared gain / step cap / scored stop, the release's
visual-only stop recorded beside it; D-A (a) the adaptive tile grid; **D-B (b) LangSAM's goal
mask** through the simulation controller's own mask branch (the evening ruling - it replaced
the spec's D-B (a) "no mask"); D-C a velocity cap of 10.0 (the release's 1.0 times 2.0 / 0.2);
D-D EMA alpha 0.8; D-E hold the last twist on a degenerate frame and record, never abort.

## The files

| file here | upstream file (md5 at 9b1208c) | kept | left out, and why |
| --- | --- | --- | --- |
| `matcher.py` | `R/features/matcher.py` (`ffa75e2932ebc37d3970cbb2219037b4`) | the whole file, byte for byte (266 lines; md5 identical) | - |
| `controller.py` | `R/core/controller.py` (`1bdc99f979dba12fadd19367efb2b862`) | the whole file, byte for byte (409 lines; md5 identical). Tests hand it a `types.SimpleNamespace` config with the simulation config's values | - |
| `log_bin_release.py` | `R/features/multi_backbone_extractor.py` (`bea6f7f32e3c749d8e742d2e6608192b`) | `MultiBackboneExtractor._log_bin`, lines 852-912, as a function `log_bin_release(x, num_patches, hierarchy, device)`: `self.num_patches` / `self.device` became the arguments | the class around it; the one-time `[BINNING]` print (869-874). The station parity test imports the real extractor from the checkout instead |
| `detector_state.py` | `R/features/langsam_lighttrack_detector.py` (`a718bec227a56fcf5c6b6f2a70c65f5d`) | lines 22-31 (`suppress_stdout`), 35 (`logger`), 71-83 (the log helpers), 179-189 (`set_classes` - `set_yolo_keyword` calls it, so it is kept though the spec's table left it out), 191-703 (`detect`, `start_tracking`, `detect_and_cache_goal`, `get_cached_goal_roi`, `reset_tracking`, `_detect_langsam`, `_select_by_spatial_prior`, `_track_frame`, `_use_cached_bbox`, `_apply_padding`, `crop_roi`, `_empty_result`, `_xyxy_to_xywh`, `_xywh_to_xyxy`) verbatim, with exactly two injections marked `TRANSCRIPTION:` - `self.langsam.predict(...)` -> `self.predict(...)` (a callable with LangSAM's signature and result dict) and `LightTrack(...)` -> `self.tracker_factory(...)` | the imports and the constructor (1-177: the `lang_sam` import, the LightTrack path discovery, the model loading), replaced by a minimal constructor that sets the same state fields (128-130, 162-177); `HAS_LIGHTTRACK` is a constant `True` (the tracker is injected) |
| `roi_glue.py` | `R/ros_interface/ros_controller.py` (`f6ff534545d025d63e47107f5953b9c8`) | class `ROSVisualServoingController`, these methods byte for byte, each under a `# ---- release lines a-b` marker: `_cache_goal_image_processing` 317-346, `_cache_goal_roi_features` 348-405 (**the mask branch included** - it runs whenever `config.mask_path` names a file, which is how the LangSAM goal mask reaches it), `get_depth` 407-428, `check_roi_tiling_switch` 594-634, `check_tiling_switch` 636-681, `detect_features` 683-748, `detect_features_original` 750-854, `calculate_uv` 856-890, `calculate_uv_roi` 892-917, `set_yolo_keyword` 919-973, the second `detect_features_roi` 1360-1571 (the one Python keeps), `detect_features_roi_tiled` 1573-1982 (its mask branch included), `_map_roi_to_full_image` 2140-2171, `ibvs` 2472-2594, `_calculate_optimal_tile_number` 2781-2837, `get_effective_tile_number` 2839-2848, `__del__` 2890-2901. Plus, marked `TRANSCRIPTION:`: a header (`rospy` is a no-op stub; `core.controller` / `features.matcher` are the vendored copies), a constructor setting the state fields of `__init__` (66-231) that the kept methods read, `begin_sample` (= run()'s per-sample resets, 2603-2643 verbatim), `step` (= one pass of run()'s loop, 2688-2699, with `publish_twist`'s clip, 438) and a `detect_features_tiled` that raises | ROS, cv_bridge, TF, the publishers and subscribers, Gazebo pose reads, the classical paths, the dangling first `detect_features_roi` (975-1140 - its body uses names never assigned; Python keeps the second definition), the full-image 6 x 6 tiled path 1142-1358 (the box-less fallback of the tiled phase: the port's one stated deviation, sim_spec.md 4.3), the visualisation helpers (2214-2470). **One difference from the spec's plan**: the spec listed the kept methods "minus the visualisation blocks" and with `rospy.log*` and `time.perf_counter` lines dropped; they are kept here byte for byte instead, behind the no-op `rospy` stub, because every visualisation call is guarded by `enable_visualization`, which the simulation config sets `false` (S:63) - so the methods are mechanically identical to the release and the guard decides, as it did there |
| `make_roi_glue.py`, `make_detector_state.py` | - | the scripts that assembled the two transcriptions from the release's lines (`python3 make_roi_glue.py ../../../../../ART-VS`); re-running them reproduces the files byte for byte, which is how a reader checks the "byte for byte" claims | - |
| `LICENSE` | `LICENSE` (`2cae802ed078c53f1d0940218dc65208`) | whole file | - |

## The simulation config the arm reproduces (`S:` = `R/configs/hollywood_500_evaluation/config_dinov3_256_6x6_tiling.yaml`, md5 `2eda068f37ca6a4fe92ec0627a09de33`)

```yaml
backbone_model: 'dinov3-small'
vit_input_size: 256
image_path: 'goalrgb.jpg'  # Hollywood poster
u_max: 1440
v_max: 1080
f_x: 1130.1787109375
f_y: 1130.1787109375
lambda_: 0.2
max_velocity: 1.0
num_pairs: 24
num_samples: 500
num_circles: 4
circle_radius_aug: 0.08
box_sample_size: [1.2, 1.2, 0.3]
velocity_convergence_threshold: 0.00008
velocity_threshold_translation: 5.0e-19
velocity_threshold_rotation: 5.0e-19
error_threshold_ratio: 0.001
error_threshold_absolute_translation: 0.1
error_threshold_absolute_rotation: 0.1
min_iterations: 300
max_iterations: 1500
max_velocity_vector_history: 200
ema_alpha: 0.8
thresh_filter_keypoints: 1
background_thresh: 0.005
use_feature_binning: true
use_tiling: true
use_hybrid_mode: true
tiling_switch_iteration: 0
hybrid_switch_threshold: 0.2
roi_tiling_switch_threshold: 0.2
tile_number: 36
tile_input_resolution: 1536
auto_calculate_tiles: false
batch_process_tiles: true
use_roi_detection: false
use_roi_tiling: false
enable_visualization: false
stable_convergence_enabled: true
stable_convergence_position_threshold: 0.1
stable_convergence_rotation_threshold: 0.1
stable_convergence_window: 50
enable_rotation_compensation: true
rotation_mode: 'discrete'
rotation_learning_rate: 1000.0
rotation_delta: 5.0
rotation_max_step: 5.0
rotation_convergence_threshold: 1.0e-05
rotation_max_iterations: 50
use_mixed_precision: false
mask_path: null
```

(comments trimmed). **What the arm changes, and why** (the owner's rulings): `use_roi_detection`
and `use_roi_tiling` on (object mode); `auto_calculate_tiles` on (D-A (a)); the detector
`langsam_lighttrack` (the code default is `yoloworld`, `R/core/config.py:142`);
`mask_path` = LangSAM's goal mask (D-B (b)); `enable_rotation_compensation` off; the camera,
the gain, the step cap and the stop are the bench's.

**Code defaults used** for the keys `S:` does not set (`R/core/config.py`, md5
`e323f072b04da9be5597deb342aa528a`): `feature_hierarchy` 1 (:39 - and the ROI calls pass no
`hierarchy` at all, so the wrapper default 1 applies either way), `hybrid_switch_threshold` is
set by `S:51` (the `Config` default would be 0.1, :50), `roi_padding_ratio` 0.1 (:136),
`yolo_confidence_threshold` 0.3 (:137, stored and never applied), `detector_type`
`yoloworld` (:142), `lighttrack_arch` `LightTrackM_Subnet` (:149), `cache_timeout` 10 (:150),
`roi_tile_number` 4 (:154, unused once the grid is automatic), `roi_tiling_switch_threshold`
0.2 (:155, shadowed by `hybrid_switch_threshold`), `robot_mode` `simulation` -> the
ground-truth stop (:180, 196).

**Two driver facts** (`R/experiments/run_multi_model_experiment_with_hollywood.py`, md5
`195bb64a2e0d96fca78a7bd2ed99dc42`): the prompt reaches the controller through
`set_yolo_keyword` (:636-639), which is where the goal is detected and the tile grid computed;
and before every sample the driver resets the detector (:688-690), runs ONE LangSAM detection
on the live frame and starts LightTrack with the raw box (:741-755) - so the loop's first box is
a LightTrack update.

**The paper lines the arm departs from** (`paper.txt` = the paper's linear text extraction,
cited in sim_spec.md): `:240-242` (the mask sentence - the owner's D-B (b) now uses LangSAM's
mask, through the simulation code's own mask branch), `:315-319` (`Kt = max(4, floor(K/n))` -
the code's `max(1, 24 // n)` is followed, the paper's rule is the known paper-vs-code gap),
`:551-553` ("the final transformer block" - the code reads `last_hidden_state`, after the final
LayerNorm).
