# Transcribed from the ART-VS release (git 9b1208c, MIT, "Alessandro Scherl and the ART-VS
# authors"): catkin_ws/ibvs/src/visual_servoing/ros_interface/ros_controller.py, the
# ROS1/Gazebo SIMULATION controller, class ROSVisualServoingController.
#
# Every method marked "release lines a-b" below is those lines of that file, byte for byte
# (assembled mechanically; tests/fast/test_objroi_sim.py re-reads the release and checks
# each block where the checkout is present). What is NOT the release's is marked
# "TRANSCRIPTION:": this header (ROS, cv_bridge, TF and the publishers are absent: `rospy`
# is a no-op stub, and the release's own `features.matcher` / `core.controller` are the
# vendored copies beside this file), the constructor (only the state fields the kept
# methods read, set as ROSVisualServoingController.__init__ sets them, lines 66-231),
# `begin_sample` (run()'s per-sample resets, lines 2603-2643 verbatim), `step` (one pass of
# run()'s loop, lines 2688-2699 + publish_twist's clip, line 438) and
# `detect_features_tiled`, which raises: the full-image 6x6 tiled fallback (lines
# 1142-1358) is the one path the bench's port does not build (sim_spec.md section 4.3).
# The visualisation calls inside the kept methods are kept too; every one is behind
# `enable_visualization`, which the simulation config sets false (S:63). See PROVENANCE.md.
import functools
import importlib.util
import os
import sys
import time
import types

import numpy as np
import torch
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    """TRANSCRIPTION: the vendored release modules, loaded by path under their own names -
    nothing is put on sys.path, so no other `controller` / `matcher` / `features` is shadowed."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_vendored_controller = _load("artvs_release_controller", "controller.py")  # core/controller.py
_vendored_matcher = _load("artvs_release_matcher", "matcher.py")           # features/matcher.py


def _release_imports(fn):
    """TRANSCRIPTION: `detect_features_roi_tiled` imports `features.matcher` inside its body
    (line 1767); while it runs - and only then - that name is the vendored copy, and whatever
    held the name before is put back (another test's own `features` package included)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        saved = {n: sys.modules.get(n) for n in ("features", "features.matcher")}
        pkg = types.ModuleType("features")
        pkg.__path__ = []
        pkg.matcher = _vendored_matcher
        sys.modules["features"], sys.modules["features.matcher"] = pkg, _vendored_matcher
        try:
            return fn(*args, **kwargs)
        finally:
            for n, mod in saved.items():
                if mod is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = mod
    return wrapper


VisualServoingController = _vendored_controller.VisualServoingController
find_correspondences_batch = _vendored_matcher.find_correspondences_batch
scale_points_from_patch = _vendored_matcher.scale_points_from_patch


class _RospyStub:
    """TRANSCRIPTION: rospy's logging calls as no-ops (the release logs; nothing is read)."""

    def logdebug(self, *a, **k):
        pass

    loginfo = logwarn = logerr = logdebug

    def logwarn_throttle(self, *a, **k):
        pass

    def sleep(self, *a, **k):
        pass

    def is_shutdown(self):
        return False


rospy = _RospyStub()


class ROSVisualServoingController:
    """The release's simulation controller, minus ROS (see the header)."""

    def __init__(self, config, feature_extractor, yolo_detector, goal_image):
        # TRANSCRIPTION: the state ROSVisualServoingController.__init__ sets (lines 66-231)
        # that the kept methods read; the detector is injected (lines 122-139 build it)
        self.config = config
        self.feature_extractor = feature_extractor
        self.vs_controller = VisualServoingController(config)
        self.roi_tiling_activated = False
        self.roi_tiling_switch_iteration = None
        self.optimal_tile_number = None
        self.consecutive_no_features_count = 0
        self.max_consecutive_no_features = 5
        self.latest_image = None
        self.latest_image_depth = None
        self.latest_pil_image = None
        self.camera_position = None
        self.orientation_quaternion = None
        self.desired_position = None
        self.desired_orientation = None
        self.feature_failure_count = 0
        self.last_num_features = 0
        self.initial_feature_error = None
        self.current_feature_error = None
        self.goal_image_resized = None
        self.goal_tensor = None
        self.goal_features = None
        self.goal_roi_resized = None
        self.goal_roi_tensor = None
        self.goal_roi_features = None
        self.goal_roi_mask_path = None
        self.yolo_detector = yolo_detector
        self.detector_name = "LangSAM+LightTrack"
        self.yolo_keyword = None
        self.goal_roi_bbox = None
        self.current_roi_bbox = None
        self.last_valid_roi_iteration = 0
        self.tracking_started = False
        self.tiling_activated = False          # first defined in run() in the release
        self.tiling_switch_iteration = None
        self.goal_image = goal_image
        self.goal_image_array = np.array(self.goal_image)
        self._cache_goal_image_processing()

    def begin_sample(self):
        # TRANSCRIPTION: run()'s per-sample resets - release lines 2603-2643, verbatim

        self.initial_feature_error = None
        self.current_feature_error = None

        # Reset tiling activation state for new sample
        self.tiling_activated = False
        self.tiling_switch_iteration = None

        # Reset ROI tiling activation state
        if hasattr(self, 'roi_tiling_activated'):
            self.roi_tiling_activated = False
            self.roi_tiling_switch_iteration = None

        # Preserve preset initial errors if they exist
        preset_trans_error = None
        preset_rot_error = None
        preset_flag = False
        if hasattr(self.vs_controller, 'initial_errors_preset') and self.vs_controller.initial_errors_preset:
            preset_trans_error = self.vs_controller.initial_error_translation
            preset_rot_error = self.vs_controller.initial_error_rotation
            preset_flag = True
            rospy.logdebug(f"Preserving preset errors: Trans={preset_trans_error:.2f}cm, Rot={preset_rot_error:.1f}°")
        
        # Reset controller state
        self.vs_controller.reset()
        self.feature_failure_count = 0
        self.consecutive_no_features_count = 0  # Reset classical method failure count
        self.tiling_activated = False
        self.tiling_switch_iteration = None
        self.roi_tiling_activated = False
        self.roi_tiling_switch_iteration = None
        
        # Restore preset initial errors after reset
        if preset_flag:
            self.vs_controller.initial_error_translation = preset_trans_error
            self.vs_controller.initial_error_rotation = preset_rot_error
            self.vs_controller.initial_errors_preset = True
            rospy.logdebug(f"Restored preset errors after reset: Trans={preset_trans_error:.2f}cm, Rot={preset_rot_error:.1f}°")
        
        # Reset current ROI cache at start of servoing
        self.current_roi_bbox = None
        self.last_valid_roi_iteration = 0

    def step(self):
        """TRANSCRIPTION: one pass of run()'s loop (lines 2688-2699): the twist the camera
        gets (publish_twist -> store_applied_velocities, the clip, line 438), or None when
        the frame is skipped - nothing is published and Gazebo keeps the last twist. The
        RuntimeError("Persistent feature detection failure") propagates: run() catches it
        and aborts the sample (lines 2692-2695)."""
        v_c = self.ibvs()
        if v_c is None:
            return None
        return self.vs_controller.store_applied_velocities(v_c)

    def detect_features_tiled(self):
        """TRANSCRIPTION: release lines 1142-1358 are not transcribed - the full-image 6x6
        tiled fallback is the one path the port does not build (sim_spec.md 4.3)."""
        raise NotImplementedError("detect_features_tiled (the box-less fallback of the tiled "
                                  "phase) is not transcribed - the stated deviation")


    # ---- release lines 317-346: _cache_goal_image_processing
    def _cache_goal_image_processing(self):
        """Cache goal image processing to avoid redundant computation."""
        cache_start = time.perf_counter()
        
        # Check if using classical features - no caching needed for classical methods
        if hasattr(self.config, 'backbone_model') and 'classical-' in self.config.backbone_model:
            rospy.logdebug("Classical feature detection - no goal image caching needed")
            return
        
        # Cache resized goal image
        rospy.logdebug("Caching goal image processing...")
        self.goal_image_resized = self.goal_image.resize((self.config.vit_input_size, self.config.vit_input_size))
        
        # Cache preprocessed tensor
        self.goal_tensor = self.feature_extractor.preprocess_pil(self.goal_image_resized)
        
        # Cache goal features
        with torch.no_grad():
            self.goal_features = self.feature_extractor.extract_descriptors(
                self.goal_tensor.to(self.feature_extractor.device),
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning
            )
        
        cache_time = (time.perf_counter() - cache_start) * 1000
        rospy.logdebug(f"Goal image processing cached in {cache_time:.2f}ms")
        rospy.logdebug(f"  - Resized image: {self.goal_image_resized.size}")
        rospy.logdebug(f"  - Tensor shape: {self.goal_tensor.shape}")
        rospy.logdebug(f"  - Features shape: {self.goal_features.shape}")

    # ---- release lines 348-405: _cache_goal_roi_features
    def _cache_goal_roi_features(self):
        """Cache goal ROI features for ROI-based detection."""
        if self.goal_roi_bbox is None:
            return

        cache_start = time.perf_counter()

        # Crop goal ROI
        goal_roi = self.yolo_detector.crop_roi(self.goal_image, self.goal_roi_bbox)
        
        # Note: We don't apply rotation to goal image since it's already at the desired orientation
        # Rotation compensation is only for aligning current image to goal image
        
        # Resize to VIT input size - force square for compatibility with ViT
        # NOTE: This causes aspect ratio distortion which affects visualization
        # The distortion is handled in the mapping functions
        self.goal_roi_resized = goal_roi.resize((self.config.vit_input_size, self.config.vit_input_size))

        # Store original ROI size for proper coordinate mapping
        self.goal_roi_original_size = goal_roi.size
        
        # Preprocess and cache tensor
        self.goal_roi_tensor = self.feature_extractor.preprocess_pil(self.goal_roi_resized)
        
        # Extract and cache features
        with torch.no_grad():
            self.goal_roi_features = self.feature_extractor.extract_descriptors(
                self.goal_roi_tensor.to(self.feature_extractor.device),
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning
            )

        # Cache ROI mask if needed
        if self.config.mask_path:
            # Load original mask
            mask_img = Image.open(self.config.mask_path).convert('L')
            
            # Crop mask to match goal ROI
            x1, y1, x2, y2 = self.goal_roi_bbox
            mask_roi = mask_img.crop((x1, y1, x2, y2))
            
            # Resize mask to VIT input size
            mask_roi_resized = mask_roi.resize(
                (self.config.vit_input_size, self.config.vit_input_size),
                Image.Resampling.LANCZOS
            )
            
            # Save temporary resized mask
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                mask_roi_resized.save(tmp.name, 'JPEG')
                self.goal_roi_mask_path = tmp.name
        
        cache_time = (time.perf_counter() - cache_start) * 1000
        rospy.logdebug(f"Goal ROI features cached in {cache_time:.2f}ms")
        rospy.logdebug(f"  - ROI bbox: {self.goal_roi_bbox}")
        rospy.logdebug(f"  - ROI features shape: {self.goal_roi_features.shape}")

    # ---- release lines 407-428: get_depth
    def get_depth(self, current_points):
        """Get depth values for the current feature points."""
        if self.latest_image_depth is None:
            rospy.logwarn("No depth image received yet")
            return None

        z_values_meter = np.zeros((len(current_points), 1))
        height, width = self.latest_image_depth.shape

        for count, point in enumerate(current_points):
            x, y = int(point[0]), int(point[1])

            # Ensure the point is within the image bounds
            if 0 <= x < width and 0 <= y < height:
                depth_value = self.latest_image_depth[y, x]
                # Convert depth value to meters
                z_values_meter[count] = depth_value / 1000.0 if depth_value != 0 else 100
            else:
                z_values_meter[count] = 100  # Out of bounds point
                print("!!!!!!!!!!!! Depth points out of bounds !!!!!!!!!!!!!!")

        return z_values_meter

    # ---- release lines 594-634: check_roi_tiling_switch
    def check_roi_tiling_switch(self):
        """Check if we should switch to ROI tiling mode."""
        # Check ROI tiling mode
        if (self.config.use_roi_detection and hasattr(self.config, 'use_roi_tiling')
            and self.config.use_roi_tiling and not self.roi_tiling_activated):

            # Don't check on iteration 0 - we need at least one iteration to establish baseline
            if self.vs_controller.iteration_count == 0:
                return

            # Check if we should use iteration-based switching
            if hasattr(self.config, 'tiling_switch_iteration') and self.config.tiling_switch_iteration > 0:
                # Iteration-based switching
                if self.vs_controller.iteration_count >= self.config.tiling_switch_iteration:
                    self.roi_tiling_activated = True
                    self.roi_tiling_switch_iteration = self.vs_controller.iteration_count
                    print(f"\n>>> SWITCHING TO ROI TILING MODE (iter {self.vs_controller.iteration_count}) - Iteration threshold reached\n")
                
                # Return early - we're using iteration-based, don't check error-based
                return
            
            # Feature-error-based switching (works in real world without ground truth)
            if (self.initial_feature_error is not None and
                self.initial_feature_error > 0 and
                self.current_feature_error is not None and
                self.vs_controller.iteration_count >= 10):  # Min iterations before switching

                # Calculate feature error reduction (0.0 = no reduction, 1.0 = 100% reduction)
                feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)

                # Get ROI tiling threshold (default 0.2 = switch at 80% reduction)
                # Check for both parameter names for backward compatibility
                roi_threshold = getattr(self.config, 'hybrid_switch_threshold',
                                      getattr(self.config, 'roi_tiling_switch_threshold', 0.2))

                # Switch when feature error has been reduced by the threshold amount
                if feature_error_reduction >= (1.0 - roi_threshold):
                    self.roi_tiling_activated = True
                    self.roi_tiling_switch_iteration = self.vs_controller.iteration_count
                    print(f"\n>>> SWITCHING TO ROI TILING MODE (iter {self.vs_controller.iteration_count}) - Feature error reduced by {feature_error_reduction*100:.1f}%")
                    print(f"    Initial error: {self.initial_feature_error:.2f} pixels, Current: {self.current_feature_error:.2f} pixels\n")

    # ---- release lines 636-681: check_tiling_switch
    def check_tiling_switch(self):
        """Check if we should switch to tiled mode for full image (MODE 2)."""
        # Only check if we haven't already switched and hybrid mode is enabled
        if self.tiling_activated or not self.config.use_hybrid_mode:
            return

        # Don't check on iteration 0 - we need at least one iteration to establish baseline
        if self.vs_controller.iteration_count == 0:
            return
        
        # Check if we should use iteration-based switching
        if hasattr(self.config, 'tiling_switch_iteration') and self.config.tiling_switch_iteration > 0:
            # Debug output on first check
            if self.vs_controller.iteration_count == 1:
                print(f"[Tiling Switch] Using iteration-based switching at iteration {self.config.tiling_switch_iteration}")
            
            # Iteration-based switching
            if self.vs_controller.iteration_count >= self.config.tiling_switch_iteration:
                self.tiling_activated = True
                self.tiling_switch_iteration = self.vs_controller.iteration_count
                print(f"\n>>> SWITCHING TO FULL IMAGE TILING MODE (iter {self.vs_controller.iteration_count}) - Iteration threshold reached\n")
            
            # Return early - we're using iteration-based, don't check error-based
            return
            
        # Feature-error-based switching (works in real world without ground truth)
        if self.vs_controller.iteration_count == 1:
            print(f"[Tiling Switch] Using feature-error-based switching (threshold: {(1-getattr(self.config, 'hybrid_switch_threshold', 0.2))*100:.0f}% reduction)")

        # Need initial feature error to calculate reduction
        if (self.initial_feature_error is not None and
            self.initial_feature_error > 0 and
            self.current_feature_error is not None):

            # Calculate feature error reduction (0.0 = no reduction, 1.0 = 100% reduction)
            feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)

            # Get hybrid switch threshold (default 0.2 = switch at 80% reduction)
            threshold = getattr(self.config, 'hybrid_switch_threshold', 0.2)

            # Switch when feature error has been reduced by the threshold amount
            if feature_error_reduction >= (1.0 - threshold):
                self.tiling_activated = True
                self.tiling_switch_iteration = self.vs_controller.iteration_count
                print(f"\n>>> SWITCHING TO FULL IMAGE TILING MODE (iter {self.vs_controller.iteration_count}) - Feature error reduced by {feature_error_reduction*100:.1f}%")
                print(f"    Initial error: {self.initial_feature_error:.2f} pixels, Current: {self.current_feature_error:.2f} pixels\n")

    # ---- release lines 683-748: detect_features
    def detect_features(self):
        """Main feature detection method."""
        # Performance tracking
        detect_start = time.perf_counter()

        # CRITICAL: Check for classical methods FIRST, before ROI detection branch
        # Classical methods (SIFT/ORB/AKAZE) have their own detection pipeline
        is_classical = hasattr(self.config, 'backbone_model') and 'classical-' in self.config.backbone_model

        if is_classical:
            # Classical methods - route to classical detection (with or without ROI)
            if self.config.use_roi_detection and self.yolo_detector is not None:
                result = self.detect_features_classical_roi()
            else:
                result = self.detect_features_classical()

            # Handle classical result
            if result is not None and result[0] is not None:
                self.consecutive_no_features_count = 0
            else:
                rospy.logwarn("[Classical] No features detected!")
                self.consecutive_no_features_count += 1
                if self.consecutive_no_features_count >= self.max_consecutive_no_features:
                    rospy.logerr(f"[Classical] No features for {self.consecutive_no_features_count} consecutive iterations - aborting")
                    raise RuntimeError("Persistent feature detection failure")

            total_time = (time.perf_counter() - detect_start) * 1000
            rospy.logdebug(f"[PERF] Total detect_features (classical): {total_time:.2f}ms")
            return result

        # ViT methods - existing logic unchanged
        # Check if we should switch to ROI tiling mode
        switch_start = time.perf_counter()
        self.check_roi_tiling_switch()
        switch_time = (time.perf_counter() - switch_start) * 1000
        if switch_time > 1:
            rospy.logdebug(f"[PERF] ROI switch check: {switch_time:.2f}ms")

        # Choose detection method for ViT
        if self.config.use_roi_detection and self.yolo_detector is not None:
            # Check if ROI tiling is enabled and we've reached the error threshold
            if (hasattr(self.config, 'use_roi_tiling') and self.config.use_roi_tiling
                and self.roi_tiling_activated):
                result = self.detect_features_roi_tiled()
            else:
                result = self.detect_features_roi()
        else:
            # MODE 2: Check for full image tiling with hybrid mode
            if (self.config.use_tiling and self.config.use_hybrid_mode):
                # Check if we should switch to tiled mode
                self.check_tiling_switch()

                if self.tiling_activated:
                    result = self.detect_features_tiled()
                else:
                    result = self.detect_features_original()
            elif self.config.use_tiling:
                # Tiling enabled but not hybrid - always use tiled
                result = self.detect_features_tiled()
            else:
                # No tiling - use original
                result = self.detect_features_original()

        total_time = (time.perf_counter() - detect_start) * 1000
        rospy.logdebug(f"[PERF] Total detect_features: {total_time:.2f}ms")
        return result

    # ---- release lines 750-854: detect_features_original
    def detect_features_original(self):
        """Original detect features method without tiling (ViT only).

        Note: Classical methods are handled at the top level of detect_features()
        and will never reach this method.
        """
        if self.latest_image is None:
            return None, None

        # Use cached goal image data - no need to resize or extract features!
        goal_image_resized = self.goal_image_resized
        desc1 = self.goal_features
        
        # Only process current image
        resize_start = time.perf_counter()
        current_image_resized = self.latest_pil_image.resize((self.config.vit_input_size, self.config.vit_input_size))
        resize_time = (time.perf_counter() - resize_start) * 1000
        rospy.logdebug(f"[PERF] Current image resize: {resize_time:.2f}ms (goal cached)")

        with torch.no_grad():
            # Only preprocess current image
            preprocess_start = time.perf_counter()
            current_tensor = self.feature_extractor.preprocess_pil(current_image_resized)
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000
            rospy.logdebug(f"[PERF] Current preprocessing: {preprocess_time:.2f}ms (goal cached)")

            # Only extract features for current image
            extract_start = time.perf_counter()
            desc2 = self.feature_extractor.extract_descriptors(
                current_tensor.to(self.feature_extractor.device),
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning
            )
            extract_time = (time.perf_counter() - extract_start) * 1000
            rospy.logdebug(f"[PERF] Current feature extraction: {extract_time:.2f}ms (goal cached)")
            
            # Check if features are on GPU
            if desc2.is_cuda:
                rospy.logdebug(f"[GPU] Features on GPU, shape: {desc2.shape}")

            # Profile feature matching
            match_start = time.perf_counter()
            rospy.logdebug(f"Calling find_correspondences_batch with mask_path: {self.config.mask_path}")
            points1, points2, sim_selected_12 = find_correspondences_batch(
                desc1, desc2,
                num_pairs=self.config.num_pairs,
                mask_path=self.config.mask_path,
                vit_input_size=self.config.vit_input_size,
                patch_size=self.feature_extractor.get_patch_size()
            )
            match_time = (time.perf_counter() - match_start) * 1000
            rospy.logdebug(f"[PERF] Feature matching: {match_time:.2f}ms")

            if points1 is None or points2 is None:
                self.feature_failure_count += 1
                self.last_num_features = 0  # No features found
                if self.feature_failure_count >= 10:
                    rospy.logerr("Feature detection failed 10 times in a row")
                    raise RuntimeError("Persistent feature detection failure")
                return None, None

            # Reset counter on successful detection
            self.feature_failure_count = 0
            
            # Store actual number of features found
            self.last_num_features = len(points1) if points1 is not None else 0

            # Check for GPU->CPU transfer  
            if torch.is_tensor(points1) and points1.is_cuda:
                size_mb = (points1.numel() + points2.numel()) * 4 / (1024 * 1024)
                rospy.logdebug(f"[GPU->CPU] Points will be transferred: {size_mb:.3f}MB")

            # Convert to VIT input space for visualization (points are in patch coordinates)
            scale = self.config.vit_input_size / int(np.sqrt(desc1.size(-2)))  
            patch_size = self.feature_extractor.get_patch_size()
            points1_vit = points1 * scale + patch_size / 2
            points2_vit = points2 * scale + patch_size / 2

            # Flip points from (x,y) to (y,x) for visualization
            # The visualization function expects points in (y,x) order
            if torch.is_tensor(points1_vit):
                points1_viz = torch.flip(points1_vit, dims=[1])
                points2_viz = torch.flip(points2_vit, dims=[1])
            else:
                points1_viz = np.flip(np.array(points1_vit), axis=1)
                points2_viz = np.flip(np.array(points2_vit), axis=1)

            # Visualize correspondences (only if visualization is enabled)
            if self.config.enable_visualization:
                visualize_correspondences_ros(
                    goal_image_resized, current_image_resized,
                    points1_viz, points2_viz,
                    None, self.bridge, self.correspondence_pub,
                    self.goal_image_pub, self.current_image_pub,
                    False, None  # No tiling
                )

            # Profile calculate_uv - pass VIT coordinates since calculate_uv expects VIT input size
            calc_start = time.perf_counter()
            result = self.calculate_uv(points1_vit.tolist(), points2_vit.tolist()), sim_selected_12
            calc_time = (time.perf_counter() - calc_start) * 1000
            if calc_time > 5:
                rospy.logdebug(f"[PERF] calculate_uv: {calc_time:.2f}ms")
            return result

    # ---- release lines 856-890: calculate_uv
    def calculate_uv(self, goal_features, current_features):
        """Calculate feature points and scale them to the real image resolution."""
        num_pairs = self.config.num_pairs

        # Profile numpy operations
        numpy_start = time.perf_counter()
        # Keep coordinates in (x,y) format - no flip needed as they match (u,v)
        s_uv_star_resized = np.asarray(goal_features)
        s_uv_resized = np.asarray(current_features)

        # Calculate the scaled feature values
        s_uv = np.zeros([num_pairs, 2], dtype=int)
        s_uv_star = np.zeros([num_pairs, 2], dtype=int)
        numpy_time = (time.perf_counter() - numpy_start) * 1000
        if numpy_time > 1:
            rospy.logdebug(f"[PERF] calculate_uv numpy ops: {numpy_time:.2f}ms")

        if len(goal_features) != num_pairs:
            rospy.logwarn("Number of extracted features doesn't match desired number of pairs")
            num_pairs = len(goal_features)
            if num_pairs < 4:
                rospy.logwarn("Too few features detected (<4). Skipping processing.")
                return s_uv_star, s_uv

        # Scale factors to convert from DINO input size to original image size
        scale_x = self.config.u_max / self.config.vit_input_size
        scale_y = self.config.v_max / self.config.vit_input_size

        for count in range(num_pairs):
            s_uv_star[count, 0] = round(s_uv_star_resized[count, 0] * scale_x)
            s_uv_star[count, 1] = round(s_uv_star_resized[count, 1] * scale_y)
            s_uv[count, 0] = round(s_uv_resized[count, 0] * scale_x)
            s_uv[count, 1] = round(s_uv_resized[count, 1] * scale_y)

        return s_uv_star, s_uv

    # ---- release lines 892-917: calculate_uv_roi
    def calculate_uv_roi(self, goal_features, current_features):
        """Calculate feature points for ROI mode - points are already in full image coordinates."""
        num_pairs = self.config.num_pairs

        
        # NOTE: No flip needed - points are already in (x,y) format which matches (u,v)
        # where x/u is horizontal and y/v is vertical
        s_uv_star = np.asarray(goal_features).astype(int)
        s_uv = np.asarray(current_features).astype(int)
        

        if len(goal_features) != num_pairs:
            rospy.logwarn("Number of extracted features doesn't match desired number of pairs")
            num_pairs = len(goal_features)
            if num_pairs < 4:
                rospy.logwarn("Too few features detected (<4). Skipping processing.")
                return np.zeros([0, 2], dtype=int), np.zeros([0, 2], dtype=int)

        # Points are already in full image coordinates, just ensure they're within bounds
        s_uv_star[:, 0] = np.clip(s_uv_star[:, 0], 0, self.config.u_max - 1)
        s_uv_star[:, 1] = np.clip(s_uv_star[:, 1], 0, self.config.v_max - 1)
        s_uv[:, 0] = np.clip(s_uv[:, 0], 0, self.config.u_max - 1)
        s_uv[:, 1] = np.clip(s_uv[:, 1], 0, self.config.v_max - 1)

        self.last_num_features = num_pairs  # Store actual number of features used
        return s_uv_star[:num_pairs], s_uv[:num_pairs]

    # ---- release lines 919-973: set_yolo_keyword
    def set_yolo_keyword(self, keyword):
        """Set the YOLO detection keyword for the current model."""
        if self.yolo_detector is not None:
            self.yolo_keyword = keyword
            self.yolo_detector.set_classes([keyword])
            
            # Reset current ROI cache when switching models
            self.current_roi_bbox = None
            self.last_valid_roi_iteration = 0
            
            # Detect and cache goal ROI
            detection = self.yolo_detector.detect_and_cache_goal(
                self.goal_image,
                keyword,
                self.config.roi_padding_ratio
            )

            if detection['bbox'] is not None:
                self.goal_roi_bbox = detection['padded_bbox']
                rospy.loginfo(f"[{self.detector_name} GOAL DETECTION] Goal image size: {self.goal_image.size}")
                rospy.loginfo(f"[{self.detector_name} GOAL DETECTION] Detected bbox: {detection['bbox']}")
                rospy.loginfo(f"[{self.detector_name} GOAL DETECTION] Padded bbox: {detection['padded_bbox']}")

                # Calculate what the bbox would be if scaled to 1440 or 1920
                goal_width = self.goal_image.size[0]
                if goal_width == 1920:
                    scale_to_1440 = 1440 / 1920
                    scaled_bbox = [
                        detection['padded_bbox'][0] * scale_to_1440,
                        detection['padded_bbox'][1],
                        detection['padded_bbox'][2] * scale_to_1440,
                        detection['padded_bbox'][3]
                    ]
                    rospy.loginfo(f"[{self.detector_name} GOAL DETECTION] If scaled to 1440: {scaled_bbox}")
                elif goal_width == 1440:
                    scale_to_1920 = 1920 / 1440
                    scaled_bbox = [
                        detection['padded_bbox'][0] * scale_to_1920,
                        detection['padded_bbox'][1],
                        detection['padded_bbox'][2] * scale_to_1920,
                        detection['padded_bbox'][3]
                    ]
                    rospy.loginfo(f"[{self.detector_name} GOAL DETECTION] If scaled to 1920: {scaled_bbox}")
                # Cache goal ROI features when ROI is detected
                self._cache_goal_roi_features()
                # Calculate optimal tile number if hybrid mode is enabled
                self._calculate_optimal_tile_number(detection['bbox'])
            else:
                rospy.logwarn(f"No detection in goal image for '{keyword}', will use full image")
                self.goal_roi_bbox = None
                # Set goal ROI features to use full image features as fallback
                self.goal_roi_features = self.goal_features
                self.goal_roi_resized = self.goal_image_resized
                self.goal_roi_tensor = self.goal_tensor
                self.goal_roi_mask_path = self.config.mask_path if hasattr(self.config, 'mask_path') else None

    # ---- release lines 1360-1571: detect_features_roi (the second definition - the one Python keeps)
    def detect_features_roi(self):
        """Detect features using ROI-based approach with YOLOWorld."""
        if self.latest_image is None or self.yolo_detector is None:
            return None, None
            
        # Profile YOLO detection
        yolo_start = time.perf_counter()
        current_detection = self.yolo_detector.detect(
            self.latest_pil_image,
            self.yolo_keyword,
            self.config.roi_padding_ratio
        )
        yolo_time = (time.perf_counter() - yolo_start) * 1000
        rospy.logdebug(f"[PERF] YOLO detection: {yolo_time:.2f}ms")
        
        # Update or use cached ROI
        if current_detection['bbox'] is not None:
            # Valid detection - update cache
            self.current_roi_bbox = current_detection['padded_bbox']
            self.last_valid_roi_iteration = self.vs_controller.iteration_count
            current_roi_bbox = self.current_roi_bbox
        else:
            # No detection - use cached ROI if available
            if self.current_roi_bbox is not None:
                current_roi_bbox = self.current_roi_bbox
                rospy.logwarn_throttle(5, f"No ROI detected, using cached ROI from iteration {self.last_valid_roi_iteration}")
            else:
                # No cached ROI available, fall back to full image
                rospy.logwarn_throttle(5, "No ROI detected and no cached ROI available, using full image")
                return self.detect_features_original()
        
        # Use cached goal ROI features
        goal_roi_bbox = self.goal_roi_bbox if self.goal_roi_bbox is not None else None
        
        # Get cached goal features and image
        if goal_roi_bbox is None:
            # No goal ROI detected - use full image features
            desc1 = self.goal_features
            goal_roi_resized = self.goal_image_resized
            roi_mask_path = self.config.mask_path if hasattr(self.config, 'mask_path') else None
            rospy.logwarn("No goal ROI detected, using full goal image features")
        else:
            desc1 = self.goal_roi_features
            goal_roi_resized = self.goal_roi_resized
            roi_mask_path = self.goal_roi_mask_path
        
        # Only need to extract current image features
        current_roi = self.yolo_detector.crop_roi(self.latest_pil_image, current_roi_bbox)
        # Store original size for proper coordinate mapping
        current_roi_original_size = current_roi.size

        # Profile current image processing
        current_process_start = time.perf_counter()

        # Resize current ROI to VIT input size (forces square, causing aspect ratio distortion)
        current_roi_resized = current_roi.resize((self.config.vit_input_size, self.config.vit_input_size))
        
        with torch.no_grad():
            # Extract features from current ROI only
            current_tensor = self.feature_extractor.preprocess_pil(current_roi_resized)
            
            desc2 = self.feature_extractor.extract_descriptors(
                current_tensor.to(self.feature_extractor.device),
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning
            )
        
        current_process_time = (time.perf_counter() - current_process_start) * 1000
        rospy.logdebug(f"[PERF] Current image processing: {current_process_time:.2f}ms (goal cached)")
        

        # Find correspondences
        rospy.logdebug(f"Calling find_correspondences_batch with mask_path: {roi_mask_path}")
        points1, points2, sim_selected_12 = find_correspondences_batch(
            desc1, desc2,
            num_pairs=self.config.num_pairs,
            mask_path=roi_mask_path,
            vit_input_size=self.config.vit_input_size,
            patch_size=self.feature_extractor.get_patch_size()
        )
        
        # ROI mask is cached, no need to clean up
        
        if points1 is None or points2 is None:
            self.feature_failure_count += 1
            if self.feature_failure_count >= 10:
                rospy.logerr("Feature detection failed 10 times in a row")
                raise RuntimeError("Persistent feature detection failure")
            return None, None
            
        # Reset counter on successful detection
        self.feature_failure_count = 0
        
        # Scale points from patch space to VIT input size
        # Get actual patch size from the model for correct coordinate scaling
        patch_size = self.feature_extractor.get_patch_size()
        num_patches_total = desc1.size(-2)
        num_patches_per_side = int(np.sqrt(num_patches_total))

        # Use the same calculation as the working old implementation
        scale = self.config.vit_input_size / num_patches_per_side


        points1_vit = points1 * scale + patch_size / 2
        points2_vit = points2 * scale + patch_size / 2

        
        # Visualize correspondences in VIT input space
        if getattr(self.config, 'enable_visualization', True):
            self._visualize_vit_space_correspondences(
                goal_roi_resized, current_roi_resized,
                points1_vit, points2_vit
            )

        # For visualization only - map to full image coordinates
        # ROIs are resized to square but we need to use ORIGINAL dimensions for proper mapping


        if goal_roi_bbox is not None:
            # Use the simple mapping function that assumes square resizing
            # This is correct because ViT always processes 224x224 images
            points1_full_viz = self._map_roi_to_full_image(points1_vit, goal_roi_bbox, None)
        else:
            # Goal points are already in full image coordinates
            points1_full_viz = points1_vit
        # For current ROI, also use the simple mapping
        points2_full_viz = self._map_roi_to_full_image(points2_vit, current_roi_bbox, None)

        
        # Visualize correspondences with ROI boxes
        # Points from regular ROI detection are in (x,y) format
        if getattr(self.config, 'enable_visualization', True):
            # DEBUG: Check if goal and current images have different dimensions
            if self.goal_image.size != self.latest_pil_image.size:
                rospy.logwarn(f"[VIZ DIMENSION ISSUE] Goal image: {self.goal_image.size}, Current image: {self.latest_pil_image.size}")
                rospy.logwarn(f"[VIZ DIMENSION ISSUE] Goal ROI bbox: {goal_roi_bbox}")
                rospy.logwarn(f"[VIZ DIMENSION ISSUE] Current ROI bbox: {current_roi_bbox}")

                # If dimensions mismatch, we need to scale the goal ROI bbox to match current image dimensions
                if goal_roi_bbox is not None:
                    goal_width, goal_height = self.goal_image.size
                    current_width, current_height = self.latest_pil_image.size

                    # Scale goal ROI bbox if widths are different
                    if goal_width != current_width:
                        scale_factor = current_width / goal_width
                        scaled_goal_roi_bbox = [
                            goal_roi_bbox[0] * scale_factor,  # x1
                            goal_roi_bbox[1],  # y1 (height is same)
                            goal_roi_bbox[2] * scale_factor,  # x2
                            goal_roi_bbox[3]   # y2 (height is same)
                        ]
                        rospy.loginfo(f"[VIZ FIX] Scaling goal ROI bbox by {scale_factor:.3f}")
                        rospy.loginfo(f"[VIZ FIX] Original goal ROI: {goal_roi_bbox}")
                        rospy.loginfo(f"[VIZ FIX] Scaled goal ROI: {scaled_goal_roi_bbox}")

                        # Also need to scale the goal points
                        points1_full_viz_scaled = points1_full_viz.clone() if torch.is_tensor(points1_full_viz) else points1_full_viz.copy()
                        if torch.is_tensor(points1_full_viz_scaled):
                            points1_full_viz_scaled[:, 0] *= scale_factor
                        else:
                            points1_full_viz_scaled[:, 0] *= scale_factor

                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            points1_full_viz_scaled, points2_full_viz,
                            scaled_goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                    else:
                        # Heights different but widths same - just use original
                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            points1_full_viz, points2_full_viz,
                            goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                else:
                    # No goal ROI bbox to scale
                    self._visualize_roi_correspondences(
                        self.goal_image, self.latest_pil_image,
                        points1_full_viz, points2_full_viz,
                        goal_roi_bbox, current_roi_bbox,
                        points_are_yx=False
                    )
            else:
                # Dimensions match - use original visualization
                self._visualize_roi_correspondences(
                    self.goal_image, self.latest_pil_image,
                    points1_full_viz, points2_full_viz,
                    goal_roi_bbox, current_roi_bbox,
                    points_are_yx=False
                )
        
        # Map points from VIT space to full image coordinates for servoing
        if goal_roi_bbox is not None:
            points1_full = self._map_roi_to_full_image(points1_vit, goal_roi_bbox, None)
        else:
            # Goal points are already in full image coordinates when no ROI
            # Scale from VIT input size to full image size
            scale_x = self.config.u_max / self.config.vit_input_size
            scale_y = self.config.v_max / self.config.vit_input_size
            points1_full = points1_vit.copy() if isinstance(points1_vit, np.ndarray) else points1_vit.cpu().numpy()
            points1_full[:, 0] *= scale_x
            points1_full[:, 1] *= scale_y
        # For current ROI servoing, use the simple mapping
        points2_full = self._map_roi_to_full_image(points2_vit, current_roi_bbox, None)
        
        
        # Use custom calculate_uv for ROI mode that doesn't double-scale
        return self.calculate_uv_roi(points1_full.tolist(), points2_full.tolist()), sim_selected_12

    # ---- release lines 1573-1982: detect_features_roi_tiled
    def detect_features_roi_tiled(self):
        """Detect features using ROI-based tiling approach."""
        if self.latest_image is None or self.yolo_detector is None:
            return None, None
            
        # Profile YOLO detection
        yolo_start = time.perf_counter()
        current_detection = self.yolo_detector.detect(
            self.latest_pil_image,
            self.yolo_keyword,
            self.config.roi_padding_ratio
        )
        yolo_time = (time.perf_counter() - yolo_start) * 1000
        rospy.logdebug(f"[PERF] YOLO detection (ROI tiled): {yolo_time:.2f}ms")
        
        # Update or use cached ROI
        if current_detection['bbox'] is not None:
            # Valid detection - update cache
            self.current_roi_bbox = current_detection['padded_bbox']
            self.last_valid_roi_iteration = self.vs_controller.iteration_count
            current_roi_bbox = self.current_roi_bbox
        else:
            # No detection - use cached ROI if available
            if self.current_roi_bbox is not None:
                current_roi_bbox = self.current_roi_bbox
                rospy.logwarn_throttle(5, f"[Tiled] No ROI detected, using cached ROI from iteration {self.last_valid_roi_iteration}")
            else:
                # No cached ROI available, fall back to full image with tiling
                rospy.logwarn_throttle(5, "[Tiled] No ROI detected and no cached ROI available, using full image with tiling")
                return self.detect_features_tiled()
        
        # Use cached goal ROI or fail for tiling mode
        goal_roi_bbox = self.goal_roi_bbox if self.goal_roi_bbox is not None else None
        
        # ROI tiling REQUIRES both bounding boxes
        if goal_roi_bbox is None:
            rospy.logerr("[ROI Tiled] FATAL: No goal ROI detected - ROI tiling mode requires object detection in goal image!")
            rospy.logerr(f"[ROI Tiled] Failed to detect '{self.yolo_keyword}' in goal image. Cannot use ROI tiling mode.")
            # Try to fall back to regular ROI mode
            rospy.logwarn("[ROI Tiled] Attempting fallback to regular ROI mode...")
            return self.detect_features_roi()
        
        # Crop ROIs
        goal_roi = self.yolo_detector.crop_roi(self.goal_image, goal_roi_bbox)
        rospy.logdebug(f"[ROI Tiled] Using goal ROI bbox: {goal_roi_bbox}")
            
        current_roi = self.yolo_detector.crop_roi(self.latest_pil_image, current_roi_bbox)
        
        # Create ROI tiling configuration
        # Use optimal tile number if calculated, otherwise use config value
        roi_tile_number = self.get_effective_tile_number()

        # Get rectangular grid dimensions (supports non-square grids like 4x1)
        if hasattr(self, 'optimal_grid_width') and hasattr(self, 'optimal_grid_height'):
            grid_width = self.optimal_grid_width
            grid_height = self.optimal_grid_height
        else:
            # Fallback to square grid
            grid_size = int(np.sqrt(roi_tile_number))
            grid_width = grid_height = grid_size

        # Only log the grid configuration once when tiling is first activated or grid changes
        if not hasattr(self, '_last_logged_grid') or self._last_logged_grid != (grid_width, grid_height):
            rospy.logdebug(f"[ROI Tiled] Using {grid_width}x{grid_height} grid ({roi_tile_number} tiles)")
            self._last_logged_grid = (grid_width, grid_height)

        # Resize ROI to rectangular dimensions that match the grid
        # Calculate total size for each dimension
        total_width = self.config.vit_input_size * grid_width
        total_height = self.config.vit_input_size * grid_height
        tile_size = self.config.vit_input_size
        
        # Profile ROI resize for tiling
        roi_resize_start = time.perf_counter()
        goal_roi_resized = goal_roi.resize((total_width, total_height))
        current_roi_resized = current_roi.resize((total_width, total_height))
        roi_resize_time = (time.perf_counter() - roi_resize_start) * 1000
        rospy.logdebug(f"[PERF] ROI resize to {total_width}x{total_height}: {roi_resize_time:.2f}ms")
        
        # Handle mask for ROI tiling mode
        mask_resized = None
        if self.config.mask_path:
            # Try to load object-specific mask first
            mask_path = self.config.mask_path
            if hasattr(self, 'yolo_keyword') and self.yolo_keyword:
                # Check if object-specific mask exists
                import os
                base_dir = os.path.dirname(mask_path)
                object_mask_path = os.path.join(base_dir, f"mask_{self.yolo_keyword}.jpg")
                if os.path.exists(object_mask_path):
                    mask_path = object_mask_path
            
            # Load mask
            mask_img = Image.open(mask_path).convert('L')
            
            if goal_roi_bbox is not None:
                # Crop mask to match goal ROI
                x1, y1, x2, y2 = [int(coord) for coord in goal_roi_bbox]
                mask_cropped = mask_img.crop((x1, y1, x2, y2))
            else:
                # No ROI, use full mask
                mask_cropped = mask_img
            
            # Resize mask to match the resized ROI
            mask_resized = mask_cropped.resize((total_width, total_height))
        
        
        all_points1 = []
        all_points2 = []
        all_sims = []
        
        # Store VIT space data for visualization
        vit_tiles_data = []
        
        # Profile tile processing
        tile_process_start = time.perf_counter()
        total_tiles = grid_width * grid_height
        rospy.logdebug(f"[PERF] Processing {total_tiles} tiles ({grid_width}x{grid_height} grid) - BATCH MODE")
        
        # First, extract all tiles
        goal_tiles = []
        current_tiles = []
        mask_tiles = []
        tile_positions = []
        
        extract_tiles_start = time.perf_counter()
        # Process each tile in the rectangular grid
        for i in range(grid_height):  # Rows
            for j in range(grid_width):  # Columns
                tile_x = j * tile_size
                tile_y = i * tile_size
                
                # Extract tiles
                goal_tile = goal_roi_resized.crop((
                    tile_x, tile_y,
                    tile_x + tile_size, tile_y + tile_size
                ))
                current_tile = current_roi_resized.crop((
                    tile_x, tile_y,
                    tile_x + tile_size, tile_y + tile_size
                ))
                
                goal_tiles.append(goal_tile)
                current_tiles.append(current_tile)
                tile_positions.append((i, j))
                
                # Extract mask tile if available
                if mask_resized is not None:
                    mask_tile = mask_resized.crop((
                        tile_x, tile_y,
                        tile_x + tile_size, tile_y + tile_size
                    ))
                    mask_tiles.append(mask_tile)
        
        extract_tiles_time = (time.perf_counter() - extract_tiles_start) * 1000
        rospy.logdebug(f"[PERF] Extract all tiles: {extract_tiles_time:.2f}ms")
        
        # Batch preprocess all tiles
        preprocess_start = time.perf_counter()
        goal_tensors = []
        current_tensors = []
        
        for goal_tile, current_tile in zip(goal_tiles, current_tiles):
            goal_tensor = self.feature_extractor.preprocess_pil(goal_tile)
            current_tensor = self.feature_extractor.preprocess_pil(current_tile)
            goal_tensors.append(goal_tensor)
            current_tensors.append(current_tensor)
        
        # Stack into batches
        goal_batch = torch.cat(goal_tensors, dim=0).to(self.feature_extractor.device)
        current_batch = torch.cat(current_tensors, dim=0).to(self.feature_extractor.device)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        rospy.logdebug(f"[PERF] Preprocess all tiles: {preprocess_time:.2f}ms")
        
        # Batch extract features
        with torch.no_grad():
            extract_start = time.perf_counter()
            # Extract features for all tiles at once
            desc1_batch = self.feature_extractor.extract_descriptors(
                goal_batch,
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning
            )
            desc2_batch = self.feature_extractor.extract_descriptors(
                current_batch,
                layer=11,
                facet='token',
                bin=self.config.use_feature_binning
            )
            extract_time = (time.perf_counter() - extract_start) * 1000
            rospy.logdebug(f"[PERF] Batch feature extraction ({total_tiles} tiles): {extract_time:.2f}ms")
        
        # Pre-compute mask tensors for all tiles to avoid repeated computation
        mask_tensors = []
        if mask_tiles:
            from features.matcher import create_patch_mask_from_pil
            # Calculate actual patch size from descriptor dimensions
            actual_patch_size = self.config.vit_input_size // int(np.sqrt(desc1_batch[0].size(-2)))
            
            mask_compute_start = time.perf_counter()
            for mask_tile in mask_tiles:
                mask_tensor = create_patch_mask_from_pil(
                    mask_tile, 
                    vit_input_size=self.config.vit_input_size,
                    patch_size=actual_patch_size
                ).to(self.feature_extractor.device)
                mask_tensors.append(mask_tensor)
            mask_compute_time = (time.perf_counter() - mask_compute_start) * 1000
            rospy.logdebug(f"[PERF] Mask tensor computation: {mask_compute_time:.2f}ms")
        
        # Process correspondences for each tile
        correspondence_start = time.perf_counter()
        for tile_idx in range(total_tiles):
            i, j = tile_positions[tile_idx]
            
            # Extract features for this tile from the batch
            desc1 = desc1_batch[tile_idx:tile_idx+1]
            desc2 = desc2_batch[tile_idx:tile_idx+1]
            
            # Use pre-computed mask tensor if available
            tile_mask_tensor = mask_tensors[tile_idx] if tile_idx < len(mask_tensors) else None
            
            # Find correspondences for this tile with mask
            # Use actual number of tiles (grid_width * grid_height) instead of configured tile number
            # since rectangular grids may have different tile counts than expected
            # Ensure at least 1 pair per tile
            pairs_per_tile = max(1, self.config.num_pairs // total_tiles)

            # Log which tile we're processing
            rospy.logdebug(f"[ROI Tiled] Processing tile ({i},{j}) - requesting {pairs_per_tile} pairs")

            points1, points2, sim = find_correspondences_batch(
                desc1, desc2,
                num_pairs=pairs_per_tile,  # Distribute pairs across actual tile count
                mask_tensor=tile_mask_tensor,  # Use pre-computed mask tensor
                vit_input_size=self.config.vit_input_size,
                patch_size=self.feature_extractor.get_patch_size()
            )

            if points1 is not None and points2 is not None:
                rospy.logdebug(f"[ROI Tiled] Tile ({i},{j}) SUCCESS: Got {len(points1)} correspondences")
                # Get tile position for offset calculation
                tile_x = j * tile_size
                tile_y = i * tile_size

                # Points from find_correspondences_batch are in [x, y] format
                # Scale points from patch space to VIT input size
                # Get actual patch size from the model for correct coordinate scaling
                patch_size = self.feature_extractor.get_patch_size()
                scale = self.config.vit_input_size / int(np.sqrt(desc1.size(-2)))
                points1_vit = points1 * scale + patch_size / 2
                points2_vit = points2 * scale + patch_size / 2

                
                # Store for VIT space visualization
                vit_tiles_data.append({
                    'tile_idx': (i, j),
                    'goal_tile': goal_tiles[tile_idx],
                    'current_tile': current_tiles[tile_idx],
                    'points1_vit': points1_vit.clone() if torch.is_tensor(points1_vit) else points1_vit.copy(),
                    'points2_vit': points2_vit.clone() if torch.is_tensor(points2_vit) else points2_vit.copy()
                })
                
                # Map from tile VIT space to resized ROI coordinates
                # Since tiles are already at VIT input size, no scaling needed for tile coordinates
                # Just add the tile offset within the resized ROI
                points1_resized_roi = torch.zeros_like(points1_vit)
                points1_resized_roi[:, 0] = tile_x + points1_vit[:, 0]  # x
                points1_resized_roi[:, 1] = tile_y + points1_vit[:, 1]  # y

                points2_resized_roi = torch.zeros_like(points2_vit)
                points2_resized_roi[:, 0] = tile_x + points2_vit[:, 0]  # x
                points2_resized_roi[:, 1] = tile_y + points2_vit[:, 1]  # y

                
                # Step 3: Map from resized ROI coordinates to original ROI coordinates
                # Then to full image coordinates
                if goal_roi_bbox is not None:
                    goal_x1, goal_y1, goal_x2, goal_y2 = goal_roi_bbox
                    goal_roi_width = goal_x2 - goal_x1
                    goal_roi_height = goal_y2 - goal_y1
                    
                    # Scale from resized ROI (total_width x total_height) to original ROI size
                    scale_x_goal = goal_roi_width / total_width
                    scale_y_goal = goal_roi_height / total_height
                    
                    points1_full = torch.zeros_like(points1_vit)
                    points1_full[:, 0] = goal_x1 + points1_resized_roi[:, 0] * scale_x_goal
                    points1_full[:, 1] = goal_y1 + points1_resized_roi[:, 1] * scale_y_goal
                else:
                    # If no ROI, scale directly to full image dimensions
                    points1_full = torch.zeros_like(points1_vit)
                    points1_full[:, 0] = points1_resized_roi[:, 0] * (self.config.u_max / total_width)
                    points1_full[:, 1] = points1_resized_roi[:, 1] * (self.config.v_max / total_height)
                    
                    # Debug logging for no goal bbox case
                    if tile_idx == 0:  # Only log once
                        rospy.logwarn(f"[ROI Tiled] No goal bbox - scaling from {total_width}x{total_height} to {self.config.u_max}x{self.config.v_max}")
                    
                current_x1, current_y1, current_x2, current_y2 = current_roi_bbox
                current_roi_width = current_x2 - current_x1
                current_roi_height = current_y2 - current_y1
                
                # Scale from resized ROI (total_width x total_height) to original ROI size
                scale_x_current = current_roi_width / total_width
                scale_y_current = current_roi_height / total_height
                
                points2_full = torch.zeros_like(points2_vit)
                points2_full[:, 0] = current_x1 + points2_resized_roi[:, 0] * scale_x_current
                points2_full[:, 1] = current_y1 + points2_resized_roi[:, 1] * scale_y_current
                
                all_points1.append(points1_full)
                all_points2.append(points2_full)
                all_sims.append(sim)
            else:
                # Log which tile failed to produce correspondences
                rospy.logerr(f"[ROI Tiled] ERROR: Tile ({i},{j}) FAILED - no correspondences returned! (requested {pairs_per_tile} pairs)")
                rospy.logerr(f"[ROI Tiled] This likely means no valid correspondences passed the distance threshold and mask filters. Enable debug logging to see details.")

        # Log correspondence processing time
        correspondence_time = (time.perf_counter() - correspondence_start) * 1000
        rospy.logdebug(f"[PERF] Correspondence processing: {correspondence_time:.2f}ms")
        
        # Total tile processing time
        total_tile_time = (time.perf_counter() - tile_process_start) * 1000
        rospy.logdebug(f"[PERF] Total ROI tile processing: {total_tile_time:.2f}ms")
        
        # Combine all points from all tiles
        if len(all_points1) > 0:
            combined_points1 = torch.cat(all_points1, dim=0)
            combined_points2 = torch.cat(all_points2, dim=0)
            combined_sims = torch.cat(all_sims, dim=0)

            # Log if we got fewer correspondences than expected
            if len(combined_points1) < self.config.num_pairs:
                rospy.logdebug(f"[ROI Tiled] Found {len(combined_points1)} correspondences from {total_tiles} tiles (requested {self.config.num_pairs})")

            # Visualize all tiles at VIT space
            if vit_tiles_data and getattr(self.config, 'enable_visualization', True):
                # Pass both dimensions to visualization function
                self._visualize_vit_space_tiled_grid(vit_tiles_data, grid_width, grid_height)


            # Select top matches based on similarity
            num_pairs = min(self.config.num_pairs, len(combined_points1))
            if len(combined_points1) > num_pairs:
                top_indices = torch.topk(combined_sims, num_pairs).indices
                final_points1 = combined_points1[top_indices]
                final_points2 = combined_points2[top_indices]
                final_sims = combined_sims[top_indices]
            else:
                final_points1 = combined_points1
                final_points2 = combined_points2
                final_sims = combined_sims
            
            
            # Visualize correspondences with ROI boxes
            # Points from ROI tiling are in (x,y) format, need to tell visualization function
            if getattr(self.config, 'enable_visualization', True):
                # Check if goal and current images have different dimensions (same fix as regular ROI)
                if self.goal_image.size != self.latest_pil_image.size:
                    goal_width, goal_height = self.goal_image.size
                    current_width, current_height = self.latest_pil_image.size

                    if goal_width != current_width and goal_roi_bbox is not None:
                        scale_factor = current_width / goal_width
                        scaled_goal_roi_bbox = [
                            goal_roi_bbox[0] * scale_factor,
                            goal_roi_bbox[1],
                            goal_roi_bbox[2] * scale_factor,
                            goal_roi_bbox[3]
                        ]
                        # Scale goal points too
                        final_points1_scaled = final_points1.clone() if torch.is_tensor(final_points1) else final_points1.copy()
                        if torch.is_tensor(final_points1_scaled):
                            final_points1_scaled[:, 0] *= scale_factor
                        else:
                            final_points1_scaled[:, 0] *= scale_factor

                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            final_points1_scaled, final_points2,
                            scaled_goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                    else:
                        self._visualize_roi_correspondences(
                            self.goal_image, self.latest_pil_image,
                            final_points1, final_points2,
                            goal_roi_bbox, current_roi_bbox,
                            points_are_yx=False
                        )
                else:
                    # Dimensions match - use original
                    self._visualize_roi_correspondences(
                        self.goal_image, self.latest_pil_image,
                        final_points1, final_points2,
                        goal_roi_bbox, current_roi_bbox,
                        points_are_yx=False  # ROI tiling coordinates are in (x,y) format
                    )

            # Store actual number of features found
            self.last_num_features = len(final_points1)

            return self.calculate_uv_roi(final_points1.tolist(), final_points2.tolist()), final_sims
        else:
            rospy.logwarn("No correspondences found in any tile")
            self.last_num_features = 0
            return None, None

    # ---- release lines 2140-2171: _map_roi_to_full_image
    def _map_roi_to_full_image(self, points_roi, roi_bbox, roi_size):
        """Map points from ROI coordinates to full image coordinates.

        This handles the case where ROI was resized to square (224x224) for ViT processing.
        Points are in VIT space (224x224) and need to be mapped back to original ROI dimensions.
        """
        if roi_bbox is None:
            # No ROI, points are already in full image space (scaled)
            scale_x = self.config.u_max / self.config.vit_input_size
            scale_y = self.config.v_max / self.config.vit_input_size
            points_full = points_roi.clone()
            points_full[:, 0] *= scale_x
            points_full[:, 1] *= scale_y
            return points_full

        # ROI dimensions in the original image
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1

        # Points are in VIT space (224x224), need to map to original ROI dimensions
        # The ROI was resized from (roi_width x roi_height) to (224 x 224)
        # So we need to scale back by the inverse of that transformation
        scale_x = roi_width / self.config.vit_input_size
        scale_y = roi_height / self.config.vit_input_size

        # Map to full image coordinates
        points_full = torch.zeros_like(points_roi)
        points_full[:, 0] = roi_x1 + (points_roi[:, 0] * scale_x)
        points_full[:, 1] = roi_y1 + (points_roi[:, 1] * scale_y)

        return points_full

    # ---- release lines 2472-2594: ibvs
    def ibvs(self):
        """Image Based Visual Servoing Method."""
        if self.latest_image is None:
            print("NO LATEST IMAGE SKIPPING")
            return

        start_time = time.time()

        # Get features with error handling
        result = self.detect_features()
        if result is None or result[0] is None:
            rospy.logwarn("Feature detection failed - skipping this iteration")
            return

        (s_uv_star, s_uv), sim_selected_12 = result
        if s_uv_star is None or s_uv is None or len(s_uv_star) < 4:
            rospy.logwarn("Insufficient features detected - skipping this iteration")
            return

        # Calculate feature error in pixel space for real-world switching
        # This is the average pixel distance between current and goal features
        feature_errors = np.linalg.norm(s_uv - s_uv_star, axis=1)  # Pixel distances for each feature
        self.current_feature_error = np.mean(feature_errors)  # Average pixel error

        # Store initial feature error on first iteration (after first successful feature detection)
        if self.initial_feature_error is None and self.vs_controller.iteration_count <= 1:
            self.initial_feature_error = self.current_feature_error
            rospy.loginfo(f"[Feature Error] Initial average pixel error: {self.initial_feature_error:.2f} pixels")

        # Log feature error reduction progress every 50 iterations
        if self.initial_feature_error is not None and self.initial_feature_error > 0:
            if self.vs_controller.iteration_count % 50 == 0:
                feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)
                rospy.loginfo(f"[Feature Error] Iter {self.vs_controller.iteration_count}: "
                            f"{self.current_feature_error:.2f} pixels ({feature_error_reduction*100:.1f}% reduction)")

        # Draw points only if visualization is enabled
        if getattr(self.config, 'enable_visualization', True):
            self.draw_points(np.array(self.latest_pil_image), s_uv, s_uv_star)

        # Transform feature points to real-world coordinates
        s_xy, s_star_xy = self.vs_controller.transform_to_real_world(s_uv, s_uv_star)

        # Get depth
        Z = self.get_depth(s_uv)
        if Z is None:
            rospy.logwarn("Failed to get depth - skipping this iteration")
            return

        # Compute control velocity
        v_c = self.vs_controller.compute_control_velocity(s_xy, s_star_xy, Z)
        
        # Update iteration statistics
        self.vs_controller.update_iteration_stats(v_c)

        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store iteration time for FPS calculation
        self.vs_controller.iteration_times.append(execution_time)
        # Get current pose error
        if hasattr(self, 'camera_position') and hasattr(self, 'desired_position') and self.camera_position is not None and self.desired_position is not None:
            position_error = np.linalg.norm(self.camera_position - self.desired_position)
            position_error_cm = position_error * 100  # Convert to cm
            
            # Calculate rotation error in degrees
            if hasattr(self, 'orientation_quaternion') and hasattr(self, 'desired_orientation') and self.orientation_quaternion is not None and self.desired_orientation is not None:
                from scipy.spatial.transform import Rotation
                R_current = Rotation.from_quat(self.orientation_quaternion)
                R_desired = Rotation.from_quat(self.desired_orientation)
                R_error = R_desired.inv() * R_current
                rotation_error_deg = np.linalg.norm(R_error.as_rotvec()) * 180 / np.pi
            else:
                rotation_error_deg = 0.0
            
            # Get mode and bbox info
            if self.roi_tiling_activated:
                mode = "ROI-Tiled"
                # Add tile number and VIT size info
                tile_num = self.get_effective_tile_number()
                mode += f" T{tile_num}"
            elif self.tiling_activated:
                mode = "Tiled"
                # Add tile number info - use the effective tile number
                effective_tiles = self._get_full_image_tile_number()
                mode += f" T{effective_tiles}"
                # Add asterisk if auto-calculated
                if not (hasattr(self.config, 'auto_calculate_tiles') and not self.config.auto_calculate_tiles):
                    mode += "*"  # Indicates auto-calculated
            elif self.config.use_roi_detection:
                mode = "ROI"
            else:
                mode = "Original"
            
            # Get current bbox dimensions if available
            bbox_info = ""
            if self.config.use_roi_detection and self.current_roi_bbox is not None:
                x1, y1, x2, y2 = self.current_roi_bbox
                width = int(x2 - x1)
                height = int(y2 - y1)
                bbox_info = f" | BBox: {width}x{height}"
            
            # Add feature count and VIT size to output
            feature_info = f" | F: {self.last_num_features}/{self.config.num_pairs}"
            vit_info = f" | VIT: {self.config.vit_input_size}"

            # Add feature error info (real-world observable metric)
            feature_error_info = ""
            if self.initial_feature_error is not None and self.initial_feature_error > 0 and self.current_feature_error is not None:
                feature_error_reduction = 1.0 - (self.current_feature_error / self.initial_feature_error)
                feature_error_info = f" | FeatErr: {self.current_feature_error:.1f}px ({feature_error_reduction*100:.0f}%↓)"

            # Check if we should display this iteration (respecting display frequency)
            display_freq = getattr(self.config, 'iteration_display_freq', 1)
            if self.vs_controller.iteration_count % display_freq == 0 or self.vs_controller.iteration_count == 1:
                print(f"Iter: {self.vs_controller.iteration_count:4d} | Time: {execution_time:4.2f}s | Trans: {position_error_cm:5.2f}cm | Rot: {rotation_error_deg:5.2f}° | Mode: {mode}{bbox_info}{feature_info}{vit_info}{feature_error_info}")
        else:
            # Simple output when error info not available
            display_freq = getattr(self.config, 'iteration_display_freq', 1)
            if self.vs_controller.iteration_count % display_freq == 0 or self.vs_controller.iteration_count == 1:
                print(f"Iter: {self.vs_controller.iteration_count:4d} | Time: {execution_time:4.2f}s")
        
        return v_c

    # ---- release lines 2781-2837: _calculate_optimal_tile_number
    def _calculate_optimal_tile_number(self, bbox):
        """
        Calculate optimal tile grid dimensions based on ROI bounding box.
        Supports rectangular grids (e.g., 4x2) for better resolution utilization.

        Args:
            bbox: Tuple (x1, y1, x2, y2) representing the bounding box
        """
        rospy.loginfo(f"[Auto Tile Calculation] Called with bbox: {bbox}")

        # Only calculate if hybrid mode is enabled and auto calculation is not disabled
        if not getattr(self.config, 'use_hybrid_mode', False):
            rospy.loginfo("[Auto Tile Calculation] Skipped - use_hybrid_mode is False")
            return

        # Check if auto calculation is explicitly disabled
        if hasattr(self.config, 'auto_calculate_tiles') and not self.config.auto_calculate_tiles:
            rospy.loginfo("Automatic tile calculation is disabled by config")
            return

        # Extract bbox dimensions
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Get VIT input size
        vit_input_size = self.config.vit_input_size

        # Calculate tiles needed for each dimension independently
        import math
        tiles_width = math.ceil(bbox_width / vit_input_size)
        tiles_height = math.ceil(bbox_height / vit_input_size)

        # Apply constraints per dimension
        min_tiles_per_dim = 1
        max_tiles_per_dim = 5  # Maximum to prevent memory issues

        tiles_width = max(min_tiles_per_dim, min(tiles_width, max_tiles_per_dim))
        tiles_height = max(min_tiles_per_dim, min(tiles_height, max_tiles_per_dim))

        # Store rectangular grid dimensions
        self.optimal_grid_width = tiles_width
        self.optimal_grid_height = tiles_height
        self.optimal_tile_number = tiles_width * tiles_height
        
        # Calculate coverage efficiency with rectangular grid
        coverage_width_pixels = tiles_width * vit_input_size
        coverage_height_pixels = tiles_height * vit_input_size
        coverage_width_ratio = coverage_width_pixels / bbox_width
        coverage_height_ratio = coverage_height_pixels / bbox_height

        rospy.loginfo(f"[Auto Tile Calculation] ROI bbox: {bbox_width:.0f}x{bbox_height:.0f}, VIT size: {vit_input_size}")
        rospy.loginfo(f"[Auto Tile Calculation] Optimal grid: {tiles_width}x{tiles_height} = {self.optimal_tile_number} tiles")
        rospy.loginfo(f"[Auto Tile Calculation] Coverage: {coverage_width_pixels}x{coverage_height_pixels} pixels ({coverage_width_ratio:.1f}x width, {coverage_height_ratio:.1f}x height)")
        
        # Note: This overrides config.tile_number when use_hybrid_mode=true AND use_roi_detection=true
        # To disable auto calculation, set auto_calculate_tiles=false in config

    # ---- release lines 2839-2848: get_effective_tile_number
    def get_effective_tile_number(self):
        """
        Get the effective tile number to use.
        Returns optimal_tile_number if calculated, otherwise config.tile_number.
        
        This method is called when ROI tiling is active to determine the tile grid size.
        """
        if self.optimal_tile_number is not None:
            return self.optimal_tile_number
        return getattr(self.config, 'roi_tile_number', 4)

    # ---- release lines 2890-2901: __del__
    def __del__(self):
        """Cleanup temporary files on object destruction."""
        # Clean up temporary ROI mask file if it exists
        if hasattr(self, 'goal_roi_mask_path') and self.goal_roi_mask_path:
            if self.goal_roi_mask_path != getattr(self.config, 'mask_path', None):
                try:
                    import os
                    if os.path.exists(self.goal_roi_mask_path):
                        os.unlink(self.goal_roi_mask_path)
                        rospy.loginfo(f"Cleaned up temporary mask file: {self.goal_roi_mask_path}")
                except Exception as e:
                    rospy.logwarn(f"Failed to clean up temporary mask: {e}")


# TRANSCRIPTION: the tiled method's in-body import is served while it runs (see the header)
ROSVisualServoingController.detect_features_roi_tiled = _release_imports(
    ROSVisualServoingController.detect_features_roi_tiled)
