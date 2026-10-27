"""Assemble tests/fixtures/reference/artvs_release/roi_glue.py from the release's lines."""
import os
import sys

RELEASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[".."] * 5, "ART-VS")
SRC = os.path.join(RELEASE, "catkin_ws/ibvs/src/visual_servoing/ros_interface/ros_controller.py")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_glue.py")
lines = open(SRC).read().split("\n")
def rng(a, b):
    return lines[a - 1:b]

KEEP = [  # (first, last, what) - 1-based inclusive line numbers of the release file
    (317, 346, "_cache_goal_image_processing"),
    (348, 405, "_cache_goal_roi_features"),
    (407, 428, "get_depth"),
    (594, 634, "check_roi_tiling_switch"),
    (636, 681, "check_tiling_switch"),
    (683, 748, "detect_features"),
    (750, 854, "detect_features_original"),
    (856, 890, "calculate_uv"),
    (892, 917, "calculate_uv_roi"),
    (919, 973, "set_yolo_keyword"),
    (1360, 1571, "detect_features_roi (the second definition - the one Python keeps)"),
    (1573, 1982, "detect_features_roi_tiled"),
    (2140, 2171, "_map_roi_to_full_image"),
    (2472, 2594, "ibvs"),
    (2781, 2837, "_calculate_optimal_tile_number"),
    (2839, 2848, "get_effective_tile_number"),
    (2890, 2901, "__del__"),
]

header = '''# Transcribed from the ART-VS release (git 9b1208c, MIT, "Alessandro Scherl and the ART-VS
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
'''
body = header.split("\n")
body += rng(2603, 2643)
body += ['''
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
''']
for a, b, what in KEEP:
    body += ["", f"    # ---- release lines {a}-{b}: {what}"]
    body += rng(a, b)
body += ["", "",
         "# TRANSCRIPTION: the tiled method's in-body import is served while it runs (see the header)",
         "ROSVisualServoingController.detect_features_roi_tiled = _release_imports(",
         "    ROSVisualServoingController.detect_features_roi_tiled)"]
text = "\n".join(body) + "\n"
open(OUT, "w").write(text)
print("written", OUT, len(text.splitlines()), "lines")
