"""Assemble tests/fixtures/reference/artvs_release/detector_state.py from the release's lines."""
import os
import sys

RELEASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[".."] * 5, "ART-VS")
SRC = os.path.join(RELEASE, "catkin_ws/ibvs/src/visual_servoing/features/langsam_lighttrack_detector.py")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_state.py")
lines = open(SRC).read().split("\n")
def rng(a, b):                      # 1-based inclusive
    return lines[a - 1:b]
header = '''# Transcribed from the ART-VS release (git 9b1208c, MIT, "Alessandro Scherl and the ART-VS
# authors"): catkin_ws/ibvs/src/visual_servoing/features/langsam_lighttrack_detector.py.
# The methods below are the release's lines, unchanged, except two injections (marked
# "TRANSCRIPTION:"): LangSAM's `self.langsam.predict` is `self.predict` (an injected callable
# with LangSAM's signature and result shape), and `LightTrack(...)` is
# `self.tracker_factory(...)` (an injected class with the wrapper's interface). The imports
# and the constructor (lines 1-177: the lang_sam import, the LightTrack path discovery, the
# model loading) are replaced by the minimal constructor below, which sets the same state
# fields the release's sets (lines 128-130, 162-177). `HAS_LIGHTTRACK` is True: the
# tracker is injected. See PROVENANCE.md.
import sys
import logging
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image

HAS_LIGHTTRACK = True  # TRANSCRIPTION: the tracker class is injected (tracker_factory)


'''
body = []
body += rng(22, 31)            # suppress_stdout
body += ["", ""]
body += rng(35, 35)            # logger
body += ["", ""]
body += rng(71, 83)            # _log_info / _log_warn / _log_debug
body += ["", ""]
cls = '''class LangSAMLightTrackDetector:
    """The release's detector state machine (TRANSCRIPTION: constructor replaced)."""

    def __init__(self, predict, tracker_factory, lighttrack_weights='', lighttrack_arch='LightTrackM_Subnet',
                 confidence_threshold=0.3, cache_timeout=10, device='cpu'):
        # TRANSCRIPTION: the release loads LangSAM and discovers LightTrack here; both injected
        self.predict = predict
        self.tracker_factory = tracker_factory
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.lighttrack_weights = lighttrack_weights
        self.lighttrack_arch = lighttrack_arch
        self.cache_timeout = cache_timeout
        # the release's state fields (lines 162-177)
        self.tracker = None
        self.state = 'DETECT'  # DETECT, TRACK, CACHE
        self.is_tracking = False
        self.cached_bbox = None
        self.cache_counter = 0
        self.text_prompt = None
        self.cached_goal_roi = None
        self.cached_goal_keyword = None
        self._logged_detection = False
        self._logged_tracking = False
'''
body += cls.split("\n")
body += rng(179, 189)          # set_classes (needed by set_yolo_keyword)
body += [""]
body += rng(191, 707 - 4)      # detect .. _xywh_to_xyxy (up to line 703)
text = header + "\n".join(body) + "\n"
old1 = "            self.tracker = LightTrack(weights=self.lighttrack_weights, arch=self.lighttrack_arch)"
new1 = ("            # TRANSCRIPTION: LightTrack(...) -> the injected tracker_factory(...)\n"
        "            self.tracker = self.tracker_factory(weights=self.lighttrack_weights, arch=self.lighttrack_arch)")
assert text.count(old1) == 1
text = text.replace(old1, new1)
old2 = "                results = self.langsam.predict([image_pil], [self.text_prompt])"
new2 = ("                # TRANSCRIPTION: self.langsam.predict -> the injected self.predict\n"
        "                results = self.predict([image_pil], [self.text_prompt])")
assert text.count(old2) == 1
text = text.replace(old2, new2)
open(OUT, "w").write(text)
print("written", OUT)
