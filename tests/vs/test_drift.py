"""The copy of vs-sim is the pin's copy, byte for byte (docs/PORT_PLAN.md section 2.4).

Hand-written (it is named in ``tools/vssim_copy.py::HAND_WRITTEN``). Runs in the copied tier:
``tools/vssim_drift.py`` re-derives every copied file from ``external/VSSIM_COMMIT`` through
the same rules and patches the copy used and refuses any difference, any file the manifest
does not name, and any entry from another commit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "vssim_drift.py"


def _drift():
    spec = importlib.util.spec_from_file_location("_vssim_drift", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_copy_is_the_pins_copy():
    drift = _drift()
    if not drift.vc.CLONE.is_dir():
        pytest.skip(f"the pinned vs-sim clone is not at {drift.vc.CLONE} (README, U0)")
    problems = drift.check()
    assert problems == [], "\n".join(problems)


def test_the_patch_list_stays_short_and_every_patch_says_why():
    vc = _drift().vc
    patches = vc.load_patches()
    assert len(patches) <= 6, sorted(patches)   # a seventh is a question for the owner
    for dest, p in patches.items():
        assert p["reason"].strip() and all(pair["reason"].strip() for pair in p["pairs"]), dest
