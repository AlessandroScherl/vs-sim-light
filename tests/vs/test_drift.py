"""The copy of vs-sim is the pin's copy, byte for byte (docs/PORT_PLAN.md section 2.4).

Hand-written; it lives outside the copy's roots on purpose. Run it right after the copied
tier, in its own pytest process (``pytest tests/vs``; the two test trees each have a
``conftest.py`` that its tests import as ``conftest``, so one process cannot hold both):
``tools/vssim_drift.py`` re-derives every copied file
from ``external/VSSIM_COMMIT`` through the same rules and patches the copy used and refuses
any difference, any file the manifest does not name, and any entry from another commit.

The second half guards the copied tier against passing *vacuously*: several vs-sim tests
loop over a directory glob (``for path in ARM_FILES.glob("*.yaml")``), and a path that
resolves to the wrong place gives an empty loop and a green test. Each such directory, as
the copied test file itself resolves it, must hold exactly as many files as at the pin.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "vssim_drift.py"
COPIED_FAST = ROOT / "vsbench" / "tests" / "fast"


def _drift():
    spec = importlib.util.spec_from_file_location("_vssim_drift", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vc():
    mod = _drift()
    if not mod.vc.CLONE.is_dir():
        pytest.skip(f"the pinned vs-sim clone is not at {mod.vc.CLONE} (README, U0)")
    return mod.vc


def test_the_copy_is_the_pins_copy(vc):
    problems = _drift().check()
    assert problems == [], "\n".join(problems)


def test_the_patch_list_stays_short_and_every_patch_says_why():
    vc = _drift().vc
    patches = vc.load_patches()
    assert len(patches) <= 6, sorted(patches)   # a seventh is a question for the owner
    for dest, p in patches.items():
        assert p["reason"].strip() and all(pair["reason"].strip() for pair in p["pairs"]), dest


def _pin_count(vc, src_dir: str, suffix: str, recursive: bool = False) -> int:
    n = 0
    for src, _b, _m in vc.source_files(vc.read_pin()):
        if not src.startswith(src_dir + "/") or not src.endswith(suffix):
            continue
        if recursive or "/" not in src[len(src_dir) + 1:]:
            n += 1
    return n


#: (copied test file, how it resolves its directory, glob, the source dir at the pin,
#: recursive?) - every directory glob in vs-sim's fast tier that walks repository files
#: (`grep -n '\.r\?glob(' tests/fast/*.py`; the others glob a tmp_path).
GLOBS = (
    ("test_control.py", lambda f: f.parents[2] / "configs" / "arms", "*.yaml", "configs/arms",
     False),
    ("test_arms.py", lambda f: f.parents[2] / "configs" / "arms", "*.yaml", "configs/arms",
     False),
    ("test_package.py", lambda f: f.parents[2] / "vs", "*.py", "vssim", False),
    ("test_package.py", lambda f: f.parents[2] / "vs" / "baselines", "*.py", "vssim/baselines",
     False),
    ("test_package.py", lambda f: f.parents[2] / "vs" / "bench", "*.py", "vssim/bench", False),
    ("test_package.py", lambda f: f.parents[2] / "vs", "**/*.py", "vssim", True),
    ("test_package.py", lambda f: f.parents[2] / "tests", "**/*.py", "tests", True),
    ("test_package.py", lambda f: f.parents[2] / "scripts", "*.py", "scripts", False),
    ("test_starts.py", lambda f: f.parents[2] / "benchmark" / "starts", "*.json",
     "benchmark/starts", False),
)


@pytest.mark.parametrize("test_file,where,pattern,src_dir,recursive", GLOBS,
                         ids=[f"{g[0]}:{g[3]}/{g[2]}" for g in GLOBS])
def test_no_copied_glob_loop_can_pass_on_an_empty_directory(vc, test_file, where, pattern,
                                                            src_dir, recursive):
    f = COPIED_FAST / test_file
    assert f.is_file(), f
    found = list(where(f.resolve()).glob(pattern))
    want = _pin_count(vc, src_dir, pattern.rsplit("*", 1)[-1], recursive)
    if src_dir == "tests":
        # tests/slow is not copied (docs/PORT_PLAN.md 2.2): the copy has the rest
        want -= _pin_count(vc, "tests/slow", ".py", True)
    assert want > 0 and len(found) == want, (test_file, src_dir, len(found), want)
