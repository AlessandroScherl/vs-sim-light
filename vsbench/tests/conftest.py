# vs-sim copy: tests/conftest.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Shared test setup.

Two tiers:

- ``tests/fast/`` - CPU, numpy, seconds. Always run.
- ``tests/slow/`` - needs Isaac Sim (the station). Marked ``@pytest.mark.slow`` and
  **skipped unless ``-m slow`` is given**.

Run the fast tier from the repo root with any Python >= 3.10 that has numpy,
PyYAML and pytest::

    python3 -m pytest tests/fast -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    """Skip ``slow`` tests unless the run explicitly asks for them (``-m slow``)."""
    if "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="needs Isaac Sim; run with -m slow on the station")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# ------------------------------------------------------------- the bench's fixtures
VALID_CONFIG = {
    "seed": 20260902,
    # `device` is required for every variant that loads an encoder and has no default:
    # the station configs say `cuda`, the fast tier says `cpu` explicitly, and nothing
    # anywhere infers one.
    "matcher": {"variant": "mutual_nn_seeded_subsample", "num_landmarks": 24,
                "device": "cpu"},
    "stop": {"rule": "dec_plateau"},
    "scoring": {"success_rule": ["predicate_a"]},
}


@pytest.fixture
def valid_config() -> dict:
    """The smallest config that `vsbench.vs.bench.config` accepts."""
    import copy

    return copy.deepcopy(VALID_CONFIG)


@pytest.fixture
def valid_config_file(tmp_path, valid_config) -> Path:
    """`valid_config` written to a YAML file in a temp directory."""
    import yaml

    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(valid_config), encoding="utf-8")
    return path


#: The cv2 versions the classical parity fixture is pinned against. It was produced
#: under 4.11.0 and reproduces bit-for-bit under 4.10.0; under 4.12 AKAZE's detector
#: returns 24 keypoints where 4.11 returned 23. That is OpenCV moving, not the port, so
#: the fixture test states its range and skips outside it rather than failing.
CV2_FIXTURE_VERSIONS = ("4.10.", "4.11.")


@pytest.fixture(scope="session")
def writable_checkout() -> Path:
    """Skip a test that must write inside the checkout when the checkout is read-only.

    On a station the repository may be owned by the container user and the host user
    cannot write into it; the `test_run.py` tests that drop a canary in the tree to
    watch `dirty_paths` see it would raise `PermissionError` there. They are testing
    git's view of a working tree, which is a laptop question; skipping is the honest
    outcome.
    """
    probe = REPO_ROOT / ".writable_probe_for_tests.tmp"
    try:
        probe.write_text("probe\n", encoding="utf-8")
        probe.unlink()
        exp = REPO_ROOT / "experiments"
        exp.mkdir(exist_ok=True)
        probe2 = exp / ".writable_probe_for_tests.tmp"
        probe2.write_text("probe\n", encoding="utf-8")
        probe2.unlink()
    except OSError as exc:
        pytest.skip(f"the checkout is not writable by this user ({exc.__class__.__name__}): "
                    "these tests watch git's view of a working tree and need to touch it")
    return REPO_ROOT
