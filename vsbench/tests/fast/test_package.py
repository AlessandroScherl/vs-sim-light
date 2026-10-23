# vs-sim copy: tests/fast/test_package.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules + patch P1  # noqa: E501
"""Package hygiene: the simulator core imports without Isaac and depends on nothing but
numpy + yaml; the baselines and the bench are the only places torch may appear; and
nothing anywhere imports the research repository this was extracted from."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import vsbench.vs

PKG = Path(vsbench.vs.__file__).resolve().parent
CORE_ALLOWED = {"numpy", "yaml", "vsbench"}
BASELINES_ALLOWED = CORE_ALLOWED | {"torch", "cv2", "PIL", "transformers", "timm", "scipy",
                                    "cns",     # the CNS checkout, on the path at run time
                                    "visp",    # ViSP's bindings, built from source, optional
                                    "lightglue", "romatch", "romav2",   # the authors' packages
                                    "imageio_ffmpeg",                    # the films' encoder
                                    "tracking",   # the LightTrack checkout (ART-VS ROI mode)
                                    "sklearn",    # DINOBot's published K-means (D3), on use
                                    "threadpoolctl"}   # ... pinned to one thread (sklearn's dep)
ISAAC_ONLY = {"isaacsim", "omni", "pxr"}
CORE_MODULES = ("camera", "scene", "scenes", "assets", "labels", "bake", "isaac_scene",
                "starts", "benchmark", "demo")


def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node, a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node, node.module.split(".")[0]


def test_the_core_imports_without_isaac_or_torch():
    """`import vsbench.vs` and every core module must not pull torch or Isaac: the simulator
    is numpy-only. Checked in a fresh interpreter so this test's own imports cannot
    mask it."""
    code = ("import sys, vsbench.vs; " + "; ".join(f"import vsbench.vs.{m}" for m in CORE_MODULES)
            + "; bad = {'torch', 'isaacsim', 'omni', 'pxr', 'refinevs'} & set(sys.modules); "
              "print(sorted(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=PKG.parent)
    assert out.stdout.strip() == "[]", out.stdout


def test_core_third_party_imports_are_numpy_and_yaml_only_and_isaac_is_lazy():
    stdlib = set(sys.stdlib_module_names)
    for path in PKG.glob("*.py"):
        for node, name in _imports(path):
            if name in stdlib or name in CORE_ALLOWED:
                continue
            if name in ISAAC_ONLY:
                assert node.col_offset > 0, f"{path.name}: top-level import of {name}"
                continue
            pytest.fail(f"{path.name} imports {name!r}, which is not a core dependency")


def test_the_baselines_and_the_bench_stay_inside_their_declared_dependencies():
    stdlib = set(sys.stdlib_module_names)
    for sub in ("baselines", "bench"):
        for path in (PKG / sub).glob("*.py"):
            for _node, name in _imports(path):
                if name in stdlib or name in BASELINES_ALLOWED:
                    continue
                pytest.fail(f"{sub}/{path.name} imports {name!r}, which is not a declared "
                            "dependency of the baselines extra")


#: Third-party sources vendored byte-for-byte as test references (MIT; their provenance is
#: beside them). Not this repository's code, so not walked - and one of them, the DINOBot
#: gist, is published with syntax errors and does not parse (it is kept verbatim as text).
REFERENCE = PKG.parent / "tests" / "fixtures" / "reference"


def test_nothing_refers_to_the_research_repository():
    for path in list(PKG.rglob("*.py")) + list((PKG.parent / "tests").rglob("*.py")) \
            + list((PKG.parent / "scripts").glob("*.py")) \
            + list((PKG.parent / "examples").glob("*.py")):
        if path.name.endswith("_fixture.py") or "make_" in path.name:
            continue        # the fixture generators ran in the source repository
        if REFERENCE in path.parents:
            continue        # vendored third-party references (see REFERENCE)
        for _node, name in _imports(path):
            assert name != "refinevs", f"{path} imports the research repository"


def test_version_and_public_names():
    assert vsbench.vs.__version__
    for name in vsbench.vs.__all__:
        assert hasattr(vsbench.vs, name)
