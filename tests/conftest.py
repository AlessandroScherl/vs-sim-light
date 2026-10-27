"""vsbench's tests.

In station mode (``VSBENCH_STATION=1``, ``source tools/cpu_env.sh``) software GL is forced at
import, so even a test that loads a GL or Vulkan library itself gets Mesa (llvmpipe /
lavapipe), never the GPU. Outside station mode nothing is forced: the tests that render use
whatever MuJoCo picks (``vsbench.glenv``).

The copied vs-sim tier (``vsbench/tests/fast``) runs in its own process:
``tools/test_copied.sh``.
"""

from __future__ import annotations

from vsbench.glenv import force_software_gl, software_gl_requested

if software_gl_requested():
    force_software_gl()


# ------------------------------------------------------------------ the assets
def scene_assets_missing(name: str) -> str | None:
    """Why scene ``name`` cannot be built here (a missing picture or converted object), or
    None. The pictures and objects are not in the repository (README, Assets)."""
    from vsbench.vs.scenes import load_scenes
    from vsbench.world import assets as A
    from vsbench.world.build import workspace_path

    spec = load_scenes()[0][name]
    raw = spec.raw
    tex = raw.get("texture") or (raw.get("ground") or {}).get("texture")
    if tex and not workspace_path(tex).exists():
        return f"{name}: picture {workspace_path(tex)} not installed"
    if spec.kind != "poster":
        for iid in spec.object_ids():
            if not (A.object_dir(iid) / "meta.json").exists():
                return f"{name}: object {iid!r} not in the object cache {A.cache_root()}"
    return None


def pytest_collection_modifyitems(config, items):
    """A module that sets ``NEEDS_SCENES = (...)`` is skipped where those scenes' assets are
    not installed."""
    import pytest

    seen: dict = {}
    for item in items:
        mod = getattr(item, "module", None)
        names = getattr(mod, "NEEDS_SCENES", ())
        if not names:
            continue
        if mod not in seen:
            seen[mod] = next((r for r in map(scene_assets_missing, names) if r), None)
        if seen[mod]:
            item.add_marker(pytest.mark.skip(reason=seen[mod]))
