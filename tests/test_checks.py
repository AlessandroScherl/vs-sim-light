"""Step 3 gates 5 to 8: zero at target, hold at target, restoring direction, negative control.

The oracle passes all three setup checks on the poster scene. The negative control is a
test-only oracle with the sign of one twist component flipped: for each of the six axes
it must fail the restoring-direction check on that axis and only there. At the target the
twist is zero whatever its sign, so the flipped oracle still passes the two target checks;
the negative control therefore runs the restoring check, with a 300-step budget instead of
3000 because the flipped axis never converges (the other axes converge in about 80 steps).
"""

import numpy as np
import pytest

from vsbench.camera import gate_camera
from vsbench.checks import (
    AXES,
    hold_at_target,
    restoring_direction,
    run_checks,
    zero_at_target,
)
from vsbench.episode import EpisodeConfig
from vsbench.methods import make_method
from vsbench.methods.oracle_pbvs import OraclePBVS
from vsbench.scenes import poster_target_pose


class FlippedOracle(OraclePBVS):
    """The oracle with the sign of one twist component flipped. Test-only, not registered."""

    name = "flipped_oracle"

    def __init__(self, axis: int, gain: float = 1.0):
        super().__init__(gain)
        self.axis = axis

    def get_control_rate(self, current_rgb, current_depth=None):
        xi, info = super().get_control_rate(current_rgb, current_depth)
        xi = xi.copy()
        xi[self.axis] = -xi[self.axis]
        return xi, info


def test_oracle_passes_zero_and_hold(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    m = make_method("oracle_pbvs")
    z = zero_at_target(r, m, T_t)
    print(f"{backend} step3 zero at target oracle_pbvs: {z.detail}")
    h = hold_at_target(r, m, T_t)
    print(f"{backend} step3 hold at target oracle_pbvs: {h.detail}")
    assert z.passed and z.numbers["v_norm"] < 1e-6 and z.numbers["w_norm"] < 1e-6
    assert h.passed and h.numbers["max_drift_m"] < 1e-6 and h.numbers["max_drift_rad"] < 1e-6


def test_oracle_passes_restoring_direction(backend, renderer_factory):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    res = restoring_direction(r, make_method("oracle_pbvs"), T_t)
    starts = res.numbers["starts"]
    print(f"{backend} step3 restoring direction oracle_pbvs: {res.detail}; per axis {res.per_axis}")
    assert len(starts) == 12 and all(s["sign_ok"] for s in starts) and all(s["status"] == "converged" for s in starts)
    # the first oracle twist is exactly minus the offset on that axis, so the offsets and
    # the expected components agree in magnitude too
    for s in starts:
        assert abs(abs(s["expected_component"]) - (0.02 if s["axis"].startswith("v") else np.deg2rad(2.0))) < 1e-12
        assert abs(s["first_component"] - s["expected_component"]) < 1e-12
    assert res.passed and all(res.per_axis.values()) and res.failed_axes() == []


def test_run_checks_and_report(backend, renderer_factory):
    from vsbench.checks import format_report

    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    results = run_checks(r, make_method("oracle_pbvs"), poster_target_pose(cam), EpisodeConfig(max_steps=300))
    text = format_report(results, "oracle_pbvs")
    assert [x.name for x in results] == ["zero_at_target", "hold_at_target", "restoring_direction"]
    assert all(x.passed for x in results) and "all: pass" in text and "FAIL" not in text


@pytest.mark.parametrize("axis", range(6), ids=AXES)
def test_negative_control(backend, renderer_factory, axis):
    cam = gate_camera()
    r = renderer_factory(backend, "poster", (640, 480))
    T_t = poster_target_pose(cam)
    cfg = EpisodeConfig(max_steps=300)
    m = FlippedOracle(axis)
    res = restoring_direction(r, m, T_t, cfg)
    bad = res.failed_axes()
    flipped = [s for s in res.numbers["starts"] if s["axis"] == AXES[axis]]
    others = [s for s in res.numbers["starts"] if s["axis"] != AXES[axis]]
    print(
        f"{backend} step3 negative control {AXES[axis]} flipped: restoring direction fails on {bad}; "
        f"flipped starts sign_ok {[s['sign_ok'] for s in flipped]} status {[s['status'] for s in flipped]}, "
        f"other axes {sum(s['passed'] for s in others)} of {len(others)} pass"
    )
    assert bad == [AXES[axis]]
    assert not res.passed
    assert all(not s["sign_ok"] and s["status"] != "converged" for s in flipped)
    assert all(s["passed"] for s in others)
    # the flipped oracle still passes the two target checks: a zero twist has no sign
    assert zero_at_target(r, m, T_t).passed and hold_at_target(r, m, T_t).passed
