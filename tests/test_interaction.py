"""The point interaction matrix: the analytic image flow. No renderer.

200 seeded pixels at depths of 0.3 to 2 m; each axis alone and one random twist, each
scaled so the median predicted flow is 1 px over ``tau``. The median gap between
``predicted_flow`` and ``exact_flow`` must be below 0.01 px and shrink about fourfold when
``tau`` is halved, which is what a first-order model of a smooth motion has to do.

The gate is on the median, like the flow it is scaled to: the linearisation error of a
point is second order in its own flow, and a forward motion moves a point at 0.3 m near
the image edge by 7 px while the median point moves 1 px, so its error is 0.1 px by
geometry, not by a bug. The largest error is printed and bounded loosely.
"""

import numpy as np
import pytest

from vsbench.camera import gate_camera
from vsbench.interaction import exact_flow, interaction_matrix, predicted_flow
from vsbench.se3 import exp_se3

AXES = ["vx", "vy", "vz", "wx", "wy", "wz", "random"]
TAU = 0.04


def sample_points(cam, n=200, seed=20):
    rng = np.random.default_rng(seed)
    px = np.column_stack([rng.uniform(0, cam.width - 1, n), rng.uniform(0, cam.height - 1, n)])
    Z = rng.uniform(0.3, 2.0, n)
    return px, Z, rng


def unit_twist(axis, rng):
    if axis == 6:
        xi = rng.normal(size=6)
        return xi / np.linalg.norm(xi)
    xi = np.zeros(6)
    xi[axis] = 1.0
    return xi


def scaled_twist(cam, px, Z, axis, rng, median_px=1.0, tau=TAU):
    """The unit twist along ``axis`` scaled so the median predicted flow over ``tau`` is ``median_px``."""
    xi = unit_twist(axis, rng)
    m = np.median(np.linalg.norm(predicted_flow(cam, px, Z, xi, tau), axis=1))
    return xi * (median_px / m)


@pytest.mark.parametrize("axis", range(7), ids=AXES)
def test_predicted_flow_matches_exact_flow(axis):
    cam = gate_camera()
    px, Z, rng = sample_points(cam)
    xi = scaled_twist(cam, px, Z, axis, rng)
    pred = predicted_flow(cam, px, Z, xi, TAU)
    exact = exact_flow(cam, px, Z, xi, TAU)
    assert np.all(np.isfinite(exact))
    med = float(np.median(np.linalg.norm(pred, axis=1)))
    assert abs(med - 1.0) < 1e-9
    err = np.linalg.norm(pred - exact, axis=1)
    err_half = np.linalg.norm(predicted_flow(cam, px, Z, xi, TAU / 2) - exact_flow(cam, px, Z, xi, TAU / 2), axis=1)
    med_err, max_err = float(np.median(err)), float(err.max())
    ratio_med = med_err / max(float(np.median(err_half)), 1e-300)
    ratio_max = max_err / max(float(err_half.max()), 1e-300)
    print(f"step2 analytic flow {AXES[axis]}: |pred - exact| median {med_err:.2e} px, max {max_err:.2e} px at 1 px median flow; x{ratio_med:.2f} (median) x{ratio_max:.2f} (max) when tau is halved")
    assert med_err < 0.01
    assert max_err < 0.2
    # first-order model: the error is second order in tau. A pure translation along x or
    # y has no linearisation error at all (the flow is exactly linear in the shift), so
    # the ratio is only checked when there is an error to speak of.
    if max_err > 1e-9:
        assert 3.0 < ratio_med < 5.0, f"median error ratio {ratio_med:.2f} when tau is halved"
        assert 3.0 < ratio_max < 5.0, f"max error ratio {ratio_max:.2f} when tau is halved"


def test_signs_follow_the_axis_table():
    """A point near the centre, in front of the camera: which way does the image move?"""
    cam = gate_camera()
    px = np.array([[cam.cx + 60.0, cam.cy + 40.0]])  # right of and below the principal point
    Z = np.array([1.0])
    tau = 0.01

    def flow(axis, value=1.0):
        xi = np.zeros(6)
        xi[axis] = value
        return exact_flow(cam, px, Z, xi, tau)[0]

    assert flow(0)[0] < 0 and abs(flow(0)[1]) < 1e-12  # vx: content moves left
    assert flow(1)[1] < 0 and abs(flow(1)[0]) < 1e-12  # vy: content moves up
    f = flow(2)
    assert f[0] > 0 and f[1] > 0  # vz: expands from the centre (this point is right-below)
    assert flow(3)[1] > 0  # wx (tilt up): content moves down
    assert flow(4)[0] < 0  # wy (pan right): content moves left
    f = flow(5)  # wz (roll clockwise from behind): content rotates counter-clockwise
    r = px[0] - np.array([cam.cx, cam.cy])
    assert r[0] * f[1] - r[1] * f[0] < 0  # counter-clockwise on a y-down image is a negative z cross


def test_interaction_matrix_entries_and_errors():
    cam = gate_camera()
    px = np.array([[cam.cx + 0.2 * cam.fx, cam.cy - 0.1 * cam.fy]])
    L = interaction_matrix(cam, px, 2.0)[0]
    x, y, Z = 0.2, -0.1, 2.0
    ref = np.array(
        [[-1 / Z, 0, x / Z, x * y, -(1 + x * x), y], [0, -1 / Z, y / Z, 1 + y * y, -x * y, -x]]
    )
    assert np.abs(L - ref).max() < 1e-12
    with pytest.raises(ValueError):
        interaction_matrix(cam, px, 0.0)
    with pytest.raises(ValueError):
        interaction_matrix(cam, np.zeros((3, 2)), np.ones(2))
    # exact flow of a point that ends behind the camera is NaN, not a wrong number
    f = exact_flow(cam, px, 0.05, np.array([0, 0, 1.0, 0, 0, 0]), 1.0)
    assert np.all(np.isnan(f))
    # and exact flow of a zero twist is exactly zero
    assert np.all(exact_flow(cam, px, 1.0, np.zeros(6), 1.0) == 0.0)
    assert np.array_equal(exp_se3(np.zeros(6)), np.eye(4))
