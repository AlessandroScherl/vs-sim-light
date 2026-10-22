"""The three setup checks a method must pass before its results mean anything.

* ``zero_at_target``: one control call at the exact target pose; the linear and the
  angular twist norms must both be below 1e-6.
* ``hold_at_target``: 50 steps from the target pose with the stop rule off; the camera
  must move less than 1e-6 m and 1e-6 rad.
* ``restoring_direction``: plus and minus 2 cm on each translation axis and plus and
  minus 2 deg on each rotation axis, 12 starts. On the first step the twist component on
  that axis must have the sign of the same component of
  ``log_se3(inv(T_wc) @ T_wc_target)``, and the episode must converge.

Every check runs the ordinary ``run_episode``, so the method is set up exactly as in a
benchmark episode. A check returns a ``CheckResult`` with a verdict per axis where that
makes sense and the numbers behind it; ``run_checks`` runs all three and
``format_report`` prints them as a table. The negative control in the tests (an oracle
with one flipped axis) shows the restoring check fails on that axis and only there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vsbench.episode import EpisodeConfig, run_episode
from vsbench.methods.base import VSMethod
from vsbench.reference import invert
from vsbench.se3 import exp_se3, log_se3, pose_error

AXES = ("vx", "vy", "vz", "wx", "wy", "wz")
TRANSLATION_OFFSET = 0.02  # m
ROTATION_OFFSET = float(np.deg2rad(2.0))  # rad
ZERO_TOL = 1e-6  # m/s and rad/s
HOLD_TOL = (1e-6, 1e-6)  # m, rad
HOLD_STEPS = 50


@dataclass
class CheckResult:
    name: str
    passed: bool
    per_axis: dict[str, bool] | None = None  # None for the checks that have no axis
    numbers: dict = field(default_factory=dict)
    detail: str = ""

    def failed_axes(self) -> list[str]:
        return [a for a, ok in (self.per_axis or {}).items() if not ok]


def axis_offsets() -> list[tuple[str, float, np.ndarray]]:
    """``(axis, sign, offset twist)`` for the 12 restoring-direction starts."""
    out = []
    for i, axis in enumerate(AXES):
        step = TRANSLATION_OFFSET if i < 3 else ROTATION_OFFSET
        for sign in (1.0, -1.0):
            xi = np.zeros(6)
            xi[i] = sign * step
            out.append((axis, sign, xi))
    return out


def zero_at_target(renderer, method: VSMethod, T_wc_target: np.ndarray, cfg: EpisodeConfig = EpisodeConfig(), scene_name: str = "poster") -> CheckResult:
    one = EpisodeConfig(dt=cfg.dt, max_steps=1, limits=cfg.limits, stop_tol=None)
    log = run_episode(renderer, method, T_wc_target, T_wc_target, one, scene_name)
    if log.meta["status"] != "timeout" or len(log) != 1:
        return CheckResult("zero_at_target", False, numbers={"status": log.meta["status"], "error": log.meta.get("error")}, detail=f"no twist: {log.meta['status']} {log.meta.get('error') or ''}")
    xi = log.twist_cmd[0]
    v, w = float(np.linalg.norm(xi[:3])), float(np.linalg.norm(xi[3:]))
    ok = v < ZERO_TOL and w < ZERO_TOL
    return CheckResult("zero_at_target", ok, numbers={"v_norm": v, "w_norm": w, "twist": xi.tolist()}, detail=f"|v| {v:.2e} m/s, |w| {w:.2e} rad/s (limit {ZERO_TOL:.0e})")


def hold_at_target(renderer, method: VSMethod, T_wc_target: np.ndarray, cfg: EpisodeConfig = EpisodeConfig(), scene_name: str = "poster") -> CheckResult:
    hold = EpisodeConfig(dt=cfg.dt, max_steps=HOLD_STEPS, limits=cfg.limits, stop_tol=None)
    log = run_episode(renderer, method, T_wc_target, T_wc_target, hold, scene_name)
    if log.meta["status"] != "timeout" or len(log) != HOLD_STEPS:
        return CheckResult("hold_at_target", False, numbers={"status": log.meta["status"], "error": log.meta.get("error")}, detail=f"did not complete {HOLD_STEPS} steps: {log.meta['status']} {log.meta.get('error') or ''}")
    # the largest drift over the whole hold, not only at the end
    poses = list(log.T_wc) + [log.final_pose()]
    errs = np.array([pose_error(T, T_wc_target) for T in poses])
    d_t, d_r = float(errs[:, 0].max()), float(errs[:, 1].max())
    v_max = float(np.linalg.norm(log.twist_applied[:, :3], axis=1).max())
    w_max = float(np.linalg.norm(log.twist_applied[:, 3:], axis=1).max())
    ok = d_t < HOLD_TOL[0] and d_r < HOLD_TOL[1]
    return CheckResult("hold_at_target", ok, numbers={"steps": HOLD_STEPS, "max_drift_m": d_t, "max_drift_rad": d_r, "max_v": v_max, "max_w": w_max}, detail=f"drift {d_t:.2e} m, {d_r:.2e} rad over {HOLD_STEPS} steps (limits {HOLD_TOL[0]:.0e}, {HOLD_TOL[1]:.0e})")


def restoring_direction(renderer, method: VSMethod, T_wc_target: np.ndarray, cfg: EpisodeConfig = EpisodeConfig(), scene_name: str = "poster") -> CheckResult:
    per_axis = {a: True for a in AXES}
    rows = []
    for axis, sign, offset in axis_offsets():
        i = AXES.index(axis)
        T0 = T_wc_target @ exp_se3(offset)
        log = run_episode(renderer, method, T0, T_wc_target, cfg, scene_name)
        expected = log_se3(invert(T0) @ T_wc_target)
        if len(log) == 0:
            first = None
            sign_ok = False
        else:
            first = log.twist_cmd[0]
            sign_ok = bool(first[i] * expected[i] > 0.0)
        converged = log.meta["status"] == "converged"
        ok = sign_ok and converged
        per_axis[axis] = per_axis[axis] and ok
        rows.append(
            {
                "axis": axis,
                "sign": sign,
                "first_twist": None if first is None else first.tolist(),
                "first_component": None if first is None else float(first[i]),
                "expected_component": float(expected[i]),
                "sign_ok": sign_ok,
                "status": log.meta["status"],
                "steps": len(log),
                "final_error_m": log.meta["final_error_m"],
                "final_error_rad": log.meta["final_error_rad"],
                "error": log.meta.get("error"),
                "passed": ok,
            }
        )
    passed = all(per_axis.values())
    bad = [f"{r['axis']}{'+' if r['sign'] > 0 else '-'}" for r in rows if not r["passed"]]
    steps = [r["steps"] for r in rows if r["status"] == "converged"]
    detail = f"{sum(r['passed'] for r in rows)} of {len(rows)} starts right and converged" + (f" (converged in {min(steps)} to {max(steps)} steps)" if steps else "") + (f"; failed: {', '.join(bad)}" if bad else "")
    return CheckResult("restoring_direction", passed, per_axis=per_axis, numbers={"starts": rows}, detail=detail)


CHECKS = (zero_at_target, hold_at_target, restoring_direction)


def run_checks(renderer, method: VSMethod, T_wc_target: np.ndarray, cfg: EpisodeConfig = EpisodeConfig(), scene_name: str = "poster") -> list[CheckResult]:
    return [c(renderer, method, T_wc_target, cfg, scene_name) for c in CHECKS]


def format_report(results: list[CheckResult], method_name: str = "") -> str:
    lines = [f"setup checks{' for ' + method_name if method_name else ''}:"]
    for r in results:
        lines.append(f"  {r.name:20s} {'pass' if r.passed else 'FAIL'}  {r.detail}")
        if r.per_axis is not None:
            lines.append("    per axis: " + "  ".join(f"{a} {'pass' if ok else 'FAIL'}" for a, ok in r.per_axis.items()))
    lines.append(f"  all: {'pass' if all(r.passed for r in results) else 'FAIL'}")
    return "\n".join(lines)
