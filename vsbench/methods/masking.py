"""Goal masks for a method: does it take one, and which of its matches does one keep.

``python -m vsbench fly ... --mask objects`` (or ``--mask my_mask.png``) hands a method a
bool ``[H, W]`` mask on the GOAL picture (True = "match here"), once per episode, through
``set_target(..., goal_mask=mask)``. The mask is known at goal time (the goal depth, or a
file); nothing about the live frame's truth is in it.

A method says it takes one in either of two ways:

* its ``set_target`` has a parameter named ``goal_mask`` (default ``None``), or
* it sets ``uses_goal_mask = True`` (and its ``set_target`` then must have that parameter;
  ``uses_goal_mask = False`` opts out even if the parameter is there).

A method that takes none flies exactly as before, and ``--mask`` on it is refused (a label
saying "masked" over a flight that never used the mask would be a lie).
"""

from __future__ import annotations

import inspect

import numpy as np


def takes_goal_mask(method) -> bool:
    """True when ``method.set_target`` accepts ``goal_mask=`` and the method has not opted out.

    Raises ``TypeError`` when ``uses_goal_mask = True`` but ``set_target`` has no
    ``goal_mask`` parameter (the harness could not hand it over).
    """
    declared = getattr(method, "uses_goal_mask", None)
    has_param = _set_target_has_goal_mask(method)
    if declared is None:
        return has_param
    if not isinstance(declared, bool):
        raise TypeError(f"uses_goal_mask must be a bool, got {declared!r}")
    if declared and not has_param:
        raise TypeError(f"{type(method).__name__} says uses_goal_mask = True but its "
                        "set_target has no goal_mask parameter (add goal_mask=None)")
    return declared


def _set_target_has_goal_mask(method) -> bool:
    fn = getattr(method, "set_target", None)
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    p = params.get("goal_mask")
    return p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        inspect.Parameter.KEYWORD_ONLY)


def mask_matches(kp_des, kp_cur, goal_mask) -> np.ndarray:
    """``keep`` (N,) bool: the matches whose GOAL point lies inside ``goal_mask``.

    ``kp_des`` (N, 2) are the matched pixels in the goal picture (x right, y down; OpenCV's
    keypoint convention, pixel centres at whole numbers), ``kp_cur`` the same matches in the
    live picture (only its length is checked: the mask is the goal's, so the live point
    plays no part). A goal point is inside when the mask is True at its nearest pixel;
    points off the picture are outside. ``goal_mask=None`` keeps every match::

        keep = mask_matches(goal_pts, live_pts, self.goal_mask)
        goal_pts, live_pts = goal_pts[keep], live_pts[keep]
    """
    kd = np.asarray(kp_des, dtype=np.float64).reshape(-1, 2)
    if kp_cur is not None:
        kc = np.asarray(kp_cur, dtype=np.float64).reshape(-1, 2)
        if len(kc) != len(kd):
            raise ValueError(f"kp_des has {len(kd)} rows, kp_cur {len(kc)}")
    if goal_mask is None:
        return np.ones(len(kd), dtype=bool)
    m = np.asarray(goal_mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"goal_mask must be a 2-D [H, W] mask, got shape {m.shape}")
    h, w = m.shape
    keep = np.zeros(len(kd), dtype=bool)
    finite = np.all(np.isfinite(kd), axis=1)
    u = np.full(len(kd), -1, dtype=np.int64)
    v = np.full(len(kd), -1, dtype=np.int64)
    u[finite] = np.floor(kd[finite, 0] + 0.5).astype(np.int64)
    v[finite] = np.floor(kd[finite, 1] + 0.5).astype(np.int64)
    inside = finite & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    keep[inside] = m[v[inside], u[inside]]
    return keep
