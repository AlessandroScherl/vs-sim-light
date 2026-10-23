"""The method contract: the ``VSMethod`` protocol, the info keys and output normalisation.

A method sees images, optional depth and the intrinsics, and returns a camera-frame
twist ``[vx, vy, vz, wx, wy, wz]`` in m/s and rad/s (the convention of ``FreeCamera``).
The harness owns the truth: only a method with ``privileged = True`` (the two test
instruments) ever receives poses, through ``observe_truth``. Nothing here imports the
episode loop, a renderer or a scene, and nothing under ``vsbench/methods/`` may.

``wants_depth`` says what depth the harness passes:

* ``"none"``: no depth at all (``desired_depth``, ``distance_prior`` and
  ``current_depth`` are ``None``).
* ``"scalar"``: one number, the ``distance_prior`` (median valid depth of the target
  view) at ``set_target``; ``current_depth`` stays ``None``.
* ``"dense"``: the depth image of the target view at ``set_target`` and of the current
  view at every ``get_control_rate``.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

DepthMode = Literal["none", "scalar", "dense"]
DEPTH_MODES: tuple[str, ...] = ("none", "scalar", "dense")

# every key of the optional info dict is optional
INFO_KEYS: tuple[str, ...] = (
    "kp_cur",  # (M, 2) float pixel coordinates of the matched keypoints in the current image
    "kp_des",  # (M, 2) the matching keypoints in the desired image
    "inliers",  # (M,) bool, True for the matches that survived the outlier rejection
    "error_norm",  # float, the method's own feature or pose error norm
    "observer_s",  # float, seconds spent detecting, matching and reading depth
    "controller_s",  # float, seconds spent in the control law
    "lost",  # bool, True when the method gave up this step (zero twist)
)

REQUIRED = ("name", "wants_depth", "privileged", "configure", "reset", "set_target", "get_control_rate")


@runtime_checkable
class VSMethod(Protocol):
    name: str
    wants_depth: DepthMode
    privileged: bool  # True only for the two test instruments

    def configure(self, K: np.ndarray, width: int, height: int, dt: float) -> None:
        """Intrinsics in pixels, image size and the control period, once per episode."""

    def reset(self) -> None:
        """Forget every state from a previous episode."""

    def set_target(self, desired_rgb: np.ndarray, desired_depth: np.ndarray | None = None, distance_prior: float | None = None) -> None:
        """The desired view, with depth in the form ``wants_depth`` asked for.

        Optional: a ``goal_mask=None`` parameter (bool ``[H, W]`` on the desired view, True =
        "match here"; ``--mask objects|<file.png>``) - see ``vsbench.methods.masking``. A
        method without it never receives a mask."""

    def get_control_rate(self, current_rgb: np.ndarray, current_depth: np.ndarray | None = None) -> Any:
        """The twist (6,), alone or as ``(twist, info)``."""

    def observe_truth(self, T_wc: np.ndarray, T_wc_target: np.ndarray) -> None:
        """Called before ``get_control_rate`` only when ``privileged`` is True."""


def check_method(method: object) -> None:
    """Raise ``TypeError`` or ``ValueError`` unless ``method`` fills the contract."""
    missing = [a for a in REQUIRED if not hasattr(method, a)]
    if missing:
        raise TypeError(f"{type(method).__name__} lacks {missing}")
    if not isinstance(getattr(method, "name"), str) or not method.name:
        raise TypeError("method.name must be a non-empty string")
    if method.wants_depth not in DEPTH_MODES:
        raise ValueError(f"wants_depth must be one of {DEPTH_MODES}, got {method.wants_depth!r}")
    if not isinstance(method.privileged, bool):
        raise TypeError("privileged must be a bool")
    if method.privileged and not hasattr(method, "observe_truth"):
        raise TypeError("a privileged method must implement observe_truth")
    from vsbench.methods.masking import takes_goal_mask

    takes_goal_mask(method)          # uses_goal_mask = True without a goal_mask parameter raises


def normalise_output(out: Any) -> tuple[np.ndarray, dict]:
    """``(twist, info)`` from what ``get_control_rate`` returned.

    The twist becomes a fresh float64 array of shape ``(6,)`` (any six-element array-like
    is accepted). A twist that is not six finite numbers raises ``ValueError``; an info
    that is not a dict or ``None`` raises ``TypeError``.
    """
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError(f"get_control_rate must return a twist or (twist, info), got a {len(out)}-tuple")
        twist, info = out
    else:
        twist, info = out, None
    try:
        xi = np.array(twist, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"twist must be six numbers, got {twist!r}") from e
    if xi.size != 6:
        raise ValueError(f"twist must have six elements, got shape {xi.shape}")
    xi = xi.reshape(6)
    if not np.all(np.isfinite(xi)):
        raise ValueError(f"twist must be finite, got {xi}")
    if info is None:
        info = {}
    elif not isinstance(info, dict):
        raise TypeError(f"info must be a dict or None, got {type(info).__name__}")
    return xi, info
