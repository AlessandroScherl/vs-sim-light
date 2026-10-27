# vs-sim copy: vssim/demo.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""A minimal episode loop and a cheating controller - to show the API, never a method.

:func:`run_episode` is the four verbs in order: ``reset(goal); render()`` to take
the goal view, ``reset(start)``, then ``render -> controller -> apply`` for
``max_steps``, and a small record of where it started and where it ended.
:func:`pose_servo` closes that loop by reading the **true** pose - which visual
servoing may never do. It exists so the examples, the tests and a first station
smoke can exercise every verb without a matcher or a control law; a real
controller has the signature ``controller(scene, goal_rgb, rgb, depth) -> twist``
and ignores ``scene``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from vsbench.vs.scene import Pose, Scene, rotation_error_deg, translation_error_mm

Controller = Callable[[Scene, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def pose_servo(scene: Scene, goal_rgb=None, rgb=None, depth=None,
               gain: float = 2.0) -> np.ndarray:
    """A proportional controller on the TRUE pose. A demo of the loop, not a method.

    Returns the optical-frame twist ``(v, w)`` that moves the camera toward
    ``scene.goal_pose`` at ``gain`` per second. The image arguments are accepted
    and ignored so it has a controller's signature.
    """
    cur, goal = scene.pose, scene.goal_pose
    R = cur.rotation
    v = gain * (R.T @ (goal.position - cur.position))          # into the optical frame
    rel = R.T @ goal.rotation
    ang = float(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))
    if ang < 1e-12:
        w = np.zeros(3)
    else:
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        w = gain * ang * axis / (2.0 * np.sin(ang))
    return np.concatenate([v, w])


def run_episode(scene: Scene, start: Pose | dict, controller: Controller = pose_servo, *,
                max_steps: int = 300, dt: float = 1.0 / 60.0,
                on_step: Callable[[int, np.ndarray, np.ndarray, Pose, np.ndarray], None]
                | None = None) -> dict:
    """One episode: goal view, start pose, ``max_steps`` of render -> control -> apply.

    ``start`` is a :class:`Pose` or a start-file entry (``cam_pos`` + ``cam_R_opt``).
    ``on_step(i, rgb, depth, pose, twist)`` is called after each control step when
    given (to save frames or telemetry). Returns a record with the start and final
    translation / rotation errors, the step count and the wall time per step.
    """
    start_pose = start if isinstance(start, Pose) else Pose.from_start(start)
    goal = scene.goal_pose
    scene.reset(goal)
    goal_rgb, _goal_depth, _ = scene.render()
    scene.reset(start_pose)
    rec = {
        "scene": scene.name,
        "start_id": start.get("start_id") if isinstance(start, dict) else None,
        "start_translation_mm": translation_error_mm(start_pose.position, goal.position),
        "start_rotation_deg": rotation_error_deg(start_pose.rotation, goal.rotation),
        "steps": 0, "dt": float(dt),
    }
    t0 = time.perf_counter()
    for i in range(1, int(max_steps) + 1):
        rgb, depth, pose = scene.render()
        twist = np.asarray(controller(scene, goal_rgb, rgb, depth), dtype=np.float64)
        if not np.all(np.isfinite(twist)):
            rec["terminated_reason"] = "non_finite_twist"
            break
        scene.apply(twist, dt)
        rec["steps"] = i
        if on_step is not None:
            on_step(i, rgb, depth, pose, twist)
    else:
        rec["terminated_reason"] = "max_steps"
    final = scene.pose
    rec["final_translation_mm"] = translation_error_mm(final.position, goal.position)
    rec["final_rotation_deg"] = rotation_error_deg(final.rotation, goal.rotation)
    rec["sec_per_step"] = (time.perf_counter() - t0) / max(rec["steps"], 1)
    rec["final_pose"] = final.as_dict()
    return rec
