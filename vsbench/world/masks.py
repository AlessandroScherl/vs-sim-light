"""``--mask``: filter the matches with a mask on the GOAL picture (owner request 2026-10-23).

    python -m vsbench fly --scene industrial1_drill --method sift_ransac --mask objects ...
    python -m vsbench fly --scene industrial1_drill --method my_ibvs --plugin examples/my_method.py \\
        --mask my_mask.png ...

Three values:

* ``none`` (default): nothing changes; every record is what it always was.
* ``objects``: the goal picture's object pixels, exactly as vs-sim's ``objmask`` cells compute
  them - the copied runner's ``vsbench.vs.labels.object_mask_from_depth(goal_depth,
  intrinsics, goal_pose)`` (the floor masked out against the ground plane, 5 mm). For a copied
  baseline this IS the protocol's knob ``matcher.goal_mask: objects``: the protocol's own
  ``<arm>_objmask`` cell when it has one (``sift_ransac_objmask``, ``artvs_objmask``,
  ``vitvs_dinov3_objmask``), else the arm's cell with that one override (stated in the
  config's ``_source``); the copied label carries ``objmask``. The rooms only: a poster's goal
  view is all ground plane, the mask keeps no pixel and the flight is refused.
* ``<path.png>``: your own mask on the goal picture (any image; grey value > 127 = keep). A
  mask of another size is resized to the goal picture with nearest-neighbour interpolation
  (stated in the record: ``goal_mask_resized_from``). For a copied baseline the arm is built
  with ``matcher.goal_mask: objects`` (the switch that turns on the matcher's goal-mask seam
  and its ``honours_goal_mask`` refusal), then :func:`attach_goal_mask` sets its
  ``goal_mask_mode`` to ``file`` - so the copied runner computes no depth outline - and wraps
  its ``reset`` to hand it the file's mask; the label's ``objmask`` token becomes
  ``filemask-<md5[:8]>`` and the record says ``goal_mask_mode: file``, ``goal_mask_source:
  file:<name>:<md5>`` (the md5 of the file's bytes). No copied line is edited.

A vsbench method (``--method``) receives the mask through ``set_target(..., goal_mask=...)``
if its ``set_target`` takes that parameter (``vsbench.methods.takes_goal_mask``); otherwise
``--mask`` is refused for it. ``vsbench.methods.mask_matches`` is the helper that drops the
matches whose goal point is outside the mask.

**GT-blind.** Both masks are known at goal time: the goal depth and the goal pose (the goal
the method is handed a picture of) or a file. Nothing is read from the live frame's truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The two named values of ``--mask``; anything else is a file path.
NAMED = ("none", "objects")
#: The source string of the depth outline (the copied debug log's word for it).
OBJECTS_SOURCE = "objects_from_depth"
#: The rule a file mask is read by (stated in the record).
FILE_RULE = "grey value > 127 = keep (cv2.IMREAD_GRAYSCALE); other sizes resized nearest-neighbour"


@dataclass
class GoalMask:
    """One ``--mask`` value, resolved on one scene's goal picture."""

    mode: str                            # "none" | "objects" | "file"
    mask: np.ndarray | None = field(default=None, repr=False)   # bool [H, W] or None
    source: str | None = None            # "objects_from_depth" | "file:<name>:<md5>"
    file: str | None = None              # the path as given
    file_md5: str | None = None          # md5 of the file's bytes
    resized_from: list | None = None     # [w, h] of the file when it was resized

    @property
    def share(self) -> float | None:
        return None if self.mask is None else float(self.mask.mean())

    @property
    def mask_md5(self) -> str | None:
        return None if self.mask is None else hashlib.md5(
            np.ascontiguousarray(self.mask, dtype=np.uint8).tobytes()).hexdigest()

    @property
    def token(self) -> str | None:
        """The label token: ``objmask`` (the copied one), ``filemask-<md5[:8]>``, or None."""
        if self.mode == "objects":
            return "objmask"
        if self.mode == "file":
            return f"filemask-{self.file_md5[:8]}"
        return None

    @property
    def tag(self) -> str:
        """The cell tag's suffix (file names): ``""``, ``_objmask``, ``_filemask_<md5[:8]>``."""
        return {"none": "", "objects": "_objmask"}.get(self.mode, f"_filemask_{(self.file_md5 or '')[:8]}")

    def record(self) -> dict:
        """The record fields ``fly`` adds to a masked episode (none for ``--mask none``)."""
        if self.mode == "none":
            return {}
        out = {"goal_mask_source": self.source, "goal_mask_md5": self.mask_md5}
        if self.mode == "file":
            # the copied runner computed no mask (mode `file`), so its share is the file's
            out.update(goal_mask_share=self.share, goal_mask_file=self.file,
                       goal_mask_file_md5=self.file_md5, goal_mask_file_rule=FILE_RULE,
                       goal_mask_resized_from=self.resized_from)
        return out

    def apply(self, rec: dict) -> dict:
        """Add :meth:`record` to an episode record; for ``objects`` check that the copied
        runner's own mask had the same share (it computes the same outline itself)."""
        if self.mode == "objects":
            got = rec.get("goal_mask_share")
            if got is None or abs(float(got) - float(self.share)) > 1e-12:
                raise RuntimeError(f"the runner's goal mask share {got} is not the bridge's "
                                   f"{self.share}: two different object masks")
        rec.update(self.record())
        return rec


def parse(value: str | None) -> tuple[str, str | None]:
    """``(mode, path)`` of a ``--mask`` value: ``none`` / ``objects`` / ``file`` + path."""
    v = (value or "none").strip()
    if v in NAMED:
        return v, None
    p = Path(v).expanduser()
    if not p.is_file():
        raise SystemExit(f"--mask {value!r}: not 'none', 'objects' or an existing image file")
    return "file", str(p)


def load_file_mask(path, width: int, height: int) -> tuple[np.ndarray, str, list | None]:
    """``(mask, md5 of the file's bytes, [w, h] it was resized from or None)``."""
    import cv2

    raw = Path(path).read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"--mask {path}: not an image OpenCV can read")
    resized = None
    if img.shape != (int(height), int(width)):
        resized = [int(img.shape[1]), int(img.shape[0])]
        img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    mask = img > 127
    if not mask.any():
        raise SystemExit(f"--mask {path}: the mask keeps no pixel (grey value > 127 = keep)")
    return mask, hashlib.md5(raw).hexdigest(), resized


def resolve(value: str | None, scene, goal_depth) -> GoalMask:
    """The ``--mask`` value on ``scene``'s goal: the goal depth (``objects``) or a file."""
    mode, path = parse(value)
    if mode == "none":
        return GoalMask("none")
    intr = scene.intrinsics
    if mode == "objects":
        from vsbench.vs.labels import object_mask_from_depth

        # the copied runner's call, argument for argument (vsbench.vs.bench.runner.run_episode)
        mask = object_mask_from_depth(goal_depth, intr, scene.goal_pose)
        if not mask.any():
            raise SystemExit(f"--mask objects on {scene.name}: the goal view is all ground plane "
                             "(a poster scene?), the object mask keeps no pixel - use a file mask")
        return GoalMask("objects", mask=mask, source=OBJECTS_SOURCE)
    mask, md5, resized = load_file_mask(path, int(intr.width), int(intr.height))
    return GoalMask("file", mask=mask, source=f"file:{Path(path).name}:{md5}", file=path,
                    file_md5=md5, resized_from=resized)


def arm_cell(protocol: dict, arm_id: str, gm: GoalMask) -> tuple[str, bool]:
    """``(protocol cell to configure, needs the goal_mask override)`` for a copied arm.

    Unmasked: the arm itself. Masked: the protocol's ``<arm>_objmask`` cell when it exists
    (the faithful, labelled cell), else the arm's cell plus ``matcher.goal_mask: objects``.
    """
    if gm.mode == "none":
        return arm_id, False
    ids = {e["id"] for e in protocol["arms"]}
    if f"{arm_id}_objmask" in ids:
        return f"{arm_id}_objmask", False
    return arm_id, True


def attach_goal_mask(arm, gm: GoalMask):
    """Hand ``gm``'s mask to an arm built with ``matcher.goal_mask: objects`` (a copied arm)
    or ``goal_mask_mode`` ``objects`` / ``file`` (a ``MethodArm``), without editing it.

    * ``file``: ``goal_mask_mode`` becomes ``file`` (the copied runner then computes no depth
      outline and hands ``reset`` none), ``reset`` is wrapped to hand the file's mask, the
      label's ``objmask`` token becomes ``filemask-<md5[:8]>`` (a copied arm; a method arm is
      built with its token).
    * ``objects``: ``reset`` is wrapped only to fill in the SAME depth outline when a caller
      hands none (the copied rig check's identity step resets without one); the runner hands
      its own, identical, mask.
    """
    if gm.mode == "none":
        return arm
    if getattr(arm, "goal_mask_mode", "none") not in ("objects", "file"):
        raise ValueError(f"{arm.name}: built without a goal-mask seam (goal_mask_mode "
                         f"{getattr(arm, 'goal_mask_mode', None)!r})")
    mask = np.asarray(gm.mask, dtype=bool)
    inner = arm.reset

    def reset(goal_rgb, goal_depth=None, goal_mask=None):
        if gm.mode == "file" and goal_mask is not None:
            raise ValueError("a file-mask cell was handed a second goal mask by its caller")
        return inner(goal_rgb, goal_depth, mask if goal_mask is None else goal_mask)

    arm.reset = reset
    if gm.mode == "file":
        if arm.goal_mask_mode == "objects":          # a copied arm: relabel its objmask token
            parts = arm.name.split("/")
            if parts.count("objmask") != 1:
                raise ValueError(f"{arm.name}: expected one 'objmask' token in the label")
            arm.name = "/".join(gm.token if p == "objmask" else p for p in parts)
        arm.goal_mask_mode = "file"
    return arm
