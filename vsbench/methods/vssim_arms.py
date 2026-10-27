"""The baselines as vsbench methods: drive them in your own loop.

``make_method("sift_ransac")`` (or ``vitvs_dinov3``, ``vitvs_dinov2``, ``artvs``,
``orb_ransac``, ``akaze_ransac``) is the published baseline the benchmark flies, behind the
``VSMethod`` contract, so you can call it yourself: ``configure`` once, ``set_target`` with the
goal picture, then ``get_control_rate`` on every live frame and apply the twist it returns
(README, "Drive the camera yourself"). No baseline code lives here: each method is the protocol
cell of the same name (``vsbench/configs/protocol.yaml``: its ``configs/arms/*.yaml`` file
deep-merged with the cell's deltas, the copied strict loader) built by the copied
``build_arm``, with one stated override, ``matcher.device`` (``auto``: CUDA when available,
the CPU in station mode; only for the arms that load a network).

The adapter, call for call:

* ``configure(K, width, height, dt)`` -> the copied ``Intrinsics`` and ``build_arm``. The
  arm (and its encoder weights) is built once per camera and reused across episodes.
* ``set_target(desired_rgb, desired_depth)`` -> ``arm.set_episode(episode, start_id)`` and
  ``arm.reset(goal_rgb, goal_depth)``. The landmark draw of the ViT arms is seeded by
  ``(episode, start_id, seed)`` (the copied ``episode_seed``); both are constructor
  parameters (default ``"vsbench"`` and ``0``).
* ``get_control_rate(rgb, depth)`` -> ``arm.step(rgb, depth)``, then the twist scaled by
  the arm's scored stop rule exactly as the benchmark loop scales it (the rule's
  ``lambda_scale``; 1.0 for ``cns_plateau``, the rule these six arms score), and the info
  dict: ``kp_cur`` / ``kp_des`` are the pixels the control law solved on, ``error_norm`` the
  copied law's ``feature_error_norm``, ``observer_s`` the matcher time, ``controller_s`` the
  solve time, ``lost`` the law's degenerate flag (fewer than ``min_matches`` rows: the copied
  law commands a zero twist).
* ``wants_depth = "dense"``: the copied arm reads the live depth map at its current pixels
  (truncating; nothing hit / invalid -> the copied 0.5 m fallback).

``scene_ransac`` (default ``homography``: a poster is a plane) answers the one scene question
the classical cells ask (``matcher.ransac: scene_default`` -> the scene's model,
``configs/scenes.yaml``); pass ``fundamental`` for a 3-D scene. It is recorded in ``params``.

The benchmark itself (``python -m vsbench fly``) flies these baselines as protocol cells
(``--arms``), never through this wrapper; a loop of your own is not the benchmark loop, and
its numbers are not the benchmark's.

Nothing here imports a loop, a renderer or a scene at import time: the copied bench and the
protocol reader are imported inside ``configure``, so ``import vsbench.methods`` stays light.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from vsbench.methods import register

#: The protocol cells registered as vsbench methods (``configs/protocol.yaml`` ids).
ARM_IDS: tuple[str, ...] = ("vitvs_dinov3", "vitvs_dinov2", "artvs",
                            "sift_ransac", "orb_ransac", "akaze_ransac")
#: a poster is a plane: the model a planar scene states in ``scenes.yaml``.
DEFAULT_SCENE_RANSAC = "homography"


#: The encoder each ViT cell loads (its arm file's ``matcher.backbone_name``).
BACKBONE_OF = {"vitvs_dinov3": "dinov3_vits16", "vitvs_dinov2": "dinov2_vits14",
               "artvs": "dinov2_vits14"}
#: Where the offline weights must sit (``local_files_only`` / ``torch.hub`` from cache):
#: under ``$HF_HOME/hub`` and ``$TORCH_HOME/hub`` (default ``~/.cache/huggingface`` / ``~/.cache/torch``).
WEIGHT_FILES = {
    "dinov3_vits16": ("hf", "models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots"),
    "dinov2_vits14": ("torch", "checkpoints/dinov2_vits14_pretrain.pth",
                      "facebookresearch_dinov2_main"),
}


def _hub_dir(kind: str):
    import os
    from pathlib import Path

    if kind == "hf":
        root = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    else:
        root = os.environ.get("TORCH_HOME") or str(Path.home() / ".cache" / "torch")
    return Path(root) / "hub"


def unavailable_reason(arm_id: str) -> str | None:
    """Why the copied arm cannot run in this interpreter, or ``None`` when it can.

    Nothing is imported or loaded: the copied bench needs ``torch`` (every arm, the matcher
    module imports it), the DINOv3 cell ``transformers`` too, the ViT cells their weights in
    the caches (``$HF_HOME`` / ``$TORCH_HOME``), the AKAZE cell an OpenCV that has
    ``AKAZE_create``. The tests skip an arm that says why.
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return "needs torch (the copied vs-sim bench imports it); not installed here"
    if arm_id == "akaze_ransac":
        import cv2

        if not hasattr(cv2, "AKAZE_create"):
            return f"OpenCV {cv2.__version__} has no AKAZE_create (pin 4.11.0.86)"
    backbone = BACKBONE_OF.get(arm_id)
    if backbone is None:
        return None
    if backbone.startswith("dinov3") and importlib.util.find_spec("transformers") is None:
        return "needs transformers for DINOv3; not installed here"
    kind, *paths = WEIGHT_FILES[backbone]
    hub = _hub_dir(kind)
    missing = [p for p in paths if not (hub / p).exists()]
    if missing:
        return (f"the {backbone} weights are not in {hub} (missing {', '.join(missing)}; "
                "README, Weights: download them once)")
    return None


def _ro(a):
    """A read-only view (or ``None``)."""
    if a is None:
        return None
    v = np.asarray(a).view()
    v.setflags(write=False)
    return v


class VssimArmMethod:
    """One copied vs-sim arm behind the ``VSMethod`` contract. Subclassed per arm id."""

    arm_id: str = ""
    wants_depth = "dense"
    privileged = False
    #: tells ``vsbench.world.method_arm.MethodArm`` that ``kp_cur`` / ``kp_des`` come with
    #: every step (so its ``cns_plateau`` stop can run)
    reports_points = True

    @classmethod
    def unavailable(cls) -> str | None:
        """Why this arm cannot run here (``None``: it can) - see :func:`unavailable_reason`."""
        return unavailable_reason(cls.arm_id)

    def __init__(self, device: str = "auto", scene_ransac: str = DEFAULT_SCENE_RANSAC,
                 episode: str = "vsbench", start_id=0):
        if not self.arm_id:
            raise TypeError("VssimArmMethod is a base: use a registered arm id")
        if scene_ransac not in ("homography", "fundamental"):
            raise ValueError("scene_ransac must be 'homography' or 'fundamental'")
        self.device = str(device)
        self.scene_ransac = str(scene_ransac)
        self.episode = str(episode)
        self.start_id = start_id
        self.params = {"arm": self.arm_id, "device": self.device,
                       "scene_ransac": self.scene_ransac, "episode": self.episode,
                       "start_id": start_id}
        self._arm = None
        self._key = None
        self._targeted = False
        self.config: dict | None = None
        self.label: str | None = None

    # -- contract ---------------------------------------------------------------------
    def configure(self, K, width, height, dt) -> None:
        K = np.asarray(K, dtype=np.float64)
        key = (tuple(float(x) for x in K.reshape(-1)), int(width), int(height))
        if self._arm is not None and key == self._key:
            return
        self.close()
        from vsbench.runtime import resolve_device
        from vsbench.vs.baselines.control import Intrinsics
        from vsbench.vs.bench.arms import build_arm
        from vsbench.world.fly import arm_config, arm_entry, load_protocol

        protocol = load_protocol()
        device = resolve_device(self.device) if self.device in ("auto", "cuda", "cpu") else self.device
        self.resolved_device = device
        cfg = arm_config(protocol, self.arm_id, device=device)
        intr = Intrinsics(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]),
                          cy=float(K[1, 2]), width=int(width), height=int(height))
        overrides = {}
        if "scene_spec" in set(arm_entry(protocol, self.arm_id).get("needs") or []):
            # the one scene fact a classical cell reads (`matcher.ransac: scene_default`)
            overrides["scene_spec"] = SimpleNamespace(name="vsbench", ransac=self.scene_ransac,
                                                      prompt=None)
        self._arm = build_arm(cfg, intrinsics=intr, **overrides)
        self._key = key
        self.config = cfg
        self.label = self._arm.name

    def reset(self) -> None:
        self._targeted = False

    def set_target(self, desired_rgb, desired_depth=None, distance_prior=None) -> None:
        if self._arm is None:
            raise RuntimeError(f"{self.arm_id}: configure first")
        self._arm.set_episode(self.episode, self.start_id)
        self._arm.reset(_ro(desired_rgb), _ro(desired_depth), None)
        self._targeted = True

    def get_control_rate(self, current_rgb, current_depth=None):
        if self._arm is None or not self._targeted:
            raise RuntimeError(f"{self.arm_id}: configure and set_target first")
        if current_depth is None:
            raise RuntimeError(f"{self.arm_id} reads the live depth map (wants_depth='dense')")
        from vsbench.vs.baselines.control import StopObservation

        arm = self._arm
        t0 = time.perf_counter()
        twist, debug, _matches = arm.step(current_rgb, current_depth)
        # the benchmark loop's step 1: the scored rule may scale the commanded twist
        # (`lambda_scale`; 1.0 under cns_plateau, the rule all six arms score)
        verdict = arm.stop.update(StopObservation.from_debug(
            debug, twist=twist, dt=float(getattr(arm.stop, "dt", arm.dt))))
        v = twist.as_array() * float(verdict.lambda_scale)
        wall = time.perf_counter() - t0
        pc = np.asarray(debug.get("pts_cur", np.zeros((0, 2))), np.float64).reshape(-1, 2)
        pg = np.asarray(debug.get("pts_goal", np.zeros((0, 2))), np.float64).reshape(-1, 2)
        match_s = float(debug.get("match_ms", 0.0)) / 1000.0
        info = {"kp_cur": pc, "kp_des": pg, "inliers": np.ones(len(pc), dtype=bool),
                "error_norm": float(debug.get("feature_error_norm", np.nan)),
                "observer_s": match_s, "controller_s": max(0.0, wall - match_s),
                "lost": bool(debug.get("degenerate", len(pc) == 0))}
        return np.array(v, dtype=np.float64), info

    def close(self) -> None:
        arm, self._arm = self._arm, None
        self._key = None
        if arm is not None:
            arm.close()


def _make(arm_id: str) -> type:
    cls = type(f"Vssim_{arm_id}", (VssimArmMethod,),
               {"arm_id": arm_id, "__module__": __name__,
                "__doc__": f"The copied vs-sim protocol cell ``{arm_id}`` as a vsbench method."})
    return register(arm_id)(cls)


for _arm_id in ARM_IDS:
    _make(_arm_id)
