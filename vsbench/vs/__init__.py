# vs-sim copy: vssim/__init__.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""vssim - a visual-servoing simulator with a frozen six-scene benchmark.

A free-flying D435i camera in Isaac Sim over six frozen scenes (three textured
posters, three cluttered 3-D rooms), a numpy mock of the same camera for tests,
and the frozen start poses every method is flown from. Four verbs and four facts
are the whole contract (:class:`vsbench.vs.scene.Scene`): ``reset``, ``render``,
``apply``, plus ``intrinsics``, ``goal_pose``, ``working_distance`` and ``pose``.

=================== ==========================================================
module              what it owns
=================== ==========================================================
``scene``           the ``Scene`` protocol, ``Pose``, the twist integration, the
                    numpy ``MockPlaneScene``
``isaac_scene``     ``IsaacScene`` - the real world (needs ``$ISAAC_PY``)
``camera``          ``configs/camera.yaml`` -> intrinsics
``scenes``          the scene specs, goal poses, scatter, baked layouts
``assets``          the asset library and USD path resolution
``labels``          ground-truth correspondence from depth + poses
``bake``            settle a new scene once and freeze its layout
``starts``          the start-pose samplers and the start-file format
``benchmark``       the six scenes x their pinned start files
=================== ==========================================================
"""

from vsbench.vs.scene import (
    D435I,
    MockPlaneScene,
    Pose,
    Scene,
    integrate_optical_velocity,
    rotation_error_deg,
    translation_error_mm,
)

__version__ = "0.1.0"

__all__ = ["D435I", "MockPlaneScene", "Pose", "Scene", "integrate_optical_velocity",
           "rotation_error_deg", "translation_error_mm", "__version__"]
