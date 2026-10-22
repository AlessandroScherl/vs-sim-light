"""VS-Bench: a Renderer interface (MuJoCo, PyBullet, Open3D backends) and a free-flying
camera moved by a 6-DoF velocity command.

Conventions (every backend converts internally, no caller sees an OpenGL frame):

* ``T_wc`` is a 4 x 4 camera-to-world pose. The optical frame is x right, y down,
  z forward (OpenCV).
* ``CameraModel.K`` is in pixels; integer pixel ``(u, v)`` is the pixel centre.
* ``render`` returns ``rgb`` uint8 (H, W, 3) and ``depth`` float32 (H, W) z-depth in
  metres, 0 where nothing was hit.
* A twist is ``[vx, vy, vz, wx, wy, wz]`` in m/s and rad/s, the body twist of the
  camera in its optical frame; ``FreeCamera.step`` applies ``T_wc @ exp_se3(twist * dt)``.
"""

from vsbench.camera import CameraModel
from vsbench.motion import FreeCamera, MotionLimits
from vsbench.scene import Box, Light, Material, Mesh, Plane, SceneSpec, Sphere
from vsbench.trajectory import Trajectory

__all__ = [
    "CameraModel",
    "SceneSpec",
    "Plane",
    "Box",
    "Sphere",
    "Mesh",
    "Light",
    "Material",
    "FreeCamera",
    "MotionLimits",
    "Trajectory",
]
__version__ = "0.1.0"
