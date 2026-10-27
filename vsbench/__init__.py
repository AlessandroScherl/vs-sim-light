"""vsbench: a lightweight visual-servoing simulator on MuJoCo.

A free-flying camera (an Intel RealSense D435i model) over textured scenes. The world is
``vsbench.world.mujoco_scene.MujocoScene``; the benchmark loop, scoring and the published
baselines are the copied vs-sim modules under ``vsbench.vs``; your own method is a class
registered with ``vsbench.methods.register``.

Conventions:

* poses are camera-to-world; the camera's optical frame is x right, y down, z forward
  (OpenCV), lengths in metres;
* ``K`` is in pixels and integer pixel ``(u, v)`` is the pixel centre;
* a frame is ``rgb`` uint8 (H, W, 3) and ``depth`` float32 (H, W) z-depth in metres, 0 where
  nothing was hit;
* a twist is ``[vx, vy, vz, wx, wy, wz]`` in m/s and rad/s, the camera's velocity in its own
  optical frame (``docs/IBVS.md`` says exactly how the world applies it).
"""

__version__ = "0.2.0"
