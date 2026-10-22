"""PyBullet backend: a ``p.DIRECT`` client holding visual shapes only, rendered by
TinyRenderer (pure software) or, with ``hardware=True``, by the ``eglRendererPlugin``.

The projection matrix is built from ``K`` (``CameraModel.gl_projection``), never from a
field of view, with a half-pixel correction for TinyRenderer's sampling positions, and
the z-buffer is linearised with ``far * near / (far - (far - near) * z)``.
TinyRenderer has one global light and no per-object unlit shader: a scene whose
materials are all ``unlit`` is drawn with the diffuse term switched off, otherwise
``unlit`` is approximated by the ambient term alone.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from vsbench.camera import CameraModel
from vsbench.meshgen import quad
from vsbench.reference import invert
from vsbench.scene import Box, Mesh, Plane, SceneSpec, Sphere, quat_wxyz

_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
# TinyRenderer rasterises at integer window coordinates, not at pixel centres: its
# image lands half a pixel right and half a pixel up of OpenGL's. Measured by the
# projection gate as a (+0.499, -0.501) px centroid offset, so the principal point
# handed to the projection matrix is moved by (-0.5, +0.5) px to cancel it.
_PIXEL_SHIFT = (-0.5, +0.5)
# TinyRenderer samples texel v = 0 at the bottom row of the image it loads (OpenGL
# style); our uvs put v = 0 at the top row, so flip v once when building the mesh.
_FLIP_V = True


def _xyzw(R: np.ndarray) -> list[float]:
    w, x, y, z = quat_wxyz(R)
    return [x, y, z, w]


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


class PyBulletRenderer:
    def __init__(self, scene: SceneSpec, cam: CameraModel, hardware: bool = False):
        import pybullet as p

        self._p = p
        self.cam = cam
        self.scene_spec = scene
        self.name = "pybullet-egl" if hardware else "pybullet-tiny"
        self.cid = p.connect(p.DIRECT)
        self._renderer_flag = p.ER_TINY_RENDERER
        if hardware:
            plugin = p.loadPlugin("eglRendererPlugin", physicsClientId=self.cid)
            if plugin < 0:
                raise RuntimeError("eglRendererPlugin is not available in this pybullet build")
            self._renderer_flag = p.ER_BULLET_HARDWARE_OPENGL
        self._tmp = tempfile.TemporaryDirectory(prefix="vsbench_pb_")
        self._numpy = bool(p.isNumpyEnabled())
        for i, obj in enumerate(scene.objects):
            self._add_object(i, obj)
        self._all_unlit = all(o.material.unlit for o in scene.objects)
        L = scene.lights[0] if scene.lights else None
        self._light_dir = np.array(L.direction if L else (0.0, 0.0, -1.0))
        self._light_color = np.array(L.color if L else (1.0, 1.0, 1.0))
        self._light_intensity = float(L.intensity) if L else 0.0
        K = cam.K.copy()
        K[0, 2] += _PIXEL_SHIFT[0]
        K[1, 2] += _PIXEL_SHIFT[1]
        self._proj = list(CameraModel(K, cam.width, cam.height, cam.near, cam.far).gl_projection().flatten(order="F"))
        self.gl_renderer = "TinyRenderer (software)" if not hardware else "eglRendererPlugin"

    # -- scene building -----------------------------------------------------------------
    def _add_shape(self, kind: str, obj, **kw):
        p = self._p
        vis = p.createVisualShape(getattr(p, kind), rgbaColor=[*obj.material.color, 1.0], physicsClientId=self.cid, **kw)
        return vis

    def _add_body(self, vis: int, pos, R, texture) -> int:
        p = self._p
        body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=list(pos), baseOrientation=_xyzw(R), physicsClientId=self.cid)
        if texture is not None:
            from PIL import Image

            path = os.path.join(self._tmp.name, f"tex_{body}.png")
            Image.fromarray(texture).save(path)
            tex = p.loadTexture(path, physicsClientId=self.cid)
            p.changeVisualShape(body, -1, textureUniqueId=tex, physicsClientId=self.cid)
        return body

    def _add_mesh(self, V, F, UV, pose, obj) -> None:
        p = self._p
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int32)
        kw = dict(vertices=V.tolist(), indices=F.ravel().tolist(), normals=_vertex_normals(V, F).tolist())
        if UV is not None:
            uv = np.asarray(UV, dtype=np.float64).copy()
            if _FLIP_V:
                uv[:, 1] = 1.0 - uv[:, 1]
            kw["uvs"] = uv.tolist()
        vis = p.createVisualShape(p.GEOM_MESH, rgbaColor=[*obj.material.color, 1.0], physicsClientId=self.cid, **kw)
        self._add_body(vis, pose[:3, 3], pose[:3, :3], obj.material.texture)

    def _add_object(self, i: int, obj) -> None:
        if isinstance(obj, Sphere):
            vis = self._add_shape("GEOM_SPHERE", obj, radius=obj.radius)
            self._add_body(vis, obj.center, np.eye(3), None)
        elif isinstance(obj, Box):
            vis = self._add_shape("GEOM_BOX", obj, halfExtents=[s / 2.0 for s in obj.size])
            self._add_body(vis, obj.pose[:3, 3], obj.pose[:3, :3], None)
        elif isinstance(obj, Plane):
            V, F, UV = quad(obj.size)
            self._add_mesh(V, F, UV, obj.pose, obj)
        elif isinstance(obj, Mesh):
            self._add_mesh(obj.vertices, obj.faces, obj.uvs, obj.pose, obj)
        else:
            raise TypeError(type(obj))

    # -- Renderer protocol --------------------------------------------------------------
    def render(self, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self._p
        cam = self.cam
        V = _CV_TO_GL @ invert(np.asarray(T_wc, dtype=np.float64))
        view = list(V.flatten(order="F"))
        diffuse = 0.0 if self._all_unlit else self._light_intensity
        ambient = 1.0 if self._all_unlit else self.scene_spec.ambient
        out = p.getCameraImage(
            cam.width,
            cam.height,
            viewMatrix=view,
            projectionMatrix=self._proj,
            lightDirection=list(-self._light_dir),
            lightColor=list(self._light_color),
            lightDistance=10.0,
            shadow=0,
            lightAmbientCoeff=ambient,
            lightDiffuseCoeff=diffuse,
            lightSpecularCoeff=0.0,
            renderer=self._renderer_flag,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.cid,
        )
        w, h, rgba, zbuf = out[0], out[1], out[2], out[3]
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)
        z = np.asarray(zbuf, dtype=np.float64).reshape(h, w)
        n, f = cam.near, cam.far
        depth = f * n / (f - (f - n) * z)
        depth[z >= 1.0 - 1e-7] = 0.0
        depth[depth >= f * (1.0 - 1e-6)] = 0.0
        return np.ascontiguousarray(rgba[:, :, :3]), depth.astype(np.float32)

    def set_light(self, direction, intensity: float, color=(1.0, 1.0, 1.0)) -> None:
        d = np.asarray(direction, dtype=np.float64)
        self._light_dir = d / np.linalg.norm(d)
        self._light_intensity = float(intensity)
        self._light_color = np.asarray(color, dtype=np.float64)

    def info(self) -> dict:
        import importlib.metadata as md

        try:
            ver = md.version("pybullet")
        except md.PackageNotFoundError:
            ver = "unknown"
        return {"variant": self.name, "gl_renderer": self.gl_renderer, "pybullet": ver, "numpy_enabled": self._numpy}

    def close(self) -> None:
        try:
            self._p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass
        self._tmp.cleanup()
