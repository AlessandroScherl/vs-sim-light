"""Open3D backend: ``OffscreenRenderer`` over Filament. ``cpu=True`` sets
``EGL_PLATFORM=surfaceless`` before the import, which on Linux gives Mesa software
rendering; on macOS and Windows Open3D always uses the GPU.

The camera is set with ``Camera.set_projection(K, near, far, W, H)`` (an intrinsic matrix,
not a field of view) and ``Camera.look_at``. Depth comes from
``render_to_depth_image(z_in_view_space=True)``. Post-processing is off so colours are
the raw shader output; ``unlit`` materials use the ``defaultUnlit`` shader.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from vsbench.camera import CameraModel
from vsbench.meshgen import quad
from vsbench.scene import Box, Mesh, Plane, SceneSpec, Sphere

# Filament reads texel v = 0 at the bottom row of the image; ours is the top row.
_FLIP_V = True
# With post-processing off, Open3D's unlit output is 0.5 * srgb_decode(colour), for flat
# colours and albedo textures alike (measured: base colour 0.5 -> 27, 1.0 -> 128; the
# poster's 128 -> 28). To hand back the raw colour, encode it first: a base colour of
# srgb_encode(2 c) decodes to 2 c and comes out as c, and a texture stored as
# srgb_encode(t) under a base colour of srgb_encode(2) comes out as t.
_UNLIT_ENCODE = True
# Destroying an OffscreenRenderer while another one is alive aborts the process
# (Filament: "destroying MaterialInstance ... still in use"), so instances are kept
# until exit. One renderer per process is the contract anyway.
_KEEP_ALIVE: list = []
# Open3D's projection puts the principal point at continuous window coordinate cx, while
# OpenCV's integer index is a pixel centre, so add half a pixel.
_HALF_PIXEL = 0.5
SUN_INTENSITY = 230000.0  # lux for intensity 1.0: a lit grey floor then matches MuJoCo within a few percent
INDIRECT_INTENSITY = 100000.0  # for ambient 1.0


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0), 1 / 2.4) - 0.055)


class Open3DRenderer:
    def __init__(self, scene: SceneSpec, cam: CameraModel, cpu: bool = True):
        if cpu and sys.platform.startswith("linux"):
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        import open3d as o3d

        _KEEP_ALIVE.append(self)

        self._o3d = o3d
        self.cam = cam
        self.scene_spec = scene
        self.name = "open3d-cpu" if cpu else "open3d-gpu"
        self.ignored_lights = 0
        W, H = cam.width, cam.height
        self._r = o3d.visualization.rendering.OffscreenRenderer(W, H)
        s = self._r.scene
        s.set_background([*scene.background, 1.0])
        s.view.set_post_processing(False)
        s.scene.enable_indirect_light(True)
        s.scene.set_indirect_light_intensity(float(scene.ambient) * INDIRECT_INTENSITY)
        if scene.lights:
            L = scene.lights[0]
            s.scene.set_sun_light(list(L.direction), list(L.color), float(L.intensity) * SUN_INTENSITY)
            s.scene.enable_sun_light(True)
            # Filament has exactly one directional light (the sun); further directional
            # lights are accepted by the API but do not light anything (measured: mean
            # luminance unchanged), so they are reported and ignored here.
            for i, L2 in enumerate(scene.lights[1:], start=1):
                s.scene.add_directional_light(f"vsbench_light{i}", list(L2.color), list(L2.direction), float(L2.intensity) * SUN_INTENSITY, bool(L2.cast_shadow))
                self.ignored_lights = i
        else:
            s.scene.enable_sun_light(False)
        s.view.set_shadowing(False)
        for i, obj in enumerate(scene.objects):
            self._add_object(i, obj)
        K = cam.K.copy()
        K[0, 2] += _HALF_PIXEL
        K[1, 2] += _HALF_PIXEL
        s.camera.set_projection(K, cam.near, cam.far, float(W), float(H))
        self.gl_renderer = "Filament headless (Vulkan on llvmpipe per Open3D's log)" if cpu else "Filament (GPU)"

    # -- scene building -----------------------------------------------------------------
    def _material(self, mat):
        o3d = self._o3d
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit" if mat.unlit else "defaultLit"
        color = np.asarray(mat.color, dtype=np.float64)
        tex = mat.texture
        if mat.unlit and _UNLIT_ENCODE:
            if tex is None:
                color = _srgb_encode(2.0 * color)
            else:
                color = _srgb_encode(2.0 * color)  # (1, 1, 1) -> encode(2)
                tex = (255.0 * _srgb_encode(tex.astype(np.float64) / 255.0) + 0.5).astype(np.uint8)
        m.base_color = [*color.tolist(), 1.0]
        m.base_roughness = 1.0
        m.base_metallic = 0.0
        m.base_reflectance = 0.0
        if tex is not None:
            m.albedo_img = o3d.geometry.Image(np.ascontiguousarray(tex))
        return m

    def _add_mesh(self, name: str, V, F, UV, pose, obj) -> None:
        o3d = self._o3d
        V = np.array(V, dtype=np.float64)  # copies: Open3D wants writeable arrays
        F = np.array(F, dtype=np.int32)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        mesh.compute_vertex_normals()
        if UV is not None and obj.material.texture is not None:
            uv = np.asarray(UV, dtype=np.float64).copy()
            if _FLIP_V:
                uv[:, 1] = 1.0 - uv[:, 1]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uv[F.ravel()])
        mesh.transform(np.asarray(pose, dtype=np.float64))
        self._r.scene.add_geometry(name, mesh, self._material(obj.material))

    def _add_object(self, i: int, obj) -> None:
        o3d = self._o3d
        name = f"vsbench_{id(self):x}_obj{i}"  # unique across renderers in one process
        if isinstance(obj, Sphere):
            mesh = o3d.geometry.TriangleMesh.create_sphere(obj.radius, resolution=48)
            mesh.compute_vertex_normals()
            mesh.translate(np.asarray(obj.center, dtype=np.float64))
            self._r.scene.add_geometry(name, mesh, self._material(obj.material))
        elif isinstance(obj, Box):
            sx, sy, sz = obj.size
            mesh = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
            mesh.compute_vertex_normals()
            mesh.translate(np.array([-sx / 2, -sy / 2, -sz / 2]))
            mesh.transform(np.asarray(obj.pose, dtype=np.float64))
            self._r.scene.add_geometry(name, mesh, self._material(obj.material))
        elif isinstance(obj, Plane):
            V, F, UV = quad(obj.size)
            self._add_mesh(name, V, F, UV, obj.pose, obj)
        elif isinstance(obj, Mesh):
            self._add_mesh(name, obj.vertices, obj.faces, obj.uvs, obj.pose, obj)
        else:
            raise TypeError(type(obj))

    # -- Renderer protocol --------------------------------------------------------------
    def render(self, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(T_wc, dtype=np.float64)
        eye = T[:3, 3]
        center = eye + T[:3, 2]
        up = -T[:3, 1]
        self._r.scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())
        rgb = np.asarray(self._r.render_to_image())
        depth = np.asarray(self._r.render_to_depth_image(z_in_view_space=True), dtype=np.float64)
        depth = np.where(np.isfinite(depth), depth, 0.0)
        depth[depth >= self.cam.far * (1.0 - 1e-6)] = 0.0
        depth[depth <= 0.0] = 0.0
        return np.ascontiguousarray(rgb[:, :, :3]).astype(np.uint8), depth.astype(np.float32)

    def set_light(self, direction, intensity: float, color=(1.0, 1.0, 1.0)) -> None:
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        s = self._r.scene.scene
        s.set_sun_light(d.tolist(), list(color), float(intensity) * SUN_INTENSITY)
        s.enable_sun_light(True)

    def info(self) -> dict:
        return {"variant": self.name, "gl_renderer": self.gl_renderer, "open3d": self._o3d.__version__, "EGL_PLATFORM": os.environ.get("EGL_PLATFORM")}

    def close(self) -> None:
        # No explicit release: clearing the scene or destroying the OffscreenRenderer
        # while another exists crashes Filament, so the instance stays until exit.
        pass
