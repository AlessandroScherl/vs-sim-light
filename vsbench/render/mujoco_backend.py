"""MuJoCo backend: an ``MjSpec`` scene, the camera on a mocap body, ``mjr_render`` plus
``mjr_readPixels`` per frame, and when ``samples > 0`` a second render into a
non-multisampled context for the depth image.

``MUJOCO_GL`` picks the context: ``egl`` (GPU, or Mesa llvmpipe when
``VSBENCH_SOFTWARE_GL=1``), ``osmesa`` (needs libOSMesa) or ``glfw`` (needs a display).
It must be set before ``import mujoco``; asking for a second value in one process fails.

Depth comes back from the reversed-Z buffer (``mjDEPTH_ZEROFAR``) and is linearised with
the same float32 frustum coefficients MuJoCo's own ``Renderer`` uses. The clipping
planes are set through ``vis.map.znear/zfar`` which MuJoCo scales by ``stat.extent``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from vsbench.camera import CameraModel
from vsbench.meshgen import quad
from vsbench.reference import invert  # noqa: F401  (kept for symmetry with other backends)
from vsbench.scene import Box, Light, Mesh, Plane, SceneSpec, Sphere, quat_wxyz

_CV_TO_GL = np.diag([1.0, -1.0, -1.0])  # optical (y down, z forward) -> MuJoCo camera (y up, z back)
# MuJoCo user-mesh texcoords already put v = 0 at the top row of the picture, like ours:
# no flip (checked by the homography gate; OBJ files are flipped by MuJoCo itself).
_FLIP_V = False


class MujocoRenderer:
    def __init__(self, scene: SceneSpec, cam: CameraModel, gl: str = "egl", samples: int = 4):
        want = gl.lower()
        have = os.environ.get("MUJOCO_GL", "").lower()
        if "mujoco" in sys.modules and have and have != want:
            raise RuntimeError(f"mujoco already imported with MUJOCO_GL={have!r}; one GL backend per process")
        os.environ["MUJOCO_GL"] = want
        # VSBENCH_SOFTWARE_GL is honoured here too, not only in make_renderer: a caller that
        # builds this class directly (tools/convert_objects.py's contact sheet did) would
        # otherwise let glvnd hand MuJoCo's EGL the first device, the GPU on the station
        from vsbench.glenv import force_software_gl, software_gl_requested

        if software_gl_requested() and want == "egl":
            force_software_gl()
        import mujoco

        self._mj = mujoco
        self.name = f"mujoco-{want}"
        self.cam = cam
        self.scene_spec = scene
        self._light_ids: list[int] = []

        spec = mujoco.MjSpec()
        spec.modelname = "vsbench"
        g = spec.visual.global_
        g.offwidth = max(int(cam.width), 640)
        g.offheight = max(int(cam.height), 480)
        spec.visual.quality.offsamples = int(samples)
        spec.visual.quality.shadowsize = 4096
        hl = spec.visual.headlight
        hl.active = 1
        hl.ambient = (scene.ambient,) * 3
        hl.diffuse = (0.0, 0.0, 0.0)
        hl.specular = (0.0, 0.0, 0.0)
        if any(c > 0 for c in scene.background):
            sky = spec.add_texture(name="vsbench_sky")
            sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
            sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
            sky.rgb1 = scene.background
            sky.rgb2 = scene.background
            sky.width = 32
            sky.height = 32

        for i, L in enumerate(scene.lights):
            self._add_light(spec, i, L)

        for i, obj in enumerate(scene.objects):
            self._add_object(spec, i, obj)

        # camera on a mocap body, intrinsics in pixels
        body = spec.worldbody.add_body(name="vsbench_cam_body")
        body.mocap = True
        body.pos = (0.0, 0.0, 1.0)
        c = body.add_camera(name="vsbench_cam")
        c.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        W, H = cam.width, cam.height
        c.resolution = (W, H)
        c.sensor_size = (W / 1000.0, H / 1000.0)  # a 1 mm pixel; only the ratio matters
        c.focal_pixel = (cam.fx, cam.fy)
        # MuJoCo's principal point is an offset from the image centre, positive to the
        # left and up (see getFrustum in engine_vis_visualize.c); OpenCV's is absolute
        # with integer indices at pixel centres.
        c.principal_pixel = (W / 2.0 - cam.cx - 0.5, H / 2.0 - cam.cy - 0.5)

        self.model = spec.compile()
        model = self.model
        model.vis.map.znear = cam.near / model.stat.extent
        model.vis.map.zfar = cam.far / model.stat.extent
        self.data = mujoco.MjData(model)
        self._cam_id = model.camera("vsbench_cam").id
        self._mocap_id = model.body("vsbench_cam_body").mocapid[0]

        self._gl = mujoco.GLContext(g.offwidth, g.offheight)
        self._gl.make_current()
        self._scn = mujoco.MjvScene(model, maxgeom=len(scene.objects) + 64)
        self._ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self._ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._ctx)
        self._samples = int(samples)
        self._ctx_depth = self._ctx
        if self._samples > 0:
            # Depth is read from a second render into a context whose offscreen buffer is
            # not multisampled: a multisampled depth buffer resolves to one sample off
            # the pixel centre (3.9 mm at the 99th percentile on a 60 degree plane), and
            # NVIDIA keeps sample positions even for MuJoCo's segmentation pass.
            model.vis.quality.offsamples = 0
            self._ctx_depth = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._ctx_depth.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._ctx_depth)
            model.vis.quality.offsamples = self._samples
            # leave the colour context's framebuffer bound: mjr_render draws into
            # whatever mjr_setBuffer bound last
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._ctx)
        self._rect = mujoco.MjrRect(0, 0, W, H)
        self._opt = mujoco.MjvOption()
        self._vcam = mujoco.MjvCamera()
        self._vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._vcam.fixedcamid = self._cam_id
        self._shadows = any(L.cast_shadow for L in scene.lights)
        self._rgb = np.empty((H, W, 3), dtype=np.uint8)
        self._depth = np.empty((H, W), dtype=np.float32)
        # depth linearisation, exactly as mujoco.Renderer does it
        zfar, znear = np.float32(cam.far), np.float32(cam.near)
        c0 = -(zfar + znear) / (zfar - znear)
        d0 = -(np.float32(2) * zfar * znear) / (zfar - znear)
        self._c = np.float32(-0.5) * c0 - np.float32(0.5)
        self._d = np.float32(-0.5) * d0
        try:
            from OpenGL import GL

            self.gl_renderer = (GL.glGetString(GL.GL_RENDERER) or b"").decode()
        except Exception:  # pragma: no cover - PyOpenGL missing or no context
            self.gl_renderer = "unknown"

    # -- scene building -----------------------------------------------------------------
    def _add_light(self, spec, i: int, L: Light) -> None:
        mujoco = self._mj
        light = spec.worldbody.add_light(name=f"vsbench_light{i}")
        if hasattr(mujoco, "mjtLightType"):
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        else:  # pragma: no cover - MuJoCo < 3.3
            light.directional = True
        light.pos = tuple(-3.0 * np.asarray(L.direction))
        light.dir = tuple(L.direction)
        light.diffuse = tuple(L.intensity * np.asarray(L.color))
        light.specular = (0.0, 0.0, 0.0)
        light.ambient = (0.0, 0.0, 0.0)
        light.castshadow = bool(L.cast_shadow)

    def _material(self, spec, name: str, mat) -> str:
        mujoco = self._mj
        m = spec.add_material(name=name)
        m.rgba = (*mat.color, 1.0)
        m.specular = 0.0
        m.shininess = 0.0
        m.reflectance = 0.0
        # unlit: the ambient term is the only shading, so emission fills the rest
        m.emission = float(max(0.0, 1.0 - self.scene_spec.ambient)) if mat.unlit else 0.0
        if mat.texture is not None:
            tex = spec.add_texture(name=name + "_tex")
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            img = np.ascontiguousarray(mat.texture)
            tex.height, tex.width, tex.nchannel = int(img.shape[0]), int(img.shape[1]), 3
            tex.data = img.tobytes()
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            m.texrepeat = (1.0, 1.0)
            m.texuniform = False
        return m.name

    def _add_mesh_geom(self, spec, name: str, V, F, UV, pose, matname: str) -> None:
        mujoco = self._mj
        mesh = spec.add_mesh(name=name + "_mesh")
        mesh.uservert = np.asarray(V, dtype=np.float32).ravel().tolist()
        mesh.userface = np.asarray(F, dtype=np.int32).ravel().tolist()
        if UV is not None:
            uv = np.asarray(UV, dtype=np.float32).copy()
            if _FLIP_V:
                uv[:, 1] = 1.0 - uv[:, 1]
            mesh.usertexcoord = uv.ravel().tolist()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        geom = spec.worldbody.add_geom(name=name)
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
        geom.material = matname
        geom.pos = tuple(pose[:3, 3])
        geom.quat = tuple(quat_wxyz(pose[:3, :3]))
        geom.contype = 0
        geom.conaffinity = 0

    def _add_object(self, spec, i: int, obj) -> None:
        mujoco = self._mj
        name = f"vsbench_obj{i}"
        matname = self._material(spec, name + "_mat", obj.material)
        if isinstance(obj, Sphere):
            g = spec.worldbody.add_geom(name=name)
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = (obj.radius, 0.0, 0.0)
            g.pos = obj.center
            g.material = matname
            g.contype = g.conaffinity = 0
        elif isinstance(obj, Box):
            g = spec.worldbody.add_geom(name=name)
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = tuple(s / 2.0 for s in obj.size)
            g.pos = tuple(obj.pose[:3, 3])
            g.quat = tuple(quat_wxyz(obj.pose[:3, :3]))
            g.material = matname
            g.contype = g.conaffinity = 0
        elif isinstance(obj, Plane):
            V, F, UV = quad(obj.size)
            self._add_mesh_geom(spec, name, V, F, UV, obj.pose, matname)
        elif isinstance(obj, Mesh):
            self._add_mesh_geom(spec, name, obj.vertices, obj.faces, obj.uvs, obj.pose, matname)
        else:
            raise TypeError(type(obj))

    # -- Renderer protocol --------------------------------------------------------------
    def render(self, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mujoco = self._mj
        T = np.asarray(T_wc, dtype=np.float64)
        self.data.mocap_pos[self._mocap_id] = T[:3, 3]
        self.data.mocap_quat[self._mocap_id] = quat_wxyz(T[:3, :3] @ _CV_TO_GL)
        mujoco.mj_forward(self.model, self.data)
        self._gl.make_current()
        mujoco.mjv_updateScene(self.model, self.data, self._opt, None, self._vcam, mujoco.mjtCatBit.mjCAT_ALL, self._scn)
        f = self._scn.flags
        f[mujoco.mjtRndFlag.mjRND_SHADOW] = self._shadows
        f[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        f[mujoco.mjtRndFlag.mjRND_FOG] = False
        f[mujoco.mjtRndFlag.mjRND_HAZE] = False
        f[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        if self._ctx_depth is not self._ctx:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._ctx)
        mujoco.mjr_render(self._rect, self._scn, self._ctx)
        if self._samples > 0:
            # colour from the multisampled context, depth from a second, single-sample
            # render in segmentation mode (no lighting work) into the depth context
            mujoco.mjr_readPixels(self._rgb, None, self._rect, self._ctx)
            f[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
            f[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._ctx_depth)
            mujoco.mjr_render(self._rect, self._scn, self._ctx_depth)
            mujoco.mjr_readPixels(None, self._depth, self._rect, self._ctx_depth)
            f[mujoco.mjtRndFlag.mjRND_SEGMENT] = False
            f[mujoco.mjtRndFlag.mjRND_IDCOLOR] = False
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._ctx)
        else:
            mujoco.mjr_readPixels(self._rgb, self._depth, self._rect, self._ctx)
        rgb = np.flipud(self._rgb).copy()
        z = np.flipud(self._depth).astype(np.float64)
        depth = self._d / (z + self._c)
        depth[z <= 0.0] = 0.0  # reversed Z: 0 is the far plane, nothing hit
        depth[depth >= self.cam.far * (1.0 - 1e-6)] = 0.0
        return rgb, depth.astype(np.float32)

    def set_light(self, direction, intensity: float, color=(1.0, 1.0, 1.0)) -> None:
        if self.model.nlight == 0:
            raise RuntimeError("scene has no light to move")
        d = np.asarray(direction, dtype=np.float64)
        d /= np.linalg.norm(d)
        self.model.light_dir[0] = d
        self.model.light_diffuse[0] = np.asarray(color, dtype=np.float64) * float(intensity)

    def info(self) -> dict:
        return {"variant": self.name, "gl_renderer": self.gl_renderer, "mujoco": self._mj.__version__, "MUJOCO_GL": os.environ.get("MUJOCO_GL"), "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID")}

    def close(self) -> None:
        for ctx in {id(self._ctx): self._ctx, id(self._ctx_depth): self._ctx_depth}.values():
            try:
                ctx.free()
            except Exception:
                pass
        try:
            self._gl.free()
        except Exception:
            pass
