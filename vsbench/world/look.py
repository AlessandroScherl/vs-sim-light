"""The look of a MuJoCo picture: ``isaac_fit_v1`` (default for every flight) or ``raw``.

Owner decision 2026-10-23 (port unit U4b): make the MuJoCo pictures look like Isaac's - "fit
one brightness/tone curve per scene to the Isaac goal pictures already on disk, raise the
light, and use a finer, softer shadow map". The geometry, the depth and every pose are the
same in both looks; only the colour picture differs.

``raw`` - the U1/U4 picture, bit for bit: ambient 0.2 plus one sun whose diffuse term renders
a lit white texel white (``vsbench.world.build.poster_lights``), MuJoCo's default 4096 shadow
map over ``vis.map.shadowclip = 1`` (one ``stat.extent``: 7.4 m across a room, 1.8 mm per
texel), no curve. Used by the gates that compare against the copied mock (U1's phase and tone
gates) and kept for anyone who needs the texture's own tone.

``isaac_fit_v1`` - read from ``vsbench/world/configs/look_isaac_fit_v1.json`` (written by
``tools/u4b_look.py fit``; the Isaac source pictures' md5s are in it):

* light: a fill (ambient) share ``a`` measured from Isaac's own shadows and shaded sides, and a
  per-scene gain ``G`` (the light raised as far as it goes without clipping more than the
  config's budget of the goal picture - Isaac's own brightest pixel is ~250/255, it never
  clips - and never past the point where MuJoCo saturates the sun's diffuse term): ambient
  ``a G``, sun ``(1 - a) G / cos(incidence)``;
* shadows: an 8192 map over a square fitted to the scene (all of its geometry seen from the
  sun, 2 % margin; 0.41 mm per texel in a room) - fine enough that the GL's filtered shadow
  compare draws a smooth edge instead of 1.8 mm stair steps;
* a per-scene, per-channel monotone tone curve applied to the RGB (never to depth):
  ``y = 255 * srgb_encode(v / (1 + v))``, ``v = floor + exp(log_exposure) * x ** gamma``,
  ``x = raw / 255`` - an exposure, a gamma, a floor and a Reinhard tone map, three numbers
  per channel, monotone by construction, stored with its 256-entry table and that table's
  md5 (the curve id is ``isaac_fit_v1/<scene>``).

Every record says which look it was drawn with (:meth:`Look.record`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOOKS = ("isaac_fit_v1", "raw")
DEFAULT_LOOK = "isaac_fit_v1"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
#: the raw look's shadow map (MuJoCo's default in vsbench's backend since Step 1)
RAW_SHADOWSIZE = 4096


def config_path(name: str) -> Path:
    return CONFIG_DIR / f"look_{name}.json"


def file_md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------- the curve
def srgb_encode(v) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def curve_model(params, x) -> np.ndarray:
    """The tone curve on ``x`` in [0, 1] -> grey levels in [0, 255] (float).

    ``params = (floor, log_exposure, gamma)``; strictly increasing for ``exp(log_exposure)
    > 0`` and ``gamma > 0``, whatever ``floor >= 0``.
    """
    floor, log_e, gamma = (float(p) for p in params)
    v = floor + np.exp(log_e) * np.power(np.clip(np.asarray(x, np.float64), 0.0, 1.0), gamma)
    return 255.0 * srgb_encode(v / (1.0 + v))


def curve_lut(channel_params) -> np.ndarray:
    """(256, 3) uint8: the curve of each channel at the 256 input levels, rounded."""
    x = np.arange(256, dtype=np.float64) / 255.0
    cols = [np.clip(np.rint(curve_model(p, x)), 0, 255) for p in channel_params]
    return np.ascontiguousarray(np.stack(cols, axis=1).astype(np.uint8))


def lut_md5(lut: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(lut, dtype=np.uint8).tobytes()).hexdigest()


def apply_lut(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Per-channel table lookup on an (H, W, 3) uint8 picture; a new array."""
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = lut[rgb[..., c], c]
    return out


# ------------------------------------------------------------------------- the light
def shadowclip_for(scene_spec, direction, margin: float = 1.02) -> float:
    """Half the side (m) of the square, seen from the sun, that holds all of the scene's
    geometry - the area a directional light's shadow map must cover (MuJoCo centres it on
    the light's line through the world origin, ``vsbench.render.mujoco_backend``)."""
    from vsbench.meshgen import quad
    from vsbench.scene import Box, Mesh, Plane, Sphere

    d = np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    a = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(a) < 1e-9:
        a = np.cross(d, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    pts = []
    for o in scene_spec.objects:
        if isinstance(o, Plane):
            V = np.asarray(quad(o.size)[0], np.float64)
        elif isinstance(o, Mesh):
            V = o.vertices
        elif isinstance(o, Box):
            s = np.asarray(o.size) / 2.0
            V = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
        elif isinstance(o, Sphere):
            r = o.radius
            V = np.asarray(o.center) + r * np.vstack([np.eye(3), -np.eye(3)])
            pts.append(V)
            continue
        else:
            raise TypeError(type(o))
        pts.append(V @ o.pose[:3, :3].T + o.pose[:3, 3])
    P = np.concatenate(pts)
    return float(np.max(np.abs(np.stack([P @ a, P @ b], axis=1)))) * float(margin)


def lit_spec(scene_spec, ambient_share: float, gain: float):
    """``scene_spec`` with the look's light: ambient ``a G`` and the same sun direction at
    ``(1 - a) G / cos(incidence)`` (a lit flat ``z = 0`` texel then renders ``G`` times its
    tone, as long as the sun's term stays <= 1, where MuJoCo saturates it)."""
    from vsbench.scene import Light, SceneSpec

    sun = scene_spec.lights[0]
    cos_inc = -float(sun.direction[2])
    a, G = float(ambient_share), float(gain)
    light = Light(direction=sun.direction, intensity=(1.0 - a) * G / cos_inc,
                  color=sun.color, cast_shadow=True)
    return SceneSpec(scene_spec.objects, (light,), a * G, scene_spec.background)


# -------------------------------------------------------------------------- the look
@dataclass
class Look:
    """How one scene's colour picture is made. ``Look.raw()`` or :func:`load_look`."""

    name: str
    scene: str | None = None
    ambient_share: float | None = None
    gain: float | None = None
    shadowsize: int = RAW_SHADOWSIZE
    shadowclip_margin: float | None = None      # None: MuJoCo's default clip (raw)
    channel_params: list | None = None          # [(floor, log_e, gamma)] * 3, or None (no curve)
    config_file: str | None = None
    config_md5: str | None = None
    extra: dict = field(default_factory=dict)
    lut: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.channel_params is not None and self.lut is None:
            self.lut = curve_lut(self.channel_params)

    @classmethod
    def raw(cls) -> Look:
        return cls("raw")

    @property
    def curve_id(self) -> str | None:
        if self.lut is None:
            return None
        return f"{self.name}/{self.scene}"

    @property
    def curve_md5(self) -> str | None:
        return None if self.lut is None else lut_md5(self.lut)

    def scene_spec(self, built_spec):
        """The plain-data scene this look renders (raw: as built)."""
        if self.ambient_share is None:
            return built_spec
        return lit_spec(built_spec, self.ambient_share, self.gain)

    def renderer_kwargs(self, spec) -> dict:
        kw = {"shadowsize": int(self.shadowsize)}
        if self.shadowclip_margin is not None:
            kw["shadowclip_m"] = shadowclip_for(spec, spec.lights[0].direction,
                                                self.shadowclip_margin)
        return kw

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return rgb if self.lut is None else apply_lut(rgb, self.lut)

    def record(self, spec=None, renderer_kwargs: dict | None = None) -> dict:
        """What a record says about the look (flat keys first, then the details)."""
        rec = {"look": self.name, "look_curve_id": self.curve_id, "look_curve_md5": self.curve_md5,
               "look_config": self.config_file, "look_config_md5": self.config_md5,
               "ambient_share": self.ambient_share, "light_gain": self.gain,
               "shadowsize": int(self.shadowsize)}
        if spec is not None:
            rec["ambient"] = round(float(spec.ambient), 9)
            rec["sun_intensity"] = round(float(spec.lights[0].intensity), 9)
        if renderer_kwargs:
            rec["shadowclip_m"] = renderer_kwargs.get("shadowclip_m")
        if self.channel_params is not None:
            rec["curve_params_rgb"] = [[float(v) for v in p] for p in self.channel_params]
        if self.name == "raw":
            rec["note"] = ("the U1/U4 picture: ambient 0.2, a lit white texel white, MuJoCo's "
                           "default shadow clip, no curve")
        return rec


def load_config(name: str = DEFAULT_LOOK) -> dict:
    p = config_path(name)
    if not p.exists():
        raise FileNotFoundError(f"look {name!r}: no {p} (tools/u4b_look.py fit writes it)")
    return json.loads(p.read_text())


def load_look(name: str, scene: str) -> Look:
    """The look ``name`` for ``scene``; ``raw`` needs no config."""
    if name == "raw":
        return Look.raw()
    if name not in LOOKS:
        raise ValueError(f"look {name!r} not in {LOOKS}")
    cfg = load_config(name)
    if scene not in cfg["scenes"]:
        raise KeyError(f"look {name!r} has no fit for scene {scene!r} "
                       f"(fitted: {sorted(cfg['scenes'])}); fly it with --look raw")
    s = cfg["scenes"][scene]
    params = [tuple(s["curve"][c]["params"]) for c in ("R", "G", "B")]
    look = Look(name, scene, ambient_share=float(cfg["lighting"]["ambient_share"]),
                gain=float(s["light_gain"]), shadowsize=int(cfg["lighting"]["shadowsize"]),
                shadowclip_margin=float(cfg["lighting"]["shadowclip_margin"]),
                channel_params=params, config_file=str(config_path(name).relative_to(
                    CONFIG_DIR.parents[2])), config_md5=file_md5(config_path(name)))
    want = s["curve"].get("lut_md5")
    if want and want != look.curve_md5:
        raise ValueError(f"look {name!r}/{scene}: the curve's table md5 {look.curve_md5} is not "
                         f"the config's {want} (numpy rounding changed? refit)")
    return look


def resolve_look(look, scene: str) -> Look:
    if isinstance(look, Look):
        return look
    return load_look(str(look or DEFAULT_LOOK), scene)
