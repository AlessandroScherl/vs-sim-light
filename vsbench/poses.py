"""Seeded camera poses in the optical convention (x right, y down, z forward)."""

from __future__ import annotations

import numpy as np

from vsbench.scene import rigid


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world pose whose optical axis points from ``eye`` to ``target``.

    ``up`` is the world direction that should appear as image-up (so image-down, the
    optical y axis, points against it).
    """
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z)
    up = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return rigid(np.column_stack([x, y, z]), eye)


def roll(T_wc: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the camera about its optical axis by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = T_wc.copy()
    T[:3, :3] = T_wc[:3, :3] @ Rz
    return T


def sample_pose(
    rng: np.random.Generator,
    T_wp: np.ndarray,
    dist: tuple[float, float] = (0.4, 1.0),
    max_slant_deg: float = 45.0,
    target_jitter: float = 0.05,
    max_roll_deg: float = 180.0,
) -> np.ndarray:
    """One pose looking at (near) the origin of plane frame ``T_wp`` from its +z side.

    Slant is the angle between the viewing direction and the plane normal, drawn
    uniformly in cos so that the flat views are not over-represented.
    """
    slant_max = np.deg2rad(max_slant_deg)
    cos_s = rng.uniform(np.cos(slant_max), 1.0)
    slant = np.arccos(cos_s)
    az = rng.uniform(0.0, 2 * np.pi)
    d = rng.uniform(*dist)
    dir_p = np.array([np.sin(slant) * np.cos(az), np.sin(slant) * np.sin(az), np.cos(slant)])
    target_p = np.array([rng.uniform(-target_jitter, target_jitter), rng.uniform(-target_jitter, target_jitter), 0.0])
    eye_p = target_p + d * dir_p
    R, t = T_wp[:3, :3], T_wp[:3, 3]
    eye_w = R @ eye_p + t
    target_w = R @ target_p + t
    # image-up along the plane's +y unless we look straight down it
    up_w = R @ np.array([0.0, 1.0, 0.0])
    T = look_at(eye_w, target_w, up_w)
    return roll(T, np.deg2rad(rng.uniform(-max_roll_deg, max_roll_deg)))


def sample_poses(seed: int, n: int, T_wp: np.ndarray, **kw) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [sample_pose(rng, T_wp, **kw) for _ in range(n)]


def orbit_poses(n: int, T_wp: np.ndarray, radius: float = 0.9, height: float = 0.7) -> list[np.ndarray]:
    """A smooth closed path around the plane origin, for the speed benchmarks."""
    out = []
    R, t = T_wp[:3, :3], T_wp[:3, 3]
    for i in range(n):
        a = 2 * np.pi * i / n
        eye_p = np.array([radius * np.cos(a), radius * np.sin(a), height])
        out.append(look_at(R @ eye_p + t, t, R @ np.array([0.0, 1.0, 0.0])))
    return out
