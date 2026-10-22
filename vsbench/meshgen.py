"""Procedural triangle meshes: the clutter scene's three meshes and the quad the
poster and every finite plane are built from in backends that want a mesh."""

from __future__ import annotations

import numpy as np


def quad(size: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A rectangle in the x-y plane, normal +z, with picture-convention uvs.

    Returns ``(vertices (4, 3), faces (2, 3), uvs (4, 2))``; uv (0, 0) is the top-left of
    the picture at local (-w/2, +h/2), matching ``vsbench.scene.Plane``.
    """
    w, h = size
    V = np.array([[-w / 2, -h / 2, 0.0], [w / 2, -h / 2, 0.0], [w / 2, h / 2, 0.0], [-w / 2, h / 2, 0.0]])
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    UV = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    return V, F, UV


def uv_sphere(radius: float, n_lat: int = 100, n_lon: int = 100) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(0.0, np.pi, n_lat + 1)
    lon = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    V = radius * np.stack([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)], -1).reshape(-1, 3)
    F = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            if i > 0:
                F.append([a, c, b])
            if i < n_lat - 1:
                F.append([b, c, d])
    return V, np.array(F, dtype=np.int32)


def torus(R: float, r: float, n_major: int = 80, n_minor: int = 100) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    V = np.stack([(R + r * np.cos(vv)) * np.cos(uu), (R + r * np.cos(vv)) * np.sin(uu), r * np.sin(vv)], -1).reshape(-1, 3)
    F = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = i * n_minor + (j + 1) % n_minor
            c = ((i + 1) % n_major) * n_minor + j
            d = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            F.append([a, c, b])
            F.append([b, c, d])
    return V, np.array(F, dtype=np.int32)


def bumpy_sphere(radius: float, n: int = 84, seed: int = 0, bump: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """A sphere with smooth radial bumps (low-frequency noise), the third clutter mesh."""
    V, F = uv_sphere(1.0, n, n)
    rng = np.random.default_rng(seed)
    coef = rng.normal(size=(6, 3))
    phase = rng.uniform(0, 2 * np.pi, size=6)
    disp = np.zeros(len(V))
    for k in range(6):
        disp += np.sin((V @ coef[k]) * 2.0 + phase[k])
    scale = radius * (1.0 + bump * disp / 6.0)
    return V * scale[:, None], F
