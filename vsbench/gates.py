"""Shared measurement code for the gates: marker detection, pose selection and ORB
matching. The tests call these; the tools reuse them for the report."""

from __future__ import annotations

import cv2
import numpy as np

from vsbench.camera import CameraModel
from vsbench.poses import sample_pose
from vsbench.reference import project
from vsbench.stdscenes import MARKER_COLORS, MARKER_POINTS, MARKER_RADIUS


def marker_pixel_radius(cam: CameraModel, z: np.ndarray) -> np.ndarray:
    return max(cam.fx, cam.fy) * MARKER_RADIUS / z


def projection_poses(cam: CameraModel, n: int, seed: int) -> list[np.ndarray]:
    """Seeded poses where all 13 markers are inside the image, apart and unoccluded."""
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < 10000:
        tries += 1
        T = sample_pose(rng, np.eye(4), dist=(0.45, 0.9), max_slant_deg=40.0, target_jitter=0.03)
        px, z = project(cam, T, MARKER_POINTS)
        if np.any(z <= 0.1):
            continue
        r = marker_pixel_radius(cam, z)
        margin = 3 * r + 4
        if np.any(px[:, 0] < margin) or np.any(px[:, 0] > cam.width - 1 - margin):
            continue
        if np.any(px[:, 1] < margin) or np.any(px[:, 1] > cam.height - 1 - margin):
            continue
        d = np.linalg.norm(px[:, None, :] - px[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        need = 3.0 * np.maximum(r[:, None], r[None, :]) + 6
        if np.any(d < need):
            continue
        out.append(T)
    if len(out) < n:
        raise RuntimeError("could not sample enough marker poses")
    return out


def detect_markers(rgb: np.ndarray, min_area: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted centroids (N, 2) and mean colours (N, 3) of bright blobs on black."""
    inten = rgb.max(axis=2).astype(np.float64)
    mask = (inten > 40).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cents, cols = [], []
    H, W = inten.shape
    vv, uu = np.mgrid[0:H, 0:W]
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        sel = labels == k
        w = inten[sel]
        cents.append([(uu[sel] * w).sum() / w.sum(), (vv[sel] * w).sum() / w.sum()])
        core = sel & (inten > 0.8 * inten[sel].max())
        cols.append(rgb[core].reshape(-1, 3).mean(axis=0) / 255.0)
    return np.array(cents).reshape(-1, 2), np.array(cols).reshape(-1, 3)


def match_markers(cam: CameraModel, T_wc: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Per-marker pixel error (13,) of the rendered centroids against the reference.

    Each expected marker takes the nearest detected blob within 6 px; the blob's colour
    must have the right dominant channels, so a swapped or mirrored marker fails.
    """
    expect, _ = project(cam, T_wc, MARKER_POINTS)
    cents, cols = detect_markers(rgb)
    if len(cents) < len(expect):
        raise AssertionError(f"found {len(cents)} blobs, expected {len(expect)}")
    errs = np.full(len(expect), np.inf)
    used = set()
    for i, e in enumerate(expect):
        d = np.linalg.norm(cents - e, axis=1)
        j = int(np.argmin(d))
        if d[j] > 6.0 or j in used:
            raise AssertionError(f"marker {i}: nearest blob at {d[j]:.1f} px (expected {e}), used={j in used}")
        want = MARKER_COLORS[i]
        got = cols[j]
        if np.argmax(got) != np.argmax(want) or np.any(got[want == 0] > 0.35 * got.max()):
            raise AssertionError(f"marker {i}: colour {got} does not look like {want}")
        used.add(j)
        errs[i] = d[j]
    return errs


def orb_displacements(img_a: np.ndarray, img_b: np.ndarray, nfeatures: int = 3000) -> np.ndarray:
    """Pixel displacements of cross-checked ORB matches between two images of one view."""
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) == 0 or len(kb) == 0:
        return np.zeros(0)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(da, db)
    if not m:
        return np.zeros(0)
    pa = np.array([ka[x.queryIdx].pt for x in m])
    pb = np.array([kb[x.trainIdx].pt for x in m])
    return np.linalg.norm(pa - pb, axis=1)


def inlier_counts(render: np.ndarray, ref: np.ndarray, tol: float = 2.0) -> dict:
    """ORB and SIFT matches between a render and its reference that agree to ``tol`` px."""
    gr = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
    gf = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    out = {}
    for name, det, norm in (("orb", cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING), ("sift", cv2.SIFT_create(), cv2.NORM_L2)):
        ka, da = det.detectAndCompute(gr, None)
        kb, db = det.detectAndCompute(gf, None)
        if da is None or db is None or len(ka) < 2 or len(kb) < 2:
            out[name] = {"matches": 0, "inliers": 0, "keypoints_render": len(ka or []), "keypoints_ref": len(kb or [])}
            continue
        m = cv2.BFMatcher(norm, crossCheck=True).match(da, db)
        pa = np.array([ka[x.queryIdx].pt for x in m]).reshape(-1, 2)
        pb = np.array([kb[x.trainIdx].pt for x in m]).reshape(-1, 2)
        disp = np.linalg.norm(pa - pb, axis=1) if len(m) else np.zeros(0)
        out[name] = {
            "matches": int(len(m)),
            "inliers": int((disp <= tol).sum()),
            "median_disp_px": float(np.median(disp)) if len(m) else None,
            "keypoints_render": len(ka),
            "keypoints_ref": len(kb),
        }
    return out


# -- step 2: rendered flow ----------------------------------------------------------------
def depth_edge_mask(depth: np.ndarray, jump: float = 0.02, margin: int = 5) -> np.ndarray:
    """Pixels within ``margin`` px of a depth edge or of the image border (bool (H, W)).

    A depth edge is a jump of more than ``jump`` metres to a 4-neighbour, or a change
    between hit (depth > 0) and miss (depth == 0). A slanted plane changes by a few
    millimetres per pixel, so it never counts.
    """
    d = np.asarray(depth, dtype=np.float64)
    hit = d > 0
    edge = np.zeros(d.shape, dtype=bool)
    for axis in (0, 1):
        a = np.take(d, np.arange(d.shape[axis] - 1), axis=axis)
        b = np.take(d, np.arange(1, d.shape[axis]), axis=axis)
        ha = np.take(hit, np.arange(d.shape[axis] - 1), axis=axis)
        hb = np.take(hit, np.arange(1, d.shape[axis]), axis=axis)
        e = (np.abs(a - b) > jump) | (ha != hb)
        if axis == 0:
            edge[:-1] |= e
            edge[1:] |= e
        else:
            edge[:, :-1] |= e
            edge[:, 1:] |= e
    k = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    out = cv2.dilate(edge.astype(np.uint8), k).astype(bool)
    out[:margin] = out[-margin:] = True
    out[:, :margin] = out[:, -margin:] = True
    return out


def shi_tomasi_corners(rgb: np.ndarray, max_corners: int = 2000, quality: float = 0.01, min_distance: int = 7) -> np.ndarray:
    """Shi-Tomasi corners (N, 2) as float pixel coordinates."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    c = cv2.goodFeaturesToTrack(g, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)
    return np.zeros((0, 2)) if c is None else c.reshape(-1, 2).astype(np.float64)


def track_lk(img_a: np.ndarray, img_b: np.ndarray, pts: np.ndarray, fb_max: float = 1.0, win: int = 21, levels: int = 4, init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade from ``img_a`` to ``img_b``: positions (N, 2) and a keep mask.

    A point is kept when the forward track succeeds and tracking it back lands within
    ``fb_max`` px of where it started (the usual forward-backward check). ``init`` (N, 2)
    seeds the forward search with a guess of where each point ends up, for displacements
    beyond what the pyramid covers.
    """
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    p0 = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    if len(p0) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    kw = dict(winSize=(win, win), maxLevel=levels, criteria=crit)
    if init is None:
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None, **kw)
    else:
        guess = np.asarray(init, dtype=np.float32).reshape(-1, 1, 2).copy()
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, guess, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, p1, None, **kw)
    fb = np.linalg.norm(p0b.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb <= fb_max)
    return p1.reshape(-1, 2).astype(np.float64), ok


def flow_agreement(measured: np.ndarray, predicted: np.ndarray, min_pred: float = 1.0) -> dict:
    """Endpoint error and direction agreement between measured and predicted flow (N, 2)."""
    m = np.asarray(measured, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    epe = np.linalg.norm(m - p, axis=1)
    pn = np.linalg.norm(p, axis=1)
    mn = np.linalg.norm(m, axis=1)
    big = pn > min_pred
    cos = np.full(len(m), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos[big] = (m[big] * p[big]).sum(1) / (pn[big] * mn[big])
    return {
        "n": int(len(m)),
        "n_over_1px": int(big.sum()),
        "median_epe_px": float(np.median(epe)) if len(m) else None,
        "p90_epe_px": float(np.percentile(epe, 90)) if len(m) else None,
        "median_cos": float(np.median(cos[big])) if big.any() else None,
        "median_pred_px": float(np.median(pn)) if len(m) else None,
    }
