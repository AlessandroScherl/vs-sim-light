# vs-sim copy: vssim/bench/debugview.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""The debug view: one flight drawn from its debug log, for a person to look at.

Purpose
-------
The owner's request of 2026-09-30: see, step by step, what each method matched, whether
the matches were right, where it was allowed to look, how far it was from the three
references and why it stopped. :func:`render_episode` reads one episode directory
written by ``check_baselines.py --telemetry --clean-film`` (``telemetry.jsonl.gz``,
``clean.mp4`` + ``clean_meta.json``, ``goal.jpg``, ``goal_mask.png``;
:mod:`vsbench.vs.bench.debuglog`) and draws 1600 x 1000 frames:

::

    header: method | scene / live object | start | step, and the three distances
    GOAL (800 x 450)                        | LIVE (800 x 450)
      goal mask shaded, ART-VS goal box       |   ART-VS live box, its source and mode
      numbered dots, one colour per pair -----+-- a thin line to the same pair's live dot
      dot size = match score                  |   thick ring green = right / red = wrong /
                                              |   grey = no truth; x = the true spot, a
                                              |   line from the dot to it = the miss;
                                              |   a yellow LANDED badge on a stop's frame
    legend (the method's cell in px and mm, the common 20-px yardstick)
    timeline of the whole flight (cursor)   | error split (sideways, height, tilt, roll),
      distances, closest approach, angle,   |   the commanded motion, DINOBot's fit
      matches, share right, median miss,    |
      stop markers, step ticks              |

:func:`render_pair` draws the landing still of every episode of one pair and an
``index.html`` table (method, start, the three distances at landing, the shares, the
median miss), so the flights worth opening can be found on a phone. Every writer takes an
``out_dir`` (default: the episode / pair directory), because the station's flight
directories belong to the container's user and the host cannot write into them.

It runs anywhere (numpy + OpenCV; no GPU, no Isaac); ``scripts/debug_view.py`` is its
command line. Nothing here is on a control path.

Provenance: new in vs-sim (2026-09-30, ``DESIGN.md`` section E on ``feat/debug-telemetry``;
layout revised after the review of the same day). Golden tests:
``tests/fast/test_debuglog.py`` (stills, a video, pixel probes of every panel, the header
fit, a read-only episode directory, the pair index).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np

from vsbench.vs.bench.debuglog import TRUTH_COMMON_PX, read_telemetry

#: The canvas and its parts, pixels.
CANVAS_W, CANVAS_H = 1600, 1000
HEADER_H = 84
PANEL_W, PANEL_H = 800, 450
LEGEND_H = 38
TIMELINE_W = 1050
BOTTOM_Y = HEADER_H + PANEL_H + LEGEND_H
BOTTOM_H = CANVAS_H - BOTTOM_Y
#: The timeline's x range (left axis labels, right margin).
TL_X0, TL_X1 = 70, TIMELINE_W - 20

FONT = 0            # cv2.FONT_HERSHEY_SIMPLEX
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
DARK = (28, 28, 28)
YELLOW = (0, 230, 255)
RIGHT_BGR = (60, 210, 60)
WRONG_BGR = (40, 40, 235)
NOTRUTH_BGR = (160, 160, 160)
NEAR_BGR = (255, 220, 0)          # the common yardstick's line (within 20 px), cyan
MISS_BGR = (60, 160, 255)         # the median miss curve, orange
#: The ring around a live dot (right / wrong / no truth), pixels.
RING_THICK = 4
#: The three references' colours in the timeline and the header: goal, size-matched,
#: depth-true.
REF_BGR = {"goal": (240, 240, 240), "size_matched": (230, 200, 60),
           "depth_true": (40, 210, 250)}
REF_NAMES = {"goal": "goal", "size_matched": "size-matched", "depth_true": "depth-true"}
#: Stop markers in the timeline.
STOP_BGR = {"own": (230, 80, 230), "scored": (40, 210, 250), "velocity_quiet": (150, 150, 150)}

#: 20 well-separated colours (BGR) for the pairs, with NO pure red and NO pure green (those
#: two are the right / wrong rings); a pair keeps its colour across steps when the method
#: has a landmark id.
PALETTE = [(255, 160, 0), (0, 215, 255), (255, 0, 200), (255, 255, 0), (180, 105, 255),
           (0, 140, 255), (255, 255, 255), (200, 120, 120), (140, 230, 240), (240, 130, 240),
           (100, 180, 255), (160, 60, 120), (255, 200, 150), (80, 120, 180), (220, 160, 255),
           (60, 200, 255), (255, 100, 100), (200, 200, 90), (120, 90, 200), (190, 150, 60)]

SOURCE_WORDS = {"langsam": "detector", "lighttrack": "tracker", "cache": "cache",
                "none": "full picture"}


# ------------------------------------------------------------------ text
def _fit(s: str, scale: float, thick: int, max_w: int) -> float:
    """The largest scale <= ``scale`` at which ``s`` is at most ``max_w`` pixels wide."""
    import cv2  # noqa: PLC0415

    while scale > 0.3 and cv2.getTextSize(str(s), FONT, scale, thick + 3)[0][0] > max_w:
        scale *= 0.95
    return scale


def _text(img, s, org, scale=0.7, colour=WHITE, thick=2, max_w: int | None = None) -> float:
    """Outlined text; with ``max_w`` it is shrunk to fit. Returns the scale used."""
    import cv2  # noqa: PLC0415

    if max_w is not None:
        scale = _fit(s, scale, thick, max_w)
    cv2.putText(img, str(s), (int(org[0]), int(org[1])), FONT, scale, (0, 0, 0), thick + 3,
                cv2.LINE_AA)
    cv2.putText(img, str(s), (int(org[0]), int(org[1])), FONT, scale, colour, thick,
                cv2.LINE_AA)
    return scale


def _text_width(s: str, scale: float, thick: int = 2) -> int:
    import cv2  # noqa: PLC0415

    return int(cv2.getTextSize(str(s), FONT, scale, thick + 3)[0][0])


def _mm(x) -> str:
    return "-" if x is None else f"{x:.1f} mm"


# ------------------------------------------------------------------ loading
class _Frames:
    """The clean film's frames by step, decoded on demand by reading forward: a still
    decodes up to its one frame, a video streams in step order - never the whole film
    in memory."""

    def __init__(self, path: Path, meta: list[dict]) -> None:
        self.path = Path(path)
        self.index = {int(m["step"]): k for k, m in enumerate(meta)}
        self.cap = None
        self.pos = 0
        self.last: tuple[int, np.ndarray] | None = None

    def get(self, step: int):
        import cv2  # noqa: PLC0415

        k = self.index.get(int(step))
        if k is None or not self.path.is_file():
            return None
        if self.last is not None and self.last[0] == k:
            return self.last[1]
        if self.cap is None or k < self.pos:
            self.close()
            self.cap, self.pos = cv2.VideoCapture(str(self.path)), 0
        img = None
        while self.pos <= k:
            ok, img = self.cap.read()
            if not ok:
                return None
            self.pos += 1
        self.last = (k, img)
        return img

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Episode:
    """One episode directory, loaded: the telemetry lines, the goal picture and mask, the
    clean frames (lazily, :meth:`frame`)."""

    def __init__(self, ep_dir) -> None:
        import cv2  # noqa: PLC0415

        self.dir = Path(ep_dir)
        lines = read_telemetry(self.dir / "telemetry.jsonl.gz")
        if not lines or lines[0].get("kind") != "meta":
            raise ValueError(f"{self.dir}: no debug telemetry (fly with --telemetry)")
        self.meta = lines[0]
        self.steps = [e for e in lines if e.get("kind") == "step"]
        self.final = next((e for e in lines if e.get("kind") == "final"), None)
        self.end = next((e for e in lines if e.get("kind") == "end"), None)
        self.error = next((e for e in lines if e.get("kind") == "error"), None)
        self.W, self.H = (int(v) for v in self.meta["image_wh"])
        goal = cv2.imread(str(self.dir / "goal.jpg"))
        self.goal = (np.zeros((PANEL_H, PANEL_W, 3), np.uint8) if goal is None
                     else cv2.resize(goal, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA))
        mask = cv2.imread(str(self.dir / "goal_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.mask = (None if mask is None else
                     cv2.resize(mask, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST) > 127)
        meta_path = self.dir / "clean_meta.json"
        self.clean_meta = (json.loads(meta_path.read_text()) if meta_path.is_file() else [])
        self._frames = _Frames(self.dir / "clean.mp4", self.clean_meta)
        self.sx, self.sy = PANEL_W / self.W, PANEL_H / self.H
        self.max_step = max([int(self.meta.get("max_iter") or 1)]
                            + [int(s["i"]) for s in self.steps])
        self.markers = self._markers()

    def frame(self, i: int):
        """The clean live frame of step ``i`` (BGR, half size; the final frame is step
        ``steps + 1``), or ``None``."""
        return self._frames.get(i)

    def close(self) -> None:
        self._frames.close()

    def _markers(self) -> list[tuple[int, str, tuple]]:
        """``[(step, label, colour)]``: the method's own stop, the scored stop, the quiet
        stop (and any other recorded rule)."""
        out = []
        stops = (self.end or {}).get("stops") or {}
        own_rules = set(self.meta.get("stops", {}).get("own") or [])
        scored = self.meta.get("stops", {}).get("scored")
        for rule, st in stops.items():
            if not isinstance(st, dict) or st.get("step") is None:
                continue
            if rule in own_rules:
                out.append((int(st["step"]), "own stop", STOP_BGR["own"]))
            elif rule == scored:
                out.append((int(st["step"]), "scored stop", STOP_BGR["scored"]))
            else:
                out.append((int(st["step"]), rule.replace("_", " "),
                            STOP_BGR.get(rule, GREY)))
        return sorted(out)

    def landing_marks(self, i: int) -> list[str]:
        """The landings (own stop, scored stop) on step ``i`` - what the film marks."""
        return [lab for (s, lab, _c) in self.markers
                if s == i and lab in ("own stop", "scored stop")]

    def landing_step(self) -> int | None:
        """The method's own stop if it fired, else the scored stop, else ``None``."""
        own = [m for m in self.markers if m[1] == "own stop"]
        sc = [m for m in self.markers if m[1] == "scored stop"]
        return (own or sc or [(None,)])[0][0]

    def step_by_i(self, i: int) -> dict | None:
        return next((s for s in self.steps if int(s["i"]) == int(i)), None)


# ------------------------------------------------------------------ panels
def _colour(k: int, lid) -> tuple:
    return PALETTE[(int(lid) if lid is not None else k) % len(PALETTE)]


def _ring_colour(letter: str) -> tuple:
    return RIGHT_BGR if letter == "R" else WRONG_BGR if letter == "W" else NOTRUTH_BGR


def _shade_mask(img, mask) -> None:
    if mask is None:
        return
    tint = np.array([60, 160, 60], np.float32)
    img[mask] = (0.6 * img[mask].astype(np.float32) + 0.4 * tint).astype(np.uint8)


def _box(img, box, sx, sy, colour, label=None) -> None:
    import cv2  # noqa: PLC0415

    if box is None:
        return
    x1, y1, x2, y2 = (float(v) for v in box)
    p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
    cv2.rectangle(img, p1, p2, colour, 2)
    if label:
        _text(img, label, (p1[0] + 4, max(22, p1[1] - 6)), 0.6, colour, 2,
              max_w=img.shape[1] - p1[0] - 8)


def _grid(img, box, grid, sx, sy, colour) -> None:
    import cv2  # noqa: PLC0415

    if box is None or not grid:
        return
    x1, y1, x2, y2 = (float(v) for v in box)
    gw, gh = int(grid[0]), int(grid[1])
    for j in range(1, gw):
        x = int((x1 + (x2 - x1) * j / gw) * sx)
        cv2.line(img, (x, int(y1 * sy)), (x, int(y2 * sy)), colour, 1)
    for i in range(1, gh):
        y = int((y1 + (y2 - y1) * i / gh) * sy)
        cv2.line(img, (int(x1 * sx), y), (int(x2 * sx), y), colour, 1)


def _shares(step: dict) -> tuple[str, str]:
    """The live panel's two headline lines: the match count, then the shares and the miss."""
    right = step.get("right") or {}
    judged = int(right.get("R", 0)) + int(right.get("W", 0))
    near = step.get("near") or {}
    parts = []
    parts.append(f"right (own cell) {right.get('R', 0)}/{judged}" if judged
                 else "right (own cell) -")
    if near.get("n"):
        parts.append(f"within {near.get('px', TRUTH_COMMON_PX):.0f} px {near['k']}/{near['n']}")
        if near.get("median_px") is not None:
            mm = near.get("median_mm")
            parts.append(f"median miss {near['median_px']:.0f} px"
                         + (f" ~ {mm:.1f} mm" if mm is not None else ""))
    else:
        parts.append("no truth for these points")
    return f"LIVE   {step.get('n', 0)} matches", "  |  ".join(parts)


def draw_match_panels(ep: Episode, step: dict | None, live_bgr, *,
                      landing: list[str] | None = None, final_caption: str = "") -> np.ndarray:
    """Panels 1-4: goal | live, the pairs, the truth rings and misses, where the method may
    look; the LANDED badge when a stop fired on this frame."""
    import cv2  # noqa: PLC0415

    g = ep.goal.copy()
    _shade_mask(g, ep.mask)
    if live_bgr is None:
        c = np.full((PANEL_H, PANEL_W, 3), 45, np.uint8)
        _text(c, "no clean film (fly with --clean-film)", (40, PANEL_H // 2), 0.8, GREY)
    else:
        c = cv2.resize(live_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
    sx, sy = ep.sx, ep.sy
    ag = (ep.meta.get("arm_goal") or {}).get("artvs")
    if ag:
        _box(g, ag.get("goal_box"), sx, sy, (0, 200, 255), "goal box")
        _grid(g, ag.get("goal_box"), ag.get("grid"), sx, sy, (0, 200, 255))
    art = (step or {}).get("artvs")
    if art:
        src = SOURCE_WORDS.get(art.get("source"), str(art.get("source")))
        mode = art.get("mode")
        if mode == "tiled" and art.get("grid"):
            mode = f"tiles {art['grid'][0]} x {art['grid'][1]}"
        elif mode == "full":
            mode = "full picture"
        _box(c, art.get("box"), sx, sy, (0, 200, 255), f"{src} | {mode}")
        if art.get("mode") == "tiled":
            _grid(c, art.get("box"), art.get("grid"), sx, sy, (0, 200, 255))
        if art.get("box") is None:
            _text(c, f"{src} | {mode}", (12, PANEL_H - 60), 0.65, (0, 200, 255))
    canvas = np.concatenate([g, c], axis=1)
    if not step:
        _text(canvas, "GOAL", (12, 30), 0.8)
        _text(canvas, f"LIVE  ({final_caption or 'after the last move'})", (PANEL_W + 12, 30),
              0.8, max_w=PANEL_W - 24)
    else:
        pr, tr = step.get("pairs") or {}, step.get("truth") or {}
        gu, gv, cu, cv_ = (pr.get(k) or [] for k in ("gu", "gv", "cu", "cv"))
        scores, lids = pr.get("score"), pr.get("lid")
        stat = tr.get("stat") or ""
        tu, tv = tr.get("tu") or [], tr.get("tv") or []
        ok = [k for k in range(len(cu)) if None not in (gu[k], gv[k], cu[k], cv_[k])]
        geo = {}
        for k in ok:
            s = scores[k] if scores and scores[k] is not None else 0.5
            geo[k] = (int(round(3 + 6 * min(1.0, max(0.0, float(s))))),
                      (int(gu[k] * sx), int(gv[k] * sy)),
                      (int(cu[k] * sx) + PANEL_W, int(cv_[k] * sy)))
        # 1. the pairing lines (thin, the pair's colour), under everything
        for k in ok:
            _rad, pg, pc = geo[k]
            cv2.line(canvas, pg, pc, _colour(k, lids[k] if lids else None), 1, cv2.LINE_AA)
        # 2. the miss: from the live dot to its true spot, in the verdict's colour, and the x
        for k in ok:
            if k < len(tu) and tu[k] is not None and tv[k] is not None:
                ring = _ring_colour(stat[k] if k < len(stat) else "?")
                t = (int(tu[k] * sx) + PANEL_W, int(tv[k] * sy))
                cv2.line(canvas, geo[k][2], t, ring, 2, cv2.LINE_AA)
                cv2.drawMarker(canvas, t, ring, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        # 3. the dots (size = score), the verdict ring, the numbers
        for k in ok:
            rad, pg, pc = geo[k]
            col = _colour(k, lids[k] if lids else None)
            cv2.circle(canvas, pg, rad, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, pc, rad, col, -1, cv2.LINE_AA)
            ring = _ring_colour(stat[k] if k < len(stat) else "?")
            cv2.circle(canvas, pc, rad + 3 + RING_THICK // 2, ring, RING_THICK, cv2.LINE_AA)
            _text(canvas, str(k + 1), (pg[0] + rad + 2, pg[1] - rad), 0.5, col, 1)
            _text(canvas, str(k + 1), (pc[0] + rad + 6, pc[1] - rad - 2), 0.5, col, 1)
        _text(canvas, "GOAL", (12, 30), 0.8)
        head, shares = _shares(step)
        _text(canvas, head, (PANEL_W + 12, 30), 0.8, max_w=PANEL_W - 24)
        _text(canvas, shares, (PANEL_W + 12, 58), 0.6, WHITE, 2, max_w=PANEL_W - 24)
    if landing:
        # the landing: a yellow border round the live half and a badge at its foot
        cv2.rectangle(canvas, (PANEL_W + 2, 2), (2 * PANEL_W - 3, PANEL_H - 3), YELLOW, 6)
        badge = "LANDED: " + " + ".join(landing)
        scale = _fit(badge, 0.8, 2, PANEL_W - 60)
        w = _text_width(badge, scale)
        x0, y1 = PANEL_W + 14, PANEL_H - 14
        cv2.rectangle(canvas, (x0, y1 - 38), (x0 + w + 20, y1), YELLOW, -1)
        cv2.putText(canvas, badge, (x0 + 10, y1 - 11), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), DARK, 3)
    return canvas


# ---------------------------------------------------------------- timeline
def _series(ep: Episode, ref: str) -> list[tuple[int, float]]:
    out = []
    for s in ep.steps:
        v = (s.get("to") or {}).get(ref)
        if v is not None and v.get("t_mm") is not None:
            out.append((int(s["i"]), float(v["t_mm"])))
    if ep.final is not None and (ep.final.get("to") or {}).get(ref):
        out.append((int(ep.final["after_step"]) + 1, float(ep.final["to"][ref]["t_mm"])))
    return out


def _X(ep: Episode, i) -> int:
    n = ep.max_step + 1
    return int(TL_X0 + (TL_X1 - TL_X0) * (float(i) - 1) / max(1.0, n - 1))


def _tick_step(n: int) -> int:
    for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if n / t <= 12:
            return t
    return 2000


#: The timeline's strips (top, bottom), pixels inside the bottom-left panel.
STRIP_A = (40, 150)     # distance to the three references, mm, log scale
STRIP_B = (178, 214)    # the angle to the goal orientation
STRIP_C = (242, 276)    # matches (bars)
STRIP_D = (304, 344)    # share right (own cell) and within 20 px, 0..100 %
STRIP_E = (372, 398)    # median miss, px
TICKS_Y = 418


def draw_timeline_base(ep: Episode) -> np.ndarray:
    """Panel 5 without the cursor: the whole flight, drawn once per episode."""
    import cv2  # noqa: PLC0415

    img = np.full((BOTTOM_H, TIMELINE_W, 3), 22, np.uint8)
    x0, x1 = TL_X0, TL_X1

    def X(i):
        return _X(ep, i)

    # strip A: distance to the three references, mm, log scale
    a0, a1 = STRIP_A
    vals = [v for ref in REF_BGR for _, v in _series(ep, ref) if v > 0]
    lo = 10 ** math.floor(math.log10(max(0.1, min(vals)))) if vals else 1.0
    hi = 10 ** math.ceil(math.log10(max(vals))) if vals else 1000.0
    hi = max(hi, lo * 10)

    def YA(v):
        v = min(max(v, lo), hi)
        return int(a1 - (a1 - a0) * (math.log10(v) - math.log10(lo))
                   / (math.log10(hi) - math.log10(lo)))

    dec = lo
    while dec <= hi * 1.001:
        y = YA(dec)
        cv2.line(img, (x0, y), (x1, y), (55, 55, 55), 1)
        _text(img, f"{dec:g}", (6, y + 6), 0.5, GREY, 1)
        dec *= 10
    for ref, col in REF_BGR.items():
        pts = [(X(i), YA(v)) for i, v in _series(ep, ref)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
    lx = x0 + 6
    _text(img, "distance (mm, log):", (lx, 28), 0.6, WHITE, 1)
    lx += 215
    for ref, col in REF_BGR.items():
        if _series(ep, ref):
            _text(img, REF_NAMES[ref], (lx, 28), 0.6, col, 2)
            lx += 20 + 14 * len(REF_NAMES[ref])
    # the closest approach to the pose that matters (depth-true when it exists)
    ref = "depth_true" if _series(ep, "depth_true") else "goal"
    ser = _series(ep, ref)
    if ser:
        ci, cvv = min(ser, key=lambda t: t[1])
        cv2.circle(img, (X(ci), YA(cvv)), 7, REF_BGR[ref], 2, cv2.LINE_AA)
        label = f"closest {cvv:.1f} mm @{ci}"
        _text(img, label, (min(X(ci) + 10, x1 - _text_width(label, 0.55, 1)), YA(cvv) - 10),
              0.55, REF_BGR[ref], 1)
    # strip B: the angle to the goal orientation, deg
    b0, b1 = STRIP_B
    ang = [(int(s["i"]), float(s["r_deg"])) for s in ep.steps if s.get("r_deg") is not None]
    amax = max([1.0] + [v for _, v in ang])
    if len(ang) > 1:
        cv2.polylines(img, [np.array([(X(i), int(b1 - (b1 - b0) * v / amax)) for i, v in ang],
                                     np.int32)], False, (200, 160, 255), 2, cv2.LINE_AA)
    _text(img, f"angle (deg, top = {amax:.1f})", (x0 + 6, b0 - 6), 0.55, (200, 160, 255), 1)
    # strip C: matches (bars)
    c0, c1 = STRIP_C
    nmax = max([1] + [int(s.get("n") or 0) for s in ep.steps])
    bw = min(4, max(0, int(0.35 * (x1 - x0) / (ep.max_step + 1))))
    for s in ep.steps:
        x = X(s["i"])
        h = int((c1 - c0) * int(s.get("n") or 0) / nmax)
        cv2.rectangle(img, (x - bw, c1), (x + bw, c1 - h), (90, 90, 90), -1)
    _text(img, f"matches (bars, top = {nmax})", (x0 + 6, c0 - 6), 0.55, WHITE, 1)
    # strip D: the two shares, 0..100 %; shaded red where fewer than half are right
    d0, d1 = STRIP_D
    cv2.rectangle(img, (x0, d0), (x1, d1), (40, 40, 40), 1)
    own, near = [], []
    for s in ep.steps:
        r = s.get("right") or {}
        judged = int(r.get("R", 0)) + int(r.get("W", 0))
        if judged:
            frac = int(r["R"]) / judged
            own.append((X(s["i"]), int(d1 - (d1 - d0) * frac)))
            if frac < 0.5:
                x = X(s["i"])
                cv2.rectangle(img, (x - max(1, bw), d0 + 1), (x + max(1, bw), d1 - 1),
                              (30, 30, 90), -1)
        nr = s.get("near") or {}
        if nr.get("n"):
            near.append((X(s["i"]), int(d1 - (d1 - d0) * int(nr["k"]) / int(nr["n"]))))
    for pts, col in ((near, NEAR_BGR), (own, RIGHT_BGR)):
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(img, pts[0], 3, col, -1)
    lab = "share right (0-100 %):"
    _text(img, lab, (x0 + 6, d0 - 6), 0.55, WHITE, 1)
    lx = x0 + 16 + _text_width(lab, 0.55, 1)
    _text(img, "own cell", (lx, d0 - 6), 0.55, RIGHT_BGR, 1)
    lx += 16 + _text_width("own cell", 0.55, 1)
    _text(img, f"within {TRUTH_COMMON_PX:.0f} px (same for every method)", (lx, d0 - 6), 0.55,
          NEAR_BGR, 1)
    # strip E: the median miss, px (linear, top = the largest)
    e0, e1 = STRIP_E
    miss = [(int(s["i"]), float((s.get("near") or {})["median_px"])) for s in ep.steps
            if (s.get("near") or {}).get("median_px") is not None]
    # at least 1.5 x the common yardstick, so its grey line is always inside the strip
    mmax = max([1.5 * TRUTH_COMMON_PX] + [v for _, v in miss])
    if len(miss) > 1:
        cv2.polylines(img, [np.array([(X(i), int(e1 - (e1 - e0) * v / mmax)) for i, v in miss],
                                     np.int32)], False, MISS_BGR, 2, cv2.LINE_AA)
    y20 = int(e1 - (e1 - e0) * min(1.0, TRUTH_COMMON_PX / mmax))
    cv2.line(img, (x0, y20), (x1, y20), (70, 70, 70), 1)
    _text(img, f"median miss (px, top = {mmax:.0f}; grey line = {TRUTH_COMMON_PX:.0f} px)",
          (x0 + 6, e0 - 6), 0.55, MISS_BGR, 1)
    # the step axis
    t = _tick_step(ep.max_step)
    for i in [1] + list(range(t, ep.max_step + 1, t)):
        x = X(i)
        cv2.line(img, (x, TICKS_Y - 12), (x, TICKS_Y - 6), GREY, 1)
        if i == 1 or x < X(ep.max_step) - 40 or i == ep.max_step:
            _text(img, str(i), (x - 8, TICKS_Y + 6), 0.45, GREY, 1)
    # the stop markers and the cap
    for k, (i, label, col) in enumerate(ep.markers):
        x = X(i)
        cv2.line(img, (x, a0 - 4), (x, e1), col, 2)
        s = f"{label} @{i}"
        _text(img, s, (min(x + 4, x1 - _text_width(s, 0.5, 1)), a0 + 14 + 18 * k), 0.5, col, 1)
    cv2.line(img, (X(ep.max_step), a0), (X(ep.max_step), e1), (90, 90, 90), 1)
    _text(img, "cap", (X(ep.max_step) - 36, a1 - 4), 0.5, GREY, 1)
    return img


def draw_cursor(base: np.ndarray, ep: Episode, i: int) -> np.ndarray:
    import cv2  # noqa: PLC0415

    img = base.copy()
    x = _X(ep, i)
    cv2.line(img, (x, 32), (x, STRIP_E[1] + 2), WHITE, 2)
    return img


# ------------------------------------------------------------ error split
def draw_error_split(ep: Episode, entry: dict | None) -> np.ndarray:
    """Panel 6: signed bars - sideways (x, y), height, tilt, roll - against the depth-true
    pose when it exists, the goal pose otherwise; the commanded motion; DINOBot's fit."""
    import cv2  # noqa: PLC0415

    w = CANVAS_W - TIMELINE_W
    img = np.full((BOTTOM_H, w, 3), 30, np.uint8)
    if not entry:
        return img
    to = entry.get("to") or {}
    ref = "depth_true" if to.get("depth_true") else "goal"
    e = (to.get(ref) or {}).get("e_mm") or [0.0, 0.0, 0.0]
    _text(img, f"error split vs the {REF_NAMES[ref]} pose", (14, 30), 0.65, REF_BGR[ref], 2,
          max_w=w - 28)
    rows = [("right (+) / left (-)", e[0], "mm"), ("down (+) / up (-)", e[1], "mm"),
            ("too high (+) / low (-)", None if e[2] is None else -e[2], "mm"),
            ("tilt", entry.get("tilt_deg"), "deg"), ("turn (roll)", entry.get("roll_deg"), "deg")]
    mm_vals = [abs(v) for _, v, u in rows if u == "mm" and v is not None]
    mm_rng = max([10.0] + mm_vals)
    deg_rng = max([2.0] + [abs(v) for _, v, u in rows if u == "deg" and v is not None])
    # labels on the left, the bars in the middle (symmetric about zero), the value on the right
    cx, half, vx = 310, 80, 405
    for k, (label, v, unit) in enumerate(rows):
        y = 70 + 46 * k
        _text(img, label, (14, y + 6), 0.55, WHITE, 1)
        cv2.line(img, (cx, y - 14), (cx, y + 14), GREY, 1)
        if v is None:
            _text(img, "-", (vx, y + 7), 0.6, GREY, 1)
            continue
        rng = mm_rng if unit == "mm" else deg_rng
        L = int(half * max(-1.0, min(1.0, float(v) / rng)))
        col = (80, 170, 255) if unit == "mm" else (200, 160, 255)
        cv2.rectangle(img, (cx, y - 10), (cx + L, y + 10), col, -1)
        fmt = f"{float(v):+.0f}" if abs(float(v)) >= 100 else f"{float(v):+.1f}"
        _text(img, f"{fmt} {unit}", (vx, y + 7), 0.6, WHITE, 1)
    _text(img, f"bars: +-{mm_rng:.0f} mm, +-{deg_rng:.1f} deg", (14, 70 + 46 * 5 - 8), 0.5,
          GREY, 1)
    # the commanded motion (what the method asked the arm for on this step)
    v = entry.get("v_cmd")
    if v and len(v) == 6 and None not in v:
        lin = 1000.0 * math.sqrt(sum(float(a) ** 2 for a in v[:3]))
        rot = math.degrees(math.sqrt(sum(float(a) ** 2 for a in v[3:])))
        _text(img, f"command: move {lin:.1f} mm/s, turn {rot:.2f} deg/s", (14, BOTTOM_H - 52),
              0.55, WHITE, 1, max_w=w - 28)
    d = entry.get("dinobot")
    if d:
        def f(x, fmt):
            return "-" if x is None else format(x, fmt)
        _text(img, f"DINOBot: buddies {f(d.get('n_buddies'), 'd')}  |  fit residual "
                   f"{f(d.get('fit_residual_mm'), '.1f')} mm  |  jump "
                   f"{f(d.get('jump_t_mm'), '.1f')} mm", (14, BOTTOM_H - 22), 0.55,
              (230, 80, 230), 1, max_w=w - 28)
    return img


# ---------------------------------------------------------------- one frame
def header_title(ep: Episode, i: int, kind: str) -> str:
    """The header's first line: method, scene / live object (and what it is), start, step."""
    m = ep.meta
    title = f"{m.get('display')} | {m.get('scene')}"
    if m.get("live_name"):
        title += f" / {m['live_name']}"
    truth = (m.get("truth") or {})
    if truth.get("object") == "scaled" and truth.get("s"):
        title += f" (same object, {100 * float(truth['s']):.0f} % size)"
    elif truth.get("object") == "none":
        title += " (another object)"
    title += f" | start {m.get('start_id')} | "
    title += ("after the last move" if kind == "final" else f"step {i}/{m.get('max_iter')}")
    return title


def _final_caption(ep: Episode) -> str:
    f = ep.final or {}
    if f.get("moved", True):
        return "after the last move"
    return f"final pose, no move after step {f.get('after_step')}: {f.get('end_reason')}"


def compose(ep: Episode, entry: dict | None, *, kind: str, timeline_base) -> np.ndarray:
    """One 1600 x 1000 frame for a telemetry ``step`` (or the ``final`` entry)."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    if kind == "final":
        i = int(ep.final["after_step"]) + 1 if ep.final else ep.max_step + 1
        step = None
    else:
        i = int(entry["i"])
        step = entry
    live = ep.frame(i)
    marks = ep.landing_marks(i) if kind != "final" else []
    _text(canvas, header_title(ep, i, kind), (14, 34), 0.8, YELLOW if marks else WHITE, 2,
          max_w=CANVAS_W - 28)
    to = (entry or {}).get("to") or {}
    parts = [(f"to {REF_NAMES[ref]} {_mm(to[ref]['t_mm'] if to.get(ref) else None)}",
              REF_BGR[ref]) for ref in REF_BGR]
    parts.append((f"angle {float((entry or {}).get('r_deg') or 0.0):.1f} deg", (200, 160, 255)))
    # the second line, shrunk as one piece so all four fit
    gap = 30
    scale = _fit("".join(p for p, _ in parts) + " " * 8, 0.75, 2, CANVAS_W - 28 - gap * 3)
    x = 14
    for s, col in parts:
        _text(canvas, s, (x, 70), scale, col, 2)
        x += _text_width(s, scale) + gap
    canvas[HEADER_H:HEADER_H + PANEL_H] = draw_match_panels(
        ep, step, live, landing=marks, final_caption=_final_caption(ep))
    cell, cell_mm = (step or {}).get("cell_px"), (step or {}).get("cell_mm")
    if cell:
        cell_txt = f"{cell[0]:.0f} x {cell[1]:.0f} px"
        if cell_mm:
            cell_txt += f" ~ {cell_mm[0]:.0f} x {cell_mm[1]:.0f} mm"
    else:
        cell_txt = "no cell size"
    _text(canvas, f"ring green = right (within one {ep.meta.get('display')} cell: {cell_txt})  |  "
                  f"red = wrong  |  grey = no truth / floor / hidden  |  x = true spot, line = "
                  f"the miss  |  big dot = sure",
          (14, HEADER_H + PANEL_H + 27), 0.58, WHITE, 1, max_w=CANVAS_W - 28)
    canvas[BOTTOM_Y:, :TIMELINE_W] = draw_cursor(timeline_base, ep, i)
    canvas[BOTTOM_Y:, TIMELINE_W:] = draw_error_split(ep, entry)
    return canvas


def _entries(ep: Episode, every: int = 1) -> list[tuple[str, dict]]:
    out = [("step", s) for k, s in enumerate(ep.steps)
           if k % max(1, every) == 0 or int(s["i"]) in {m[0] for m in ep.markers}]
    if ep.final is not None:
        out.append(("final", ep.final))
    return out


def _out(ep: Episode, out_dir) -> Path:
    d = Path(out_dir) if out_dir is not None else ep.dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_still(ep_dir, which="landing", *, out_dir=None, name: str | None = None) -> Path:
    """``debug_<step>.png`` (or ``debug_final.png``) for a step number, ``landing`` (the
    method's own stop, else the scored stop, else the final frame) or ``final``; into
    ``out_dir`` (default the episode directory), named ``name`` if given."""
    import cv2  # noqa: PLC0415

    ep = Episode(ep_dir)
    try:
        base = draw_timeline_base(ep)
        step_i = None
        if str(which) == "landing":
            step_i = ep.landing_step()
            if step_i is None and ep.steps:
                # read at the limit: the last logged step shows the matches the flight ended on
                # (the final frame, after the last move, has none)
                step_i = int(ep.steps[-1]["i"])
        elif str(which) != "final":
            step_i = int(which)
        entry = ep.step_by_i(step_i) if step_i is not None else None
        if entry is None:
            img, default = compose(ep, ep.final, kind="final", timeline_base=base), \
                "debug_final.png"
        else:
            img, default = compose(ep, entry, kind="step", timeline_base=base), \
                f"debug_{step_i}.png"
        out = _out(ep, out_dir) / (name or default)
        cv2.imwrite(str(out), img)
    finally:
        ep.close()
    return out


def render_episode(ep_dir, *, every: int = 1, out_dir=None) -> Path | None:
    """``debug.mp4``: one frame per logged step (every ``every``-th, landings always) and
    the final frame; landings held 1 s, the final frame 2 s. Written into the episode
    directory as ``debug.mp4``, or into ``out_dir`` as ``<episode>_debug.mp4``."""
    import cv2  # noqa: PLC0415

    from vsbench.vs.bench import films

    ep = Episode(ep_dir)
    dest = _out(ep, out_dir)
    base = draw_timeline_base(ep)
    tmp = dest / f"_debug_frames_{ep.dir.name}"
    tmp.mkdir(exist_ok=True)
    holds: dict[str, float] = {}
    landing = {m[0] for m in ep.markers}
    try:
        for k, (kind, entry) in enumerate(_entries(ep, every)):
            name = f"dbg_{k:05d}.jpg"
            cv2.imwrite(str(tmp / name), compose(ep, entry, kind=kind, timeline_base=base),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            if kind == "final":
                holds[name] = films.FINAL_HOLD_S
            elif int(entry["i"]) in landing:
                holds[name] = films.LANDING_HOLD_S
        out = films.encode_episode(tmp, scale=1.0, keep_frames=False, holds=holds,
                                   pattern="dbg_*.jpg", out_name="debug.mp4", crf=23)
        final = None
        if out is not None:
            final = dest / ("debug.mp4" if out_dir is None else f"{ep.dir.name}_debug.mp4")
            out.replace(final)
    finally:
        ep.close()
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()
    return final


# ------------------------------------------------------------- one pair
def _landing_summary(ep: Episode) -> dict:
    """What the pair index shows for one episode: where it landed and how it matched."""
    i = ep.landing_step()
    entry = ep.step_by_i(i) if i is not None else None
    where = ("own stop" if i is not None and "own stop" in ep.landing_marks(i)
             else "scored stop" if i is not None else "final pose")
    src = entry if entry is not None else (ep.final or {})
    to = src.get("to") or {}
    row = {"episode": ep.dir.name, "method": ep.meta.get("display"),
           "start": ep.meta.get("start_id"), "landing": where,
           "step": i if i is not None else (ep.final or {}).get("after_step"),
           "angle_deg": src.get("r_deg")}
    for ref in REF_BGR:
        row[ref] = (to.get(ref) or {}).get("t_mm")
    # match quality: at the landing step, or - read at the limit - at the last logged step
    mentry = entry if entry is not None else (ep.steps[-1] if ep.steps else None)
    right, near = (mentry or {}).get("right") or {}, (mentry or {}).get("near") or {}
    judged = int(right.get("R", 0)) + int(right.get("W", 0))
    row["right_own_cell"] = f"{right.get('R', 0)}/{judged}" if judged else "-"
    row["within_px"] = f"{near['k']}/{near['n']}" if near.get("n") else "-"
    row["median_miss_px"] = near.get("median_px")
    return row


def render_pair(pair_dir, *, out_dir=None) -> Path:
    """The landing still of every episode of one pair (``<pair>/films/*/``) and one
    ``index.html`` table linking them; into ``out_dir`` (default ``<pair>/debug_view``)."""
    pair = Path(pair_dir)
    dest = Path(out_dir) if out_dir is not None else pair / "debug_view"
    dest.mkdir(parents=True, exist_ok=True)
    eps = sorted(p.parent for p in pair.glob("films/*/telemetry.jsonl.gz"))
    if not eps and (pair / "telemetry.jsonl.gz").is_file():
        eps = [pair]
    rows = []
    for ep_dir in eps:
        # one unreadable episode (the recorder failed in begin, or a kill before the first flush)
        # must not cost the index of the whole pair: it gets a row that says so
        try:
            png = render_still(ep_dir, "landing", out_dir=dest, name=f"{ep_dir.name}.png")
            ep = Episode(ep_dir)
        except (ValueError, OSError, EOFError, KeyError, json.JSONDecodeError) as exc:
            rows.append({"method": ep_dir.name, "start": "-", "landing": f"unreadable ({exc})",
                         "step": None, "goal": None, "size_matched": None, "depth_true": None,
                         "angle_deg": None, "right_own_cell": "-", "within_px": "-",
                         "median_miss_px": None, "still": ""})
            continue
        try:
            rows.append({**_landing_summary(ep), "still": png.name})
        finally:
            ep.close()
    (dest / "index.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def cell(v, fmt="{:.1f}"):
        return "-" if v is None else (fmt.format(v) if isinstance(v, (int, float)) else str(v))

    head = ("<tr><th>method</th><th>start</th><th>landing</th><th>to goal mm</th>"
            "<th>to size-matched mm</th><th>to depth-true mm</th><th>angle deg</th>"
            f"<th>right (own cell)</th><th>within {TRUTH_COMMON_PX:.0f} px</th>"
            "<th>median miss px</th><th>still</th></tr>")
    body = []
    for r in rows:
        tds = [html.escape(cell(r["method"])), html.escape(cell(r["start"])),
               html.escape(f"{r['landing']} @{cell(r['step'], '{}')}")]
        tds += [cell(r[k]) for k in ("goal", "size_matched", "depth_true", "angle_deg")]
        tds += [html.escape(r["right_own_cell"]), html.escape(r["within_px"]),
                cell(r["median_miss_px"], "{:.0f}")]
        still = html.escape(r["still"])
        tds.append(f'<a href="{still}"><img src="{still}" width="320"></a>' if still else "-")
        body.append("<tr>" + "".join(f"<td>{t}</td>" for t in tds) + "</tr>")
    body = "".join(body)
    page = ("<!doctype html><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width'><title>debug view: " + html.escape(pair.name)
            + "</title><style>body{font-family:sans-serif;font-size:15px}table{border-collapse:"
            "collapse}td,th{border:1px solid #ccc;padding:4px 6px;text-align:right}"
            "td:first-child,td:nth-child(2),td:nth-child(3){text-align:left}</style>"
            f"<h2>{html.escape(pair.name)}: where each flight landed</h2>"
            "<p>Landing = the method's own stop if it fired, else the scored stop, else the "
            "final pose. Distances in mm. 'Right (own cell)' uses each method's own matching "
            f"cell, which differs between methods; 'within {TRUTH_COMMON_PX:.0f} px' is the "
            "same yardstick for all. Tap a picture for the full still.</p>"
            f"<table>{head}{body}</table>")
    out = dest / "index.html"
    out.write_text(page, encoding="utf-8")
    return out
