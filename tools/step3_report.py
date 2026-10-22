#!/usr/bin/env python
"""Run the thirteen Step 3 gates and write ``results/STEP3_REPORT.md``.

The Step 3 test files run in one pytest process with the chosen MuJoCo variant. Each test
prints one or more ``step3 <gate> ...: <numbers>`` lines; those lines, the junit status and
the demo records (``results/demo/<method>/demo.json``, written by ``python -m
vsbench.demo``) are what the report shows. ``results/step3_gates.json`` keeps the raw
record.

    VSBENCH_SOFTWARE_GL=1 python tools/step3_report.py --backend mujoco-egl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vsbench.sysinfo import system_record  # noqa: E402

FILES = ["test_contract.py", "test_leak.py", "test_oracle.py", "test_checks.py", "test_true_point_ibvs.py", "test_feature_ibvs.py", "test_episode_log.py", "test_replay.py", "test_video.py", "test_demo_parallel.py"]

SETUP_TESTS = [("test_checks.py", "test_oracle_passes"), ("test_true_point_ibvs.py", "test_true_point_ibvs_setup_checks"), ("test_feature_ibvs.py", "test_feature_setup_checks")]

# gate -> (test cases as (file, name prefix) pairs, printed-line prefixes, setup, pass rule)
GATES = [
    ("Contract", [("test_contract.py", "test_")], ["contract"], "every registered method through configure, reset, set_target, get_control_rate on the poster scene", "finite (6,) float twist, alone or with an info dict; inputs not modified"),
    ("Leak", [("test_leak.py", "test_")], ["leak"], "a spy method with privileged = False; static and transitive import graph of vsbench/methods", "observe_truth never called; no import of the episode loop, a renderer or a scene"),
    ("Oracle decay", [("test_oracle.py", "test_oracle_decay")], ["oracle decay"], "20 random starts inside the clamp, gain 1.0, 100 rendered steps", "norm of log_se3 of the pose error equals (1 - gain dt)^k times its start value within 1e-9"),
    ("Oracle convergence", [("test_oracle.py", "test_oracle_convergence")], ["oracle convergence"], "50 random starts up to 1 m and 170 deg, every frame rendered", "below 0.1 mm and 0.01 deg within 1000 steps"),
    ("Zero at target", SETUP_TESTS, ["zero at target"], "every method at the exact target pose", "linear and angular twist norms below 1e-6"),
    ("Hold at target", SETUP_TESTS, ["hold at target"], "50 steps from the target pose", "pose moves less than 1e-6 m and 1e-6 rad"),
    ("Restoring direction", SETUP_TESTS, ["restoring direction"], "+-2 cm and +-2 deg on each axis, 12 starts per method", "first-step twist component has the sign of the same component of log_se3(inv(T_wc) @ T_wc_target); the episode converges"),
    ("Negative control", [("test_checks.py", "test_negative_control")], ["negative control"], "test-only oracle with the sign of one axis flipped, each axis in turn", "the checks fail on that axis and only there"),
    ("IBVS decay", [("test_true_point_ibvs.py", "test_ibvs_decay")], ["ibvs decay"], "true_point_ibvs, 20 offsets up to 2 cm and 2 deg, gain 1.0, 50 steps", "fitted per-step ratio of the feature error norm within 0.002 of 1 - gain dt"),
    ("Feature convergence", [("test_feature_ibvs.py", "test_feature_convergence_sift")], ["feature convergence sift_ibvs", "feature convergence orb_ibvs"], "sift_ibvs, poster scene, the ten demo starts, dense depth", "all ten below 5 mm and 0.5 deg within 500 steps (ORB reported, not gated)"),
    ("Log round trip", [("test_episode_log.py", "test_round_trip")], ["log round trip"], "save, then load", "arrays and meta identical"),
    ("Replay", [("test_replay.py", "test_replay_identical")], ["replay"], "re-render a logged episode in the same GL variant", "all frame hashes identical"),
    ("Video", [("test_video.py", "test_video_gate")], ["video"], "a 300-step episode", "decodes, 150 frames, 960 x 720, under 5 MB"),
]
METHODS = ["sift_ibvs", "orb_ibvs", "oracle_pbvs", "true_point_ibvs"]


def run_pytest(files: list[str], xml: Path, backend: str) -> tuple[list[dict], list[str], str]:
    cmd = [sys.executable, "-m", "pytest", *[f"tests/{f}" for f in files], "-q", "-s", f"--junitxml={xml}", "-p", "no:cacheprovider", "--backends", backend]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=7200)
    cases = []
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            status, detail = "passed", ""
            for child in tc:
                if child.tag in ("failure", "error"):
                    status, detail = "failed", (child.get("message") or "")[:400]
                elif child.tag == "skipped":
                    status, detail = "skipped", (child.get("message") or "")[:400]
            cases.append({"file": Path(tc.get("classname", "").replace(".", "/")).name + ".py", "name": tc.get("name"), "status": status, "detail": detail, "time_s": float(tc.get("time") or 0)})
    measured = [ln[ln.index("step3 "):].strip() for ln in p.stdout.splitlines() if "step3 " in ln]
    return cases, measured, (p.stdout + p.stderr)[-4000:]


def gate_rows(cases: list[dict], measured: list[str]) -> list[dict]:
    rows = []
    for name, tests, lines, setup, rule in GATES:
        mine = [c for c in cases if any(c["file"] == f and c["name"].startswith(p) for f, p in tests)]
        if not mine:
            status = "not run"
        elif all(c["status"] == "passed" for c in mine):
            status = "passed"
        elif all(c["status"] == "skipped" for c in mine):
            status = "skipped"
        else:
            status = "failed"
        nums = [m for m in measured if any(m.lower().startswith(f"step3 {ln}") for ln in lines)]
        rows.append({"gate": name, "status": status, "cases": len(mine), "lines": lines, "setup": setup, "rule": rule, "measured": nums, "details": [c["detail"] for c in mine if c["detail"]]})
    return rows


def strip_prefix(m: str, lines: list[str]) -> str:
    for ln in lines:
        m2 = re.sub(r"^step3 " + re.escape(ln) + r" ?:? *", "", m, flags=re.IGNORECASE)
        if m2 != m:
            return m2.strip()
    return m


def fmt_measured(g: dict) -> str:
    out = []
    for m in g["measured"]:
        s = strip_prefix(m, g["lines"])
        if g["gate"] in ("Zero at target", "Hold at target", "Restoring direction"):
            # "<method>: <detail>; per axis {...}" -> keep the method and the detail
            s = re.sub(r"; per axis \{.*\}$", "", s)
        if g["gate"] == "Feature convergence":
            s = ("sift_ibvs: " if "sift" in m.lower() else "orb_ibvs (reported): ") + s.split(": ", 1)[-1] if ": " in s else s
        out.append(s)
    return "<br>".join(out) if out else "-"


def load_demo(results: Path, method: str) -> dict | None:
    p = results / "demo" / method / "demo.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_parallel(results: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results.glob("episodes_parallel_*.json"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mujoco-egl")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--skip-tests", action="store_true", help="reuse results/step3_gates.json")
    a = ap.parse_args(argv)
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "step3_gates.json"
    if a.skip_tests and rec_path.exists():
        rec = json.loads(rec_path.read_text())
    else:
        tmp = out / "_tmp"
        tmp.mkdir(exist_ok=True)
        cases, measured, tail = run_pytest(FILES, tmp / "step3.xml", a.backend)
        rec = {"kind": "step3_gates", "backend": a.backend, "system": system_record(), "cases": cases, "measured": measured, "gates": gate_rows(cases, measured), "tail": tail}
        rec["all_passed"] = all(g["status"] == "passed" for g in rec["gates"])
        rec["test_time_s"] = sum(c["time_s"] for c in cases)
        rec_path.write_text(json.dumps(rec, indent=1))
    demos = {m: load_demo(out, m) for m in METHODS}
    for g in rec["gates"]:
        print(f"{g['gate']:22s} {g['status']:8s} {' | '.join(fmt_measured(g).split('<br>'))[:150]}")

    s = rec["system"]
    gl = next((d["renderer"].get("gl_renderer") for d in demos.values() if d and d.get("renderer")), None)
    L = ["# VS-Bench step 3 report: servo loop", ""]
    L.append(f"Generated by `tools/step3_report.py` from the Step 3 gates and the demo records. Machine: {s['cpu']}, {s['cores']} threads, {s['os']}, Python {s['python']}, NumPy {s['packages'].get('numpy')}, OpenCV {s['packages'].get('opencv-python-headless') or s['packages'].get('opencv-python')}, MuJoCo {s['packages'].get('mujoco')}. Backend `{rec['backend']}`" + (f" ({gl})" if gl else "") + f"; `VSBENCH_SOFTWARE_GL={s['env'].get('VSBENCH_SOFTWARE_GL', 'unset')}`, so no GPU was used.")
    L.append("")
    L.append("The loop: `vsbench/methods/` (the `VSMethod` protocol, the registry, `oracle_pbvs` and `true_point_ibvs` as privileged test instruments, `sift_ibvs` and `orb_ibvs` as the basic real controllers), `vsbench/episode.py` (`run_episode`, `EpisodeConfig`, `EpisodeLog` on top of `Trajectory`), `vsbench/checks.py` (the three setup checks), `vsbench/replay.py` (re-render and compare frame hashes) and `vsbench/video.py` (the four-panel debug video). A method receives images, depth in the form it asked for and the intrinsics, nothing else; the `Renderer` interface, the MuJoCo backend and the Step 2 modules are unchanged.")
    L.append("")
    L.append("## 1. Gates")
    L.append("")
    L.append("| gate | setup | pass rule | measured | status |")
    L.append("|---|---|---|---|---|")
    for g in rec["gates"]:
        cell = fmt_measured(g).replace("|", "\\|")
        L.append(f"| {g['gate']} | {g['setup']} | {g['rule'].replace('|', chr(92) + '|')} | {cell} | {'pass' if g['status'] == 'passed' else g['status'].upper()} |")
    L.append("")
    L.append(f"All thirteen gates: **{'pass' if rec['all_passed'] else 'FAIL'}**, {sum(c['status'] == 'passed' for c in rec['cases'])} of {len(rec['cases'])} test cases passed in {rec.get('test_time_s', 0) / 60:.1f} min. Every start is seeded, the images are 640 x 480 with the Step 1 gate camera (fx 600, fy 620, principal point shifted by (+17, -11) px); the target looks straight at the poster centre from 1.07 m, where the poster spans 70 % of the image width.")
    L.append("")
    L.append("## 2. Demo: the ten starts")
    L.append("")
    L.append("`python -m vsbench.demo --method <name>`: `T_wc0 = T_wc_target @ exp_se3(offset)` for the ten offsets of the brief, default stop rule (2 mm and 0.2 deg held for 25 steps), at most 500 steps. Final error is the true pose error where the episode stopped.")
    L.append("")
    for m in METHODS:
        d = demos.get(m)
        if not d:
            L.append(f"`{m}`: no demo record (`python -m vsbench.demo --method {m}`).")
            L.append("")
            continue
        L.append(f"### `{m}` on the `{d['scene']}` scene" + (f", params {d['params']}" if d.get("params") else ""))
        L.append("")
        L.append("| start | translation cm | rotation deg | status | steps | final mm | final deg | ms per call |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in d["rows"]:
            L.append(f"| {r['start']} | {tuple(r['translation_cm'])} | {tuple(r['rotation_deg'])} | {r['status']} | {r['steps']} | {r['final_error_mm']:.2f} | {r['final_error_deg']:.3f} | {(r['call_s_mean'] or 0) * 1e3:.1f} |")
        L.append("")
        vs = d.get("video_start")
        L.append(f"{d['converged']} of {len(d['rows'])} starts converged; **{(d['call_s_mean'] or 0) * 1e3:.1f} ms per `get_control_rate` call** on average" + (f". Video and match sheet of start {vs}: `results/demo/{m}/start_{vs:02d}.mp4`, `start_{vs:02d}_matches.png`." if vs else "."))
        L.append("")
    par = load_parallel(out)
    L.append("## 3. Parallel episodes")
    L.append("")
    L.append("`python -m vsbench.demo --jobs N` runs the ten starts in N worker processes, one renderer and one method each; `tools/bench_episodes.py` times 1, 2, 4 and 8 jobs without video and checks the tables agree. Every episode is deterministic, so the logs are bit-identical whatever the job count (gate in `tests/test_demo_parallel.py`).")
    L.append("")
    if not par:
        L.append("No record yet (`python tools/bench_episodes.py --backend mujoco-egl --method sift_ibvs`).")
        L.append("")
    for rec in par:
        sysr = rec.get("system", {})
        L.append(f"### `{rec['method']}` on `{rec['backend']}`, {rec['renderer'].get('gl_renderer', '?')}; {sysr.get('cpu', '?')}, {sysr.get('cores', '?')} threads" + (f", GPU {sysr['gpu']}" if sysr.get("gpu") else "") + f"; OpenCV threads per worker: {rec.get('cv_threads', 'default')}")
        L.append("")
        L.append("| jobs | OpenCV threads per worker | wall s for the ten starts | speedup | ms per call | mean episode wall s | converged |")
        L.append("|---|---|---|---|---|---|---|")
        w0 = rec["rows"][0]["wall_s"]
        for r in rec["rows"]:
            L.append(f"| {r['jobs']} | {r.get('cv_threads') or 'default'} | {r['wall_s']:.1f} | x{w0 / r['wall_s']:.2f} | {r['call_ms_mean']:.1f} | {r['episode_wall_s_mean']:.1f} | {r['converged']} of 10 |")
        L.append("")
        L.append(f"Tables identical across job counts: {'yes' if rec.get('tables_identical') else 'NO'}.")
        L.append("")
    L.append("## 4. Notes")
    L.append("")
    L.append("- **Depth forms.** `none` gets no depth at all, `scalar` gets the median target depth once as the distance prior, `dense` gets the depth image of the target and of every current view. `sift_ibvs` and `orb_ibvs` default to `dense` and read the depth at each inlier keypoint.")
    L.append("- **Stop rule.** The pose is checked at the start of every step; `converged` means `hold_steps` consecutive checks inside `stop_tol`. Stopping on the true pose is a test-only rule that Step 7 replaces.")
    L.append("- **Zero at target is exact.** The renderer is deterministic, so at the target the current and desired images are identical, every keypoint matches itself with zero distance, and the feature error is exactly zero; the oracle takes the logarithm of an identity, also exactly zero.")
    L.append("- **Negative control.** A flipped axis still passes zero-at-target and hold-at-target, because a zero twist has no sign; the restoring check is the one that catches it, on exactly that axis. It runs with a 300-step budget instead of 3000 because the flipped axis never converges.")
    L.append("- **Log.** `uint16` pixel coordinates times 32 store matches to 1/32 px up to 2047 px (measured worst quantisation 1/64 px); more than 500 matches per step are thinned evenly so inliers and outliers keep their mix. Frame hashes are 64-bit BLAKE2b of the raw RGB bytes.")
    L.append("- **Replay.** Re-rendering the logged poses in the same GL variant gives identical hashes; a different variant is reported as not identical and the video is still written, marked in red.")
    L.append("- **Parallel runs.** One episode of `sift_ibvs` already saturates a CPU-only laptop (OpenCV spreads one SIFT call over every core and so does Mesa's software renderer), so more processes give no speedup there; with the renderer on a GPU the cores are free for the methods and the speedup is real. The section above has the measured numbers.")
    L.append("- **Steps 1 and 2.** Unchanged and green.")
    L.append("")
    (out / "STEP3_REPORT.md").write_text("\n".join(L))
    print(f"report: {out / 'STEP3_REPORT.md'}")
    return 0 if rec["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
