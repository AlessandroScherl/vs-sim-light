#!/usr/bin/env python
"""Run the four gates for each backend variant, one pytest process per gate file so a
backend that keeps one renderer per process (the contract) is never asked for more than
one scene family per process. Writes results/gates_<backend>.json.

    python tools/run_gates.py --backends mujoco-egl pybullet-tiny open3d-cpu
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
from vsbench.render.base import BACKENDS, package_installed, probe_backend  # noqa: E402
from vsbench.sysinfo import system_record  # noqa: E402

GATES = ["projection", "depth", "homography", "determinism"]


def run_gate(backend: str, gate: str, tmp: Path) -> dict:
    xml = tmp / f"{backend}_{gate}.xml"
    cmd = [sys.executable, "-m", "pytest", f"tests/test_{gate}.py", "-q", "-s", "--backends", backend, f"--junitxml={xml}", "-p", "no:cacheprovider"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    cases = []
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            status = "passed"
            detail = ""
            for child in tc:
                if child.tag in ("failure", "error"):
                    status = "failed"
                    detail = (child.get("message") or "")[:400]
                elif child.tag == "skipped":
                    status = "skipped"
                    detail = (child.get("message") or "")[:400]
            cases.append({"name": tc.get("name"), "status": status, "detail": detail, "time_s": float(tc.get("time") or 0)})
    numbers = [ln for ln in p.stdout.splitlines() if ln.startswith(backend)]
    status = "passed" if cases and all(c["status"] == "passed" for c in cases) else ("skipped" if cases and all(c["status"] == "skipped" for c in cases) else "failed")
    return {"gate": gate, "status": status, "cases": cases, "measured": numbers, "returncode": p.returncode, "tail": (p.stdout + p.stderr)[-1500:] if status == "failed" else ""}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--label", default=None, help="record label and file suffix (single backend only)")
    a = ap.parse_args(argv)
    if a.label and len(a.backends) != 1:
        ap.error("--label needs exactly one backend")
    out_dir = Path(a.out_dir)
    tmp = out_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for backend in a.backends:
        label = a.label or backend
        rec = {"kind": "gates", "backend": backend, "label": label, "system": system_record(), "gates": []}
        if not package_installed(backend):
            rec["available"] = False
            rec["reason"] = f"package {BACKENDS[backend][3]!r} not installed"
        else:
            ok, why = probe_backend(backend)
            rec["available"] = ok
            rec["reason"] = "" if ok else why
            if ok:
                rec["renderer"] = json.loads(why) if why.startswith("{") else {}
                for gate in GATES:
                    g = run_gate(backend, gate, tmp)
                    rec["gates"].append(g)
                    print(f"{label:16s} {gate:12s} {g['status']:8s} {' | '.join(re.sub(r'^' + re.escape(backend) + r' ', '', m) for m in g['measured'])}")
        if not rec["available"]:
            print(f"{backend:14s} unavailable: {rec['reason'].splitlines()[-1] if rec['reason'] else ''}")
        rec["all_passed"] = rec["available"] and bool(rec["gates"]) and all(g["status"] == "passed" for g in rec["gates"])
        (out_dir / f"gates_{label}.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
