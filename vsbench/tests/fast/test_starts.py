# vs-sim copy: tests/fast/test_starts.py @ 7ac261d7384861b1d1ad0cd645c6e51b5046b650 (tools/vssim_copy.py); edits: rewrite rules  # noqa: E501
"""Start-set parity: the frozen files regenerate from their recorded seeds.

The two references are the benchmark's own hollywood files (``benchmark/starts/``):

- ``hollywood_starts_v2.json`` - the campaign file built on the station on
  2026-08-20 (seed 20260820, md5 ``9a242541b0d6d32f2e399bfc971e368d``);
- ``hollywood_starts_near.json`` - the terminal-phase stratum (seed 20260902,
  md5 ``7e768de50e0e39346def7097115dfc63``).

**Content, not md5.** Regeneration reproduces every id, cell, count and share
exactly, and every rotation matrix to ~4e-16: the frozen files were written on the
station's numpy/BLAS and the residual is the matrix product in ``look_at_R`` and
``roll_about_z``, not the code. An md5 comparison would fail on any other machine
and would prove nothing about the sampler, so the comparison is structural plus a
numeric tolerance, and the md5 is asserted for the *fixture* (that the reference
has not been edited).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vsbench.vs import starts

STARTS = Path(__file__).resolve().parents[2] / "benchmark" / "starts"
WIDE_FILE = STARTS / "hollywood_starts_v2.json"
NEAR_FILE = STARTS / "hollywood_starts_near.json"
WIDE_MD5 = "9a242541b0d6d32f2e399bfc971e368d"
NEAR_MD5 = "7e768de50e0e39346def7097115dfc63"

#: The station's residual on a regenerated rotation matrix (measured 4.4e-16).
POSE_TOL = 1e-12


@pytest.fixture(scope="module")
def frozen_wide() -> dict:
    return json.loads(WIDE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_near() -> dict:
    return json.loads(NEAR_FILE.read_text(encoding="utf-8"))


def _goal(payload: dict):
    g = payload["goal"]
    return np.array(g["cam_pos"]), np.array(g["cam_R_opt"]), float(g["distance_m"])


def _assert_starts_equal(got: dict, want: dict) -> None:
    assert set(got) == set(want)
    for sid, w in want.items():
        g = got[sid]
        assert set(g) == set(w), sid
        for field in ("start_id", "cell", "circle", "tier", "roll_band"):
            assert g[field] == w[field], f"{sid}.{field}"
        for field in ("roll_deg", "pos_err_m", "rot_err_deg"):
            if w[field] is None:
                assert g[field] is None, f"{sid}.{field}"
            else:
                assert g[field] == pytest.approx(w[field], abs=POSE_TOL), f"{sid}.{field}"
        assert np.allclose(g["cam_pos"], w["cam_pos"], atol=POSE_TOL), sid
        assert np.allclose(g["cam_R_opt"], w["cam_R_opt"], atol=POSE_TOL), sid


def test_the_reference_start_files_are_the_published_ones():
    assert hashlib.md5(WIDE_FILE.read_bytes()).hexdigest() == WIDE_MD5
    assert hashlib.md5(NEAR_FILE.read_bytes()).hexdigest() == NEAR_MD5


def test_regenerating_the_campaign_file_reproduces_it(frozen_wide):
    g, R, d = _goal(frozen_wide)
    got = starts.build_payload("hollywood", g, R, d, seed=frozen_wide["seed"],
                               hardware=frozen_wide["hardware"],
                               created=frozen_wide["created"])
    for field in ("schema", "version", "scene", "seed", "created", "hardware", "scheme"):
        assert got[field] == frozen_wide[field], field
    assert got["goal"] == frozen_wide["goal"]
    assert got["cells"] == frozen_wide["cells"]          # every id list, in order
    _assert_starts_equal(got["starts"], frozen_wide["starts"])
    assert set(got["stats"]) == set(frozen_wide["stats"])
    for cell, want in frozen_wide["stats"].items():
        if isinstance(want, dict):
            assert json.dumps(sorted(want)) == json.dumps(sorted(got["stats"][cell])), cell


def test_the_campaign_statistics_reproduce_to_machine_precision(frozen_wide):
    g, R, d = _goal(frozen_wide)
    got = starts.build_payload("hollywood", g, R, d, seed=frozen_wide["seed"])["stats"]
    want = frozen_wide["stats"]
    for cell in ("wide", "wide100", "wide20", "wideroll10", "narrow", "wide_le120_subset"):
        for k, v in want[cell].items():
            assert got[cell][k] == pytest.approx(v, abs=1e-10), f"{cell}.{k}"
    assert got["wide_roll_band_shares"] == want["wide_roll_band_shares"]
    # the published-parity SUB-DISTRIBUTION: restricting the widened wide cell to the
    # papers' +-120 deg cap must land in the papers' band (~47 +- 16 cm / 74 +- 28 deg,
    # the `reference` row). hollywood's is 46 cm / 80 deg - inside it, which is the
    # comparability claim the 2026-08-20 widening had to survive.
    sub = got["wide_le120_subset"]
    assert got["reference"].startswith("papers report")
    assert abs(sub["pos_cm_mean"] - 47) <= 16 and abs(sub["rot_deg_mean"] - 74) <= 28
    assert round(got["wide_roll_band_shares"]["ge150"], 3) == 0.176


def test_regenerating_the_near_stratum_reproduces_it(frozen_near):
    g, R, d = _goal(frozen_near)
    got = starts.build_near_payload("hollywood", g, R, d, seed=frozen_near["seed"],
                                    hardware=frozen_near["hardware"],
                                    created=frozen_near["created"])
    for field in ("schema", "version", "scene", "seed", "created", "probe", "scheme",
                  "goal", "cells"):
        assert got[field] == frozen_near[field], field
    _assert_starts_equal(got["starts"], frozen_near["starts"])
    for k, v in frozen_near["stats"]["near"].items():
        assert got["stats"]["near"][k] == pytest.approx(v, abs=1e-10), k


def test_the_near_stratum_is_where_the_probe_says_it_is(frozen_near):
    """3-8 mm and at most 3 deg - the stratum's whole point is starting inside the parking phase."""
    entries = list(frozen_near["starts"].values())
    assert len(entries) == starts.NEAR["n"] == 60
    assert starts.NEAR["seed"] == 20260902
    pos_mm = [e["pos_err_m"] * 1000.0 for e in entries]
    rot = [e["rot_err_deg"] for e in entries]
    assert min(pos_mm) >= 3.0 and max(pos_mm) <= 8.0
    assert max(rot) <= 3.0
    assert all(e["cell"] == "near" and e["roll_deg"] is None for e in entries)


def test_the_near_file_is_its_own_file_and_carries_no_campaign_cell(frozen_near):
    assert set(frozen_near["cells"]) == {"near"}
    with pytest.raises(ValueError, match="unknown start cell"):
        starts.starts_for_cell(frozen_near, "wide100")


def test_the_draw_order_of_the_wide_cell_is_frozen(frozen_wide):
    """Positions for ALL 500 samples first, then the thetas circle by circle.

    Re-ordering a single `rng` call re-deals every campaign start silently, so the
    order is pinned twice: the frozen file's first start must come out, and the
    plausible alternative (drawing each circle's positions and thetas together)
    must NOT - otherwise the test would pass on either implementation.
    """
    g, R, d = _goal(frozen_wide)
    got = starts.sample_wide(g, R, d, seed=starts.SEED)
    want = frozen_wide["starts"]["w1_000"]
    assert got[0]["cam_pos"] == pytest.approx(want["cam_pos"], abs=POSE_TOL)
    assert np.allclose(got[0]["cam_R_opt"], want["cam_R_opt"], atol=POSE_TOL)

    n_c, per = starts.WIDE["n_circles"], starts.WIDE["per_circle"]
    rng = np.random.default_rng(starts.SEED)
    interleaved_first_theta = None
    for _ in range(n_c):
        rng.uniform(-0.6, 0.6, size=(per, 2))
        rng.uniform(-0.15, 0.15, size=(per, 1))
        theta = rng.uniform(-np.pi, np.pi, size=per)
        interleaved_first_theta = theta[0] if interleaved_first_theta is None \
            else interleaved_first_theta
    rng = np.random.default_rng(starts.SEED)
    rng.uniform(-0.6, 0.6, size=(n_c * per, 2))
    rng.uniform(-0.15, 0.15, size=(n_c * per, 1))
    frozen_first_theta = rng.uniform(-np.pi, np.pi, size=per)[0]
    assert interleaved_first_theta != frozen_first_theta


def test_the_wide20_cell_is_a_prefix_of_wide100(frozen_wide):
    """Appendability: extending the admission set to the campaign set only appends."""
    ids100 = frozen_wide["cells"]["wide100"]["ids"]
    ids20 = frozen_wide["cells"]["wide20"]["ids"]
    per = starts.WIDE_CAMPAIGN_PER_CIRCLE
    for c in range(starts.WIDE["n_circles"]):
        block = ids100[c * per:(c + 1) * per]
        assert block[:starts.WIDE_ADMISSION_PER_CIRCLE] == \
            ids20[c * starts.WIDE_ADMISSION_PER_CIRCLE:(c + 1) * starts.WIDE_ADMISSION_PER_CIRCLE]
    roll = frozen_wide["cells"]["wideroll10"]["ids"]
    assert len(roll) == 10 and not set(roll) & set(ids20)
    assert all(abs(frozen_wide["starts"][s]["roll_deg"]) >= starts.EXTREME_ROLL_MIN_DEG
               for s in roll)


# ------------------------------------------------------------ the set reads
def test_the_thirds_partition_is_deterministic_and_thirty_three_each(frozen_wide):
    """The wide-100 band in three thirds by initial rotation."""
    rows = [dict(frozen_wide["starts"][i], start_rotation_deg=frozen_wide["starts"][i]
                 ["rot_err_deg"]) for i in frozen_wide["cells"]["wide100"]["ids"]]
    thirds = starts.roll_thirds(rows)
    assert [len(thirds[k]) for k in starts.THIRDS] == [33, 33, 34]
    again = starts.roll_thirds(list(reversed(rows)))
    assert [[r["start_id"] for r in again[k]] for k in starts.THIRDS] == \
           [[r["start_id"] for r in thirds[k]] for k in starts.THIRDS]
    lo = max(r["start_rotation_deg"] for r in thirds["small"])
    hi = min(r["start_rotation_deg"] for r in thirds["medium"])
    assert lo <= hi
    assert (min(r["start_rotation_deg"] for r in thirds["large"])
            >= max(r["start_rotation_deg"] for r in thirds["medium"]))


def test_the_capped_population_is_the_screening_regime(frozen_wide):
    """"The first 30 of the frozen wide-100 starts per scene with roll <= 90 deg"."""
    capped = starts.capped_population(frozen_wide, cap=90.0, n=30)
    assert len(capped) == 30
    assert all(v <= 90.0 for v in capped.values())
    ordered = [i for i in frozen_wide["cells"]["wide100"]["ids"] if i in capped]
    assert list(capped) == ordered          # stored order, never sorted by roll


# --------------------------------------------------------------- file rules
def test_write_payload_pins_its_own_checksum(tmp_path, frozen_near):
    p = tmp_path / "x_starts_near.json"
    digest = starts.write_payload(p, frozen_near)
    assert digest == starts.payload_md5(frozen_near)
    assert p.with_suffix(".json.md5").read_text().split()[0] == digest
    assert starts.load_payload(p, expect_md5=digest)["scene"] == "hollywood"
    with pytest.raises(ValueError, match="md5"):
        starts.load_payload(p, expect_md5="0" * 32)


def test_load_payload_refuses_a_file_that_is_not_a_start_set(tmp_path):
    p = tmp_path / "nope.json"
    p.write_text(json.dumps({"schema": "something_else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a starts_v2 file"):
        starts.load_payload(p)


def test_check_goal_match_refuses_a_drifted_goal(frozen_wide):
    g, R, d = _goal(frozen_wide)
    starts.check_goal_match(frozen_wide, g, R, d)               # the real goal passes
    with pytest.raises(ValueError, match="goal mismatch"):
        starts.check_goal_match(frozen_wide, g + np.array([0.0, 0.0, 0.01]), R, d)
    with pytest.raises(ValueError, match="goal mismatch"):
        starts.check_goal_match(frozen_wide, g, starts.roll_about_z(R, 1.0), d)
    with pytest.raises(ValueError, match="goal mismatch"):
        starts.check_goal_match(frozen_wide, g, R, d + 0.01)


def test_starts_for_cell_stamps_the_requested_cell(frozen_wide):
    thirty = starts.starts_for_cell(frozen_wide, "wide100_30")
    assert len(thirty) == 30
    assert all(s["cell"] == "wide100" for s in thirty)
    assert [s["start_id"] for s in thirty] == frozen_wide["cells"]["wide100"]["ids"][:30]
    roll = starts.starts_for_cell(frozen_wide, "wideroll10")
    assert all(s["cell"] == "wideroll10" for s in roll)   # they are `wide` starts underneath
    with pytest.raises(ValueError, match="out of range"):
        starts.starts_for_cell(frozen_wide, "wide100_500")


def test_cell_size_static_needs_no_file():
    assert starts.cell_size_static("wide100") == 100
    assert starts.cell_size_static("wide20_5") == 5
    assert starts.cell_size_static("near") == 60
    with pytest.raises(ValueError, match="unknown start cell"):
        starts.cell_size_static("nonsense")


# ---------------------------------------------- the simple cell (instance suite, 2026-09-23)
def _simple_goal():
    from vsbench.vs.scenes import R_LOOK_DOWN

    return np.array([0.0, 0.0, 0.5]), R_LOOK_DOWN.copy(), 0.25


def test_the_simple_cell_is_near_the_goal_looks_at_the_object_and_rolls_less_than_60_deg():
    g, R_g, d = _simple_goal()
    cell = starts.sample_simple(g, R_g, d)
    assert len(cell) == starts.SIMPLE["n"] == 10
    assert [s["roll_deg"] for s in cell] == list(starts.SIMPLE["rolls_deg"])
    assert sorted(s["roll_deg"] for s in cell[:5]) == [-54.0, -30.0, -6.0, 18.0, 42.0]
    assert max(abs(s["roll_deg"]) for s in cell) < 60.0
    aim0 = starts.goal_look_at(g, R_g, d)
    for s in cell:
        pos, R = np.asarray(s["cam_pos"]), np.asarray(s["cam_R_opt"])
        off = np.linalg.norm(pos - g) / d
        assert 0.10 - 1e-12 <= off <= np.hypot(0.35, 0.40) + 1e-12
        assert s["offset_frac"] == pytest.approx(off)
        # the optical axis passes within the aim jitter (5 % of d) of the goal's aim point
        z = R[:, 2]
        miss = np.linalg.norm(np.cross(aim0 - pos, z))
        assert miss <= 0.05 * d + 1e-9
        # the roll about the viewing axis is the listed one: undo the look-at, read it
        look = starts.look_at_R(pos, pos + z * np.linalg.norm(aim0 - pos), y_ref_down=R_g[:, 1])
        rel = look.T @ R
        assert np.degrees(np.arctan2(rel[1, 0], rel[0, 0])) == pytest.approx(s["roll_deg"])
        assert s["cell"] == "simple" and s["start_id"].startswith("s_")


def test_the_draw_order_of_the_simple_cell_is_frozen():
    """Re-ordering one rng call re-deals every start: two numbers pin the whole order."""
    g, R_g, d = _simple_goal()
    cell = starts.sample_simple(g, R_g, d)
    np.testing.assert_allclose(cell[0]["cam_pos"], [0.02330787231627801, -0.08367627696220649,
                                                    0.5885480244739123], atol=1e-12)
    np.testing.assert_allclose(cell[9]["cam_pos"], [-0.06404818378546226, -0.03378666195480877,
                                                    0.5351389806377127], atol=1e-12)
    again = starts.sample_simple(g, R_g, d)
    assert [s["cam_pos"] for s in again] == [s["cam_pos"] for s in cell]
    other = starts.sample_simple(g, R_g, d, seed=1)
    assert other[0]["cam_pos"] != cell[0]["cam_pos"]


def test_the_simple_payload_is_its_own_file_with_a_five_start_prefix(tmp_path):
    g, R_g, d = _simple_goal()
    payload = starts.build_simple_payload("inst_x", g, R_g, d, created="2026-09-23")
    assert payload["schema"] == starts.SCHEMA and list(payload["cells"]) == ["simple"]
    assert payload["seed"] == starts.SIMPLE["seed"] and payload["probe"] == "instance_suite"
    assert payload["stats"]["simple"]["n"] == 10 and payload["stats"]["simple5"]["n"] == 5
    digest = starts.write_payload(tmp_path / "inst_x_starts_simple.json", payload)
    back = starts.load_payload(tmp_path / "inst_x_starts_simple.json", expect_md5=digest)
    five = starts.starts_for_cell(back, "simple_5")
    assert [s["start_id"] for s in five] == ["s_00", "s_01", "s_02", "s_03", "s_04"]
    assert all(s["cell"] == "simple" and s["starts_seed"] == starts.SIMPLE["seed"] for s in five)
    assert starts.cell_size_static("simple") == 10 and starts.cell_size_static("simple_5") == 5
