"""Stress generator v2: design balance, determinism, factor effects and screening runner."""

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments.stress_instances import (
    FACTORS,
    candidate_specs,
    design_points,
    stress_problem,
)

SCRIPTS = Path(__file__).parents[1] / "scripts"
BASE = dict.fromkeys(FACTORS, 0)


def test_fractional_design_is_balanced_and_pairwise_orthogonal():
    points = design_points()
    assert len(points) == 16 and len({tuple(p.values()) for p in points}) == 16
    for f in FACTORS:
        assert sum(p[f] for p in points) == 8
    for a, b in combinations(FACTORS, 2):
        cells = [(p[a], p[b]) for p in points]
        assert all(cells.count(c) == 4 for c in ((0, 0), (0, 1), (1, 0), (1, 1)))
    assert points[0] == BASE and points[15] == dict.fromkeys(FACTORS, 1)
    specs = candidate_specs()
    assert len(specs) == 64 and len({s["key"] for s in specs}) == 64
    assert len({s["seed"] for s in specs}) == 64


def test_generation_is_deterministic_and_level_zero_matches_component33_setting():
    a = stress_problem(50, 3, BASE, 7)
    assert a == stress_problem(50, 3, BASE, 7) and a != stress_problem(50, 3, BASE, 8)
    tariffs = a.regions[0].tariffs
    assert tariffs[0].end - tariffs[0].start == 4800
    assert [t.price for t in tariffs[:3]] == [0.35, 0.8, 0.5]
    assert a.regions[0].demand_rate == 5.0
    assert max(j.release for j in a.jobs) <= 3600
    assert all(j.kv_delay in (15, 30, 60) for j in a.jobs)


def test_stressed_levels_change_only_their_pressure():
    for factor in FACTORS:
        levels = dict(BASE, **{factor: 1})
        p = stress_problem(100, 3, levels, 11)
        counts = [sum(m.region == r for m in p.instances) for r in range(3)]
        assert min(counts) >= 2
        horizon = max(j.release for j in p.jobs) + sum(
            j.prefill.duration + j.decode.duration + j.kv_delay for j in p.jobs
        )
        assert all(r.tariffs[-1].end >= horizon for r in p.regions)
        if factor == "tariff":
            t = p.regions[0].tariffs
            assert (
                t[0].end == 1800 and max(x.price for x in t[:3]) / min(x.price for x in t[:3]) == 5
            )
            # Regions are phase-shifted: their first-period prices differ in level.
            assert len({p.regions[r].tariffs[0].price / (1 + 0.15 * r) for r in range(3)}) == 3
        if factor == "arrival":
            assert max(j.release for j in p.jobs) > 3600
        if factor == "demand":
            assert p.regions[0].demand_rate == 0.5
        if factor == "region":
            assert counts[0] <= counts[1] <= counts[2]
            assert all(j.kv_delay in (60, 120, 240) for j in p.jobs)
        if factor == "jobs":
            assert max(j.decode.duration for j in p.jobs) > 900
    with pytest.raises(ValueError):
        stress_problem(50, 4, BASE, 1)
    with pytest.raises(ValueError):
        stress_problem(50, 3, dict(BASE, load=2), 1)


def test_screen_prepare_seals_generated_inputs(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "run_stress_screen", SCRIPTS / "run_stress_screen.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = runner.prepare(tmp_path / "pilot", pilot=True)
    assert manifest["runs"] == 12 and len(manifest["blocks"]) == 4
    runner.validate_inputs(tmp_path / "pilot")
    key = [k for k in manifest["blocks"] if k.startswith("st_n20")][0]
    s = json.loads((tmp_path / "pilot" / "specs" / (key + ".json")).read_text())
    assert s["seconds"] == 2 * s["base_seconds"] == 90
    check = runner.preflight(tmp_path / "pilot", key)
    assert {a: v["fresh_exact_verified"] for a, v in check["arms"].items()} == dict.fromkeys(
        runner.ARMS, 140
    )
    assert runner.arm_config("BILL_C2X", 300).seconds == 600
