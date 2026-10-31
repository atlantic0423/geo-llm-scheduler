"""LSV2 research arm: reference-path identity, scheduler accounting and component rules."""

import random
from collections import deque
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.lsv2_study import (
    ARMS,
    lsv2_config,
    operator_config,
    run_arm,
    study_config,
)
from geo_llm_scheduler.experiments.rl23_support import study_config as reference_config
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.search import (
    A8Gate,
    LSV2Config,
    LSV2Search,
    SlidingWindowBandit,
    box_hv,
    local_box,
    segment_of,
)
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager

# Digests of (population, Archive) after two generations, computed on the unchanged
# baseline 22eeb2f with ``slot_assignment.run_arm``; seed 16601, case n20 R3M6.
GOLDEN = {
    "BILL_RANDOM": (
        340,
        "70e59479dfaa8a897796831c0df6c8ec994844d09410b538bd5d93bce4dc19b2",
        "8b0943a42c69059353be9c4b4a2ef75232c2138cb2a7f136daaa72d0cac9731e",
    ),
    "MAIN_RANDOM": (
        478,
        "6a148b0d6af0e3c8efdcbcd23d1884158002adb9b14a790de3ea5289fa5a0d8f",
        "64962bc570595b08e2ceb2acdb137669780037b98fa9e5b7c57a6a1dcdbb5d7d",
    ),
}


@pytest.fixture(scope="module")
def problem():
    return case_problem(20, 3, 6, 202610160020, "arrival", 0)


@pytest.mark.parametrize("arm", ["BILL_RANDOM", "MAIN_RANDOM"])
def test_reference_arms_match_original_path_solution_by_solution(problem, arm):
    """BILL/MAIN arms through the LSV2 study entry equal the pre-seam reference digests."""
    cfg = replace(study_config(arm, 16601, 1e6), generations=2)
    result = run_arm(problem, arm, cfg, {})
    exact, population, archive = GOLDEN[arm]
    assert result.gateway.counts["exact"] == exact
    assert digest([solution(c) for c in result.population]) == population
    assert digest([solution(c) for c in result.archive.members]) == archive


def test_arm_configs_are_declared():
    assert ARMS == ("BILL_RANDOM", "MAIN_RANDOM", "LSV2", "LSV2_NOGAP")
    bill = reference_config("BILL_PERMUTED", 3, 60.0)
    assert study_config("LSV2", 3, 60.0) == study_config("LSV2_NOGAP", 3, 60.0) == bill
    assert operator_config(3, 60.0).a3_region_policy == "tariff"
    assert lsv2_config("BILL_RANDOM") is None
    gap, nogap = lsv2_config("LSV2"), lsv2_config("LSV2_NOGAP")
    assert gap is not None and nogap is not None
    assert gap.time_share == 0.20 and gap.seed_rule == "gap"
    assert replace(nogap, seed_rule="gap") == gap and nogap.flow_end_shares == (0.8, 0.15, 0.05)
    assert "A6" not in gap.operators and {"P2_FLOW", "P2_MIXED", "P3", "A8"} <= set(gap.operators)
    with pytest.raises(ValueError):
        LSV2Config(time_share=0.0)
    with pytest.raises(ValueError):
        LSV2Config(operators=("A6",))


def test_segment_box_and_local_hv():
    assert [segment_of((x, 0.5)) for x in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0)] == [0, 0, 1, 1, 2, 2]
    front = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    assert local_box(front, (0.5, 0.5)) == (0.0, 1.0, 0.0, 1.0)
    x_lo, x_hi, y_lo, y_hi = local_box(front, (0.0, 1.0))
    assert x_lo == float("-inf") and (x_hi, y_lo, y_hi) == (0.5, 0.5, 1.1)
    box = (0.0, 1.0, 0.0, 1.0)
    # Only (0.5, 0.5) attains the box interior; (0.25, 0.25) attains [0.25, 1]^2.
    assert box_hv(front, box) == pytest.approx(0.25)
    assert box_hv(front + [(0.25, 0.25)], box) == pytest.approx(0.75**2)
    assert box_hv([(2.0, 2.0)], box) == 0.0
    assert box_hv(front, (1.0, 1.0, 0.0, 1.0)) == 0.0


def test_sliding_window_bandit_explores_then_exploits():
    bandit = SlidingWindowBandit(window=6, exploration=0.0)
    rng = random.Random(0)
    first = {bandit.choose(0, ["x", "y"], rng)[0] for _ in range(20)}
    assert first == {"x", "y"}
    bandit.update(0, "x", 1.0)
    bandit.update(0, "y", 0.1)
    assert bandit.choose(0, ["x", "y"], rng) == (
        "x",
        {"rule": "ucb", "candidates": 2, "score": 1.0, "mean": 1.0, "window_pulls": 1},
    )
    # Another segment's arms are separate; window forgets old pulls.
    assert bandit.choose(1, ["x"], rng)[1]["rule"] == "unplayed"
    for _ in range(6):
        bandit.update(0, "y", 0.5)
    assert bandit.choose(0, ["x", "y"], rng)[1]["rule"] == "unplayed"
    summary = bandit.summary()["arms"]
    assert {(a["segment"], a["operator"], a["pulls"]) for a in summary} == {
        ("flow_end", "x", 1),
        ("flow_end", "y", 7),
    }
    with pytest.raises(ValueError):
        bandit.choose(0, [], rng)


def test_a8_gate_closes_on_empty_calls_and_probes():
    gate = A8Gate(window=4, min_calls=3, empty_rate=0.75, probe_interval=3)
    for _ in range(3):
        assert gate.available(2)
        gate.record(2, True)
    assert [gate.available(2) for _ in range(6)] == [False, False, True, False, False, True]
    assert gate.available(0)
    gate.record(2, False)
    gate.record(2, False)
    assert gate.available(2)
    assert gate.history[2] == deque([True, True, False, False], maxlen=4)


def _seeded_state(problem, seed=16601):
    cfg = study_config("LSV2", seed, 1e6)
    gateway = EvaluationGateway(problem, Archive())
    streams = RNGManager(cfg.seed)
    members = charged_initializer(problem, cfg, "BILL_MOEAD", {})(gateway, streams)
    return cfg, gateway, streams, members


class FakeClock:
    """Deterministic LS timer: every reading advances by one millisecond."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1e-3
        return self.now


def _run_search(problem, seed_rule="gap", elapsed=(0.0, 0.5)):
    cfg, gateway, streams, population = _seeded_state(problem)
    search = LSV2Search(
        LSV2Config(seed_rule=seed_rule, frame_steps=6),
        cfg,
        operator_config(cfg.seed, 1e6),
        timer=FakeClock(),
    )
    captured = []
    gateway.candidate_sink = captured.append
    for value in elapsed:
        search(population, gateway, streams, lambda v=value: v)
    return search, gateway, population, captured


def test_search_respects_time_share_and_is_reproducible(problem):
    search, gateway, population, captured = _run_search(problem)
    assert search.search_start == 0.0 and search.trajectories > 0
    # Stops once LS time reaches its share; at most one trajectory beyond it.
    assert search.ls_seconds >= 0.20 * 0.5
    again, *_ = _run_search(problem)
    assert again.steps == search.steps
    assert again.summary() == search.summary()
    rows = search.steps
    assert {r["segment"] for r in rows} <= {"flow_end", "middle", "bill_end"}
    assert all(r["operator"] in r["available"] and "A6" not in r["available"] for r in rows)
    assert sum(r["exact"] for r in rows) == sum(
        v for k, v in gateway.counts.items() if k.startswith("exact:LSV2:")
    )
    assert all(r["hv_gain"] >= -1e-12 for r in rows)
    assert {r["frame"] for r in rows} == set(range(search.frame_id + 1))
    # Every feasible LS candidate was offered to the Archive; exact is the only truth.
    ls = [c for c in captured if c.origin.startswith("LSV2:")]
    assert ls and gateway.counts["archive_attempts:LSV2:A1"] == sum(
        c.evaluation.feasible for c in ls if c.origin == "LSV2:A1"
    )
    for c in ls[:40]:
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
        if c.origin in ("LSV2:A1", "LSV2:P2_FLOW", "LSV2:P2_MIXED"):
            assert decode(problem, c.genotype) == c.schedule
    assert len(population) == 100 and all(c.evaluation.feasible for c in population)


def test_chain_acceptance_and_reflow_follow_declared_rules(problem):
    search, gateway, population, _ = _run_search(problem, elapsed=(0.0, 1.0))
    rows = search.steps
    by_traj: dict = {}
    for r in rows:
        by_traj.setdefault(r["trajectory"], []).append(r)
    for traj in by_traj.values():
        for a, b in zip(traj, traj[1:]):
            assert b["start_objectives"] == a["next_objectives"]
        for r in traj:
            if r["chain"] == "stay":
                assert r["next_objectives"] == r["start_objectives"]
            if r["chain"] == "dominate":
                nf, nb = r["next_objectives"]
                sf, sb = r["start_objectives"]
                assert nf <= sf and nb <= sb and (nf, nb) != (sf, sb)
            if r["chain"] == "gap":
                assert r["entered_archive"] > 0
    assert any(r["chain"] != "stay" for r in rows)
    assert search.counts.get("reflow_replaced", 0) == sum(r["reflow_replaced"] for r in rows)


def test_reflow_replaces_only_improved_neighbors_of_associated_direction(problem):
    cfg, gateway, streams, population = _seeded_state(problem)
    search = LSV2Search(LSV2Config(), cfg, operator_config(cfg.seed, 1e6), timer=FakeClock())
    lambdas = weights(len(population))
    best_flow = min(gateway.archive.members, key=lambda c: c.evaluation.objectives)
    before = list(population)
    ctx = NormalizationContext(gateway.ideal, maximum(population))
    direction = associate(ctx.normalize(best_flow), lambdas)
    replaced = search.reflow(population, gateway, [best_flow])
    changed = [j for j, (a, b) in enumerate(zip(before, population)) if a is not b]
    assert replaced == len(changed) <= cfg.replacement_cap
    assert set(changed) <= set(search.neighbors[direction])
    for j in changed:
        assert population[j] is best_flow
        assert ctx.scalar(best_flow, lambdas[j]) < ctx.scalar(before[j], lambdas[j])
    assert search.reflow(population, gateway, []) == 0


def test_nogap_rule_starts_mostly_from_flow_end(problem):
    search, *_ = _run_search(problem, seed_rule="flow_end", elapsed=(0.0, 1.0))
    starts = [r for r in search.steps if r["step"] == 0]
    frame_points = [r["start_normalized"][0] for r in starts]
    assert starts and sum(x < 1 / 3 for x in frame_points) / len(frame_points) > 0.5


def test_frozen_frame_hv_reference():
    frame = FrozenFrame((0.0, 0.0), (1.0, 1.0))
    assert frame.hv([(0.0, 0.0)]) == pytest.approx(1.21)


def _load_script(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_prepare_binds_original_inputs_and_preflight(tmp_path, monkeypatch, problem):
    import json

    from test_polish_policy_campaign import historical_fixture

    runner = _load_script("run_lsv2")
    old = tmp_path / "old"
    historical_fixture(old)
    manifest = runner.prepare(tmp_path / "formal", old)
    assert manifest["runs"] == 128 and len(manifest["blocks"]) == 32
    assert manifest["lsv2_configs"]["BILL_RANDOM"] is None
    assert manifest["lsv2_configs"]["LSV2_NOGAP"]["seed_rule"] == "flow_end"
    for key in manifest["blocks"]:
        spec = json.loads((tmp_path / "formal" / "specs" / (key + ".json")).read_text())
        assert spec["seed"] == 16601 and set(spec["order"]) == set(ARMS)
    runner.validate_inputs(tmp_path / "formal")
    monkeypatch.setattr(runner, "load_instance", lambda path: problem)
    check = runner.preflight(tmp_path / "formal", manifest["blocks"][0])
    assert check["policies"]["bill4"]["fresh_exact_verified"] == 140
    pilot = runner.prepare(tmp_path / "pilot", old, pilot=True)
    assert pilot["runs"] == 16
    assert runner.required("LSV2")[-2:] == ("lsv2_steps.json.gz", "lsv2_summary.json")
    assert "lsv2_steps.json.gz" not in runner.required("MAIN_RANDOM")
    flat = _load_script("slot_assignment_dispatch")
    topology = {c: ("0", str(c % 4)) for c in range(8)}
    assert flat.cpu_order(set(range(8)), topology, 3, frozenset({0})) == [1, 2, 3]


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="worker uses fcntl/affinity")
@pytest.mark.parametrize("arm", ["LSV2", "MAIN_RANDOM"])
def test_worker_writes_audited_artifacts(tmp_path, monkeypatch, problem, arm):
    import gzip
    import json
    import os

    from test_polish_policy_campaign import historical_fixture

    runner = _load_script("run_lsv2")
    old = tmp_path / "old"
    historical_fixture(old)
    monkeypatch.setattr(runner, "PILOT_SECONDS", {n: 6 for n in (20, 50, 100, 200)})
    manifest = runner.prepare(tmp_path / "pilot", old, pilot=True)
    root = tmp_path / "pilot"
    key = manifest["blocks"][0]
    monkeypatch.setattr(runner, "load_instance", lambda path: problem)
    runner.preflight(root, key)
    runner.worker(root, key, arm, sorted(os.sched_getaffinity(0))[0])
    assert runner.complete(root, key, arm, manifest["source"])
    out = root / "runs" / key / arm
    with gzip.open(out / "archive_events.json.gz", "rt") as f:
        events = json.load(f)
    assert events and all(a[0] <= b[0] for a, b in zip(events, events[1:]))
    summary = json.loads((out / "summary.json").read_text())
    if arm == "LSV2":
        assert summary["lsv2"]["steps"] > 0 and 0.05 < summary["lsv2"]["share"] < 0.5


def test_uniform_selector_is_a_matched_control(problem):
    from geo_llm_scheduler.experiments.lsv2_study import PANEL_ARMS

    assert PANEL_ARMS == ("BILL_RANDOM", "MAIN_RANDOM", "LSV2", "LSV2_RANDSEL")
    rand = lsv2_config("LSV2_RANDSEL")
    assert rand is not None and replace(rand, selector="bandit") == lsv2_config("LSV2")
    assert study_config("LSV2_RANDSEL", 3, 60.0) == study_config("LSV2", 3, 60.0)
    with pytest.raises(ValueError):
        LSV2Config(selector="greedy")
    cfg, gateway, streams, population = _seeded_state(problem)
    runs = []
    for _ in range(2):
        cfg, gateway, streams, population = _seeded_state(problem)
        search = LSV2Search(rand, cfg, operator_config(cfg.seed, 1e6), timer=FakeClock())
        for value in (0.0, 0.5):
            search(population, gateway, streams, lambda v=value: v)
        runs.append(search.steps)
    assert runs[0] == runs[1] and runs[0]
    assert all(
        r["bandit"]["rule"] == "uniform" and r["operator"] in r["available"] for r in runs[0]
    )
    assert len({r["operator"] for r in runs[0]}) > 3


def test_panel_prepare_copies_selected_stress_inputs(tmp_path):
    import json

    from geo_llm_scheduler.experiments.stress_instances import FACTORS, stress_problem
    from geo_llm_scheduler.io.loaders import save_instance

    runner = _load_script("run_lsv2")
    source = tmp_path / "screen"
    (source / "inputs").mkdir(parents=True)
    items = []
    for i, n in enumerate((20, 50, 100, 200)):
        levels = dict.fromkeys(FACTORS, i % 2)
        path = source / "inputs" / f"st_n{n}_d{i:02d}.json"
        save_instance(stress_problem(n, 3, levels, 100 + i), path)
        items.append(
            dict(
                key=path.stem,
                panel="dev" if i % 2 else "holdout",
                n=n,
                regions=3,
                levels=levels,
                input=f"inputs/{path.name}",
                input_sha256=runner.sha(path),
            )
        )
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps(dict(instances=items)))
    manifest = runner.prepare_panel(tmp_path / "formal", panel, source)
    assert manifest["runs"] == 16 and tuple(manifest["arms"]) == runner.PANEL_ARMS
    assert manifest["lsv2_configs"]["LSV2_RANDSEL"]["selector"] == "uniform"
    runner.validate_inputs(tmp_path / "formal")
    spec = json.loads(
        (tmp_path / "formal" / "specs" / (manifest["blocks"][3] + ".json")).read_text()
    )
    assert spec["seconds"] == 2400 and spec["seed"] == runner.PANEL_SEED and spec["panel"] == "dev"
    assert set(spec["order"]) == set(runner.PANEL_ARMS)
