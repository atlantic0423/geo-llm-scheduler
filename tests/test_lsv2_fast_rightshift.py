"""Right-only feasibility, episode credit, shared time, controls and persisted evidence."""

import copy
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from random import Random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import (
    Candidate,
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.lsv2_fast_rightshift import (
    ARMS,
    BASE,
    NO_P3,
    POLISH_ARMS,
    RUN_SEED,
    arm_config,
    configuration_record,
    make_search,
    search_config,
)
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.polish import FastRightShiftPolish
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search
from geo_llm_scheduler.utils.rng import RNGManager

CAL = dict(rho=0.001, h=0.01, w_cutoff=1.5)
ALL = next(a for a, pair in POLISH_ARMS.items() if pair == ("all", "all"))


def synthetic(n=2, *, left_only=False):
    tariffs = (Tariff(0, 50, 1 if left_only else 10), Tariff(50, 5000, 10 if left_only else 1))
    problem = ProblemInstance(
        tuple(Job(str(i), 0, Profile(10, 1, 1), Profile(10, 1, 1)) for i in range(n)),
        (Region("r", tariffs, 0.1),),
        tuple(ServingInstance(str(i), 0, 2, 0.1, 1) for i in range(n)),
    )
    genotype = Genotype(tuple(i for i in range(n) for _ in range(2)), tuple(range(n)) * 2)
    schedule = Schedule(
        tuple(t for i in range(n) for t in ((60, 70) if left_only else (0, 100 + i * 20)))
    )
    return problem, Candidate(genotype, schedule, evaluate(problem, genotype, schedule))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


@pytest.mark.parametrize("n", [2, 20, 25, 100, 200])
def test_targets_right_only_exact_move_caps_and_reproducibility(n):
    problem, current = synthetic(n)
    for all_targets in (False, True):
        op = FastRightShiftPolish(all_targets)
        rng = Random(9)
        batch = op.propose(problem, current, 3, Config(), rng)
        same = op.propose(problem, current, 3, Config(), Random(9))
        assert batch == same and batch.proposals
        assert batch.diagnostics["eligible"] == n
        assert batch.diagnostics["targets"] == (n if all_targets else min(n, max(20, (n + 4) // 5)))
        assert max(batch.diagnostics["moves"]) <= max(5, (n + 9) // 10)
        for prop in batch.proposals:
            exact = evaluate(problem, prop.genotype, prop.schedule)
            assert exact.feasible and exact.flow == current.evaluation.flow
            assert exact.bill < current.evaluation.bill
            assert prop.schedule.starts[1::2] == current.schedule.starts[1::2]
            assert all(a >= b for a, b in zip(prop.schedule.starts, current.schedule.starts))
            assert all(o % 2 == 0 for o in prop.selected)
    if n <= 20:
        assert FastRightShiftPolish(False).propose(
            problem, current, 3, Config(), Random(8)
        ) == FastRightShiftPolish(True).propose(problem, current, 3, Config(), Random(8))


def test_no_right_slack_does_not_become_a_left_move():
    problem, current = synthetic(left_only=True)
    assert FlowSafeBillMoves().propose(problem, current, 3, Config(), Random(1)).proposals
    batch = FastRightShiftPolish().propose(problem, current, 3, Config(), Random(1))
    assert batch.diagnostics["eligible"] == 0 and not batch.proposals and batch.attempts == 0
    with pytest.raises(ValueError):
        FastRightShiftPolish().propose(problem, current, 0, Config(), Random(1))


def test_original_p3_matches_frozen_source_and_rng():
    path = Path(__file__).parent / "reference/lsv2_flowsafe_0413113.py"
    spec = importlib.util.spec_from_file_location("p3_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for left in (False, True):
        problem, current = synthetic(25, left_only=left)
        old_rng, new_rng = Random(7), Random(7)
        assert module.FlowSafeBillMoves().propose(
            problem, current, 3, Config(), old_rng
        ) == FlowSafeBillMoves().propose(problem, current, 3, Config(), new_rng)
        assert old_rng.getstate() == new_rng.getstate()


def test_ten_configs_and_fail_closed_masks():
    assert len(ARMS) == 10 and len(POLISH_ARMS) == 8 and RUN_SEED == 16901
    for arm in ARMS:
        config = search_config(arm)
        assert config.budget == config.trajectory_steps == config.polish_budget == 3
        assert config.time_share == 0.2
        assert ("P3" in config.operators) == (arm == BASE)
        record = configuration_record(arm, RUN_SEED, 100, CAL)
        assert record["lsv2"] == asdict(config)
        assert record["qspec"]["gamma"] == 0.9
        assert arm_config(arm, RUN_SEED, 100) == arm_config(BASE, RUN_SEED, 100)
    for kwargs in (
        dict(polish_region="bad"),
        dict(polish_targets="bad"),
        dict(polish_budget=0),
        dict(polish_region="all"),
    ):
        with pytest.raises(ValueError):
            LSV2Config(**kwargs)
    for f, args in ((search_config, ("bad",)), (arm_config, ("bad", 1, 2))):
        with pytest.raises(ValueError):
            f(*args)


def _prepared(arm):
    problem = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    cfg = replace(arm_config(arm, RUN_SEED, 100), population=6, neighborhood=4)
    gateway = EvaluationGateway(problem, Archive())
    pop = [gateway.evaluate(g) for g in initial_genotypes(problem, cfg, Random(3))[:6]]
    search = make_search(arm, cfg, CAL, QTable(), training=True)
    search.timer = Clock()
    streams = RNGManager(RUN_SEED)
    search(pop, gateway, streams, lambda: 0.0)
    return search, pop, gateway, streams


def test_polish_runs_after_terminal_and_does_not_change_q_credit(monkeypatch):
    original = FastRightShiftPolish.propose
    search, pop, gateway, streams = _prepared(ALL)
    calls = []

    def probe(self, *args):
        assert search._pending is None
        assert search.steps[-1]["q_transition"]["terminal"]
        calls.append(copy.deepcopy(asdict(search.table)))
        return original(self, *args)

    monkeypatch.setattr(FastRightShiftPolish, "propose", probe)
    search._trajectory(pop, gateway, streams, lambda: 10.0)
    assert len(calls) == 1 and calls[0] == asdict(search.table)
    assert len(search.steps) == 3 and len(search.polish_steps) == 1
    assert all("P3" not in r["available"] for r in search.steps)
    assert all("P3" not in r["q_transition"]["next_available"] for r in search.steps)
    reference, rp, rg, rs = _prepared(NO_P3)
    reference._trajectory(rp, rg, rs, lambda: 10.0)
    assert search.steps == reference.steps
    assert asdict(search.table) == asdict(reference.table)
    for c in gateway.archive.members + pop:
        assert evaluate(gateway.problem, c.genotype, c.schedule) == c.evaluation
    assert search.summary()["polish_seconds"] > 0


def test_external_rightshift_admits_exact_and_refreshes_all_entrants():
    problem, current = synthetic()
    cfg = replace(arm_config(ALL, RUN_SEED, 100), population=2, neighborhood=2)
    search = make_search(ALL, cfg, CAL, QTable(), training=True)
    search.timer = Clock()
    gateway = EvaluationGateway(problem, Archive())
    gateway.archive.consider(current)
    pop = [current, current]
    search._polish(
        current,
        FrozenFrame.of([current.evaluation.objectives]),
        0,
        pop,
        gateway,
        RNGManager(1),
        lambda: 10,
    )
    row = search.polish_steps[-1]
    assert row["accepted"] and row["bill_gain"] > 0 and row["entered_archive"] > 0
    assert row["exact"] == gateway.counts["exact:LSV2:RIGHTSHIFT"] <= 3
    assert gateway.counts["ssgs"] == 0 and search.table.values == {}
    assert row["reflow_replaced"] > 0
    for c in gateway.archive.members:
        assert len(c.schedule.resource_wait) == problem.operation_count
        assert c.evaluation == evaluate(problem, c.genotype, c.schedule)


@pytest.mark.parametrize(
    "region,norm,expected",
    [
        ("flow_end", 0.0, False),
        ("middle", 1 / 3, False),
        ("bill_end", 2 / 3, False),
        ("bill_end", 0.1, True),
    ],
)
def test_region_uses_current_endpoint_and_existing_boundaries(region, norm, expected):
    problem, current = synthetic()
    cfg = replace(arm_config(ALL, RUN_SEED, 100), population=2, neighborhood=2)
    search = make_search(ALL, cfg, CAL, QTable(), training=True)
    search.config = replace(search.config, polish_region=region)
    search.timer = Clock()
    flow, bill = current.evaluation.objectives
    frame = (
        FrozenFrame((0, 0), (flow / norm, bill))
        if norm
        else FrozenFrame((flow, 0), (flow + 1, bill))
    )
    gateway = EvaluationGateway(problem, Archive())
    search._polish(current, frame, 0, [current] * 2, gateway, RNGManager(1), lambda: 10)
    assert (search.polish_steps[-1]["status"] == "region_skip") == expected


@pytest.mark.parametrize("reason", ["deadline_skip", "deadline_partial"])
def test_polish_respects_deadline_and_shared_allowance(reason):
    problem, current = synthetic()
    cfg = replace(arm_config(ALL, RUN_SEED, 100), population=2, neighborhood=2)
    search = make_search(ALL, cfg, CAL, QTable(), training=True)
    search.timer = Clock()
    clock = iter([10.0, 101.0]) if reason == "deadline_partial" else None
    elapsed = (
        (lambda: next(clock)) if clock else (lambda: 101.0 if reason == "deadline_skip" else 10.0)
    )
    gateway = EvaluationGateway(problem, Archive())
    search._polish(
        current,
        FrozenFrame.of([current.evaluation.objectives]),
        0,
        [current] * 2,
        gateway,
        RNGManager(1),
        elapsed,
    )
    assert search.polish_steps[-1]["status"] == reason
    assert gateway.counts["exact"] == 0


def test_scheduler_charges_polish_once_and_does_not_offer_extra_share(monkeypatch):
    cfg = replace(arm_config(ALL, RUN_SEED, 100), population=2, neighborhood=2)
    search = make_search(ALL, cfg, CAL, QTable(), training=True)
    clock = [0.0]
    search.timer = lambda: clock[0]
    search.search_start = 0

    def trajectory(*args):
        clock[0] += 0.3
        search.polish_seconds += 0.2

    monkeypatch.setattr(search, "_trajectory", trajectory)
    search([], None, RNGManager(1), lambda: 1 + clock[0])
    assert search.ls_seconds == pytest.approx(0.3)
    assert search.polish_seconds == pytest.approx(0.2)


def test_uniform_controller_can_use_same_polish_without_q():
    problem, current = synthetic()
    cfg = replace(arm_config(ALL, RUN_SEED, 100), population=2, neighborhood=2)
    search = LSV2Search(replace(search_config(ALL), selector="uniform"), cfg, cfg, timer=Clock())
    gateway = EvaluationGateway(problem, Archive())
    gateway.evaluate(current.genotype, current.schedule)
    streams = RNGManager(1)
    pop = [current] * 2
    search(pop, gateway, streams, lambda: 0)
    search._trajectory(pop, gateway, streams, lambda: 10)
    assert search.polish_steps and len(search.steps) == 3
    assert all(r["bandit"]["rule"] == "uniform" for r in search.steps)


def test_single_job_runner_seals_separate_polish_logs(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "rightshift_runner", scripts / "run_lsv2_fast_rightshift.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        input_path = tmp_path / "input.json"
        save_instance(case_problem(20, 3, 6, 202610160020, "arrival", 0), input_path)
        calibration_path = tmp_path / "cal.json"
        calibration_path.write_text(json.dumps(CAL), encoding="utf-8")
        out = tmp_path / "out"
        summary = runner.run_job(input_path, calibration_path, out, ALL, 0.3)
        seal = json.loads((out / "complete.json").read_text())
        assert "rightshift_steps.json.gz" in seal["files"]
        assert all(runner.sha(out / name) == row["sha256"] for name, row in seal["files"].items())
        assert summary["lsv2"]["config"]["polish_region"] == "all"
        with pytest.raises(FileExistsError):
            runner.run_job(input_path, calibration_path, out, ALL, 0.3)
    finally:
        sys.path.remove(str(scripts))
