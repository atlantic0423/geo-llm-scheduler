"""Differential checks against frozen bf719d6, including floating-point edges."""

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.diagnostics.workload import imbalance, machine_loads, region_loads
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.comparison13 import (
    arm_config,
    case_problem,
    initialization_factory,
)
from geo_llm_scheduler.experiments.initial_bill_mixture import guided_genotype
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import generate
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators._assignment_loads import AssignmentWorkloads
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.scheduling import resources, ssgs
from geo_llm_scheduler.utils.rng import RNGManager
from research.performance_20261019 import reference_runtime as reference


def test_frozen_reference_sources_are_original_git_blobs():
    checksums = {
        "initial_bill_index.py": "887e749f69e8e1eb05596a0fc77bc9a7755e828ca648d4a4fafa5020524fc6c1",
        "pareto.py": "015b5c1214186ecdbb6dd667c5a2723fe53ba547d9d8ccd2b67509205fe46bf5",
        "resources.py": "1ba56e61fcf34005ca741dd720f4e0e5ca003bbb551bf30dd4d7b2f2166d7f69",
        "ssgs.py": "a929ea765cc281253c21e8cdc34a5cb19585078252b07984366c76da2a30b7bd",
        "structural.py": "cd5ec26f533b7be6e656a55871d8ad6552b038bb389204d372a43745a80a08d1",
        "trajectory.py": "a452f841da94934f9b6acc9721a9e34508cc408e8de56561d4c8e14eda5990ca",
    }
    root = Path(reference.__file__).parent / "frozen"
    for name, expected in checksums.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("n", (20, 50, 100, 200))
@pytest.mark.parametrize("pressure", ("arrival", "kv", "cheap_capacity", "demand"))
def test_decode_full_schedule_matches_original(n, pressure):
    problem = case_problem(n, 3, 12, 191019, pressure, 1)
    for mode in (0, 7, 9):
        genotype = generate(problem, mode, random.Random(51 + mode), 0.1)
        old = reference.ssgs.decode(problem, genotype)
        new = ssgs.decode(problem, genotype)
        assert new == old
        assert evaluate(problem, genotype, new) == evaluate(problem, genotype, old)


def test_calendar_fractional_near_capacity_and_frozen_future_events(problem):
    jobs = tuple(
        replace(
            job,
            release=i * 0.1,
            prefill=replace(job.prefill, duration=100.1, compute=0.5000000004),
            decode=replace(job.decode, duration=200.3, vram=4.0000000004),
        )
        for i, job in enumerate(problem.jobs)
    )
    problem = replace(problem, jobs=jobs)
    genotype = Genotype((0, 0, 0, 0), (1, 0, 1, 0))
    assert ssgs.decode(problem, genotype) == reference.ssgs.decode(problem, genotype)
    schedule = Schedule((-1.0, -1.0, 50.1, 150.2))
    for operation in (0, 1):
        for est in (0.0, 50.1, 150.2, 350.5):
            assert resources.earliest(problem, genotype, schedule, operation, est) == (
                reference.resources.earliest(problem, genotype, schedule, operation, est)
            )
            for excluded in (frozenset(), frozenset({2, 3})):
                assert resources.fits(problem, genotype, schedule, operation, est, excluded) == (
                    reference.resources.fits(problem, genotype, schedule, operation, est, excluded)
                )
    impossible = replace(jobs[0], prefill=replace(jobs[0].prefill, compute=1.1))
    with pytest.raises(ValueError, match="cannot fit"):
        resources.ResourceCalendar(replace(problem, jobs=(impossible,))).earliest(0, 0, 0.0)


@pytest.mark.parametrize("region_count,machines", ((2, 12), (3, 6), (3, 24), (5, 12)))
def test_assignment_scores_order_candidates_and_rng_exact(region_count, machines):
    problem = case_problem(12, region_count, machines, 191020, "kv", 1)
    # Nonintegral sums exercise the requirement to preserve addition order.
    problem = replace(
        problem,
        jobs=tuple(
            replace(
                job,
                prefill=replace(job.prefill, duration=job.prefill.duration + 0.1),
                decode=replace(job.decode, duration=job.decode.duration + 0.3),
            )
            for job in problem.jobs
        ),
    )
    genotype = generate(problem, 7, random.Random(83), 0.1)
    schedule = ssgs.decode(problem, genotype)
    candidate = Candidate(genotype, schedule, evaluate(problem, genotype, schedule))
    indexed = AssignmentWorkloads(problem, genotype)
    for job in range(len(problem.jobs)):
        for prefill, decode in paths(problem, job):
            ms = list(genotype.ms)
            ms[2 * job : 2 * job + 2] = prefill, decode
            changed = Genotype(tuple(ms), genotype.os)
            loads = machine_loads(problem, changed)
            assert indexed.path_load(job, prefill, decode) == max(loads[prefill], loads[decode])
            assert indexed.region_imbalance(job, problem.instances[prefill].region) == imbalance(
                region_loads(problem, changed)
            )
    for action in (1, 2, 3):
        for policy in ("load", "tariff"):
            config = Config(method="full", a3_region_policy=policy)
            new_rng, old_rng = random.Random(84), random.Random(84)
            new = StructuralOperator(action).propose(problem, candidate, 6, config, new_rng)
            old = reference.structural.StructuralOperator(action).propose(
                problem, candidate, 6, config, old_rng
            )
            assert new == old
            assert new_rng.getstate() == old_rng.getstate()


def test_archive_live_identity_cache_preserves_ties_mutation_and_counters():
    rng = random.Random(191019)
    new, old = Archive(), reference.pareto.Archive()
    for i in range(160):
        flow = float(rng.randrange(1, 16)) + (i % 3) * 4e-10
        bill = 16.0 - int(flow) + (i % 5) * 4e-10
        candidate = Candidate(
            Genotype((0, 0), (0, 0)),
            Schedule((float(i % 31) * 1e-9, 20.0)),
            EvaluationResult(i % 17 != 0, flow, bill, (), ()),
            str(i),
        )
        if i == 45:
            new.members = list(reversed(new.members))
            old.members = list(reversed(old.members))
        if i == 90:
            new.members.clear()
            old.members.clear()
        assert new.consider(candidate) == old.consider(candidate)
        assert new.members == old.members
        assert (new.attempts, new.insertions, new.peak_size) == (
            old.attempts,
            old.insertions,
            old.peak_size,
        )
        assert len(new._identities) <= len(new.members)


@pytest.mark.parametrize("arm", ("M11_I4", "MOEAD"))
def test_charged_initializer_all_candidates_hashes_rng_and_original_exact(arm):
    problem = case_problem(12, 3, 6, 191019, "arrival", 1)
    config = replace(arm_config(arm, 71, 300), seconds=None, generations=1)

    def initial(use_reference):
        from geo_llm_scheduler.engine.evaluation import EvaluationGateway

        evidence = {}
        manager = RNGManager(config.seed)
        gateway = EvaluationGateway(problem, Archive())
        calls = []
        gateway.candidate_sink = calls.append
        initializer = initialization_factory(problem, config, 4, 1191019, evidence)
        if use_reference:
            with reference.reference_runtime():
                population = initializer(gateway, manager)
        else:
            population = initializer(gateway, manager)
        return population, gateway, calls, manager.snapshot(), evidence

    a, b = initial(False), initial(True)
    assert a[0] == b[0] and a[2] == b[2] and a[3] == b[3]
    assert a[1].counts == b[1].counts
    assert a[1].archive.members == b[1].archive.members
    for key in ("initial_population_sha256", "initial_archive_sha256"):
        assert a[4][key] == b[4][key]
    for candidate in a[2]:
        assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation


@pytest.mark.parametrize("seed", (17, 71))
def test_full_engine_candidate_stream_trace_q_and_final_rng_match(seed):
    problem = case_problem(12, 3, 6, 191021, "kv", 1)
    config = replace(
        arm_config("M11_I4", seed, 300),
        seconds=None,
        generations=2,
        population=12,
        neighborhood=4,
        trigger_mode="always",
    )

    def execute(use_reference):
        calls, snapshots = [], []

        def observe(event, data):
            if event == "candidate":
                calls.append(data["candidate"])
            elif event == "offspring":
                snapshots.append(data["streams"].snapshot())

        if use_reference:
            with reference.reference_runtime():
                result = run(problem, config, observer=observe)
        else:
            result = run(problem, config, observer=observe)
        return result, calls, snapshots

    (a, ac, ar), (b, bc, br) = execute(False), execute(True)
    assert ac == bc and ar == br
    assert a.population == b.population and a.archive.members == b.archive.members
    assert a.gateway.counts == b.gateway.counts
    assert a.controller.q == b.controller.q and a.controller.updates == b.controller.updates
    assert deterministic_trace(a.trace) == deterministic_trace(b.trace)


def test_guided_output_and_rng_unchanged_against_old_resource_bindings():
    problem = case_problem(20, 3, 12, 191019, "arrival", 1)
    a, b = random.Random(22), random.Random(22)
    current = guided_genotype(problem, "BILL", a, 0.1, accelerated=True)
    with reference.reference_runtime():
        original = guided_genotype(problem, "BILL", b, 0.1, accelerated=True)
    assert current == original and a.getstate() == b.getstate()


def test_rejected_trajectory_reuses_pure_state_diagnostics(problem, monkeypatch):
    from geo_llm_scheduler.engine import trajectory
    from geo_llm_scheduler.engine.evaluation import EvaluationGateway
    from geo_llm_scheduler.rl.controller import Controller

    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    gateway = EvaluationGateway(problem, Archive())
    candidate = gateway.evaluate(genotype)
    config = Config(method="full", enabled_operators=(4,), polish=False)
    observed = []
    original = trajectory.severities

    def counted(p, current):
        observed.append(current)
        return original(p, current)

    monkeypatch.setattr(trajectory, "severities", counted)
    result, steps = trajectory.improve(
        candidate,
        0,
        0,
        0.0,
        ((0.5, 0.5),),
        candidate.evaluation.objectives,
        gateway,
        Controller(config),
        config,
        RNGManager(71),
        [candidate],
    )
    assert result is candidate and not any(step["accepted"] for step in steps)
    assert len(steps) == config.rl_steps and observed == [candidate]
