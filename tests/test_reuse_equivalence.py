"""Differential checks against original Git blobs, not duplicated fast-path logic."""

import hashlib
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation import exact
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import generate
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.operators import active_pack, right_shift
from geo_llm_scheduler.scheduling import resources, ssgs, timing
from geo_llm_scheduler.utils.numeric import identity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.performance_reuse_20261020 import reference_runtime as reference  # noqa: E402


def test_reference_is_original_git_content():
    root = Path(reference.__file__).parent
    metadata = json.loads((root / "REFERENCE.json").read_text())
    assert metadata["commit"] == "b148c4addd207fe30a6b2a43d953a04afc35d3a1"
    for name, spec in metadata["files"].items():
        assert (
            hashlib.sha256((root / "frozen" / (name + ".py")).read_bytes()).hexdigest()
            == (spec["sha256"])
        )


@pytest.mark.parametrize("size", (20, 50, 100, 200))
@pytest.mark.parametrize("pressure", ("arrival", "kv", "cheap_capacity", "demand"))
def test_exact_decomposition_all_windows_and_diagnostics_are_identical(size, pressure):
    problem = case_problem(size, 3, 12, 20261020, pressure, 1)
    genotype = generate(problem, 7, random.Random(19), 0.1)
    schedule = ssgs.decode(problem, genotype)
    old = reference.original["exact"]
    assert exact.violations(problem, genotype, schedule) == old.violations(
        problem, genotype, schedule
    )
    assert exact.evaluate(problem, genotype, schedule) == old.evaluate(problem, genotype, schedule)
    for r in range(len(problem.regions)):
        assert exact.power_segments(problem, genotype, schedule, r) == old.power_segments(
            problem, genotype, schedule, r
        )
    assert ssgs.refresh_diagnostics(problem, genotype, schedule) == reference.original[
        "ssgs"
    ].refresh_diagnostics(problem, genotype, schedule)


def test_resource_view_partial_schedule_exclusion_and_changed_input_rejection(problem):
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((-1.0, -1.0, 50.1, 150.2))
    view = resources.ResourceView(problem, genotype, schedule)
    view.check_binding(problem, genotype, schedule)
    for machine in range(len(problem.instances)):
        expected = resources.intervals(problem, genotype, schedule, machine)
        assert view.events(machine) == tuple(sorted({t for _, a, b in expected for t in (a, b)}))
        assert view.activity_bounds(machine) == (
            tuple(sorted(a for _, a, b in expected if a < b)),
            tuple(sorted(b for _, a, b in expected if a < b)),
        )
        assert view.events(machine) is view.events(machine)
        assert view.activity_bounds(machine) is view.activity_bounds(machine)
        for excluded in (frozenset(), frozenset({2}), frozenset({0, 1, 2, 3})):
            assert view.intervals(machine, excluded) == resources.intervals(
                problem, genotype, schedule, machine, excluded
            )
    for o in (0, 1):
        for t in (0.0, 50.1, 150.2, 350.5):
            assert view.fits(o, t) == resources.fits(problem, genotype, schedule, o, t)
            assert view.earliest(o, t) == resources.earliest(problem, genotype, schedule, o, t)
    view.intervals(0).clear()
    assert view.items[0]
    for p, g, s in (
        (replace(problem), genotype, schedule),
        (problem, replace(genotype), schedule),
        (problem, genotype, replace(schedule, starts=(0.0, 1.0, 2.0, 3.0))),
    ):
        with pytest.raises(ValueError, match="different MS/timing"):
            timing.move(p, g, s, 0, 0.0, resources=view)
        with pytest.raises(ValueError, match="different MS/timing"):
            timing.critical_times(p, g, s, 0, resources=view)
        with pytest.raises(ValueError, match="different MS/timing"):
            timing.packing_moves(p, g, s, resources=view)


@pytest.mark.parametrize("mode", (0, 7, 9))
def test_packing_order_all_stages_rng_and_polish_archive_match(mode):
    problem = case_problem(20, 3, 6, 20261021, "demand", 1)
    genotype = generate(problem, mode, random.Random(81), 0.1)
    schedule = ssgs.decode(problem, genotype)
    candidate = Candidate(genotype, schedule, exact.evaluate(problem, genotype, schedule))

    def exercise():
        rng = random.Random(51)
        moves = timing.packing_moves(problem, genotype, schedule)
        selection = active_pack.select_packing_moves(problem, candidate, 6, rng)
        # Frozen module dataclasses have different Python identities. Compare fields.
        selection = (selection.raw, selection.representatives, selection.pool, selection.selected)
        state = reference.current["severity"].severities(problem, candidate)
        gateway = EvaluationGateway(problem, Archive())
        observed = []
        gateway.candidate_sink = observed.append
        stats = right_shift.PolishStats()
        best = right_shift.polish(problem, candidate, gateway, stats)
        return (
            moves,
            selection,
            state,
            best,
            observed,
            dict(vars(stats)),
            gateway.archive.members,
            gateway.archive.attempts,
            dict(gateway.counts),
            rng.getstate(),
        )

    new = exercise()
    with reference.reference_runtime():
        old = exercise()
    assert new == old


@pytest.mark.parametrize("count", (0, 1, 2, 17, 80))
def test_a7_rank_crowding_duplicate_ties_layers_and_fractions(count):
    rng = random.Random(20261020)
    moves = [(i, float(i), rng.randrange(9) / 10, rng.randrange(9) / 10) for i in range(count)]
    assert active_pack.rank_crowding(moves) == reference.original["active_pack"].rank_crowding(
        moves
    )


def test_archive_snapshot_mutations_detached_set_and_accounting(problem):
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = ssgs.decode(problem, genotype)
    candidate = Candidate(genotype, schedule, exact.evaluate(problem, genotype, schedule))
    archive = Archive()
    assert archive.identity_keys() == set()
    archive.consider(candidate)
    snapshot = archive.identity_keys()
    archive.members += [replace(candidate, origin="same phenotype")]
    assert archive.identity_keys() == {identity(c) for c in archive.members}
    archive.members.reverse()
    assert archive.identity_keys() == snapshot
    archive.members.clear()
    assert archive.identity_keys() == set() and snapshot
    assert not archive._identities
    snapshot.clear()
    assert (archive.attempts, archive.insertions) == (1, 1)


@pytest.mark.parametrize(
    "changes", ({}, {"polish": False}, {"trigger_mode": "strict", "trigger_delta": 0})
)
def test_fixed_engine_candidate_archive_q_trace_rng_identical(changes):
    problem = case_problem(12, 3, 6, 20261022, "kv", 1)
    config = (
        replace(
            Config(method="full", population=12, neighborhood=4, generations=2, seed=420),
            trigger_mode="always",
            **changes,
        )
        if "trigger_mode" not in changes
        else Config(
            method="full", population=12, neighborhood=4, generations=2, seed=420, **changes
        )
    )

    def execute():
        candidates, snapshots = [], []

        def observe(event, data):
            if event == "candidate":
                candidates.append(data["candidate"])
            elif event == "offspring":
                snapshots.append(data["streams"].snapshot())

        result = run(problem, config, observer=observe)
        return (
            result.population,
            result.archive.members,
            result.gateway.counts,
            result.controller.q,
            result.controller.updates,
            deterministic_trace(result.trace),
            candidates,
            snapshots,
        )

    new = execute()
    with reference.reference_runtime():
        old = execute()
    assert new == old


@pytest.mark.parametrize("starts", ((0.0,), (math.nan, 0, 0, 0), (-1.0, 0, 0, 0), (0, 0, 0, 0)))
def test_reference_invalid_timing_diagnostics_order(problem, starts):
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule(starts)
    assert exact.evaluate(problem, genotype, schedule) == reference.original["exact"].evaluate(
        problem, genotype, schedule
    )


@pytest.mark.parametrize("ms", ((-1, 0, 0, 0), (99, 0, 0, 0), (True, 0, 0, 0), (0,)))
def test_assignment_validation_same_error_and_order(problem, ms):
    genotype = Genotype(ms, (0, 1, 0, 1))
    errors = []
    for function in (validate_genotype, reference.original["validation"].validate_genotype):
        with pytest.raises(ValueError) as caught:
            function(problem, genotype)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]


@pytest.mark.parametrize("region_count,machines", ((2, 6), (3, 12), (5, 24)))
def test_ten_initialization_modes_paths_order_and_final_rng(region_count, machines):
    from geo_llm_scheduler.moead.variation import paths

    problem = case_problem(20, region_count, machines, 20261023, "cheap_capacity", 1)
    for job in range(len(problem.jobs)):
        for region in (None, *range(region_count)):
            assert paths(problem, job, region) == reference.original["variation"].paths(
                problem, job, region
            )
    for mode in range(10):
        rng = random.Random(510 + mode)
        new = generate(problem, mode, rng, 0.1), rng.getstate()
        with reference.reference_runtime():
            rng = random.Random(510 + mode)
            old = reference.original["generators"].generate(problem, mode, rng, 0.1), rng.getstate()
        assert new == old


@pytest.mark.parametrize("action", range(1, 9))
def test_each_action_fixed_trajectory_keeps_all_candidates_q_and_rng(action):
    problem = case_problem(12, 3, 6, 20261024, "demand", 1)
    config = Config(
        method="full",
        population=10,
        neighborhood=4,
        generations=1,
        seed=71,
        trigger_mode="always",
        enabled_operators=(action,),
        rl_steps=2,
    )

    def exercise():
        candidates, snapshots = [], []

        def observe(event, data):
            if event == "candidate":
                candidates.append(data["candidate"])
            elif event == "offspring":
                snapshots.append(data["streams"].snapshot())

        result = run(problem, config, observer=observe)
        return (
            candidates,
            snapshots,
            result.population,
            result.archive.members,
            result.gateway.counts,
            result.controller.q,
            deterministic_trace(result.trace),
        )

    new = exercise()
    with reference.reference_runtime():
        old = exercise()
    assert new == old


def test_power_hidden_boundaries_idle_tail_fractional_sums_and_tariff_edges(problem):
    from geo_llm_scheduler.domain.models import Tariff

    jobs = tuple(
        replace(
            job,
            release=0.0,
            kv_delay=0.0,
            prefill=replace(job.prefill, duration=300.1, compute=0.1),
            decode=replace(job.decode, duration=900.2, compute=0.1),
        )
        for job in problem.jobs
    )
    regions = tuple(
        replace(
            r,
            tariffs=(
                Tariff(0, 300.1, 0.1000000001),
                Tariff(300.1, 900, 0.9),
                Tariff(900, 1800, 0.0),
                Tariff(1800, 7200, 1.1000000001),
            ),
        )
        for r in problem.regions
    )
    problem = replace(problem, jobs=jobs, regions=regions)
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((0.0, 900.0, 0.1, 900.1))
    assert exact.evaluate(problem, genotype, schedule) == reference.original["exact"].evaluate(
        problem, genotype, schedule
    )
    for region in range(len(problem.regions)):
        assert exact.power_segments(problem, genotype, schedule, region) == reference.original[
            "exact"
        ].power_segments(problem, genotype, schedule, region)


@pytest.mark.parametrize("order", ([], [1, 1], [0, 0, 1, 1], [0, 1, 2, 1, 0, 2], list(range(40))))
def test_lazy_swap_pairs_every_index_slice_and_rng_match_original_list(order):
    from geo_llm_scheduler.moead.variation import _DifferentJobPairs

    old = [
        (i, j) for i in range(len(order)) for j in range(i + 1, len(order)) if order[i] != order[j]
    ]
    new = _DifferentJobPairs(order)
    assert list(new) == old and len(new) == len(old)
    assert new[::-1] == old[::-1] and new[1::2] == old[1::2]
    with pytest.raises(IndexError):
        new[len(new)]
    if old:
        assert new[-1] == old[-1]
        for seed in range(30):
            a, b = random.Random(seed), random.Random(seed)
            assert a.choice(new) == b.choice(old)
            assert a.getstate() == b.getstate()


def test_static_direction_cache_matches_reference_and_tracks_epsilon(monkeypatch):
    from geo_llm_scheduler.moead.core import weights
    from geo_llm_scheduler.rl import state

    rng = random.Random(42)
    old = reference.original["state"]
    for count in (2, 12, 24, 100):
        rays = weights(count)
        for vector in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (-1.0, 1.0)):
            assert state.associate(vector, rays) == old.associate(vector, rays)
        for _ in range(30):
            vector = rng.random(), rng.random()
            assert state.associate(vector, rays) == old.associate(vector, rays)
    monkeypatch.setattr(state, "EPS_DIRECTION", 0.1)
    monkeypatch.setattr(old, "EPS_DIRECTION", 0.1)
    assert state.associate((0.1, 0.9), weights(100)) == old.associate((0.1, 0.9), weights(100))
    for count in range(2, 25):
        state.associate((0.1, 0.9), weights(count))
    assert state._reference_rays.cache_info().currsize <= 16


def test_power_sum_overflow_retains_original_zero_term_nan_behavior(problem):
    from geo_llm_scheduler.domain.models import Tariff
    from geo_llm_scheduler.io.validation import validate_problem

    region = replace(
        problem.regions[0], tariffs=(Tariff(0, 3600, 1), Tariff(3600, 7200, 1)), demand_rate=1
    )
    problem = replace(
        problem,
        regions=(region,),
        jobs=tuple(replace(job, release=0.0, kv_delay=0.0) for job in problem.jobs),
        instances=tuple(
            replace(server, region=0, idle_kw=1e308, active_kw=1e308)
            for server in problem.instances
        ),
    )
    validate_problem(problem)
    genotype = Genotype((0, 0, 1, 1), (0, 1, 0, 1))
    schedule = ssgs.decode(problem, genotype)
    new = exact.evaluate(problem, genotype, schedule)
    old = reference.original["exact"].evaluate(problem, genotype, schedule)
    assert new.feasible == old.feasible
    assert new.flow == old.flow and new.windows == old.windows and new.demand == old.demand
    assert math.isnan(new.tou) and math.isnan(old.tou)
