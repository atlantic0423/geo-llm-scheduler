"""Independent exact, Archive, conditional-sampling and sequential-move contracts."""

from dataclasses import replace
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
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.evaluation.prefill import PrefillEvaluationContext
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.right_shift_policies import (
    PolishPostprocess,
    policy_config,
    run_policy,
)
from geo_llm_scheduler.initialization.generators import generate
from geo_llm_scheduler.operators.polish_policies import focus_prefills, policy_polish
from geo_llm_scheduler.operators.right_shift import PolishStats, right_shift_times
from geo_llm_scheduler.scheduling.resources import ResourceView
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.scheduling.timing import move
from geo_llm_scheduler.utils.numeric import TOL, identity
from geo_llm_scheduler.utils.rng import RNGManager


@pytest.fixture
def shift_problem():
    return ProblemInstance(
        tuple(Job(str(i), 0, Profile(100, 0.5, 1), Profile(100, 0.5, 1)) for i in range(2)),
        (Region("r", (Tariff(0, 500, 1), Tariff(500, 3000, 0.1)), 0),),
        tuple(ServingInstance(str(i), 0, 8, 0.1, 1.1) for i in range(2)),
    )


def parent_for(problem):
    g = Genotype((0, 0, 1, 1), (0, 1, 0, 1))
    s = Schedule((0, 1000, 200, 1300))
    return Candidate(g, s, evaluate(problem, g, s))


@pytest.mark.parametrize("n", (20, 50, 100, 200))
@pytest.mark.parametrize("pressure", ("arrival", "kv", "cheap_capacity", "demand"))
def test_reused_single_prefill_exact_matches_every_float(n, pressure):
    p = case_problem(n, 3, 12, 2026102059, pressure, 1)
    g = generate(p, 7, Random(271), 0.1)
    s = decode(p, g)
    delay = max(s.end(p, o) for o in range(p.operation_count)) + 1000
    s = replace(s, starts=tuple(t + delay if o % 2 else t for o, t in enumerate(s.starts)))
    parent = Candidate(g, s, evaluate(p, g, s))
    ctx = PrefillEvaluationContext(p, parent)
    checked = 0
    for o in range(0, p.operation_count, 2):
        times = right_shift_times(p, g, s, o, ctx.resources)
        for t in times[:2]:
            candidate = move(p, g, s, o, t, resources=ctx.resources)
            if candidate is not None:
                assert ctx.evaluate(g, candidate) == evaluate(p, g, candidate)
                checked += 1
        if checked >= 4:
            break
    assert parent.evaluation.feasible and checked > 0


def test_context_binding_invalid_parent_and_infeasible_fallback(shift_problem):
    p = shift_problem
    c = parent_for(p)
    ctx = PrefillEvaluationContext(p, c)
    with pytest.raises(ValueError, match="verified"):
        PrefillEvaluationContext(p, replace(c, evaluation=replace(c.evaluation, tou=2)))
    for schedule in [Schedule((0, 1001, 200, 1300)), Schedule((1, 1000, 201, 1300)), c.schedule]:
        with pytest.raises(ValueError, match="exactly one"):
            ctx.evaluate(c.genotype, schedule)
    with pytest.raises(ValueError, match="another genotype"):
        ctx.evaluate(replace(c.genotype), Schedule((50, 1000, 200, 1300)))
    invalid = Schedule((-1, 1000, 200, 1300))
    assert ctx.evaluate(c.genotype, invalid) == evaluate(p, c.genotype, invalid)
    invalid = Schedule((1000, 1000, 200, 1300))
    assert ctx.evaluate(c.genotype, invalid) == evaluate(p, c.genotype, invalid)


def test_reference_cached_has_same_complete_candidate_stream_and_archive(shift_problem):
    p = shift_problem
    c = parent_for(p)
    observed = []
    outputs = []
    for reuse in (False, True):
        gateway = EvaluationGateway(p, Archive())
        gateway.archive.consider(c)
        seen = []
        gateway.candidate_sink = lambda candidate: seen.append(candidate)
        stats = PolishStats()
        result = policy_polish(p, c, gateway, "reference_reuse", stats, reuse=reuse)
        observed.append([identity(x) for x in seen])
        outputs.append((identity(result), gateway.archive.identity_keys(), stats))
        assert stats.exact == gateway.counts["archive_attempts:polish"] == len(seen)
    assert observed[0] == observed[1] and outputs[0] == outputs[1]


@pytest.mark.parametrize("policy", ("two_focus", "sequential_scan"))
def test_new_policies_reference_and_reuse_match_continuous_moves(shift_problem, policy):
    p = shift_problem
    c = parent_for(p)
    results = []
    for reuse in (False, True):
        gateway = EvaluationGateway(p, Archive())
        stats = PolishStats()
        seen = []
        gateway.candidate_sink = lambda candidate: seen.append(candidate)
        result = policy_polish(p, c, gateway, policy, stats, reuse=reuse)
        assert stats.accepted == 2
        assert result.schedule.starts[0] > c.schedule.starts[0]
        assert result.schedule.starts[2] > c.schedule.starts[2]
        assert result.schedule.starts[1::2] == c.schedule.starts[1::2]
        assert result.genotype is c.genotype and result.evaluation.flow == c.evaluation.flow
        assert result.evaluation.bill < c.evaluation.bill and result.evaluation == evaluate(
            p, c.genotype, result.schedule
        )
        assert result.schedule.resource_wait and stats.exact == len(seen)
        results.append(
            (identity(result), [identity(x) for x in seen], gateway.archive.identity_keys(), stats)
        )
    assert results[0] == results[1]


def test_focus_distinct_eligible_unique_activity_and_tied_peaks(shift_problem):
    p = shift_problem
    c = parent_for(p)
    assert focus_prefills(p, c) == (0, 2)
    c = replace(c, schedule=Schedule((900, 1000, 1200, 1300)))
    c = replace(c, evaluation=evaluate(p, c.genotype, c.schedule))
    assert focus_prefills(p, c) == ()
    other = ResourceView(p, c.genotype, Schedule(c.schedule.starts))
    with pytest.raises(ValueError, match="different"):
        focus_prefills(p, c, other)
    # A higher-power Prefill hidden entirely by another operation owns no
    # exclusive energy; selecting by raw duration*power would double-count it.
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = Schedule((0, 1000, 0, 1300))
    c = Candidate(g, s, evaluate(p, g, s))
    assert focus_prefills(p, c) == (0, 2)


def test_prefill_reuse_rechecks_reference_capacity_near_tolerance(shift_problem):
    p = replace(
        shift_problem,
        jobs=tuple(
            replace(j, prefill=replace(j.prefill, compute=0.5000000006)) for j in shift_problem.jobs
        ),
    )
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = Schedule((0, 1000, 200, 1300))
    c = Candidate(g, s, evaluate(p, g, s))
    ctx = PrefillEvaluationContext(p, c)
    shifted = Schedule((200, 1000, 200, 1300))
    assert ctx.evaluate(g, shifted) == evaluate(p, g, shifted)
    assert not ctx.evaluate(g, shifted).feasible


def test_context_tail_window_tied_peaks_and_tariff_boundary(shift_problem):
    p = replace(shift_problem, regions=(replace(shift_problem.regions[0], demand_rate=9),))
    c = parent_for(p)
    ctx = PrefillEvaluationContext(p, c)
    for start in (400, 500, 800, 900):
        s = replace(c.schedule, starts=(start, *c.schedule.starts[1:]))
        fast = ctx.evaluate(c.genotype, s)
        assert fast == evaluate(p, c.genotype, s)
        assert len(fast.windows[0]) == 2


def test_conditional_sampling_separate_stream_and_calls(shift_problem):
    p = shift_problem
    c = parent_for(p)
    gateway = EvaluationGateway(p, Archive())
    streams = RNGManager(292)
    reference = RNGManager(292)
    streams.stream("variation")
    before = streams.snapshot()["variation"]
    sample = reference.stream("polish_sampling")
    counts = {"ls": 0, "no_ls": 0}
    process = PolishPostprocess("SPARSE_SCAN")
    # An already tight parent makes the test a sampling check, not thousands
    # of expensive polish calls. Both classes still create the same RNG draws.
    s = Schedule((900, 1000, 1200, 1300))
    c = Candidate(c.genotype, s, evaluate(p, c.genotype, s))
    for i in range(200):
        hit = i % 2 == 0
        group = "ls" if hit else "no_ls"
        counts[group] += int(sample.random() < (0.05 if hit else 0.10))
        process(c, hit, gateway, streams)
    for group in counts:
        assert gateway.counts["polish_calls:" + group] == counts[group]
        assert gateway.counts["polish_sampling_draws:" + group] == 100
    assert streams.snapshot()["variation"] == before
    assert streams.snapshot()["polish_sampling"] == reference.snapshot()["polish_sampling"]


def test_default_engine_and_identity_callback_same_fixed_work(problem):
    cfg = Config(population=4, neighborhood=2, generations=1, seed=104, method="plain")
    original = run(problem, cfg)
    transformed = run(problem, cfg, offspring_transform=lambda c, hit, gateway, streams: c)
    assert original.gateway.counts == transformed.gateway.counts
    assert [identity(c) for c in original.population] == [
        identity(c) for c in transformed.population
    ]
    assert original.archive.identity_keys() == transformed.archive.identity_keys()


def test_no_polish_reference_rng_and_unknown_modes(shift_problem):
    p = shift_problem
    c = parent_for(p)
    s = RNGManager(27)
    gateway = EvaluationGateway(p, Archive())
    for arm in ("NO_POLISH", "BILL_MOEAD"):
        assert PolishPostprocess(arm)(c, True, gateway, s) is c
    assert s.snapshot() == {}
    with pytest.raises(ValueError):
        PolishPostprocess("invalid")
    with pytest.raises(ValueError):
        policy_config("invalid", 1, 10)
    with pytest.raises(ValueError):
        policy_polish(p, c, gateway, "invalid")
    with pytest.raises(ValueError):
        right_shift_times(p, c.genotype, c.schedule, 1, ResourceView(p, c.genotype, c.schedule))
    with pytest.raises(ValueError):
        run_policy(p, "NO_POLISH", replace(policy_config("NO_POLISH", 1, 10), fixed_budget=3), {})
    with pytest.raises(ValueError):
        gateway.evaluate(c.genotype, prefill_context=PrefillEvaluationContext(p, c))
    assert TOL.cost > 0


@pytest.mark.parametrize("trigger", ("always", "preference", "strict"))
def test_reference_hook_preserves_original_trajectory_q_and_all_candidates(problem, trigger):
    cfg = Config(
        population=6,
        neighborhood=3,
        generations=2,
        seed=731,
        method="full",
        trigger_mode=trigger,
        rl_steps=2,
    )
    results = []
    for reuse_hook in (False, True):
        events = []
        streams = []

        def observe(event, data):
            if event == "candidate":
                events.append((identity(data["candidate"]), data["candidate"].evaluation))
            if event == "offspring":
                streams.append(data["streams"].snapshot())

        result = run(
            problem,
            replace(cfg, polish=False) if reuse_hook else cfg,
            observer=observe,
            offspring_transform=PolishPostprocess("REFERENCE_REUSE") if reuse_hook else None,
        )
        results.append(
            (
                events,
                streams,
                [identity(c) for c in result.population],
                result.archive.identity_keys(),
                result.controller.q,
                result.controller.selections,
                result.gateway.counts["trigger_successes"],
                result.gateway.counts["exact"],
                result.gateway.counts["polish_exact"],
            )
        )
    assert results[0] == results[1]
