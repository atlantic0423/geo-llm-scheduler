"""Paired cloning, direction routing and A7 physical-unit proxy counterexamples."""

import random
from dataclasses import asdict, replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.domain.models import (
    Candidate,
    EvaluationResult,
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.evaluation import EvaluationCapReached, EvaluationGateway
from geo_llm_scheduler.engine.offspring import create_offspring
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.p2 import P2_ARMS, p2_config
from geo_llm_scheduler.moead.core import (
    NormalizationContext,
    neighborhoods,
    replace_neighbors,
    weights,
)
from geo_llm_scheduler.moead.replacement import replacement_neighborhood
from geo_llm_scheduler.moead.variation import reproduce
from geo_llm_scheduler.operators.active_pack import ActivePack, select_packing_moves
from geo_llm_scheduler.operators.packing_proxy import PackingBillProxy
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.scheduling.timing import move, pack_gain
from geo_llm_scheduler.utils.rng import RNGManager


def delayed(problem):
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((0, 300, 200, 300))
    return Candidate(genotype, schedule, evaluate(problem, genotype, schedule))


@pytest.mark.parametrize(
    "options",
    [
        {"offspring_policy": "unknown"},
        {"replacement_policy": "unknown"},
        {"a7_representative_policy": "unknown"},
        {"clone_probability": -0.01},
        {"clone_probability": 1.01},
        {"clone_probability": float("nan")},
        {"offspring_policy": "phenotype_clone", "method": "plain"},
        {"replacement_policy": "direction", "method": "nsga2"},
        {"a7_representative_policy": "bill_proxy", "method": "nsga2_memetic"},
    ],
)
def test_invalid_experimental_config(options):
    with pytest.raises(ValueError):
        Config(**{"method": "full", **options})


def test_p2_definitions_are_single_factor_and_yaml_roundtrips(tmp_path):
    import yaml

    baseline = p2_config("F6")
    changed = {
        "F6": set(),
        "CG": {"offspring_policy"},
        "CT": {"offspring_policy"},
        "RPERM": {"replacement_policy"},
        "RDIR": {"replacement_policy"},
        "A7B": {"a7_representative_policy"},
    }
    for arm in P2_ARMS:
        config = p2_config(arm)
        assert {k for k, v in asdict(config).items() if v != asdict(baseline)[k]} == changed[arm]
        path = tmp_path / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
        assert load_config(path) == config
    for base in (Config(), replace(baseline, fixed_budget=3), p2_config("CT")):
        with pytest.raises(ValueError):
            p2_config("F6", base)
    with pytest.raises(ValueError):
        p2_config("unknown")


def test_cg_redecodes_and_ct_freezes_whole_phenotype_and_archives(problem):
    incumbent = delayed(problem)
    assert incumbent.evaluation.feasible
    assert decode(problem, incumbent.genotype).starts != incumbent.schedule.starts
    outputs = {}
    for arm in ("CG", "CT"):
        config = replace(p2_config(arm), clone_probability=1)
        gateway = EvaluationGateway(problem, Archive())
        other_schedule = replace(
            incumbent.schedule, starts=tuple(t + 50 for t in incumbent.schedule.starts)
        )
        other = replace(
            incumbent,
            schedule=other_schedule,
            evaluation=evaluate(problem, incumbent.genotype, other_schedule),
        )
        child = create_offspring(
            [other, incumbent, other], 1, (1, 0), config, gateway, RNGManager(9)
        )
        outputs[arm] = child
        assert child.genotype == incumbent.genotype
        assert child.evaluation == evaluate(problem, child.genotype, child.schedule)
        assert gateway.counts["exact"] == gateway.counts["feasible"] == 1
        assert gateway.archive.members == [child]
        assert gateway.counts["ssgs"] == (1 if arm == "CG" else 0)
    assert outputs["CT"].schedule == incumbent.schedule
    assert outputs["CT"].schedule is not incumbent.schedule
    assert outputs["CG"].schedule.starts == decode(problem, incumbent.genotype).starts
    assert incumbent == delayed(problem)


def test_clone_routing_paired_stream_and_zero_probability_variation(problem):
    incumbent = delayed(problem)
    policies = ("genotype_clone", "phenotype_clone")
    routes = []
    for policy in policies:
        streams = RNGManager(92)
        gateway = EvaluationGateway(problem, Archive())
        config = Config(method="full", offspring_policy=policy, clone_probability=0.4)
        children = [
            create_offspring([incumbent] * 3, 1, (1, 0), config, gateway, streams)
            for _ in range(30)
        ]
        routes.append([c.origin.startswith("clone_") for c in children])
    assert routes[0] == routes[1]
    assert any(routes[0]) and not all(routes[0])
    outputs = []
    for policy in ("variation", *policies):
        gateway = EvaluationGateway(problem, Archive())
        outputs.append(
            create_offspring(
                [incumbent] * 3,
                1,
                (1, 0),
                Config(method="full", offspring_policy=policy, clone_probability=0),
                gateway,
                RNGManager(9),
            )
        )
    assert outputs[0] == outputs[1] == outputs[2]


def test_disabled_clone_matches_original_variation_rng_exactly(problem):
    incumbent = delayed(problem)
    population = [incumbent] * 3
    config = Config(method="full", neighbor_probability=0)
    streams = RNGManager(19)
    original = RNGManager(19)
    rng = original.stream("variation")
    pool = (1, 0) if rng.random() < config.neighbor_probability else (0, 1, 2)
    a, b = rng.sample(pool, 2)
    expected = reproduce(problem, population[a].genotype, population[b].genotype, config, rng)
    gateway = EvaluationGateway(problem, Archive())
    child = create_offspring(population, 1, (1, 0), config, gateway, streams)
    assert child.genotype == expected
    assert streams.snapshot() == original.snapshot()


@pytest.mark.parametrize("arm", ["CG", "CT"])
def test_clones_respect_exact_cap_before_building(problem, arm):
    incumbent = delayed(problem)
    gateway = EvaluationGateway(problem, Archive(), exact_cap=1)
    gateway.evaluate(incumbent.genotype)
    before = dict(gateway.counts)
    with pytest.raises(EvaluationCapReached):
        create_offspring(
            [incumbent] * 3,
            0,
            (0, 1),
            replace(p2_config(arm), clone_probability=1),
            gateway,
            RNGManager(1),
        )
    assert dict(gateway.counts) == before


def test_direction_routing_inverse_rays_endpoints_ties_and_rng_isolation(problem):
    base = delayed(problem)
    lambdas = weights(5)
    neighbors = neighborhoods(lambdas, 2)
    context = NormalizationContext((0, 0), (10, 10))
    streams = RNGManager(7)
    for vector, target in (((0, 10), 4), ((10, 0), 0), ((5, 5), 2), ((0, 0), 0)):
        c = replace(base, evaluation=EvaluationResult(True, *vector, (0,), ((),)))
        assert (
            replacement_neighborhood("direction", 2, c, neighbors, lambdas, context, streams)
            == neighbors[target]
        )
    assert streams.snapshot() == {}
    zero = replace(base, evaluation=EvaluationResult(True, 5, 5, (0,), ((),)))
    assert (
        replacement_neighborhood(
            "direction", 3, zero, neighbors, lambdas, NormalizationContext((5, 5), (5, 5)), streams
        )
        == neighbors[0]
    )
    assert (
        replacement_neighborhood("birth", 3, zero, neighbors, lambdas, context, streams)
        == neighbors[3]
    )
    a = replacement_neighborhood("permuted", 2, base, neighbors, lambdas, context, streams)
    b = replacement_neighborhood("permuted", 2, base, neighbors, lambdas, context, RNGManager(7))
    assert a == b and set(a) == set(neighbors[2])
    assert set(streams.snapshot()) == {"P2:replacement"}
    with pytest.raises(ValueError):
        replacement_neighborhood("unknown", 2, base, neighbors, lambdas, context, streams)


def test_routing_keeps_neighbor_own_weights_and_cap(problem):
    base = delayed(problem)

    def point(flow, bill):
        return replace(base, evaluation=EvaluationResult(True, flow, bill, (0,), ((),)))

    lambdas = weights(3)
    context = NormalizationContext((0, 0), (10, 10))
    population = [point(8, 2), point(8, 8), point(2, 8)]
    child = point(1, 7)
    order = replacement_neighborhood(
        "direction", 0, child, neighborhoods(lambdas, 3), lambdas, context, RNGManager(1)
    )
    assert replace_neighbors(population, child, order, lambdas, context, 1) == 1
    assert population[2] is child
    assert population[0].evaluation.objectives == (8, 2)
    assert population[1].evaluation.objectives == (8, 8)


def test_proxy_matches_fixed_horizon_local_bill_and_avoids_overlap_double_count(problem):
    incumbent = delayed(problem)
    schedule = move(problem, incumbent.genotype, incumbent.schedule, 0, 200)
    assert schedule is not None
    proxy = PackingBillProxy(problem, incumbent)
    score = proxy.score(0, 200)
    assert score == pytest.approx(-20.5)  # -0.5 TOU and -20 demand, 100s unique area
    actual = evaluate(problem, incumbent.genotype, schedule)
    assert score == pytest.approx(actual.bill - incumbent.evaluation.bill)
    assert proxy.score(0, 0) == 0
    assert len(proxy.others) == 1
    assert incumbent == delayed(problem)


def test_proxy_tariff_units_zero_demand_and_tied_peak_average(problem):
    region = Region("r0", (Tariff(0, 100, 0.1), Tariff(100, 10000, 0.5)), 0)
    problem = replace(problem, regions=(region, problem.regions[1]))
    incumbent = delayed(problem)
    assert PackingBillProxy(problem, incumbent).score(0, 100) == pytest.approx(1)
    # Explicit tied-window unit fixture: averaging both old peaks must cancel a transfer.
    flat = replace(
        problem,
        regions=(
            replace(region, tariffs=(Tariff(0, 10000, 0),), demand_rate=2),
            problem.regions[1],
        ),
    )
    tied = replace(incumbent, evaluation=replace(incumbent.evaluation, windows=((100, 100), ())))
    assert PackingBillProxy(flat, tied).score(0, 1000) == pytest.approx(0)
    empty = replace(tied, evaluation=replace(tied.evaluation, windows=((), ())))
    assert PackingBillProxy(flat, empty).score(0, 1000) == 0


def test_proxy_new_peak_counterexample_still_requires_exact_gate():
    p = ProblemInstance(
        tuple(
            Job(str(i), 0, Profile(d, 0.25, 1), Profile(10, 0.25, 1))
            for i, d in enumerate((900, 700, 100))
        ),
        (Region("r", (Tariff(0, 10000, 0),), 1),),
        tuple(ServingInstance(str(i), 0, 8, 0, 100) for i in range(3)),
    )
    g = Genotype((0, 2, 1, 2, 0, 2), (0, 1, 2, 0, 1, 2))
    s = Schedule((0, 1800, 900, 1800, 1000, 1800))
    c = Candidate(g, s, evaluate(p, g, s))
    new = move(p, g, s, 0, 900)
    assert c.evaluation.feasible and new is not None
    assert pack_gain(p, g, s, new, 0) == 100
    assert PackingBillProxy(p, c).score(0, 900) == pytest.approx(-100)
    actual = evaluate(p, g, new)
    assert actual.feasible and actual.bill - c.evaluation.bill == pytest.approx(700 / 9)
    assert new.horizon(p) == s.horizon(p)


def test_a7_proxy_prescreen_never_calls_complete_evaluator_and_freezes_others(problem, monkeypatch):
    incumbent = delayed(problem)

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw A7 prescreen must not invoke the full evaluator")

    monkeypatch.setattr("geo_llm_scheduler.evaluation.exact.evaluate", forbidden)
    monkeypatch.setattr("geo_llm_scheduler.engine.evaluation.evaluate", forbidden)
    config = p2_config("A7B")
    audit = select_packing_moves(
        problem, incumbent, 6, random.Random(3), representative_policy="bill_proxy"
    )
    batch = ActivePack().propose(problem, incumbent, 6, config, random.Random(3))
    assert batch.proposals and len(batch.proposals) == len(audit.selected) <= 6
    assert set(audit.selected) <= set(audit.pool) <= set(audit.representatives) <= set(audit.raw)
    for proposal in batch.proposals:
        assert proposal.schedule is not None and proposal.genotype == incumbent.genotype
        assert sum(a != b for a, b in zip(proposal.schedule.starts, incumbent.schedule.starts)) == 1
        assert evaluate(problem, proposal.genotype, proposal.schedule).feasible
    with pytest.raises(ValueError):
        select_packing_moves(problem, incumbent, 6, random.Random(3), representative_policy="bad")


def test_a7_changes_first_representative_only(problem, monkeypatch):
    incumbent = delayed(problem)
    raw = [(0, 200.0, 100.0, 0.0), (0, 150.0, 50.0, 0.0), (0, 175.0, 75.0, 0.0)]
    monkeypatch.setattr("geo_llm_scheduler.operators.active_pack.packing_moves", lambda *a: raw)
    monkeypatch.setattr(
        PackingBillProxy, "score", lambda self, op, t: {200.0: 3, 150.0: -1, 175.0: 2}[t]
    )
    normal = select_packing_moves(problem, incumbent, 6, random.Random(1))
    proxy = select_packing_moves(
        problem, incumbent, 6, random.Random(1), representative_policy="bill_proxy"
    )
    assert normal.raw == proxy.raw == tuple(raw)
    assert normal.representatives == (raw[0], raw[2])
    assert proxy.representatives == (raw[1], raw[0])
    assert len(normal.pool) == len(proxy.pool) == 2
    assert len(normal.selected) == len(proxy.selected) == 2


def test_a7_real_tariff_counterexample_changes_first_representative():
    problem = ProblemInstance(
        tuple(
            Job(str(i), 0, Profile(d, 0.25, 1), Profile(50, 0.25, 1))
            for i, d in enumerate((300, 200, 100))
        ),
        (Region("r", (Tariff(0, 900, 0.1), Tariff(900, 1800, 1), Tariff(1800, 10000, 0.1)), 0),),
        tuple(ServingInstance(str(i), 0, 8, 0, 100) for i in range(2)),
    )
    genotype = Genotype((0, 1, 0, 1, 0, 1), (0, 1, 2, 0, 1, 2))
    schedule = Schedule((0, 2400, 1000, 2400, 2000, 2400))
    incumbent = Candidate(genotype, schedule, evaluate(problem, genotype, schedule))
    normal = select_packing_moves(problem, incumbent, 6, random.Random(1))
    proxy = select_packing_moves(
        problem, incumbent, 6, random.Random(1), representative_policy="bill_proxy"
    )
    first_normal = next(m for m in normal.representatives if m[0] == 0)
    first_proxy = next(m for m in proxy.representatives if m[0] == 0)
    assert normal.raw == proxy.raw
    assert first_normal[1:3] == (900, 200)
    assert first_proxy[1:3] == (1800, 100)
    normal_schedule = move(problem, genotype, schedule, 0, first_normal[1])
    proxy_schedule = move(problem, genotype, schedule, 0, first_proxy[1])
    assert normal_schedule is not None and proxy_schedule is not None
    assert evaluate(problem, genotype, normal_schedule).bill > incumbent.evaluation.bill
    assert evaluate(problem, genotype, proxy_schedule).bill < incumbent.evaluation.bill
