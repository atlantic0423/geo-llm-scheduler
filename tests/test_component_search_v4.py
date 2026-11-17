"""Thirty-slot coverage, frozen-default equivalence and exact feasibility checks."""

from dataclasses import asdict, replace
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, Region, ServingInstance, Tariff
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.offspring import create_offspring
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.component_search_v4 import (
    NUMERIC,
    SLOTS,
    Components,
    initializer,
    offspring_factory,
    run_components,
)
from geo_llm_scheduler.moead.component_selection import (
    TerminalSelector,
    crowding,
    genotype_distance,
    mutate_cross_region,
    order_replacements,
    select_parents,
    select_pool,
    tariff_proxy,
)
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.utils.rng import RNGManager


def panel():
    base = synthetic(6)[0]
    return replace(
        base,
        regions=(
            Region("a", (Tariff(0, 5000, 1),), 0.1),
            Region("b", (Tariff(0, 5000, 2),), 0.1),
            Region("c", (Tariff(0, 5000, 3),), 0.1),
            Region("d", (Tariff(0, 5000, 4),), 0.1),
            Region("e", (Tariff(0, 5000, 5),), 0.1),
        ),
        instances=tuple(ServingInstance(str(i), i // 2, 2, 0.1, 1) for i in range(10)),
    )


def members():
    problem = panel()
    gateway = EvaluationGateway(problem, Archive())
    pop = initializer(
        problem,
        Config(population=10, neighborhood=5),
        Components(initialization="legal_random"),
        {},
    )(gateway, RNGManager(9))
    return problem, gateway, pop


@pytest.mark.parametrize("slot,option", [(s, o) for s, options in SLOTS.items() for o in options])
def test_all_thirty_candidates_execute_in_engine(slot, option):
    evidence = {}
    observed = []
    result = run_components(
        panel(),
        replace(Components(), **{slot: option}),
        19,
        60,
        CAL,
        evidence,
        generations=1,
        observer=lambda event, data: observed.append(event),
    )
    assert len(result.population) == 100 and result.offspring_count == 100
    assert "structure" in observed
    assert evidence["components"][slot] == option
    assert evidence["terminal_budget"] == 3 and evidence["terminal_share"] == 0.05
    for candidate in result.population + result.archive.members:
        exact = evaluate(panel(), candidate.genotype, candidate.schedule)
        assert exact.feasible and exact.objectives == candidate.evaluation.objectives


def test_default_offspring_random_stream_and_outputs_identical():
    problem, gateway, pop = members()
    config = Config(population=10, neighborhood=5)
    old_streams, new_streams = RNGManager(77), RNGManager(77)
    other = EvaluationGateway(problem, Archive())
    other.ideal = gateway.ideal
    for _ in range(100):
        old = create_offspring(pop, 0, tuple(range(5)), config, gateway, old_streams)
        new = offspring_factory(Components())(pop, 0, tuple(range(5)), config, other, new_streams)
        assert old.genotype == new.genotype and old.evaluation == new.evaluation
        assert (
            old_streams.stream("variation").getstate() == new_streams.stream("variation").getstate()
        )


@pytest.mark.parametrize("slot", list(SLOTS) + list(NUMERIC))
def test_invalid_values_rejected(slot):
    with pytest.raises(ValueError):
        Components(**{slot: "invalid"})


def test_counts_numeric_domains_and_frozen_mechanisms():
    assert sum(map(len, SLOTS.values())) == 30
    assert len(SLOTS) == len(NUMERIC) == 8
    for name, options in NUMERIC.items():
        for value in options:
            spec = replace(Components(), **{name: value})
            config = spec.engine_config(7, 1200)
            assert config.population == 100
            if not name.startswith("ls_"):
                assert getattr(config, name) == value
    with pytest.raises(ValueError):
        Components(ls_depth=3.0)
    with pytest.raises(ValueError):
        Components(replacement_cap=True)
    with pytest.raises(TypeError):
        Components(unknown=True)


def test_parent_semantics_and_distance():
    _, gateway, pop = members()
    ctx = NormalizationContext(gateway.ideal, maximum(pop))
    pool = tuple(range(10))
    for rule in SLOTS["parents"]:
        for seed in range(15):
            a, b = select_parents(pop, 4, pool, rule, ctx, (0.3, 0.7), Random(seed))
            assert a != b and a in pool and b in pool
            if rule == "incumbent_partner":
                assert a == 4
    assert genotype_distance(pop[0].genotype, pop[0].genotype) == 0
    assert genotype_distance(Genotype((), ()), Genotype((), ())) == 0
    for a in pop:
        for b in pop:
            assert 0 <= genotype_distance(a.genotype, b.genotype) <= 1
            assert genotype_distance(a.genotype, b.genotype) == genotype_distance(
                b.genotype, a.genotype
            )
    with pytest.raises(ValueError):
        select_parents(pop, 0, (0,), "random_pair", ctx, (0.5, 0.5), Random(1))


@pytest.mark.parametrize("rule", SLOTS["cross_region_mutation"])
def test_cross_region_legal_exact_and_bounded_shortlists(rule):
    problem, _, pop = members()
    for seed in range(30):
        g = pop[0].genotype
        target = mutate_cross_region(problem, g, rule, Random(seed))
        assert target == mutate_cross_region(problem, g, rule, Random(seed))
        changed = [j for j in range(6) if target.ms[2 * j : 2 * j + 2] != g.ms[2 * j : 2 * j + 2]]
        assert len(changed) == 1
        j = changed[0]
        assert problem.instances[target.ms[2 * j]].region != problem.instances[g.ms[2 * j]].region
        assert EvaluationGateway(problem, Archive()).evaluate(target).evaluation.feasible
        if rule == "tou_proxy":
            from geo_llm_scheduler.moead.variation import paths

            legal = [
                p
                for p in paths(problem, j)
                if problem.instances[p[0]].region != problem.instances[g.ms[2 * j]].region
            ]
            best = sorted(legal, key=lambda p: (tariff_proxy(problem, j, p), p))[:3]
            assert target.ms[2 * j : 2 * j + 2] in best
    one, c = synthetic(2)
    assert mutate_cross_region(one, c.genotype, rule, Random(1)) == c.genotype


def test_replacement_scope_and_order():
    _, gateway, pop = members()
    ctx = NormalizationContext(gateway.ideal, maximum(pop))
    w = weights(10)
    neighbors = (4, 3, 5, 2)
    for rule in SLOTS["replacement_order"]:
        ordered = order_replacements(pop, pop[0], neighbors, w, ctx, rule, Random(2))
        assert set(ordered) == set(neighbors)
        if rule == "fixed":
            assert ordered == neighbors
        if rule == "improvement_first":
            scores = [ctx.scalar(pop[j], w[j]) - ctx.scalar(pop[0], w[j]) for j in ordered]
            assert scores == sorted(scores, reverse=True)
    with pytest.raises(ValueError):
        order_replacements(pop, pop[0], neighbors, w, ctx, "random")
    with pytest.raises(ValueError):
        order_replacements(pop, pop[0], neighbors, w, ctx, "bad")


def test_pool_unique_dominance_and_crowding():
    _, _, pop = members()
    selected = select_pool(pop + pop, 5)
    assert len(selected) == len({c.genotype for c in selected}) == 5
    assert selected == select_pool(pop + pop, 5)
    with pytest.raises(ValueError):
        select_pool([pop[0]], 2)
    assert crowding([]) == [] and crowding([(1, 1)]) == [0]
    assert crowding([(0, 2), (1, 1), (2, 0)]) == [float("inf"), 2, float("inf")]


def test_proxy_power_kv_and_tariff_boundaries():
    from geo_llm_scheduler.domain.models import PowerModel

    problem = panel()
    job = replace(problem.jobs[0], kv_delay=5)
    problem = replace(problem, jobs=(job, *problem.jobs[1:]), power_model=PowerModel(0.4, 1, 0.25))
    expected = sum(
        (0.1 + problem.floor_kw(m) + problem.load_kw(o, m)) * 10 / 3600
        for o, m in enumerate((0, 1))
    )
    assert tariff_proxy(problem, 0, (0, 1)) == pytest.approx(expected)
    tariffs = (Tariff(0, 12, 1), Tariff(12, 5000, 4))
    problem = replace(
        problem, regions=(replace(problem.regions[0], tariffs=tariffs), *problem.regions[1:])
    )
    same = tariff_proxy(problem, 0, (0, 0))
    assert tariff_proxy(problem, 0, (0, 1)) > same  # KV delays Decode into expensive interval.
    problem = replace(
        problem,
        regions=(replace(problem.regions[0], tariffs=(Tariff(0, 15, 1),)), *problem.regions[1:]),
    )
    with pytest.raises(ValueError, match="coverage"):
        tariff_proxy(problem, 0, (0, 1))


def test_terminal_orders_and_persistent_alternation():
    from golden.test_moead import candidate

    todo = [candidate(1, 9, 0), candidate(3, 7, 1), candidate(5, 5, 2), candidate(7, 3, 3)]
    front = [c.evaluation.objectives for c in todo]
    assert TerminalSelector("flow_ascending")(todo, front) is todo[0]
    assert TerminalSelector("bill_ascending")(todo, front) is todo[-1]
    assert TerminalSelector("crowding_first")(todo, front) is todo[0]
    alternating = TerminalSelector("alternate_extremes")
    assert alternating(todo, front) is todo[0]
    assert alternating(todo[1:], front) is todo[-1]
    assert alternating(todo[1:-1], front) is todo[1]
    with pytest.raises(ValueError):
        TerminalSelector("bad")
    with pytest.raises(ValueError):
        alternating([], [])


@pytest.mark.parametrize("seed", range(4))
def test_mixed_configurations_exact(seed):
    rng = Random(seed)
    spec = Components(**{k: rng.choice(v) for k, v in {**SLOTS, **NUMERIC}.items()})
    searches = []
    result = run_components(
        panel(), spec, seed, 60, CAL, {}, generations=1, search_sink=searches.append
    )
    assert result.offspring_count == 100
    search = searches[0]
    assert search.config.budget == spec.ls_width
    assert search.config.trajectory_steps == spec.ls_depth
    assert search.terminal_proposal_budget == search.config.polish_budget == 3
    assert search.moead.terminal_ops == ("A7", "A8")
    assert all(c.evaluation.feasible for c in result.archive.members)


def test_pool_evaluates_all_two_populations_and_archives():
    problem = panel()
    gateway = EvaluationGateway(problem, Archive())
    seen = []
    gateway.candidate_sink = seen.append
    evidence = {}
    result = initializer(
        problem,
        Config(),
        Components(initialization="quality_diversity_pool"),
        evidence,
    )(gateway, RNGManager(8))
    assert len(result) == 100 and evidence["pool_size"] == 200
    assert len(seen) >= 200
    assert all(c.evaluation.feasible for c in seen)
    assert asdict(Components())["initialization"] == "current_bill4"


def test_default_complete_run_matches_adopted_algorithm(monkeypatch):
    import sys

    from test_throughput_equivalence import FakeClock

    from geo_llm_scheduler.experiments import s4_moead

    results = []
    for legacy in (True, False):
        clock = FakeClock(0.002)
        for name, module in tuple(sys.modules.items()):
            if name.startswith("geo_llm_scheduler.") and callable(
                getattr(module, "perf_counter", None)
            ):
                monkeypatch.setattr(module, "perf_counter", clock)
        searches = []

        def sink(search):
            search.timer = clock
            searches.append(search)

        if legacy:
            config = replace(s4_moead.arm_config("GS4_GUIDED_RAND_SWITCH_D4", 5, 50), generations=3)
            result = s4_moead.run_arm(
                panel(),
                "GS4_GUIDED_RAND_SWITCH_D4",
                config,
                CAL,
                {},
                search_sink=sink,
                guided_timer=clock,
            )
        else:
            result = run_components(
                panel(),
                Components(),
                5,
                50,
                CAL,
                {},
                generations=3,
                search_sink=sink,
                guided_timer=clock,
            )
        results.append(
            (
                [(c.genotype, c.schedule, c.evaluation) for c in result.population],
                [(c.genotype, c.schedule, c.evaluation) for c in result.archive.members],
                dict(result.gateway.counts),
                searches[0].steps,
                searches[0].moead_counts,
            )
        )
    assert results[0] == results[1]
    assert results[0][3], "Must exercise real local-search steps"


@pytest.mark.parametrize("width", [1, 3, 6])
def test_terminal_proposal_and_exact_caps_are_independent_of_ls_width(width, monkeypatch):
    import geo_llm_scheduler.lsv2.moead_search as module
    from geo_llm_scheduler.experiments.s4_moead import make_search
    from geo_llm_scheduler.operators.base import Proposal, ProposalBatch

    problem, gateway, pop = members()
    search = make_search("GS4_GUIDED_RAND_SWITCH_D4", Components().engine_config(1, 60), CAL)
    search.config = replace(search.config, budget=width)
    search.terminal_proposal_budget = 3
    seen = []
    original = search._propose
    operator_budgets = []

    class TimingProbe:
        def propose(self, problem, current, budget, config, rng):
            operator_budgets.append(budget)
            return ProposalBatch([Proposal(current.genotype, current.schedule)])

    search._timing = {"A7": TimingProbe(), "A8": TimingProbe()}

    def propose(name, current, frame, rng, problem, proposal_budget=None):
        seen.append((name, proposal_budget, current.genotype))
        return original(name, current, frame, rng, problem, proposal_budget=proposal_budget)

    caps = []
    original_execute = module.execute

    def execute(batch, current, budget, *args):
        caps.append(budget)
        return original_execute(batch, current, budget, *args)

    monkeypatch.setattr(search, "_propose", propose)
    monkeypatch.setattr(module, "execute", execute)
    search._terminal_extra(
        pop[0], [c.evaluation.objectives for c in pop], pop, gateway, RNGManager(2), lambda: 1
    )
    assert [row[:2] for row in seen] == [("A7", 3), ("A8", 3)]
    assert all(row[2] == pop[0].genotype for row in seen)
    assert caps and all(cap == 3 for cap in caps)
    assert operator_budgets == [3, 3]
    assert search.config.budget == width


@pytest.mark.parametrize("rule", SLOTS["terminal_target_order"])
def test_terminal_sweep_calls_all_three_refinements_in_selected_order(rule, monkeypatch):
    from golden.test_moead import candidate

    from geo_llm_scheduler.experiments.s4_moead import make_search
    from geo_llm_scheduler.lsv2.polish_selection import physical_key

    problem, gateway, pop = members()
    targets = [candidate(1, 9, 0), candidate(3, 7, 1), candidate(5, 5, 2), candidate(7, 3, 3)]
    gateway.archive.members[:] = targets
    search = make_search("GS4_GUIDED_RAND_SWITCH_D4", Components().engine_config(1, 60), CAL)
    search.terminal_selector = TerminalSelector(rule)
    search.retain_steps = False
    seen = []
    monkeypatch.setattr(search, "_polish", lambda target, *a: seen.append(("shift", target)))
    monkeypatch.setattr(
        search, "_terminal_extra", lambda target, *a: seen.append(("A7/A8", target))
    )
    search._terminal_sweep(pop, gateway, RNGManager(2), lambda: 0)
    expected = {
        "flow_ascending": [0, 1, 2, 3],
        "bill_ascending": [3, 2, 1, 0],
        "alternate_extremes": [0, 3, 1, 2],
        "crowding_first": [0, 3, 1, 2],
    }[rule]
    assert seen == [(stage, targets[i]) for i in expected for stage in ("shift", "A7/A8")]
    assert search._terminal_done == {physical_key(c) for c in targets}


def test_cli_space_and_single_run(tmp_path, monkeypatch):
    import importlib
    import json
    from pathlib import Path

    from geo_llm_scheduler.io.loaders import save_instance

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    cli = importlib.import_module("run_component_search_v4")
    space = tmp_path / "space.json"
    cli.main(["--export-space", str(space)])
    assert sum(map(len, json.loads(space.read_text())["slots"].values())) == 30
    instance = tmp_path / "instance.json"
    save_instance(panel(), instance)
    calibration = tmp_path / "cal.json"
    calibration.write_text(json.dumps(CAL))
    config = tmp_path / "configuration.json"
    config.write_text(json.dumps(asdict(Components())))
    output = tmp_path / "result.json"
    args = [
        "--instance",
        str(instance),
        "--calibration",
        str(calibration),
        "--configuration",
        str(config),
        "--output",
        str(output),
        "--seconds",
        "60",
        "--generations",
        "1",
    ]
    cli.main(args)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["offspring_count"] == 100 and len(result["population"]) == 100
    with pytest.raises(SystemExit):
        cli.main(args)
    with pytest.raises(SystemExit):
        cli.main([])
