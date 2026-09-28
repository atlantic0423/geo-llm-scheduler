"""E15 policy, paired-instance, fairness and NSGA-II transfer counterexamples."""

import random
from dataclasses import replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.engine.trigger import triggered
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.nsga2 import environmental_selection, run_nsga2
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.macrosearch.adaptive import (
    AdaptiveBudget,
    action_severity,
    empirical_percentile,
    gap_budgets,
)
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.utils.rng import RNGManager


def _candidate(flow: float, bill: float, index: int = 0) -> Candidate:
    return Candidate(
        Genotype((index, 0), (0, 0)),
        Schedule((0.0, 1.0)),
        EvaluationResult(True, flow, bill, (), ()),
    )


def test_tariff_pair_changes_only_prices_and_validates():
    h, t = generate_e15_pair(3, 1001)
    assert h.jobs == t.jobs and h.instances == t.instances
    assert [r.name for r in h.regions] == [r.name for r in t.regions]
    assert any(a.tariffs != b.tariffs for a, b in zip(h.regions, t.regions))
    assert any(a.demand_rate != b.demand_rate for a, b in zip(h.regions, t.regions))


def test_severity_v2_action_history_warmup_and_bins():
    config = Config(population=10, neighborhood=3, budget_policy="severity_v2")
    controller = AdaptiveBudget()
    population = [_candidate(10, 10)]
    values = (0.01, 0, 0, 0, 0, 0)
    context = NormalizationContext((0, 0), (100, 100))
    for _ in range(5):
        budget, details = controller.choose(
            1, values, 0, 0, population, [], weights(10), context, config
        )
        assert budget == 6 and details["history_count"] < 5
    low, low_details = controller.choose(
        1, (0, 0, 0, 0, 0, 0), 0, 0, population, [], weights(10), context, config
    )
    high, high_details = controller.choose(
        1, (10, 0, 0, 0, 0, 0), 0, 0, population, [], weights(10), context, config
    )
    assert low == 3 and low_details["action_percentile"] < 0.2
    assert high == 10 and high_details["action_percentile"] >= 0.8
    assert len(controller.histories[2]) == 0
    assert empirical_percentile([1, 1, 1], 1) == 0.5
    assert action_severity(7, (0, 0, 0, 0, 0, 0.16), 0, 0, config) == 1


def test_coverage_v2_gap_bins_uniform_gate_and_cache():
    assert gap_budgets((0.1,) * 8, (3, 6, 10))[0] == (6,) * 8
    bins = gap_budgets((0, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4), (3, 6, 10))[0]
    assert set(bins) == {3, 6, 10}
    config = Config(population=10, neighborhood=3, budget_policy="coverage_v2")
    controller = AdaptiveBudget()
    population = [_candidate(1, 5), _candidate(5, 1)]
    context = NormalizationContext((0, 0), (10, 10))
    args = (1, (0,) * 6, 0, 2, population, [], weights(10), context, config)
    controller.choose(*args)
    controller.choose(*args)
    assert controller.counts["coverage_recompute_count"] == 1
    assert controller.counts["coverage_cache_hit"] == 1
    controller.choose(
        1, (0,) * 6, 0, 2, population + [_candidate(2, 3)], [], weights(10), context, config
    )
    assert controller.counts["coverage_recompute_count"] == 2


def test_sequential_continuation_and_exact_archive_dispatch():
    class Evaluator:
        ideal = (0.0, 0.0)

        def __init__(self, scores):
            self.scores = scores
            self.evaluated = []

        def evaluate(self, genotype, schedule=None, origin="structural"):
            index = genotype.ms[0]
            self.evaluated.append(index)
            return _candidate(self.scores[index], self.scores[index], index)

    incumbent = _candidate(20, 20)
    proposals = ProposalBatch([Proposal(Genotype((i, 0), (0, 0))) for i in range(10)], 10)
    for scores, expected in (
        ([30] * 10, 3),
        ([10] + [30] * 9, 6),
        ([10, 30, 30, 5] + [30] * 6, 10),
    ):
        evaluator = Evaluator(scores)
        result = execute(
            proposals,
            incumbent,
            10,
            (0.5, 0.5),
            NormalizationContext((0, 0), (100, 100)),
            evaluator,
            "test",
            sequential=True,
        )
        assert result.effective == expected
        assert evaluator.evaluated == list(range(expected))
        assert result.sequential["final_B_eff"] == expected
    evaluator = Evaluator([10] + [30] * 9)
    fixed = execute(
        proposals,
        incumbent,
        6,
        (0.5, 0.5),
        NormalizationContext((0, 0), (100, 100)),
        evaluator,
        "test",
    )
    assert fixed.effective == 6 and evaluator.evaluated == list(range(6))


def test_equal_exact_cap_and_nsga2_memetic_survival(problem):
    base = Config(
        population=10, neighborhood=3, generations=3, exact_evaluation_cap=15, polish=False
    )
    initial = initial_genotypes(problem, base, RNGManager(7).stream("initialization"))
    plain = run(problem, base, initial)
    assert plain.gateway.counts["exact"] == 15
    assert plain.termination_reason == "exact_evaluation_cap"
    assert len(plain.trace) == 5  # stopped inside the first generation
    nsga = run_nsga2(problem, replace(base, method="nsga2"), initial)
    assert nsga.gateway.counts["exact"] == 15
    assert nsga.termination_reason == "exact_evaluation_cap"
    memetic = run_nsga2(
        problem,
        replace(base, method="nsga2_memetic", trigger_mode="always", budget_policy="sequential"),
        initial,
    )
    assert memetic.gateway.counts["exact"] == 15
    assert memetic.termination_reason == "exact_evaluation_cap"
    assert len(memetic.population) == 10
    assert environmental_selection(nsga.population, 5) == environmental_selection(
        nsga.population, 5
    )


def test_auxiliary_parent_direction_does_not_make_trigger_tautological():
    config = Config(
        population=10, neighborhood=3, trigger_mode="strict", trigger_quality_gate=False
    )
    context = NormalizationContext((0, 0), (10, 10))
    parent = _candidate(1, 9)
    child = _candidate(9, 1)
    index = associate(context.normalize(parent), weights(10))
    assert associate(context.normalize(child), weights(10)) != index
    assert not triggered(child, parent, index, weights(10), context, config, random.Random(1))


def test_memetic_nsga2_keeps_rank_crowding_survival(problem, monkeypatch):
    import geo_llm_scheduler.experiments.nsga2 as nsga_module

    original = nsga_module.environmental_selection
    selected = []

    def audit(population, size):
        result = original(population, size)
        selected.append(result)
        return result

    monkeypatch.setattr(nsga_module, "environmental_selection", audit)
    config = Config(
        population=10,
        neighborhood=3,
        generations=1,
        method="nsga2_memetic",
        trigger_mode="always",
        rl_steps=1,
        fixed_budget=3,
        polish=False,
    )
    initial = initial_genotypes(problem, config, RNGManager(9).stream("initialization"))
    result = run_nsga2(problem, config, initial)
    assert selected and result.population == selected[-1]
    assert result.gateway.counts["trigger_hits"] > 0


def test_equal_eval_progress_uses_consumed_cap(problem):
    config = Config(
        population=10,
        neighborhood=3,
        generations=1_000_000,
        exact_evaluation_cap=30,
        method="full",
        trigger_mode="always",
        rl_steps=1,
        fixed_budget=3,
        polish=False,
    )
    initial = initial_genotypes(problem, config, RNGManager(12).stream("initialization"))
    result = run(problem, config, initial)
    steps = [step for row in result.trace for step in row["steps"]]
    assert steps and 0.3 < steps[0]["progress"] < 0.6
    assert result.gateway.counts["exact"] == 30
