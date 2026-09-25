"""Counterexamples and replay checks for the standard NSGA-II baseline."""

import random
from dataclasses import replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.experiments.nsga2 import (
    crowding_distance,
    environmental_selection,
    nondominated_fronts,
    rank_and_crowding,
    run_nsga2,
    tournament,
)
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager


def candidate(flow: float, bill: float, index: int) -> Candidate:
    return Candidate(
        Genotype((0, 0), (0, 0)),
        Schedule((float(index), float(index + 1))),
        EvaluationResult(True, flow, bill, (), ()),
    )


def test_fast_sort_and_crowding_counterexample():
    population = [
        candidate(1, 5, 0),
        candidate(2, 4, 1),
        candidate(3, 3, 2),
        candidate(4, 2, 3),
        candidate(5, 1, 4),
        candidate(4, 5, 5),
    ]
    assert nondominated_fronts(population) == [[0, 1, 2, 3, 4], [5]]
    distances = crowding_distance(population, [0, 1, 2, 3, 4])
    assert distances[0] == distances[4] == float("inf")
    assert distances[2] == 1.0
    rank, crowding = rank_and_crowding(population)
    assert rank == [0, 0, 0, 0, 0, 1]
    assert crowding[5] == float("inf")
    assert environmental_selection(population, 3) == [population[0], population[4], population[1]]


def test_tournament_uses_rank_then_crowding():
    assert tournament([1, 0], [float("inf"), 0], random.Random(9)) == 1
    assert tournament([0, 0], [0, 2], random.Random(9)) == 1


def test_nsga2_replay_with_frozen_initial_population():
    problem = load_instance("examples/smoke.json")
    config = Config(population=10, neighborhood=3, generations=2, method="nsga2", seed=19)
    initial = initial_genotypes(problem, config, RNGManager(19).stream("initialization"))
    first = run_nsga2(problem, config, initial)
    second = run_nsga2(problem, replace(config), initial)
    assert [c.evaluation.objectives for c in first.population] == [
        c.evaluation.objectives for c in second.population
    ]
    assert first.gateway.counts["exact"] == second.gateway.counts["exact"]
    assert len(first.population) == config.population
    assert first.termination_reason == second.termination_reason == "generation_limit"
