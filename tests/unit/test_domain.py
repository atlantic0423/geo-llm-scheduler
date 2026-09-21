"""Contract tests including invalid input and reproducible IO."""

from dataclasses import FrozenInstanceError, replace

import pytest

from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.io.validation import validate_genotype, validate_problem
from geo_llm_scheduler.utils.rng import RNGManager


def test_immutable_and_assignment(problem):
    with pytest.raises(FrozenInstanceError):
        problem.jobs = ()
    assert not hasattr(Schedule((0, 1, 2, 3)), "assignment")
    validate_problem(problem)
    validate_genotype(problem, Genotype((0, 1, 2, 2), (0, 1, 0, 1)))
    for g in (Genotype((0, 2, 0, 0), (0, 1, 0, 1)), Genotype((0, 0, 0, 0), (0, 0, 0, 1))):
        with pytest.raises(ValueError):
            validate_genotype(problem, g)


def test_roundtrip(problem, tmp_path):
    path = tmp_path / "instance.json"
    save_instance(problem, path)
    assert load_instance(path) == problem


def test_bad_profiles(problem):
    for value in (-1, float("nan"), float("inf")):
        bad = replace(problem.jobs[0], release=value)
        with pytest.raises(ValueError):
            validate_problem(replace(problem, jobs=(bad,)))


def test_rng_independence():
    a, b = RNGManager(42), RNGManager(42)
    a.stream("unrelated").random()
    assert [a.stream("variation").random() for _ in range(10)] == [
        b.stream("variation").random() for _ in range(10)
    ]
