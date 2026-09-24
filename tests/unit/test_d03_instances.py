"""D03 v2 generator and seed-separation contracts."""

from geo_llm_scheduler.experiments.d03_instances import (
    CONDITIONS,
    TYPE_I_CALIBRATION_SEEDS,
    TYPE_I_EVALUATION_SEEDS,
    TYPE_II_CALIBRATION_SEEDS,
    TYPE_II_EVALUATION_SEEDS,
    generate_type_i,
    generate_type_ii,
    materialize,
)
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.io.validation import validate_problem


def test_d03_seeds_are_disjoint_and_generators_are_deterministic():
    assert set(TYPE_I_CALIBRATION_SEEDS).isdisjoint(TYPE_I_EVALUATION_SEEDS)
    assert set(TYPE_II_CALIBRATION_SEEDS).isdisjoint(TYPE_II_EVALUATION_SEEDS)
    for condition in CONDITIONS:
        first = generate_type_i(condition, "moderate", 12, 71)
        assert first == generate_type_i(condition, "moderate", 12, 71)
        validate_problem(first)
    transition = generate_type_ii("broad", 12, 81)
    assert transition == generate_type_ii("broad", 12, 81)
    validate_problem(transition)


def test_d03_materialized_instance_roundtrips_with_full_horizon(tmp_path):
    record = materialize(tmp_path, "type_ii", "transition", "burst", 12, 9, "evaluation")
    problem = load_instance(record["path"])
    conservative = max(job.release for job in problem.jobs) + sum(
        job.prefill.duration + job.decode.duration + job.kv_delay for job in problem.jobs
    )
    assert all(region.tariffs[-1].end > conservative for region in problem.regions)
    assert record["role"] == "evaluation"
