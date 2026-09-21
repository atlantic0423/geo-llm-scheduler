"""D02 generator validity, frozen ranges, and reproducible manifest coverage."""

import json

from geo_llm_scheduler.experiments.mixed_instances import (
    MIXED_SCENARIO,
    generate_mixed_instance,
    materialize_mixed_instance,
    mixed_parameters,
)
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.io.validation import validate_problem


def test_mixed_instance_is_deterministic_valid_and_has_full_tariff_horizon():
    first = generate_mixed_instance(50, 11)
    second = generate_mixed_instance(50, 11)
    assert first == second
    assert first.jobs != generate_mixed_instance(50, 22).jobs
    validate_problem(first)
    conservative_horizon = max(job.release for job in first.jobs) + sum(
        job.prefill.duration + job.decode.duration + job.kv_delay for job in first.jobs
    )
    assert all(region.tariffs[-1].end > conservative_horizon for region in first.regions)


def test_mixed_instance_uses_homogeneous_regions_and_frozen_nondirectional_ranges():
    parameters = mixed_parameters()
    problem = generate_mixed_instance(50, 33)
    assert len(problem.regions) == 3
    assert len(problem.instances) == 9
    hardware = {
        (instance.vram, instance.idle_kw, instance.active_kw) for instance in problem.instances
    }
    assert len(hardware) == 1
    assert {instance.region for instance in problem.instances} == {0, 1, 2}
    assert [
        sum(instance.region == region for instance in problem.instances) for region in range(3)
    ] == [
        3,
        3,
        3,
    ]
    assert "no post-hoc calibration" in parameters["calibration_policy"]
    for job in problem.jobs:
        assert job.prefill.duration in parameters["prefill_s"]
        assert job.decode.duration in parameters["decode_s"]
        assert job.prefill.compute in parameters["compute"]
        assert job.decode.compute in parameters["compute"]
        assert job.prefill.vram in parameters["vram_gb"]
        assert job.decode.vram in parameters["vram_gb"]
        assert job.kv_delay in parameters["kv_s"]


def test_mixed_manifest_records_all_sampled_ranges_and_roundtrips(tmp_path):
    record = materialize_mixed_instance(tmp_path, 12, 55)
    assert record["scenario"] == MIXED_SCENARIO
    assert json.loads(record["parameters_json"]) == mixed_parameters()
    assert load_instance(record["path"]) == generate_mixed_instance(12, 55)
