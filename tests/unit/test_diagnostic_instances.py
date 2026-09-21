"""Deterministic D01 scenario construction and manifest boundaries."""

import csv
import json

import pytest

from geo_llm_scheduler.experiments.diagnostic_instances import (
    DEMAND_VARIANTS,
    SCENARIOS,
    diagnostic_parameters,
    generate_diagnostic_instance,
    materialize_diagnostic_instance,
    write_manifest,
)
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.io.validation import validate_problem


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios_are_deterministic_valid_and_covered(scenario):
    variant = "multi_tied_peak" if scenario == "D5_demand" else None
    first = generate_diagnostic_instance(scenario, 12, 7, variant)
    second = generate_diagnostic_instance(scenario, 12, 7, variant)
    assert first == second
    validate_problem(first)
    conservative_horizon = max(job.release for job in first.jobs) + sum(
        job.prefill.duration + job.decode.duration + job.kv_delay for job in first.jobs
    )
    assert all(region.tariffs[-1].end > conservative_horizon for region in first.regions)
    different = generate_diagnostic_instance(scenario, 12, 8, variant)
    assert first.jobs != different.jobs


@pytest.mark.parametrize("variant", DEMAND_VARIANTS)
def test_demand_variants_are_explicit(variant):
    problem = generate_diagnostic_instance("D5_demand", 10, 1, variant)
    assert problem.regions[0].demand_rate == 20


def test_manifest_matches_serialized_instance(tmp_path):
    record = materialize_diagnostic_instance(tmp_path, "D2_kv", 8, 11)
    manifest = tmp_path / "manifest.csv"
    write_manifest([record], manifest)
    row = next(csv.DictReader(manifest.open(encoding="utf-8")))
    assert row["instance_id"] == record["instance_id"]
    assert json.loads(row["parameters_json"]) == diagnostic_parameters("D2_kv")
    assert load_instance(row["path"]) == generate_diagnostic_instance("D2_kv", 8, 11)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_structured_parameters_describe_generated_instance(scenario):
    variant = "mixed_peak" if scenario == "D5_demand" else None
    parameters = diagnostic_parameters(scenario, variant)
    problem = generate_diagnostic_instance(scenario, 100, 17, variant)

    def in_grid(value, component):
        return (
            component["start_s"] <= value <= component["end_s"]
            and (value - component["start_s"]) % component["step_s"] == 0
        )

    release = parameters["release"]
    for job in problem.jobs:
        assert job.prefill.duration in parameters["prefill_s"]
        assert job.decode.duration in parameters["decode_s"]
        assert job.prefill.compute in parameters["compute"]
        assert job.decode.compute in parameters["compute"]
        assert job.prefill.vram in parameters["vram_gb"]
        assert job.decode.vram in parameters["vram_gb"]
        assert job.kv_delay in parameters["kv_s"]
        if release["kind"] == "uniform_grid":
            assert in_grid(job.release, release)
        elif release["kind"] == "mixture":
            assert any(in_grid(job.release, component) for component in release["components"])
        elif release["kind"] == "mixed_peak":
            assert any(
                in_grid(job.release, component) for component in release["mixed_peak"]["components"]
            )
    assert {region.demand_rate for region in problem.regions} == {parameters["demand_rate_cny_kw"]}
    if "flat_tariff_cny_kwh" in parameters:
        assert {tariff.price for region in problem.regions for tariff in region.tariffs} == {
            parameters["flat_tariff_cny_kwh"]
        }
    elif "flat_region_prices_cny_kwh" in parameters:
        assert [region.tariffs[0].price for region in problem.regions] == parameters[
            "flat_region_prices_cny_kwh"
        ]
    else:
        expected = [segment[2] for segment in parameters["tou_cny_kwh"]]
        assert [tariff.price for tariff in problem.regions[0].tariffs] == expected


def test_invalid_scenario_and_variant_are_rejected():
    with pytest.raises(ValueError):
        generate_diagnostic_instance("unknown", 2, 1)
    with pytest.raises(ValueError):
        generate_diagnostic_instance("D0_balanced", 2, 1, "mixed_peak")
    with pytest.raises(ValueError):
        generate_diagnostic_instance("D5_demand", 2, 1, "unknown")
