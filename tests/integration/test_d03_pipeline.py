"""D03 matrix identities and expanded diagnostic logging."""

import csv
import importlib.util
import sys
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.diagnostic_instances import materialize_diagnostic_instance
from geo_llm_scheduler.experiments.diagnostics import DiagnosticRunSpec, run_diagnostic


def load_pipeline():
    path = Path(__file__).parents[2] / "scripts" / "run_d03_v2_pipeline.py"
    spec = importlib.util.spec_from_file_location("d03_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_d03_matrix_contract_is_disjoint_and_complete():
    module = load_pipeline()
    first, second = module.calibration_runs()
    assert len(first) == 42
    assert len(second) == 9
    frozen = {
        "type_i_profiles": {condition: "moderate" for condition in module.CONDITIONS},
        "type_ii_profile": "broad",
    }
    type_i, type_ii = module.evaluation_runs(frozen)
    assert len(type_i) == 42
    assert len(type_ii) == 15
    calibration_seeds = {run.instance_seed for run in first + second}
    evaluation_seeds = {run.instance_seed for run in type_i + type_ii}
    assert calibration_seeds.isdisjoint(evaluation_seeds)


def test_d03_observational_fields_and_generation_zero_snapshot(tmp_path):
    record = materialize_diagnostic_instance(tmp_path / "instances", "D0_balanced", 4, 3)
    config = Config(
        population=4,
        neighborhood=2,
        generations=1,
        method="full",
        trigger_mode="always",
        rl_steps=1,
        fixed_budget=1,
        initialization_attempts=200,
        polish=False,
    )
    spec = DiagnosticRunSpec("d03_test", "D03_test", 4, 3, 101, 1, str(record["path"]))
    run_diagnostic(spec, config, tmp_path / "outputs")
    directory = tmp_path / "outputs" / "d03_test" / "runs" / spec.run_id
    with (directory / "rl_steps.csv").open(newline="", encoding="utf-8") as handle:
        step = next(csv.DictReader(handle))
    assert step["ratio_resource"] == str(float(step["severity_resource"]) / 0.2)
    assert {"q_selected_before", "q_selected_after", "argmax_q_action", "B_eff"} <= set(step)
    with (directory / "initial_severity.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert (directory / "trigger_funnel.csv").stat().st_size > 0
