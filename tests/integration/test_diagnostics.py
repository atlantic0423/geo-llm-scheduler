"""D01 artifact isolation, aggregation recount and fixed-generation replay."""

import csv
from dataclasses import replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.diagnostic_instances import materialize_diagnostic_instance
from geo_llm_scheduler.experiments.diagnostics import (
    DiagnosticRunSpec,
    aggregate_diagnostics,
    run_diagnostic,
)


def test_diagnostic_replay_direct_fields_and_aggregate_recount(tmp_path):
    record = materialize_diagnostic_instance(tmp_path / "instances", "D0_balanced", 6, 4)
    base = replace(
        Config(),
        population=10,
        neighborhood=3,
        generations=1,
        method="full",
        trigger_mode="always",
        rl_steps=1,
        fixed_budget=2,
        enabled_operators=(1,),
    )
    common = dict(
        scenario="D0_balanced",
        jobs=6,
        instance_seed=4,
        algorithm_seed=9,
        generations=1,
        instance_path=str(record["path"]),
    )
    pilot = DiagnosticRunSpec(phase="pilot", **common)
    main = DiagnosticRunSpec(phase="main", **common)
    first = run_diagnostic(pilot, base, tmp_path / "outputs")
    second = run_diagnostic(main, base, tmp_path / "outputs")
    assert first["trace_hash"] == second["trace_hash"]
    assert (tmp_path / "outputs" / "pilot" / "runs" / pilot.run_id).exists()
    assert (tmp_path / "outputs" / "main" / "runs" / main.run_id).exists()

    step_path = tmp_path / "outputs" / "pilot" / "runs" / pilot.run_id / "rl_steps.csv"
    steps = list(csv.DictReader(step_path.open(encoding="utf-8")))
    assert steps
    required = {
        "run_id",
        "state_id",
        "preference",
        "dominant_condition",
        "search_progress",
        "severity_resource",
        "selected_action",
        "explore_or_greedy",
        "raw_construction_attempts",
        "construction_seconds",
    }
    assert required <= set(steps[0])

    result = aggregate_diagnostics(tmp_path / "outputs")
    assert result["runs"] == 2
    assert result["rl_steps"] == 20
    counts_path = tmp_path / "outputs" / "aggregate" / "state_action_counts.csv"
    counts = list(csv.DictReader(counts_path.open(encoding="utf-8")))
    assert sum(int(row["selections"]) for row in counts) == result["rl_steps"]
    assert result["state_action_rows"] == 2 * 42 * 8
