"""Campaign sealing and production-time boundary checks for the three-arm runner."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.run import run

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("llm_runner", ROOT / "scripts/run_llm_flash.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_three_arm_orders_balance_slots_and_directed_predecessors():
    orders = [runner.balanced_order(i) for i in range(6)]
    assert len(set(tuple(o) for o in orders)) == 6
    assert all(set(o) == set(runner.ARMS) for o in orders)
    for position in range(3):
        assert all(sum(o[position] == a for o in orders) == 2 for a in runner.ARMS)
    assert all(
        sum((a, b) in list(zip(o, o[1:])) for o in orders) == 2
        for a in runner.ARMS
        for b in runner.ARMS
        if a != b
    )


def test_safe_boundary_wait_exhausting_budget_stops_before_new_offspring(problem, monkeypatch):
    module = __import__("geo_llm_scheduler.engine.run", fromlist=["run"])
    clock = [0.0]
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])

    def observe(event, data):
        if event == "before_offspring":
            clock[0] = 2.0

    initial = [Genotype((0, 0, 1, 1), (0, 1, 0, 1)), Genotype((1, 1, 0, 0), (1, 0, 1, 0))]
    config = replace(
        Config(),
        population=2,
        neighborhood=2,
        seconds=1,
        generations=1,
        method="plain",
        polish=False,
    )
    result = run(problem, config, initial, observer=observe, safe_boundary_deadline=True)
    assert result.offspring_count == 0 and result.termination_reason == "time_budget"


def test_input_seal_detects_mutation(tmp_path):
    # Artificial provenance fixture is never a formal experiment input.
    old = tmp_path / "old"
    old.mkdir()
    (old / "specs").mkdir()
    (old / "inputs").mkdir()
    blocks, files = [], {}
    for i in range(32):
        key = f"b{i}_s16101"
        blocks.append(key)
        input_name = f"inputs/{i}.json"
        (old / input_name).write_text("{}")
        value = dict(
            seed=16101,
            n=20,
            regions=2,
            machines=2,
            pressure="arrival",
            variant=0,
            base_seed=i,
            seconds=300,
            input=input_name,
            input_sha256=runner.sha(old / input_name),
        )
        (old / "specs" / f"{key}.json").write_text(json.dumps(value))
        files[input_name] = runner.sha(old / input_name)
        files[f"specs/{key}.json"] = runner.sha(old / "specs" / f"{key}.json")
    (old / "manifest.json").write_text(
        json.dumps(
            dict(
                commit="b148c4addd207fe30a6b2a43d953a04afc35d3a1",
                runs=1056,
                blocks=blocks,
                files=files,
            )
        )
    )
    (old / "manifest.sha256.json").write_text(
        json.dumps(dict(sha256=runner.sha(old / "manifest.json")))
    )
    root = tmp_path / "new"
    result = runner.prepare(root, old)
    assert result["runs"] == 192 and len(result["blocks"]) == 64 and len(result["arms"]) == 3
    assert result["learning_parameters"]["model"] == "deepseek-flash"
    assert runner.validate_inputs(root) == result
    changed = root / next(iter(result["files"]))
    changed.write_text("mutated")
    with pytest.raises(AssertionError):
        runner.validate_inputs(root)
