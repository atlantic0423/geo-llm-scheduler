"""Night pipeline matrix identity and incomplete-artifact detection."""

import importlib.util
import json
import sys
from pathlib import Path


def load_pipeline_module():
    path = Path(__file__).parents[2] / "scripts" / "run_diagnostic_night_pipeline.py"
    spec = importlib.util.spec_from_file_location("night_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pipeline_matrices_are_disjoint_and_complete():
    module = load_pipeline_module()
    d01 = module.d01_identities()
    d02 = module.d02_identities()
    assert len(d01) == 63
    assert len(d02) == 15
    assert len({identity.run_id for identity in d01}) == 63
    assert len({identity.run_id for identity in d02}) == 15
    assert {identity.phase for identity in d01} == {"main"}
    assert {identity.phase for identity in d02} == {"mixed"}
    assert not ({identity.run_id for identity in d01} & {identity.run_id for identity in d02})


def test_pipeline_accepts_only_complete_exact_identity(tmp_path, monkeypatch):
    module = load_pipeline_module()
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path)
    identity = module.d02_identities()[0]
    identity.directory.mkdir(parents=True)
    for name in module.REQUIRED_ARTIFACTS:
        (identity.directory / name).write_text("{}\n", encoding="utf-8")
    metadata = {
        "run_id": identity.run_id,
        "phase": identity.phase,
        "scenario": identity.scenario,
        "jobs": identity.jobs,
        "instance_seed": identity.instance_seed,
        "algorithm_seed": identity.algorithm_seed,
        "generations": identity.generations,
    }
    (identity.directory / "diagnostic.json").write_text(json.dumps(metadata), encoding="utf-8")
    (identity.directory / "summary.json").write_text(
        json.dumps({"termination_reason": "generation_limit"}), encoding="utf-8"
    )
    (identity.directory / "qtable.json").write_text("{}", encoding="utf-8")
    (identity.directory / "rl_steps.csv").write_text(
        "state_id,selected_action\n0,A1\n", encoding="utf-8"
    )
    assert module.validate_run(identity) == (True, "complete")
    (identity.directory / "summary.json").write_text(
        json.dumps({"termination_reason": "time_limit"}), encoding="utf-8"
    )
    valid, reason = module.validate_run(identity)
    assert not valid
    assert reason == "termination:time_limit"
