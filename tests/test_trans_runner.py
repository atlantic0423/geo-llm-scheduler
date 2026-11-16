"""All eight runnable entries preserve provenance, exact checks and non-overwrite behavior."""

import json

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler.experiments.trans_runner import ALGORITHMS, NATIVE, digest, main, run_case
from geo_llm_scheduler.io.loaders import save_instance


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_eight_entries_persist_valid_complete_artifacts(tmp_path, algorithm):
    instance = tmp_path / "input.json"
    problem, _ = synthetic(4)
    save_instance(problem, instance)
    output = tmp_path / "result"
    summary = run_case(
        algorithm,
        instance,
        output,
        seconds=10,
        calibration=CAL,
        population_size=6 if algorithm in NATIVE else None,
        max_generations=1,
    )
    assert summary["independent_exact_audit"] == "passed" and summary["archive_size"] > 0
    config = json.loads((output / "config.json").read_text(encoding="utf8"))
    assert config["instance_sha256"] == digest(instance)
    assert config["engineering_override"]
    assert config["source"]["files"]
    if algorithm.startswith("A"):
        table = json.loads((output / "q_table.json").read_text())
        assert (table is None) == (algorithm in ("A2_RANDOM_OPERATOR", "A4_DIRECT_SHIFT"))
        if table is not None:
            assert "values" in table and "visits" in table
    for artifact in json.loads((output / "manifest.json").read_text()):
        assert digest(output / artifact["path"]) == artifact["sha256"]
    with pytest.raises(FileExistsError):
        run_case(algorithm, instance, output, seconds=10, calibration=CAL)


def test_runner_rejects_unspecified_calibration_and_invalid_configuration(tmp_path):
    problem, _ = synthetic(4)
    instance = tmp_path / "input.json"
    save_instance(problem, instance)
    for kwargs in (
        {"algorithm": "bad"},
        {"algorithm": ALGORITHMS[0]},
        {"algorithm": ALGORITHMS[0], "calibration": CAL, "population_size": 6},
        {"algorithm": "CMOEA_CD_PD", "seconds": -1},
    ):
        with pytest.raises(ValueError):
            run_case(instance=instance, output=tmp_path / "no_result", **({"seconds": 10} | kwargs))
        assert not (tmp_path / "no_result").exists()
    with pytest.raises(ValueError):
        run_case("CMOEA_CD_PD", instance, tmp_path / "failed", seconds=10, population_size=2)
    assert json.loads((tmp_path / "failed/status.json").read_text())["status"] == "failed"


def test_cli_calibration_key_and_execution(tmp_path, monkeypatch, capsys):
    problem, _ = synthetic(4)
    instance = tmp_path / "input.json"
    save_instance(problem, instance)
    calibration = tmp_path / "cal.json"
    calibration.write_text(json.dumps({"fold": CAL}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--algorithm",
            ALGORITHMS[0],
            "--instance",
            str(instance),
            "--output",
            str(tmp_path / "cli"),
            "--seconds",
            "10",
            "--max-generations",
            "1",
            "--calibration",
            str(calibration),
            "--calibration-key",
            "fold",
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--algorithm",
            ALGORITHMS[0],
            "--instance",
            str(instance),
            "--output",
            str(tmp_path / "bad"),
            "--calibration-key",
            "fold",
        ],
    )
    with pytest.raises(SystemExit):
        main()
