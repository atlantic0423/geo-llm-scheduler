"""Portable worker artifact export and conservative memory admission boundaries."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.rl.controller import Controller

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        "export_test_" + name, ROOT / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_fixture(problem, tmp_path, monkeypatch):
    runner = load_script("run_llm_flash")
    root = tmp_path / "synthetic-export-campaign"
    root.mkdir()
    (root / "inputs").mkdir()
    (root / "specs").mkdir()
    input_path = root / "inputs" / "fixture.json"
    input_path.write_text("{}")
    key = "synthetic_n2_seed7"
    spec = dict(
        key=key,
        host=runner.socket.gethostname(),
        seed=7,
        seconds=300,
        input="inputs/fixture.json",
        input_sha256=runner.sha(input_path),
    )
    spec_path = root / "specs" / (key + ".json")
    runner.atomic_json(spec_path, spec)
    manifest = dict(
        source="synthetic-frozen",
        host=runner.socket.gethostname(),
        commit="fixture",
        files={"specs/" + key + ".json": runner.sha(spec_path)},
        learning_parameters=dict(model="deepseek-flash", calls_max=3),
    )
    runner.atomic_json(root / "manifest.json", manifest)
    initial = dict(
        initial_population_sha256="fixture-population", initial_archive_sha256="fixture-archive"
    )
    runner.atomic_json(
        root / "preflight" / (key + ".json"),
        dict(
            source=manifest["source"],
            input_sha256=spec["input_sha256"],
            policies=dict(bill4=dict(evidence=initial)),
        ),
    )
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: calls.append(("lock", args[1]))),
    )
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(
            RLIMIT_AS=9,
            RUSAGE_SELF=0,
            setrlimit=lambda *args: calls.append(("limit", args)),
            getrusage=lambda *args: SimpleNamespace(ru_maxrss=1024),
        ),
    )
    monkeypatch.setattr(
        runner.os, "sched_setaffinity", lambda *args: calls.append(("cpu", args)), raising=False
    )
    monkeypatch.setattr(runner, "source_hash", lambda: "synthetic-frozen")
    monkeypatch.setattr(runner, "load_instance", lambda path: problem)

    def fake_run(p, arm, cfg, evidence, *, observer, parameters):
        gateway = EvaluationGateway(p, Archive())
        candidate = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))
        population = [candidate] * 100
        evidence.update(policy="bill4", **initial)
        observer("initial_ready", dict(population=population, gateway=gateway))
        observer("candidate", dict(candidate=candidate, gateway=gateway))
        observer("rl_step", dict(action=1, accepted=False))
        if arm == "LLM_FLASH":
            parameters["policy_sink"](
                dict(
                    accepted=True,
                    response=dict(
                        returned_model="deepseek-flash", finish_reason="stop", content="fixture"
                    ),
                )
            )
            evidence["llm"] = dict(requests=1, valid_requests=1)
        return SimpleNamespace(
            population=population,
            archive=gateway.archive,
            gateway=gateway,
            elapsed=300.1,
            controller=Controller(cfg),
            termination_reason="time_budget",
            offspring_count=1,
            adaptive=None,
        )

    monkeypatch.setattr(runner, "run_policy", fake_run)
    return runner, root, key, manifest, calls


@pytest.mark.parametrize("arm", ["BILL_MOEAD", "MAIN_Q", "LLM_FLASH"])
def test_worker_writes_serializable_frozen_parameters_and_independently_validated_artifacts(
    export_fixture, arm
):
    runner, root, key, manifest, calls = export_fixture
    runner.worker(root, key, arm, cpu=3)
    out = root / "runs" / key / arm
    summary = json.loads((out / "summary.json").read_text())
    assert summary["learning_parameters"] == manifest["learning_parameters"]
    assert "policy_sink" not in summary["learning_parameters"]
    assert summary["final_exact_verified"] == 101
    assert summary["population"] == 100 and summary["archive"] == 1
    assert summary["production_seconds"] == 300.1
    assert runner.complete(root, key, arm, manifest["source"])
    assert not (out / "failure.json").exists()
    assert {name for name in runner.REQUIRED} <= {p.name for p in out.iterdir()}
    with gzip.open(out / "initial.json.gz", "rt", encoding="utf-8") as stream:
        assert len(json.load(stream)) == 100
    assert any(name == "limit" for name, _ in calls)
    assert any(name == "cpu" and args == (0, {3}) for name, args in calls)
    # Actual complete validation must reject a changed artifact, not merely trust a marker.
    (out / "global_prompt.txt").write_text("mutated", encoding="utf-8")
    with pytest.raises(AssertionError):
        runner.complete(root, key, arm, manifest["source"])


def test_llm_worker_cannot_be_accepted_when_every_policy_request_fails(export_fixture, monkeypatch):
    runner, root, key, manifest, _ = export_fixture
    original = runner.run_policy

    def invalid_llm(*args, **kwargs):
        parameters = dict(kwargs["parameters"])
        sink = parameters["policy_sink"]
        parameters["policy_sink"] = lambda record: sink({**record, "accepted": False})
        kwargs["parameters"] = parameters
        result = original(*args, **kwargs)
        evidence = args[3]
        evidence["llm"]["valid_requests"] = 0
        return result

    monkeypatch.setattr(runner, "run_policy", invalid_llm)
    with pytest.raises(AssertionError):
        runner.worker(root, key, "LLM_FLASH", cpu=3)
    out = root / "runs" / key / "LLM_FLASH"
    failure = json.loads((out / "failure.json").read_text())
    assert "LLM treatment had no valid policy" in failure["traceback"]
    assert (out / "policy_calls.jsonl.gz").exists()


@pytest.mark.parametrize(
    "available_gib,active,group_gb,disk_gib,expected",
    [
        (255, 0, 10, 300, True),
        (254, 0, 10, 300, False),
        (400, 74, 10, 300, False),
        (500, 95, 100, 300, True),
        (500, 0, 359, 300, False),
        (500, 0, 10, 219, False),
    ],
)
def test_admission_reserves_growth_of_active_workers_and_http_children(
    monkeypatch, tmp_path, available_gib, active, group_gb, disk_gib, expected
):
    dispatcher = load_script("llm_flash_dispatch")
    original = Path.read_text

    def proc_contents(path, *args, **kwargs):
        if str(path).replace("\\", "/") == "/proc/meminfo":
            return f"MemAvailable: {int(available_gib * 1024**2)} kB\n"
        if str(path).replace("\\", "/") == "/proc/self/cgroup":
            return "0::/fixture\n"
        if path.name == "memory.current":
            return str(int(group_gb * 10**9))
        return original(path, *args, **kwargs)

    original_exists = Path.exists
    monkeypatch.setattr(Path, "read_text", proc_contents)
    monkeypatch.setattr(Path, "exists", lambda p: p.name == "memory.current" or original_exists(p))
    monkeypatch.setattr(
        dispatcher.shutil, "disk_usage", lambda root: SimpleNamespace(free=disk_gib * 1024**3)
    )
    assert dispatcher.resources_available(tmp_path, active_count=active) is expected
