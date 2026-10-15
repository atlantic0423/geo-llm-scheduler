"""Counterexamples for budget separation, observer isolation and offline exact replay."""

import gzip
import importlib.util
import json
import platform
import random
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments import macro_width_worker, runner
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.macro_width import (
    ARMS,
    SCENARIOS,
    CallSampler,
    arm_config,
    gzip_json,
    read_json,
    stress_pair,
)
from geo_llm_scheduler.experiments.macro_width_replay import replay_call, replay_run
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager


@pytest.mark.parametrize(
    "fields",
    [
        {"macro_generation_budget": -1},
        {"macro_generation_budget": True},
        {"method": "full", "macro_generation_budget": 5},
        {"method": "full", "macro_generation_budget": 11},
        {"method": "full", "budget_policy": "random", "macro_generation_budget": 10},
        {"macro_diagnostics": 1},
    ],
)
def test_invalid_budget_interventions(fields):
    with pytest.raises(ValueError):
        Config(**fields)


def test_arm_scope_and_native_default():
    baseline = Config(method="full")
    configs = {a: arm_config(baseline, a) for a in ARMS}
    assert [c.fixed_budget for c in configs.values()] == [6, 3, 6, 10]
    assert [c.macro_generation_budget for c in configs.values()] == [0, 10, 10, 10]
    assert all(c.rl_state_policy == "dominant" and c.rl_steps == 5 for c in configs.values())
    with pytest.raises(ValueError):
        arm_config(baseline, "invalid")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stress_tariff_pairing_and_legality(scenario):
    h, t = stress_pair(6, 1520001, scenario)
    assert h.jobs == t.jobs and h.instances == t.instances
    assert all(h.eligible(o) for o in range(h.operation_count))
    if scenario == "region_asymmetric":
        assert len(h.instances) == 7
    with pytest.raises(ValueError):
        stress_pair(6, 1520001, "invalid")


def test_sampling_does_not_change_fixed_generation_native_search():
    p = stress_pair(5, 1520002, "mixed")[1]
    c = arm_config(
        Config(population=6, neighborhood=3, generations=2, trigger_mode="always"), "NATIVE6"
    )
    plain = run(p, replace(c, macro_diagnostics=False))
    sampler = CallSampler(random.Random(42))

    def observe(event, context):
        if event == "macro_before":
            sampler.before(context)
        if event == "macro_after":
            sampler.after(context)

    sampled = run(p, c, observer=observe)
    assert deterministic_trace(plain.trace) == deterministic_trace(sampled.trace)
    assert (
        plain.population == sampled.population and plain.archive.members == sampled.archive.members
    )
    assert sampler.rows and len(sampler.rows) <= 24


@pytest.mark.parametrize("arm", ARMS)
def test_all_saved_action_calls_replay_exact_and_rng(arm):
    p = stress_pair(5, 1520003, "burst")[1]
    c = arm_config(Config(population=6, neighborhood=3, generations=2, trigger_mode="always"), arm)
    sampler = CallSampler(random.Random(9))

    def observe(event, context):
        if event == "macro_before":
            sampler.before(context)
        if event == "macro_after":
            sampler.after(context)

    run(p, c, observer=observe)
    for row in json.loads(json.dumps(sampler.rows)).values():
        replay = replay_call(p, c, row)
        assert replay["online_rng_verified"]
        gains = [replay["prefix"][str(b)]["gain"] for b in (3, 6, 10)]
        assert gains == sorted(gains)
        assert replay["seq_missed_gain"] >= -1e-9
    row = next(iter(json.loads(json.dumps(sampler.rows)).values()))
    row["attempts"] += 1
    with pytest.raises(ValueError, match="RNG"):
        replay_call(p, c, row)


def test_gzip_evidence_roundtrip_and_fixed_header(tmp_path):
    data = {"items": [0.1, 2, {"immutable": "中文"}]}
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    gzip_json(a, data)
    gzip_json(b, data)
    assert a.read_bytes() == b.read_bytes() and read_json(a) == data
    plain = tmp_path / "x.json"
    plain.write_bytes(gzip.decompress(a.read_bytes()))
    assert read_json(plain) == data


def test_complete_matrix_base_ownership_native_initial_and_williams(tmp_path, monkeypatch):
    ms = importlib.util.spec_from_file_location(
        "macro_campaign", "scripts/run_macro_width_campaign.py"
    )
    module = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(module)
    monkeypatch.setattr(
        module, "git_metadata", lambda *a: {"git_commit": "fixture", "git_dirty": False}
    )
    monkeypatch.setattr(module, "stress_pair", lambda n, s, sc: stress_pair(4, s, sc))
    bases = set()
    runs = 0
    hours = 0
    for node in (0, 1):
        root = tmp_path / str(node)
        matrix = module.prepare(root, Path.cwd(), node=node, expected_host="fixture")
        local = set()
        for block in matrix["blocks"]:
            specs = [read_json(root / "specs" / (k + ".json")) for k in block["runs"]]
            assert (
                len({s["instance_hash"] for s in specs})
                == len({s["initial_hash"] for s in specs})
                == 1
            )
            assert {s["arm"] for s in specs} == set(ARMS)
            local.add(specs[0]["base_seed"])
        assert not local & bases
        bases |= local
        runs += matrix["run_count"]
        hours += matrix["nominal_worker_hours"]
        assert len(local) == (14 if node == 0 else 26)
        with pytest.raises(FileExistsError):
            module.prepare(root, Path.cwd(), node=node)
    assert len(bases) == 40 and runs == 640 and hours == 320


def test_transactional_compressed_worker_and_idempotent_probe(tmp_path, monkeypatch):
    """Run actual search/export/exact checks, then reject changed compressed evidence."""
    root = tmp_path / "campaign"
    (root / "inputs").mkdir(parents=True)
    (root / "specs").mkdir()
    (root / "runs").mkdir()
    p = stress_pair(5, 1520004, "mixed")[1]
    c = arm_config(
        Config(population=6, neighborhood=3, generations=2, trigger_mode="always"), "POOL3"
    )
    save_instance(p, root / "inputs/problem.json")
    initial = initial_genotypes(p, c, RNGManager(c.seed).stream("initialization"))
    atomic_json(root / "inputs/initial.json", [asdict(g) for g in initial])
    spec = {
        "key": "fixture",
        "arm": "POOL3",
        "config": asdict(c),
        "instance": "inputs/problem.json",
        "initial": "inputs/initial.json",
        "instance_hash": file_hash(root / "inputs/problem.json"),
        "initial_hash": file_hash(root / "inputs/initial.json"),
        "source_commit": "fixture",
        "source_hash": "fixture",
        "expected_host": platform.node(),
    }
    atomic_json(root / "specs/fixture.json", spec)

    def metadata(*args):
        return {"git_commit": "fixture", "git_dirty": False, "git_branch": "fixture"}

    monkeypatch.setattr(macro_width_worker, "canonical_source_hash", lambda: "fixture")
    monkeypatch.setattr(macro_width_worker, "git_metadata", metadata)
    monkeypatch.setattr(runner, "git_metadata", metadata)
    macro_width_worker.execute(root, "fixture")
    output = root / "runs/fixture"
    assert macro_width_worker.validate_complete(output, spec) == (True, "complete")
    assert not (output / "archive.json").exists() and read_json(output / "archive.json.gz")
    first = replay_run(root, "fixture")
    assert first == replay_run(root, "fixture") and first["calls"] > 0
    archive = output / "archive.json.gz"
    original = archive.read_bytes()
    archive.write_bytes(original + b"tamper")
    assert not macro_width_worker.validate_complete(output, spec)[0]
    archive.write_bytes(original)
    marker = root / "replay/fixture.json.complete.json"
    data = read_json(marker)
    data["source_marker_hash"] = "tamper"
    atomic_json(marker, data)
    with pytest.raises(ValueError, match="changed"):
        replay_run(root, "fixture")
