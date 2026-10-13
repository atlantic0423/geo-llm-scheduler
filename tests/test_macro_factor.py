"""Counterexamples for cross-call RNG coupling and frozen factor evidence."""

import importlib.util
import platform
import random
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments import macro_factor, macro_factor_worker, runner
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.macro_factor import ARMS, arm_config, quota_used_bytes
from geo_llm_scheduler.experiments.macro_width import read_json, stress_pair
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager, operator_call_rng


@pytest.mark.parametrize(
    "fields",
    [
        {"macro_rng_policy": "bad"},
        {"macro_rng_policy": True},
        {"macro_rng_policy": "call"},
        {"method": "full", "budget_policy": "random", "macro_rng_policy": "call"},
    ],
)
def test_invalid_policy_scope(fields):
    with pytest.raises(ValueError):
        Config(**fields)


def test_shared_preserves_object_and_state():
    parent = random.Random(1213)
    state = parent.getstate()
    assert operator_call_rng(parent, "shared") is parent
    assert parent.getstate() == state
    with pytest.raises(ValueError):
        operator_call_rng(parent, "bad")
    assert parent.getstate() == state


def test_call_parent_consumption_independent_of_internal_budget():
    six, ten, reference = [random.Random(343) for _ in range(3)]
    for _ in range(5):
        seed = reference.getrandbits(128)
        children = [operator_call_rng(six, "call"), operator_call_rng(ten, "call")]
        assert children[0].getstate() == children[1].getstate() == random.Random(seed).getstate()
        for _ in range(6):
            children[0].random()
        for _ in range(10):
            children[1].random()
        assert six.getstate() == ten.getstate() == reference.getstate()
    # Shared streams provide a counterexample: different internal work affects the next call.
    six, ten = random.Random(343), random.Random(343)
    for _ in range(6):
        operator_call_rng(six, "shared").random()
    for _ in range(10):
        operator_call_rng(ten, "shared").random()
    assert six.getstate() != ten.getstate()


def test_four_arm_scope_no_heavy_observer():
    configs = [arm_config(Config(), arm) for arm in ARMS]
    assert [c.macro_generation_budget for c in configs] == [0, 10, 0, 10]
    assert [c.macro_rng_policy for c in configs] == ["shared", "shared", "call", "call"]
    assert all(c.fixed_budget == 6 and not c.macro_diagnostics for c in configs)
    assert all(c.a3_region_policy == "tariff" and c.rl_steps == 5 for c in configs)
    with pytest.raises(ValueError):
        arm_config(Config(), "invalid")


@pytest.mark.parametrize("arm", ARMS)
def test_fixed_generation_reproducible_and_default_equivalence(arm):
    problem = stress_pair(5, 1580001, "mixed")[1]
    config = arm_config(
        Config(population=6, neighborhood=3, generations=2, trigger_mode="always"), arm
    )
    first, second = run(problem, config), run(problem, config)
    assert deterministic_trace(first.trace) == deterministic_trace(second.trace)
    assert first.population == second.population and first.archive.members == second.archive.members
    if arm == "NATIVE_SHARED":
        from dataclasses import replace

        from geo_llm_scheduler.experiments.macro_width import arm_config as old_config

        legacy = run(problem, replace(old_config(config, "NATIVE6"), macro_diagnostics=False))
        assert deterministic_trace(first.trace) == deterministic_trace(legacy.trace)
        assert (
            first.population == legacy.population
            and first.archive.members == legacy.archive.members
        )


def test_actual_formal_matrix_balance_pairing_and_separation(tmp_path, monkeypatch):
    loader = importlib.util.spec_from_file_location(
        "factor_campaign", "scripts/run_macro_factor_campaign.py"
    )
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    monkeypatch.setattr(
        module, "git_metadata", lambda *a: {"git_commit": "fixture", "git_dirty": False}
    )
    monkeypatch.setattr(module, "stress_pair", lambda n, s, sc: stress_pair(4, s, sc))
    bases, runs, hours = set(), 0, 0
    for node in (0, 1):
        root = tmp_path / str(node)
        manifest = module.prepare(root, Path.cwd(), node=node, expected_host="fixture")
        local, positions, edges = set(), Counter(), Counter()
        for block in manifest["blocks"]:
            specs = [read_json(root / "specs" / (k + ".json")) for k in block["runs"]]
            assert (
                len({s["instance_hash"] for s in specs})
                == len({s["initial_hash"] for s in specs})
                == 1
            )
            assert {s["arm"] for s in specs} == set(ARMS)
            assert {s["algorithm_seed"] for s in specs} <= {1313, 2313}
            for index, spec in enumerate(specs):
                assert spec["config"]["fixed_budget"] == 6 and spec["expected_host"] == "fixture"
                positions[(spec["arm"], index)] += 1
                if index:
                    edges[(specs[index - 1]["arm"], spec["arm"])] += 1
            local.add(specs[0]["base_seed"])
        assert len(set(positions.values())) == len(set(edges.values())) == 1
        assert not local & bases
        bases |= local
        runs += manifest["run_count"]
        hours += manifest["nominal_worker_hours"]
        assert len(local) == (14 if node == 0 else 26)
        with pytest.raises(FileExistsError):
            module.prepare(root, Path.cwd())
    assert len(bases) == 40 and runs == 640 and hours == 320
    assert min(bases) == 1585001 and max(bases) == 1595020


@pytest.mark.parametrize("arm", ARMS)
def test_transactional_exact_worker_without_probe_and_tamper(tmp_path, monkeypatch, arm):
    root = tmp_path / "campaign"
    for directory in ("inputs", "specs", "runs"):
        (root / directory).mkdir(parents=True)
    problem = stress_pair(5, 1580002, "mixed")[1]
    config = arm_config(
        Config(population=6, neighborhood=3, generations=2, trigger_mode="always"), arm
    )
    save_instance(problem, root / "inputs/problem.json")
    initial = initial_genotypes(problem, config, RNGManager(config.seed).stream("initialization"))
    atomic_json(root / "inputs/initial.json", [asdict(g) for g in initial])
    spec = {
        "key": "fixture",
        "arm": arm,
        "config": asdict(config),
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

    monkeypatch.setattr(macro_factor_worker, "canonical_source_hash", lambda: "fixture")
    monkeypatch.setattr(macro_factor_worker, "git_metadata", metadata)
    monkeypatch.setattr(runner, "git_metadata", metadata)
    macro_factor_worker.execute(root, "fixture")
    output = root / "runs/fixture"
    assert macro_factor_worker.validate_complete(output, spec) == (True, "complete")
    assert read_json(output / "diagnostics.json.gz")["calls"] == {}
    assert not (root / "replay").exists()
    assert read_json(output / "summary.json")["call_sample_size"] == 0
    archive = output / "archive.json.gz"
    original = archive.read_bytes()
    archive.write_bytes(original + b"tamper")
    assert not macro_factor_worker.validate_complete(output, spec)[0]


CAMPAIGN = Path("/workspace/owner/macro_factor_20261013")
ENOENT = f"du: cannot access '{CAMPAIGN}/formal_node0/ops/heartbeats/.n50.json.123.tmp': No such file or directory\n"


@pytest.mark.parametrize(
    "stderr",
    [
        "",
        ENOENT.replace("No such file or directory", "Permission denied"),
        ENOENT.replace("macro_factor_20261013", "other"),
        ENOENT.replace(".123.tmp", ".123.json"),
        ENOENT + "du: unrelated failure\n",
    ],
)
def test_quota_permanent_or_unrelated_failures_propagate(stderr):
    error = subprocess.CalledProcessError(1, ["du"], stderr=stderr)

    def reader(*args, **kwargs):
        raise error

    with pytest.raises(subprocess.CalledProcessError) as caught:
        quota_used_bytes(CAMPAIGN, reader)
    assert caught.value is error


def test_quota_atomic_retry_exact_measurement_and_persistent_closed(monkeypatch):
    monkeypatch.setattr(macro_factor.time, "sleep", lambda _: None)
    calls = []

    def reader(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command, stderr=ENOENT)
        return "4567\t/workspace/owner\n"

    assert quota_used_bytes(CAMPAIGN, reader) == 4567 and len(calls) == 2
    assert all(x == ["du", "-sb", str(CAMPAIGN.parent)] for x in calls)

    def failing(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, stderr=ENOENT.encode())

    calls.clear()
    assert quota_used_bytes(CAMPAIGN, failing) == 100 * 1024**3 and len(calls) == 4

    def negative(*args, **kwargs):
        return "-1\t/workspace/owner\n"

    with pytest.raises(ValueError):
        quota_used_bytes(CAMPAIGN, negative)
