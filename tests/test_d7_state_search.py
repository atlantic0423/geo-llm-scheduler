"""D7 state/action semantics, paired matrix identities and transactional outputs."""

import importlib.util
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import (
    Candidate,
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import d7_worker, runner
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash, validate_complete
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.operators.structural import StructuralOperator, relocation_tou_proxy
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.rl.state import decode, extract, state_count
from geo_llm_scheduler.utils.rng import RNGManager


@pytest.mark.parametrize("policy,size", [("dominant", 42), ("pooled", 6), ("compound", 84)])
def test_state_dimensions_coexistence_and_terminal_backup(policy, size):
    config = Config(method="full", rl_state_policy=policy)
    single = tuple(t * (3 if i == 3 else 0.5) for i, t in enumerate(config.severity_thresholds))
    multiple = tuple(
        t * (3 if i == 3 else 2 if i == 0 else 0.5)
        for i, t in enumerate(config.severity_thresholds)
    )
    a, b = (extract(0, v, 0, config) for v in (single, multiple))
    assert state_count(policy) == size
    assert decode(a, policy)[0::2] == ("Electricity", "Improving")
    assert (a != b) is (policy == "compound")
    if policy == "compound":
        assert a % 2 == 0 and b == a + 1
    controller = Controller(config)
    assert len(controller.q) == len(controller.visits) == size
    controller.q[-1][0] = 1000
    controller.update(a, 1, 5, size - 1, terminal=True)
    assert controller.q[a][0] == pytest.approx(1.5)
    controller.update(a, 1, 5, size - 1)
    assert controller.q[a][0] == pytest.approx(1.5 + 0.3 * (5 + 0.7 * 1000 - 1.5))
    for s in range(size):
        assert controller.select(s, 0, random.Random(s))[0] in range(1, 9)
        assert decode(s, policy)[0] in ("Flow", "Balanced", "Electricity")
    with pytest.raises(ValueError):
        decode(size, policy)
    with pytest.raises(ValueError):
        controller.available_actions(size)


def test_policy_boundaries_and_pooled_condition_removal():
    with pytest.raises(ValueError):
        state_count("unknown")
    for kwargs in (
        {"rl_state_policy": "bad"},
        {"a3_region_policy": "bad"},
        {"rl_state_policy": "pooled"},
        {"method": "full", "rl_state_policy": "compound", "action_mask_policy": "no_a6"},
    ):
        with pytest.raises(ValueError):
            Config(**kwargs)
    c = Config(method="full", rl_state_policy="pooled")
    assert extract(0, (0,) * 6, 0, c) == extract(0, (1,) * 6, 0, c) == 4
    assert extract(0, (1,) * 6, 5, c) == 5


def test_tariff_ranking_changes_targets_with_exact_ground_truth_and_boundaries():
    p = ProblemInstance(
        (Job("j", 0, Profile(100, 1, 1), Profile(200, 1, 1)),),
        tuple(
            Region(str(i), (Tariff(0, 50, price), Tariff(50, 10000, price * 2)), 0)
            for i, price in enumerate((0.5, 0.9, 0.3, 0.1))
        ),
        tuple(ServingInstance(str(i), i, 8, 10, 100) for i in range(4)),
    )
    g, schedule = Genotype((0, 0), (0, 0)), Schedule((0, 100))
    x = Candidate(g, schedule, evaluate(p, g, schedule))
    proxy = relocation_tou_proxy(p, x)
    assert proxy[0, 3] == pytest.approx(90 * (50 * 0.1 + 50 * 0.2) / 3600)
    baseline = Config(method="full")
    operator = StructuralOperator(3)
    old, attempts, _ = operator._assignments(p, x, baseline)
    new, new_attempts, _ = operator._assignments(p, x, replace(baseline, a3_region_policy="tariff"))
    assert old[0].ms == (1, 1) and new[0].ms == (3, 3)
    assert attempts == new_attempts and set(old) == set(new)
    batch = operator.propose(
        p, x, 1, replace(baseline, a3_region_policy="tariff"), random.Random(1)
    )
    assert len(batch.proposals) == 1 and batch.proposals[0].genotype.ms == (3, 3)
    # The ranking helper must not mutate the incumbent or claim exact marginal bills.
    assert x.evaluation == evaluate(p, g, schedule)


@pytest.mark.parametrize("policy", ["pooled", "compound"])
def test_alternate_state_engine_keeps_exact_archive_and_default_path(problem, policy):
    c = Config(population=6, neighborhood=3, generations=2, method="full", trigger_mode="always")
    old = run(problem, c)
    explicit = run(problem, replace(c, rl_state_policy="dominant", a3_region_policy="load"))
    assert deterministic_trace(old.trace) == deterministic_trace(explicit.trace)
    changed = run(problem, replace(c, rl_state_policy=policy))
    assert len(changed.controller.q) == state_count(policy)
    assert changed.trace
    for candidate in changed.population + changed.archive.members:
        assert candidate.evaluation == evaluate(problem, candidate.genotype, candidate.schedule)


def test_freeze_disjoint_same_initials_single_factors_and_williams(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("d7_campaign", Path("scripts/run_d7_campaign.py"))
    assert spec and spec.loader
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    monkeypatch.setattr(
        campaign, "git_metadata", lambda *a: {"git_dirty": False, "git_commit": "fixture"}
    )
    manifests = []
    bases = []
    for node in range(2):
        root = tmp_path / f"formal{node}"
        matrix = campaign.prepare(root, Path.cwd(), node=node)
        manifests.append(matrix)
        seen = set()
        positions = {a: [0] * 8 for a in campaign.ARMS}
        for block in matrix["blocks"]:
            specs = [json.loads((root / "specs" / f"{k}.json").read_text()) for k in block["runs"]]
            assert (
                len({s["instance_hash"] for s in specs})
                == len({s["initial_hash"] for s in specs})
                == 1
            )
            assert {s["arm"] for s in specs} == set(campaign.ARMS)
            seen.add(specs[0]["base_seed"])
            baseline = next(s["config"] for s in specs if s["arm"] == "BASE")
            for position, s in enumerate(specs):
                positions[s["arm"]][position] += 1
                changed = {key for key in baseline if baseline[key] != s["config"][key]} - {
                    "output"
                }
                assert len(changed) == (s["arm"] != "BASE")
        assert all(counts == [8] * 8 for counts in positions.values())
        bases.append(seen)
        assert matrix["run_count"] == 512 and len(matrix["blocks"]) == 64
        assert matrix["nominal_worker_hours"] == 192
        with pytest.raises(FileExistsError):
            campaign.prepare(root, Path.cwd(), node=node)
    assert not bases[0] & bases[1] and len(bases[0] | bases[1]) == 32


@pytest.mark.parametrize(
    "arm,kwargs",
    [
        ("BASE", {}),
        ("STATE6", {"rl_state_policy": "pooled"}),
        ("STATE84", {"rl_state_policy": "compound"}),
        ("B3", {"fixed_budget": 3}),
        ("B10", {"fixed_budget": 10}),
        ("SEQ", {"budget_policy": "sequential"}),
        ("A3_TOU", {"a3_region_policy": "tariff"}),
        ("RPERM", {"replacement_policy": "permuted"}),
    ],
)
def test_worker_frozen_inputs_final_exact_bounded_observation_and_tamper(
    tmp_path, problem, monkeypatch, arm, kwargs
):
    provenance = {"git_dirty": False, "git_commit": "fixture", "git_branch": "test"}
    monkeypatch.setattr(d7_worker, "git_metadata", lambda *a: provenance)
    monkeypatch.setattr(runner, "git_metadata", lambda *a: provenance)
    c = Config(
        population=4, neighborhood=2, generations=2, method="full", trigger_mode="always", **kwargs
    )
    (tmp_path / "inputs").mkdir()
    save_instance(problem, tmp_path / "inputs/problem.json")
    gs = initial_genotypes(problem, c, RNGManager(1).stream("initialization"))
    atomic_json(tmp_path / "inputs/initial.json", [asdict(g) for g in gs])
    spec = {
        "key": "key",
        "arm": arm,
        "instance": "inputs/problem.json",
        "initial": "inputs/initial.json",
        "instance_hash": file_hash(tmp_path / "inputs/problem.json"),
        "initial_hash": file_hash(tmp_path / "inputs/initial.json"),
        "source_hash": canonical_source_hash(),
        "source_commit": "fixture",
        "config": asdict(c),
    }
    atomic_json(tmp_path / "specs/key.json", spec)
    d7_worker.execute(tmp_path, "key")
    output = tmp_path / "runs/key"
    assert validate_complete(output, spec) == (True, "complete")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["light_budget_histogram"]
    assert (
        sum(
            v["calls"] for k, v in summary["light_operator_totals"].items() if k.startswith("state")
        )
        > 0
    )
    assert len(json.loads((output / "qtable.json").read_text())["q"]) == state_count(
        c.rl_state_policy
    )
    (output / "config.json").write_text("{}")
    assert validate_complete(output, spec)[0] is False
