"""Initial slot-assignment study: unchanged reference path, pure permutation, sealed runner."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.experiments.rl23_support import study_config as reference_config
from geo_llm_scheduler.experiments.slot_assignment import (
    ARMS,
    direction_order,
    reference_policy,
    run_arm,
    slot_policy,
    study_config,
)
from geo_llm_scheduler.utils.numeric import identity

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake(flow, bill, tag):
    return SimpleNamespace(evaluation=SimpleNamespace(objectives=(flow, bill)), tag=tag)


def problem20():
    return case_problem(20, 3, 6, 202610160020, "arrival", 0)


def test_direction_order_puts_fast_flow_on_flow_weight_slots():
    members = [fake(3, 1, "slow"), fake(1, 3, "fast"), fake(2, 2, "mid"), fake(2, 2, "tie")]
    ordered = direction_order(members)
    assert [m.tag for m in ordered] == ["slow", "tie", "mid", "fast"]
    assert {id(m) for m in ordered} == {id(m) for m in members}
    assert direction_order(members) == ordered


def test_arm_mapping_and_shared_reference_configs():
    assert ARMS == ("BILL_RANDOM", "BILL_SORTED", "MAIN_RANDOM", "MAIN_SORTED")
    assert {reference_policy(a) for a in ARMS} == {"BILL_PERMUTED", "STRICT_TWO_FOCUS"}
    assert [slot_policy(a) for a in ARMS] == ["random", "sorted", "random", "sorted"]
    for arm in ARMS:
        assert study_config(arm, 7, 300) == reference_config(reference_policy(arm), 7, 300)
    with pytest.raises(ValueError):
        study_config("BILL_SHUFFLED", 7, 300)


@pytest.mark.parametrize("prefix", ("BILL", "MAIN"))
def test_random_arm_replays_reference_and_sorted_only_permutes(prefix):
    problem = problem20()
    cfg = replace(study_config(prefix + "_RANDOM", 16101, 300), generations=1, seconds=None)
    original = run_policy(problem, reference_policy(prefix + "_RANDOM"), cfg, {})
    initial: dict[str, list] = {}

    def capture(arm):
        def observe(event, data):
            if event == "initial_ready":
                initial[arm] = list(data["population"])

        return observe

    random_evidence: dict = {}
    sorted_evidence: dict = {}
    replay = run_arm(problem, prefix + "_RANDOM", cfg, random_evidence, observer=capture("r"))
    ordered = run_arm(problem, prefix + "_SORTED", cfg, sorted_evidence, observer=capture("s"))
    assert [identity(c) for c in replay.population] == [identity(c) for c in original.population]
    assert [identity(c) for c in replay.archive.members] == [
        identity(c) for c in original.archive.members
    ]
    assert sorted(identity(c) for c in initial["r"]) == sorted(identity(c) for c in initial["s"])
    assert [c.evaluation.objectives for c in initial["s"]] == sorted(
        (c.evaluation.objectives for c in initial["s"]), reverse=True
    )
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert random_evidence[field] == sorted_evidence[field]
    assert all(c.evaluation.feasible for c in ordered.population + ordered.archive.members)


def test_run_policy_rejects_order_that_is_not_a_permutation():
    cfg = replace(study_config("BILL_RANDOM", 16101, 300), generations=1, seconds=None)
    with pytest.raises(ValueError):
        run_policy(problem20(), "BILL_PERMUTED", cfg, {}, initial_order=lambda m: m[:-1])


def test_campaign_binds_original_inputs_with_one_seed(tmp_path, monkeypatch):
    from test_polish_policy_campaign import historical_fixture

    runner = load_script("run_slot_assignment")
    old = tmp_path / "old"
    historical_fixture(old)
    manifest = runner.prepare(tmp_path / "new", old)
    assert manifest["runs"] == 128 and len(manifest["blocks"]) == 32
    assert manifest["slot_policies"]["MAIN_SORTED"] == "sorted"
    for key in manifest["blocks"]:
        spec = json.loads((tmp_path / "new" / "specs" / (key + ".json")).read_text())
        assert spec["seed"] == 16501 and set(spec["order"]) == set(ARMS)
    runner.validate_inputs(tmp_path / "new")
    monkeypatch.setattr(runner, "load_instance", lambda path: problem20())
    check = runner.preflight(tmp_path / "new", manifest["blocks"][0])
    assert check["policies"]["bill4"]["fresh_exact_verified"] == 140
    pilot = runner.prepare(tmp_path / "pilot", old, pilot=True)
    assert pilot["runs"] == 16
    assert {
        json.loads((tmp_path / "pilot" / "specs" / (k + ".json")).read_text())["seconds"]
        for k in pilot["blocks"]
    } == set(runner.PILOT_SECONDS.values())


def test_dispatch_excludes_whole_cores_and_runs_longest_budgets_first(tmp_path):
    flat = load_script("slot_assignment_dispatch")
    topology = {c: ("0", str(c % 4)) for c in range(8)}
    assert flat.cpu_order(set(range(8)), topology, 6, frozenset({0})) == [1, 2, 3, 5, 6, 7]
    with pytest.raises(ValueError):
        flat.cpu_order(set(range(8)), topology, 7, frozenset({0}))
    (tmp_path / "specs").mkdir()
    for key, seconds in (("short", 300), ("long", 2400)):
        (tmp_path / "specs" / (key + ".json")).write_text(
            json.dumps(dict(order=["B", "A"], seconds=seconds))
        )
    manifest = dict(blocks=["short", "long"], arms=["A", "B"], source="s")
    fake_runner = SimpleNamespace(
        complete=lambda root, key, arm, source: key == "short" and arm == "A"
    )
    assert flat.pending_tasks(tmp_path, manifest, fake_runner) == [
        ("long", "B", 2400),
        ("long", "A", 2400),
        ("short", "B", 300),
    ]
