"""Initialize honestly, isolate components and avoid false overnight guarantees."""

import importlib.util
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.engine.trajectory import improve
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import component_random as study
from geo_llm_scheduler.experiments.comparison13 import arm_config as old_config
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager


def test_roster_and_initialization_are_explicit_and_no_handicap_parameters():
    assert len(study.ARMS) == len(set(study.ARMS)) == 26
    for arm in study.ARMS:
        cfg = study.arm_config(arm, 19, 300)
        assert cfg.population == 100 and cfg.neighborhood == 20
        assert cfg.crossover_probability == Config().crossover_probability == 0.9
        assert cfg.mutation_probability == Config().mutation_probability == 0.2
        if arm in study.EXTERNAL_ARMS:
            assert study.initializer_policy(arm) == "random"
            assert cfg.method != "full" and not cfg.polish
            assert cfg.replacement_policy == "birth" and cfg.budget_policy == "fixed"
    assert study.initializer_policy("MIXED_MOEAD") == "mixed"
    assert study.initializer_policy("BILL_MOEAD") == "bill4"
    with pytest.raises(ValueError):
        study.arm_config("missing", 1, 10)


def test_all_eight_single_operators_and_hierarchical_controls_are_isolated():
    for action in range(1, 9):
        cfg = study.arm_config(f"ONLY_A{action}", 1, 300)
        assert cfg.enabled_operators == (action,)
        assert cfg.fixed_budget == cfg.rl_steps == 1
        assert cfg.controller == "random" and not cfg.polish
        assert cfg.a3_region_policy == ("tariff" if action == 3 else "load")
    replacements = study.arm_config("RPERM_ONLY", 1, 300)
    assert replacements.replacement_policy == "permuted"
    assert replacements.trigger_mode == "fixed" and replacements.fixed_ls_probability == 0
    left = study.arm_config("POOL_RANDOM_B6_L1", 1, 300)
    right = study.arm_config("POOL_Q_B6_L1", 1, 300)
    assert asdict(replace(left, controller="qlearning")) == asdict(right)
    assert study.arm_config("POOL_Q_B6_L5", 1, 300).rl_steps == 5
    strict = study.arm_config("POOL_Q_STRICT_B6_L5", 1, 300)
    assert (strict.trigger_mode, strict.trigger_delta) == ("strict", 0)
    assert study.arm_config("POOL_Q_SEVERITY_L5", 1, 300).budget_policy == "severity_v2"
    assert study.arm_config("POOL_Q_COVERAGE_L5", 1, 300).budget_policy == "coverage_v2"
    assert study.arm_config("FULL_CURRENT_I4", 1, 300).polish
    assert not study.arm_config("FULL_STRICT_NO_POLISH_I4", 1, 300).polish
    # The full method is present intact, not a single-component substitute.
    assert asdict(study.arm_config("FULL_CURRENT_I4", 1, 300)) == asdict(
        old_config("M11_I4", 1, 300)
    )


def test_external_random_initializer_is_identical_price_blind_and_not_ours(monkeypatch):
    problem = case_problem(6, 2, 4, 73, "arrival", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("External baselines cannot use designed initialization")

    monkeypatch.setattr(study, "traced_original", forbidden)
    monkeypatch.setattr(study, "initialization_factory", forbidden)
    signatures = []
    for arm in study.EXTERNAL_ARMS:
        cfg = replace(study.arm_config(arm, 19, 300), population=8, neighborhood=4)
        gateway = EvaluationGateway(problem, Archive())
        evidence = {}
        members = study.charged_initializer(problem, cfg, arm, evidence)(gateway, RNGManager(19))
        assert len({c.genotype for c in members}) == 8
        assert gateway.counts["exact"] == gateway.archive.attempts == 8
        assert all(c.origin == "pure_random_initialization" for c in members)
        assert all(c.evaluation == evaluate(problem, c.genotype, c.schedule) for c in members)
        signatures.append(([c.genotype for c in members], evidence["initial_population_sha256"]))
    assert all(row == signatures[0] for row in signatures)
    changed = replace(problem, regions=tuple(replace(r, demand_rate=999) for r in problem.regions))
    members = study.charged_initializer(changed, cfg, arm, {})(
        EvaluationGateway(changed, Archive()), RNGManager(19)
    )
    assert [c.genotype for c in members] == signatures[0][0]


def test_random_initializer_fails_at_explicit_draw_limit_without_heuristic_fallback():
    problem = case_problem(6, 2, 4, 73, "arrival", 0)
    cfg = Config(population=8, neighborhood=4, initialization_attempts=1)
    gateway = EvaluationGateway(problem, Archive())
    with pytest.raises(ValueError, match="attempt limit"):
        study.charged_initializer(problem, cfg, "EXT_MOEAD", {})(gateway, RNGManager(1))
    assert gateway.counts["exact"] == gateway.archive.attempts == 0


def test_mixed_and_bill_are_separate_and_charge_all_unselected_candidates():
    problem = case_problem(12, 3, 12, 20, "arrival", 0)
    for arm, exact in (("MIXED_MOEAD", 100), ("BILL_MOEAD", 140)):
        cfg = study.arm_config(arm, 33, 300)
        gateway = EvaluationGateway(problem, Archive())
        evidence = {}
        members = study.charged_initializer(problem, cfg, arm, evidence)(gateway, RNGManager(33))
        assert len(members) == len({c.genotype for c in members}) == 100
        assert gateway.counts["exact"] == gateway.archive.attempts == exact
        if arm == "MIXED_MOEAD":
            assert Counter(c.origin for c in members) == {
                f"initial_mode_{i}": 10 for i in range(10)
            }
            assert evidence["random_draw_attempts"] is None
        else:
            assert sum(c.origin == "bill_seed_BILL" for c in members) == 10
            assert evidence["trials"] == 4


@pytest.mark.parametrize("arm", study.EXTERNAL_ARMS)
def test_each_external_engine_runs_random_population_without_memetic_work(arm):
    problem = case_problem(6, 2, 4, 73, "arrival", 0)
    cfg = replace(
        study.arm_config(arm, 19, 300), population=8, neighborhood=4, generations=1, seconds=None
    )
    evidence = {}
    events = []
    result = study.run_arm(
        problem, cfg, arm, evidence, observer=lambda event, data: events.append(event)
    )
    assert result.offspring_count == 8
    assert events.count("initial_ready") == 1
    assert evidence["policy"] == "random"
    assert result.gateway.counts["polish_calls"] == 0
    assert not sum(map(sum, result.controller.selections))
    assert all(c.evaluation.feasible for c in result.population + result.archive.members)


def test_polish_only_callback_cannot_select_an_action_or_update_q():
    problem = case_problem(6, 2, 4, 73, "arrival", 0)
    cfg = Config(
        population=6, neighborhood=2, generations=1, seed=19, method="full", trigger_mode="always"
    )
    result = run(problem, cfg, trajectory_override=study.polish_only)
    assert result.offspring_count == result.gateway.counts["polish_calls"] == 6
    assert not sum(map(sum, result.controller.selections))
    assert all(value == 0 for row in result.controller.q for value in row)
    assert not any(k.startswith("proposals:A") for k in result.gateway.counts)


def test_omitted_research_callback_retains_default_trace_counts_q_and_rng():
    problem = case_problem(6, 2, 4, 73, "arrival", 0)
    cfg = Config(
        population=6, neighborhood=2, generations=1, seed=19, method="full", trigger_mode="always"
    )
    states = [[], []]

    def observer(index):
        def observe(event, data):
            if event == "offspring":
                states[index].append(data["streams"].snapshot())

        return observe

    left = run(problem, cfg, observer=observer(0))
    right = run(problem, cfg, trajectory_override=improve, observer=observer(1))
    assert [solution(c) for c in left.population] == [solution(c) for c in right.population]
    assert [solution(c) for c in left.archive.members] == [
        solution(c) for c in right.archive.members
    ]
    assert left.gateway.counts == right.gateway.counts and left.controller.q == right.controller.q
    assert states[0] == states[1] and states[0]


def test_matrix_has_independent_parents_not_independent_variants_and_model_is_not_launch():
    specs = study.study_specs()
    assert len(specs) == len({s["old_key"] for s in specs}) == 40
    assert len({s["base_seed"] for s in specs}) == 20
    assert [s["n"] for s in specs] == [20, 50, 100, 200] * 10
    assert not {s["base_seed"] for s in specs} & {
        s["base_seed"] for s in study.study_specs(reserve=True)
    }
    for base in {s["base_seed"] for s in specs}:
        assert {(s["variant"], s["seed"]) for s in specs if s["base_seed"] == base} == {
            (0, 16101),
            (1, 16101),
        }
    plan = study.planning_summary()
    assert plan["status"] == "planning_only_not_dispatched" and plan["runs"] == 1040
    assert plan["production_solver_hours"] == 325
    assert plan["modeled_completion_hours"] > plan["ideal_lower_bound_hours"]
    assert plan["memory_cap_bytes"] == 500_000_000_000
    for workers, overhead in ((0, 30), (48, -1), (48, float("nan"))):
        with pytest.raises(ValueError):
            study.planning_summary(workers, overhead)


def test_even_williams_templates_balance_positions_and_directed_predecessors():
    rows = [study.balanced_order(i) for i in range(len(study.ARMS))]
    assert all(set(row) == set(study.ARMS) for row in rows)
    for i in range(len(study.ARMS)):
        assert Counter(row[i] for row in rows) == dict.fromkeys(study.ARMS, 1)
    adjacent = Counter((a, b) for row in rows for a, b in zip(row, row[1:]))
    assert len(adjacent) == 26 * 25 and set(adjacent.values()) == {1}


def test_runner_copies_bound_old_input_and_seals_matrix_preflight_and_queue(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_component_random.py"
    loader = importlib.util.spec_from_file_location("component_runner_test", script)
    module = importlib.util.module_from_spec(loader)
    assert loader.loader
    loader.loader.exec_module(module)
    old = tmp_path / "historical"
    (old / "inputs").mkdir(parents=True)
    (old / "specs").mkdir()
    s = dict(
        old_key="tiny",
        n=12,
        regions=3,
        machines=12,
        pressure="arrival",
        variant=0,
        base_seed=20,
        seed=33,
        seconds=60,
    )
    case = old / "inputs/original.json"
    save_instance(case_problem(12, 3, 12, 20, "arrival", 0), case)
    (old / "specs/tiny.json").write_text(
        json.dumps(dict(s, input="inputs/original.json", input_sha256=module.sha(case)))
    )
    monkeypatch.setattr(module, "study_specs", lambda: [s.copy()])
    root = tmp_path / "new"
    manifest = module.prepare(root, old)
    assert module.validate_inputs(root) == manifest
    assert (root / "inputs/original.json").read_bytes() == case.read_bytes()
    assert module.queue(root, manifest) == [("tiny", arm) for arm in study.balanced_order(0)]
    assert not module.complete(root, "tiny", study.ARMS[0], manifest["source"])
    with pytest.raises(FileExistsError):
        module.prepare(root, old)
    preflight = module.preflight(root, "tiny")
    assert {k: v["fresh_exact_verified"] for k, v in preflight["policies"].items()} == {
        "random": 100,
        "mixed": 100,
        "bill4": 140,
    }
    (root / "inputs/original.json").write_text("{}")
    with pytest.raises(AssertionError):
        module.validate_inputs(root)
    with pytest.raises(AssertionError):
        module.preflight(root, "tiny")


def test_plan_queue_model_reproduces_exact_runner_order(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_component_random.py"
    loader = importlib.util.spec_from_file_location("component_queue_test", script)
    module = importlib.util.module_from_spec(loader)
    assert loader.loader
    loader.loader.exec_module(module)
    (tmp_path / "specs").mkdir()
    specs = study.study_specs()
    keys = []
    for i, s in enumerate(specs):
        key = s["old_key"]
        keys.append(key)
        (tmp_path / "specs" / (key + ".json")).write_text(
            json.dumps(dict(s, key=key, order=study.balanced_order(i)))
        )
    queue = module.queue(tmp_path, {"blocks": keys})
    assert len(queue) == len(set(queue)) == 1040
    assert [specs[keys.index(k)]["n"] for k, a in queue] == [20, 50, 100, 200] * 260
    assert all({a for k, a in queue if k == key} == set(study.ARMS) for key in keys)
