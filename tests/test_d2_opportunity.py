"""Prospective feature isolation, matched exact labels, recovery and holdout tests."""

import json
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments import d1_routing, d2_campaign
from geo_llm_scheduler.experiments.d2_analysis import (
    analyse_opportunity,
    paired_summary,
    ridge_predictions,
    shuffled_features,
)
from geo_llm_scheduler.experiments.d2_observation import (
    CONTROL_NAMES,
    OPPORTUNITY_NAMES,
    OpportunityObserver,
    control_features,
    opportunity_features,
)
from geo_llm_scheduler.experiments.d2_probe import action_probe, d1_cost_twins, probe_panel
from geo_llm_scheduler.experiments.p1_observation import SemanticRecorder, restore_candidate
from geo_llm_scheduler.moead.core import NormalizationContext


def source_for(problem):
    gateway = EvaluationGateway(problem, Archive())
    return gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))


def captured_panel(problem):
    source = source_for(problem)
    observer = OpportunityObserver(problem, 10, 12, 3)
    gateway = EvaluationGateway(problem, Archive())
    gateway.ideal = source.evaluation.objectives
    values = {
        "population": [source] * 6,
        "gateway": gateway,
        "stagnation": [0] * 6,
        "elapsed": 1,
        "subproblem": 0,
        "generation": 1,
    }
    observer("before_offspring", values)
    return observer.export()["panels"][0]


def test_features_are_pre_action_finite_and_empty_zero(problem, monkeypatch):
    source = source_for(problem)
    monkeypatch.setattr(
        "geo_llm_scheduler.experiments.d2_probe.evaluate",
        lambda *a: pytest.fail("feature extraction evaluated future candidates"),
    )
    features = opportunity_features(problem, source)
    assert len(features) == len(OPPORTUNITY_NAMES) and all(v >= 0 for v in features)
    panel = captured_panel(problem)
    controls = control_features(problem, source, source, panel, panel["sources"][0])
    assert len(controls) == len(CONTROL_NAMES)
    empty = ProblemInstance((), (), ())
    assert opportunity_features(empty, source) == (0,) * len(OPPORTUNITY_NAMES)


def test_observer_is_bounded_causal_and_does_not_advance_algorithm_rng(problem):
    config = Config(
        population=6,
        neighborhood=3,
        generations=3,
        seed=31,
        method="full",
        trigger_mode="always",
        enabled_operators=(1,),
    )
    signatures = []
    for collect in (False, True):
        recorder = SemanticRecorder()
        observer = OpportunityObserver(problem, 100, 8, 3)

        def callback(event, values):
            recorder(event, values)
            if collect:
                observer(event, values)

        result = run(problem, config, retain_trace=False, observer=callback)
        signatures.append(recorder.signature(result))
        assert len(observer.panels) <= 9
    assert signatures[0] == signatures[1]
    panel = captured_panel(problem)
    assert all(s["history_calls"] == 0 and s["history"] == 0 for s in panel["sources"])
    assert panel["mate"]["evaluation"]["feasible"]


@pytest.mark.parametrize("action", range(9))
def test_action_budget_and_all_outputs_exact(problem, action):
    source = source_for(problem)
    context = NormalizationContext((0, 0), (10000, 10000))
    result = action_probe(problem, source, source, action, Config(), (0.5, 0.5), context, 42)
    assert 0 <= result["exact"] <= 3
    assert result["audit_checks"] == result["exact"] + 1
    assert restore_candidate(result["retained"]).evaluation.feasible
    again = action_probe(problem, source, source, action, Config(), (0.5, 0.5), context, 42)
    assert result["retained"] == again["retained"] and result["outputs"] == again["outputs"]


def test_panel_keeps_degenerate_sources_and_uses_common_comparison(problem):
    panel = captured_panel(problem)
    result = probe_panel(problem, panel, Config(), 31)
    assert result["unique_sources"] == 1 and result["pool_size"] == 3
    assert len(result["rows"]) == 27
    for action in range(9):
        rows = [r for r in result["rows"] if r["action"] == action]
        assert len({json.dumps(r["comparison_context"]) for r in rows}) == 1
        assert len({r["gain"] for r in rows}) == 1


def test_cheap_d1_guard_skips_incompatible_intent_and_preserves_outputs(problem, monkeypatch):
    source = source_for(problem)
    target = replace(source.genotype, ms=(1, 1, 1, 1))
    calls = []
    original = d1_routing.extract_intent

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(d1_routing, "extract_intent", counted)
    common = (
        problem,
        source,
        target,
        (0.5, 0.5),
        NormalizationContext((0, 0), (10000, 10000)),
        2,
        False,
        30,
    )
    reference = d1_routing.routing_probe(*common, policy_groups=("CONDITIONAL",))
    fast = d1_routing.routing_probe(
        *common, precheck_structure=True, policy_groups=("CONDITIONAL",)
    )
    assert len(calls) == 1
    a, b = (r["groups"]["CONDITIONAL"] for r in (reference, fast))
    assert (a["route"], a["starts"], a["objectives"], a["exact"]) == (
        b["route"],
        b["starts"],
        b["objectives"],
        b["exact"],
    )
    with pytest.raises(ValueError):
        d1_routing.routing_probe(*common, policy_groups=("unknown",))
    twins = d1_cost_twins(problem, captured_panel(problem), Config(), 31)
    assert len(twins["rows"]) == 4 and all(r["same"] for r in twins["rows"])


def test_ridge_training_isolation_shuffle_capacity_and_exact_statistics():
    train = [
        {
            "controls": [float(i)],
            "opportunities": [float(i % 2)],
            "gain": float(i),
            "key": "train",
            "panel": "s0",
            "action": 0,
        }
        for i in range(6)
    ]
    test = [{**train[0], "controls": [8.0], "gain": -999, "key": "test"}]
    expected = ridge_predictions(train, test, True)
    assert expected == ridge_predictions(train, [{**test[0], "gain": 999}], True)
    shuffled = shuffled_features(train, 4)
    assert sorted(r["opportunities"] for r in shuffled) == sorted(r["opportunities"] for r in train)
    assert [r["controls"] for r in shuffled] == [r["controls"] for r in train]
    assert paired_summary([1.0] * 8, 31)["p"] == 2 / 256
    assert paired_summary([], 31)["bases"] == 0


def test_preparation_worker_hashes_and_interrupted_recovery(tmp_path, problem, monkeypatch):
    repository = tmp_path / "repository"
    (repository / "docs/experiments").mkdir(parents=True)
    (repository / d2_campaign.PROTOCOL).write_text("protocol", encoding="utf-8")
    # Freeze an engineering fixture, never a scientific 50/100-job result.
    monkeypatch.setattr(
        d2_campaign, "git_metadata", lambda *a: {"git_commit": "test-sha", "git_dirty": False}
    )
    monkeypatch.setattr(
        d2_campaign, "load_config", lambda p: Config(population=6, neighborhood=3, method="plain")
    )
    monkeypatch.setattr(d2_campaign, "generate_e15_pair", lambda n, b: (problem, problem))
    root = tmp_path / "campaign"
    manifest = d2_campaign.prepare_opportunity(root, repository, 0.08)
    assert len(manifest["keys"]) == 12 and manifest["heldout_bases"] == []
    # This fixture uses its own protocol repository.
    original = d2_campaign.frozen_manifest

    def frozen(_):
        return manifest

    monkeypatch.setattr(d2_campaign, "frozen_manifest", frozen)
    key = manifest["keys"][0]
    d2_campaign.execute_opportunity_job(root, "samples", key)
    d2_campaign.execute_opportunity_job(root, "probes", key)
    assert d2_campaign.validate_opportunity_job(root, "samples", key)[0]
    assert d2_campaign.validate_opportunity_job(root, "probes", key)[0]
    incomplete = root / "samples/.unfinished.attempt-999"
    incomplete.mkdir()
    result = d2_campaign.recover_interrupted(root)
    assert result["quarantined"] and not incomplete.exists()
    unfinished_key = manifest["keys"][1]
    d2_campaign.execute_opportunity_job(root, "samples", unfinished_key)
    migrated = d2_campaign.recover_interrupted(root, migrate=True)
    assert migrated["migrate"] and migrated["quarantined"]
    assert not (root / "samples" / unfinished_key).exists()
    assert d2_campaign.validate_opportunity_job(root, "samples", key)[0]
    assert d2_campaign.validate_opportunity_job(root, "probes", key)[0]
    (root / "samples" / key / "data.json").write_text("tampered")
    assert not d2_campaign.validate_opportunity_job(root, "samples", key)[0]
    assert original is not frozen


def test_crashes_are_preserved_and_not_automatically_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(d2_campaign, "frozen_manifest", lambda r: {"keys": []})
    root = tmp_path / "campaign"
    (root / "ops/failures").mkdir(parents=True)
    (root / "ops/failures/crash.json").write_text("{}")
    with pytest.raises(ValueError, match="crashes"):
        d2_campaign.recover_interrupted(root)


def test_analysis_failure_records_execution_without_false_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(d2_campaign, "frozen_manifest", lambda r: {"keys": []})
    monkeypatch.setattr(d2_campaign, "keep_awake", lambda value: None)

    def broken_analysis(root):
        raise ValueError("analysis evidence failure")

    monkeypatch.setattr(
        "geo_llm_scheduler.experiments.d2_analysis.analyse_opportunity", broken_analysis
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "manifest.json").write_text("{}")
    result = d2_campaign.run_opportunity(tmp_path, 1, 1)
    assert result["state"] == "failed" and result["execution_complete"]
    assert not result["analysis_complete"] and result["active"] == []
    evidence = json.loads((tmp_path / "ops/failures/analysis.json").read_text())
    assert "analysis evidence failure" in evidence["traceback"]
    with pytest.raises(ValueError, match="failure evidence"):
        d2_campaign.run_opportunity(tmp_path, 1, 1)


def test_complete_analysis_heldout_selection_and_inventory(tmp_path, problem, monkeypatch):
    manifest = {
        "keys": ["development", "heldout"],
        "pilot": False,
        "source_commit": "test",
        "source_hash": "test-hash",
        "development_bases": [1],
        "heldout_bases": [2],
        "scope": "fixture screening",
    }
    monkeypatch.setattr(d2_campaign, "frozen_manifest", lambda root: manifest)
    monkeypatch.setattr(d2_campaign, "validate_opportunity_job", lambda *a: (True, "fixture"))
    panel = captured_panel(problem)
    record = {
        "d2": probe_panel(problem, panel, Config(), 41),
        "d1": d1_cost_twins(problem, panel, Config(), 41),
    }
    (tmp_path / "specs").mkdir()
    (tmp_path / "analysis").mkdir()
    for base, key in enumerate(manifest["keys"], 1):
        (tmp_path / "specs" / f"{key}.json").write_text(
            json.dumps(
                {
                    "base_seed": base,
                    "split": key,
                    "jobs": 2,
                }
            )
        )
        path = tmp_path / "probes" / key
        path.mkdir(parents=True)
        (path / "data.json").write_text(json.dumps({"panels": [record]}))
    # Windows byte locking makes the active lease unreadable. Analysis excludes
    # this transient control file while retaining all immutable job evidence.
    with d2_campaign.campaign_lease(tmp_path / "ops/supervisor.lock"):
        report = analyse_opportunity(tmp_path)
    assert report["rows"] == 54 and report["passes_primary_screen"] is False
    assert report["heldout_bases"] == [2]
    assert set(report["primary_contrasts"]) == {"BASE", "SHUFFLED"}
    assert len(report["heldout_action_selection"]) == 9
    assert report["panel_diagnostics"]["panels"] == 2
    assert set(report["action_diagnostics"]) == {str(a) for a in range(9)}
    assert (tmp_path / "analysis/heldout_decisions.csv").exists()
    inventory = json.loads((tmp_path / "analysis/files_manifest.json").read_text())
    assert inventory and "ops/supervisor.lock" not in {r["path"] for r in inventory}
