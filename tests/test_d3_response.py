"""Pre-action separation, same-pool gain bound, shard pairing and tamper failures."""

import csv
import json
from dataclasses import replace

import pytest
from test_d2_opportunity import captured_panel, source_for

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import d2_campaign
from geo_llm_scheduler.experiments.d3_response import (
    analyse_response,
    analyse_response_joint,
    probe_response_panel,
    replacement_value,
    response_features,
    shuffle_opportunities,
)
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.moead.core import NormalizationContext


def test_gain_uses_each_incumbent_weight_and_only_two_slots(problem):
    x = source_for(problem)
    context = NormalizationContext((0, 0), tuple(v * 2 for v in x.evaluation.objectives))
    assert replacement_value(x, [x] * 4, [(0, 1), (1, 0), (0.5, 0.5), (0.2, 0.8)], context) == 0

    def point(flow, bill):
        return replace(
            x,
            evaluation=replace(
                x.evaluation, flow=flow, tou=bill, demand=(0.0,) * len(x.evaluation.demand)
            ),
        )

    context = NormalizationContext((0, 0), (10, 10))
    targets = [point(10, 4), point(4, 10), point(8, 8), point(6, 6)]
    weights = [(1, 0), (0, 1), (0.5, 0.5), (0.2, 0.8)]
    assert replacement_value(point(2, 2), targets, weights, context) == pytest.approx(0.4)
    assert replacement_value(point(2, 2), targets, [(0.5, 0.5)] * 4, context) != pytest.approx(0.4)
    with pytest.raises(ValueError):
        replacement_value(x, [x], [], context)


def test_all_features_precede_action_calls_and_shared_pool_bound(problem, monkeypatch):
    import geo_llm_scheduler.experiments.d3_response as d3

    original = d3.action_probe
    count = []
    prepared = []
    extract = d3.response_features

    def record_features(*args):
        assert not count
        value = extract(*args)
        prepared.append(value)
        return value

    def action(*args):
        assert len(prepared) == 1
        count.append(args[3])
        return original(*args)

    monkeypatch.setattr(d3, "response_features", record_features)
    monkeypatch.setattr(d3, "action_probe", action)
    panel = captured_panel(problem)
    result = probe_response_panel(problem, panel, Config(), 73)
    assert sorted(count) == list(range(1, 9))
    assert len(result["rows"]) == len(panel["sources"]) * 8
    assert result["exact"] <= 24
    assert all(r["gain"] <= r["post_gain"] + 1e-12 for r in result["rows"])
    assert all(p["raw"]["exact"] <= 3 for p in result["pools"])


def test_direction_interaction_changes_while_source_vector_is_fixed(problem):
    panel = captured_panel(problem)
    targets = [restore_candidate(s["candidate"]) for s in panel["sources"]]
    rows, _ = response_features(
        problem, restore_candidate(panel["mate"]), targets, panel["sources"], panel
    )
    assert len({tuple(r["source_opportunities"]) for r in rows}) == 1
    assert len({tuple(r["opportunities"]) for r in rows}) > 1


def test_shuffle_preserves_controls_and_pool_consistency():
    rows = [
        {
            "key": str(k),
            "panel": "s0_p0",
            "jobs": 50,
            "stage": 0,
            "preference": 0,
            "action": 1,
            "source_opportunities": [float(k + 1)],
            "weight_flow": w,
            "controls": [k, w],
            "opportunities": [(k + 1) * w, (k + 1) * w**2],
        }
        for k in range(6)
        for w in (0.2, 0.8)
    ]
    changed = shuffle_opportunities(rows, 55)
    assert [r["controls"] for r in changed] == [r["controls"] for r in rows]
    assert any(a["opportunities"] != b["opportunities"] for a, b in zip(rows, changed))
    for i in range(0, len(rows), 2):
        assert changed[i]["opportunities"][0] / 0.2 == pytest.approx(
            changed[i + 1]["opportunities"][0] / 0.8
        )


def test_d3_pair_frozen_split_and_checkpoint_tamper(tmp_path, problem, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs/experiments").mkdir(parents=True)
    (repo / d2_campaign.PROTOCOL).write_text("common sampling")
    (repo / "docs/experiments/d3_response.md").write_text("D3 protocol")
    monkeypatch.setattr(
        d2_campaign, "git_metadata", lambda *a: {"git_commit": "fixture", "git_dirty": False}
    )
    monkeypatch.setattr(
        d2_campaign, "load_config", lambda p: Config(population=6, neighborhood=3, method="plain")
    )
    monkeypatch.setattr(d2_campaign, "generate_e15_pair", lambda n, b: (problem, problem))
    root = tmp_path / "campaign"
    manifest = d2_campaign.prepare_opportunity(root, repo, 0.1, "D3", 1)
    assert manifest["node"] == 1 and len(manifest["keys"]) == 12
    assert set(manifest["development_bases"]) == {894002, 904002}
    key = manifest["keys"][0]
    manifest["keys"] = [key]
    monkeypatch.setattr(d2_campaign, "frozen_manifest", lambda p: manifest)
    d2_campaign.execute_opportunity_job(root, "samples", key)
    d2_campaign.execute_opportunity_job(root, "probes", key)
    report = analyse_response(root)
    assert report["pairs"] == 1 and report["panels"] == 9
    assert report["joint_quality"].startswith("UNVERIFIED")
    data = root / "probes" / key / "data.json"
    original = json.loads(data.read_text())
    assert all(not p["d1"]["rows"] for p in original["panels"])
    data.write_text("tampered")
    with pytest.raises(ValueError, match="artifact hash"):
        analyse_response(root)


def test_formal_shards_use_disjoint_bases_with_identical_global_split(
    tmp_path, problem, monkeypatch
):
    repo = tmp_path / "repo"
    (repo / "docs/experiments").mkdir(parents=True)
    (repo / d2_campaign.PROTOCOL).write_text("common sampling")
    (repo / "docs/experiments/d3_response.md").write_text("D3 protocol")
    monkeypatch.setattr(
        d2_campaign, "git_metadata", lambda *a: {"git_commit": "fixture", "git_dirty": False}
    )
    monkeypatch.setattr(
        d2_campaign, "load_config", lambda p: Config(population=6, neighborhood=3, method="plain")
    )
    monkeypatch.setattr(d2_campaign, "generate_e15_pair", lambda n, b: (problem, problem))
    shards = [
        d2_campaign.prepare_opportunity(tmp_path / f"node{n}", repo, None, "D3", n) for n in (0, 1)
    ]
    assert all(len(m["keys"]) == 144 and m["source_worker_hours"] == 108 for m in shards)
    assert not set(shards[0]["keys"]) & set(shards[1]["keys"])
    assert shards[0]["development_bases"] == shards[1]["development_bases"]
    assert len(shards[0]["development_bases"]) == 32 and len(shards[0]["heldout_bases"]) == 16


def test_joint_analysis_never_fits_heldout_labels_and_saves_decision_indices(tmp_path, monkeypatch):
    import geo_llm_scheduler.experiments.d3_response as d3

    roots = [tmp_path / f"node{n}" for n in (0, 1)]
    manifests = {}
    for node, root in enumerate(roots):
        (root / "specs").mkdir(parents=True)
        keys = []
        for base in range(node, 48, 2):
            key = str(base)
            keys.append(key)
            (root / "specs" / f"{key}.json").write_text(
                json.dumps(
                    {
                        "base_seed": base,
                        "jobs": 50 if base < 24 else 100,
                        "split": "development" if base < 32 else "heldout",
                    }
                )
            )
            (root / "probes" / key).mkdir(parents=True)
            rows = [
                {
                    "index": i,
                    "action": a,
                    "gain": i * (base + 1) / 1000,
                    "post_gain": (base + 1) / 1000,
                    "controls": [i],
                    "source_opportunities": [base + 1],
                    "opportunities": [(base + 1) * w, (base + 1) * w**2],
                    "weight_flow": w,
                    "distance": i,
                    "source_gain": i,
                }
                for a in range(1, 9)
                for i, w in enumerate((0.2, 0.8))
            ]
            (root / "probes" / key / "data.json").write_text(
                json.dumps({"panels": [{"d2": {"panel": "s0_p0", "rows": rows}}]})
            )
        manifests[root] = {
            "node": node,
            "pilot": False,
            "kind": "D3_response_v1",
            "source_hash": "fixture",
            "source_commit": "fixture",
            "keys": keys,
        }
    monkeypatch.setattr(d2_campaign, "frozen_manifest", lambda root: manifests[root])
    monkeypatch.setattr(d3, "analyse_response", lambda root: {"status": "SHARD_VALIDATED"})
    calls = []
    original = d3.ridge_predictions

    def fitted(train, test, opportunities):
        assert all(r["base"] < 32 for r in train)
        assert all(r["base"] >= 32 for r in test)
        calls.append(opportunities)
        return original(train, test, opportunities)

    monkeypatch.setattr(d3, "ridge_predictions", fitted)
    report = analyse_response_joint(roots, tmp_path / "analysis")
    assert len(calls) == 24 and report["heldout_bases"] == 16
    assert set(report["primary"]) == {"RESPONSE-BASE", "RESPONSE-RGAIN", "RESPONSE-SHUFFLED"}
    decisions = list(csv.DictReader((tmp_path / "analysis/heldout_decisions.csv").open()))
    assert len(decisions) == 128 and all(row["RGAIN_index"] == "1" for row in decisions)
    assert all(float(row["RESPONSE"]) <= float(row["POST_RGAIN"]) for row in decisions)
    with pytest.raises(ValueError, match="independent analysis"):
        analyse_response_joint(roots, roots[0] / "analysis")
    with pytest.raises(ValueError, match="identical source"):
        analyse_response_joint([roots[0], roots[0]], tmp_path / "bad")
