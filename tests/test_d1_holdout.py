"""Source isolation, artifact tampering, routed worker and clustered statistics."""

import json
from dataclasses import replace

import pytest

from geo_llm_scheduler.domain.models import Candidate, Genotype, Profile
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import d1_holdout as campaign
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d1_holdout_analysis import (
    analyse_holdout,
    holm_adjust,
    paired_inference,
)
from geo_llm_scheduler.experiments.d1_routing import GROUPS
from geo_llm_scheduler.experiments.p2_worker import REQUIRED
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.scheduling.ssgs import decode


@pytest.fixture
def imported_pilot(tmp_path, problem, monkeypatch):
    # Test IO/provenance on a small instance; actual 50/100-job construction
    # and resource envelopes are exercised by the separate real-source pilot.
    toy_keys = tuple(k.replace("n50_", "n2_").replace("n100_", "n2_") for k in campaign.PILOT_KEYS)
    monkeypatch.setattr(campaign, "PILOT_KEYS", toy_keys)
    p2 = tmp_path / "p2"
    (p2 / "inputs").mkdir(parents=True)
    (p2 / "specs").mkdir()
    (p2 / "runs").mkdir()
    for key in campaign.PILOT_KEYS:
        n = int(key.split("_")[0][1:])
        base = int(key.split("_")[1][1:])
        tariff = key.split("_")[2]
        seed = int(key.split("_")[3][1:])
        instance = replace(
            problem,
            jobs=tuple(
                replace(
                    problem.jobs[0],
                    name=f"j{i}",
                    prefill=Profile(1, 0.5, 2),
                    decode=Profile(2, 0.5, 2),
                )
                for i in range(n)
            ),
        )
        name = f"inputs/{key}.json"
        save_instance(instance, p2 / name)
        initial = f"inputs/{key}_initial.json"
        atomic_json(p2 / initial, [])
        population = []
        for index in range(4):
            ms = tuple(index % 3 for _ in range(2 * n))
            order = tuple(range(n)) * 2 if index < 3 else tuple(reversed(range(n))) * 2
            g = Genotype(ms, order)
            schedule = decode(instance, g)
            population.append(solution(Candidate(g, schedule, evaluate(instance, g, schedule))))
        config = {"population": 4, "seconds": 600 if n == 50 else 1200}
        spec = {
            "key": key,
            "arm": "F6",
            "jobs": n,
            "base_seed": base,
            "tariff": tariff,
            "algorithm_seed": seed,
            "config": config,
            "source_commit": "old",
            "source_hash": "old-hash",
            "instance": name,
            "instance_hash": file_hash(p2 / name),
            "initial": initial,
            "initial_hash": file_hash(p2 / initial),
        }
        atomic_json(p2 / "specs" / f"{key}.json", spec)
        output = p2 / "runs" / key
        output.mkdir()
        for artifact in REQUIRED:
            atomic_json(output / artifact, {})
        (output / "trace.jsonl").write_text("")
        atomic_json(output / "population.json", population)
        atomic_json(output / "archive.json", population)
        atomic_json(output / "config.json", config)
        atomic_json(
            output / "summary.json",
            {
                "canonical_source_hash": "old-hash",
                "git_commit": "old",
                "git_dirty": False,
                "instance_hash": spec["instance_hash"],
                "offspring_count": 1,
            },
        )
        atomic_json(
            output / "complete.json",
            {"input_hash": digest(spec), "files": {a: file_hash(output / a) for a in REQUIRED}},
        )
    atomic_json(p2 / "manifest.json", {"frozen": True})
    monkeypatch.setattr(
        campaign, "git_metadata", lambda: {"git_commit": "unit", "git_dirty": False}
    )
    root = tmp_path / "pilot"
    manifest = campaign.prepare_holdout(root, p2, pilot=True)
    manifest["quota_seconds"]["2"] = 1.0
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"manifest.json": file_hash(root / "manifest.json")})
    return root, p2, manifest


def test_import_worker_exact_and_paired_route_count(imported_pilot):
    root, p2, manifest = imported_pilot
    assert len(manifest["keys"]) == 4 and manifest["cases_planned"] == 96
    assert set(manifest["bases"]).issubset(campaign.DEVELOPMENT_BASES)
    assert set(manifest["bases"]).isdisjoint(campaign.HOLDOUT_BASES)
    for key in manifest["keys"]:
        summary = campaign.execute_holdout_job(root, key)
        assert summary["cases"] + summary["skipped"] == 24
        assert campaign.validate_holdout_job(root, key) == (True, "complete")
        with pytest.raises(FileExistsError):
            campaign.execute_holdout_job(root, key)
    assert all(file_hash(p2 / n) == h for n, h in manifest["original_p2_files"].items())
    result = campaign.run_holdout(root, 1)
    assert result["status"] == "complete" and result["completed"] == 4
    with pytest.raises(ValueError):
        analyse_holdout(root, 100)
    key = manifest["keys"][0]
    (root / "jobs" / key / "data.json").write_text("{}")
    assert campaign.validate_holdout_job(root, key)[1] == "artifact hash mismatch"
    with pytest.raises(ValueError):
        campaign.run_holdout(root, 1)


def test_import_rejects_bad_matrix_hash_and_dirty_provenance(imported_pilot, monkeypatch):
    root, p2, manifest = imported_pilot
    with pytest.raises(FileExistsError):
        campaign.prepare_holdout(root, p2, True)
    with pytest.raises(ValueError):
        campaign.prepare_holdout(root.parent / "formal", p2)
    monkeypatch.setattr(campaign, "git_metadata", lambda: {"git_commit": "unit", "git_dirty": True})
    with pytest.raises(ValueError):
        campaign.prepare_holdout(root.parent / "dirty", p2, True)
    with pytest.raises(ValueError):
        campaign.execute_holdout_job(root, manifest["keys"][0])


def test_queue_stop_failure_and_resource_bounds(imported_pilot):
    root, _, manifest = imported_pilot
    for workers in (0, 9):
        with pytest.raises(ValueError):
            campaign.run_holdout(root, workers)
    (root / "stop.request").write_text("pause before dispatch")
    assert campaign.run_holdout(root, 1)["status"] == "paused"
    assert not list((root / "jobs").iterdir())
    key = manifest["keys"][0]
    atomic_json(root / "ops" / f"failure_{key}.json", {"preserved": True})
    with pytest.raises(ValueError):
        campaign.run_holdout(root, 1)


def test_base_inference_exact_signs_and_holm_replay():
    positive = paired_inference([0.01] * 16, [50] * 8 + [100] * 8, 200)
    assert positive["n_bases"] == 16 and positive["p_exact"] == 2 / 65536
    assert positive["ci95"] == pytest.approx([0.01, 0.01])
    assert paired_inference([0] * 16, [50] * 8 + [100] * 8, 200)["p_exact"] == 1
    assert holm_adjust([0.01, 0.04, 0.2]) == pytest.approx([0.03, 0.08, 0.2])
    assert paired_inference([0.02, -0.01], [50, 100], 200) == paired_inference(
        [0.02, -0.01], [50, 100], 200
    )
    for effects, sizes in (([], []), ([float("nan")], [50]), ([1], [])):
        with pytest.raises(ValueError):
            paired_inference(effects, sizes)
    with pytest.raises(ValueError):
        holm_adjust([1.1])


def test_supervisor_records_failure_without_retrying_or_dispatching_more(
    imported_pilot, monkeypatch
):
    root, _, _ = imported_pilot
    launches = []

    class FailedWorker:
        pid = 1234

        def __init__(self, command, **kwargs):
            launches.append(command)

        def poll(self):
            return 1

    monkeypatch.setattr(campaign.subprocess, "Popen", FailedWorker)
    with pytest.raises(RuntimeError):
        campaign.run_holdout(root, 1)
    assert len(launches) == 1
    assert json.loads((root / "ops/exit.json").read_text())["status"] == "failed"
    assert len(list((root / "ops").glob("failure_*.json"))) == 1


def test_analysis_uses_bases_instead_of_duplicated_cases(tmp_path, monkeypatch):
    import geo_llm_scheduler.experiments.d1_holdout_analysis as analysis

    monkeypatch.setattr(analysis, "validate_holdout_job", lambda *_: (True, "complete"))
    bases = sorted(campaign.HOLDOUT_BASES)
    keys = []
    for index, base in enumerate(bases):
        for replica in range(6):
            key = f"{base}_{replica}"
            keys.append(key)
            costs = {
                g: {
                    "relative_gain": gain,
                    "seconds": 0.01,
                    "exact": 3,
                    "feasible": 3,
                    "on_time_exact": 3,
                    "extraction_seconds": 0.001,
                    "projection_seconds": 0,
                    "construction_seconds": 0.008,
                    "exact_seconds": 0.001,
                    "failed_projection": 0,
                    "overshoot_seconds": 0,
                    "route": "A7",
                    "objectives": [200 * (1 - gain), 20 * (1 - gain)],
                    "intent_nonzero": 0,
                    "proposals": [],
                }
                for g, gain in zip(GROUPS, (0.02, 0.03 + index * 0.0001, 0.01))
            }
            row = {
                "sample_index": 0,
                "kind": "OS",
                "magnitude": 1,
                "preference": 0,
                "base_scalar": 1,
                "shared_seconds": 0.002,
                "groups": costs,
                "context": {"ideal": [0, 0], "maximum": [200, 20]},
                "weight": [0.5, 0.5],
                "base_objectives": [200, 20],
            }
            atomic_json(tmp_path / "jobs" / key / "data.json", {"rows": [row, row]})
            atomic_json(
                tmp_path / "jobs" / key / "summary.json",
                {
                    "base_seed": base,
                    "jobs": 50 if index < 8 else 100,
                    "tariff": "H",
                    "skipped": 0,
                    "source_snapshots": 1,
                    "planning_seconds": 0.01,
                    "elapsed": 0.1,
                    "rss_peak_gib": 0.01,
                },
            )
    (tmp_path / "analysis").mkdir()
    atomic_json(
        tmp_path / "manifest.json",
        {"pilot": False, "keys": keys, "bases": bases, "scope": "test", "normalization": "fixed"},
    )
    report = analyse_holdout(tmp_path, 200)
    assert report["n_bases"] == 16 and report["cases"] == 192
    assert report["primary"]["CONDITIONAL-ALWAYS_A7"]["n_bases"] == 16
    assert report["passes_quality_screen"]
    assert (tmp_path / "analysis/report.md").exists()
