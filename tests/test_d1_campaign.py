"""Frozen source/probe artifacts, recovery and independent-base accounting."""

import json

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import d1_campaign as campaign
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash


@pytest.fixture
def frozen_campaign(problem, tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    root = tmp_path / "campaign"
    provenance = {"git_commit": "frozen", "git_dirty": False, "git_branch": "research/test"}
    monkeypatch.setattr(campaign, "git_metadata", lambda *args: provenance)
    config = Config(population=4, neighborhood=2, rl_steps=1, polish=False)
    monkeypatch.setattr(campaign, "load_config", lambda *_: config)
    monkeypatch.setattr(campaign, "generate_e15_pair", lambda *_: (problem, problem))
    manifest = campaign.prepare(root, repository, 0.3)
    return root, repository, manifest


def test_full_matrix_minimum_ten_hours_and_paired_initialization(frozen_campaign, monkeypatch):
    root, repository, pilot = frozen_campaign
    assert len(pilot["samples"]) == 4 and len(pilot["probes"]) == 12
    with pytest.raises(FileExistsError):
        campaign.prepare(root, repository, 0.3)
    full_root = root.parent / "full"
    full = campaign.prepare(full_root, repository)
    assert len(full["samples"]) == 80 and len(full["probes"]) == 240
    assert full["source_worker_hours"] == 60
    assert full["minimum_source_wall_hours"] == 10
    specs = [json.loads(p.read_text()) for p in (full_root / "specs").glob("*.json")]
    assert len({s["base_seed"] for s in specs}) == 8
    for spec in specs:
        counterpart = replace_key = spec["key"].replace("_H_", "_T_")
        if counterpart == spec["key"]:
            continue
        paired = json.loads((full_root / "specs" / f"{replace_key}.json").read_text())
        assert paired["initial_hash"] == spec["initial_hash"]
        assert paired["algorithm_seed"] == spec["algorithm_seed"]
    for p, sha in full["input_files"].items():
        assert file_hash(full_root / p) == sha


def test_end_to_end_source_and_probe_exact_hashes_and_base_count(frozen_campaign):
    root, _, manifest = frozen_campaign
    for key in manifest["samples"]:
        campaign.execute(root, "samples", key)
        assert campaign.validate_job(root, "samples", key) == (True, "complete")
        source = json.loads((root / "samples" / key / "summary.json").read_text())
        assert source["checkpoints"] == 3 and source["elapsed"] >= 0.3
        with pytest.raises(FileExistsError):
            campaign.execute(root, "samples", key)
    for key in manifest["probes"]:
        campaign.execute(root, "probes", key)
        assert campaign.validate_job(root, "probes", key) == (True, "complete")
        summary = json.loads((root / "probes" / key / "summary.json").read_text())
        assert summary["quartets"] + summary["skipped"] == 24
    report = campaign.analyse(root, manifest)
    assert report["independent_bases"] == 2  # Not twelve stage jobs or repeated sources.
    assert report["cases"] > 0
    assert (root / "analysis/paired_cases.csv").stat().st_size > 0
    sample_key = manifest["samples"][0]
    checkpoint = root / f"checkpoints/{sample_key}_s0.json"
    checkpoint.write_text("{}")
    assert not campaign.validate_job(root, "samples", sample_key)[0]
    assert not campaign.validate_job(root, "probes", f"{sample_key}_s0")[0]
    with pytest.raises(ValueError):
        campaign.analyse(root, manifest)


def test_interrupted_source_quarantines_dependent_probes_without_deletion(frozen_campaign):
    root, _, manifest = frozen_campaign
    key = manifest["samples"][0]
    attempt = root / "samples" / f".{key}.attempt-123"
    atomic_json(attempt / "partial.json", {"unfinished": True})
    atomic_json(root / f"checkpoints/{key}_s0.json", {"old_source": True})
    atomic_json(root / f"probes/{key}_s0/data.json", {"prior_probe": True})
    assert not campaign.recover_job(root, "samples", key)
    preserved = list((root / "quarantine").glob("*"))
    assert len(preserved) == 1
    assert (preserved[0] / attempt.name / "partial.json").exists()
    assert (preserved[0] / f"{key}_s0.json").exists()
    assert (preserved[0] / f"{key}_s0/data.json").exists()
    assert not (root / f"checkpoints/{key}_s0.json").exists()
    atomic_json(root / f"ops/failures/samples_{key}.json", {"reason": "crash"})
    with pytest.raises(RuntimeError):
        campaign.recover_job(root, "samples", key)


def test_invalid_inputs_and_frozen_provenance_are_rejected(frozen_campaign, monkeypatch):
    root, repository, manifest = frozen_campaign
    with pytest.raises(ValueError):
        campaign.prepare(root.parent / "bad", repository, -1)
    with pytest.raises(ValueError):
        campaign.execute(root, "bad_role", "k")
    monkeypatch.setattr(
        campaign, "git_metadata", lambda *_: {"git_commit": "wrong", "git_dirty": False}
    )
    with pytest.raises(ValueError):
        campaign.execute(root, "samples", manifest["samples"][0])
    monkeypatch.setattr(
        campaign, "git_metadata", lambda *_: {"git_commit": "wrong", "git_dirty": True}
    )
    with pytest.raises(ValueError):
        campaign.prepare(root.parent / "dirty", repository)
    with pytest.raises(ValueError):
        campaign.supervise(root, 0, 1)
    (root / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        campaign.supervise(root, 1, 1)


def test_supervisor_stop_and_leases_preserve_a_resumable_queue(frozen_campaign):
    root, _, manifest = frozen_campaign
    (root / "stop.request").write_text("user stop")
    with campaign.campaign_lease(root / "ops/other.lock"):
        with pytest.raises((OSError, BlockingIOError)):
            with campaign.campaign_lease(root / "ops/other.lock"):
                pass
    campaign.supervise(root, 1, 1)
    status = json.loads((root / "ops/status.json").read_text())
    assert status["state"] == "paused" and status["active"] == []
    assert status["completed_samples"] == 0
    assert len(manifest["samples"]) == 4
