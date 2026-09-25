"""Recovery and admission checks on the durable campaign supervisor."""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.model import CampaignSettings, JobSpec
from geo_llm_scheduler.experiments.campaign.scheduler import Supervisor
from geo_llm_scheduler.experiments.campaign.support import atomic_json


def make_supervisor(repo: Path, root: Path, retries: int = 1) -> Supervisor:
    """Create an isolated checkpoint without launching the experiment DAG."""
    settings = replace(CampaignSettings(), max_job_retries=retries)
    atomic_json(
        root / "launch.json",
        {"deadline_epoch": time.time() + 120, "settings": asdict(settings)},
    )
    return Supervisor(repo, root)


def make_spec() -> JobSpec:
    """Return a stable job identity sufficient for checkpoint transition tests."""
    return JobSpec(
        "e11_trigger",
        "fixed",
        71,
        101,
        {"seconds": 1},
        "instance.json",
        "instance-sha",
        "initial.json",
        "initial-sha",
        "source",
        "commit",
    )


def test_dead_worker_is_retried_then_permanent_and_incomplete_artifact_quarantined(
    tmp_path, monkeypatch
):
    supervisor = make_supervisor(tmp_path, tmp_path / "campaign")
    try:
        spec = make_spec()
        supervisor.store.add(spec)
        output = supervisor.root / spec.stage / "runs" / spec.key
        output.mkdir(parents=True)
        (output / "summary.json").write_text("{broken", encoding="utf-8")
        supervisor.store.transition(spec.key, "RUNNING", pid=999999, slot=0, attempt=1)
        monkeypatch.setattr(
            "geo_llm_scheduler.experiments.campaign.scheduler.alive", lambda _pid: False
        )
        supervisor._reconcile()
        first = supervisor.store.get(spec.key)
        assert first["state"] == "FAILED_RETRYABLE"
        assert not output.exists()
        assert list((supervisor.root / "quarantine").iterdir())
        supervisor.store.transition(spec.key, "RUNNING", pid=999999, slot=0, attempt=2)
        supervisor._reconcile()
        assert supervisor.store.get(spec.key)["state"] == "FAILED_PERMANENT"
    finally:
        supervisor.store.close()


def test_valid_completed_job_recovered_without_reexecution(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path, tmp_path / "campaign")
    try:
        spec = make_spec()
        supervisor.store.add(spec)
        output = supervisor.root / spec.stage / "runs" / spec.key
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps({"elapsed": 1.5}))
        monkeypatch.setattr(
            "geo_llm_scheduler.experiments.campaign.scheduler.validate_result",
            lambda _output, _spec: (True, "complete"),
        )
        supervisor._reconcile()
        recovered = supervisor.store.get(spec.key)
        assert recovered["state"] == "SUCCEEDED"
        assert recovered["attempt"] == 0
        assert spec.key in supervisor.verified_successes
        supervisor._reconcile()
        assert supervisor.store.get(spec.key)["attempt"] == 0
    finally:
        supervisor.store.close()


def test_nonzero_worker_exit_rejects_even_complete_artifact(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path, tmp_path / "campaign")
    try:
        spec = make_spec()
        supervisor.store.add(spec)
        supervisor.store.transition(spec.key, "RUNNING", pid=999999, slot=0, attempt=1)
        output = supervisor.root / spec.stage / "runs" / spec.key
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps({"elapsed": 1.5}))
        monkeypatch.setattr(
            "geo_llm_scheduler.experiments.campaign.scheduler.validate_result",
            lambda _output, _spec: (True, "complete"),
        )

        class ExitedWorker:
            def wait(self, timeout):
                assert timeout == 5
                return 1

        supervisor.children[spec.key] = ExitedWorker()
        supervisor._reconcile()
        assert supervisor.store.get(spec.key)["state"] == "FAILED_RETRYABLE"
        assert not output.exists()
    finally:
        supervisor.store.close()


def test_deadline_and_low_disk_never_launch_new_worker(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path, tmp_path / "campaign")
    try:
        spec = make_spec()
        supervisor.store.add(spec)
        supervisor.resource_plan = {"workers": 2}
        launches = []
        monkeypatch.setattr(supervisor, "_launch", lambda row, slot: launches.append((row, slot)))
        supervisor.monotonic_deadline = time.monotonic() - 1
        supervisor._fill_workers()
        assert not launches
        supervisor.monotonic_deadline = time.monotonic() + 120
        monkeypatch.setattr(
            "geo_llm_scheduler.experiments.campaign.scheduler.free_disk_gb", lambda _path: 0.0
        )
        supervisor._fill_workers()
        assert not launches
        assert (supervisor.root / "disk_incident.json").exists()
    finally:
        supervisor.store.close()
