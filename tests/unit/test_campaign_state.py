"""Durable checkpoint identity and retry state transition tests."""

from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.state_store import StateStore


def spec() -> JobSpec:
    return JobSpec(
        "stage", "arm", 1, 2, {"seconds": 1}, "instance.json", "a", "initial.json", "b", "c", "d"
    )


def test_sqlite_add_is_idempotent_and_hash_change_refused(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    job = spec()
    store.add(job)
    store.add(job)
    assert len(store.rows()) == 1
    store.transition(job.key, "RUNNING", pid=123, slot=0, attempt=1)
    store.close()
    resumed = StateStore(tmp_path / "state.sqlite3")
    assert resumed.get(job.key)["attempt"] == 1
    assert resumed.get(job.key)["state"] == "RUNNING"
    changed = JobSpec(**{**job.__dict__, "source_hash": "different"})
    try:
        resumed.add(changed)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("Input hash change was reused")
    resumed.close()
