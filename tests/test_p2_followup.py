"""Four-arm atomic recovery and disjoint, tariff-paired host assignment."""

import json
import runpy
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments.campaign.support import atomic_json

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_p2_followup.py"))


def test_four_arm_completion_and_foreign_partial_recovery(tmp_path, monkeypatch):
    block = {"key": "b", "runs": list(SCRIPT["ARMS"])}
    for key in block["runs"]:
        atomic_json(tmp_path / "specs" / f"{key}.json", {"key": key})
    monkeypatch.setitem(
        SCRIPT["recover_block"].__globals__, "validate_complete", lambda *_: (True, "ok")
    )
    for key in block["runs"][:3]:
        atomic_json(tmp_path / "runs" / key / "complete.json", {"host": "old"})
    assert not SCRIPT["recover_block"](tmp_path, block, "old")
    assert not SCRIPT["recover_block"](tmp_path, block, "new")
    assert len(list((tmp_path / "quarantine").glob("*/*/complete.json"))) == 3
    for key in block["runs"]:
        atomic_json(tmp_path / "runs" / key / "complete.json", {"host": "new"})
    assert SCRIPT["recover_block"](tmp_path, block, "new")
    atomic_json(tmp_path / "ops/failures/A7B.json", {"reason": "crash"})
    with pytest.raises(RuntimeError, match="Recorded crash"):
        SCRIPT["recover_block"](tmp_path, block, "new")


def test_frozen_host_shards_cover_new_bases_without_tariff_split(tmp_path, monkeypatch):
    globals_ = SCRIPT["prepare"].__globals__
    monkeypatch.setitem(
        globals_,
        "git_metadata",
        lambda *_: {
            "git_commit": "frozen",
            "git_dirty": False,
        },
    )
    repository = Path(__file__).resolve().parents[1]
    matrices = [SCRIPT["prepare"](tmp_path / str(node), repository, node=node) for node in (0, 1)]
    bases = []
    for node, matrix in enumerate(matrices):
        assert matrix["run_count"] == 480
        assert matrix["nominal_worker_hours"] == 360
        specs = [json.loads(p.read_text()) for p in (tmp_path / str(node) / "specs").glob("*.json")]
        identities = {(s["jobs"], s["base_seed"]) for s in specs}
        assert len(identities) == 12
        bases.append(identities)
        for jobs, base in identities:
            subset = [s for s in specs if (s["jobs"], s["base_seed"]) == (jobs, base)]
            assert {s["tariff"] for s in subset} == {"H", "T"}
            assert len({s["algorithm_seed"] for s in subset}) == 5
            assert {s["arm"] for s in subset} == set(SCRIPT["ARMS"])
    assert not bases[0] & bases[1]
    assert len(bases[0] | bases[1]) == 24
