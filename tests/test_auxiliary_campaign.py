"""Independent input, provenance, selection, streaming and transactional-run tests."""

import gzip
import hashlib
import json
import random
import socket
from collections import Counter
from dataclasses import asdict, replace

import pytest

from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.experiments.auxiliary_instances import PRESSURES, TOPOLOGIES, paired_instance
from geo_llm_scheduler.experiments.runner import digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.macrosearch.trajectory_budget import ARM_SPECS
from geo_llm_scheduler.utils.numeric import schedule_identity
from geo_llm_scheduler.utils.rng import RNGManager
from research.auxiliary_walk_20261016.campaign import (
    make_config,
    memory_available,
    ready_spec,
    stage_specs,
    williams,
)
from research.auxiliary_walk_20261016.common import sha, source, stream_json, validate_complete
from research.auxiliary_walk_20261016.preflight import thresholds
from research.auxiliary_walk_20261016.selection import choose, nondominated
from research.auxiliary_walk_20261016.worker import Observer, execute_run


@pytest.mark.parametrize("pressure", PRESSURES)
def test_pairs_change_only_declared_factor(pressure):
    a, b = [paired_instance(20, 3, 12, pressure, v, random.Random(16442)) for v in (0, 1)]
    assert a != b
    if pressure in ("arrival", "kv"):
        field = "release" if pressure == "arrival" else "kv_delay"
        assert a.regions == b.regions and a.instances == b.instances
        assert a.jobs == tuple(
            replace(y, **{field: getattr(x, field)}) for x, y in zip(a.jobs, b.jobs)
        )
    elif pressure == "demand":
        assert a.jobs == b.jobs and a.instances == b.instances
        assert a.regions == tuple(replace(r, demand_rate=0.2) for r in b.regions)
    else:
        assert a.jobs == b.jobs and a.regions == b.regions
        assert sorted(Counter(m.region for m in a.instances).values()) == sorted(
            Counter(m.region for m in b.instances).values()
        )
        assert sum(m.region == 0 for m in a.instances) > sum(m.region == 0 for m in b.instances)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_generated_dimensions_and_explicit_rng(topology):
    r, m = topology
    one, two = random.Random(33117), random.Random(33117)
    a = paired_instance(200, r, m, "kv", 0, one)
    b = paired_instance(200, r, m, "kv", 1, two)
    assert one.getstate() == two.getstate()
    assert len(a.instances) == len(b.instances) == m and len(a.regions) == r
    with pytest.raises(ValueError):
        paired_instance(20, r, m, "invalid", 0, random.Random(1))


@pytest.mark.parametrize("count", (4, 18))
def test_williams_periods_and_predecessors_are_balanced(count):
    rows = [williams(count, i) for i in range(count)]
    assert all(set(row) == set(range(count)) for row in rows)
    assert all(len({row[j] for row in rows}) == count for j in range(count))
    assert set(Counter((a, b) for row in rows for a, b in zip(row, row[1:])).values()) == {1}
    with pytest.raises(ValueError):
        williams(3, 0)


def test_streaming_hashes_are_independent_and_no_giant_serialization(tmp_path):
    path = tmp_path / "data.json.gz"
    values = ({"item": i, "large": list(range(100))} for i in range(100))
    meta = stream_json(path, values)
    raw = gzip.decompress(path.read_bytes())
    assert len(json.loads(raw)) == meta["count"] == 100
    assert hashlib.sha256(raw).hexdigest() == meta["raw_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]


def test_screening_choice_is_preregistered_and_matches_aid():
    rows = [
        {
            "spec": {"case": "case", "parent": "parent", "arm": arm},
            "archive_points": [(1, 1)] if arm == "A1" else [(2, 2)],
            "diagnostic_counts": {"full_request:3": 1, "full_request:6": 2, "full_request:10": 1},
        }
        for arm in ARM_SPECS
    ]
    choice = choose(rows)
    assert choice["selected"] == "A1" and choice["random_aid"]
    assert choice["probabilities"] == [0.25, 0.5, 0.25]
    cfg = make_config("R0", 1, 10, {}, choice)
    assert cfg.auxiliary_random_aid and cfg.auxiliary_random_probabilities == (0.25, 0.5, 0.25)
    assert nondominated([(1, 2), (2, 1), (2, 2), (1, 3)]) == [(1, 2), (2, 1)]
    with pytest.raises(ValueError):
        choose(rows[:-1])


def test_specs_keep_parent_host_and_frozen_config():
    record = {
        "stage": "confirm",
        "case": "c",
        "parent": "p",
        "host": "h",
        "node": 0,
        "seed": 161001,
        "initial_config": {"seconds": 600},
        "order_row": 15,
    }
    plan = {
        "records": [record],
        "calibration": {"D2": (0.01, 0.1)},
        "stage_arms": {"confirm": ["S0", "W1", "SELECTED", "R0"]},
    }
    choice = {"selected": "D2", "probabilities": (0.2, 0.6, 0.2), "random_aid": False}
    specs = stage_specs(plan, "confirm", choice)
    assert {s["arm"] for s in specs} == {"S0", "W1", "D2", "R0"}
    assert {s["host"] for s in specs} == {"h"} and {s["parent"] for s in specs} == {"p"}
    assert all(
        s["config"]["seconds"] == 600 and s["selection_hash"] == digest(choice) for s in specs
    )
    with pytest.raises(ValueError):
        stage_specs(plan, "confirm")


def test_each_case_executes_its_williams_sequence_serially():
    pending = [
        {"case": "a", "arm": "second"},
        {"case": "a", "arm": "third"},
        {"case": "b", "arm": "first"},
    ]
    active = [{"case": "a", "arm": "first"}]
    assert ready_spec(pending, active) == pending[2]
    assert ready_spec(pending, []) == pending[0]
    assert ready_spec(pending[:2], active) is None


def test_memory_guard_reclaims_only_clean_inactive_file_pages():
    assert memory_available(8192, 8000, {"inactive_file": 7000}) == 7192
    assert (
        memory_available(
            8192,
            8000,
            {"inactive_file": 7000, "file_dirty": 500, "file_writeback": 100, "shmem": 600},
        )
        == 5992
    )
    assert memory_available(8192, 8000, {"active_file": 7000}) == 192
    assert memory_available(8192, 9000, {}) == 0


def test_independent_calibration_retains_zero_ties_and_rejects_invalid_signals():
    assert thresholds([0, 0, 0, 0, 0, 1]) == [0, 0]
    assert thresholds([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) == [0.2, 0.4]
    for values in ([], [float("nan")], [-0.01], [1.01]):
        with pytest.raises(ValueError):
            thresholds(values)


def test_worker_native_initialization_exact_and_tamper_gate(problem, tmp_path):
    cfg = replace(
        make_config("D2", 15171, 30, {}, population=4), generations=1, trigger_mode="always"
    )
    save_instance(problem, tmp_path / "input.json")
    initial = initial_genotypes(problem, cfg, RNGManager(cfg.seed).stream("initialization"))
    spec = {
        "id": "test",
        "host": socket.gethostname(),
        "arm": "D2",
        "input": "input.json",
        "input_sha256": sha(tmp_path / "input.json"),
        "config": asdict(cfg),
        "initial_hash": digest([asdict(g) for g in initial]),
    }
    provenance = source()
    marker = execute_run(spec, tmp_path, tmp_path / "run", provenance)
    assert validate_complete(tmp_path / "run", spec, provenance) == marker
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["offspring"] == 4 and summary["independent_exact"] >= 4
    assert len(json.loads((tmp_path / "run" / "q.json").read_text())["q"]) == 42
    with pytest.raises(ValueError):
        validate_complete(tmp_path / "run", {**spec, "id": "tampered"}, provenance)
    (tmp_path / "run" / "q.json").write_text("{}")
    with pytest.raises(ValueError):
        validate_complete(tmp_path / "run", spec, provenance)


def test_physical_identity_ignores_equivalent_os(problem):
    from geo_llm_scheduler.archive.pareto import Archive
    from geo_llm_scheduler.engine.evaluation import EvaluationGateway

    gateway = EvaluationGateway(problem, Archive())
    a = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 0, 1, 1)))
    b = replace(a, genotype=replace(a.genotype, os=(1, 1, 0, 0)))
    assert schedule_identity(a) == schedule_identity(b)


def test_recovery_uses_fixed_context_and_all_historical_candidates(problem):
    from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Schedule
    from geo_llm_scheduler.moead.core import NormalizationContext

    def point(value):
        return Candidate(
            Genotype((0, 0, 1, 1), (0, 0, 1, 1)),
            Schedule((value,) * 4),
            EvaluationResult(True, value, 1, (), ()),
        )

    positions = [point(v) for v in (10, 12, 10, 11)]
    histories = [[10, 12], [10, 12, 10, 8], [10, 12, 10, 8, 11]]
    rows = [
        {
            "accepted": True,
            "regression": j != 1,
            "revisit": False,
            "effective": 6,
            "attempts": 6,
            "action": 1,
            "budget": 6,
            "budget_details": {"credit_used": 6 * (j + 1)},
            "historical_objectives": [(v, 1) for v in values],
        }
        for j, values in enumerate(histories)
    ]
    observer = Observer(problem, 30, 1)
    observer(
        "auxiliary_episode",
        {
            "positions": positions,
            "records": rows,
            "context": NormalizationContext((0, 0), (20, 20)),
            "weight": (1, 0),
            "selected": point(8),
            "arm": "W1",
        },
    )
    assert observer.counts["breakthrough_after:1"] == 1
    assert observer.counts["position_recovery_after:1"] == 1
    assert observer.counts["unrecovered_positions"] == 1
    assert observer.counts["last_best_gap_sum"] == pytest.approx(0.15)
