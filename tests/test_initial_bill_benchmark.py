"""Reference replay, isolated exact budget, nested RNG and four-arm integrity."""

import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.initial_bill_benchmark import (
    METHODS,
    ORDERS,
    benchmark_population,
    traced_original,
)
from geo_llm_scheduler.experiments.initial_bill_mixture import seed_bank
from geo_llm_scheduler.experiments.initial_frontier import diagnostic_problem
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager


def test_trace_preserves_original_output_and_modes():
    p = diagnostic_problem(12, 1479011, "T", "mixed")
    cfg = Config(seed=11001)
    parents = traced_original(p, cfg)
    assert [c.genotype for c in parents] == initial_genotypes(
        p, cfg, RNGManager(cfg.seed).stream("initialization")
    )
    assert Counter(c.origin for c in parents) == {f"initial_mode_{i}": 10 for i in range(10)}


@pytest.mark.parametrize("method", METHODS)
def test_independent_population_exact_budget_and_clocks(method):
    p = diagnostic_problem(12, 1479012, "T", "mixed")
    pop, front, stats = benchmark_population(p, Config(seed=11001), method, 9744)
    assert len(pop) == len({c.genotype for c in pop}) == 100
    assert front and all(c.evaluation.feasible for c in pop)
    assert stats["new_candidate_exact"] == (0 if method == "BASE" else int(method[-2:]) * 4)
    assert stats["final_independent_exact_verified"] == 100
    assert stats["production_wall_seconds"] > 0 and stats["production_cpu_seconds"] > 0
    assert stats["total_elapsed_seconds"] >= stats["production_wall_seconds"]
    assert stats["audit_wall_seconds"] > 0
    quota = 10 if method == "BASE" else (100 - int(method[-2:])) // 10
    assert Counter(c.origin for c in pop if c.origin.startswith("initial_mode")) == {
        f"initial_mode_{i}": quota for i in range(10)
    }


def test_bill10_constructs_ten_only_and_matches_bill20_prefix():
    p = diagnostic_problem(12, 1479013, "H", "mixed")
    parents = traced_original(p, Config(seed=11001))
    a, sa = seed_bank(p, parents, random.Random(97), 4, families=("BILL",), size=10)
    b, sb = seed_bank(p, parents, random.Random(97), 4, families=("BILL",), size=20)
    assert set(a) == set(b) == {"BILL"}
    assert a["BILL"] == b["BILL"][:10]
    assert sa["exact"] == 40 and sb["exact"] == 80
    assert len(sa["observations"]) == 40


def test_williams_position_and_predecessor_balance():
    assert all(sorted(row) == [0, 1, 2, 3] for row in ORDERS)
    for position in range(4):
        assert {row[position] for row in ORDERS} == {0, 1, 2, 3}
    assert Counter((a, b) for row in ORDERS for a, b in zip(row, row[1:])) == {
        (a, b): 1 for a in range(4) for b in range(4) if a != b
    }


def test_bad_families_and_resource_guard():
    p = diagnostic_problem(12, 1479014, "T", "mixed")
    parents = traced_original(p, Config(seed=11001))
    for families, size in (((), 10), (("PORTFOLIO",), 20), (("BILL", "BILL"), 10), (("BILL",), 12)):
        with pytest.raises(ValueError):
            seed_bank(p, parents, random.Random(1), 4, families=families, size=size)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_initial_bill_benchmark import dispatch_allowed

    assert dispatch_allowed({"available_memory_gib": 1.6, "free_disk_gib": 12}, 0, 500, 300)
    assert not dispatch_allowed({"available_memory_gib": 1.4, "free_disk_gib": 12}, 0, 500, 300)


def test_frozen_input_worker_complete_and_tamper_rejection(tmp_path, monkeypatch):
    """Exercise preparation, worker, all four P100 exports and hash rejection."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_initial_bill_benchmark as runner

    from geo_llm_scheduler.experiments.campaign.support import atomic_json
    from geo_llm_scheduler.experiments.campaign.validation import file_hash

    monkeypatch.setattr(runner, "git_metadata", lambda: {"git_dirty": False, "git_commit": "test"})
    monkeypatch.setattr(runner, "HOSTS", (platform.node(), platform.node()))
    old = tmp_path / "old"
    old.mkdir()
    instance = "instances/test.json"
    (old / "instances").mkdir()
    save_instance(diagnostic_problem(12, 1479021, "T", "mixed"), old / instance)
    key = "12_mixed_1479021_T_11001"
    spec = {
        "key": key,
        "jobs": 12,
        "scenario": "mixed",
        "base_seed": 1479021,
        "tariff": "T",
        "seed": 11001,
        "instance": instance,
        "instance_sha256": file_hash(old / instance),
    }
    atomic_json(old / "specs" / (key + ".json"), spec)
    m = {
        "node": 0,
        "source_commit": "5bb666874a86ae59e38aa2513a1c4f7ed1fa8fcb",
        "blocks": [key],
        "input_files": {"specs/" + key + ".json": file_hash(old / "specs" / (key + ".json"))},
    }
    atomic_json(old / "manifest.json", m)
    atomic_json(old / "manifest.sha256.json", {"sha256": file_hash(old / "manifest.json")})
    root = tmp_path / "new"
    assert runner.prepare(root, old)["population_count"] == 4
    runner.worker(root, key)
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    assert runner.complete(root, key, spec)
    hashes = [
        json.loads((root / "runs" / key / arm / "summary.json").read_text())[
            "parent_genotypes_sha256"
        ]
        for arm in METHODS
    ]
    assert len(set(hashes)) == 1
    p = root / "runs" / key / "BILL10/population.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.complete(root, key, spec)
