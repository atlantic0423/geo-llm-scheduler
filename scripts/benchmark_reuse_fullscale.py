"""Confirm one fixed P100 large-instance run against the frozen implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from geo_llm_scheduler.experiments.comparison13 import case_problem  # noqa: E402
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash  # noqa: E402
from research.performance_reuse_20261020 import reference_runtime as reference  # noqa: E402
from scripts.benchmark_reuse_performance import fixed_work  # noqa: E402


def main() -> None:
    """Alternate old/new ordering and require complete deterministic equality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bill", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or args.size < 2 or args.population < 2:
        parser.error("Require at least two pairs, jobs and population members")
    args.output.mkdir(parents=True, exist_ok=True)
    source = canonical_source_hash()
    problem = case_problem(args.size, 3, 12, 20261020, "kv", 1)
    evidence = {
        "reference_commit": "b148c4addd207fe30a6b2a43d953a04afc35d3a1",
        "optimized_canonical_source_hash": source,
        "python": sys.version,
        "platform": platform.platform(),
        "jobs": args.size,
        "regions": 3,
        "instances": 12,
        "instance_seed": 20261020,
        "algorithm_seed": 20261020,
        "initialization_seed": 220261020 if args.bill else None,
        "pressure": "kv",
        "version": 1,
        "population": args.population,
        "generations": 1,
        "initializer": "BILL4" if args.bill else "ten-mode mixed",
        "fixed_work_includes_initialization": True,
        "attempts": [],
        "status": "running",
    }
    for repeat in range(args.repeats):
        pair = {}
        for old in (True, False) if repeat % 2 == 0 else (False, True):
            label = "reference" if old else "optimized"
            print(json.dumps({"repeat": repeat, "starting": label}), flush=True)
            with reference.reference_runtime() if old else nullcontext():
                wall, cpu = perf_counter(), process_time()
                value = fixed_work(problem, args.population, args.bill)
                elapsed = perf_counter() - wall, process_time() - cpu
            digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            pair[label] = {"wall": elapsed[0], "cpu": elapsed[1], "digest": digest}
            assert canonical_source_hash() == source, "Source changed during benchmark"
            print(json.dumps({"repeat": repeat, "finished": label, **pair[label]}), flush=True)
        assert pair["reference"]["digest"] == pair["optimized"]["digest"], pair
        evidence["attempts"].append(pair)
        (args.output / "benchmark.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
    old_wall = statistics.median(p["reference"]["wall"] for p in evidence["attempts"])
    new_wall = statistics.median(p["optimized"]["wall"] for p in evidence["attempts"])
    evidence.update(
        status="complete",
        reference_wall_median=old_wall,
        optimized_wall_median=new_wall,
        wall_reduction=1 - new_wall / old_wall,
        speedup=old_wall / new_wall,
    )
    (args.output / "benchmark.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "wall_reduction": evidence["wall_reduction"]}))


if __name__ == "__main__":
    main()
