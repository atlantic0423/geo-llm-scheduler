"""Paired fixed-work, local-only benchmarks against exact frozen b148c4a sources."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import platform
import pstats
import random
import statistics
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from geo_llm_scheduler.archive.pareto import Archive  # noqa: E402
from geo_llm_scheduler.config import Config  # noqa: E402
from geo_llm_scheduler.diagnostics import severity  # noqa: E402
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance  # noqa: E402
from geo_llm_scheduler.engine import evaluation  # noqa: E402
from geo_llm_scheduler.engine.run import run  # noqa: E402
from geo_llm_scheduler.evaluation import exact  # noqa: E402
from geo_llm_scheduler.experiments.comparison13 import (  # noqa: E402
    arm_config,
    case_problem,
    initialization_factory,
)
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash  # noqa: E402
from geo_llm_scheduler.experiments.runner import deterministic_trace  # noqa: E402
from geo_llm_scheduler.initialization.generators import generate  # noqa: E402
from geo_llm_scheduler.operators import active_pack, peak_coalition, right_shift  # noqa: E402
from geo_llm_scheduler.scheduling import ssgs  # noqa: E402
from research.performance_reuse_20261020 import reference_runtime as reference  # noqa: E402


def fixed_work(problem: ProblemInstance, population: int, bill: bool) -> dict:
    """Hash all candidates, Q and RNG at fixed generation count, including setup."""
    config = replace(
        arm_config("M11_I4", 20261020, 300),
        seconds=None,
        generations=1,
        population=population,
        neighborhood=min(20, population),
    )
    candidates = hashlib.sha256()
    streams = []

    def observe(event: str, data: dict) -> None:
        if event == "candidate":
            candidates.update(json.dumps(asdict(data["candidate"]), sort_keys=True).encode())
        elif event == "offspring":
            streams.append(data["streams"].snapshot())

    evidence = {}
    result = run(
        problem,
        config,
        observer=observe,
        initializer=initialization_factory(problem, config, 4, 220261020, evidence)
        if bill
        else None,
    )
    return {
        "population": [asdict(c) for c in result.population],
        "archive": [asdict(c) for c in result.archive.members],
        "trace": deterministic_trace(result.trace),
        "candidate_stream": candidates.hexdigest(),
        "counts": dict(result.gateway.counts),
        "q": result.controller.q,
        "rng": streams,
        "offspring": result.offspring_count,
        "initial_hashes": {k: v for k, v in evidence.items() if k.endswith("sha256")},
    }


def main() -> None:
    """Alternate execution order and reject any deterministic output mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", help="Optional workload name substring")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two alternating repeats")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    source_hash = canonical_source_hash()

    def benchmark(name: str, function: Callable[[], object]) -> None:
        if args.only and args.only not in name:
            return
        attempts = []
        for repeat in range(args.repeats):
            pair = {}
            for old in (True, False) if repeat % 2 == 0 else (False, True):
                with reference.reference_runtime() if old else nullcontext():
                    wall, cpu = perf_counter(), process_time()
                    value = function()
                    elapsed = (perf_counter() - wall, process_time() - cpu)
                pair["reference" if old else "optimized"] = {
                    "wall": elapsed[0],
                    "cpu": elapsed[1],
                    "digest": hashlib.sha256(
                        json.dumps(value, sort_keys=True, default=asdict).encode()
                    ).hexdigest(),
                }
            if pair["reference"]["digest"] != pair["optimized"]["digest"]:
                raise AssertionError((name, pair))
            attempts.append(pair)
            print(json.dumps({"name": name, "repeat": repeat, **pair}), flush=True)
        old_wall = statistics.median(p["reference"]["wall"] for p in attempts)
        new_wall = statistics.median(p["optimized"]["wall"] for p in attempts)
        if canonical_source_hash() != source_hash:
            raise AssertionError("Optimized source changed during benchmark")
        rows.append(
            {
                "name": name,
                "reference_wall_median": old_wall,
                "optimized_wall_median": new_wall,
                "wall_reduction": 1 - new_wall / old_wall,
                "attempts": attempts,
            }
        )
        (args.output / "benchmark.json").write_text(
            json.dumps(
                {
                    "reference_commit": "b148c4addd207fe30a6b2a43d953a04afc35d3a1",
                    "optimized_canonical_source_hash": source_hash,
                    "python": sys.version,
                    "platform": platform.platform(),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for size in (20, 50, 100, 200):
        problem = case_problem(size, 3, 12, 20261020, "kv", 1)
        genotype = generate(problem, 7, random.Random(730), 0.1)
        schedule = ssgs.decode(problem, genotype)
        candidate = Candidate(genotype, schedule, exact.evaluate(problem, genotype, schedule))
        benchmark(
            f"exact_{size}_x12",
            lambda: [exact.evaluate(problem, genotype, schedule) for _ in range(12)],
        )
        benchmark(f"decode_{size}_x6", lambda: [ssgs.decode(problem, genotype) for _ in range(6)])
        benchmark(
            f"refresh_{size}_x6",
            lambda: [ssgs.refresh_diagnostics(problem, genotype, schedule) for _ in range(6)],
        )
        benchmark(f"severity_{size}", lambda: severity.severities(problem, candidate))

        def propose() -> object:
            rng = random.Random(51)
            result = active_pack.ActivePack().propose(problem, candidate, 6, Config(), rng)
            return asdict(result), rng.getstate()

        benchmark(f"A7_{size}", propose)

        def propose8() -> object:
            rng = random.Random(52)
            result = peak_coalition.PeakCoalition().propose(problem, candidate, 6, Config(), rng)
            return asdict(result), rng.getstate()

        benchmark(f"A8_{size}", propose8)
        benchmark(f"generate_energy_{size}", lambda: generate(problem, 9, random.Random(53), 0.1))

        def polish() -> object:
            gateway = evaluation.EvaluationGateway(problem, Archive())
            calls = []
            gateway.candidate_sink = calls.append
            stats = right_shift.PolishStats()
            best = right_shift.polish(problem, candidate, gateway, stats)
            return best, calls, gateway.archive.members, dict(gateway.counts), vars(stats)

        benchmark(f"polish_{size}", polish)
        population = 100 if size <= 50 else 24
        benchmark(f"full_P{population}_g1_{size}", lambda: fixed_work(problem, population, False))
        if size <= 50:
            benchmark(f"full_BILL4_P100_g1_{size}", lambda: fixed_work(problem, 100, True))
    if args.profile:
        problem = case_problem(200, 3, 12, 20261020, "kv", 1)
        for label, old in (("reference", True), ("optimized", False)):
            with reference.reference_runtime() if old else nullcontext():
                profiler = cProfile.Profile()
                profiler.runcall(fixed_work, problem, 24, False)
            profiler.dump_stats(str(args.output / (label + ".prof")))
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(
                55
            )
            (args.output / (label + ".txt")).write_text(stream.getvalue(), encoding="utf-8")
    print(json.dumps({"status": "complete", "paired_workloads": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
