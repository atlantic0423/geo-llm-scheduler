"""Local paired fixed-work benchmarks; never connect to experiment servers."""

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

# Support a source checkout without changing its installed package selection.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from geo_llm_scheduler.config import Config  # noqa: E402
from geo_llm_scheduler.domain.models import (  # noqa: E402
    Candidate,
    EvaluationResult,
    Genotype,
    ProblemInstance,
    Schedule,
)
from geo_llm_scheduler.engine.run import run  # noqa: E402
from geo_llm_scheduler.evaluation.exact import evaluate  # noqa: E402
from geo_llm_scheduler.experiments.comparison13 import (  # noqa: E402
    arm_config,
    case_problem,
    initialization_factory,
)
from geo_llm_scheduler.experiments.runner import deterministic_trace, digest  # noqa: E402
from geo_llm_scheduler.initialization.generators import generate  # noqa: E402
from geo_llm_scheduler.scheduling.ssgs import decode  # noqa: E402
from research.performance_20261019 import reference_runtime as reference  # noqa: E402


def timed(function: Callable[[], object], use_reference: bool) -> dict:
    """Measure production work without profiling; return a deterministic digest."""
    with reference.reference_runtime() if use_reference else nullcontext():
        wall, cpu = perf_counter(), process_time()
        value = function()
        wall, cpu = perf_counter() - wall, process_time() - cpu
    return {"wall": wall, "cpu": cpu, "digest": digest(value)}


def fixed_run(problem: ProblemInstance, bill: bool) -> dict:
    """Run the exact same finite work, recording candidates, Q and final RNG."""
    config = replace(arm_config("M11_I4", 191019, 300), seconds=None, generations=1)
    candidates = hashlib.sha256()
    final_rng = {}

    def observer(event, data):
        nonlocal final_rng
        if event == "candidate":
            candidates.update(json.dumps(asdict(data["candidate"]), sort_keys=True).encode())
        elif event == "offspring":
            final_rng = data["streams"].snapshot()

    evidence = {}
    result = run(
        problem,
        config,
        observer=observer,
        initializer=initialization_factory(problem, config, 4, 1191019, evidence) if bill else None,
    )
    return {
        "population": [asdict(c) for c in result.population],
        "archive": [asdict(c) for c in result.archive.members],
        "trace": deterministic_trace(result.trace),
        "candidate_stream": candidates.hexdigest(),
        "counts": dict(result.gateway.counts),
        "q": result.controller.q,
        "rng": final_rng,
        "offspring": result.offspring_count,
    }


def main() -> None:
    """Write paired timing evidence, rejecting any fixed-work digest mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two alternating-order repeats are required")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    def benchmark(name, function):
        attempts = []
        for repeat in range(args.repeats):
            pair = {}
            for old in (True, False) if repeat % 2 == 0 else (False, True):
                pair["reference" if old else "optimized"] = timed(function, old)
            assert pair["reference"]["digest"] == pair["optimized"]["digest"], name
            attempts.append(pair)
            print(json.dumps({"name": name, "repeat": repeat, **pair}), flush=True)
        reference_wall = statistics.median(a["reference"]["wall"] for a in attempts)
        optimized_wall = statistics.median(a["optimized"]["wall"] for a in attempts)
        reference_cpu = statistics.median(a["reference"]["cpu"] for a in attempts)
        optimized_cpu = statistics.median(a["optimized"]["cpu"] for a in attempts)
        rows.append(
            {
                "name": name,
                "reference_wall_median": reference_wall,
                "optimized_wall_median": optimized_wall,
                "wall_reduction": 1 - optimized_wall / reference_wall,
                "reference_cpu_median": reference_cpu,
                "optimized_cpu_median": optimized_cpu,
                "cpu_reduction": 1 - optimized_cpu / reference_cpu
                if min(reference_cpu, optimized_cpu) >= 0.05
                else None,
                "attempts": attempts,
            }
        )
        (args.output / "benchmark.json").write_text(
            json.dumps(
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "cpu_ratio_min_seconds": 0.05,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    for size in (20, 50, 100, 200):
        problem = case_problem(size, 3, 12, 191019, "kv", 1)
        genotypes = [generate(problem, 7, random.Random(730 + i), 0.1) for i in range(12)]

        def decode_workload():
            decoder = sys.modules["geo_llm_scheduler.scheduling.ssgs"].decode
            return [
                asdict(decoder(problem, genotype))
                for _ in range(max(2, 200 // size))
                for genotype in genotypes
            ]

        benchmark(f"decode_{size}_{12 * max(2, 200 // size)}", decode_workload)
        schedule = decode(problem, genotypes[0])
        candidate = Candidate(genotypes[0], schedule, evaluate(problem, genotypes[0], schedule))

        def assignment_workload():
            operator = sys.modules["geo_llm_scheduler.operators.structural"].StructuralOperator
            return [
                asdict(operator(action).propose(problem, candidate, 6, Config(), random.Random(77)))
                for action in (2, 3)
            ]

        benchmark(f"A2_A3_{size}", assignment_workload)

    points = [
        Candidate(
            Genotype((0,) * 200, (0,) * 200),
            Schedule((float(i),) * 200),
            EvaluationResult(True, float(i), float(300 - i), (), ()),
        )
        for i in range(300)
    ]

    def archive_workload():
        archive_class = sys.modules["geo_llm_scheduler.archive.pareto"].Archive
        archive = archive_class()
        outcomes = [archive.consider(point) for point in points + points]
        return outcomes, [asdict(c) for c in archive.members], archive.insertions, archive.attempts

    benchmark("archive_300_frontier_600_attempts", archive_workload)
    for size in (20, 50):
        problem = case_problem(size, 3, 12, 191019, "kv", 1)
        for bill in (False, True):
            benchmark(
                f"full_P100_g1_{size}_{'BILL4' if bill else 'original'}",
                lambda: fixed_run(problem, bill),
            )

    problem = case_problem(20, 3, 12, 191019, "kv", 1)
    for label, old in (("reference20_matched", True), ("optimized20", False)):
        with reference.reference_runtime() if old else nullcontext():
            profiler = cProfile.Profile()
            profiler.runcall(fixed_run, problem, True)
        profiler.dump_stats(str(args.output / f"{label}.prof"))
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(45)
        (args.output / f"{label}.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
