"""Compare incremental and full-exact scoring on the same feasible right-shift probes.

This measures the scoring kernel, not complete-optimizer quality or speedup.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

from run_lsv2 import sha, source_hash

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.billing import PowerProfile
from geo_llm_scheduler.lsv2.flowsafe import prefill_times
from geo_llm_scheduler.scheduling.resources import _fits_intervals
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import TOL, less


def benchmark(path: Path, repeats: int = 3) -> dict:
    """Verify every probe delta, then alternate timed kernel order on fixed work."""
    problem = load_instance(path)
    genotype = initial_genotypes(problem, Config(seed=16901), Random(16901))[0]
    schedule = decode(problem, genotype)
    baseline = evaluate(problem, genotype, schedule)
    profile = PowerProfile(problem, genotype.ms, schedule.starts)
    probes = []
    for o in range(0, problem.operation_count, 2):
        lo, hi = legal_bounds(problem, genotype, schedule, o)
        lo = max(lo, schedule.starts[o])
        if not less(lo, hi, TOL.time):
            continue
        machine = genotype.ms[o]
        others = [
            (q, schedule.starts[q], schedule.end(problem, q))
            for q in range(problem.operation_count)
            if q != o and genotype.ms[q] == machine
        ]
        for t in prefill_times(problem, machine, o, lo, hi, others):
            if not less(schedule.starts[o], t, TOL.time) or not _fits_intervals(
                problem, machine, o, t, others
            ):
                continue
            starts = list(schedule.starts)
            starts[o] = t
            probes.append(
                (
                    {machine: {o: (t, t + problem.profile(o).duration)}},
                    replace(schedule, starts=tuple(starts)),
                )
            )
            if len(probes) >= 256:
                break
        if len(probes) >= 256:
            break
    if not probes:
        raise ValueError("No right-shift probes for benchmark")

    def fast() -> list[float]:
        return [profile.delta_for(change) for change, _ in probes]

    def exact() -> list[float]:
        results = [evaluate(problem, genotype, moved) for _, moved in probes]
        assert all(r.feasible and r.flow == baseline.flow for r in results)
        return [r.bill - baseline.bill for r in results]

    values, reference = fast(), exact()
    errors = [abs(a - b) for a, b in zip(values, reference)]
    assert max(errors) <= TOL.cost
    samples: dict[str, list[float]] = {"incremental": [], "full_exact": []}
    for i in range(repeats):
        order = (("incremental", fast), ("full_exact", exact))
        for name, function in order if i % 2 == 0 else order[::-1]:
            tick = perf_counter()
            function()
            samples[name].append(perf_counter() - tick)
    return dict(
        input_sha256=sha(path),
        n=len(problem.jobs),
        probes=len(probes),
        max_abs_bill_delta_error=max(errors),
        samples=samples,
        incremental_median=median(samples["incremental"]),
        full_exact_median=median(samples["full_exact"]),
        kernel_speedup=median(samples["full_exact"]) / median(samples["incremental"]),
    )


def main() -> None:
    """Write a source-bound microbenchmark without running any search campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        scope="fixed-probe kernel only",
        source_sha256=source_hash(),
        cases=[benchmark(p) for p in args.input],
    )
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
