"""Fixed-input selector overhead and unchanged endpoint regression against frozen Git source."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

from run_lsv2 import sha, source_hash

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.polish_selection import (
    POOL_POLICIES,
    PolishSelector,
    candidate_pool,
    physical_fingerprint,
)
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.utils.numeric import TOL, close
from geo_llm_scheduler.utils.rng import RNGManager

REFERENCE = "93137d61508a92e50465fc03acff3ba36e9074ab"


def benchmark(input_path: Path, population_path: Path, repeats: int = 20) -> dict:
    """Audit historical inputs, time selection, compare endpoint outcomes and cost."""
    problem = load_instance(input_path)
    with gzip.open(population_path, "rt", encoding="utf-8") as stream:
        pop = [restore_candidate(row) for row in json.load(stream)]
    restored_deltas = []
    for c in pop:
        exact = evaluate(problem, c.genotype, c.schedule)
        assert c.evaluation.feasible and exact.feasible
        assert close(exact.flow, c.evaluation.flow, TOL.time)
        assert close(exact.tou, c.evaluation.tou, TOL.cost)
        assert all(close(a, b, TOL.cost) for a, b in zip(exact.demand, c.evaluation.demand))
        assert all(
            close(a, b, TOL.power)
            for x, y in zip(exact.windows, c.evaluation.windows)
            for a, b in zip(x, y)
        )
        restored_deltas.append(abs(exact.bill - c.evaluation.bill))
    frame = FrozenFrame.of([c.evaluation.objectives for c in pop])
    ideal = tuple(min(c.evaluation.objectives[k] for c in pop) for k in range(2))
    context = NormalizationContext(ideal, maximum(pop))
    lambdas = weights(len(pop))
    samples = {}
    for policy in POOL_POLICIES:
        selector, rng = PolishSelector(), Random(16901)
        samples[policy] = []
        for _ in range(repeats):
            tick = perf_counter()
            pick = selector.choose(policy, pop, pop[0], frame, context, lambdas, rng)
            samples[policy].append(perf_counter() - tick)
            selector.charge(pick.region, 0.001)
    samples["fingerprint"] = []
    for _ in range(repeats):
        tick = perf_counter()
        physical_fingerprint(pop[0])
        samples["fingerprint"].append(perf_counter() - tick)
    # Frozen endpoint implementation is loaded from Git, never copied from current code.
    source = subprocess.check_output(
        ["git", "show", REFERENCE + ":src/geo_llm_scheduler/lsv2/search.py"], text=True
    )
    reference = types.ModuleType("frozen_rightshift_search")
    sys.modules[reference.__name__] = reference
    exec(compile(source, REFERENCE + "/search.py", "exec"), reference.__dict__)
    cfg = Config(population=len(pop), neighborhood=min(20, len(pop)), seconds=100000)
    config = replace(LSV2Config(), operators=("A1",), polish_region="all")

    def endpoint(cls, targets):
        gateway = EvaluationGateway(problem, Archive())
        for c in pop:
            gateway.archive.consider(c)
        gateway.ideal = ideal
        search = cls(replace(config, polish_targets=targets), cfg, cfg)
        current_pop = list(pop)
        tick = perf_counter()
        search._polish(pop[0], frame, 0, current_pop, gateway, RNGManager(16901), lambda: 0)
        seconds = perf_counter() - tick
        return seconds, (current_pop, gateway.archive.members, dict(gateway.counts))

    for targets in ("partial", "all"):
        for i in range(3):
            order = (("frozen", reference.LSV2Search), ("current", LSV2Search))
            results = []
            for name, cls in order if i % 2 == 0 else order[::-1]:
                seconds, outcome = endpoint(cls, targets)
                samples.setdefault(name + "_endpoint_" + targets, []).append(seconds)
                results.append(outcome)
            assert results[0] == results[1], "Endpoint phenotype/archive/count regression"
    return dict(
        n=len(problem.jobs),
        input_sha256=sha(input_path),
        population_sha256=sha(population_path),
        population_size=len(pop),
        unique_physical_inputs=len(candidate_pool(pop, pop[0])),
        endpoint_reference=REFERENCE,
        endpoint_outcome_equivalence=True,
        cross_platform_max_bill_delta=max(restored_deltas),
        samples=samples,
        median_seconds={k: median(v) for k, v in samples.items()},
        scope="Fixed historical input; selector overhead and endpoint regression, not search quality",
    )


def main() -> None:
    """Write one source-bound local benchmark without starting optimizer campaigns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.input, args.population)
    result["source_sha256"] = source_hash()
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}))


if __name__ == "__main__":
    main()
