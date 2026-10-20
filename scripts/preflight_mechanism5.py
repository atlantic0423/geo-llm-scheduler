"""Audit each frozen initializer in independent processes before formal dispatch."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.comparison13 import initialization_factory
from geo_llm_scheduler.experiments.mechanism5 import mechanism_config
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager


def audit_one(root_string: str, key: str) -> dict:
    """Construct all 140 candidates and independently recheck their exact results."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    root = Path(root_string)
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    destination = root / "initialization_preflight" / (key + ".json")
    tick = time.perf_counter()
    try:
        problem = load_instance(root / spec["input"])
        config = mechanism_config("A_CURRENT", spec["seed"], spec["seconds"])
        candidates = []
        gateway = EvaluationGateway(problem, Archive())
        gateway.candidate_sink = candidates.append
        evidence = {}
        population = initialization_factory(problem, config, 4, spec["bank_seed"], evidence)(
            gateway, RNGManager(config.seed)
        )
        production = time.perf_counter() - tick
        assert len(population) == len({c.genotype for c in population}) == 100
        assert len(candidates) == gateway.counts["exact"] == 140
        for candidate in candidates:
            assert candidate.evaluation.feasible
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
        result = dict(
            key=key,
            success=True,
            production_seconds=production,
            audit_seconds=time.perf_counter() - tick - production,
            candidates_verified=len(candidates),
            initial_population_sha256=evidence["initial_population_sha256"],
            initial_archive_sha256=evidence["initial_archive_sha256"],
            bank_rng_sha256=evidence["bank_stats"]["rng_state_sha256"],
            config=asdict(config),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        atomic_json(destination, result)
        return result
    except BaseException:
        atomic_json(destination, dict(key=key, success=False, traceback=traceback.format_exc()))
        raise


def preflight(root: Path, parallel: int) -> None:
    """Validate frozen inputs and stop acceptance on any initializer failure."""
    from run_mechanism5 import validate_inputs

    manifest = validate_inputs(root)
    target = root / "initialization_preflight"
    target.mkdir(exist_ok=False)
    tick = time.time()
    results = []
    with ProcessPoolExecutor(
        max_workers=parallel, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(audit_one, str(root), key) for key in manifest["blocks"]]
        try:
            for future in as_completed(futures):
                results.append(future.result())
                atomic_json(
                    target / "status.json",
                    dict(
                        updated=time.time(),
                        pid=os.getpid(),
                        completed=len(results),
                        total=len(futures),
                    ),
                )
        except BaseException:
            for future in futures:
                future.cancel()
            atomic_json(
                target / "failure.json", dict(updated=time.time(), traceback=traceback.format_exc())
            )
            raise
    atomic_json(
        target / "complete.json",
        dict(
            success=True,
            inputs=len(results),
            exact_verified=sum(r["candidates_verified"] for r in results),
            completed=time.time(),
            elapsed=time.time() - tick,
            source=manifest["source"],
            commit=manifest["commit"],
            records=sorted(results, key=lambda r: r["key"]),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=48)
    args = parser.parse_args()
    preflight(args.root, args.parallel)
