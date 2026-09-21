"""Profile a fixed synthetic workload; this is engineering evidence, not algorithm quality."""

import cProfile
import io
import json
import pstats
import time
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.runner import deterministic_trace, digest
from geo_llm_scheduler.experiments.synthetic import synthetic

config = Config(population=20, neighborhood=5, generations=2, method="full", seed=7)
problem = synthetic(12, 2, 2, 7)
profiler = cProfile.Profile()
started = time.perf_counter()
result = profiler.runcall(run, problem, config)
elapsed = time.perf_counter() - started
stream = io.StringIO()
pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
Path("outputs").mkdir(exist_ok=True)
report = {
    "config": asdict(config),
    "jobs": 12,
    "elapsed_under_profiler": elapsed,
    "counts": dict(result.gateway.counts),
    "timings": dict(result.gateway.seconds),
    "trace_hash": digest(deterministic_trace(result.trace)),
}
Path("outputs/profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
Path("outputs/profile.txt").write_text(stream.getvalue(), encoding="utf-8")
print(json.dumps(report, indent=2))
print(stream.getvalue())
