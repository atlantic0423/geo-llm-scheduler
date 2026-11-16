"""Time SSGS decode, exact evaluation and LS PowerProfile separately, new vs frozen 582237f.

Fixed workload: archived genotypes (and their archived timings) of ``per_size`` blocks
per task size from campaign roots. Each component runs ``repeats`` times per path,
alternating the path order, and the median is reported. No profiler is attached.

Usage:
    python -m research.throughput_20261109.benchmark_components RESULTS_ROOT
        [--per-size K] [--repeats R] [--out data/benchmark_components.json]
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation import exact
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2 import billing
from geo_llm_scheduler.scheduling import ssgs

from .reference_runtime import reference_runtime

Case = tuple[ProblemInstance, Genotype, Schedule]


def load_cases(base: Path, per_size: int) -> dict[int, list[Case]]:
    """Archived (problem, genotype, timing) triples for ``per_size`` blocks per size."""
    out: dict[int, list[Case]] = {}
    for root in sorted(base.glob("formal_panel_shard*")):
        for spec_path in sorted((root / "specs").glob("*.json")):
            if spec_path.name.endswith(".sha256.json"):
                continue
            spec = json.loads(spec_path.read_text(encoding="utf8"))
            n = spec["n"]
            blocks = {id(c[0]) for c in out.get(n, [])}
            if len(blocks) >= per_size:
                continue
            problem = load_instance(root / spec["input"])
            for arm in spec["order"]:
                with gzip.open(
                    root / "runs" / spec["key"] / arm / "archive.json.gz", "rt", encoding="utf8"
                ) as f:
                    for x in json.load(f):
                        g = Genotype(tuple(x["genotype"]["ms"]), tuple(x["genotype"]["os"]))
                        out.setdefault(n, []).append(
                            (problem, g, Schedule(tuple(x["schedule"]["starts"])))
                        )
    return out


COMPONENTS: dict[str, Callable[[Case], object]] = {
    "decode": lambda c: ssgs.decode(c[0], c[1]),
    "exact": lambda c: exact.evaluate(c[0], c[1], c[2]),
    "power_profile": lambda c: billing.PowerProfile(c[0], c[1].ms, c[2].starts).with_horizon(
        c[2].horizon(c[0]) + 900.0
    ),
}


def _time(cases: list[Case], fn: Callable[[Case], object]) -> float:
    start = time.perf_counter()
    for c in cases:
        fn(c)
    return time.perf_counter() - start


def benchmark(cases: dict[int, list[Case]], repeats: int) -> dict:
    """Median seconds per component, path and size, with the reference/new ratio."""
    result: dict[str, dict] = {}
    for n, items in sorted(cases.items()):
        row: dict[str, dict] = {}
        for name, fn in COMPONENTS.items():
            new, old = [], []
            for r in range(repeats):
                for path in (0, 1) if r % 2 == 0 else (1, 0):
                    if path == 0:
                        new.append(_time(items, fn))
                    else:
                        with reference_runtime():
                            old.append(_time(items, fn))
            a, b = statistics.median(new), statistics.median(old)
            row[name] = dict(new=a, reference=b, speedup=b / a)
        result[str(n)] = dict(cases=len(items), **row)
    return result


def main() -> None:
    """Command-line entry."""
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--per-size", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = dict(
        root=a.root.parent.name,
        per_size=a.per_size,
        repeats=a.repeats,
        sizes=benchmark(load_cases(a.root, a.per_size), a.repeats),
    )
    text = json.dumps(out, indent=2)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
