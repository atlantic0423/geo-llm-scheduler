"""Replay archived genotypes through the new and the frozen 582237f code; require identity.

Each archived genotype is decoded and exact-evaluated by both paths; its archived
timing is exact-evaluated too, and so is a random structural neighbour (one OS swap
and one same-Region MS change), which also exercises infeasible-looking edits. Both
paths must return equal ``Schedule`` and ``EvaluationResult`` values, and each
archived timing's LS ``PowerProfile`` must agree on every window and TOU value.

Usage (outside the test suite; reads local campaign results):
    python -m research.throughput_20261109.golden_replay RESULTS_ROOT [RESULTS_ROOT ...]
        [--per-size K] [--seed S] [--out data/golden_replay.json]
where each RESULTS_ROOT holds ``formal_panel_shard*/`` campaign roots.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation import exact
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2 import billing
from geo_llm_scheduler.scheduling import ssgs

from .reference_runtime import reference_runtime


def neighbour(problem: ProblemInstance, genotype: Genotype, rng: random.Random) -> Genotype:
    """One OS swap and one MS change inside the partner operation's Region."""
    os_ = list(genotype.os)
    i, k = rng.randrange(len(os_)), rng.randrange(len(os_))
    os_[i], os_[k] = os_[k], os_[i]
    ms = list(genotype.ms)
    o = rng.randrange(len(ms))
    ms[o] = rng.choice(problem.eligible(o, problem.instances[ms[o ^ 1]].region))
    return Genotype(tuple(ms), tuple(os_))


def _profile_values(problem: ProblemInstance, genotype: Genotype, schedule: Schedule) -> tuple:
    pp = billing.PowerProfile(problem, genotype.ms, schedule.starts)
    later = pp.with_horizon(pp.horizon + 1234.5)
    return pp.tou, pp.windows, pp.bill(), later.tou, later.windows


def _one(problem: ProblemInstance, genotype: Genotype, archived: Schedule | None) -> tuple:
    decoded = ssgs.decode(problem, genotype)
    out = [decoded, exact.evaluate(problem, genotype, decoded)]
    if archived is not None:
        out += [
            exact.evaluate(problem, genotype, archived),
            _profile_values(problem, genotype, archived),
        ]
    return tuple(out)


def replay(roots: list[Path], per_size: int, seed: int) -> dict:
    """Compare both paths on ``per_size`` blocks per task size from each campaign root."""
    rng = random.Random(seed)
    totals: dict[int, dict[str, float]] = {}
    for base in roots:
        for root in sorted(base.glob("formal_panel_shard*")):
            chosen: dict[int, int] = {}
            for spec_path in sorted((root / "specs").glob("*.json")):
                if spec_path.name.endswith(".sha256.json"):
                    continue
                spec = json.loads(spec_path.read_text(encoding="utf8"))
                n = spec["n"]
                if chosen.get(n, 0) >= per_size:
                    continue
                chosen[n] = chosen.get(n, 0) + 1
                problem = load_instance(root / spec["input"])
                cases: list[tuple[Genotype, Schedule | None]] = []
                for arm in spec["order"]:
                    path = root / "runs" / spec["key"] / arm / "archive.json.gz"
                    with gzip.open(path, "rt", encoding="utf8") as f:
                        for x in json.load(f):
                            g = Genotype(tuple(x["genotype"]["ms"]), tuple(x["genotype"]["os"]))
                            cases.append((g, Schedule(tuple(x["schedule"]["starts"]))))
                            cases.append((neighbour(problem, g, rng), None))
                stat = totals.setdefault(n, dict(cases=0, new=0.0, reference=0.0))
                start = time.perf_counter()
                new = [_one(problem, g, archived) for g, archived in cases]
                stat["new"] += time.perf_counter() - start
                with reference_runtime():
                    start = time.perf_counter()
                    old = [_one(problem, g, archived) for g, archived in cases]
                    stat["reference"] += time.perf_counter() - start
                for i, (a, b) in enumerate(zip(new, old, strict=True)):
                    if a != b:
                        raise AssertionError(f"mismatch: {spec['key']} case {i}")
                stat["cases"] += len(cases)
    return {
        str(n): dict(v, speedup=v["reference"] / v["new"] if v["new"] else None)
        for n, v in sorted(totals.items())
    }


def main() -> None:
    """Command-line entry: replay, print and optionally save the comparison."""
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--per-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = dict(
        roots=[r.parent.name for r in a.roots],
        per_size=a.per_size,
        seed=a.seed,
        identical=True,
        sizes=replay(a.roots, a.per_size, a.seed),
    )
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
