"""Validate the real initializer against a frozen campaign's preflight hashes."""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
from pathlib import Path

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.initial_pool_retention import pool_initializer
from geo_llm_scheduler.experiments.s4_ablation import arm_config
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, nondominated
from geo_llm_scheduler.utils.rng import RNGManager


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def check_one(root: Path, out: Path, key: str) -> dict:
    """Run original generation plus opt-in retention and independently audit it."""
    spec = _read(root / "formal/specs" / (key + ".json"))
    path = root / "formal" / spec["input"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["input_sha256"]
    expected = _read(root / "server_startup/formal/preflight" / (key + ".json"))["policies"][
        "bill4"
    ]
    problem = load_instance(path)
    cfg = arm_config("A0_FULL_S4", spec["seed"], spec["seconds"])
    gw = EvaluationGateway(problem, Archive(), decode_cache_size=cfg.decode_cache_size)
    evidence = {}
    members = pool_initializer(problem, cfg, evidence)(gw, RNGManager(spec["seed"]))
    assert evidence["legacy_initial_population_sha256"] == expected["initial_population_sha256"]
    assert evidence["initial_archive_sha256"] == expected["initial_archive_sha256"]
    assert evidence["retention"]["pool_count"] == gw.counts["exact"] == 140
    front = set(nondominated([c.evaluation.objectives for c in gw.archive.members]))
    pop_points = {c.evaluation.objectives for c in members}
    assert front <= pop_points
    for c in (members[0], members[-1], gw.archive.members[-1]):
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
    frame = FrozenFrame.of(list(front))
    row = dict(
        key=key,
        n=spec["n"],
        seed=spec["seed"],
        evidence=evidence,
        exact_checks=3,
        archive_points=len(front),
        archive_covered=sum(p in pop_points for p in front),
        population_hv=frame.hv(list(pop_points)),
        archive_hv=frame.hv(list(front)),
    )
    (out / (key + ".json")).write_text(json.dumps(row, indent=2), encoding="utf8")
    return row


def main() -> None:
    """Check all sealed cases using bounded local workers and write aggregates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    keys = _read(args.reference_root / "formal/manifest.json")["blocks"]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_one, args.reference_root, args.output, key) for key in keys]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(row["key"], flush=True)
    summary = dict(
        cases=len(rows),
        exact_checks=sum(r["exact_checks"] for r in rows),
        all_preflight_hashes_matched=True,
        all_archive_points_retained=True,
        by_size={},
    )
    for n in (50, 100, 200):
        group = [r["evidence"] for r in rows if r["n"] == n]
        costs = [r["retention"]["selection_seconds"] for r in group]
        summary["by_size"][str(n)] = dict(
            cases=len(group),
            selection_mean_s=statistics.mean(costs),
            selection_max_s=max(costs),
            overhead_mean_s=statistics.mean(
                r["initialization_seconds"] - r["legacy_initialization_seconds"] for r in group
            ),
            selected_bill_mean=statistics.mean(
                r["retention"]["selected_origins"].get("bill_seed_BILL", 0) for r in group
            ),
        )
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
