"""Whole working-version runs at a fixed wall budget: new vs frozen 582237f path.

Each (instance, path) runs in its own process (``one`` sub-command) so no state is
shared; the two paths of a block run concurrently in the same wall-clock window. The
comparison is the number of exact evaluations (and SSGS decodes and LS steps)
completed in the same wall budget. This
measures throughput only; the search trajectories legitimately diverge once more
evaluations fit in the budget.

Usage:
    python -m research.throughput_20261109.benchmark_runs plan CAMPAIGN_ROOT
        --keys BLOCK1 BLOCK2 ... --seconds 120 --seed 13 --repeats 2 --out data/benchmark_runs.json
where CAMPAIGN_ROOT is a ``formal_panel_shard*`` root whose specs name each block's input
and calibration fold.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.io.loaders import load_instance

from .reference_runtime import reference_runtime

ARM = "N7_P1_SHARE"


def one(instance: Path, calibration: dict, seconds: float, seed: int, path: str) -> dict:
    """Run the working version once on ``path`` ("new" or "reference") and count work."""
    problem = load_instance(instance)
    config = s4_moead.arm_config(ARM, seed, seconds)
    searches: list = []
    evidence: dict = {}

    def go():  # noqa: ANN202
        return s4_moead.run_arm(
            problem, ARM, config, calibration, evidence, search_sink=searches.append
        )

    if path == "reference":
        with reference_runtime():
            result = go()
    else:
        result = go()
    counts = result.gateway.counts
    return dict(
        path=path,
        exact=counts["exact"],
        ssgs=counts["ssgs"],
        ls_steps=len(searches[0].steps),
        elapsed=result.elapsed,
        archive=len(result.gateway.archive.members),
        initialization=result.gateway.seconds.get("initialization", 0.0),
        cpu=time.process_time(),
    )


def plan(root: Path, keys: list[str], seconds: float, seed: int, repeats: int) -> dict:
    """Run each block's two paths concurrently, ``repeats`` rounds; report count ratios.

    The pair shares one wall-clock window, so machine-state drift (frequency,
    background load) affects both paths alike; rounds use distinct solver seeds.
    """
    folds = json.loads((root / "calibration.json").read_text(encoding="utf8"))
    rows = []
    for key in keys:
        spec = json.loads((root / "specs" / f"{key}.json").read_text(encoding="utf8"))
        n, fold = spec["n"], folds[spec["calibration_fold"]]
        for r in range(repeats):
            procs = {
                path: subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "research.throughput_20261109.benchmark_runs",
                        "one",
                        str(root / spec["input"]),
                        json.dumps(fold),
                        str(seconds),
                        str(seed + r),
                        path,
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                )
                for path in ("new", "reference")
            }
            pair = {}
            for path, proc in procs.items():
                out, _ = proc.communicate()
                if proc.returncode:
                    raise RuntimeError(f"{key} {path} failed")
                pair[path] = json.loads(out.strip().splitlines()[-1])
            print(key, r, pair, flush=True)
            rows.append(
                dict(
                    key=key,
                    n=n,
                    round=r,
                    **pair,
                    exact_ratio=pair["new"]["exact"] / pair["reference"]["exact"],
                    ssgs_ratio=pair["new"]["ssgs"] / pair["reference"]["ssgs"],
                )
            )
    sizes = sorted({r["n"] for r in rows})
    return dict(
        arm=ARM,
        seconds=seconds,
        seed=seed,
        repeats=repeats,
        concurrent_pairs=True,
        rows=rows,
        median_exact_ratio={
            str(n): statistics.median(r["exact_ratio"] for r in rows if r["n"] == n) for n in sizes
        },
    )


def main() -> None:
    """Command-line entry: ``plan`` (driver) or ``one`` (single process run)."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("root", type=Path)
    a.add_argument("--keys", nargs="+", required=True)
    a.add_argument("--seconds", type=float, default=120.0)
    a.add_argument("--seed", type=int, default=13)
    a.add_argument("--repeats", type=int, default=2)
    a.add_argument("--out", type=Path)
    b = sub.add_parser("one")
    b.add_argument("instance", type=Path)
    b.add_argument("fold")
    b.add_argument("seconds", type=float)
    b.add_argument("seed", type=int)
    b.add_argument("path", choices=("new", "reference"))
    args = p.parse_args()
    if args.cmd == "one":
        print(
            json.dumps(
                one(args.instance, json.loads(args.fold), args.seconds, args.seed, args.path)
            )
        )
        return
    out = plan(args.root, args.keys, args.seconds, args.seed, args.repeats)
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
