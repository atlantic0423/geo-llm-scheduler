"""Analyze one run's external-polish log; optional exact first-move replay is offline."""

import argparse
import gzip
import json
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.polish_diagnostics import inspect_start, repetition_summary


def main() -> None:
    """Require explicit snapshot indices for potentially expensive offline scans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-seconds", type=float)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--indices", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.indices and args.input is None:
        parser.error("--input is required for exact replay")
    if args.out.exists():
        parser.error("Output must be a new path")
    with gzip.open(args.log, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    result = repetition_summary(rows, args.run_seconds)
    if args.indices:
        problem = load_instance(args.input)
        result["replays"] = [dict(index=i, **inspect_start(problem, rows[i])) for i in args.indices]
    atomic_json(args.out, result)


if __name__ == "__main__":
    main()
