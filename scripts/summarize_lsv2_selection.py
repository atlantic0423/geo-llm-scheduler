"""Summarize one run's rightshift_steps.json[.gz] with explicit denominators."""

import argparse
import gzip
import json
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.lsv2.selection_summary import summarize_selection


def main() -> None:
    """Read one chronological run log; preserve the original source unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.log.resolve() == args.out.resolve():
        raise ValueError("Summary output must not overwrite its source log")
    opener = gzip.open if args.log.suffix == ".gz" else open
    with opener(args.log, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError("Expected one run's chronological row array")
    atomic_json(args.out, summarize_selection(rows))


if __name__ == "__main__":
    main()
