"""Run one explicit local v4 configuration, or export its declared search space."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.experiments.component_search_v4 import (
    NUMERIC,
    SLOTS,
    Components,
    run_components,
)
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance


def main(argv: list[str] | None = None) -> None:
    """Execute only explicit inputs; write a self-contained result without overwriting files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-space", type=Path)
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--generations", type=int, help="Local smoke-test cap only")
    args = parser.parse_args(argv)
    if args.export_space:
        with args.export_space.open("x", encoding="utf-8") as stream:
            json.dump(
                dict(slots=SLOTS, numeric=NUMERIC, default=asdict(Components())),
                stream,
                ensure_ascii=False,
                indent=2,
            )
        return
    if not all((args.instance, args.calibration, args.output, args.seconds)):
        parser.error("--instance, --calibration, --output and --seconds are required")
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    data = json.loads(args.configuration.read_text(encoding="utf-8")) if args.configuration else {}
    if "mutation_weights" in data:
        data["mutation_weights"] = tuple(data["mutation_weights"])
    spec = Components(**data)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    evidence: dict = {}
    result = run_components(
        load_instance(args.instance),
        spec,
        args.seed,
        args.seconds,
        calibration,
        evidence,
        generations=args.generations,
    )
    payload = dict(
        protocol="component-search-v4/local",
        evidence=evidence,
        input_sha256=hashlib.sha256(args.instance.read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        termination=result.termination_reason,
        elapsed=result.elapsed,
        offspring_count=result.offspring_count,
        counts=dict(result.gateway.counts),
        population=[solution(c) for c in result.population],
        archive=[solution(c) for c in result.archive.members],
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
