"""Run one explicit S4 retention treatment; never alter the sealed 14-arm matrix."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from geo_llm_scheduler.experiments.initial_pool_retention import POLICIES
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.experiments.s4_ablation import arm_config, configuration_record, run_arm
from geo_llm_scheduler.io.loaders import load_instance


def main() -> None:
    """Reuse a frozen input/seed/calibration and include retention in wall time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Existing frozen campaign formal/ directory",
    )
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention", choices=POLICIES, default="global_pool")
    parser.add_argument("--seconds", type=float, help="Explicit local diagnostic override")
    parser.add_argument("--generations", type=int, help="Optional local smoke limit")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a separate result path")
    spec = json.loads(
        (args.reference_root / "specs" / (args.key + ".json")).read_text(encoding="utf8")
    )
    path = args.reference_root / spec["input"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["input_sha256"]
    cal = json.loads((args.reference_root / "calibration.json").read_text(encoding="utf8"))[
        spec["calibration_fold"]
    ]
    seconds = args.seconds if args.seconds is not None else spec["seconds"]
    if seconds <= 0 or (args.generations is not None and args.generations <= 0):
        parser.error("Seconds and generations must be positive")
    cfg = arm_config("A0_FULL_S4", spec["seed"], seconds)
    if args.generations is not None:
        cfg = replace(cfg, generations=args.generations)
    evidence = {}
    result = run_arm(
        load_instance(path), "A0_FULL_S4", cfg, cal, evidence, initial_retention=args.retention
    )
    record = dict(
        key=args.key,
        input_sha256=spec["input_sha256"],
        configuration=configuration_record(
            "A0_FULL_S4", cfg, cal, initial_retention=args.retention
        ),
        evidence=evidence,
        elapsed=result.elapsed,
        termination=result.termination_reason,
        offspring=result.offspring_count,
        counts=dict(result.gateway.counts),
        population=[solution(c) for c in result.population],
        archive=[solution(c) for c in result.archive.members],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as file:
        json.dump(record, file, indent=2)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                elapsed=result.elapsed,
                offspring=result.offspring_count,
                retention=args.retention,
            )
        )
    )


if __name__ == "__main__":
    main()
