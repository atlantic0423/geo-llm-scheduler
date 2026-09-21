"""Generate, resume, and aggregate the independent D02 mixed validation matrix."""

import argparse
from pathlib import Path

from geo_llm_scheduler.config import load_config
from geo_llm_scheduler.experiments.diagnostic_instances import write_manifest
from geo_llm_scheduler.experiments.diagnostics import (
    DiagnosticRunSpec,
    aggregate_diagnostics,
    run_diagnostic,
)
from geo_llm_scheduler.experiments.mixed_instances import (
    MIXED_ALGORITHM_SEEDS,
    MIXED_INSTANCE_SEEDS,
    MIXED_SCENARIO,
    materialize_mixed_instance,
)


def main() -> None:
    """Execute one explicit D02 workflow stage with isolated output identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "main", "aggregate"), required=True)
    parser.add_argument("--config", default="configs/diagnostics/mixed/base.yaml")
    parser.add_argument("--instances", default="instances/diagnostic_mixed")
    parser.add_argument("--output", default="outputs/diagnostics")
    parser.add_argument("--jobs", type=int, default=50)
    parser.add_argument("--instance-seeds", nargs="+", type=int, default=MIXED_INSTANCE_SEEDS)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int, default=MIXED_ALGORITHM_SEEDS)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    instance_root = Path(args.instances)
    manifest = instance_root / "manifest.csv"
    output_root = Path(args.output)
    if args.stage == "aggregate":
        print(
            aggregate_diagnostics(
                output_root,
                phases=("mixed",),
                manifest=manifest,
                aggregate_subdir="aggregate/d02",
            )
        )
        return

    records = [
        materialize_mixed_instance(instance_root, args.jobs, seed) for seed in args.instance_seeds
    ]
    write_manifest(records, manifest)
    if args.stage == "generate":
        print({"instances": len(records), "manifest": str(manifest)})
        return

    config = load_config(args.config)
    specs = [
        DiagnosticRunSpec(
            "mixed",
            MIXED_SCENARIO,
            args.jobs,
            instance_seed,
            algorithm_seed,
            args.generations,
            str(instance_root / MIXED_SCENARIO / f"jobs{args.jobs}_seed{instance_seed}.json"),
        )
        for instance_seed in args.instance_seeds
        for algorithm_seed in args.algorithm_seeds
    ]
    for index, spec in enumerate(specs, start=1):
        summary = run_diagnostic(spec, config, output_root, resume=args.resume)
        print(
            {
                "completed": f"{index}/{len(specs)}",
                "run_id": spec.run_id,
                "elapsed": summary["elapsed"],
                "archive_size": summary["archive_size"],
            }
        )
    print(
        aggregate_diagnostics(
            output_root,
            phases=("mixed",),
            manifest=manifest,
            aggregate_subdir="aggregate/d02",
        )
    )


if __name__ == "__main__":
    main()
