"""Generate and run the D01 scale, pilot and multi-seed diagnostic matrices."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from geo_llm_scheduler.config import load_config
from geo_llm_scheduler.experiments.diagnostic_instances import (
    SCENARIOS,
    materialize_diagnostic_instance,
    write_manifest,
)
from geo_llm_scheduler.experiments.diagnostics import (
    DiagnosticRunSpec,
    aggregate_diagnostics,
    run_diagnostic,
)


def existing_manifest(path: Path) -> list[dict[str, object]]:
    """Read prior instance records so separate stages share one manifest."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def store_manifest(path: Path, records: list[dict[str, object]]) -> None:
    """Merge manifest records by deterministic instance identity."""
    merged = {str(row["instance_id"]): row for row in existing_manifest(path)}
    merged.update({str(row["instance_id"]): row for row in records})
    write_manifest(list(merged.values()), path)


def record_spec(
    instance_root: Path,
    phase: str,
    scenario: str,
    jobs: int,
    instance_seed: int,
    algorithm_seed: int,
    generations: int,
    variant: str | None = None,
) -> tuple[dict[str, object], DiagnosticRunSpec]:
    """Materialize one instance and bind it to an independent algorithm run."""
    record = materialize_diagnostic_instance(instance_root, scenario, jobs, instance_seed, variant)
    return record, DiagnosticRunSpec(
        phase,
        scenario,
        jobs,
        instance_seed,
        algorithm_seed,
        generations,
        str(record["path"]),
        variant,
    )


def main() -> None:
    """Execute one explicit D01 workflow stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("generate", "scale", "pilot", "main", "aggregate"), required=True
    )
    parser.add_argument("--phase-label")
    parser.add_argument("--config", default="configs/diagnostics/main/base.yaml")
    parser.add_argument("--instances", default="instances/diagnostic")
    parser.add_argument("--output", default="outputs/diagnostics")
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--job-sizes", nargs="+", type=int, default=(50, 100, 200))
    parser.add_argument("--instance-seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--algorithm-seeds", nargs="+", type=int, default=(101, 202, 303))
    parser.add_argument("--generations", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    instance_root = Path(args.instances)
    manifest_path = instance_root / "manifest.csv"
    output_root = Path(args.output)
    if args.stage == "aggregate":
        print(
            aggregate_diagnostics(
                output_root,
                phases=("main",),
                manifest=manifest_path,
                aggregate_subdir="aggregate/d01",
            )
        )
        return

    config = load_config(args.config)
    if args.seconds is not None:
        config = replace(config, seconds=args.seconds)
    generations = args.generations or config.generations
    records: list[dict[str, object]] = []
    specs: list[DiagnosticRunSpec] = []
    if args.stage == "scale":
        for jobs in args.job_sizes:
            record, spec = record_spec(
                instance_root,
                "scale",
                "D0_balanced",
                jobs,
                args.instance_seeds[0],
                args.algorithm_seeds[0],
                generations,
            )
            records.append(record)
            specs.append(spec)
    else:
        instance_seeds = (
            args.instance_seeds if args.stage in ("main", "generate") else args.instance_seeds[:1]
        )
        algorithm_seeds = args.algorithm_seeds if args.stage == "main" else args.algorithm_seeds[:1]
        phase = args.phase_label or ("main" if args.stage in ("main", "generate") else "pilot")
        variants = {1: "single_peak", 2: "multi_tied_peak", 3: "mixed_peak"}
        for scenario in SCENARIOS:
            for instance_seed in instance_seeds:
                variant = (
                    variants.get(instance_seed, "mixed_peak") if scenario == "D5_demand" else None
                )
                record = materialize_diagnostic_instance(
                    instance_root, scenario, args.jobs, instance_seed, variant
                )
                records.append(record)
                if args.stage != "generate":
                    for algorithm_seed in algorithm_seeds:
                        specs.append(
                            DiagnosticRunSpec(
                                phase,
                                scenario,
                                args.jobs,
                                instance_seed,
                                algorithm_seed,
                                generations,
                                str(record["path"]),
                                variant,
                            )
                        )
    store_manifest(manifest_path, records)
    if args.stage == "generate":
        print({"instances": len(records), "manifest": str(manifest_path)})
        return
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
            phases=(phase,),
            manifest=manifest_path,
            aggregate_subdir=f"aggregate/{'d01' if phase == 'main' else phase}",
        )
    )


if __name__ == "__main__":
    main()
