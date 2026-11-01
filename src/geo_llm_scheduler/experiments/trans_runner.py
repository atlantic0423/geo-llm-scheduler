"""Local CLI for four S4 arms and four attributed native-method P/D adaptations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_ablation
from geo_llm_scheduler.experiments.trans_cmoeaaop import run_cmoeaaop
from geo_llm_scheduler.experiments.trans_cmoeacd import run_cmoeacd
from geo_llm_scheduler.experiments.trans_cmoflcs import run_cmoflcs
from geo_llm_scheduler.experiments.trans_common import Individual, SearchContext
from geo_llm_scheduler.experiments.trans_pslgp import run_pslgp
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch
from geo_llm_scheduler.lsv2.search import LSV2Search

NATIVE: dict[str, Callable[[SearchContext, int], list[Individual]]] = {
    "CMOEA_CD_PD": run_cmoeacd,
    "CMOEA_AOP_PD": run_cmoeaaop,
    "CMOFLCS_PD": run_cmoflcs,
    "PSLGP_H_PD": run_pslgp,
}
ALGORITHMS = (*s4_ablation.ARMS, *NATIVE)


def digest(path: Path) -> str:
    """SHA256 a local input or artifact without depending on modification timestamps."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    """Atomically persist finite JSON; NaN/Infinity never masquerade as valid evidence."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def source_record() -> dict:
    """Record Git identity and all package-file hashes, including uncommitted development."""
    root = Path(__file__).resolve().parents[3]
    git: dict = {}
    for name, command in (
        ("commit", ["rev-parse", "HEAD"]),
        ("branch", ["branch", "--show-current"]),
        ("status", ["status", "--porcelain"]),
    ):
        result = subprocess.run(
            ["git", "-C", str(root), *command], capture_output=True, text=True, check=False
        )
        git[name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    git["files"] = {
        str(p.relative_to(root)).replace("\\", "/"): digest(p)
        for p in sorted((root / "src").rglob("*.py"))
    }
    return git


def run_case(
    algorithm: str,
    instance: Path,
    output: Path,
    *,
    seed: int = 16901,
    seconds: float | None = None,
    calibration: dict | None = None,
    population_size: int | None = None,
    max_generations: int | None = None,
    wait_genes: bool = True,
) -> dict:
    """Run a single non-overwriting case, persist exact schedules and independent audit.

    Calibration is mandatory for S4: the runner never invents reward/state scales.
    Native population defaults are retained (100; PSLGP 1000). Size overrides
    and generation limits are explicitly labeled engineering checks.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown algorithm")
    problem = load_instance(instance)
    budget = seconds if seconds is not None else s4_ablation.BUDGETS[len(problem.jobs)]
    if budget <= 0 or not float("-inf") < budget < float("inf"):
        raise ValueError("Finite positive budget required")
    if algorithm in s4_ablation.ARMS and (
        calibration is None or not {"rho", "h", "w_cutoff"} <= calibration.keys()
    ):
        raise ValueError("Supply an explicit frozen S4 calibration record")
    if algorithm in s4_ablation.ARMS and population_size is not None:
        raise ValueError("S4 ablations retain their population of 100")
    output.mkdir(parents=True, exist_ok=False)
    record: dict = {
        "protocol": "s4-trans-python/v1",
        "algorithm": algorithm,
        "seed": seed,
        "instance": instance.name,
        "instance_sha256": digest(instance),
        "jobs": len(problem.jobs),
        "seconds_including_initialization": budget,
        "calibration": calibration,
        "population": population_size or (1000 if algorithm == "PSLGP_H_PD" else 100),
        "max_generations": max_generations,
        "engineering_override": population_size is not None or max_generations is not None,
        "representation": "two_tree_dispatch"
        if algorithm == "PSLGP_H_PD"
        else f"{'4N' if wait_genes else '3N'}_keys"
        if algorithm in NATIVE
        else "S4_MS_OS_timing",
        "archive_policy": "all_complete_feasible_exact_candidates; passive for native ports",
        "source": source_record(),
    }
    write_json(output / "config.json", record)
    write_json(output / "status.json", {"status": "running"})
    try:
        evidence: dict = {}
        if algorithm in s4_ablation.ARMS:
            cfg = s4_ablation.arm_config(algorithm, seed, budget)
            if max_generations is not None:
                cfg = replace(cfg, generations=max_generations)
            assert calibration is not None
            write_json(
                output / "algorithm_config.json",
                s4_ablation.configuration_record(algorithm, cfg, calibration),
            )
            searches: list[LSV2Search] = []
            result = s4_ablation.run_arm(
                problem, algorithm, cfg, calibration, evidence, search_sink=searches.append
            )
            gateway, population, trace, elapsed = (
                result.gateway,
                result.population,
                result.trace,
                result.elapsed,
            )
            write_json(output / "initialization.json", evidence)
            search = searches[0]
            write_json(
                output / "q_table.json",
                asdict(search.table) if isinstance(search, QLearningSearch) else None,
            )
            write_json(
                output / "local_search.json",
                {
                    "steps": search.steps,
                    "polish": search.polish_steps,
                },
            )
        else:
            ctx = SearchContext(
                problem, seed, budget, wait_genes=wait_genes, max_generations=max_generations
            )
            returned = NATIVE[algorithm](ctx, int(record["population"]))
            elapsed = perf_counter() - ctx.started
            gateway, population, trace = ctx.gateway, [x.candidate for x in returned], ctx.trace
            evidence = {"adapter_queries": ctx.evaluations, "generations": ctx.generations}
        # Independent audit after the budget; no additional candidate search or selection.
        for candidate in gateway.archive.members + population:
            checked = evaluate(problem, candidate.genotype, candidate.schedule)
            if not checked.feasible or checked != candidate.evaluation:
                raise AssertionError("Independent exact audit failed")
        write_json(output / "archive.json", [asdict(c) for c in gateway.archive.members])
        write_json(output / "native_population.json", [asdict(c) for c in population])
        write_json(output / "trace.json", trace)
        summary = {
            "status": "complete",
            "algorithm": algorithm,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "budget_seconds": budget,
            "overrun_seconds": max(0.0, elapsed - budget),
            "archive_size": len(gateway.archive.members),
            "population_size": len(population),
            "counts": dict(gateway.counts),
            "component_seconds": dict(gateway.seconds),
            "independent_exact_audit": "passed",
            "run_evidence": evidence,
        }
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "complete"})
        write_json(
            output / "manifest.json",
            [
                {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(output.glob("*.json"))
            ],
        )
        return summary
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": repr(exc)})
        raise


def main() -> None:
    """Parse one experiment task; never launch an implicit matrix or server workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=16901)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-key")
    parser.add_argument("--population", type=int)
    parser.add_argument("--max-generations", type=int)
    parser.add_argument("--earliest-only", action="store_true")
    args = parser.parse_args()
    calibration = (
        json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    )
    if args.calibration_key:
        if calibration is None:
            parser.error("--calibration-key requires --calibration")
        calibration = calibration[args.calibration_key]
    summary = run_case(
        args.algorithm,
        args.instance,
        args.output,
        seed=args.seed,
        seconds=args.seconds,
        calibration=calibration,
        population_size=args.population,
        max_generations=args.max_generations,
        wait_genes=not args.earliest_only,
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
