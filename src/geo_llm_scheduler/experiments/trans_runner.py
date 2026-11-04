"""Run one S4 arm, Transactions adaptation or historical enhanced baseline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import ext_hybrid, s4_ablation
from geo_llm_scheduler.experiments.trans_cmoeaaop import run_cmoeaaop
from geo_llm_scheduler.experiments.trans_cmoeacd import run_cmoeacd
from geo_llm_scheduler.experiments.trans_cmoflcs import run_cmoflcs
from geo_llm_scheduler.experiments.trans_common import Individual, SearchContext
from geo_llm_scheduler.experiments.trans_hybrid import HybridSearchContext
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
ALGORITHMS = (*s4_ablation.ARMS, *NATIVE, *ext_hybrid.ARMS)
HYBRID_NATIVE: dict[str, Callable[[SearchContext, int], list[Individual]]] = {
    "CMOEA_CD_HYBRID_PD": run_cmoeacd,
    "CMOEA_AOP_HYBRID_PD": run_cmoeaaop,
    "CMOFLCS_HYBRID_PD": run_cmoflcs,
}
HYBRID = tuple(HYBRID_NATIVE)
BALANCED_EXTERNAL = {name + "_B4050": name for name in ext_hybrid.ARMS}
BALANCED_NATIVE = {name + "_B4050": fn for name, fn in HYBRID_NATIVE.items()}
AVAILABLE_ALGORITHMS = (*ALGORITHMS, *HYBRID, *BALANCED_EXTERNAL, *BALANCED_NATIVE)
ALL_NATIVE = NATIVE | HYBRID_NATIVE | BALANCED_NATIVE


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
    campaign_output: bool = False,
) -> dict:
    """Run a single non-overwriting case, persist exact schedules and independent audit.

    Calibration is mandatory for S4: the runner never invents reward/state scales.
    Native population defaults are retained (100; PSLGP 1000). Size overrides
    and generation limits are explicitly labeled engineering checks.
    """
    if algorithm not in AVAILABLE_ALGORITHMS:
        raise ValueError("Unknown algorithm")
    problem = load_instance(instance)
    budget = seconds if seconds is not None else s4_ablation.BUDGETS[len(problem.jobs)]
    if budget <= 0 or not float("-inf") < budget < float("inf"):
        raise ValueError("Finite positive budget required")
    if algorithm in s4_ablation.ARMS and (
        calibration is None or not {"rho", "h", "w_cutoff"} <= calibration.keys()
    ):
        raise ValueError("Supply an explicit frozen S4 calibration record")
    if algorithm not in ALL_NATIVE and population_size is not None:
        raise ValueError("S4 and enhanced baselines retain their population of 100")
    if algorithm in (*HYBRID, *BALANCED_NATIVE) and not wait_genes:
        raise ValueError("Enhanced methods require waiting genes for Polish replay")
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
        if algorithm in ALL_NATIVE
        else "enhanced_hybrid_MS_OS"
        if algorithm in (*ext_hybrid.ARMS, *BALANCED_EXTERNAL)
        else "S4_MS_OS_timing",
        "archive_policy": "all_complete_feasible_exact_candidates; passive for native ports",
        "enhancement": (
            "EXTH 20-heuristic + 1 BILL + 79 random; old single-Prefill Polish, 20% per generation"
            if algorithm in HYBRID
            else "40 heuristic + 10 BILL (4 trials) + 50 random; old single-Prefill Polish, 20%"
            if algorithm in (*BALANCED_EXTERNAL, *BALANCED_NATIVE)
            else None
        ),
        "source": source_record(),
    }
    write_json(output / "config.json", record)
    write_json(output / "status.json", {"status": "running"})
    last_beat: float | None = None
    started = perf_counter()

    def report(gw: EvaluationGateway) -> None:
        nonlocal last_beat
        now = perf_counter()
        if last_beat is not None and now - last_beat < 30:
            return
        last_beat = now
        data = dict(
            elapsed=now - started,
            exact=gw.counts["exact"],
            front=[c.evaluation.objectives for c in gw.archive.members],
        )
        write_json(output / "heartbeat.json", data)
        with (output / "checkpoints.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, allow_nan=False) + "\n")

    def observe(event: str, data: dict) -> None:
        if "gateway" in data:
            report(data["gateway"])

    def artifact(name: str, data: object) -> None:
        if not campaign_output:
            write_json(output / (name + ".json"), data)
            return
        target = output / (name + ".json.gz")
        temporary = target.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temporary.replace(target)

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
                problem,
                algorithm,
                cfg,
                calibration,
                evidence,
                search_sink=searches.append,
                observer=observe if campaign_output else None,
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
            artifact(
                "local_search",
                {
                    "steps": search.steps,
                    "polish": search.polish_steps,
                },
            )
        elif algorithm in (*ext_hybrid.ARMS, *BALANCED_EXTERNAL):
            base_arm = BALANCED_EXTERNAL.get(algorithm, algorithm)
            cfg = ext_hybrid.arm_config(base_arm, seed, budget)
            if max_generations is not None:
                cfg = replace(cfg, generations=max_generations)
            write_json(output / "algorithm_config.json", asdict(cfg))
            result = ext_hybrid.run_arm(
                problem,
                cfg,
                base_arm,
                evidence,
                observer=observe if campaign_output else None,
                balanced=algorithm in BALANCED_EXTERNAL,
            )
            gateway, population, trace, elapsed = (
                result.gateway,
                result.population,
                result.trace,
                result.elapsed,
            )
            write_json(output / "initialization.json", evidence)
        else:
            ctx = (
                HybridSearchContext(
                    problem,
                    seconds=budget,
                    seed=seed,
                    wait_genes=wait_genes,
                    max_generations=max_generations,
                    balanced=algorithm in BALANCED_NATIVE,
                )
                if algorithm in (*HYBRID, *BALANCED_NATIVE)
                else SearchContext(
                    problem, seed, budget, wait_genes=wait_genes, max_generations=max_generations
                )
            )
            if campaign_output:
                ctx.gateway.candidate_sink = lambda _: report(ctx.gateway)
            returned = ALL_NATIVE[algorithm](ctx, int(record["population"]))
            elapsed = perf_counter() - ctx.started
            gateway, population, trace = ctx.gateway, [x.candidate for x in returned], ctx.trace
            evidence = {"adapter_queries": ctx.evaluations, "generations": ctx.generations}
            if isinstance(ctx, HybridSearchContext):
                evidence.update(initializations=ctx.hybrid_evidence, polish=ctx.polish_counts)
                write_json(output / "initialization.json", ctx.hybrid_evidence)
        # Independent audit after the budget; no additional candidate search or selection.
        for candidate in gateway.archive.members + population:
            checked = evaluate(problem, candidate.genotype, candidate.schedule)
            if not checked.feasible or checked != candidate.evaluation:
                raise AssertionError("Independent exact audit failed")
        artifact("archive", [asdict(c) for c in gateway.archive.members])
        artifact("native_population", [asdict(c) for c in population])
        artifact("trace", trace)
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
                for p in sorted(output.iterdir())
                if p.is_file()
            ],
        )
        return summary
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": repr(exc)})
        raise


def main() -> None:
    """Parse one experiment task; never launch an implicit matrix or server workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=AVAILABLE_ALGORITHMS)
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
