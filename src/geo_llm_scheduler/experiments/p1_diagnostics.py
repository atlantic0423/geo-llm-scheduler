"""Read-only H1-H4 counterfactuals, isolated from online RNG and run budgets."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.metrics import hypervolume
from geo_llm_scheduler.experiments.p1_observation import replacement_audit, restore_candidate
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.operators.active_pack import ActivePack, select_packing_moves
from geo_llm_scheduler.operators.base import Operator
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.scheduling.timing import move
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL, identity, less
from geo_llm_scheduler.utils.rng import RNGManager


def _archive(members: list[Candidate]) -> Archive:
    archive = Archive()
    archive.members = list(members)
    archive.peak_size = len(members)
    return archive


def prefix_audit(
    incumbent: Candidate,
    candidates: list[Candidate],
    context: NormalizationContext,
    weight: tuple[float, float],
    archive_members: list[Candidate],
) -> list[dict]:
    """Compare 3/6/10 prefixes with independent archives and only observed ideals.

    A fixed descriptive HV range is derived offline from the whole common pool;
    this range never enters scalar comparisons or any online controller.
    """
    all_points = [c.evaluation.objectives for c in archive_members + [incumbent] + candidates]
    lo = tuple(min(p[k] for p in all_points) for k in range(2))
    hi = tuple(max(p[k] for p in all_points) for k in range(2))

    def hv(members: list[Candidate]) -> float:
        points = [
            tuple(
                (c.evaluation.objectives[k] - lo[k]) / max(hi[k] - lo[k], EPS_NORM)
                for k in range(2)
            )
            for c in members
        ]
        return hypervolume(points, (1.1, 1.1))  # type: ignore[arg-type]

    before_keys = {identity(c) for c in archive_members}
    baseline_hv = hv(archive_members)
    rows = []
    for count in (3, 6, 10):
        seen = candidates[:count]
        ideal = tuple(
            min(
                [context.ideal[k]]
                + [c.evaluation.objectives[k] for c in seen if c.evaluation.feasible]
            )
            for k in range(2)
        )
        prefix_context = NormalizationContext(ideal, context.maximum)  # type: ignore[arg-type]
        initial_score = prefix_context.scalar(incumbent, weight)
        best = min(
            [initial_score]
            + [prefix_context.scalar(c, weight) for c in seen if c.evaluation.feasible]
        )
        archive = _archive(archive_members)
        for c in seen:
            archive.consider(c)
        rows.append(
            {
                "prefix": count,
                "actual_exact": len(seen),
                "ideal": ideal,
                "incumbent_scalar": initial_score,
                "best_scalar": best,
                "scalar_gain": initial_score - best,
                "net_retained": len({identity(c) for c in archive.members} - before_keys),
                "hv_gain": hv(archive.members) - baseline_hv,
                "hv_range": {"minimum": lo, "maximum": hi, "reference": [1.1, 1.1]},
            }
        )
    return rows


def _continuation(
    problem: ProblemInstance,
    candidate: Candidate,
    config: Config,
    context: NormalizationContext,
    weight: tuple[float, float],
    seed: int,
    name: str,
) -> dict:
    gateway = EvaluationGateway(problem, Archive())
    gateway.ideal = context.ideal
    current = candidate
    streams = RNGManager(seed)
    rows = []
    for step, action in enumerate((7, 8, 7, 8, 7)):
        operator: Operator = ActivePack() if action == 7 else PeakCoalition()
        start = perf_counter()
        batch = operator.propose(problem, current, 6, config, streams.stream(f"{name}:{step}"))
        construction = perf_counter() - start
        result = execute(
            batch,
            current,
            6,
            weight,
            NormalizationContext(gateway.ideal, context.maximum),
            gateway,
            f"A{action}",
        )
        accepted = result.best is not None and less(
            result.context.scalar(result.best, weight),
            result.context.scalar(current, weight),
            TOL.scalar,
        )
        if accepted:
            assert result.best is not None
            current = replace(
                result.best,
                schedule=refresh_diagnostics(problem, result.best.genotype, result.best.schedule),
            )
        rows.append(
            {
                "action": action,
                "accepted": accepted,
                "effective": result.effective,
                "attempts": batch.attempts,
                "diagnostics": batch.diagnostics,
                "construction_seconds": construction,
                "objectives": current.evaluation.objectives,
            }
        )
    stats = PolishStats()
    current = polish(problem, current, gateway, stats)
    return {
        "final": solution(current),
        "counts": dict(gateway.counts),
        "timings": dict(gateway.seconds),
        "steps": rows,
        "polish_exact": stats.exact,
    }


def diagnose_checkpoint(problem: ProblemInstance, checkpoint: dict, config: Config) -> list[dict]:
    """Generate independent common pools and H1-H4 evidence for sampled phenotypes."""
    population = [restore_candidate(c) for c in checkpoint["population"]]
    archive = [restore_candidate(c) for c in checkpoint["archive"]]
    context = NormalizationContext(tuple(checkpoint["ideal"]), tuple(checkpoint["maximum"]))
    lambdas = weights(len(population))
    operators: dict[int, Operator] = {a: StructuralOperator(a) for a in range(1, 7)}
    operators.update({7: ActivePack(), 8: PeakCoalition()})
    rows: list[dict] = []
    for sample in checkpoint["sample"]:
        if sample["index"] is None:
            rows.append({"sample": sample, "missing": True})
            continue
        index = sample["index"]
        incumbent = population[index]
        if evaluate(problem, incumbent.genotype, incumbent.schedule) != incumbent.evaluation:
            raise ValueError("Checkpoint disagrees with exact evaluator")
        name = f"P1:{checkpoint['fraction']}:{sample['stratum']}:{index}"
        streams = RNGManager(config.seed)
        redecode_gateway = EvaluationGateway(problem, Archive())
        decoded = redecode_gateway.evaluate(incumbent.genotype, origin="H1:redecode")
        row: dict = {
            "fraction": checkpoint["fraction"],
            "sample": sample,
            "incumbent": solution(incumbent),
            "redecoded": solution(decoded),
            "h1_exact": 1,
            "checkpoint_validation_exact": 1,
            "scalar_original": context.scalar(incumbent, lambdas[index]),
            "scalar_redecoded": context.scalar(decoded, lambdas[index]),
            "bill_loss_ratio": (decoded.evaluation.bill - incumbent.evaluation.bill)
            / max(incumbent.evaluation.bill, TOL.cost),
            "actions": [],
        }
        if config.method == "full":
            start = perf_counter()
            # Both starts use the same named streams; their feasible pools may differ.
            row["continuation_retained"] = _continuation(
                problem, incumbent, config, context, lambdas[index], config.seed, name
            )
            row["continuation_redecoded"] = _continuation(
                problem, decoded, config, context, lambdas[index], config.seed, name
            )
            row["continuation_seconds"] = perf_counter() - start
        for action, operator in operators.items():
            gateway = EvaluationGateway(problem, _archive(archive))
            gateway.ideal = context.ideal
            rng = streams.stream(f"{name}:A{action}")
            start = perf_counter()
            original_state = rng.getstate()
            batch = operator.propose(problem, incumbent, 10, config, rng)
            construction_seconds = perf_counter() - start
            result = execute(batch, incumbent, 10, lambdas[index], context, gateway, f"A{action}")
            evaluated = list(result.evaluated)
            common: dict = {
                "action": action,
                "attempts": batch.attempts,
                "proposal_count": len(batch.proposals),
                "exact": result.effective,
                "construction_seconds": construction_seconds,
                "timings": dict(gateway.seconds),
                "construction": batch.diagnostics,
                "instrumentation": batch.instrumentation,
                "candidates": [solution(c) for c in evaluated],
                "h2_context_basis": "frozen checkpoint ideal and population maximum",
                "replacement": [
                    replacement_audit(
                        population,
                        c,
                        context,
                        index,
                        config,
                        streams.stream(f"{name}:H2:A{action}"),
                    )
                    for c in evaluated
                ],
                "prefixes": prefix_audit(incumbent, evaluated, context, lambdas[index], archive),
            }
            if action == 7:
                audit_rng = streams.stream(f"{name}:A7-audit")
                audit_rng.setstate(original_state)
                audit = select_packing_moves(problem, incumbent, 10, audit_rng)
                selected_set = set(audit.selected)
                remaining = [m for m in audit.raw if m not in selected_set]
                raw_sample = streams.stream(f"{name}:A7-raw").sample(
                    remaining, min(40, len(remaining))
                )
                raw_gateway = EvaluationGateway(problem, Archive())
                raw = []
                for m in raw_sample:
                    schedule = move(problem, incumbent.genotype, incumbent.schedule, m[0], m[1])
                    if schedule is None:
                        raise ValueError("A7 raw move lost feasibility")
                    candidate = raw_gateway.evaluate(incumbent.genotype, schedule, "A7:raw")
                    raw.append(
                        {
                            "move": m,
                            "representative": m in audit.representatives,
                            "pool": m in audit.pool,
                            "candidate": solution(candidate),
                        }
                    )
                common["raw_audit"] = {
                    "raw_count": len(audit.raw),
                    "representatives": audit.representatives,
                    "pool": audit.pool,
                    "selected": audit.selected,
                    "extra_exact": len(raw),
                    "sample": raw,
                    "sampling_probability": len(raw) / len(remaining) if remaining else 0.0,
                    "interpretation": "uniform pre-outcome sample; sampled regret is not full-pool oracle regret",
                }
            row["actions"].append(common)
        row["identity"] = digest(identity(incumbent))
        rows.append(row)
    return rows


def diagnose_run(problem: ProblemInstance, output: Path, config: Config) -> dict:
    """Process checkpoints one at a time, with a separate diagnostic completion marker."""
    start = perf_counter()
    names = sorted(output.glob("checkpoint_*.json.gz"))
    summary: dict = {
        "checkpoints": len(names),
        "sampled": 0,
        "missing": 0,
        "common_exact": 0,
        "raw_exact": 0,
        "redecode_exact": 0,
        "validation_exact": 0,
        "continuation_exact": 0,
        "polish_exact": 0,
    }
    for name in names:
        atomic_json(
            output / "heartbeat.json",
            {"phase": "diagnostics", "checkpoint": name.name, "elapsed": perf_counter() - start},
        )
        with gzip.open(name, "rt", encoding="utf-8") as h:
            checkpoint = json.load(h)
        rows = diagnose_checkpoint(problem, checkpoint, config)
        with gzip.open(
            output / name.name.replace("checkpoint", "diagnostic"),
            "wt",
            encoding="utf-8",
            compresslevel=1,
        ) as h:
            json.dump(rows, h)
        for row in rows:
            if row.get("missing"):
                summary["missing"] += 1
                continue
            summary["sampled"] += 1
            summary["redecode_exact"] += row["h1_exact"]
            summary["validation_exact"] += row["checkpoint_validation_exact"]
            for common in row["actions"]:
                summary["common_exact"] += common["exact"]
                summary["raw_exact"] += common.get("raw_audit", {}).get("extra_exact", 0)
            for key in ("continuation_retained", "continuation_redecoded"):
                if key in row:
                    summary["polish_exact"] += row[key]["polish_exact"]
                    summary["continuation_exact"] += (
                        row[key]["counts"].get("exact", 0) - row[key]["polish_exact"]
                    )
    summary["elapsed"] = perf_counter() - start
    atomic_json(output / "diagnostics_summary.json", summary)
    return summary
