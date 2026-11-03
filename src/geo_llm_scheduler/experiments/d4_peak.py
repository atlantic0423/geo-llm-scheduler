"""Offline A8 recipe funnel and conservative fixed-horizon peak certificates."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.peak_coalition import (
    PeakCoalition,
    removal_potential,
    support_segments,
)
from geo_llm_scheduler.scheduling.resources import intervals
from geo_llm_scheduler.utils.numeric import TOL
from geo_llm_scheduler.utils.rng import RNGManager


def residual_certificate(
    problem: ProblemInstance, source: Candidate, members: frozenset[int], region: int
) -> dict:
    """Give a necessary peak bound only when a frozen tail fixes the global horizon.

    Uses the instance active union, fixed 900-second tail divisor and all windows.
    A conservative numerical margin permits inconclusive cases, not false pruning.
    This diagnostic never changes the recipe or suppresses its counterfactual repair.
    """
    horizon = source.schedule.horizon(problem)
    fixed = any(
        o not in members and source.schedule.end(problem, o) == horizon
        for o in range(problem.operation_count)
    )
    windows = source.evaluation.windows[region]
    segments = support_segments(
        problem, source.genotype, source.schedule, region, tuple(range(len(windows)))
    )
    lower = tuple(p - removal_potential(segments, members, b) for b, p in enumerate(windows))
    threshold = max(windows, default=0.0) - TOL.power
    denied = fixed and max(lower, default=0.0) > threshold + TOL.power * 0.001
    return {
        "applicable": fixed,
        "horizon": horizon,
        "lower_windows": lower,
        "threshold": threshold,
        "denied": denied,
    }


def audit_residual(
    problem: ProblemInstance, source: Candidate, members: frozenset[int], region: int
) -> tuple[float, ...]:
    """Independently integrate the nonselected operations and idle to the fixed tail."""
    horizon = source.schedule.horizon(problem)
    machines = [m for m, s in enumerate(problem.instances) if s.region == region]
    fixed = {
        m: [
            (a, b)
            for o, a, b in intervals(problem, source.genotype, source.schedule, m)
            if o not in members
        ]
        for m in machines
    }
    points = sorted({0.0, horizon} | {t for ops in fixed.values() for a, b in ops for t in (a, b)})
    windows = [0.0] * math.ceil(horizon / 900)
    for a, b in zip(points, points[1:]):
        midpoint = (a + b) / 2
        power = sum(
            problem.instances[m].active_kw
            if any(s <= midpoint < e for s, e in fixed[m])
            else problem.instances[m].idle_kw
            for m in machines
        )
        for w in range(len(windows)):
            windows[w] += power * max(0.0, min(b, (w + 1) * 900) - max(a, w * 900)) / 900
    return tuple(windows)


def probe_peak_panel(problem: ProblemInstance, panel: dict, config: Config, seed: int) -> dict:
    """Observe original A8, auditing all recipes; no favorable-source filtering."""
    context = NormalizationContext(
        tuple(panel["context"]["ideal"]), tuple(panel["context"]["maximum"])
    )
    weight = tuple(panel["weight"])
    rows = []
    order = list(range(len(panel["sources"])))
    RNGManager(seed).stream("D4:source_order").shuffle(order)
    checks = 0
    for index in order:
        source = restore_candidate(panel["sources"][index]["candidate"])
        if evaluate(problem, source.genotype, source.schedule) != source.evaluation:
            raise ValueError("D4 source failed independent exact")
        checks += 1
        tick = perf_counter()
        batch = PeakCoalition(diagnose=True).propose(
            problem, source, 3, config, RNGManager(seed).stream("D4:A8")
        )
        construction = perf_counter() - tick
        gateway = EvaluationGateway(problem, Archive())
        gateway.ideal = context.ideal
        tick = perf_counter()
        result = execute(batch, source, 3, weight, context, gateway, "D4:A8")
        evaluation_seconds = perf_counter() - tick
        tick = perf_counter()
        recipes = batch.diagnostics.get("trace", [])
        for recipe in recipes:
            members = frozenset(recipe["members"])
            region = batch.diagnostics["region"]
            start = perf_counter()
            certificate = residual_certificate(problem, source, members, region)
            certificate["seconds"] = perf_counter() - start
            independent = audit_residual(problem, source, members, region)
            if len(independent) != len(certificate["lower_windows"]) or any(
                abs(a - b) > TOL.power * 0.01
                for a, b in zip(independent, certificate["lower_windows"])
            ):
                raise ValueError("D4 active-union certificate mismatch")
            if certificate["denied"] and recipe["stage"] in ("proposal", "duplicate"):
                raise ValueError("D4 false rejection of strict peak reduction")
            if certificate["applicable"] and "repaired_starts" in recipe:
                schedule = Schedule(tuple(recipe["repaired_starts"]))
                if schedule.horizon(problem) != certificate["horizon"]:
                    raise ValueError("D4 frozen-tail hypothesis violated")
                exact = evaluate(problem, source.genotype, schedule)
                if not exact.feasible or any(
                    b + TOL.power * 0.01 < a for a, b in zip(independent, exact.windows[region])
                ):
                    raise ValueError("D4 residual is not a lower bound")
                checks += 1
            recipe["certificate"] = certificate
        for candidate in result.evaluated:
            if evaluate(problem, candidate.genotype, candidate.schedule) != candidate.evaluation:
                raise ValueError("D4 proposal failed independent exact")
            checks += 1
        outputs = [c for c in result.evaluated if c.evaluation.feasible]
        baseline = result.context.scalar(source, weight)
        gains = [baseline - result.context.scalar(c, weight) for c in outputs]
        bill = source.evaluation.objectives[1]
        rows.append(
            {
                "index": index,
                "action": 8,
                "source": solution(source),
                "outputs": [solution(c) for c in result.evaluated],
                "selection_context": asdict(result.context),
                "weight": weight,
                "recipes": recipes,
                "attempts": batch.attempts,
                "empty": result.effective == 0,
                "exact": result.effective,
                "bill_improved": sum(c.evaluation.objectives[1] < bill for c in outputs),
                "scalar_improved": sum(g > TOL.scalar for g in gains),
                "gain": max([0.0, *gains]),
                "construction_seconds": construction,
                "call_stage": "attempted" if batch.attempts else "no_region_or_positive_peak",
                "evaluation_seconds": evaluation_seconds,
                "audit_seconds": perf_counter() - tick,
            }
        )
    return {
        "panel": panel["panel"],
        "rows": rows,
        "audit_checks": checks,
        "mode": "observe_only_same_original_A8",
    }


def analyse_peak(root: Path) -> dict:
    """Validate the whole paired matrix and aggregate base-level failure evidence."""
    from geo_llm_scheduler.experiments.d2_campaign import frozen_manifest, validate_opportunity_job

    manifest = frozen_manifest(root)
    if manifest["kind"] != "D4_peak_diagnosis_v1":
        raise ValueError("Not a D4 matrix")
    bases: dict[int, Counter] = {}
    costs: dict[int, dict[str, float]] = {}
    for key in manifest["keys"]:
        for role in ("samples", "probes"):
            valid, reason = validate_opportunity_job(root, role, key)
            if not valid:
                raise ValueError(f"{role}/{key}: {reason}")
        spec = json.loads((root / "specs" / f"{key}.json").read_text())
        tally = bases.setdefault(spec["base_seed"], Counter())
        base_cost = costs.setdefault(spec["base_seed"], defaultdict(float))
        data = json.loads((root / "probes" / key / "data.json").read_text())
        for panel in data["panels"]:
            for row in panel["d2"]["rows"]:
                tally.update(
                    calls=1,
                    empty=int(row["empty"]),
                    exact=row["exact"],
                    bill_improved=row["bill_improved"],
                    scalar_improved=row["scalar_improved"],
                )
                for recipe in row["recipes"]:
                    cert = recipe["certificate"]
                    tally.update(
                        recipes=1, applicable=int(cert["applicable"]), denied=int(cert["denied"])
                    )
                    tally.update(
                        nonempty_recipes=int(bool(recipe["members"])),
                        prunable_recipes=int(bool(recipe["members"]) and cert["denied"]),
                    )
                    base_cost["certificate_seconds"] += cert["seconds"]
                    base_cost["repair_seconds"] += recipe.get("repair_seconds", 0.0)
                    if cert["denied"]:
                        base_cost["denied_repair_seconds"] += recipe.get("repair_seconds", 0.0)
                    tally[recipe["stage"]] += 1
    report = {
        "status": "ANALYZED",
        "source_commit": manifest["source_commit"],
        "source_hash": manifest["source_hash"],
        "bases": bases,
        "base_costs": costs,
        "total": dict(sum(bases.values(), Counter())),
        "false_rejections": 0,
        "online_benefit": "UNVERIFIED",
        "scope": manifest["scope"],
    }
    rates = np.asarray(
        [
            value["prunable_recipes"] / max(1, value["nonempty_recipes"])
            for _, value in sorted(bases.items())
        ]
    )
    if len(rates):
        rng = np.random.default_rng(2026100604)
        means = rates[rng.integers(0, len(rates), (20000, len(rates)))].mean(axis=1)
        report["prunable_nonempty_base_mean"] = float(rates.mean())
        report["prunable_nonempty_base_ci"] = np.quantile(means, (0.025, 0.975)).tolist()
        report["candidate_gate"] = (
            not manifest["pilot"] and report["prunable_nonempty_base_ci"][0] > 0.1
        )
    report["timing_scope"] = (
        "Offline after-repair diagnostics; costs do not establish online savings."
    )
    atomic_json(root / "analysis/report.json", report)
    return report
