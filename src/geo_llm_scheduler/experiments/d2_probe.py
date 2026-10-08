"""Matched source interventions, independent exact audits and D1 cost twins."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.d1_routing import routing_probe
from geo_llm_scheduler.experiments.d2_observation import control_features, opportunity_features
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.moead.variation import reproduce
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import Operator, Proposal, ProposalBatch
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.utils.numeric import TOL, identity, less
from geo_llm_scheduler.utils.rng import RNGManager

ACTIONS = tuple(range(9))


def action_probe(
    problem: ProblemInstance,
    source: Candidate,
    mate: Candidate,
    action: int,
    config: Config,
    weight: tuple[float, float],
    context: NormalizationContext,
    seed: int,
) -> dict:
    """Execute one prespecified action with B=3 and its own updated ideal.

    Action zero varies only the first parent, retaining a fixed second parent
    and common random seed. A1-A8 are separate donor-opportunity diagnostics.
    There is no time-based acceptance censoring in this fixed-exact panel.
    """
    if action not in ACTIONS:
        raise ValueError("Unknown diagnostic action")
    tick = perf_counter()
    if action == 0:
        batch = ProposalBatch(
            [
                Proposal(
                    reproduce(
                        problem,
                        source.genotype,
                        mate.genotype,
                        config,
                        RNGManager(seed).stream(f"D2:variation:{rep}"),
                    )
                )
                for rep in range(3)
            ],
            3,
        )
    else:
        operator: Operator = (
            StructuralOperator(action)
            if action <= 6
            else ActivePack()
            if action == 7
            else PeakCoalition()
        )
        batch = operator.propose(problem, source, 3, config, RNGManager(seed).stream("D2:action"))
    construction = perf_counter() - tick
    gateway = EvaluationGateway(problem, Archive())
    gateway.ideal = context.ideal
    tick = perf_counter()
    result = execute(batch, source, 3, weight, context, gateway, f"D2:A{action}")
    evaluation_seconds = perf_counter() - tick
    baseline = mate if action == 0 else source
    candidates = [baseline] + [c for c in result.evaluated if c.evaluation.feasible]
    best = min(candidates, key=lambda c: (result.context.scalar(c, weight), identity(c)))
    retained = (
        best
        if less(
            result.context.scalar(best, weight), result.context.scalar(baseline, weight), TOL.scalar
        )
        else baseline
    )
    audited = [source, *result.evaluated]
    tick = perf_counter()
    for candidate in audited:
        if evaluate(problem, candidate.genotype, candidate.schedule) != candidate.evaluation:
            raise ValueError("D2 independent exact mismatch")
    return {
        "action": action,
        "retained": solution(retained),
        "selection_context": asdict(result.context),
        "outputs": [solution(c) for c in result.evaluated],
        "exact": result.effective,
        "attempts": batch.attempts,
        "empty": result.effective == 0,
        "improved": less(
            result.context.scalar(retained, weight),
            result.context.scalar(baseline, weight),
            TOL.scalar,
        ),
        "construction_seconds": construction,
        "evaluation_seconds": evaluation_seconds,
        "seconds": construction + evaluation_seconds,
        "audit_checks": len(audited),
        "audit_seconds": perf_counter() - tick,
    }


def probe_panel(problem: ProblemInstance, panel: dict, config: Config, seed: int) -> dict:
    """Extract all prospective features before generating any matched labels."""
    context = NormalizationContext(
        tuple(panel["context"]["ideal"]), tuple(panel["context"]["maximum"])
    )
    weight = tuple(panel["weight"])
    mate = restore_candidate(panel["mate"])
    if evaluate(problem, mate.genotype, mate.schedule) != mate.evaluation:
        raise ValueError("Mate failed exact verification")
    sources = [restore_candidate(item["candidate"]) for item in panel["sources"]]
    prepared = []
    for source, item in zip(sources, panel["sources"]):
        if evaluate(problem, source.genotype, source.schedule) != source.evaluation:
            raise ValueError("Prospective source failed exact verification")
        tick = perf_counter()
        control = control_features(problem, source, mate, panel, item)
        control_seconds = perf_counter() - tick
        tick = perf_counter()
        opportunities = opportunity_features(problem, source)
        prepared.append(
            {
                "slot": item["slot"],
                "controls": control,
                "opportunities": opportunities,
                "control_seconds": control_seconds,
                "feature_seconds": perf_counter() - tick,
                "history_calls": item["history_calls"],
            }
        )
    rows = []
    for action in ACTIONS:
        # Randomized order controls systematic machine/thermal ordering.
        order = list(range(len(sources)))
        RNGManager(seed).stream(f"D2:source_order:{action}").shuffle(order)
        for index in order:
            row = action_probe(
                problem,
                sources[index],
                mate,
                action,
                config,
                weight,
                context,
                seed + action * 100003,
            )
            rows.append({**prepared[index], **row, "index": index})
    # Comparison scores evaluate retained outputs without reselecting proposals
    # using candidates belonging to another source.
    for action in ACTIONS:
        subset = [row for row in rows if row["action"] == action]
        all_feasible = [source.evaluation.objectives for source in sources]
        all_feasible += [
            (c["evaluation"]["flow"], c["evaluation"]["tou"] + sum(c["evaluation"]["demand"]))
            for row in subset
            for c in row["outputs"]
            if c["evaluation"]["feasible"]
        ]
        ideal = [
            min(context.ideal[m], mate.evaluation.objectives[m], *(v[m] for v in all_feasible))
            for m in range(2)
        ]
        common = NormalizationContext((ideal[0], ideal[1]), context.maximum)
        for row in subset:
            retained = restore_candidate(row["retained"])
            row["gain"] = common.scalar(mate, weight) - common.scalar(retained, weight)
            row["comparison_context"] = asdict(common)
    return {
        "panel": panel["panel"],
        "stage": panel["stage"],
        "preference": panel["preference"],
        "rows": rows,
        "unique_sources": len({identity(c) for c in sources}),
        "pool_size": len(sources),
        "audit_checks": 1 + len(sources) + sum(row["audit_checks"] for row in rows),
    }


def d1_cost_twins(problem: ProblemInstance, panel: dict, config: Config, seed: int) -> dict:
    """Compare reference and cheap structural guard; record every binding cap."""
    source = restore_candidate(panel["sources"][0]["candidate"])
    mate = restore_candidate(panel["mate"])
    context = NormalizationContext(
        tuple(panel["context"]["ideal"]), tuple(panel["context"]["maximum"])
    )
    targets = [
        (
            "variation",
            reproduce(
                problem,
                source.genotype,
                mate.genotype,
                config,
                RNGManager(seed).stream("D1C:variation"),
            ),
        )
    ]
    for action in (1, 4, 5):
        batch = StructuralOperator(action).propose(
            problem, source, 3, config, RNGManager(seed).stream(f"D1C:A{action}")
        )
        targets.append(
            (f"A{action}", batch.proposals[0].genotype if batch.proposals else source.genotype)
        )
    rows = []
    for i, (path, target) in enumerate(targets):
        order = [False, True]
        RNGManager(seed + i).stream("D1C:order").shuffle(order)
        twins = {}
        for fast in order:
            result = routing_probe(
                problem,
                source,
                target,
                tuple(panel["weight"]),
                context,
                seed + i,
                False,
                10 if len(problem.jobs) == 50 else 30,
                allow_unchanged=True,
                normalization="updated",
                precheck_structure=fast,
                policy_groups=("CONDITIONAL",),
            )
            record = result["groups"]["CONDITIONAL"]
            for proposal in record["proposals"]:
                from geo_llm_scheduler.domain.models import Schedule

                exact = evaluate(problem, target, Schedule(tuple(proposal["starts"])))
                if exact.feasible != proposal["feasible"] or exact.objectives != tuple(
                    proposal["objectives"]
                ):
                    raise ValueError("D1 cost twin exact mismatch")
            twins["fast" if fast else "reference"] = result
        reference, fast_result = (twins[k]["groups"]["CONDITIONAL"] for k in ("reference", "fast"))
        binding = bool(reference["overshoot_seconds"] or fast_result["overshoot_seconds"])
        same = all(
            reference[field] == fast_result[field]
            for field in ("route", "starts", "objectives", "exact")
        )
        if not binding and not same:
            raise ValueError("Cheap guard changed a nonbinding result")
        rows.append(
            {
                "path": path,
                "unchanged": target == source.genotype,
                "same": same,
                "binding": binding,
                "twins": twins,
            }
        )
    return {"panel": panel["panel"], "rows": rows}
