"""Paired, bounded initialization preprocessing; defaults and operators stay unchanged."""

from __future__ import annotations

import random
from dataclasses import asdict, replace
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationCapReached, EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.runner import digest
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, close, identity, less

ARMS = ("RAW", "TIME16", "TIME64", "STRUCT16", "STRUCT64")


def select_cost_safe(
    problem: ProblemInstance, parent: Candidate, pool: list[Candidate]
) -> Candidate:
    """Keep one lowest-Bill strict improvement whose Flow does not increase.

    Selection includes the parent, is independent of pool order, and uses stable
    identity ties. This diagnostic policy is not MOEA/D weight acceptance.
    """
    eligible = [
        c
        for c in pool
        if c.evaluation.feasible
        and not less(parent.evaluation.flow, c.evaluation.flow, TOL.time)
        and less(c.evaluation.bill, parent.evaluation.bill, TOL.cost)
    ]
    if not eligible:
        return parent
    minimum = min(c.evaluation.bill for c in eligible)
    tied = [c for c in eligible if close(c.evaluation.bill, minimum, TOL.cost)]
    best = min(tied, key=lambda c: (c.evaluation.flow, identity(c)))
    return replace(best, schedule=refresh_diagnostics(problem, best.genotype, best.schedule))


def preprocess(
    problem: ProblemInstance,
    parent: Candidate,
    arm: str,
    rng: random.Random,
) -> tuple[Candidate, Archive, dict]:
    """Apply one same-parent budgeted move and independently verify every evaluation.

    TIME truncates the original deterministic Polish enumeration after 16/64
    feasible exact evaluations. It keeps MS/OS, all Decode times and all other
    operations frozen. A small budget is an enumeration prefix, not evidence
    that no other timing improvement exists.

    STRUCT interleaves up to 32 A3-tariff and 32 A5 same-parent proposals using
    separate explicit random streams, then evaluates the 16/64 prefix. Both
    budgets construct the identical maximum pool. No candidate chaining or
    evolutionary iteration occurs. Unequal realized counts are reported.
    """
    if arm not in ARMS:
        raise ValueError("Unknown initialization preprocessing arm")
    if not parent.evaluation.feasible:
        raise ValueError("Preprocessing requires a feasible exact parent")
    budget = 0 if arm == "RAW" else int(arm[-2:])
    archive = Archive()
    archive.consider(parent)
    gateway = EvaluationGateway(problem, archive, exact_cap=budget)
    pool: list[Candidate] = []
    ledger: list[dict] = []
    verify_seconds = 0.0

    def verify(candidate: Candidate) -> None:
        nonlocal verify_seconds
        tick = perf_counter()
        independent = evaluate(problem, candidate.genotype, candidate.schedule)
        verify_seconds += perf_counter() - tick
        if independent != candidate.evaluation:
            raise AssertionError("Independent preprocessing exact mismatch")
        pool.append(candidate)
        ledger.append(
            {
                "genotype_sha256": digest(asdict(candidate.genotype)),
                "starts_sha256": digest(candidate.schedule.starts),
                "feasible": independent.feasible,
                "flow": independent.flow,
                "bill": independent.bill,
                "tou": independent.tou,
                "demand": list(independent.demand),
                "origin": candidate.origin,
            }
        )

    gateway.candidate_sink = verify
    tick = perf_counter()
    bounded = False
    proposal_seconds = 0.0
    operator_stats: dict = {}
    if arm.startswith("TIME"):
        stats = PolishStats()
        try:
            polish(problem, parent, gateway, stats)
        except EvaluationCapReached:
            bounded = True
        operator_stats = asdict(stats)
    elif arm.startswith("STRUCT"):
        tick_proposals = perf_counter()
        streams = [random.Random(rng.getrandbits(128)) for _ in range(2)]
        cfg = replace(Config(), method="full", a3_region_policy="tariff")
        batches = [
            StructuralOperator(action).propose(problem, parent, 32, cfg, stream)
            for action, stream in zip((3, 5), streams, strict=True)
        ]
        proposals = []
        seen = {parent.genotype}
        for position in range(max((len(b.proposals) for b in batches), default=0)):
            for batch in batches:
                if position < len(batch.proposals):
                    proposal = batch.proposals[position]
                    if proposal.genotype not in seen:
                        seen.add(proposal.genotype)
                        proposals.append(proposal)
        proposal_seconds = perf_counter() - tick_proposals
        operator_stats = {
            "maximum_pool": len(proposals),
            "actions": [b.diagnostics for b in batches],
        }
        for proposal in proposals[:budget]:
            gateway.evaluate(proposal.genotype, origin="STRUCT")
        bounded = len(proposals) > budget
    selected = select_cost_safe(problem, parent, pool)
    elapsed = perf_counter() - tick
    check_tick = perf_counter()
    if evaluate(problem, selected.genotype, selected.schedule) != selected.evaluation:
        raise AssertionError("Selected candidate failed final independent exact")
    final_verify_seconds = perf_counter() - check_tick
    if arm.startswith("TIME"):
        if selected.genotype != parent.genotype:
            raise AssertionError("Timing preprocessing changed genotype")
        changed = [
            i
            for i, (a, b) in enumerate(zip(parent.schedule.starts, selected.schedule.starts))
            if a != b
        ]
        if len(changed) > 1 or any(i % 2 for i in changed):
            raise AssertionError("Timing preprocessing moved a frozen operation")
        if selected.evaluation.flow != parent.evaluation.flow:
            raise AssertionError("Prefill preprocessing changed Flow")
        if gateway.counts["ssgs"]:
            raise AssertionError("Timing preprocessing called SSGS")
    if less(parent.evaluation.flow, selected.evaluation.flow, TOL.time) or less(
        parent.evaluation.bill, selected.evaluation.bill, TOL.cost
    ):
        raise AssertionError("Cost-safe selection worsened an objective")
    return (
        selected,
        archive,
        {
            "arm": arm,
            "exact_budget": budget,
            "exact": gateway.counts["exact"],
            "ssgs": gateway.counts["ssgs"],
            "independent_exact_verified": len(pool),
            "selected_exact_verified": True,
            "cap_truncated": bounded,
            "accepted": identity(selected) != identity(parent),
            "counts": dict(gateway.counts),
            "seconds": dict(gateway.seconds),
            "elapsed": elapsed,
            "production_seconds": elapsed - verify_seconds,
            "independent_exact_seconds": verify_seconds + final_verify_seconds,
            "proposal_seconds": proposal_seconds,
            "operator": operator_stats,
            "ledger": ledger,
        },
    )
