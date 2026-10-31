"""Small exact probes of post-BILL4 headroom; no instance rejection or winner selection."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive, dominates
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.lsv2_study import operator_config, study_config
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import TOL, identity
from geo_llm_scheduler.utils.rng import RNGManager


def probe_candidates(members: list[Candidate], count: int = 5) -> list[Candidate]:
    """Pick objective-ordered quantiles without using operator outcomes."""
    if not members or count < 1:
        raise ValueError("Nonempty candidate list and positive count required")
    ranked = sorted(members, key=lambda c: (c.evaluation.objectives, identity(c)))
    indices = sorted({round(k * (len(ranked) - 1) / max(1, count - 1)) for k in range(count)})
    return [ranked[k] for k in indices]


def audit_problem(problem: ProblemInstance, seed: int = 16901, *, operators: bool = True) -> dict:
    """Initialize unchanged BILL4, verify exact, and probe five frozen frontier parents.

    Independent B=3 calls do not chain. Returns structural headroom and exact
    timing witnesses, not an estimate of the global optimum or online benefit.
    """
    started = perf_counter()
    cfg = study_config("BILL_RANDOM", seed, 600)
    opcfg = operator_config(seed, 600)
    gateway = EvaluationGateway(problem, Archive())
    captured: list[Candidate] = []
    gateway.candidate_sink = captured.append
    evidence: dict = {}
    members = charged_initializer(problem, cfg, "BILL_MOEAD", evidence)(gateway, RNGManager(seed))
    assert len(members) == len({c.genotype for c in members}) == 100
    assert len(captured) == 140
    for c in captured:
        assert c.evaluation.feasible and evaluate(problem, c.genotype, c.schedule) == c.evaluation
    initial = list(gateway.archive.members)
    points = [c.evaluation.objectives for c in initial]
    # Use the entire evaluated initial population's range, avoiding a near-zero
    # span when a tiny initialization frontier makes normalized gains unstable.
    frame = FrozenFrame(
        (min(c.evaluation.flow for c in captured), min(c.evaluation.bill for c in captured)),
        (max(c.evaluation.flow for c in captured), max(c.evaluation.bill for c in captured)),
    )
    sampled = probe_candidates(initial)
    search = LSV2Search(LSV2Config(budget=3, trajectory_steps=3), cfg, opcfg)
    streams = RNGManager(seed + 1000000)
    rows = []
    witnesses = []
    all_points = list(points)
    for parent_index, c in enumerate(sampled):
        slack = [
            max(0.0, legal_bounds(problem, c.genotype, c.schedule, o)[1] - c.schedule.starts[o])
            for o in range(0, problem.operation_count, 2)
        ]
        names = ["RIGHTSHIFT"]
        if operators:
            names += ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "P2_FLOW", "P2_MIXED"]
        for name in names:
            tick = perf_counter()
            rng = streams.stream(f"probe:{parent_index}:{name}")
            batch = (
                ReusedCommonPolish().propose(problem, c, 3, opcfg, rng)
                if name == "RIGHTSHIFT"
                else search._propose(name, c, frame, rng, problem)
            )
            trial_gateway = EvaluationGateway(problem, Archive())
            trials = [trial_gateway.evaluate(p.genotype, p.schedule, name) for p in batch.proposals]
            feasible = [p for p in trials if p.evaluation.feasible]
            if name == "RIGHTSHIFT":
                for trial in trials:
                    assert trial.evaluation.feasible
                    assert trial.genotype == c.genotype
                    assert trial.schedule.starts[1::2] == c.schedule.starts[1::2]
                    assert trial.evaluation.flow == c.evaluation.flow
                    assert trial.evaluation.bill < c.evaluation.bill - TOL.cost
                for trial, delta in zip(trials, batch.diagnostics["incremental_deltas"]):
                    assert abs(trial.evaluation.bill - c.evaluation.bill - delta) < 1e-7
                if trials:
                    best = min(trials, key=lambda x: x.evaluation.bill)
                    witnesses.append(
                        {"parent_index": parent_index, "before": asdict(c), "after": asdict(best)}
                    )
            new = [p.evaluation.objectives for p in feasible]
            all_points.extend(new)
            rows.append(
                {
                    "parent_index": parent_index,
                    "operator": name,
                    "seconds": perf_counter() - tick,
                    "proposals": len(trials),
                    "feasible": len(feasible),
                    "dominates_parent": sum(dominates(p, c) for p in feasible),
                    "initial_hv_gain": frame.hv_gain(points, new),
                    "best_bill_reduction_fraction": max(
                        [
                            (c.evaluation.bill - p.evaluation.bill) / c.evaluation.bill
                            for p in feasible
                        ],
                        default=0.0,
                    ),
                    "positive_slack_jobs": sum(s > TOL.time for s in slack),
                    "slack_max_s": max(slack),
                    "slack_mean_s": sum(slack) / len(slack),
                    "diagnostics": batch.diagnostics,
                }
            )
    return {
        "status": "complete",
        "solver_seed": seed,
        "initialization": evidence,
        "initial_population_count": len(members),
        "initial_exact_checks": len(captured),
        "initial_front_count": len(initial),
        "sampled_front_count": len(sampled),
        "initial_front": points,
        "hv_frame": asdict(frame),
        "initial_hv": frame.hv(points),
        "all_probe_hv_gain": frame.hv_gain(points, all_points),
        "best_initial_flow": min(p[0] for p in points),
        "flow_lower_bound": sum(j.prefill.duration + j.decode.duration for j in problem.jobs),
        "rows": rows,
        "rightshift_witnesses": witnesses,
        "elapsed_s": perf_counter() - started,
    }
