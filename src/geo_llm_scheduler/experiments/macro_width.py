"""Isolated common-pool width design and bounded, RNG-independent call sampling."""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance, Profile
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.io.validation import validate_problem

ARMS = ("NATIVE6", "POOL3", "POOL6", "POOL10")
SCENARIOS = ("mixed", "burst", "kv_high", "resource_high", "region_asymmetric", "demand_high")


def arm_config(baseline: Config, arm: str) -> Config:
    """Keep A3/Q42/trajectory fixed and vary native versus common-pool evaluation."""
    if arm not in ARMS:
        raise ValueError("Unknown MacroSearch width arm")
    return replace(
        baseline,
        method="full",
        a3_region_policy="tariff",
        rl_state_policy="dominant",
        replacement_policy="birth",
        budget_policy="fixed",
        fixed_budget={"NATIVE6": 6, "POOL3": 3, "POOL6": 6, "POOL10": 10}[arm],
        macro_generation_budget=0 if arm == "NATIVE6" else 10,
        macro_diagnostics=True,
    )


def stress_pair(jobs: int, seed: int, scenario: str) -> tuple[ProblemInstance, ProblemInstance]:
    """Apply declared stress without tuning it against observed action incidence."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown stress scenario")
    rows = []
    for problem in generate_e15_pair(jobs, seed):
        if scenario == "burst":
            problem = replace(problem, jobs=tuple(replace(j, release=0.0) for j in problem.jobs))
        elif scenario == "kv_high":
            problem = replace(
                problem, jobs=tuple(replace(j, kv_delay=4 * j.kv_delay) for j in problem.jobs)
            )
        elif scenario == "resource_high":

            def raised(profile: Profile) -> Profile:
                return replace(
                    profile,
                    compute=min(0.95, 1.5 * profile.compute),
                    vram=min(15.5, 2 * profile.vram),
                )

            problem = replace(
                problem,
                jobs=tuple(
                    replace(j, prefill=raised(j.prefill), decode=raised(j.decode))
                    for j in problem.jobs
                ),
            )
        elif scenario == "region_asymmetric":
            problem = replace(
                problem,
                instances=tuple(
                    m for i, m in enumerate(problem.instances) if m.region != 0 or i == 0
                ),
            )
        elif scenario == "demand_high":
            problem = replace(
                problem,
                regions=tuple(replace(r, demand_rate=5 * r.demand_rate) for r in problem.regions),
            )
        horizon = (
            max(j.release for j in problem.jobs)
            + sum(j.prefill.duration + j.decode.duration + j.kv_delay for j in problem.jobs)
            + 86400
        )
        problem = replace(
            problem,
            regions=tuple(
                replace(
                    r,
                    tariffs=r.tariffs[:-1]
                    + (replace(r.tariffs[-1], end=max(horizon, r.tariffs[-1].end)),),
                )
                for r in problem.regions
            ),
        )
        validate_problem(problem)
        rows.append(problem)
    return rows[0], rows[1]


def candidate_data(candidate: Candidate) -> dict:
    """Serialize exact replay essentials without redundant cached diagnostics."""
    return {
        "ms": candidate.genotype.ms,
        "os": candidate.genotype.os,
        "starts": candidate.schedule.starts,
        "objectives": candidate.evaluation.objectives,
        "resource_wait": candidate.schedule.resource_wait,
        "intentional_wait": candidate.schedule.intentional_wait,
        "feasible": candidate.evaluation.feasible,
    }


def gzip_json(path: Path, data: object) -> None:
    """Write deterministic gzip JSON and verify decoded data before publication."""
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(raw, compresslevel=3, mtime=0))
    if gzip.decompress(path.read_bytes()) != raw:
        raise ValueError("Compressed artifact did not round-trip")


def read_json(path: Path) -> Any:
    """Read plain or gzip JSON without changing frozen files."""
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


class CallSampler:
    """Uniform reservoir per action and wall-progress third; at most 24 snapshots.

    Sampling uses its own explicit RNG and never touches the solver RNG manager.
    Missing buckets are retained as missing; no action is forced to fill them.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.counts: dict[str, int] = {}
        self.rows: dict[str, dict] = {}
        self.pending: str | None = None

    def before(self, context: dict) -> None:
        """Sample a frozen call before any candidate updates ideal or Archive."""
        self.pending = None
        bucket = f"A{context['action']}:stage{min(2, int(3 * context['progress']))}"
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        if self.rng.randrange(self.counts[bucket]):
            return
        batch = context["batch"]
        shared = context["context"]
        self.rows[bucket] = {
            "action": context["action"],
            "state": context["state"],
            "progress": context["progress"],
            "budget": context["budget"],
            "generation_budget": context["generation_budget"],
            "weight": context["weight"],
            "ideal": shared.ideal,
            "maximum": shared.maximum,
            "incumbent": candidate_data(context["source"]),
            "archive_objectives": [
                c.evaluation.objectives for c in context["gateway"].archive.members
            ],
            "attempts": batch.attempts,
            "construction_seconds": context["construction_seconds"],
            "rng_before": context["operator_rng_before"],
            "rng_after": context["operator_rng_after"],
            "proposals": [
                {
                    "ms": p.genotype.ms,
                    "os": p.genotype.os,
                    "starts": None if p.schedule is None else p.schedule.starts,
                    "selected": p.selected,
                }
                for p in batch.proposals
            ],
        }
        self.pending = bucket

    def after(self, context: dict) -> None:
        """Bind sampled calls to actual online exact values without new evaluations."""
        if self.pending is not None:
            self.rows[self.pending]["online_evaluated"] = [
                candidate_data(c) for c in context["result"].evaluated
            ]
        self.pending = None
