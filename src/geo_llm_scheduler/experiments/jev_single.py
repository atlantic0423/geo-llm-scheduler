"""Opt-in B1 Jev treatment with one full-job probability ranking for A6."""

from __future__ import annotations

import math
import random
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.jev_candidates import DirectCandidates
from geo_llm_scheduler.experiments.jev_decisions import JevSettings, Requester, isolated_request
from geo_llm_scheduler.experiments.jev_prompts import BACKGROUND_EN
from geo_llm_scheduler.experiments.jev_search import SINGLE_ARM, JevPolicy
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.operators.structural import operation_order

PROMPT_VERSION = "jev_single_v1_b1_a6_top10"
BACKGROUND = (
    BACKGROUND_EN.replace(
        "constructs at most SIX distinct candidates", "constructs at most ONE complete candidate"
    )
    .replace(
        "The PROGRAM samples each selection independently; you return probabilities.",
        "The PROGRAM samples ordinary selections independently; you return probabilities. "
        "EXCEPTION: A6 ranks ALL Jobs once and takes the highest-probability 10 percent "
        "(at least two, capped by the Job count). A6 does NOT epsilon-sample destroy Jobs. "
        "Probability ties use the program's explicit RNG.",
    )
    .replace(
        "A6 destroy ~10%, >=2 jobs, ORIGINAL random legal repair;",
        "A6 one full-Job ranking selects top ~10%, >=2 jobs; ORIGINAL random legal repair;",
    )
    .replace(
        "Program RNG samples your Choice distribution;",
        "Program RNG samples ordinary Choice distributions; A6 instead takes the top set;",
    )
)
RANK_EN = (
    "For A6, return one probability distribution over ALL supplied Jobs, ranking their "
    "priority for joint destroy and repair from current_solution. The program takes the "
    "top selected_count distinct Jobs together, not one Job per request and not epsilon "
    "sampling. Consider the actual weight, waits, KV, assignment, tariffs, demand and "
    "history. Proxies are not exact gains. Original random legal repair follows. "
    "Every supplied Job must have a probability; probabilities must sum to one."
)
RANK_ZH = (
    "针对A6，一次返回全部给定工件的概率分布，对当前解中适合联合破坏修复的工件排序。"
    "程序一次取概率最高的selected_count个不同工件，不逐工件请求，也不做epsilon抽样。"
    "依据实际权重、等待、KV、分配、电价、需量与历史；代理指标不是精确收益。"
    "后续保留原随机合法修复；每个工件都必须有概率，概率和为一。"
)


class TopFractionCandidates(DirectCandidates):
    """Preserve other legal moves; A6 uses one ranking and one original random repair."""

    def _destroy(
        self, p: ProblemInstance, x: Candidate, budget: int, config: Config, rng: random.Random
    ) -> ProposalBatch:
        if budget != 1:
            raise ValueError("Top-fraction A6 requires exactly one candidate")
        n = len(p.jobs)
        if not n:
            return ProposalBatch()
        if n > 255:
            raise ValueError("One-request A6 Choice supports at most 255 Jobs; never truncate")
        count = min(n, max(2, math.ceil(config.a6_destroy_ratio * n)))
        options = {str(j): dict(job_id=j) for j in range(n)}
        request_index = len(self.session.records)
        answer = self.session.ask(
            dict(self.state, decision_kind="a6_job_ranking", selected_count=count),
            "selection",
            dict(type="choice", instructions=RANK_EN, criteria=options),
        )
        tie_rng = self._streams.stream("a6_top_fraction") if self._streams is not None else rng
        if answer is None:
            selected = tie_rng.sample(range(n), count)
            probabilities = None
            fallback = "uniform_subset_no_valid_ranking"
        else:
            probabilities = answer["probabilities"]
            ties = {j: tie_rng.random() for j in range(n)}
            selected = sorted(range(n), key=lambda j: (-probabilities[str(j)], ties[j]))[:count]
            fallback = None
        self.session.selections.append(
            dict(
                kind="a6_job_ranking",
                selected=selected.copy(),
                selected_option=dict(job_ids=selected.copy()),
                probabilities=probabilities,
                fallback=fallback,
                options=n,
                selected_count=count,
                selection_rule="top_fraction_rng_ties",
                request_index=request_index if len(self.session.records) > request_index else None,
            )
        )
        order = [o for o in operation_order(x.genotype) if o // 2 not in selected]
        ms = list(x.genotype.ms)
        for j in selected:
            region = rng.choice([r for r in range(len(p.regions)) if paths(p, j, r)])
            ms[2 * j : 2 * j + 2] = rng.choice(paths(p, j, region))
            a = rng.randrange(len(order) + 1)
            order.insert(a, 2 * j)
            order.insert(rng.randrange(a + 1, len(order) + 1), 2 * j + 1)
        genotype = Genotype(tuple(ms), tuple(o // 2 for o in order))
        batch = ProposalBatch(attempts=1)
        if genotype != x.genotype:
            batch.proposals.append(
                Proposal(genotype, selected=tuple(2 * j + s for j in selected for s in (0, 1)))
            )
        return batch


class SingleJevPolicy(JevPolicy):
    """Fresh per-step epsilon-greedy actions, B1 and the explicit A6 ranking exception."""

    def __init__(
        self,
        settings: JevSettings,
        seconds: float,
        seed: int,
        requester: Requester = isolated_request,
        sink: Callable[[dict], None] | None = None,
    ) -> None:
        super().__init__(settings, seconds, seed, requester, sink)
        self.session.background = BACKGROUND
        self.session.prompt_version = PROMPT_VERSION

    def candidates(self, action: int, state: dict) -> DirectCandidates:
        """Use the one-request full-job rank only for this isolated treatment."""
        return TopFractionCandidates(action, self.session, state)

    def snapshot(
        self,
        gateway: EvaluationGateway,
        candidate: Candidate | None = None,
        weight: tuple[float, float] | None = None,
    ) -> dict:
        """Bind every action/object/stop question to B1 and the A6 sampling exception."""
        state = super().snapshot(gateway, candidate, weight)
        state["sampling_rule"].update(
            treatment=SINGLE_ARM,
            candidate_budget=1,
            a6_destroy_selection="one_full_job_distribution_then_top_10_percent_rng_ties",
            a6_epsilon_sampling=False,
            scope="this_run_only; ordinary selections independent; A6 one top set",
        )
        return state
