"""Population-20% round selection and fresh Flash action/target/stop control."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import asdict, replace
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.jev_decisions import Requester
from geo_llm_scheduler.experiments.jev_search import JevPolicy
from geo_llm_scheduler.experiments.jev_single import TopFractionCandidates
from geo_llm_scheduler.experiments.llm_search import ObservedEvaluator
from geo_llm_scheduler.experiments.llm_search import run_policy as reference_run
from geo_llm_scheduler.experiments.llm_search import study_config as reference_config
from geo_llm_scheduler.experiments.right_shift_policies import PolishPostprocess
from geo_llm_scheduler.experiments.round_decisions import RoundSession, isolated_request
from geo_llm_scheduler.experiments.round_prompts import BACKGROUND, VERSION
from geo_llm_scheduler.experiments.round_responses import ResponsesProfile
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import (
    NormalizationContext,
    maximum,
    neighborhoods,
    replace_neighbors,
    weights,
)
from geo_llm_scheduler.moead.replacement import replacement_neighborhood
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.structural import operation_order
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, preference
from geo_llm_scheduler.scheduling.resources import est_for
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, identity, less
from geo_llm_scheduler.utils.rng import RNGManager

ARM = "LLM_ROUND20"
ARMS = ("BILL_MOEAD", "MAIN_Q", ARM)
SEED_REASONS = [
    "COVERAGE_GAP",
    "STAGNATION",
    "STRUCTURAL_OPPORTUNITY",
    "TIMING_OPPORTUNITY",
    "UNDEREXPLORED",
]
STOP_REASONS = ["CONTINUE_OPPORTUNITY", "LOW_EXPECTED_GAIN", "SEARCH_COST", "NO_OPPORTUNITY"]


def study_config(arm: str, seed: int, seconds: float) -> Config:
    """Freeze references; only the new arm defers B1 local search to round boundaries."""
    if arm not in ARMS:
        raise ValueError("Unknown three-arm treatment")
    if arm != ARM:
        return reference_config(arm, seed, seconds)
    return replace(reference_config("BILL_MOEAD", seed, seconds), fixed_budget=1, rl_steps=5)


def seed_pool(
    population: list[Candidate], round_id: int
) -> tuple[list[dict], dict[str, Candidate]]:
    """Deduplicate whole phenotypes while retaining all original fine-slot bindings."""
    rows: list[dict] = []
    candidates: dict[str, Candidate] = {}
    seen: dict[tuple, dict] = {}
    for index, candidate in enumerate(population):
        key = identity(candidate)
        if key in seen:
            seen[key]["allowed_direction_ids"].append(index)
            continue
        solution_id = f"r{round_id}_s{index}"
        row = dict(
            solution_id=solution_id,
            phenotype_id=hashlib.sha256(repr(key).encode()).hexdigest()[:24],
            allowed_direction_ids=[index],
            flow_s=candidate.evaluation.flow,
            bill_cny=candidate.evaluation.bill,
            tou_cny=candidate.evaluation.tou,
            demand_cny=candidate.evaluation.demand,
            mean_resource_wait_s=sum(candidate.schedule.resource_wait)
            / max(1, len(candidate.schedule.starts)),
            structure_id=hashlib.sha256(repr(candidate.genotype).encode()).hexdigest()[:24],
        )
        rows.append(row)
        candidates[solution_id] = candidate
        seen[key] = row
    return rows, candidates


def action_mask(problem: ProblemInstance, candidate: Candidate) -> dict[str, bool]:
    """Apply necessary eligibility tests, without spending hidden exact evaluations.

    True means potentially applicable: finite A7/A8 construction can still return
    empty after its full resource/peak checks. False means certainly inapplicable.
    """
    g, schedule = candidate.genotype, candidate.schedule
    order = operation_order(g)
    indices = {o: i for i, o in enumerate(order)}
    waits = [
        max(0.0, schedule.starts[o] - est_for(problem, g, schedule, o))
        for o in range(problem.operation_count)
    ]
    a1 = a2 = a3 = a4 = a5 = False
    for job in range(len(problem.jobs)):
        region = problem.instances[g.ms[2 * job]].region
        a1 |= any(m != g.ms[o] for o in (2 * job, 2 * job + 1) for m in problem.eligible(o, region))
        for a, b in paths(problem, job):
            if (a, b) != g.ms[2 * job : 2 * job + 2]:
                a2 |= problem.instances[a].region == region
                a3 |= problem.instances[a].region != region
        for o in (2 * job, 2 * job + 1):
            lower = indices[o - 1] + 1 if o % 2 else 0
            a4 |= waits[o] > TOL.time and lower < indices[o]
        a5 |= waits[2 * job] + waits[2 * job + 1] > TOL.time and indices[2 * job] > 0
    occupied = Counter(g.ms)
    a7 = any(n > 1 for n in occupied.values())
    a8 = any(r.demand_rate > 0 for r in problem.regions) and candidate.evaluation.bill > 0
    return {
        f"A{i}": bool(v) for i, v in enumerate((a1, a2, a3, a4, a5, bool(problem.jobs), a7, a8), 1)
    }


class RoundPolicy(JevPolicy):
    """Reuse observable full-job snapshots and legal target builders, not Jev HTTP."""

    session: RoundSession

    def __init__(
        self,
        config: Config,
        requester: Requester = isolated_request,
        sink: Callable[[dict], None] | None = None,
        observer: Callable[[str, dict], None] | None = None,
        *,
        profile: ResponsesProfile | None = None,
    ) -> None:
        from geo_llm_scheduler.experiments.jev_decisions import JevSettings

        super().__init__(
            JevSettings(epsilon_g=0.9),
            config.seconds or 1,
            config.seed,
            requester,
            exclude_api_wait=True,
        )
        self.session = RoundSession(config.seconds or 1, requester, sink, profile=profile)
        self.model_label = profile.model if profile else "flash"
        self.begin = self.session.begin
        self.config, self.observer = config, observer
        self.rounds: list[dict] = []
        self.last_ls: dict[str, dict] = {}
        self.current_round = 0

    def snapshot(
        self,
        gateway: EvaluationGateway,
        candidate: Candidate | None = None,
        weight: tuple[float, float] | None = None,
    ) -> dict:
        """Use the latest actual incumbent and full lossless English operation tables."""
        state = super().snapshot(gateway, candidate, weight)
        state["sampling_rule"].update(
            treatment=ARM, candidate_budget=1, a6_epsilon_sampling=False, a6_rule="top_10_percent"
        )
        state["budget"].update(
            search_elapsed_s=self.elapsed(),
            search_remaining_s=max(0.0, self.seconds - self.elapsed()),
        )
        state.update(round_id=self.current_round, seed_pool_policy="population_reentry20")
        front = sorted({c.evaluation.objectives for c in gateway.archive.members})
        state["archive_representatives"] = (
            [front[i] for i in sorted({round(k * (len(front) - 1) / 9) for k in range(10)})]
            if front
            else []
        )
        return state

    def notify(
        self, event: str, values: dict, population: list[Candidate], gateway: EvaluationGateway
    ) -> None:
        """Keep actual wall and search clocks available to every production observer."""
        if self.observer:
            self.observer(
                event,
                dict(
                    population=population,
                    gateway=gateway,
                    elapsed=time.perf_counter() - self.begin,
                    search_elapsed_s=self.elapsed(),
                    api_blocked_s=self.session.excluded_api_wait_s,
                    **values,
                ),
            )

    def improve_seed(
        self,
        seed: Candidate,
        index: int,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
    ) -> Candidate:
        """Execute at most B1/L5; ask a fresh action and a post-execution boolean stop."""
        current = seed
        lambdas = weights(len(population))
        group = PREFERENCE_LABELS[preference(index, len(population))]
        records: list[dict] = []
        for step in range(self.config.rl_steps):
            if self.session.expired():
                break
            started, wall_started = self.elapsed(), time.perf_counter()
            state = self.snapshot(gateway, current, lambdas[index])
            state["current_subproblem"].update(
                index=index,
                preference_group=group,
                normalization=asdict(NormalizationContext(gateway.ideal, maximum(population))),
            )
            state["trajectory"] = dict(
                next_step=step + 1, step_cap=5, completed_steps=records.copy()
            )
            mask = action_mask(gateway.problem, current)
            state["eligibility_contract"] = (
                "False excludes; True is necessary, construction may still be empty"
            )
            if not any(mask.values()) or self.session.expired():
                break
            answer = self.session.request("STEP_ACTION", state, mask)
            scores = answer["action_scores"] if answer else {k: int(v) for k, v in mask.items()}
            rng = streams.stream("round_action")
            if rng.random() < 0.9:
                selected = rng.choice([k for k, v in scores.items() if v == max(scores.values())])
            else:
                selected = rng.choices(list(scores), weights=list(scores.values()), k=1)[0]
            action = int(selected[1:])
            state["current_action"] = selected
            before_selection = len(self.session.selections)
            tick = self.elapsed()
            batch = TopFractionCandidates(action, self.session, state).propose(
                gateway.problem, current, 1, self.config, streams.stream(f"round_targets_A{action}")
            )
            gateway.seconds[f"construction:A{action}"] += self.elapsed() - tick
            gateway.counts[f"construction_attempts:A{action}"] += batch.attempts
            gateway.counts[f"proposals:A{action}"] += len(batch.proposals)
            if self.session.expired():
                break
            observed = ObservedEvaluator(gateway)
            hv_before = self.archive_value(gateway.archive.members)
            result = execute(
                batch,
                current,
                1,
                lambdas[index],
                NormalizationContext(gateway.ideal, maximum(population)),
                observed,
                selected,
            )
            before = result.context.scalar(current, lambdas[index])
            prior = current
            if result.best is not None and less(
                result.context.scalar(result.best, lambdas[index]), before, TOL.scalar
            ):
                current = result.best
                if action in (7, 8):
                    current = replace(
                        current,
                        schedule=refresh_diagnostics(
                            gateway.problem, current.genotype, current.schedule
                        ),
                    )
            after = result.context.scalar(current, lambdas[index])
            accepted = current is not prior
            cost = self.elapsed() - started
            hv_gain = max(0.0, self.archive_value(gateway.archive.members) - hv_before)
            stats = self.stats[group].setdefault(selected, {})
            for key, value in dict(
                calls=1,
                candidate_attempts=batch.attempts,
                complete_exact_candidates=result.exact_evaluations,
                feasible_exact_candidates=result.feasible_count,
                local_accepts=int(accepted),
                archive_new_objective_events=observed.new_objectives,
                calls_with_archive_gain=int(observed.new_objectives > 0),
                no_candidate_calls=int(not result.feasible_count),
                dominated_archive_members=observed.dominated_members,
                wall_s=cost,
                scalar_gain=max(0.0, before - after),
                online_hv_gain=hv_gain,
            ).items():
                stats[key] = stats.get(key, 0) + value
            self.observed = True
            terminal = step + 1 == self.config.rl_steps or self.session.expired()
            row = dict(
                round_id=self.current_round,
                subproblem=index,
                weight=lambdas[index],
                step=step,
                action=action,
                accepted=accepted,
                action_source=self.model_label if answer else "uniform_legal_fallback",
                exact_evaluations=result.exact_evaluations,
                feasible=result.feasible_count,
                candidate_objectives=[c.evaluation.objectives for c in result.evaluated],
                candidate_feasible=[c.evaluation.feasible for c in result.evaluated],
                comparison_ideal=result.context.ideal,
                comparison_maximum=result.context.maximum,
                scalar_before=before,
                scalar_after=after,
                archive_new_objective_events=observed.new_objectives,
                archive_hv_gain=hv_gain,
                action_seconds=cost,
                selection_start=before_selection,
                selection_end=len(self.session.selections),
                stop=None,
                terminal=terminal,
                stop_reason="depth_or_deadline" if terminal else None,
            )
            if not terminal:
                stop_state = self.snapshot(gateway, current, lambdas[index])
                stop_state["trajectory_feedback"] = [*records, row.copy()]
                stop_answer = self.session.request(
                    "STEP_STOP", stop_state, {}, reason_codes=STOP_REASONS
                )
                row["stop"] = stop_answer["stop"] if stop_answer else True
                terminal = bool(row["stop"]) or self.session.expired()
                row["stop_reason"] = (
                    (self.model_label + "_stop" if stop_answer else "stop_fallback")
                    if terminal
                    else None
                )
            row.update(
                terminal=terminal,
                search_seconds=self.elapsed() - started,
                seconds=time.perf_counter() - wall_started,
            )
            records.append(row)
            self.notify("rl_step", row, population, gateway)
            if terminal:
                break
        return current

    def generation(
        self,
        generation: int,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        stagnation: list[int],
    ) -> None:
        """Freeze a full-round P100 pool and refine exactly its 20% selected quota."""
        if self.session.expired():
            return
        started = self.elapsed()
        self.current_round = generation + 1
        rows, candidates = seed_pool(population, self.current_round)
        context = NormalizationContext(gateway.ideal, maximum(population))
        lambdas = weights(len(population))
        for row in rows:
            candidate = candidates[row["solution_id"]]
            row["total_kv_delay_s"] = sum(
                job.kv_delay * (candidate.genotype.ms[2 * j] != candidate.genotype.ms[2 * j + 1])
                for j, job in enumerate(gateway.problem.jobs)
            )
            row["directions"] = [
                dict(
                    direction_id=i,
                    weight=lambdas[i],
                    stagnation=stagnation[i],
                    scalar=context.scalar(candidates[row["solution_id"]], lambdas[i]),
                )
                for i in row["allowed_direction_ids"]
            ]
            row["last_ls"] = self.last_ls.get(row["phenotype_id"])
        count = min(len(rows), math.ceil(0.2 * len(population)))
        if not count or self.session.expired():
            return
        state = self.snapshot(gateway)
        state["candidate_solutions"] = rows
        state["population_size_before_dedup"] = len(population)
        answer = self.session.request(
            "ROUND_SELECT",
            state,
            {},
            limits=dict(selected_count=count, fraction=0.2, max_depth=5, candidate_budget=1),
            reason_codes=SEED_REASONS,
        )
        if answer:
            selected = answer["selected"]
        else:
            selected = [
                dict(
                    solution_id=row["solution_id"],
                    direction_id=row["allowed_direction_ids"][0],
                    reason_code="UNDEREXPLORED",
                )
                for row in streams.stream("round_seed_fallback").sample(rows, count)
            ]
        round_row = dict(
            round_id=self.current_round,
            selected=selected,
            quota=count,
            pool_size=len(rows),
            population_size=len(population),
            started=0,
            finished=0,
            source=self.model_label if answer else "uniform_legal_fallback",
        )
        self.rounds.append(round_row)
        neighbor_map = neighborhoods(lambdas, self.config.neighborhood)
        for item in selected:
            if self.session.expired():
                break
            index = item["direction_id"]
            seed = candidates[item["solution_id"]]
            round_row["started"] += 1
            gateway.counts["round_ls_started"] += 1
            current = self.improve_seed(seed, index, population, gateway, streams)
            if not self.session.expired():
                current = PolishPostprocess("TWO_FOCUS")(current, True, gateway, streams)
            ctx = NormalizationContext(gateway.ideal, maximum(population))
            neighbors = replacement_neighborhood(
                self.config.replacement_policy, index, current, neighbor_map, lambdas, ctx, streams
            )
            previous = population[index]
            replace_neighbors(
                population, current, neighbors, lambdas, ctx, self.config.replacement_cap
            )
            if less(
                ctx.scalar(population[index], lambdas[index]),
                ctx.scalar(previous, lambdas[index]),
                TOL.scalar,
            ):
                stagnation[index] = 0
            self.last_ls[hashlib.sha256(repr(identity(seed)).encode()).hexdigest()[:24]] = dict(
                search_elapsed_s=self.elapsed(), round_id=self.current_round
            )
            round_row["finished"] += 1
        gateway.seconds["round_local_search"] += self.elapsed() - started
        self.notify("round_complete", dict(round_record=round_row), population, gateway)


def run_policy(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    requester: Requester = isolated_request,
    request_sink: Callable[[dict], None] | None = None,
    observer: Callable[[str, dict], None] | None = None,
    initial: list[Genotype] | None = None,
    profile: ResponsesProfile | None = None,
) -> RunResult:
    """Run two unchanged references or the explicitly frozen population-reentry treatment."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    if arm != ARM:
        return reference_run(problem, arm, config, evidence, observer=observer)
    expected = study_config(arm, config.seed, config.seconds or 1)
    if initial is None and replace(config, generations=expected.generations) != expected:
        raise ValueError("Undeclared production configuration")
    policy = RoundPolicy(config, requester, request_sink, observer, profile=profile)
    factory = charged_initializer(problem, config, "FULL_CURRENT_I4", evidence)

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        policy.begin = policy.session.begin = time.perf_counter()
        members = (
            factory(gateway, streams) if initial is None else [gateway.evaluate(g) for g in initial]
        )
        policy.initial_context = NormalizationContext(gateway.ideal, maximum(members))
        if observer:
            observer(
                "initial_ready",
                dict(
                    population=members,
                    gateway=gateway,
                    search_elapsed_s=policy.elapsed(),
                    elapsed=policy.elapsed(),
                ),
            )
        return members

    def observe(event: str, data: dict) -> None:
        if event == "initialization":
            policy.begin = policy.session.begin = time.perf_counter() - data["elapsed"]
        if observer:
            observer(
                event,
                dict(
                    data,
                    search_elapsed_s=data.get("elapsed", 0) - policy.session.excluded_api_wait_s,
                    api_blocked_s=policy.session.excluded_api_wait_s,
                ),
            )

    result = run(
        problem,
        config,
        initializer=initialize,
        observer=observe,
        retain_trace=False,
        safe_boundary_deadline=True,
        budget_excluded_seconds=lambda: policy.session.excluded_api_wait_s,
        generation_transform=policy.generation,
    )
    evidence["round_llm"] = dict(
        prompt_version=profile.version if profile else VERSION,
        model=profile.model if profile else "deepseek-flash",
        background_sha256=hashlib.sha256(BACKGROUND.encode()).hexdigest(),
        entry_policy="population_reentry20",
        fraction=0.2,
        rounds=policy.rounds,
        valid_by_task=policy.session.valid_by_task,
        fallback_by_task=policy.session.fallback_by_task,
        requests=len(policy.session.records),
        input_tokens=policy.session.input_tokens,
        output_tokens=policy.session.output_tokens,
        input_tokens_or_reserved=policy.session.charged_or_reserved_tokens,
        api_blocked_s=policy.session.excluded_api_wait_s,
        production_wall_s=result.elapsed,
        search_elapsed_s=result.elapsed - policy.session.excluded_api_wait_s,
        intervals=policy.session.transport_intervals,
        selections=policy.session.selections,
    )
    return result
