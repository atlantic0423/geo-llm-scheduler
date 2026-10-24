"""Two Jev arms: fresh per-step actions, direct targets and post-step stopping."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, replace
from typing import Any, Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.jev_candidates import DirectCandidates
from geo_llm_scheduler.experiments.jev_decisions import (
    DecisionDeadline,
    DecisionSession,
    JevSettings,
    Requester,
    epsilon_sample,
    isolated_request,
)
from geo_llm_scheduler.experiments.jev_prompts import (
    ACTION_CRITERIA,
    BACKGROUND_EN,
    PROMPT_VERSION,
    PROMPTS,
)
from geo_llm_scheduler.experiments.llm_features import (
    objective_summary,
    static_problem,
)
from geo_llm_scheduler.experiments.llm_search import LLMPolicy, ObservedEvaluator
from geo_llm_scheduler.experiments.llm_search import study_config as flash_config
from geo_llm_scheduler.experiments.right_shift_policies import PolishPostprocess
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, maximum
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, preference
from geo_llm_scheduler.scheduling.resources import est_for
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, less
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = ("JEV_E05", "JEV_E09")


def study_config(arm: str, seed: int, seconds: float) -> Config:
    """Return the matched BILL4/P100 strict B6/L5 and two-focus production setup."""
    if arm not in ARMS:
        raise ValueError("Unknown Jev arm")
    return flash_config("LLM_FLASH", seed, seconds)


def _without_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_names(v) for k, v in value.items() if k != "name"}
    if isinstance(value, list):
        return [_without_names(v) for v in value]
    return value


class JevPolicy(LLMPolicy):
    """Same exact search seam as Flash; no Q learning and no free model waiting."""

    def __init__(
        self,
        settings: JevSettings,
        seconds: float,
        seed: int,
        requester: Requester = isolated_request,
        sink: Callable[[dict], None] | None = None,
    ) -> None:
        super().__init__(seconds, seed)
        self.session = DecisionSession(settings, seconds, requester, sink)
        self.begin = self.session.begin
        self.settings = settings
        self.problem_data: dict = {}

    def snapshot(
        self,
        gateway: EvaluationGateway,
        candidate: Candidate | None = None,
        weight: tuple[float, float] | None = None,
    ) -> dict:
        """Build a fact-only English-schema state with all Jobs and no future oracle."""
        if not self.problem_data:
            self.problem_data = _without_names(static_problem(gateway.problem))
        state: dict = dict(
            instance=self.problem_data,
            action_history=self.feedback(),
            budget=dict(
                total_s=self.seconds,
                remaining_s=max(0.0, self.seconds - self.elapsed()),
                wait_used_s=self.session.wait_s,
                wait_cap_s=self.settings.max_wait_fraction * self.seconds,
            ),
            archive_summary=objective_summary(gateway.archive.members),
        )
        if candidate is not None:
            p, g, s = gateway.problem, candidate.genotype, candidate.schedule
            jobs = []
            for j, job in enumerate(p.jobs):
                ops = (2 * j, 2 * j + 1)
                jobs.append(
                    dict(
                        job_id=j,
                        release_s=job.release,
                        kv_if_separate_s=job.kv_delay,
                        actual_kv_s=job.kv_delay * (g.ms[2 * j] != g.ms[2 * j + 1]),
                        operations=[
                            [
                                o,
                                "P" if o % 2 == 0 else "D",
                                g.ms[o],
                                p.instances[g.ms[o]].region,
                                s.starts[o],
                                s.end(p, o),
                                p.profile(o).duration,
                                p.profile(o).compute,
                                p.profile(o).vram,
                                s.resource_wait[o],
                                max(0.0, s.starts[o] - est_for(p, g, s, o)),
                            ]
                            for o in ops
                        ],
                    )
                )
            state.update(
                all_jobs=jobs,
                operation_columns=[
                    "operation",
                    "phase",
                    "machine",
                    "geographic_region",
                    "start_s",
                    "end_s",
                    "duration_s",
                    "compute",
                    "vram_gb",
                    "resource_wait_s",
                    "flow_wait_s",
                ],
                current_subproblem=dict(fine_weight=weight),
                current_solution=dict(
                    fingerprint=hashlib.sha256(repr((g, s.starts)).encode()).hexdigest(),
                    MS=g.ms,
                    OS=g.os,
                    starts_s=s.starts,
                    flow_s=candidate.evaluation.flow,
                    bill_cny=candidate.evaluation.bill,
                    tou_cny=candidate.evaluation.tou,
                    demand_cny=candidate.evaluation.demand,
                    all_region_windows_kw=candidate.evaluation.windows,
                ),
            )
        return state

    def update(self, data: dict) -> None:
        """Disable inherited background updates; only an entered LS step asks for actions."""

    def feedback(self) -> dict:
        """Keep cumulative completed-action feedback across fresh per-step decisions."""
        result = super().feedback()
        result["window_definition"] = "completed_LS_actions_since_run_start"
        return result

    def __call__(
        self,
        child: Candidate,
        index: int,
        stagnant_count: int,
        progress: float,
        weights: tuple[tuple[float, float], ...],
        maximum: tuple[float, float],
        gateway: EvaluationGateway,
        controller: Any,
        config: Config,
        streams: RNGManager,
        population: list[Candidate] | None = None,
        adaptive: Any = None,
        *,
        observer: Callable[[str, dict], None] | None = None,
        operator_overrides: dict | None = None,
    ) -> tuple[Candidate, list[dict]]:
        """Execute at most five steps; ask Noul after steps 1-4, never after step five."""
        if operator_overrides:
            raise ValueError("No undeclared Jev operator override")
        group = PREFERENCE_LABELS[preference(index, config.population)]
        current, failures = child, 0
        records: list[dict] = []
        for step in range(min(5, config.rl_steps)):
            if self.session.expired():
                break
            start = time.perf_counter()
            tick = time.perf_counter()
            state = self.snapshot(gateway, current, weights[index])
            state["current_subproblem"].update(
                index=index,
                preference_group=group,
                normalization=asdict(NormalizationContext(gateway.ideal, maximum)),
            )
            state["trajectory"] = dict(
                next_step=step + 1,
                step_cap=min(5, config.rl_steps),
                completed_steps=[dict(row) for row in records],
            )
            self.feature_s += time.perf_counter() - tick
            request_index = len(self.session.records)
            tick = time.perf_counter()
            try:
                answer = self.session.ask(
                    state,
                    "action",
                    dict(
                        type="choice",
                        instructions=PROMPTS["action"][0].format(group=group),
                        criteria=ACTION_CRITERIA,
                    ),
                )
            except DecisionDeadline:
                break
            action_query_s = time.perf_counter() - tick
            if self.session.expired():
                break
            action_request = request_index if len(self.session.records) > request_index else None
            probabilities = (
                answer["probabilities"]
                if answer is not None
                else {a: 1 / 8 for a in ACTION_CRITERIA}
            )
            if answer is not None:
                self.revision += 1
            action = int(
                epsilon_sample(
                    probabilities, self.settings.epsilon_g, streams.stream("jev_action")
                )[1:]
            )
            state["current_action"] = f"A{action}"
            state["budget"] = dict(
                state["budget"],
                remaining_s=max(0.0, self.seconds - self.elapsed()),
                wait_used_s=self.session.wait_s,
            )
            tick = time.perf_counter()
            selected_before = len(self.session.selections)
            batch = DirectCandidates(action, self.session, state).propose(
                gateway.problem,
                current,
                config.fixed_budget,
                config,
                streams.stream(f"jev_targets_A{action}"),
            )
            construction = time.perf_counter() - tick
            gateway.seconds[f"construction:A{action}"] += construction
            gateway.counts[f"construction_attempts:A{action}"] += batch.attempts
            gateway.counts[f"proposals:A{action}"] += len(batch.proposals)
            if self.session.expired():
                break
            measured = ObservedEvaluator(gateway)
            hv_before = self.archive_value(gateway.archive.members)
            result = execute(
                batch,
                current,
                config.fixed_budget,
                weights[index],
                NormalizationContext(gateway.ideal, maximum),
                measured,
                f"A{action}",
                structural_observer=(
                    lambda target: observer(
                        "structure",
                        dict(
                            source=current,
                            target=target,
                            path=f"A{action}",
                            step=step,
                            weight=weights[index],
                        ),
                    )
                )
                if observer
                else None,
            )
            prior = current
            before = result.context.scalar(current, weights[index])
            if result.best is not None and less(
                result.context.scalar(result.best, weights[index]), before, TOL.scalar
            ):
                current = result.best
                if action in (7, 8):
                    current = replace(
                        current,
                        schedule=refresh_diagnostics(
                            gateway.problem, current.genotype, current.schedule
                        ),
                    )
            after = result.context.scalar(current, weights[index])
            accepted = current is not prior
            failures = 0 if accepted or measured.new_objectives else failures + 1
            action_seconds = time.perf_counter() - start
            stats = self.stats[group].setdefault(f"A{action}", {})
            for key, value in dict(
                calls=1,
                candidate_attempts=batch.attempts,
                complete_exact_candidates=result.exact_evaluations,
                feasible_exact_candidates=result.feasible_count,
                local_accepts=int(accepted),
                archive_new_objective_events=measured.new_objectives,
                calls_with_archive_gain=int(measured.new_objectives > 0),
                no_candidate_calls=int(result.feasible_count == 0),
                dominated_archive_members=measured.dominated_members,
                wall_s=action_seconds,
                scalar_gain=max(0.0, before - after),
                online_hv_gain=max(0.0, self.archive_value(gateway.archive.members) - hv_before),
            ).items():
                stats[key] = stats.get(key, 0) + value
            self.observed = True
            terminal = step + 1 >= min(5, config.rl_steps) or self.session.expired()
            reason = "step_cap_or_deadline" if terminal else None
            row = dict(
                step=step,
                completed_step=step + 1,
                action=action,
                preference=group,
                epsilon_g=self.settings.epsilon_g,
                action_probabilities=probabilities,
                action_request=action_request,
                action_source="jev_current_step" if answer is not None else "uniform_fallback",
                action_query_s=action_query_s,
                weight=weights[index],
                scalar_before=before,
                scalar_after=after,
                accepted=accepted,
                candidate_objectives=[c.evaluation.objectives for c in result.evaluated],
                exact_evaluations=result.exact_evaluations,
                feasible=result.feasible_count,
                archive_new_objective_events=measured.new_objectives,
                consecutive_no_gain=failures,
                action_seconds=action_seconds,
                selection_start=selected_before,
                selection_end=len(self.session.selections),
                stop_noul=None,
            )
            if not terminal:
                tick = time.perf_counter()
                stop_state = self.snapshot(gateway, current, weights[index])
                stop_state["trajectory"] = dict(
                    row, remaining_steps=min(5, config.rl_steps) - step - 1
                )
                self.feature_s += time.perf_counter() - tick
                try:
                    answer = self.session.ask(
                        stop_state,
                        "stop",
                        dict(
                            type="noul",
                            instructions=PROMPTS["stop"][0],
                            criteria=dict(
                                true=PROMPTS["stop_true"][0], false=PROMPTS["stop_false"][0]
                            ),
                        ),
                    )
                except DecisionDeadline:
                    answer = None
                if answer is not None:
                    row["stop_noul"] = answer["noul"]
                    terminal = answer["noul"] > self.settings.stop_threshold
                    reason = "jev_stop" if terminal else None
                elif failures >= 2:
                    terminal, reason = True, "api_unavailable_local_no_gain_2"
                if self.session.expired():
                    terminal, reason = True, "deadline"
            row.update(terminal=terminal, stop_reason=reason, seconds=time.perf_counter() - start)
            records.append(row)
            if observer:
                observer("rl_step", row)
            if terminal:
                break
        gateway.seconds["jev_wait"] = self.session.wait_s
        gateway.seconds["jev_features"] = self.feature_s
        return current, records


def run_policy(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    requester: Requester = isolated_request,
    settings: JevSettings | None = None,
    observer: Callable[[str, dict], None] | None = None,
    request_sink: Callable[[dict], None] | None = None,
    initial: list[Genotype] | None = None,
) -> RunResult:
    """Run an isolated Jev arm; explicit initial genotypes support deterministic audits."""
    expected = 0.5 if arm == "JEV_E05" else 0.9
    if arm not in ARMS:
        raise ValueError("Unknown Jev arm")
    settings = settings or JevSettings(epsilon_g=expected)
    if settings.epsilon_g != expected:
        raise ValueError("Arm/epsilon mismatch")
    if (
        config.trigger_mode != "strict"
        or config.trigger_delta != 0
        or config.fixed_budget != 6
        or config.rl_steps != 5
        or not config.trigger_quality_gate
    ):
        raise ValueError("Jev requires strict/delta0/B6/L5")
    expected_config = study_config(arm, config.seed, config.seconds or 1)
    if (
        initial is None
        and replace(config, generations=expected_config.generations) != expected_config
    ):
        raise ValueError("Undeclared production configuration; use study_config")
    if requester is isolated_request and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise ValueError("TYPESAFE_API_KEY is required for a live Jev run")
    policy = JevPolicy(settings, config.seconds or 1, config.seed, requester, request_sink)
    factory = charged_initializer(problem, config, "FULL_CURRENT_I4", evidence)

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        policy.begin = policy.session.begin = time.perf_counter()
        members = (
            [gateway.evaluate(g) for g in initial]
            if initial is not None
            else factory(gateway, streams)
        )
        policy.initial_context = NormalizationContext(gateway.ideal, maximum(members))
        return members

    def observe(event: str, data: dict) -> None:
        if event == "initialization":
            policy.begin = policy.session.begin = time.perf_counter() - data["elapsed"]
        if observer:
            observer(event, data)

    result = run(
        problem,
        config,
        initializer=initialize,
        observer=observe,
        retain_trace=False,
        trajectory_override=policy,
        offspring_transform=PolishPostprocess("TWO_FOCUS"),
        safe_boundary_deadline=True,
    )
    evidence["jev"] = dict(
        arm=arm,
        settings=asdict(settings),
        prompt_version=PROMPT_VERSION,
        background_sha256=hashlib.sha256(BACKGROUND_EN.encode()).hexdigest(),
        requests=len(policy.session.records),
        valid_requests=sum(r["accepted"] for r in policy.session.records),
        action_request_mode="per_ls_step",
        action_requests=sum("action" in r["request"]["questions"] for r in policy.session.records),
        valid_action_requests=policy.revision,
        jev_treatment_valid=policy.revision > 0,
        actual_input_tokens=policy.session.input_tokens,
        actual_output_tokens=policy.session.output_tokens,
        billed_or_reserved_input_tokens=policy.session.charged_or_reserved_tokens,
        wait_s=policy.session.wait_s,
        feature_s=policy.feature_s,
        selections=policy.session.selections,
        requests_evidence=policy.session.records,
        provider_transport="official_https" if requester is isolated_request else "injected",
    )
    result.gateway.seconds["jev_wait"] = policy.session.wait_s
    result.gateway.seconds["jev_features"] = policy.feature_s
    return result
