"""Three-arm low-frequency LLM policy; production waits and telemetry are charged."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.llm_features import (
    objective_summary,
    preference_summaries,
    static_problem,
)
from geo_llm_scheduler.experiments.llm_focus import FocusStructural
from geo_llm_scheduler.experiments.llm_global_prompt import SYSTEM_PROMPT
from geo_llm_scheduler.experiments.llm_protocol import (
    ACTIONS,
    FOCUSES,
    GROUPS,
    POLICY_SCHEMA,
    REQUEST_SCHEMA,
    default_policies,
    parse_policy,
    probabilities,
)
from geo_llm_scheduler.experiments.metrics import hypervolume
from geo_llm_scheduler.experiments.right_shift_policies import PolishPostprocess
from geo_llm_scheduler.experiments.rl23 import run_policy as reference_run
from geo_llm_scheduler.experiments.rl23_support import study_config as reference_config
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, maximum
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import Operator
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, preference
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, close, less
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = ("BILL_MOEAD", "MAIN_Q", "LLM_FLASH")


def _policy_dump(policies: dict) -> list[dict]:
    return [
        dict(
            group=p.group,
            action_weights=dict(zip(ACTIONS, p.action_weights)),
            target_focus=p.target_focus,
            stop_after_no_gain=p.stop_after_no_gain,
            rationale=p.rationale,
        )
        for p in policies.values()
    ]


def study_config(arm: str, seed: int, seconds: float) -> Config:
    """Use identical initialization and matched historical reference paths."""
    if arm not in ARMS:
        raise ValueError("Unknown three-arm treatment")
    return reference_config(
        "BILL_PERMUTED" if arm == "BILL_MOEAD" else "STRICT_TWO_FOCUS", seed, seconds
    )


def isolated_request(payload: dict, timeout: float) -> dict:
    """Enforce one hard wall deadline including shared queue and HTTP child lifetime."""
    deadline = time.perf_counter() + timeout
    handle = None
    directory = os.environ.get("LLM_REQUEST_LOCK_DIR")
    try:
        if directory:
            fcntl = importlib.import_module("fcntl")

            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            while handle is None:
                for slot in range(16):
                    candidate = (root / f"slot{slot}.lock").open("a")
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        handle = candidate
                        break
                    except BlockingIOError:
                        candidate.close()
                if handle is None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("request_queue_timeout")
                    time.sleep(min(0.05, remaining))
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("request_deadline")
        process = subprocess.Popen(
            [sys.executable, "-m", "geo_llm_scheduler.experiments.llm_http_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            output, _ = process.communicate(
                json.dumps(dict(payload=payload, timeout=remaining), allow_nan=False),
                timeout=max(0.001, deadline - time.perf_counter()),
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise TimeoutError("http_hard_timeout") from None
        value = json.loads(output)
        if process.returncode != 0 or not value.get("ok"):
            raise RuntimeError("provider_request_failed")
        return value["response"]
    finally:
        if handle is not None:
            handle.close()


class ObservedEvaluator:
    """Preserve every exact/Archive dispatch and count actual new-objective events."""

    def __init__(self, gateway: EvaluationGateway) -> None:
        self.gateway = gateway
        self.new_objectives = 0
        self.dominated_members = 0

    @property
    def ideal(self) -> tuple[float, float]:
        """Expose the original gateway's live ideal to MacroSearch."""
        return self.gateway.ideal

    @ideal.setter
    def ideal(self, value: tuple[float, float]) -> None:
        self.gateway.ideal = value

    def evaluate(
        self, genotype: Genotype, schedule: Schedule | None = None, origin: str = "structural"
    ) -> Candidate:
        """Observe Archive changes without suppressing any complete candidate."""
        before = self.gateway.archive.members[:]
        insertions = self.gateway.archive.insertions
        result = self.gateway.evaluate(genotype, schedule, origin)
        if self.gateway.archive.insertions > insertions:
            same = any(
                close(c.evaluation.flow, result.evaluation.flow, TOL.time)
                and close(c.evaluation.bill, result.evaluation.bill, TOL.cost)
                for c in before
            )
            self.new_objectives += int(not same)
            self.dominated_members += max(0, len(before) + 1 - len(self.gateway.archive.members))
        return result


class LLMPolicy:
    """Run-local policy, measured interval feedback and safe-boundary updates."""

    def __init__(
        self,
        seconds: float,
        seed: int,
        requester: Callable[[dict, float], dict] = isolated_request,
        sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.seconds, self.seed = seconds, seed
        self.requester, self.sink = requester, sink
        self.begin = 0.0
        self.policies = default_policies()
        self.revision = 0
        self.checkpoint = 0
        self.wait_s = 0.0
        self.feature_s = 0.0
        self.activation = 0.0
        self.initial_context: NormalizationContext | None = None
        self.calls: list[dict] = []
        self.stats: dict[str, dict[str, dict]] = {g: {} for g in GROUPS}
        self.observed = False
        self.processed = 0

    def elapsed(self) -> float:
        """Use the engine's production origin, including initialization."""
        return time.perf_counter() - self.begin

    def archive_value(self, members: list[Candidate]) -> float:
        """Initialization-frozen HV proxy; no later union or true-front knowledge."""
        if self.initial_context is None:
            return 0.0
        points = [self.initial_context.normalize(c) for c in members]
        return hypervolume([p for p in points if p[0] <= 1.1 and p[1] <= 1.1], (1.1, 1.1))

    def feedback(self) -> dict:
        """Report exposure-aware time shares and benefit rates with null zero denominators."""
        total = sum(s.get("wall_s", 0.0) for group in self.stats.values() for s in group.values())
        elapsed = max(0.0, self.elapsed() - self.activation)
        groups: dict[str, dict[str, dict]] = {}
        fields = (
            "calls",
            "candidate_attempts",
            "complete_exact_candidates",
            "feasible_exact_candidates",
            "local_accepts",
            "archive_new_objective_events",
            "calls_with_archive_gain",
            "no_candidate_calls",
            "dominated_archive_members",
            "wall_s",
            "scalar_gain",
            "online_hv_gain",
        )
        for g in GROUPS:
            groups[g] = {}
            for a in ACTIONS:
                stats = {
                    k: self.stats[g].get(a, {}).get(k, 0 if self.observed else None) for k in fields
                }
                calls, wall = stats["calls"], stats["wall_s"]
                stats.update(
                    mean_wall_per_call_s=wall / calls if calls else None,
                    wall_fraction_of_all_action_time=wall / total
                    if total and wall is not None
                    else None,
                    wall_fraction_of_interval_production=wall / elapsed
                    if elapsed and wall is not None
                    else None,
                )
                for key in (
                    "local_accepts",
                    "calls_with_archive_gain",
                    "online_hv_gain",
                    "dominated_archive_members",
                    "scalar_gain",
                ):
                    stats[key + "_per_s"] = stats[key] / wall if wall else None
                groups[g][a] = stats
        return dict(
            history_status="observed" if self.observed else "unobserved",
            window_definition="since_last_valid_policy_activation",
            start_elapsed_s=self.activation if self.observed else None,
            window_s=elapsed if self.observed else None,
            by_group=groups,
        )

    def update(self, data: dict) -> None:
        """Attempt at most three updates; merge passed checkpoints without catch-up calls."""
        elapsed = self.elapsed()
        if self.checkpoint >= 3 or elapsed >= self.seconds:
            return
        if self.checkpoint and elapsed < self.seconds * self.checkpoint / 3:
            return
        checkpoint = self.checkpoint
        self.checkpoint = min(3, max(checkpoint + 1, int(3 * elapsed / self.seconds) + 1))
        timeout = min(
            20.0,
            0.02 * self.seconds,
            self.seconds - elapsed - 0.05,
            min(60.0, 0.05 * self.seconds) - self.wait_s,
        )
        if timeout <= 0.1:
            return
        tick = time.perf_counter()
        gateway = data["gateway"]
        population = data["population"]
        context = NormalizationContext(gateway.ideal, maximum(population))
        request_id = f"seed{self.seed}_request{len(self.calls)}"
        snapshot_id = hashlib.sha256(
            json.dumps(
                [gateway.counts["exact"], [c.evaluation.objectives for c in population]]
            ).encode()
        ).hexdigest()
        payload: dict = dict(
            schema_version=REQUEST_SCHEMA,
            feature_contract_version="cost_frontier_v1",
            request_id=request_id,
            snapshot_id=snapshot_id,
            update_index=checkpoint,
            budget=dict(
                total_s=self.seconds,
                elapsed_s=elapsed,
                remaining_s=self.seconds - elapsed,
                requests_used=len(self.calls),
                requests_max=3,
                llm_wait_used_s=self.wait_s,
                llm_wait_remaining_cap_s=min(60.0, 0.05 * self.seconds) - self.wait_s,
            ),
            problem=static_problem(gateway.problem),
            population_summary=objective_summary(population),
            archive_summary=objective_summary(gateway.archive.members),
            association_context=asdict(context),
            preference_summaries=preference_summaries(
                gateway.problem, population, gateway.archive.members, context
            ),
            recent_feedback=self.feedback(),
            previous_policy=_policy_dump(self.policies) if self.observed else None,
            online_hv_proxy=dict(
                context=asdict(self.initial_context) if self.initial_context else None,
                reference=[1.1, 1.1],
                range="initialization context; excludes points outside reference; not final evaluation HV",
                current=self.archive_value(gateway.archive.members),
            ),
            cost_context=dict(
                feature_seconds=self.feature_s,
                llm_wait_seconds=self.wait_s,
                polish_seconds=gateway.seconds.get("polish", 0),
                ssgs_seconds=gateway.seconds.get("ssgs", 0),
                exact_seconds=gateway.seconds.get("exact", 0),
                note="ssgs/exact overlap action wall; do not add them",
            ),
            allowed_output=dict(
                schema_version=POLICY_SCHEMA,
                focuses=FOCUSES,
                action_exploration=0.1,
                destroy_uniform_mix=0.2,
                focus_actions=["A3", "A6"],
                max_trajectory_steps=5,
                max_exact_per_step=6,
            ),
        )
        self.feature_s += time.perf_counter() - tick
        # Feature construction is charged too; refresh the remaining budget just before IO.
        elapsed = self.elapsed()
        payload["budget"].update(elapsed_s=elapsed, remaining_s=max(0.0, self.seconds - elapsed))
        timeout = min(timeout, self.seconds - elapsed - 0.05)
        record = dict(
            request=payload,
            prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            policy_revision_before=self.revision,
            activation_boundary=dict(
                processed=self.processed,
                generation=data.get("generation"),
                subproblem=data.get("subproblem"),
            ),
            accepted=False,
        )
        started = time.perf_counter()
        try:
            if timeout <= 0.1:
                raise TimeoutError("feature_budget_exhausted")
            response = self.requester(payload, timeout)
            record["response"] = response
            if (
                self.elapsed() >= self.seconds
                or response.get("finish_reason") != "stop"
                or response.get("returned_model") != "deepseek-flash"
            ):
                raise ValueError("late_truncated_or_wrong_model")
            policies = parse_policy(response["content"], request_id, snapshot_id)
            self.policies = policies
            self.revision += 1
            self.activation = self.elapsed()
            self.stats = {g: {} for g in GROUPS}
            self.observed = True
            record.update(
                accepted=True,
                policy_revision_after=self.revision,
                activated_elapsed_s=self.activation,
            )
        except (ValueError, RuntimeError, TimeoutError, OSError, json.JSONDecodeError):
            record["failure_reason"] = "invalid_or_unavailable_response"
        duration = time.perf_counter() - started
        self.wait_s += duration
        gateway.seconds["llm_wait"] += duration
        gateway.seconds["llm_features"] = self.feature_s
        record.update(
            wait_s=duration,
            completed_elapsed_s=self.elapsed(),
            effective_policy=_policy_dump(self.policies),
        )
        self.calls.append(record)
        if self.sink:
            self.sink(record)

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
        operator_overrides: dict[int, Operator] | None = None,
    ) -> tuple[Candidate, list[dict]]:
        """Execute same-incumbent exact MacroSearch using policy probability and early-stop."""
        if operator_overrides:
            raise ValueError("LLM study has no undeclared operator overrides")
        group = PREFERENCE_LABELS[preference(index, config.population)]
        policy = self.policies[group]
        probs = probabilities(policy)
        current = child
        records = []
        failures = 0
        operators: dict[int, Operator] = {a: StructuralOperator(a) for a in range(1, 7)}
        operators.update({7: ActivePack(), 8: PeakCoalition()})
        for step in range(config.rl_steps):
            if self.elapsed() >= self.seconds:
                break
            start = time.perf_counter()
            action = streams.stream("llm_action").choices(range(1, 9), weights=probs, k=1)[0]
            operator = operators[action]
            if action in (3, 6):
                operator = FocusStructural(action, policy.target_focus)
            context = NormalizationContext(gateway.ideal, maximum)
            tick = time.perf_counter()
            fallback = None
            try:
                batch = operator.propose(
                    gateway.problem,
                    current,
                    config.fixed_budget,
                    config,
                    streams.stream(f"A{action}"),
                )
            except (ValueError, IndexError, ArithmeticError):
                fallback = "focus_diagnostic_failure"
                batch = operators[action].propose(
                    gateway.problem,
                    current,
                    config.fixed_budget,
                    config,
                    streams.stream(f"A{action}"),
                )
            construction = time.perf_counter() - tick
            gateway.seconds[f"construction:A{action}"] += construction
            gateway.counts[f"construction_attempts:A{action}"] += batch.attempts
            gateway.counts[f"proposals:A{action}"] += len(batch.proposals)
            measured = ObservedEvaluator(gateway)
            hv_before = self.archive_value(gateway.archive.members)

            def observe_target(target: Genotype) -> None:
                if observer:
                    observer(
                        "structure",
                        dict(
                            source=current,
                            target=target,
                            path=f"A{action}",
                            step=step,
                            context=context,
                            weight=weights[index],
                        ),
                    )

            result = execute(
                batch,
                current,
                config.fixed_budget,
                weights[index],
                context,
                measured,
                f"A{action}",
                structural_observer=observe_target if observer else None,
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
            hv_gain = max(0.0, self.archive_value(gateway.archive.members) - hv_before)
            accepted = current is not prior
            failures = 0 if accepted or measured.new_objectives else failures + 1
            terminal = (
                failures >= policy.stop_after_no_gain
                or step == config.rl_steps - 1
                or self.elapsed() >= self.seconds
            )
            duration = time.perf_counter() - start
            row = dict(
                step=step,
                action=action,
                preference=group,
                policy_revision=self.revision,
                probabilities=probs,
                requested_focus=policy.target_focus,
                applied_focus=policy.target_focus
                if action in (3, 6) and not fallback
                else "ORIGINAL",
                fallback_reason=fallback,
                budget=config.fixed_budget,
                exact_evaluations=result.exact_evaluations,
                feasible=result.feasible_count,
                attempts=batch.attempts,
                proposals=len(batch.proposals),
                candidate_objectives=[c.evaluation.objectives for c in result.evaluated],
                candidate_feasible=[c.evaluation.feasible for c in result.evaluated],
                comparison_ideal=result.context.ideal,
                comparison_maximum=result.context.maximum,
                weight=weights[index],
                scalar_before=before,
                scalar_after=after,
                accepted=accepted,
                archive_new_objective_events=measured.new_objectives,
                dominated_archive_members=measured.dominated_members,
                archive_hv_gain=hv_gain,
                action_seconds=duration,
                seconds=duration,
                construction_seconds=construction,
                consecutive_no_gain=failures,
                terminal=terminal,
                stop_reason="no_gain_threshold"
                if failures >= policy.stop_after_no_gain
                else "horizon_or_budget"
                if terminal
                else None,
            )
            stats = self.stats[group].setdefault(f"A{action}", {})
            increments = dict(
                calls=1,
                candidate_attempts=batch.attempts,
                complete_exact_candidates=result.exact_evaluations,
                feasible_exact_candidates=result.feasible_count,
                local_accepts=int(accepted),
                archive_new_objective_events=measured.new_objectives,
                calls_with_archive_gain=int(measured.new_objectives > 0),
                no_candidate_calls=int(result.feasible_count == 0),
                dominated_archive_members=measured.dominated_members,
                wall_s=duration,
                scalar_gain=max(0.0, before - after),
                online_hv_gain=hv_gain,
            )
            for key, value in increments.items():
                stats[key] = stats.get(key, 0) + value
            self.observed = True
            records.append(row)
            if observer:
                observer("rl_step", row)
            if terminal:
                break
        return current, records


def run_policy(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    parameters: dict | None = None,
) -> RunResult:
    """Keep controls on their original path; inject only the explicit LLM research seam."""
    if arm not in ARMS:
        raise ValueError("Unknown treatment")
    if arm != "LLM_FLASH":
        return reference_run(
            problem,
            "BILL_PERMUTED" if arm == "BILL_MOEAD" else "STRICT_TWO_FOCUS",
            config,
            evidence,
            observer=observer,
        )
    if replace(
        config, generations=study_config(arm, config.seed, config.seconds or 1).generations
    ) != study_config(arm, config.seed, config.seconds or 1):
        raise ValueError("Undeclared LLM configuration")
    parameters = parameters or {}
    policy = LLMPolicy(
        config.seconds or 1,
        config.seed,
        requester=parameters.get("requester", isolated_request),
        sink=parameters.get("policy_sink"),
    )
    factory = charged_initializer(problem, config, "FULL_CURRENT_I4", evidence)

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        policy.begin = time.perf_counter()
        members = factory(gateway, streams)
        policy.initial_context = NormalizationContext(gateway.ideal, maximum(members))
        if observer:
            observer("initial_ready", dict(population=members, gateway=gateway))
        return members

    def observe(event: str, data: dict) -> None:
        if event == "offspring":
            policy.processed = data["processed"]
        if event == "initialization":
            # Derive exactly the engine's begin, including validation-before-initializer overhead.
            policy.begin = time.perf_counter() - data["elapsed"]
        if event in ("initialization", "before_offspring"):
            policy.update(data)
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
    evidence["llm"] = dict(
        model="deepseek-flash",
        thinking="disabled",
        schema=POLICY_SCHEMA,
        requests=len(policy.calls),
        valid_requests=sum(c["accepted"] for c in policy.calls),
        wait_s=policy.wait_s,
        feature_s=policy.feature_s,
        revision=policy.revision,
        prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    )
    return result
