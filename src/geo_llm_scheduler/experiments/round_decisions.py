"""Four bounded Flash decisions; only synchronous transport blocking is excluded."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Callable

from geo_llm_scheduler.experiments.jev_compact import compact_state
from geo_llm_scheduler.experiments.jev_decisions import (
    DecisionDeadline,
    DecisionSession,
    JevSettings,
    Requester,
)
from geo_llm_scheduler.experiments.llm_protocol import _reject_constant, _unique_object
from geo_llm_scheduler.experiments.round_prompts import (
    A6_TOP_SET,
    ROUND_SELECT,
    STEP_ACTION,
    STEP_STOP,
    STEP_TARGET,
    VERSION,
)


def isolated_request(payload: dict, timeout: float) -> dict:
    """Bound one English-only HTTPS child, kill/reap it on timeout, never retry."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid request deadline")
    process = subprocess.Popen(
        [sys.executable, "-m", "geo_llm_scheduler.experiments.round_http_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        output, _ = process.communicate(
            json.dumps(dict(payload=payload, timeout=timeout), allow_nan=False), timeout=timeout
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError("Flash transport deadline") from None
    value = json.loads(output)
    if process.returncode or not value.get("ok"):
        raise RuntimeError("Flash transport unavailable")
    return value["response"]


def validate_response(packet: dict, answer: dict) -> dict:
    """Reject stale IDs, incomplete scores, illegal bindings and non-boolean stops."""
    task = packet["task"]
    payload = {
        "ROUND_SELECT": "selected",
        "STEP_ACTION": "action_scores",
        "STEP_TARGET": "option_scores",
        "STEP_STOP": "stop",
    }[task]
    keys = {"request_id", "snapshot_id", payload}
    if task == "STEP_STOP":
        keys.add("reason_code")
    if not isinstance(answer, dict) or set(answer) != keys:
        raise ValueError("Response schema mismatch")
    if any(answer[k] != packet[k] for k in ("request_id", "snapshot_id")):
        raise ValueError("Response snapshot mismatch")
    if task == "ROUND_SELECT":
        selected = answer["selected"]
        if not isinstance(selected, list) or len(selected) != packet["limits"]["selected_count"]:
            raise ValueError("Selection quota mismatch")
        candidates = {c["solution_id"]: c for c in packet["state"]["candidate_solutions"]}
        seen = set()
        for item in selected:
            if not isinstance(item, dict) or set(item) != {
                "solution_id",
                "direction_id",
                "reason_code",
            }:
                raise ValueError("Invalid selected entry")
            candidate = candidates.get(item["solution_id"])
            if candidate is None or type(item["direction_id"]) is not int:
                raise ValueError("Unknown seed/direction")
            if item["direction_id"] not in candidate["allowed_direction_ids"]:
                raise ValueError("Unbound direction")
            if (
                candidate["phenotype_id"] in seen
                or item["reason_code"] not in packet["reason_codes"]
            ):
                raise ValueError("Duplicate seed or invalid reason")
            seen.add(candidate["phenotype_id"])
    elif task == "STEP_STOP":
        if type(answer["stop"]) is not bool or answer["reason_code"] not in packet["reason_codes"]:
            raise ValueError("Invalid stop decision")
    else:
        scores = answer[payload]
        options = packet["allowed_options"]
        if not isinstance(scores, dict) or set(scores) != set(options):
            raise ValueError("Scores must cover every option")
        for key, value in scores.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid priority score")
            if task == "STEP_ACTION" and not options[key] and value != 0:
                raise ValueError("Masked action scored positive")
        if not 0 < math.fsum(scores.values()) < float("inf"):
            raise ValueError("No finite positive priority mass")
    return answer


class RoundSession(DecisionSession):
    """Adapter retaining the existing legal target generators and explicit RNGs."""

    def __init__(
        self,
        seconds: float,
        requester: Requester = isolated_request,
        sink: Callable[[dict], None] | None = None,
        *,
        max_requests: int = 32768,
        max_input_tokens: int = 1_000_000_000,
    ) -> None:
        super().__init__(
            JevSettings(
                epsilon_g=0.9,
                request_timeout_s=30,
                max_requests=max_requests,
                max_input_tokens=max_input_tokens,
            ),
            seconds,
            requester,
            sink,
            exclude_api_wait=True,
        )
        self.requester = requester
        self.valid_by_task: dict[str, int] = {}
        self.fallback_by_task: dict[str, int] = {}
        self.transport_intervals: list[tuple[float, float]] = []
        self.round_context: dict = {}

    def request(self, task: str, state: dict, options: dict, **extra: object) -> dict | None:
        """Charge local construction/validation; exclude just the requester interval."""
        if self.expired():
            raise DecisionDeadline("Search deadline")
        if (
            len(self.records) >= self.settings.max_requests
            or self.charged_or_reserved_tokens + self.reservation > self.settings.max_input_tokens
        ):
            self.fallback_by_task[task] = self.fallback_by_task.get(task, 0) + 1
            return None
        text = {
            "ROUND_SELECT": ROUND_SELECT,
            "STEP_ACTION": STEP_ACTION,
            "STEP_TARGET": STEP_TARGET,
            "STEP_STOP": STEP_STOP,
        }[task]
        if state.get("decision_kind") == "a6_job_ranking":
            text += "\n" + A6_TOP_SET
        state = dict(
            state,
            budget=dict(
                state.get("budget", {}),
                search_elapsed_s=self.elapsed(),
                search_remaining_s=max(0.0, self.seconds - self.elapsed()),
            ),
        )
        payload_name = {
            "ROUND_SELECT": "selected",
            "STEP_ACTION": "action_scores",
            "STEP_TARGET": "option_scores",
            "STEP_STOP": "stop",
        }[task]
        request_id = f"q{len(self.records)}"
        # Targets share the same actual incumbent and conditional parent fields;
        # request_id distinguishes otherwise identical snapshots.
        digest = hashlib.sha256(
            json.dumps(state, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()[:24]
        packet: dict = dict(
            prompt_version=VERSION,
            task=task,
            request_id=request_id,
            snapshot_id=digest,
            instructions=text,
            state=compact_state(state),
            allowed_options=options,
            response_contract=dict(
                required=["request_id", "snapshot_id", payload_name],
                score_keys=list(options),
                extra_fields=False,
                answer_keys_are_exact=True,
                forbidden_transport_fields=["type", "response_format"],
            ),
            **extra,
        )
        if task == "STEP_STOP":
            packet["response_contract"]["required"].append("reason_code")
        key = os.environ.get("DEEPSEEK_API_KEY")
        wire = json.dumps(packet, ensure_ascii=False, allow_nan=False)
        if not wire.isascii() or key and key in wire:
            raise ValueError("Only secret-free English wire data are allowed")
        self.charged_or_reserved_tokens += self.reservation
        record = dict(request=packet, accepted=False, failure_reason=None)
        started = time.perf_counter()
        answer = None
        try:
            start = time.perf_counter()
            try:
                response = self.requester(packet, self.settings.request_timeout_s)
            finally:
                end = time.perf_counter()
                self.excluded_api_wait_s += end - start
                self.transport_intervals.append((start - self.begin, end - self.begin))
            if not isinstance(response, dict):
                raise ValueError("Invalid response envelope")
            usage = response.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("Invalid usage")
            tokens = usage.get("prompt_tokens")
            outputs = usage.get("completion_tokens")
            if type(tokens) is int and tokens >= 0:
                self.charged_or_reserved_tokens += tokens - self.reservation
                self.input_tokens += tokens
                if tokens > self.reservation:
                    record["provider_usage_exceeds_context_reservation"] = True
            if type(outputs) is int and outputs >= 0:
                self.output_tokens += outputs
            record["response"] = response
            if response.get("finish_reason") != "stop":
                raise ValueError("Incomplete answer")
            content = response["content"]
            if key and key in content:
                raise ValueError("Credential in provider response")
            answer = validate_response(
                packet,
                json.loads(
                    content, object_pairs_hook=_unique_object, parse_constant=_reject_constant
                ),
            )
            if self.expired():
                raise DecisionDeadline("Search deadline after answer")
            record["accepted"] = True
            self.valid_by_task[task] = self.valid_by_task.get(task, 0) + 1
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, TimeoutError):
            answer = None
            record["failure_reason"] = "invalid_or_unavailable"
            self.fallback_by_task[task] = self.fallback_by_task.get(task, 0) + 1
        duration = time.perf_counter() - started
        record.update(
            wall_s=duration,
            api_blocked_s=end - start,
            search_elapsed_s=self.elapsed(),
            valid_by_task=dict(self.valid_by_task),
        )
        self.wait_s = self.excluded_api_wait_s
        if self.sink:
            self.sink(record)
        self.records.append(
            record
            if self.sink is None
            else dict(
                accepted=record["accepted"],
                task=task,
                api_blocked_s=end - start,
                failure_reason=record["failure_reason"],
            )
        )
        return answer

    def ask(self, state: dict, name: str, question: dict) -> dict | None:
        """Adapt A6's existing one-ranking interface to finite nonnegative scores."""
        answer = self.request("STEP_TARGET", state, question["criteria"])
        if answer is None:
            return None
        scores = answer["option_scores"]
        total = math.fsum(scores.values())
        return dict(probabilities={k: v / total for k, v in scores.items()})

    def choose(self, kind: str, options: dict[str, dict], state: dict, rng: random.Random) -> str:
        """Use every legal conditional option; no legacy 255-item truncation."""
        if self.expired():
            raise DecisionDeadline("Search deadline")
        if not options:
            raise ValueError("Empty target set")
        fallback: str | None
        if len(options) == 1:
            selected, answer = next(iter(options)), None
            fallback = "singleton_no_request"
        else:
            answer = self.request("STEP_TARGET", dict(state, decision_kind=kind), options)
            scores = answer["option_scores"] if answer else {k: 1 for k in options}
            if rng.random() < 0.9:
                peak = max(scores.values())
                selected = rng.choice([k for k, v in scores.items() if v == peak])
            else:
                selected = rng.choices(list(scores), weights=list(scores.values()), k=1)[0]
            fallback = None if answer else "uniform_legal_fallback"
        self.selections.append(
            dict(
                kind=kind,
                selected=selected,
                selected_option=options[selected],
                fallback=fallback,
                options=len(options),
            )
        )
        return selected
