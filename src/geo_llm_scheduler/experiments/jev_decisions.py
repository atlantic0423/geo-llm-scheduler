"""Pinned Jev decisions, explicit epsilon sampling and bounded English-only requests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

from geo_llm_scheduler.experiments.jev_compact import compact_state
from geo_llm_scheduler.experiments.jev_prompts import BACKGROUND_EN, PROMPT_VERSION, PROMPTS

Requester = Callable[[dict, float], dict]


class DecisionDeadline(RuntimeError):
    """No new model or candidate work may start after the production deadline."""


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Invalid probability")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Invalid probability")
    return result


def validate_answer(response: dict, model: str, name: str, question: dict) -> dict:
    """Bind model, question identity, types, options, maximum and token usage."""
    if (
        not isinstance(response, dict)
        or response.get("model") != model
        or not isinstance(response.get("answers"), dict)
        or set(response["answers"]) != {name}
    ):
        raise ValueError("Response binding mismatch")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("Invalid usage object")
    if any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")
    ):
        raise ValueError("Invalid usage")
    answer = response["answers"][name]
    if not isinstance(answer, dict):
        raise ValueError("Invalid answer object")
    if answer.get("type") != question["type"]:
        raise ValueError("Answer type mismatch")
    if question["type"] == "noul":
        _number(answer.get("noul"))
    else:
        p = answer.get("probabilities", {})
        if not isinstance(p, dict) or set(p) != set(question["criteria"]):
            raise ValueError("Option mismatch")
        values = {k: _number(v) for k, v in p.items()}
        if not math.isclose(math.fsum(values.values()), 1, abs_tol=1e-6):
            raise ValueError("Distribution not normalized")
        if answer.get("choice") not in values:
            raise ValueError("Unknown choice")
        if values[answer["choice"]] < max(values.values()) - 1e-6:
            raise ValueError("Choice is not a maximum")
        _number(answer.get("confidence"))
    return answer


def epsilon_sample(probabilities: dict[str, float], epsilon_g: float, rng: random.Random) -> str:
    """Greedy with probability epsilon_g; otherwise sample the full Jev distribution."""
    if (
        not probabilities
        or type(epsilon_g) not in (int, float)
        or not math.isfinite(epsilon_g)
        or not 0 <= epsilon_g <= 1
    ):
        raise ValueError("Invalid epsilon sampling input")
    values = [_number(p) for p in probabilities.values()]
    if not math.isclose(math.fsum(values), 1, abs_tol=1e-6):
        raise ValueError("Invalid sampling distribution")
    keys = list(probabilities)
    if rng.random() < epsilon_g:
        peak = max(values)
        return rng.choice([k for k in keys if probabilities[k] == peak])
    return rng.choices(keys, weights=values, k=1)[0]


@dataclass(frozen=True)
class JevSettings:
    """Research limits are per run, explicit, and independent of account balance."""

    epsilon_g: float = 0.5
    model: str = "jev-1.14.0"
    request_timeout_s: float = 15.0
    max_requests: int = 512
    max_input_tokens: int = 4_000_000
    max_wait_fraction: float = 0.5
    stop_threshold: float = 0.5
    long_context: bool = False
    compact_prompt: bool = True

    def __post_init__(self) -> None:
        if self.epsilon_g not in (0.5, 0.9) or type(self.epsilon_g) is bool:
            raise ValueError("Two declared epsilon arms are 0.5 and 0.9")
        if type(self.long_context) is not bool or type(self.compact_prompt) is not bool:
            raise ValueError("Context switches must be explicit booleans")
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", self.model):
            raise ValueError("Pin a versioned Jev model")
        if any(type(v) is not int or v < 1 for v in (self.max_requests, self.max_input_tokens)):
            raise ValueError("Positive request/token limits required")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("Invalid request timeout")
        if not 0 < self.max_wait_fraction <= 1 or not 0 <= self.stop_threshold <= 1:
            raise ValueError("Invalid wait/stop settings")


def isolated_request(payload: dict, timeout: float, *, long_context: bool = False) -> dict:
    """Kill/reap the HTTPS child at the deadline; no retry, key in environment only."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError("No HTTP budget")
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-m", "geo_llm_scheduler.experiments.jev_http_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        output, _ = process.communicate(
            json.dumps(
                {"payload": payload, "timeout": timeout, "long_context": long_context},
                allow_nan=False,
            ),
            timeout=max(0.001, timeout - (time.perf_counter() - started)),
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TimeoutError("Jev hard deadline") from None
    result = json.loads(output)
    if process.returncode or not result.get("ok"):
        raise RuntimeError("Jev unavailable")
    return result["response"]


class DecisionSession:
    """One production clock, request/token reservations and complete replay records."""

    # Public default model contract: at most 64k input tokens for an entire request.
    reservation = 65536

    def __init__(
        self,
        settings: JevSettings,
        seconds: float,
        requester: Requester = isolated_request,
        sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.settings, self.seconds = settings, seconds
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite production seconds required")
        self.requester = (
            partial(isolated_request, long_context=settings.long_context)
            if requester is isolated_request
            else requester
        )
        self.sink = sink
        self.reservation = 262144 if settings.long_context else 65536
        self.begin = time.perf_counter()
        self.wait_s = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.charged_or_reserved_tokens = 0
        self.records: list[dict] = []
        self.selections: list[dict] = []

    def elapsed(self) -> float:
        """Read actual end-to-end production wall time."""
        return time.perf_counter() - self.begin

    def expired(self) -> bool:
        """Expose the hard boundary to finite candidate construction."""
        return self.elapsed() >= self.seconds

    def ask_many(self, state: dict, questions: dict[str, dict]) -> dict | None:
        """Batch independent questions once; preserve a reservation on unknown usage."""
        if self.expired():
            raise DecisionDeadline("Production deadline")
        if (
            len(self.records) >= self.settings.max_requests
            or self.charged_or_reserved_tokens + self.reservation > self.settings.max_input_tokens
        ):
            return None
        timeout = min(
            self.settings.request_timeout_s,
            self.seconds - self.elapsed(),
            self.settings.max_wait_fraction * self.seconds - self.wait_s,
        )
        if timeout <= 0.001:
            return None
        payload = dict(
            model=self.settings.model,
            state=dict(
                problem_contract=BACKGROUND_EN,
                prompt_version=PROMPT_VERSION,
                **(compact_state(state) if self.settings.compact_prompt else state),
            ),
            questions=questions,
        )
        wire = json.dumps(payload, allow_nan=False)
        key = os.environ.get("TYPESAFE_API_KEY")
        if key and key in wire:
            raise ValueError("Credential in payload")
        self.charged_or_reserved_tokens += self.reservation
        record = dict(
            request=payload,
            request_sha256=hashlib.sha256(wire.encode()).hexdigest(),
            accepted=False,
            failure_reason=None,
            reserved_input_tokens=self.reservation,
        )
        started = time.perf_counter()
        answer = None
        try:
            response = self.requester(payload, timeout)
            if not isinstance(response, dict):
                raise ValueError("Invalid response object")
            if key and key in json.dumps(response):
                raise ValueError("Credential in response")
            record["response"] = response
            # Reported usage remains billable even if answer/model validation later fails.
            usage = response.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("Invalid usage object")
            tokens = usage.get("input_tokens")
            outputs = usage.get("output_tokens")
            if type(tokens) is int and tokens >= 0:
                self.charged_or_reserved_tokens += tokens - self.reservation
                self.input_tokens += tokens
                record["reserved_input_tokens"] = tokens
                if tokens > self.reservation:
                    record["provider_usage_exceeds_context_reservation"] = True
                    raise ValueError("Provider usage exceeds declared context limit")
            if type(outputs) is int and outputs >= 0:
                self.output_tokens += outputs
            if set(response.get("answers", {})) != set(questions):
                raise ValueError("Question batch mismatch")
            answer = {
                name: validate_answer(
                    dict(response, answers={name: response["answers"][name]}),
                    self.settings.model,
                    name,
                    question,
                )
                for name, question in questions.items()
            }
            if self.expired() or time.perf_counter() - started > timeout:
                raise TimeoutError("Late answer")
            record["accepted"] = True
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, TimeoutError):
            answer = None
            record.pop("response", None)  # Arbitrary provider error text never enters evidence.
            record["failure_reason"] = "invalid_unavailable_or_late"
        duration = time.perf_counter() - started
        self.wait_s += duration
        record["wait_s"] = duration
        self.records.append(record)
        if self.sink:
            self.sink(record)
        return answer

    def ask(self, state: dict, name: str, question: dict) -> dict | None:
        """Ask one question using the same batch validation and accounting boundary."""
        answers = self.ask_many(state, {name: question})
        return None if answers is None else answers[name]

    def choose(
        self,
        kind: str,
        options: dict[str, dict],
        state: dict,
        rng: random.Random,
    ) -> str:
        """Choose from every eligible option, recursively partitioning beyond 255."""
        if self.expired():
            raise DecisionDeadline("Production deadline")
        if not options:
            raise ValueError("Empty candidate set")
        keys = list(options)
        if len(keys) == 1:
            self.selections.append(
                dict(
                    kind=kind,
                    selected=keys[0],
                    selected_option=options[keys[0]],
                    epsilon_g=self.settings.epsilon_g,
                    probabilities={keys[0]: 1.0},
                    fallback="singleton_no_request",
                    options=1,
                )
            )
            return keys[0]
        original_options = options
        if len(keys) > 255:
            blocks = {f"block_{i}": keys[i : i + 255] for i in range(0, len(keys), 255)}
            descriptions = {
                k: dict(
                    count=len(v),
                    first_id=v[0],
                    last_id=v[-1],
                    first=options[v[0]],
                    last=options[v[-1]],
                )
                for k, v in blocks.items()
            }
            block = self.choose("partition", descriptions, dict(state, decision_kind=kind), rng)
            return self.choose(kind, {k: options[k] for k in blocks[block]}, state, rng)
        question = dict(type="choice", instructions=PROMPTS[kind][0], criteria=options)
        answer = self.ask(dict(state, decision_kind=kind), "selection", question)
        if answer is None:
            selected = rng.choices(
                keys,
                weights=[options[k].get("count", 1) if kind == "partition" else 1 for k in keys],
                k=1,
            )[0]
            probabilities = None
            fallback = "uniform_eligible_or_size_weighted_partition"
        else:
            probabilities = answer["probabilities"]
            selected = epsilon_sample(probabilities, self.settings.epsilon_g, rng)
            fallback = None
        self.selections.append(
            dict(
                kind=kind,
                selected=selected,
                selected_option=options[selected],
                epsilon_g=self.settings.epsilon_g,
                probabilities=probabilities,
                fallback=fallback,
                options=len(original_options),
            )
        )
        return selected
