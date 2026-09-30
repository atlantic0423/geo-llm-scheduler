"""Fail-closed TypeSafe System One HTTP adapter (official contract, 2026-09-30)."""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter, sleep
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class JevError(RuntimeError):
    """An unavailable or invalid Jev decision; callers must fail the run."""


@dataclass(frozen=True)
class Question:
    """One narrow Choice or Noul question with versioned instructions and criteria."""

    kind: Literal["choice", "noul"]
    instructions: str
    criteria: Mapping[str, str]

    def payload(self) -> dict:
        """Validate and build the official wire shape; never invent an options field."""
        if not self.instructions.strip() or self.kind not in ("choice", "noul"):
            raise JevError("Invalid question")
        if self.kind == "choice" and not 1 <= len(self.criteria) <= 255:
            raise JevError("Choice requires 1 to 255 criteria")
        if self.kind == "noul" and set(self.criteria) != {"true", "false"}:
            raise JevError("Noul criteria must describe true and false")
        if any(not key or not isinstance(value, str) for key, value in self.criteria.items()):
            raise JevError("Invalid criteria")
        return {
            "type": self.kind,
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Answer:
    """Validated raw decision values; probabilities are not project-calibrated labels."""

    kind: str
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    noul: float | None = None


@dataclass(frozen=True)
class Response:
    """Secret-free parsed answer metadata including total request/retry wall time."""

    model: str
    answers: dict[str, Answer]
    usage: dict[str, int]
    elapsed: float
    retries: int
    http_statuses: tuple[int, ...]


@dataclass(frozen=True)
class HttpReply:
    """Minimal injectable HTTP result; server bodies must not enter error messages."""

    status: int
    body: bytes = field(default=b"", repr=False)
    retry_after: float | None = None


class Transport(Protocol):
    """Dependency-injection boundary for live HTTP and deterministic mock transports."""

    def __call__(self, body: bytes, key: str, timeout: float) -> HttpReply:
        """Post bytes to the fixed official endpoint with a bounded timeout."""
        ...


def http_transport(body: bytes, key: str, timeout: float) -> HttpReply:
    """Use stdlib HTTPS, capturing status but never echoing credential-bearing errors."""
    request = Request(
        ENDPOINT,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as result:
            return HttpReply(result.status, result.read())
    except HTTPError as error:
        try:
            delay = float(error.headers.get("Retry-After", "0"))
        except ValueError:
            delay = 0.0
        return HttpReply(error.code, b"", delay)
    except (URLError, OSError):
        raise JevError("TypeSafe transport unavailable") from None


def _probability(value: object) -> float:
    if type(value) not in (float, int):
        raise JevError("Invalid numeric decision")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise JevError("Invalid numeric decision")
    return number


def parse_response(payload: object, questions: Mapping[str, Question], model: str) -> Response:
    """Reject schema errors, unexpected options and changed models without fallback."""
    if not isinstance(payload, dict) or payload.get("model") != model:
        raise JevError("Jev model mismatch or invalid response")
    raw = payload.get("answers")
    usage = payload.get("usage")
    if not isinstance(raw, dict) or set(raw) != set(questions):
        raise JevError("Answer identifiers do not match request")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("input_tokens", "output_tokens")
    ):
        raise JevError("Invalid token usage")
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        answer = raw[name]
        if not isinstance(answer, dict) or answer.get("type") != question.kind:
            raise JevError("Answer type mismatch")
        if question.kind == "noul":
            answers[name] = Answer("noul", noul=_probability(answer.get("noul")))
            continue
        choice, probabilities = answer.get("choice"), answer.get("probabilities")
        if (
            not isinstance(choice, str)
            or choice not in question.criteria
            or not isinstance(probabilities, dict)
            or set(probabilities) != set(question.criteria)
        ):
            raise JevError("Invalid Choice option or distribution")
        validated = {option: _probability(value) for option, value in probabilities.items()}
        if not math.isclose(sum(validated.values()), 1.0, abs_tol=1e-6):
            raise JevError("Choice probabilities do not sum to one")
        if validated[choice] < max(validated.values()) - 1e-6:
            raise JevError("Selected option is not a maximum-probability option")
        answers[name] = Answer(
            "choice", choice, validated, confidence=_probability(answer.get("confidence"))
        )
    return Response(
        model,
        answers,
        {key: usage[key] for key in ("input_tokens", "output_tokens")},
        0.0,
        0,
        (),
    )


class JevClient:
    """Pinned-model client with explicit RNG, bounded retries and optional deadline."""

    def __init__(
        self,
        model: str,
        rng: random.Random,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_attempts: int = 3,
        transport: Transport = http_transport,
        clock: Callable[[], float] = perf_counter,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        """Read the secret from environment by default; mutable aliases are forbidden."""
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
            raise JevError("An explicit versioned Jev model ID is required")
        key = os.environ.get("TYPESAFE_API_KEY") if api_key is None else api_key
        if not key or not key.strip():
            raise JevError("TYPESAFE_API_KEY is not configured")
        if not math.isfinite(timeout) or timeout <= 0 or max_attempts not in (1, 2, 3):
            raise JevError("Invalid timeout or bounded attempt count")
        self._model = model
        self._key = key
        self._rng = rng
        self._timeout = timeout
        self._attempts = max_attempts
        self._transport = transport
        self._clock = clock
        self._sleep = sleeper

    @property
    def model(self) -> str:
        """Expose the immutable pinned model identifier used throughout this client."""
        return self._model

    def decide(
        self, state: dict, questions: Mapping[str, Question], *, deadline: float | None = None
    ) -> Response:
        """Return a validated decision or raise; retries/waits count against the deadline."""
        if not questions:
            raise JevError("At least one question is required")
        body = json.dumps(
            {
                "model": self.model,
                "state": state,
                "questions": {name: question.payload() for name, question in questions.items()},
            },
            allow_nan=False,
            ensure_ascii=False,
        ).encode()
        if self._key.encode() in body:
            raise JevError("Credential detected in decision payload")
        begin = self._clock()
        statuses: list[int] = []
        for attempt in range(self._attempts):
            remaining = self._timeout if deadline is None else deadline - self._clock()
            if remaining <= 0:
                raise JevError("Decision deadline exhausted")
            try:
                reply = self._transport(body, self._key, min(self._timeout, remaining))
            except (JevError, TimeoutError, OSError):
                # Suppress original messages and chained exceptions: transports may contain keys.
                reply = HttpReply(0)
            statuses.append(reply.status)
            if 200 <= reply.status < 300:
                if deadline is not None and self._clock() >= deadline:
                    raise JevError("Decision arrived after deadline")
                if self._key.encode() in reply.body:
                    raise JevError("Credential detected in API response")
                try:
                    parsed = parse_response(json.loads(reply.body), questions, self.model)
                except (ValueError, UnicodeDecodeError):
                    raise JevError("Invalid TypeSafe JSON") from None
                return Response(
                    parsed.model,
                    parsed.answers,
                    parsed.usage,
                    self._clock() - begin,
                    attempt,
                    tuple(statuses),
                )
            retryable = reply.status in (0, 429) or 500 <= reply.status < 600
            if not retryable or attempt + 1 == self._attempts:
                raise JevError(
                    f"TypeSafe decision failed (HTTP {reply.status}; attempts {attempt + 1})"
                )
            delay = 0.25 * 2**attempt + self._rng.uniform(0.0, 0.1)
            if reply.retry_after is not None and math.isfinite(reply.retry_after):
                delay = max(delay, min(2.0, max(0.0, reply.retry_after)))
            if deadline is not None and self._clock() + delay >= deadline:
                raise JevError("Retry would exceed decision deadline")
            self._sleep(delay)
        raise AssertionError("Unreachable retry state")
