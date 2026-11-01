"""Restricted strategy parsing, probability conversion and single-shot HTTPS client."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from geo_llm_scheduler.experiments.llm_global_prompt import SYSTEM_PROMPT

GROUPS = ("Flow", "Balanced", "Electricity")
ACTIONS = tuple(f"A{i}" for i in range(1, 9))
FOCUSES = (
    "FLOW_WAIT",
    "RESOURCE_WAIT",
    "KV_DELAY",
    "REGION_LOAD",
    "TOU_EXPOSURE",
    "DEMAND_PEAK",
    "MIXED",
)
REQUEST_SCHEMA = "search_policy_v2"
POLICY_SCHEMA = "search_policy_response_v2"
_STOPS = (1, 2, 3, 5)
_MAX_RESPONSE_BYTES = 256 * 1024


@dataclass(frozen=True)
class GroupPolicy:
    """One immutable coarse-preference policy; action tuple order is A1 through A8."""

    group: str
    action_weights: tuple[float, ...]
    target_focus: str
    stop_after_no_gain: int
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or self.group not in GROUPS:
            raise ValueError("Unknown preference group")
        if not isinstance(self.action_weights, tuple) or len(self.action_weights) != 8:
            raise ValueError("Exactly eight action weights are required")
        if any(not _valid_number(w, 0, 100) for w in self.action_weights):
            raise ValueError("Action weights must be finite numbers in [0,100]")
        if math.fsum(self.action_weights) <= 0:
            raise ValueError("Action weights need a positive sum")
        if not isinstance(self.target_focus, str) or self.target_focus not in FOCUSES:
            raise ValueError("Unknown target focus")
        if type(self.stop_after_no_gain) is not int or self.stop_after_no_gain not in _STOPS:
            raise ValueError("Invalid stopping threshold")
        if not isinstance(self.rationale, str) or len(self.rationale) > 300:
            raise ValueError("Rationale must contain at most 300 characters")


def _valid_number(value: Any, lower: float, upper: float) -> bool:
    try:
        return type(value) in (int, float) and lower <= value <= upper and math.isfinite(value)
    except OverflowError:
        return False


def default_policies() -> dict[str, GroupPolicy]:
    """Return a fresh uniform/original-generation L5 fallback for all three groups."""
    return {g: GroupPolicy(g, (1.0,) * 8, "MIXED", 5, "Uniform fallback") for g in GROUPS}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("Non-finite JSON constant")


def _exact_keys(value: Any, expected: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unexpected or missing policy fields")


def parse_policy(content: str, request_id: str, snapshot_id: str) -> dict[str, GroupPolicy]:
    """Validate an entire v2 response atomically, rejecting duplicates and stale bindings.

    Raises ValueError for malformed JSON, non-finite constants, extra/missing fields or any
    invalid group. The caller retains its previous effective policy on failure.
    """
    if not isinstance(content, str) or not isinstance(request_id, str):
        raise ValueError("Policy content and request ID must be strings")
    if not isinstance(snapshot_id, str):
        raise ValueError("Snapshot ID must be a string")
    try:
        value = json.loads(
            content, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        _exact_keys(value, {"schema_version", "request_id", "snapshot_id", "policies"})
        if value["schema_version"] != POLICY_SCHEMA:
            raise ValueError("Unexpected policy schema")
        if value["request_id"] != request_id or value["snapshot_id"] != snapshot_id:
            raise ValueError("Policy response does not match the current request snapshot")
        items = value["policies"]
        if not isinstance(items, list) or len(items) != 3:
            raise ValueError("Exactly three preference policies are required")
        parsed: dict[str, GroupPolicy] = {}
        for item in items:
            _exact_keys(
                item, {"group", "action_weights", "target_focus", "stop_after_no_gain", "rationale"}
            )
            _exact_keys(item["action_weights"], set(ACTIONS))
            policy = GroupPolicy(
                group=item["group"],
                action_weights=tuple(item["action_weights"][a] for a in ACTIONS),
                target_focus=item["target_focus"],
                stop_after_no_gain=item["stop_after_no_gain"],
                rationale=item["rationale"],
            )
            if policy.group in parsed:
                raise ValueError("Duplicate preference group")
            parsed[policy.group] = policy
        return {group: parsed[group] for group in GROUPS}
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ValueError("Invalid search policy response") from None


def probabilities(policy: GroupPolicy, epsilon: float = 0.10) -> tuple[float, ...]:
    """Normalize A1..A8 weights and mix fixed uniform exploration, without drawing RNG."""
    if not _valid_number(epsilon, 0, 1):
        raise ValueError("Exploration must be a finite number in [0,1]")
    total = math.fsum(policy.action_weights)
    return tuple((1 - epsilon) * (w / total) + epsilon / 8 for w in policy.action_weights)


class _NoRedirect(HTTPRedirectHandler):
    """Prevent an HTTP redirect from forwarding the Authorization header elsewhere."""

    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


class DeepSeekClient:
    """Single-shot non-thinking Chat Completions client with credential-safe failures.

    Credentials are loaded only when constructing a client, never during module import.
    Returned envelopes exclude request headers, URLs, raw HTTP error bodies and credentials.
    The caller must require finish_reason == 'stop' before parsing the policy content.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 2048,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        key = os.environ.get("DEEPSEEK_API_KEY") if api_key is None else api_key
        if not isinstance(key, str) or not key.strip() or any(c.isspace() for c in key):
            raise ValueError("A valid DeepSeek credential is required")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A model name is required")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("A nonempty system prompt is required")
        try:
            parts = urlsplit(base_url)
            if (
                parts.scheme != "https"
                or not parts.hostname
                or any(c.isspace() for c in base_url)
                or parts.username is not None
                or parts.password is not None
                or parts.query
                or parts.fragment
            ):
                raise ValueError("Invalid endpoint")
        except (ValueError, TypeError):
            raise ValueError("base_url must be an HTTPS endpoint without credentials") from None
        self._api_key = key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Make exactly one HTTPS JSON request; return sanitized response content and usage.

        Network/HTTP/provider-format failures raise RuntimeError without remote bodies, URLs
        or exception details. Invalid caller arguments raise ValueError before any network IO.
        """
        if not isinstance(payload, dict) or not _valid_number(timeout, 0, float("inf")):
            raise ValueError("Payload must be an object and timeout a positive finite number")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        deadline = time.monotonic() + timeout
        try:
            user_content = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise ValueError("Payload must be finite serializable JSON") from None
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            with build_opener(_NoRedirect()).open(request, timeout=remaining) as response:
                raw = self._read_response(response, deadline)
            result = json.loads(
                raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            choice = result["choices"][0]
            content = choice["message"]["content"]
            finish = choice["finish_reason"]
            model = result["model"]
            response_id = result.get("id")
            if not all(isinstance(v, str) for v in (content, finish, model)):
                raise ValueError("Invalid response fields")
            if response_id is not None and not isinstance(response_id, str):
                raise ValueError("Invalid response ID")
            usage = result.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("Invalid usage")
            return {
                "content": content.replace(self._api_key, "[REDACTED]"),
                "usage": self._sanitize(usage),
                "returned_model": model.replace(self._api_key, "[REDACTED]"),
                "finish_reason": finish.replace(self._api_key, "[REDACTED]"),
                "id": response_id.replace(self._api_key, "[REDACTED]") if response_id else None,
            }
        except HTTPError as error:
            raise RuntimeError(f"DeepSeek HTTP request failed (status {error.code})") from None
        except (URLError, OSError, TimeoutError):
            raise RuntimeError("DeepSeek HTTPS request failed") from None
        except (ValueError, TypeError, KeyError, IndexError, RecursionError):
            raise RuntimeError("DeepSeek returned an invalid response envelope") from None

    def _read_response(self, response: Any, deadline: float) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            # read1 makes at most one raw read; every read gets the remaining total
            # deadline instead of a renewed full timeout for each arriving chunk.
            sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if sock is not None:
                sock.settimeout(remaining)
            chunk = response.read1(min(16384, _MAX_RESPONSE_BYTES + 1 - size))
            if time.monotonic() >= deadline:
                raise TimeoutError
            if not chunk:
                return b"".join(chunks)
            size += len(chunk)
            if size > _MAX_RESPONSE_BYTES:
                raise RuntimeError("DeepSeek response exceeds the configured size limit")
            chunks.append(chunk)

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, dict):
            return {self._sanitize(k): self._sanitize(v) for k, v in value.items()}
        return value
