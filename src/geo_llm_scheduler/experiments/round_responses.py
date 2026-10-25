"""Bounded Responses transport for the explicitly selected round-search model."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from geo_llm_scheduler.experiments.llm_protocol import (
    DeepSeekClient,
    _NoRedirect,
    _reject_constant,
    _unique_object,
)
from geo_llm_scheduler.experiments.round_prompts import BACKGROUND


@dataclass(frozen=True)
class ResponsesProfile:
    """Freeze gateway/model/effort, without storing authentication in configuration."""

    model: str = "gpt-6.1-sol"
    base_url: str = "https://v1.cdks.work/v1"
    reasoning_effort: str = "high"
    request_timeout_s: float = 120.0
    max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.model not in (
            "gpt-6.1-sol",
            "gpt-5.5",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
        ):
            raise ValueError("An explicitly supported research model is required")
        if self.base_url != "https://v1.cdks.work/v1":
            raise ValueError("Gateway must match the user-authorized endpoint")
        if self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Invalid reasoning effort")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("Invalid request timeout")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("Invalid output bound")

    @property
    def version(self) -> str:
        """Return a model-bound research identity distinct from the old Flash arm."""
        return "round_responses_v031_population20_b1_l5_" + self.model


def request(profile: ResponsesProfile, payload: dict, timeout: float) -> dict:
    """Make one HTTPS request; no redirects, automatic retry or model substitution.

    This normalized envelope retains usage and HTTP classifications, not raw
    provider errors or authentication. Fatal authentication/balance/model errors
    are passed to the parent so it can stop the scientific run immediately.
    """
    key = os.environ.get("ROUND_RESPONSES_API_KEY", "")
    if not key or any(c.isspace() for c in key):
        raise ValueError("A current-run credential is required")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid request timeout")
    wire = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if not wire.isascii() or key in wire:
        raise ValueError("Only secret-free English data may be sent")
    body = dict(
        model=profile.model,
        instructions=BACKGROUND,
        input=[dict(role="developer", content=BACKGROUND), dict(role="user", content=wire)],
        store=False,
        service_tier="default",
        reasoning=dict(effort=profile.reasoning_effort),
        text=dict(format=dict(type="json_object")),
        max_output_tokens=profile.max_output_tokens,
    )
    deadline = time.monotonic() + timeout
    http = Request(
        profile.base_url + "/responses",
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-openai-actor-authorization": "local-image-extension",
        },
        method="POST",
    )
    reader = DeepSeekClient(key)
    try:
        with build_opener(_NoRedirect()).open(http, timeout=timeout) as response:
            raw = reader._read_response(response, deadline)
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if key.encode() in raw:
            return dict(fatal_error="credential_in_response", http_status=200)
        if value.get("model") != profile.model:
            return dict(fatal_error="model_identity_mismatch", http_status=200)
        chunks = []
        for item in value.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        return dict(fatal_error="provider_refusal", http_status=200)
                    if part.get("type") == "output_text":
                        chunks.append(part["text"])
        usage = value.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("Invalid usage envelope")
        if (
            type(usage.get("output_tokens")) is int
            and usage["output_tokens"] > profile.max_output_tokens
        ):
            return dict(fatal_error="output_usage_exceeds_limit", http_status=200)
        normalized = dict(usage)
        for original, canonical in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            if original in usage:
                normalized[canonical] = usage[original]
        return dict(
            content="".join(chunks),
            returned_model=value["model"],
            finish_reason="stop" if value.get("status") == "completed" and chunks else "incomplete",
            usage=normalized,
            id=value.get("id"),
            http_status=200,
            provider_parameters={
                k: value.get(k)
                for k in ("store", "reasoning", "text", "max_output_tokens", "service_tier")
            },
        )
    except HTTPError as error:
        fatal = "http_" + str(error.code) if error.code in (400, 401, 402, 403, 404) else None
        return dict(fatal_error=fatal, transport_error="http_error", http_status=error.code)
    except (URLError, OSError, TimeoutError):
        return dict(transport_error="network_or_deadline")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return dict(transport_error="invalid_response_envelope")


def isolated_request(profile: ResponsesProfile, payload: dict, timeout: float) -> dict:
    """Bound one child process; kill/reap at the total deadline without retries."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid request timeout")
    child = subprocess.Popen(
        [sys.executable, "-m", "geo_llm_scheduler.experiments.round_responses"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        output, _ = child.communicate(
            json.dumps(
                dict(profile=asdict(profile), payload=payload, timeout=timeout), allow_nan=False
            ),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        return dict(transport_error="child_deadline")
    if child.returncode:
        return dict(transport_error="child_failure")
    try:
        result = json.loads(output)
        if not isinstance(result, dict):
            raise ValueError("Not an envelope")
        return result
    except (ValueError, TypeError):
        return dict(transport_error="child_invalid_envelope")


def main() -> None:
    """Accept secret-free input and use only the child's authentication environment."""
    value = json.load(sys.stdin)
    result = request(ResponsesProfile(**value["profile"]), value["payload"], value["timeout"])
    print(json.dumps(result, allow_nan=False, ensure_ascii=True))


if __name__ == "__main__":
    main()
