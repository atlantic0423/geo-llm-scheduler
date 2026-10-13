"""Frozen four-arm generation-rule / call-RNG experiment; no default adoption."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from geo_llm_scheduler.config import Config

ARMS = ("NATIVE_SHARED", "POOL_SHARED", "NATIVE_CALL", "POOL_CALL")


def arm_config(baseline: Config, arm: str) -> Config:
    """Fix evaluation at six and vary only generation rules and call RNG policy."""
    if arm not in ARMS:
        raise ValueError("Unknown macro factor arm")
    return replace(
        baseline,
        method="full",
        a3_region_policy="tariff",
        rl_state_policy="dominant",
        replacement_policy="birth",
        budget_policy="fixed",
        fixed_budget=6,
        macro_generation_budget=10 if arm.startswith("POOL") else 0,
        macro_rng_policy="call" if arm.endswith("CALL") else "shared",
        macro_diagnostics=False,
    )


def quota_used_bytes(
    campaign: Path,
    reader: Callable[..., Any] = subprocess.check_output,
) -> int:
    """Boundedly retry only disappearing heartbeat files; otherwise fail closed.

    Persistent recognized ENOENT closes the 100 GiB quota guard. Other errors
    propagate rather than bypassing the guard. No file or cache is modified.
    """
    prefix = re.escape(str(campaign))
    pattern = re.compile(
        rf"du: cannot access '{prefix}/(?:formal|pilot)_node[01]/ops/heartbeats/"
        r"\.[^/']+\.json\.\d+\.tmp': No such file or directory"
    )
    for attempt in range(4):
        try:
            raw = reader(["du", "-sb", str(campaign.parent)], text=True, stderr=subprocess.PIPE)
            value = int(raw.split()[0])
            if value < 0:
                raise ValueError("Negative quota measurement")
            return value
        except subprocess.CalledProcessError as error:
            stderr = error.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            lines = (stderr or "").strip().splitlines()
            if error.returncode != 1 or not lines or not all(pattern.fullmatch(x) for x in lines):
                raise
            print(f"Atomic heartbeat quota read retry {attempt + 1}", file=sys.stderr, flush=True)
            if attempt < 3:
                time.sleep(0.05 * 2**attempt)
    return 100 * 1024**3
