"""Frozen campaign settings and portable job identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CampaignSettings:
    """Validated resource and experiment matrix settings from one YAML file."""

    duration_hours: float = 60
    grace_minutes: float = 25
    dependency_poll_seconds: int = 180
    heartbeat_seconds: int = 30
    screening_seconds: float = 180
    final_seconds: float = 300
    jobs_per_instance: int = 50
    population: int = 100
    screening_instance_seeds: tuple[int, ...] = (71, 72, 73)
    final_instance_seeds: tuple[int, ...] = (71, 72, 73, 74, 75)
    algorithm_seeds: tuple[int, ...] = (101, 202, 303)
    max_workers: int = 2
    memory_reserve_gb: float = 4
    memory_per_worker_gb: float = 3
    min_free_disk_gb: float = 20
    max_job_retries: int = 2
    include_optional_macro_arms: bool = False
    e13_extra_seed_start: int = 404
    e13_extra_seed_stop: int = 9999
    base_config: str = "configs/default.yaml"
    d03_raw_root: str = "outputs/diagnostics/d03_v2/runs"

    def __post_init__(self) -> None:
        if self.duration_hours <= 0 or self.grace_minutes < 0:
            raise ValueError("Invalid campaign deadline")
        if self.screening_seconds <= 0 or self.final_seconds <= 0:
            raise ValueError("Experiment budgets must be positive")
        if min(self.population, self.jobs_per_instance, self.max_workers) < 2:
            raise ValueError("Campaign needs at least two jobs, population members and workers")
        if self.dependency_poll_seconds < 1 or self.heartbeat_seconds < 1:
            raise ValueError("Poll and heartbeat intervals must be positive")
        if self.memory_per_worker_gb <= 0 or self.min_free_disk_gb <= 0:
            raise ValueError("Resource limits must be positive")
        if (
            not self.screening_instance_seeds
            or not self.final_instance_seeds
            or not self.algorithm_seeds
        ):
            raise ValueError("Frozen seed matrices cannot be empty")
        if self.max_job_retries < 0 or self.e13_extra_seed_stop < self.e13_extra_seed_start:
            raise ValueError("Invalid retry or extra seed range")


def load_settings(path: str | Path) -> CampaignSettings:
    """Read campaign YAML and reject accidental or unsupported fields."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(CampaignSettings)}:
        raise ValueError("Unknown campaign settings")
    for key in ("screening_instance_seeds", "final_instance_seeds", "algorithm_seeds"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return CampaignSettings(**raw)


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible campaign metadata."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def settings_hash(settings: CampaignSettings) -> str:
    """Identify the complete frozen campaign settings across resume attempts."""
    return stable_hash(asdict(settings))


@dataclass(frozen=True)
class JobSpec:
    """One independent experiment identified by stage, arm, instance and algorithm seed."""

    stage: str
    arm: str
    instance_seed: int
    algorithm_seed: int
    config: dict[str, Any]
    instance_path: str
    instance_hash: str
    initial_path: str
    initial_hash: str
    source_hash: str
    source_commit: str

    @property
    def key(self) -> str:
        """Produce a stable, path-safe identity for idempotent checkpointing."""
        return f"{self.stage}__{self.arm}__i{self.instance_seed}__a{self.algorithm_seed}"

    @property
    def input_hash(self) -> str:
        """Hash every input relevant to result reuse."""
        return stable_hash(asdict(self))
