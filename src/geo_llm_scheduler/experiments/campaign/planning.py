"""Freeze instances, paired initial populations, resource pilot and stage configurations."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.experiments.campaign.model import (
    CampaignSettings,
    JobSpec,
    settings_hash,
)
from geo_llm_scheduler.experiments.campaign.support import (
    atomic_json,
    available_memory_gb,
    free_disk_gb,
    process_peak_rss_gb,
    safe_worker_count,
)
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.mixed_instances import materialize_mixed_instance
from geo_llm_scheduler.experiments.runner import save_result, source_digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager


def repository_commit(repo: Path) -> str:
    """Return the exact checked-out source revision used by every worker."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10
    ).strip()


def _relative(path: Path, repo: Path) -> str:
    return path.relative_to(repo).as_posix()


def freeze_initial(
    repo: Path,
    root: Path,
    instance_seed: int,
    algorithm_seed: int,
    instance_path: Path,
    settings: CampaignSettings,
) -> tuple[str, str]:
    """Materialize one shared genotype population for Full, MOEA/D and NSGA-II."""
    output = root / "initial_populations" / f"i{instance_seed}_a{algorithm_seed}.json"
    if not output.exists():
        problem = load_instance(instance_path)
        base = load_config(repo / settings.base_config)
        config = replace(
            base,
            population=settings.population,
            neighborhood=min(base.neighborhood, settings.population),
        )
        genotypes = initial_genotypes(
            problem, config, RNGManager(algorithm_seed).stream("initialization")
        )
        atomic_json(output, [asdict(g) for g in genotypes])
    return _relative(output, repo), file_hash(output)


def freeze_inputs(repo: Path, root: Path, settings: CampaignSettings) -> dict[str, Any]:
    """Validate and freeze all core cells plus a deterministic optional extra-seed range."""
    if free_disk_gb(repo) < settings.min_free_disk_gb:
        raise OSError("Not enough free disk for campaign preflight")
    manifest = root / "campaign_manifest.json"
    source = source_digest()
    commit = repository_commit(repo)
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if (data.get("settings_hash"), data.get("source_hash"), data.get("source_commit")) != (
            settings_hash(settings),
            source,
            commit,
        ):
            raise ValueError("Frozen campaign code/config changed; refusing to reuse results")
        for record in data["instances"].values():
            if file_hash(repo / record["path"]) != record["sha256"]:
                raise ValueError("Frozen instance hash mismatch")
        for record in data["initial_populations"].values():
            if file_hash(repo / record["path"]) != record["sha256"]:
                raise ValueError("Frozen initial population hash mismatch")
        return data
    root.mkdir(parents=True, exist_ok=True)
    instance_seeds = sorted(set(settings.screening_instance_seeds + settings.final_instance_seeds))
    instances: dict[str, Any] = {}
    initial: dict[str, Any] = {}
    for instance_seed in instance_seeds:
        record = materialize_mixed_instance(
            root / "instances", settings.jobs_per_instance, instance_seed
        )
        absolute = Path(str(record["path"]))
        record["path"] = _relative(absolute, repo)
        instances[str(instance_seed)] = record
        for algorithm_seed in settings.algorithm_seeds:
            path, sha = freeze_initial(
                repo, root, instance_seed, algorithm_seed, absolute, settings
            )
            initial[f"{instance_seed}:{algorithm_seed}"] = {"path": path, "sha256": sha}
    data = {
        "campaign_id": root.name,
        "settings": asdict(settings),
        "settings_hash": settings_hash(settings),
        "source_hash": source,
        "source_commit": commit,
        "instances": instances,
        "initial_populations": initial,
        "screening_cells": [
            [i, a] for i in settings.screening_instance_seeds for a in settings.algorithm_seeds
        ],
        "final_cells": [
            [i, a] for i in settings.final_instance_seeds for a in settings.algorithm_seeds
        ],
        "extra_algorithm_seed_range": [settings.e13_extra_seed_start, settings.e13_extra_seed_stop],
        "purpose": "provisional 60h configuration freeze and E13 paired comparison",
    }
    atomic_json(manifest, data)
    return data


def resource_pilot(repo: Path, root: Path, settings: CampaignSettings) -> dict[str, Any]:
    """Run a one-generation real instance pilot and freeze conservative worker capacity."""
    path = root / "resource_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    before_disk = free_disk_gb(repo)
    before_memory = available_memory_gb()
    instance_path = (
        root
        / "instances"
        / "D02_mixed"
        / (f"jobs{settings.jobs_per_instance}_seed{settings.screening_instance_seeds[0]}.json")
    )
    problem = load_instance(instance_path)
    config = replace(
        load_config(repo / settings.base_config),
        instance=_relative(instance_path, repo),
        population=settings.population,
        neighborhood=min(
            load_config(repo / settings.base_config).neighborhood, settings.population
        ),
        generations=1,
        method="plain",
        seed=settings.algorithm_seeds[0],
        seconds=None,
    )
    from geo_llm_scheduler.engine.run import run  # Avoid import in lightweight status paths.

    start = time.monotonic()
    result = run(problem, config)
    pilot_seconds = time.monotonic() - start
    pilot_peak_rss = process_peak_rss_gb()
    pilot_output = root / "stage00_preflight" / "pilot_run"
    save_result(result, config, file_hash(instance_path), pilot_output)
    pilot_bytes = sum(path.stat().st_size for path in pilot_output.rglob("*") if path.is_file())
    mandatory_arms = 4 + 3 + 6 + 3 + 3 + 3 + 3
    screening_runs = (
        mandatory_arms * len(settings.screening_instance_seeds) * len(settings.algorithm_seeds)
    )
    final_runs = 3 * len(settings.final_instance_seeds) * len(settings.algorithm_seeds)
    allowed_cpus = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    worker_memory_budget = max(settings.memory_per_worker_gb, 1.5 * pilot_peak_rss)
    workers = safe_worker_count(
        allowed_cpus,
        available_memory_gb(),
        settings.memory_reserve_gb,
        worker_memory_budget,
        settings.max_workers,
    )
    plan = {
        "pilot_method": "plain",
        "pilot_generations": 1,
        "pilot_elapsed_s": pilot_seconds,
        "pilot_process_peak_rss_gb": pilot_peak_rss,
        "pilot_exact_evaluations": result.gateway.counts.get("exact", 0),
        "pilot_archive_peak": result.archive.peak_size,
        "pilot_output_bytes": pilot_bytes,
        "free_disk_gb_before": before_disk,
        "free_disk_gb_after": free_disk_gb(repo),
        "available_memory_gb_before": before_memory,
        "available_memory_gb_after": available_memory_gb(),
        "cpu_logical": os.cpu_count(),
        "cpu_allowed": allowed_cpus,
        "workers": workers,
        "worker_memory_budget_gb": worker_memory_budget,
        "worker_cpu_quota": "one pinned logical CPU per active worker where supported",
        "thread_env": [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ],
        "estimated_mandatory_output_gb": round(
            pilot_bytes
            * (screening_runs + final_runs)
            * max(1, settings.screening_seconds / max(pilot_seconds, 1e-9))
            / 1024**3,
            3,
        ),
    }
    if workers < 1:
        raise MemoryError("Insufficient available memory for even one worker")
    atomic_json(path, plan)
    return plan


def config_for_arm(
    base: Config,
    stage: str,
    arm: str,
    selected: dict[str, Any],
    thresholds: dict[str, list[float]] | None = None,
) -> Config:
    """Build one validated ablation configuration without altering algorithm defaults."""
    updates: dict[str, Any] = {"method": "full", "polish": True, "polish_mode": "trajectory"}
    if stage == "e14_online":
        if thresholds is None:
            raise ValueError("E14 candidate thresholds are unavailable")
        updates["severity_thresholds"] = tuple(thresholds[arm])
    else:
        updates["severity_thresholds"] = tuple(selected.get("thresholds", (0.2,) * 6))
    if stage in (
        "e11_trigger",
        "e12_delta",
        "e09_search",
        "e06_macrosearch",
        "e01_controller",
        "e04_polish",
        "e13_full",
    ):
        updates["trigger_mode"] = selected.get("trigger_mode", "preference")
    if stage == "e11_trigger":
        updates["trigger_mode"] = arm
    if stage == "e12_delta":
        if arm == "off":
            updates["trigger_quality_gate"] = False
        else:
            updates["trigger_delta"] = float(arm)
    elif stage in ("e09_search", "e06_macrosearch", "e01_controller", "e04_polish", "e13_full"):
        updates["trigger_delta"] = selected.get("trigger_delta", 0.1)
        updates["trigger_quality_gate"] = selected.get("trigger_quality_gate", True)
    if stage == "e09_search":
        updates["rl_steps"] = int(arm)
    elif stage in ("e06_macrosearch", "e01_controller", "e04_polish", "e13_full"):
        updates["rl_steps"] = int(selected.get("rl_steps", 5))
    if stage == "e06_macrosearch":
        updates["budget_policy"] = arm
    elif stage in ("e01_controller", "e04_polish", "e13_full"):
        updates["budget_policy"] = selected.get("budget_policy", "fixed")
    if stage == "e01_controller":
        updates["controller"] = arm
    elif stage in ("e04_polish", "e13_full"):
        updates["controller"] = selected.get("controller", "qlearning")
    if stage == "e04_polish":
        updates["polish_mode"] = arm
    elif stage == "e13_full":
        updates["polish_mode"] = selected.get("polish_mode", "trajectory")
    return replace(base, **updates)


def make_spec(
    manifest: dict[str, Any],
    stage: str,
    arm: str,
    instance_seed: int,
    algorithm_seed: int,
    config: Config,
) -> JobSpec:
    """Bind an arm to immutable source, instance and shared initialization hashes."""
    instance = manifest["instances"][str(instance_seed)]
    initial = manifest["initial_populations"][f"{instance_seed}:{algorithm_seed}"]
    return JobSpec(
        stage,
        arm,
        instance_seed,
        algorithm_seed,
        asdict(config),
        instance["path"],
        instance["sha256"],
        initial["path"],
        initial["sha256"],
        manifest["source_hash"],
        manifest["source_commit"],
    )
