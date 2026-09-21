"""Uncalibrated mixed random instances for D02 behavioral validation."""

import hashlib
import json
import random
from pathlib import Path

from geo_llm_scheduler.domain.models import (
    Job,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.io.validation import validate_problem

MIXED_SCENARIO = "D02_mixed"
MIXED_INSTANCE_SEEDS = (11, 22, 33, 44, 55)
MIXED_ALGORITHM_SEEDS = (101, 202, 303)


def mixed_parameters() -> dict[str, object]:
    """Return the frozen D02 sampling convention recorded in every manifest row."""
    return {
        "scenario": MIXED_SCENARIO,
        "purpose": "general-purpose random validation without a target DominantCondition",
        "calibration_policy": "frozen before observing state incidence; no post-hoc calibration",
        "regions": 3,
        "servers_per_region": 3,
        "compute_capacity": 1.0,
        "vram_capacity_gb": 16,
        "idle_kw": 10,
        "active_kw": 100,
        "demand_rate_cny_kw": 5.0,
        "release": {
            "kind": "mixture",
            "weights": [0.8, 0.2],
            "dispersed": {"start_s": 0, "end_s": 5400, "step_s": 60},
            "mild_bursts": {
                "centers_s": [900, 2700, 4500],
                "offsets_s": [-120, -60, 0, 60, 120],
            },
        },
        "prefill_s": [60, 120, 240, 360],
        "decode_s": [180, 300, 600, 900],
        "compute": [0.15, 0.3, 0.45, 0.6],
        "vram_gb": [2, 4, 6],
        "kv_s": [0, 30, 60, 120],
        "tou_cny_kwh": [
            [0, 900, 0.42],
            [900, 1800, 0.72],
            [1800, 2700, 0.35],
            [2700, 3600, 0.62],
            [3600, "coverage", 0.48],
        ],
        "units": {"time": "s", "power": "kW", "energy": "kWh", "currency": "CNY"},
    }


def _grid(rng: random.Random, start: int, end: int, step: int) -> float:
    return float(rng.randrange(start // step, end // step + 1) * step)


def generate_mixed_instance(jobs: int, instance_seed: int) -> ProblemInstance:
    """Generate one deterministic D02 instance without inspecting state incidence."""
    if type(jobs) is not int or jobs < 1 or type(instance_seed) is not int:
        raise ValueError("Positive jobs and integer instance_seed required")
    rng = random.Random(instance_seed)
    requests = []
    for index in range(jobs):
        if rng.random() < 0.8:
            release = _grid(rng, 0, 5400, 60)
        else:
            release = float(rng.choice((900, 2700, 4500)) + rng.choice((-120, -60, 0, 60, 120)))
        prefill = Profile(
            rng.choice((60, 120, 240, 360)),
            rng.choice((0.15, 0.3, 0.45, 0.6)),
            rng.choice((2, 4, 6)),
        )
        decode = Profile(
            rng.choice((180, 300, 600, 900)),
            rng.choice((0.15, 0.3, 0.45, 0.6)),
            rng.choice((2, 4, 6)),
        )
        requests.append(
            Job(
                f"{MIXED_SCENARIO}_j{index}", release, prefill, decode, rng.choice((0, 30, 60, 120))
            )
        )
    coverage = (
        max(job.release for job in requests)
        + sum(job.prefill.duration + job.decode.duration + job.kv_delay for job in requests)
        + 86400
    )
    tariffs = (
        Tariff(0, 900, 0.42),
        Tariff(900, 1800, 0.72),
        Tariff(1800, 2700, 0.35),
        Tariff(2700, 3600, 0.62),
        Tariff(3600, coverage, 0.48),
    )
    regions = tuple(Region(f"r{region}", tariffs, 5.0) for region in range(3))
    instances = tuple(
        ServingInstance(f"r{region}m{server}", region, 16, 10, 100)
        for region in range(3)
        for server in range(3)
    )
    problem = ProblemInstance(tuple(requests), regions, instances)
    validate_problem(problem)
    return problem


def materialize_mixed_instance(
    root: str | Path, jobs: int, instance_seed: int
) -> dict[str, object]:
    """Write one canonical D02 JSON instance and its manifest record."""
    path = Path(root) / MIXED_SCENARIO / f"jobs{jobs}_seed{instance_seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_instance(generate_mixed_instance(jobs, instance_seed), path)
    payload = path.read_bytes()
    return {
        "instance_id": f"{MIXED_SCENARIO}_jobs{jobs}_iseed{instance_seed}",
        "scenario": MIXED_SCENARIO,
        "variant": "",
        "jobs": jobs,
        "instance_seed": instance_seed,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parameters_json": json.dumps(mixed_parameters(), ensure_ascii=False, sort_keys=True),
    }
