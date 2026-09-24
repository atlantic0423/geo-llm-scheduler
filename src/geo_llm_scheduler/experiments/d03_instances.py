"""D03 Type-I coverage and Type-II severity-competition instance families."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import cast

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

CONDITIONS = ("Normal", "Resource", "KV", "Region", "TOU", "Demand", "Compressible")
TYPE_I_PROFILES = ("soft", "moderate", "strong")
TYPE_II_PROFILES = ("broad", "burst", "long")
ALGORITHM_SEEDS = (101, 202, 303)
TYPE_I_CALIBRATION_SEEDS = (1701, 1702)
TYPE_II_CALIBRATION_SEEDS = (1801, 1802, 1803)
TYPE_I_EVALUATION_SEEDS = (7101, 7102)
TYPE_II_EVALUATION_SEEDS = (8101, 8102, 8103, 8104, 8105)


def _grid(rng: random.Random, end: int, step: int) -> float:
    return float(rng.randrange(0, end // step + 1) * step)


def _hardware() -> tuple[ServingInstance, ...]:
    return tuple(ServingInstance(f"r{r}m{m}", r, 16, 10, 100) for r in range(3) for m in range(3))


def _tariffs(kind: str, region: int, end: float) -> tuple[Tariff, ...]:
    if kind == "tou":
        return (
            Tariff(0, 900, 0.42),
            Tariff(900, 1800, 0.76),
            Tariff(1800, 2700, 0.34),
            Tariff(2700, 3600, 0.66),
            Tariff(3600, end, 0.48),
        )
    if kind == "region":
        return (Tariff(0, end, (0.34, 0.50, 0.74)[region]),)
    return (Tariff(0, end, 0.48),)


def _type_i_ranges(condition: str, profile: str) -> dict[str, object]:
    if condition not in CONDITIONS or profile not in TYPE_I_PROFILES:
        raise ValueError("Unknown Type-I condition/profile")
    level = TYPE_I_PROFILES.index(profile)
    rows: dict[str, dict[str, object]] = {
        "Normal": dict(
            release=(7200, 120),
            p=(45, 60, 90),
            d=(90, 120, 180),
            c=(0.06, 0.08, 0.10),
            v=(1, 2),
            kv=(0, 10),
            tariff="flat",
            demand=1.0,
        ),
        "Resource": dict(
            release=((1800, 900, 480)[level], 30),
            p=(120, 300, 480),
            d=(300, 600, 900),
            c=((0.30, 0.40, 0.50), (0.40, 0.52, 0.64), (0.48, 0.60, 0.72))[level],
            v=(4, 6, 8),
            kv=(0, 30),
            tariff="flat",
            demand=1.0,
        ),
        "KV": dict(
            release=(3600, 60),
            p=(30, 60, 90),
            d=(60, 120, 180),
            c=(0.06, 0.10, 0.14),
            v=(1, 2),
            kv=((300, 600, 900), (600, 900, 1200), (900, 1500, 2100))[level],
            tariff="flat",
            demand=1.0,
        ),
        "Region": dict(
            release=(3000, 60),
            p=(120, 240, 360),
            d=(300, 600, 900),
            c=(0.10, 0.15, 0.20),
            v=(2, 4),
            kv=(0, 30),
            tariff="region",
            demand=1.0,
        ),
        "TOU": dict(
            release=(1800, 60),
            p=(240, 480, 600),
            d=(600, 900, 1200),
            c=(0.08, 0.12, 0.16),
            v=(2, 4),
            kv=(0, 30),
            tariff="tou",
            demand=1.0,
        ),
        "Demand": dict(
            release=((600, 300, 120)[level], 30),
            p=(180, 300, 480),
            d=(480, 720, 900),
            c=(0.12, 0.20, 0.28),
            v=(2, 4),
            kv=(0, 30),
            tariff="flat",
            demand=5.0,
        ),
        "Compressible": dict(
            release=((10800, 9000, 7200)[level], 120),
            p=(60, 120, 240),
            d=(600, 900, 1200),
            c=(0.06, 0.10, 0.14),
            v=(1, 2),
            kv=(0, 30),
            tariff="flat",
            demand=1.0,
        ),
    }
    return rows[condition]


def type_i_parameters(condition: str, profile: str) -> dict[str, object]:
    """Return all Type-I sampled ranges for manifest provenance."""
    return {
        "suite": "D03",
        "type": "I",
        "target_condition": condition,
        "profile": profile,
        "regions": 3,
        "servers_per_region": 3,
        "jobs": 50,
        "units": {"time": "s", "power": "kW", "energy": "kWh", "currency": "CNY"},
        **_type_i_ranges(condition, profile),
    }


def generate_type_i(condition: str, profile: str, jobs: int, seed: int) -> ProblemInstance:
    """Generate a deterministic condition-directed but non-labelled Type-I instance."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    p = _type_i_ranges(condition, profile)
    release_spec = cast(tuple[int, int], p["release"])
    prefill_choices = cast(tuple[float, ...], p["p"])
    decode_choices = cast(tuple[float, ...], p["d"])
    compute_choices = cast(tuple[float, ...], p["c"])
    vram_choices = cast(tuple[float, ...], p["v"])
    kv_choices = cast(tuple[float, ...], p["kv"])
    rng = random.Random(seed)
    requests = tuple(
        Job(
            f"D03I_{condition}_{i}",
            _grid(rng, *release_spec),
            Profile(
                rng.choice(prefill_choices), rng.choice(compute_choices), rng.choice(vram_choices)
            ),
            Profile(
                rng.choice(decode_choices), rng.choice(compute_choices), rng.choice(vram_choices)
            ),
            rng.choice(kv_choices),
        )
        for i in range(jobs)
    )
    end = (
        max(j.release for j in requests)
        + sum(j.prefill.duration + j.decode.duration + j.kv_delay for j in requests)
        + 86400
    )
    tariff_kind = cast(str, p["tariff"])
    demand_rate = cast(float, p["demand"])
    regions = tuple(Region(f"r{r}", _tariffs(tariff_kind, r, end), demand_rate) for r in range(3))
    problem = ProblemInstance(requests, regions, _hardware())
    validate_problem(problem)
    return problem


def type_ii_parameters(profile: str) -> dict[str, object]:
    """Return a moderate multi-mechanism Type-II search region."""
    if profile not in TYPE_II_PROFILES:
        raise ValueError("Unknown Type-II profile")
    settings = {
        "broad": dict(
            burst=0.35,
            release=6000,
            p=(60, 180, 360, 540),
            d=(180, 480, 720, 1080),
            c=(0.10, 0.24, 0.42, 0.60),
            kv=(0, 90, 450, 900),
        ),
        "burst": dict(
            burst=0.60,
            release=4200,
            p=(90, 240, 420),
            d=(300, 600, 900),
            c=(0.15, 0.30, 0.50),
            kv=(0, 120, 600, 900),
        ),
        "long": dict(
            burst=0.25,
            release=7800,
            p=(60, 240, 480, 600),
            d=(300, 600, 900, 1200),
            c=(0.08, 0.20, 0.38, 0.56),
            kv=(0, 120, 600, 1200),
        ),
    }[profile]
    return {
        "suite": "D03",
        "type": "II",
        "profile": profile,
        "purpose": "severity competition",
        "regions": 3,
        "servers_per_region": 3,
        "jobs": 50,
        "vram_gb": [2, 4, 6],
        "demand_rate_cny_kw": 5.0,
        "tou_cny_kwh": [0.42, 0.72, 0.36, 0.66, 0.48],
        "units": {"time": "s", "power": "kW", "energy": "kWh", "currency": "CNY"},
        **settings,
    }


def generate_type_ii(profile: str, jobs: int, seed: int) -> ProblemInstance:
    """Generate one deterministic Type-II instance with competing mechanisms."""
    p = type_ii_parameters(profile)
    burst = cast(float, p["burst"])
    release_end = cast(int, p["release"])
    prefill_choices = cast(tuple[float, ...], p["p"])
    decode_choices = cast(tuple[float, ...], p["d"])
    compute_choices = cast(tuple[float, ...], p["c"])
    kv_choices = cast(tuple[float, ...], p["kv"])
    rng = random.Random(seed)
    requests = []
    for i in range(jobs):
        release = (
            float(rng.choice((300, 1500, 3000)) + rng.choice((-120, -60, 0, 60, 120)))
            if rng.random() < burst
            else _grid(rng, release_end, 60)
        )
        requests.append(
            Job(
                f"D03II_{profile}_{i}",
                release,
                Profile(
                    rng.choice(prefill_choices), rng.choice(compute_choices), rng.choice((2, 4, 6))
                ),
                Profile(
                    rng.choice(decode_choices), rng.choice(compute_choices), rng.choice((2, 4, 6))
                ),
                rng.choice(kv_choices),
            )
        )
    frozen = tuple(requests)
    end = (
        max(j.release for j in frozen)
        + sum(j.prefill.duration + j.decode.duration + j.kv_delay for j in frozen)
        + 86400
    )
    tariffs = (
        Tariff(0, 900, 0.42),
        Tariff(900, 1800, 0.72),
        Tariff(1800, 2700, 0.36),
        Tariff(2700, 3600, 0.66),
        Tariff(3600, end, 0.48),
    )
    problem = ProblemInstance(
        frozen, tuple(Region(f"r{r}", tariffs, 5.0) for r in range(3)), _hardware()
    )
    validate_problem(problem)
    return problem


def materialize(
    root: str | Path, suite_type: str, family: str, profile: str, jobs: int, seed: int, role: str
) -> dict[str, object]:
    """Write a D03 instance plus complete deterministic manifest row."""
    problem = (
        generate_type_i(family, profile, jobs, seed)
        if suite_type == "type_i"
        else generate_type_ii(profile, jobs, seed)
    )
    scenario = f"D03I_{family}" if suite_type == "type_i" else f"D03II_{profile}"
    parameters = (
        type_i_parameters(family, profile)
        if suite_type == "type_i"
        else type_ii_parameters(profile)
    )
    path = Path(root) / role / scenario / f"jobs{jobs}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_instance(problem, path)
    return {
        "instance_id": f"{scenario}_j{jobs}_i{seed}",
        "scenario": scenario,
        "variant": profile,
        "suite_type": suite_type,
        "role": role,
        "target_condition": family if suite_type == "type_i" else "",
        "profile": profile,
        "jobs": jobs,
        "instance_seed": seed,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters_json": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
    }
