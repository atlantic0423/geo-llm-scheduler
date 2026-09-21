"""Controlled reproducible instances for D01 behavior diagnostics."""

import csv
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

SCENARIOS = (
    "D0_balanced",
    "D1_resource",
    "D2_kv",
    "D3_region",
    "D4_tou",
    "D5_demand",
    "D6_compressible",
)
DEMAND_VARIANTS = ("single_peak", "multi_tied_peak", "mixed_peak")


def _choice(rng: random.Random, values: tuple[float, ...]) -> float:
    return rng.choice(values)


def _release(rng: random.Random, start: int, end: int, step: int = 60) -> float:
    return float(rng.randrange(start // step, end // step + 1) * step)


def _job(scenario: str, index: int, rng: random.Random, variant: str | None) -> Job:
    """Generate one scenario-specific request without assigning a target state label."""
    if scenario == "D0_balanced":
        release = _release(rng, 0, 3600)
        prefill = Profile(_choice(rng, (60, 120, 240)), _choice(rng, (0.15, 0.2)), 2)
        decode = Profile(_choice(rng, (180, 300, 600)), _choice(rng, (0.15, 0.2)), 2)
        kv = _choice(rng, (0, 15, 30))
    elif scenario == "D1_resource":
        release = _release(rng, 0, 300) if rng.random() < 0.75 else _release(rng, 300, 900)
        prefill = Profile(
            _choice(rng, (120, 300, 600)), _choice(rng, (0.45, 0.6, 0.75)), _choice(rng, (4, 6, 8))
        )
        decode = Profile(
            _choice(rng, (300, 600, 900)), _choice(rng, (0.45, 0.6, 0.75)), _choice(rng, (4, 6, 8))
        )
        kv = _choice(rng, (0, 15, 30))
    elif scenario == "D2_kv":
        release = _release(rng, 0, 1800)
        prefill = Profile(_choice(rng, (60, 120)), _choice(rng, (0.1, 0.15)), 2)
        decode = Profile(_choice(rng, (120, 240)), _choice(rng, (0.1, 0.15)), 2)
        kv = _choice(rng, (600, 900, 1200))
    elif scenario == "D3_region":
        release = _release(rng, 0, 1800)
        prefill = Profile(_choice(rng, (120, 240, 300)), _choice(rng, (0.1, 0.15)), 2)
        decode = Profile(_choice(rng, (300, 600)), _choice(rng, (0.1, 0.15)), 2)
        kv = _choice(rng, (0, 15, 30))
    elif scenario == "D4_tou":
        release = _release(rng, 0, 1200)
        prefill = Profile(_choice(rng, (300, 600)), _choice(rng, (0.1, 0.15)), 2)
        decode = Profile(_choice(rng, (600, 900, 1200)), _choice(rng, (0.1, 0.15)), 2)
        kv = _choice(rng, (0, 15, 30))
    elif scenario == "D5_demand":
        if variant == "single_peak":
            release = _release(rng, 0, 120, 30) if rng.random() < 0.8 else _release(rng, 1800, 3600)
        elif variant == "multi_tied_peak":
            base = 0 if index % 2 == 0 else 900
            release = float(base + rng.randrange(0, 5) * 30)
        else:
            draw = rng.random()
            if draw < 0.65:
                release = _release(rng, 0, 120, 30)
            elif draw < 0.95:
                release = _release(rng, 900, 1020, 30)
            else:
                release = _release(rng, 1800, 2700)
        prefill = Profile(_choice(rng, (300, 600)), _choice(rng, (0.15, 0.2)), 2)
        decode = Profile(_choice(rng, (600, 900)), _choice(rng, (0.15, 0.2)), 2)
        kv = _choice(rng, (0, 15, 30))
    else:
        release = _release(rng, 0, 7200)
        prefill = Profile(_choice(rng, (60, 120)), _choice(rng, (0.1, 0.15)), 2)
        decode = Profile(_choice(rng, (600, 900)), _choice(rng, (0.1, 0.15)), 2)
        kv = _choice(rng, (0, 15))
    return Job(f"{scenario}_j{index}", release, prefill, decode, kv)


def _tariffs(scenario: str, region: int, coverage: float) -> tuple[Tariff, ...]:
    """Return complete contiguous scenario tariffs on the global second axis."""
    if scenario == "D4_tou":
        return (
            Tariff(0, 900, 0.25),
            Tariff(900, 1800, 1.25),
            Tariff(1800, 2700, 0.20),
            Tariff(2700, coverage, 0.45),
        )
    if scenario == "D0_balanced":
        return (Tariff(0, coverage, 0.45),)
    if scenario == "D3_region":
        return (Tariff(0, coverage, (0.25, 0.75, 1.25)[region]),)
    return (Tariff(0, coverage, 0.45),)


def diagnostic_parameters(scenario: str, variant: str | None = None) -> dict[str, object]:
    """Describe the complete generator convention stored in every manifest row."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown diagnostic scenario: {scenario}")
    if scenario == "D5_demand":
        variant = variant or "mixed_peak"
        if variant not in DEMAND_VARIANTS:
            raise ValueError(f"Unknown D5 variant: {variant}")
    elif variant is not None:
        raise ValueError("Only D5 accepts a variant")
    scenario_profiles: dict[str, dict[str, object]] = {
        "D0_balanced": {
            "release": {"kind": "uniform_grid", "start_s": 0, "end_s": 3600, "step_s": 60},
            "prefill_s": [60, 120, 240],
            "decode_s": [180, 300, 600],
            "compute": [0.15, 0.2],
            "vram_gb": [2],
            "kv_s": [0, 15, 30],
            "flat_tariff_cny_kwh": 0.45,
        },
        "D1_resource": {
            "release": {
                "kind": "mixture",
                "weights": [0.75, 0.25],
                "components": [
                    {"start_s": 0, "end_s": 300, "step_s": 60},
                    {"start_s": 300, "end_s": 900, "step_s": 60},
                ],
            },
            "prefill_s": [120, 300, 600],
            "decode_s": [300, 600, 900],
            "compute": [0.45, 0.6, 0.75],
            "vram_gb": [4, 6, 8],
            "kv_s": [0, 15, 30],
            "flat_tariff_cny_kwh": 0.45,
        },
        "D2_kv": {
            "release": {"kind": "uniform_grid", "start_s": 0, "end_s": 1800, "step_s": 60},
            "prefill_s": [60, 120],
            "decode_s": [120, 240],
            "compute": [0.1, 0.15],
            "vram_gb": [2],
            "kv_s": [600, 900, 1200],
            "flat_tariff_cny_kwh": 0.45,
        },
        "D3_region": {
            "release": {"kind": "uniform_grid", "start_s": 0, "end_s": 1800, "step_s": 60},
            "prefill_s": [120, 240, 300],
            "decode_s": [300, 600],
            "compute": [0.1, 0.15],
            "vram_gb": [2],
            "kv_s": [0, 15, 30],
            "flat_region_prices_cny_kwh": [0.25, 0.75, 1.25],
        },
        "D4_tou": {
            "release": {"kind": "uniform_grid", "start_s": 0, "end_s": 1200, "step_s": 60},
            "prefill_s": [300, 600],
            "decode_s": [600, 900, 1200],
            "compute": [0.1, 0.15],
            "vram_gb": [2],
            "kv_s": [0, 15, 30],
            "tou_cny_kwh": [
                [0, 900, 0.25],
                [900, 1800, 1.25],
                [1800, 2700, 0.2],
                [2700, "coverage", 0.45],
            ],
        },
        "D5_demand": {
            "release": {
                "kind": variant,
                "single_peak": {
                    "weights": [0.8, 0.2],
                    "components": [
                        {"start_s": 0, "end_s": 120, "step_s": 30},
                        {"start_s": 1800, "end_s": 3600, "step_s": 60},
                    ],
                },
                "multi_tied_peak": {
                    "assignment": "job_index_parity",
                    "batch_starts_s": [0, 900],
                    "jitter_start_s": 0,
                    "jitter_end_s": 120,
                    "step_s": 30,
                },
                "mixed_peak": {
                    "weights": [0.65, 0.30, 0.05],
                    "components": [
                        {"start_s": 0, "end_s": 120, "step_s": 30},
                        {"start_s": 900, "end_s": 1020, "step_s": 30},
                        {"start_s": 1800, "end_s": 2700, "step_s": 60},
                    ],
                },
            },
            "prefill_s": [300, 600],
            "decode_s": [600, 900],
            "compute": [0.15, 0.2],
            "vram_gb": [2],
            "kv_s": [0, 15, 30],
            "flat_tariff_cny_kwh": 0.45,
        },
        "D6_compressible": {
            "release": {"kind": "uniform_grid", "start_s": 0, "end_s": 7200, "step_s": 60},
            "prefill_s": [60, 120],
            "decode_s": [600, 900],
            "compute": [0.1, 0.15],
            "vram_gb": [2],
            "kv_s": [0, 15],
            "flat_tariff_cny_kwh": 0.45,
        },
    }
    profiles = scenario_profiles[scenario]
    return {
        "scenario": scenario,
        "variant": variant,
        "regions": 3,
        "servers_per_region": 3,
        "compute_capacity": 1.0,
        "vram_capacity_gb": 16,
        "idle_kw": 10,
        "active_kw": 100,
        "demand_rate_cny_kw": 20.0 if scenario == "D5_demand" else 0.2,
        "units": {"time": "s", "power": "kW", "energy": "kWh", "currency": "CNY"},
        **profiles,
    }


def generate_diagnostic_instance(
    scenario: str, jobs: int, instance_seed: int, variant: str | None = None
) -> ProblemInstance:
    """Generate a deterministic scenario instance with a separate instance seed."""
    if type(jobs) is not int or jobs < 1 or type(instance_seed) is not int:
        raise ValueError("Positive jobs and integer instance_seed required")
    parameters = diagnostic_parameters(scenario, variant)
    resolved = parameters["variant"] if scenario == "D5_demand" else None
    variant = resolved if isinstance(resolved, str) else None
    rng = random.Random(instance_seed)
    requests = tuple(
        _job(scenario, i, rng, variant if isinstance(variant, str) else None) for i in range(jobs)
    )
    coverage = (
        max(job.release for job in requests)
        + sum(job.prefill.duration + job.decode.duration + job.kv_delay for job in requests)
        + 86400
    )
    demand_rate = 20.0 if scenario == "D5_demand" else 0.2
    regions = tuple(Region(f"r{r}", _tariffs(scenario, r, coverage), demand_rate) for r in range(3))
    instances = tuple(
        ServingInstance(f"r{r}m{m}", r, 16, 10, 100) for r in range(3) for m in range(3)
    )
    problem = ProblemInstance(requests, regions, instances)
    validate_problem(problem)
    return problem


def materialize_diagnostic_instance(
    root: str | Path,
    scenario: str,
    jobs: int,
    instance_seed: int,
    variant: str | None = None,
) -> dict[str, object]:
    """Write one canonical JSON instance and return its complete manifest record."""
    parameters = diagnostic_parameters(scenario, variant)
    resolved = parameters["variant"] if scenario == "D5_demand" else None
    variant = resolved if isinstance(resolved, str) else None
    suffix = f"_{variant}" if variant else ""
    path = Path(root) / scenario / f"jobs{jobs}_seed{instance_seed}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_instance(
        generate_diagnostic_instance(
            scenario, jobs, instance_seed, variant if isinstance(variant, str) else None
        ),
        path,
    )
    payload = path.read_bytes()
    return {
        "instance_id": f"{scenario}_jobs{jobs}_iseed{instance_seed}{suffix}",
        "scenario": scenario,
        "variant": variant or "",
        "jobs": jobs,
        "instance_seed": instance_seed,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parameters_json": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
    }


def write_manifest(records: list[dict[str, object]], path: str | Path) -> None:
    """Write deterministic manifest rows suitable for direct readback checks."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance_id",
        "scenario",
        "variant",
        "jobs",
        "instance_seed",
        "path",
        "sha256",
        "parameters_json",
    ]
    ordered = sorted(records, key=lambda row: str(row["instance_id"]))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
