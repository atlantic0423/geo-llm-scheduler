"""Reproducible, mechanism-oriented stress inputs; never select by solver wins.

Profiles remain machine-independent and all machines support both stages.
The ten strata are overlapping stress mechanisms, not ten isolated ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from random import Random

from geo_llm_scheduler.domain.models import (
    Job,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.io.validation import validate_problem


@dataclass(frozen=True)
class Family:
    """Declared synthetic mechanisms and their input-level control parameters."""

    key: str
    title: str
    operators: tuple[str, ...]
    slot: int = 900
    peak: float = 2.0
    low: float = 0.4
    demand: float = 0.6
    kv: tuple[int, ...] = (30, 60, 120)
    burst: int = 120
    machines_per_job: float = 0.09
    cheap_scarce: bool = False
    # interactive, compute-heavy, memory-heavy, long-decode, concurrent-prefill
    mix: tuple[int, ...] = (2, 2, 2, 2, 2)


FAMILIES = (
    Family("F01", "突发拥塞与长短任务交错", ("A1", "A4", "A5", "P2_FLOW"), burst=30),
    Family("F02", "算力显存互补与碎片", ("A1", "A2", "A7"), mix=(1, 3, 3, 1, 2)),
    Family("F03", "KV路径与资源竞争", ("A2", "A5", "P2_FLOW"), kv=(180, 300, 450)),
    Family("F04", "区域电价交替反转", ("A3", "P2_MIXED", "RIGHTSHIFT"), slot=1200),
    Family("F05", "低价区域容量稀缺", ("A3", "P2_MIXED"), cheap_scarce=True),
    Family("F06", "忙碌区间碎片与并发打包", ("A7", "RIGHTSHIFT"), mix=(3, 1, 1, 2, 3)),
    Family("F07", "并列峰与需量分时冲突", ("A8", "P2_MIXED", "RIGHTSHIFT"), demand=2.4),
    Family(
        "F08",
        "高低价交替与Decode排队",
        ("RIGHTSHIFT", "A1", "A4"),
        slot=600,
        peak=2.4,
        low=0.35,
        mix=(1, 1, 1, 3, 4),
    ),
    Family(
        "F09",
        "并发Prefill联合右移",
        ("RIGHTSHIFT_COMMON", "A7"),
        slot=900,
        mix=(1, 1, 1, 2, 5),
        machines_per_job=0.08,
    ),
    Family(
        "F10",
        "多机制耦合",
        ("A1", "A2", "A3", "A4", "A5", "A7", "A8", "P2", "RIGHTSHIFT"),
        slot=750,
        demand=1.2,
        kv=(90, 180, 300),
        burst=60,
        cheap_scarce=True,
        mix=(1, 2, 2, 2, 3),
    ),
)


@dataclass(frozen=True)
class InstanceSpec:
    """One independent input, with an input RNG seed distinct from a solver seed."""

    jobs: int
    family: str
    replicate: int
    regions: int
    seed: int

    @property
    def key(self) -> str:
        """Stable input identifier, unaffected by audit results."""
        return f"mx_n{self.jobs:03d}_{self.family.lower()}_r{self.replicate:02d}"


PANEL_SIZES = (50, 100, 200)
# Large-scale extension (user, 2026-11-08): the same families, replicates and seed
# formula at 500 jobs; it never changes the frozen 120-input panel.
EXTENDED_SIZES = (500,)


def panel_specs(sizes: tuple[int, ...] = PANEL_SIZES) -> tuple[InstanceSpec, ...]:
    """Return ten strata x ``sizes`` x four independent seeds (default: the frozen 120)."""
    if not sizes or not set(sizes) <= set(PANEL_SIZES + EXTENDED_SIZES):
        raise ValueError("Unsupported stress size")
    return tuple(
        InstanceSpec(n, f.key, r, 3 if r <= 2 else 5, 202610310000 + n * 100 + i * 10 + r)
        for n in sizes
        for i, f in enumerate(FAMILIES, 1)
        for r in range(1, 5)
    )


def _profiles(kind: int, rng: Random) -> tuple[Profile, Profile]:
    if kind == 0:
        return (
            Profile(rng.choice((45, 60, 90)), 0.45, 3),
            Profile(rng.choice((120, 180, 300)), 0.30, 6),
        )
    if kind == 1:
        return (
            Profile(rng.choice((240, 360, 600)), rng.choice((0.55, 0.65, 0.75)), 3),
            Profile(rng.choice((300, 600, 900)), 0.20, rng.choice((10, 12))),
        )
    if kind == 2:
        return (
            Profile(rng.choice((120, 240, 360)), 0.25, rng.choice((8, 10))),
            Profile(rng.choice((450, 900, 1350)), rng.choice((0.55, 0.70)), 4),
        )
    if kind == 3:
        return (
            Profile(rng.choice((90, 150, 240)), 0.35, 4),
            Profile(rng.choice((1200, 1800, 2700)), rng.choice((0.50, 0.65)), 12),
        )
    return (
        Profile(rng.choice((300, 450, 600, 750)), rng.choice((0.18, 0.22, 0.28)), 3),
        Profile(rng.choice((600, 1200, 1800)), rng.choice((0.60, 0.75)), 14),
    )


def _tariffs(f: Family, region: int, count: int, horizon: float) -> tuple[Tariff, ...]:
    # Four alternating plateaus; region-specific offsets reverse the cheap-region ranking.
    # TOU edges deliberately need not coincide with fixed 900-second demand windows.
    phase = round(4 * f.slot * region / count / 30) * 30
    prices = (f.peak, f.low, 0.8 * f.peak, 1.3 * f.low)
    scale = 0.8 if f.cheap_scarce and region == 0 else 1 + 0.04 * region
    edges = sorted(
        {0.0, horizon}
        | {
            float(t - phase)
            for t in range(0, int(horizon + phase) + f.slot, f.slot)
            if 0 < t - phase < horizon
        }
    )
    return tuple(
        Tariff(a, b, prices[int((a + phase) // f.slot) % 4] * scale)
        for a, b in zip(edges, edges[1:])
    )


def mechanism_problem(spec: InstanceSpec) -> ProblemInstance:
    """Generate one valid frozen-model input without inspecting any solver output."""
    families = {f.key: f for f in FAMILIES}
    if spec.jobs not in PANEL_SIZES + EXTENDED_SIZES or spec.family not in families:
        raise ValueError("Unsupported stress size or family")
    if spec.regions not in (3, 5) or spec.replicate not in range(1, 5):
        raise ValueError("Expected three/five regions and replicate 1..4")
    f, rng = families[spec.family], Random(spec.seed)
    kinds = [k for k, weight in enumerate(f.mix) for _ in range(weight)]
    kinds = (kinds * ceil(spec.jobs / len(kinds)))[: spec.jobs]
    rng.shuffle(kinds)
    jobs = []
    for i, kind in enumerate(kinds):
        prefill, decode = _profiles(kind, rng)
        # Three recurrent arrival waves. 20% background arrivals retain non-burst work.
        wave = rng.randrange(3)
        release = wave * 2 * f.slot + rng.randrange(f.burst // 15 + 1) * 15
        if rng.random() < 0.2:
            release = rng.randrange((4 * f.slot + f.burst) // 15 + 1) * 15
        jobs.append(Job(f"j{i:03d}", release, prefill, decode, rng.choice(f.kv)))
    # Identity is unrelated to release/size ranks, avoiding privileged OS labels.
    count = max(2 * spec.regions, ceil(spec.jobs * f.machines_per_job))
    counts = [2] * spec.regions
    for i in range(count - sum(counts)):
        region = 1 + i % (spec.regions - 1) if f.cheap_scarce else i % spec.regions
        counts[region] += 1
    machines = tuple(
        ServingInstance(f"r{r}m{k}", r, 16, 0.06, 0.4)
        for r, amount in enumerate(counts)
        for k in range(amount)
    )
    # A fully serial schedule on any machine is covered, plus a full price cycle.
    horizon = (
        max(j.release for j in jobs)
        + sum(j.prefill.duration + j.decode.duration + j.kv_delay for j in jobs)
        + 8 * f.slot
    )
    regions = tuple(
        Region(f"r{r}", _tariffs(f, r, spec.regions, horizon), f.demand * (1 + 0.05 * r))
        for r in range(spec.regions)
    )
    problem = ProblemInstance(tuple(jobs), regions, machines)
    validate_problem(problem)
    return problem


def input_features(problem: ProblemInstance) -> dict[str, float | int]:
    """Report input structure only; these numbers are not optimality-gap estimates."""
    jobs = problem.jobs
    work = sum(p.duration * p.compute for j in jobs for p in (j.prefill, j.decode))
    return {
        "jobs": len(jobs),
        "regions": len(problem.regions),
        "machines": len(problem.instances),
        "release_span_s": max(j.release for j in jobs) - min(j.release for j in jobs),
        "compute_work_s": work,
        "compute_work_per_machine_s": work / len(problem.instances),
        "flow_lower_bound_s": sum(j.prefill.duration + j.decode.duration for j in jobs),
        "concurrent_prefill_jobs": sum(
            j.prefill.compute <= 0.3 and j.prefill.vram <= 4 for j in jobs
        ),
        "long_decode_jobs": sum(j.decode.duration >= 1200 for j in jobs),
        "max_kv_s": max(j.kv_delay for j in jobs),
        "price_min": min(t.price for r in problem.regions for t in r.tariffs),
        "price_max": max(t.price for r in problem.regions for t in r.tariffs),
        "demand_rate_min": min(r.demand_rate for r in problem.regions),
        "demand_rate_max": max(r.demand_rate for r in problem.regions),
    }
