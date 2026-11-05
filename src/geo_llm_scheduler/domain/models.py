"""Immutable problem data and value objects; MS alone owns assignment.

Operation indices are 2*job for Prefill and 2*job+1 for Decode.
All times are seconds and all costs are CNY.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """Deterministic duration and incremental resource demands."""

    duration: float
    compute: float
    vram: float


@dataclass(frozen=True)
class Job:
    """A released two-stage request with region-local KV handoff delay."""

    name: str
    release: float
    prefill: Profile
    decode: Profile
    kv_delay: float = 0.0


@dataclass(frozen=True)
class Tariff:
    """Half-open global-time tariff segment, in CNY/kWh."""

    start: float
    end: float
    price: float


@dataclass(frozen=True)
class Region:
    """Region tariff and fixed-window demand rate, in CNY/kW."""

    name: str
    tariffs: tuple[Tariff, ...]
    demand_rate: float


@dataclass(frozen=True)
class ServingInstance:
    """Atomic server; VRAM excludes persistent model/runtime reservation."""

    name: str
    region: int
    vram: float
    idle_kw: float
    active_kw: float
    phases: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class PowerModel:
    """Load-dependent active power shared by every serving instance.

    While at least one operation runs, an instance draws
    ``idle + (active - idle) * (beta + (1 - beta) * load)``, where ``load`` sums
    ``kappa_phase * compute`` over the running operations. ``beta`` is the share of
    the dynamic range drawn as soon as the GPU is busy. Because the capacity rule
    keeps the running Compute sum at most one and both kappas are at most one,
    power never exceeds ``active``. ``beta == 1`` is the original two-state model.
    """

    beta: float = 1.0
    kappa_prefill: float = 1.0
    kappa_decode: float = 1.0

    @property
    def legacy(self) -> bool:
        """True when power is the original idle/active two-state model."""
        return self.beta == 1.0


@dataclass(frozen=True)
class ProblemInstance:
    """Shared immutable input; indices reference tuple positions."""

    jobs: tuple[Job, ...]
    regions: tuple[Region, ...]
    instances: tuple[ServingInstance, ...]
    power_model: PowerModel = PowerModel()

    @property
    def operation_count(self) -> int:
        """Return two operations per job."""
        return 2 * len(self.jobs)

    def floor_kw(self, machine: int) -> float:
        """Power above idle drawn on a machine whenever any operation runs there."""
        server = self.instances[machine]
        return self.power_model.beta * (server.active_kw - server.idle_kw)

    def load_kw(self, operation: int, machine: int) -> float:
        """Additional power one running operation draws on the given machine."""
        model = self.power_model
        if model.legacy:
            return 0.0
        server = self.instances[machine]
        kappa = model.kappa_decode if operation % 2 else model.kappa_prefill
        return (
            (1.0 - model.beta)
            * (server.active_kw - server.idle_kw)
            * kappa
            * self.profile(operation).compute
        )

    def profile(self, operation: int) -> Profile:
        """Return the homogeneous profile of an operation."""
        job = self.jobs[operation // 2]
        return job.prefill if operation % 2 == 0 else job.decode

    def eligible(self, operation: int, region: int | None = None) -> tuple[int, ...]:
        """Return capacity- and phase-eligible server indices."""
        p = self.profile(operation)
        return tuple(
            m
            for m, server in enumerate(self.instances)
            if operation % 2 in server.phases
            and p.compute <= 1
            and p.vram <= server.vram
            and (region is None or server.region == region)
        )


@dataclass(frozen=True)
class Genotype:
    """MS is the sole assignment authority; OS has each job twice."""

    ms: tuple[int, ...]
    os: tuple[int, ...]


@dataclass(frozen=True)
class Schedule:
    """Timing and diagnostics only: no independent assignment state."""

    starts: tuple[float, ...]
    resource_wait: tuple[float, ...] = ()
    intentional_wait: tuple[float, ...] = ()

    def end(self, problem: ProblemInstance, operation: int) -> float:
        """Compute completion from immutable duration and start."""
        return self.starts[operation] + problem.profile(operation).duration

    def horizon(self, problem: ProblemInstance) -> float:
        """Return the actual batch end (zero for an empty batch)."""
        return max(
            (self.end(problem, o) for o in range(1, problem.operation_count, 2)), default=0.0
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Exact objective decomposition and region-indexed window diagnostics."""

    feasible: bool
    flow: float
    tou: float
    demand: tuple[float, ...]
    windows: tuple[tuple[float, ...], ...]
    violations: tuple[str, ...] = ()

    @property
    def bill(self) -> float:
        """Return total TOU plus regional Demand Charge."""
        return self.tou + sum(self.demand)

    @property
    def objectives(self) -> tuple[float, float]:
        """Return the minimizing objective vector."""
        return self.flow, self.bill


@dataclass(frozen=True)
class Candidate:
    """A complete evaluated solution with immutable provenance."""

    genotype: Genotype
    schedule: Schedule
    evaluation: EvaluationResult
    origin: str = "unknown"
