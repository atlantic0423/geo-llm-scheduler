"""Opt-in terminal admission bounds; heuristics never certify an improvement.

All quantities use the incumbent's fixed assignment. Bounds relax precedence,
release and capacity conflicts; they are optimistic, not attainable schedules.
"""

from dataclasses import dataclass
from math import ceil

from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.scheduling.resources import union_length
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import TOL


@dataclass(frozen=True)
class GatePolicy:
    """Archive score quantile floors and an optional per-genotype invocation cap."""

    shift: float | None = None
    pack: float | None = None
    peak: float | None = None
    cap: int | None = None


POLICIES = {
    "TG_BOUND": GatePolicy(0.0, 0.0, 0.0),
    "TG_SHIFT": GatePolicy(shift=0.5),
    "TG_PACK": GatePolicy(pack=0.5),
    "TG_PEAK": GatePolicy(peak=0.5),
    "TG_LOOSE": GatePolicy(0.25, 0.25, 0.25),
    "TG_MEDIUM": GatePolicy(0.5, 0.5, 0.5),
    "TG_STRICT": GatePolicy(0.75, 0.75, 0.75),
    "TG_CAPPED": GatePolicy(0.5, 0.5, 0.5, 3),
}


def opportunity(problem: ProblemInstance, candidate: Candidate, operator: str) -> float:
    """Return a nonnegative dimensionless optimistic opportunity score.

    SHIFT: maximum movable Prefill slack / its duration, not a bill bound.
    A7: upper bound on reducible active-union time / current total union.
    A8: upper bound on reducible total demand bill / current total bill,
    assuming the existing A8 no-extension-of-horizon rule and nonnegative tariffs.
    """
    if operator not in ("SHIFT", "A7", "A8"):
        raise ValueError("Unknown terminal operator")
    starts, ms = candidate.schedule.starts, candidate.genotype.ms
    if operator == "SHIFT":
        return max(
            (
                max(
                    0.0,
                    legal_bounds(problem, candidate.genotype, candidate.schedule, o)[1]
                    - starts[o]
                    - TOL.time,
                )
                / max(problem.profile(o).duration, TOL.time)
                for o in range(0, problem.operation_count, 2)
            ),
            default=0.0,
        )
    groups: list[list[int]] = [[] for _ in problem.instances]
    for o, m in enumerate(ms):
        groups[m].append(o)
    lower, active = [], []
    for m, ops in enumerate(groups):
        profiles = [problem.profile(o) for o in ops]
        lower.append(
            max(
                max((p.duration for p in profiles), default=0.0),
                sum(p.duration * p.compute for p in profiles),
                sum(p.duration * p.vram for p in profiles) / max(problem.instances[m].vram, 1e-12),
            )
        )
        active.append(
            union_length([(starts[o], starts[o] + problem.profile(o).duration) for o in ops])
        )
    if operator == "A7":
        return sum(max(0.0, u - lo - TOL.time) for u, lo in zip(active, lower)) / max(
            sum(active), TOL.time
        )
    horizon = candidate.schedule.horizon(problem)
    windows = max(1, ceil(horizon / 900.0))
    # Any feasible batch must finish each Job after release + P + KV + D.
    horizon_lower = max(
        (
            j.release
            + j.prefill.duration
            + j.decode.duration
            + j.kv_delay * (ms[2 * i] != ms[2 * i + 1])
            for i, j in enumerate(problem.jobs)
        ),
        default=0.0,
    )
    area = [0.0 for _ in problem.regions]  # kW seconds
    for m, server in enumerate(problem.instances):
        area[server.region] += server.idle_kw * horizon_lower + problem.floor_kw(m) * lower[m]
        area[server.region] += sum(
            problem.load_kw(o, m) * problem.profile(o).duration for o in groups[m]
        )
    saving = sum(
        region.demand_rate
        * max(0.0, max(candidate.evaluation.windows[r], default=0.0) - area[r] / (900.0 * windows))
        for r, region in enumerate(problem.regions)
    )
    return max(0.0, saving - TOL.cost) / max(candidate.evaluation.bill, TOL.cost)


class TerminalGate:
    """Run-local deterministic admission; no random draws and no archive mutation."""

    def __init__(self, policy: str) -> None:
        self.policy = POLICIES[policy]
        self.counts: dict[str, int] = {}
        self.visits: dict[tuple, int] = {}
        self.scores: dict[tuple, float] = {}

    def admit(
        self,
        problem: ProblemInstance,
        candidate: Candidate,
        operator: str,
        peers: list[Candidate] | None = None,
    ) -> bool:
        """Admit positive scores at/above the archive quantile, preserving score ties."""
        threshold = {"SHIFT": self.policy.shift, "A7": self.policy.pack, "A8": self.policy.peak}[
            operator
        ]
        key = (candidate.genotype, operator)
        if threshold is None:
            reason, accepted = "unchanged", True
        else:

            def score_of(c: Candidate) -> float:
                physical = (c.genotype.ms, c.schedule.starts, operator)
                if physical not in self.scores:
                    self.scores[physical] = opportunity(problem, c, operator)
                return self.scores[physical]

            score = score_of(candidate)
            cutoff = 0.0
            if threshold > 0:
                # Match the terminal sweep: one representative per objective vector.
                representatives = {
                    c.evaluation.objectives: c for c in reversed(peers or [candidate])
                }
                values = sorted(score_of(c) for c in representatives.values())
                cutoff = values[min(len(values) - 1, int(threshold * len(values)))]
            accepted = score > 0 and score >= cutoff
            reason = "admitted" if accepted else "bound_skip"
            if (
                accepted
                and self.policy.cap is not None
                and self.visits.get(key, 0) >= self.policy.cap
            ):
                reason, accepted = "cap_skip", False
        self.counts[operator + ":" + reason] = self.counts.get(operator + ":" + reason, 0) + 1
        if accepted:
            self.visits[key] = self.visits.get(key, 0) + 1
        return accepted
