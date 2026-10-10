"""Bit-identical fixed-horizon W scores from an insertion-local event sweep."""

from bisect import bisect_right

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import regional_windows


class InsertionWindows:
    """Cache a partial schedule once, retaining every original integration breakpoint.

    Flags use integer activity counts, so overlapping operations form a union.
    Each trial retains the reference machine summation and segment/window order;
    no subtraction of previously rounded window areas changes score ties.
    The object owns no RNG and never survives an insertion step.
    """

    def __init__(
        self,
        problem: ProblemInstance,
        genotype: Genotype,
        schedule: Schedule,
        region: int,
        horizon: float,
    ) -> None:
        """Sweep present operations, including idle instances, to a fixed horizon."""
        self.problem, self.genotype, self.horizon = problem, genotype, horizon
        self.machines = tuple(m for m, s in enumerate(problem.instances) if s.region == region)
        self.machine_index = {m: i for i, m in enumerate(self.machines)}
        events: dict[float, list[tuple[int, int]]] = {0.0: [], horizon: []}
        counts = [0] * len(self.machines)
        for o, a in enumerate(schedule.starts):
            if a < 0 or genotype.ms[o] not in self.machine_index:
                continue
            i = self.machine_index[genotype.ms[o]]
            b = a + problem.profile(o).duration
            if a <= 0 < b:
                counts[i] += 1
            if 0 < a <= horizon:
                events.setdefault(a, []).append((i, 1))
            if 0 < b <= horizon:
                events.setdefault(b, []).append((i, -1))
        self.points = tuple(sorted(events))
        self.values: list[tuple[float, ...]] = []
        for point in self.points:
            for i, change in events[point]:
                counts[i] += change
            self.values.append(
                tuple(
                    problem.instances[m].active_kw if counts[i] else problem.instances[m].idle_kw
                    for i, m in enumerate(self.machines)
                )
            )
        self.powers = tuple(sum(v) for v in self.values)

    def windows(self, operation: int, start: float) -> tuple[float, ...]:
        """Score one absent operation inserted at start, bit-identically to reference."""
        if start < 0:
            return regional_windows(
                tuple((a, b, p) for a, b, p in zip(self.points, self.points[1:], self.powers)),
                self.horizon,
            )
        m = self.genotype.ms[operation]
        i = self.machine_index.get(m)
        end = start + self.problem.profile(operation).duration
        points = (
            sorted(set(self.points) | {t for t in (start, end) if 0 <= t <= self.horizon})
            if i is not None
            else self.points
        )
        segments = []
        for a, b in zip(points, points[1:]):
            old = bisect_right(self.points, a) - 1
            power = self.powers[old]
            if i is not None and start <= a < end:
                values = self.values[old]
                if values[i] != self.problem.instances[m].active_kw:
                    power = sum(
                        self.problem.instances[m].active_kw if j == i else v
                        for j, v in enumerate(values)
                    )
            segments.append((a, b, power))
        return regional_windows(tuple(segments), self.horizon)
