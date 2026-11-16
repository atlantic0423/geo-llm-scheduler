"""Instance-local, arithmetic-preserving indexes for research BILL seeds.

Operation lists retain operation-index order, including the temporary Prefill.
The tariff index skips zero-overlap terms but uses the reference fsum formula;
it deliberately does not subtract antiderivatives or regroup resource sums.
"""

from bisect import bisect_left, bisect_right, insort
from math import fsum

from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.utils.numeric import TOL, less


class ResourceCalendar:
    """Maintain committed prefix operations separately for each machine."""

    def __init__(self, problem: ProblemInstance) -> None:
        self.problem = problem
        self.items: list[list[tuple[int, float, float]]] = [[] for _ in problem.instances]

    def commit(self, machine: int, operation: int, start: float) -> None:
        """Insert a newly scheduled operation in reference operation-index order."""
        insort(
            self.items[machine],
            (operation, start, start + self.problem.profile(operation).duration),
        )

    def earliest(
        self,
        machine: int,
        operation: int,
        est: float,
        temporary: tuple[int, float, float] | None = None,
    ) -> float:
        """Find the reference earliest feasible event with an optional temporary P."""
        others = [x for x in self.items[machine] if x[0] != operation]
        if temporary is not None and temporary[0] != operation:
            insort(others, temporary)
        profile = self.problem.profile(operation)
        events = sorted({est} | {b for _, _, b in others if b >= est})
        for start in events:
            end = start + profile.duration
            points = sorted(
                {start, end} | {t for _, a, b in others for t in (a, b) if start < t < end}
            )
            valid = True
            for a, b in zip(points, points[1:]):
                if b <= a:
                    continue
                active = [self.problem.profile(o) for o, s, e in others if s <= a < e]
                if less(1.0, profile.compute + sum(q.compute for q in active), TOL.compute) or less(
                    self.problem.instances[machine].vram,
                    profile.vram + sum(q.vram for q in active),
                    TOL.vram,
                ):
                    valid = False
                    break
            if valid:
                return start
        raise ValueError("Operation cannot fit on assigned instance")


class TariffIndex:
    """Cache exact interval integrals within one construction, never globally."""

    def __init__(self, problem: ProblemInstance) -> None:
        self.problem = problem
        self.starts = [tuple(t.start for t in r.tariffs) for r in problem.regions]
        self.ends = [tuple(t.end for t in r.tariffs) for r in problem.regions]
        self.cache: dict[tuple[int, float, float], float] = {}
        self.idle = [
            sum(s.idle_kw for s in problem.instances if s.region == r)
            for r in range(len(problem.regions))
        ]

    def integral(self, problem: ProblemInstance, region: int, a: float, b: float) -> float:
        """Return the same finite nonnegative fsum as the reference tariff scan."""
        if problem is not self.problem:
            raise ValueError("Tariff index belongs to another immutable input")
        tariffs = problem.regions[region].tariffs
        if not 0 <= a <= b <= tariffs[-1].end:
            raise ValueError("Prefix time is outside declared tariffs")
        key = (region, a, b)
        if key not in self.cache:
            left = bisect_right(self.ends[region], a)
            right = bisect_left(self.starts[region], b)
            self.cache[key] = fsum(
                max(0.0, min(b, t.end) - max(a, t.start)) * t.price for t in tariffs[left:right]
            )
        return self.cache[key]
