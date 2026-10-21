"""Instance-local, arithmetic-preserving indexes for research BILL seeds.

Operation lists retain operation-index order, including the temporary Prefill.
The tariff index skips zero-overlap terms but uses the reference fsum formula;
it deliberately does not subtract antiderivatives or regroup resource sums.
"""

from bisect import bisect_left, bisect_right
from math import fsum

from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.scheduling.resources import ResourceCalendar as ResourceCalendar


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
