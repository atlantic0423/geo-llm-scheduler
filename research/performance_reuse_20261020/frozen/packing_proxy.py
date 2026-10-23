"""A7 local active-area cost proxy, without evaluating complete raw candidates."""

from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.evaluation.exact import tied_peaks
from geo_llm_scheduler.scheduling.resources import intervals


class PackingBillProxy:
    """Cache frozen other-operation intervals and incumbent peak-window ties.

    The score integrates the selected instance's active-indicator difference.
    Demand uses the mean first-order change at the old tied peaks. New peak
    takeover and global horizon/idle-tail changes deliberately remain unmodeled;
    the score is neither an exact bill nor a candidate acceptance criterion.
    """

    def __init__(self, problem: ProblemInstance, incumbent: Candidate) -> None:
        self.problem = problem
        self.incumbent = incumbent
        self.peaks = tuple(tied_peaks(w) for w in incumbent.evaluation.windows)
        self.others: dict[int, list[tuple[int, float, float]]] = {}

    def score(self, operation: int, start: float) -> float:
        """Return predicted local bill delta in CNY; smaller is preferred.

        Only immutable incumbent data is read. No Schedule/Candidate is built,
        no exact gateway is called, and no extra RNG or future ideal is used.
        """
        problem, incumbent = self.problem, self.incumbent
        machine = incumbent.genotype.ms[operation]
        server = problem.instances[machine]
        region = problem.regions[server.region]
        power = server.active_kw - server.idle_kw
        old = incumbent.schedule.starts[operation]
        duration = problem.profile(operation).duration
        if operation not in self.others:
            self.others[operation] = intervals(
                problem, incumbent.genotype, incumbent.schedule, machine, frozenset({operation})
            )
        others = self.others[operation]
        points = sorted(
            {old, old + duration, start, start + duration}
            | {t for _, a, b in others for t in (a, b)}
        )
        peaks = self.peaks[server.region]
        cost = 0.0
        for a, b in zip(points, points[1:]):
            if any(s <= a < e for _, s, e in others):
                continue
            change = int(start <= a < start + duration) - int(old <= a < old + duration)
            if not change:
                continue
            tariff_area = sum(
                max(0.0, min(b, tariff.end) - max(a, tariff.start)) * tariff.price
                for tariff in region.tariffs
            )
            cost += power * change * tariff_area / 3600
            if peaks:
                peak_area = sum(max(0.0, min(b, (w + 1) * 900) - max(a, w * 900)) for w in peaks)
                cost += region.demand_rate * power * change * peak_area / (900 * len(peaks))
        return cost
