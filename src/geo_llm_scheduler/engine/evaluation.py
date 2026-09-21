"""Single evaluation gateway: every feasible exact result reaches the Archive."""

import time
from collections import Counter, defaultdict

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.scheduling.ssgs import decode


class EvaluationGateway:
    """Own run-level exact/rebuild counters, ideal history and archive dispatch."""

    def __init__(self, problem: ProblemInstance, archive: Archive):
        self.problem = problem
        self.archive = archive
        self.ideal = (float("inf"), float("inf"))
        self.counts: Counter[str] = Counter()
        self.seconds: dict[str, float] = defaultdict(float)

    def evaluate(
        self, genotype: Genotype, schedule: Schedule | None = None, origin: str = "structural"
    ) -> Candidate:
        """Rebuild structural proposals; directly evaluate retained timing proposals."""
        if schedule is None:
            start = time.perf_counter()
            schedule = decode(self.problem, genotype)
            duration = time.perf_counter() - start
            self.seconds["ssgs"] += duration
            self.seconds["ssgs:" + origin] += duration
            self.counts["ssgs"] += 1
        start = time.perf_counter()
        result = evaluate(self.problem, genotype, schedule)
        duration = time.perf_counter() - start
        self.seconds["exact"] += duration
        self.seconds["exact:" + origin] += duration
        self.counts["exact"] += 1
        self.counts["exact:" + origin] += 1
        candidate = Candidate(genotype, schedule, result, origin)
        if result.feasible:
            self.counts["feasible"] += 1
            self.counts["feasible:" + origin] += 1
            self.ideal = (min(self.ideal[0], result.flow), min(self.ideal[1], result.bill))
            self.counts["archive_attempts:" + origin] += 1
            insertions = self.archive.insertions
            start = time.perf_counter()
            self.archive.consider(candidate)
            duration = time.perf_counter() - start
            self.seconds["archive"] += duration
            self.seconds["archive:" + origin] += duration
            self.counts["archive_insertions:" + origin] += self.archive.insertions - insertions
        return candidate
