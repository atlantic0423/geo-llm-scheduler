"""Single evaluation gateway: every feasible exact result reaches the Archive."""

import time
from collections import Counter, OrderedDict, defaultdict
from typing import Callable

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import (
    Candidate,
    EvaluationResult,
    Genotype,
    ProblemInstance,
    Schedule,
)
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.evaluation.prefill import PrefillEvaluationContext
from geo_llm_scheduler.scheduling.ssgs import decode


class EvaluationCapReached(RuntimeError):
    """The run has consumed its frozen exact-evaluation allowance."""


class EvaluationGateway:
    """Own run-level exact/rebuild counters, ideal history and archive dispatch.

    ``decode_cache_size`` > 0 keeps a bounded LRU map from a structural genotype to
    its SSGS schedule and exact result. Both are deterministic functions of the
    immutable problem and genotype, so a hit returns the identical candidate
    content. Logical counters (``ssgs``, ``exact``, feasible, archive attempts) and
    archive dispatch are unchanged; hits are counted separately as
    ``ssgs_cache_hits`` and their lookup time as ``decode_cache`` seconds.
    ``audit_decode_cache`` recomputes every hit and asserts equality (tests only).
    """

    def __init__(
        self,
        problem: ProblemInstance,
        archive: Archive,
        exact_cap: int | None = None,
        decode_cache_size: int = 0,
    ):
        if type(decode_cache_size) is not int or decode_cache_size < 0:
            raise ValueError("Decode cache size must be a non-negative integer")
        self.problem = problem
        self.archive = archive
        self.ideal = (float("inf"), float("inf"))
        self.counts: Counter[str] = Counter()
        self.seconds: dict[str, float] = defaultdict(float)
        self.exact_cap = exact_cap
        self.candidate_sink: Callable[[Candidate], None] | None = None
        self.decode_cache_size = decode_cache_size
        self.audit_decode_cache = False
        self._decoded: OrderedDict[Genotype, tuple[Schedule, EvaluationResult]] = OrderedDict()

    def _cached(self, genotype: Genotype) -> tuple[Schedule, EvaluationResult] | None:
        """Return and refresh a structural cache entry, auditing it when requested."""
        if not self.decode_cache_size:
            return None
        start = time.perf_counter()
        hit = self._decoded.get(genotype)
        if hit is not None:
            self._decoded.move_to_end(genotype)
        self.seconds["decode_cache"] += time.perf_counter() - start
        if hit is not None and self.audit_decode_cache:
            schedule = decode(self.problem, genotype)
            assert schedule == hit[0] and evaluate(self.problem, genotype, schedule) == hit[1]
        return hit

    def _remember(self, genotype: Genotype, schedule: Schedule, result: EvaluationResult) -> None:
        """Insert a fresh structural result, evicting the least recently used entry."""
        if self.decode_cache_size:
            self._decoded[genotype] = (schedule, result)
            if len(self._decoded) > self.decode_cache_size:
                self._decoded.popitem(last=False)

    def evaluate(
        self,
        genotype: Genotype,
        schedule: Schedule | None = None,
        origin: str = "structural",
        *,
        prefill_context: PrefillEvaluationContext | None = None,
    ) -> Candidate:
        """Rebuild structural proposals; directly evaluate retained timing proposals."""
        if self.exact_cap is not None and self.counts["exact"] >= self.exact_cap:
            raise EvaluationCapReached("Exact evaluation cap reached")
        if prefill_context is not None and (
            prefill_context.problem is not self.problem or schedule is None
        ):
            raise ValueError("Prefill exact reuse requires this gateway's input and timing")
        structural = schedule is None
        hit = self._cached(genotype) if structural else None
        if hit is not None:
            schedule, result = hit
            self.counts["ssgs"] += 1
            self.counts["ssgs_cache_hits"] += 1
        else:
            if schedule is None:
                start = time.perf_counter()
                schedule = decode(self.problem, genotype)
                duration = time.perf_counter() - start
                self.seconds["ssgs"] += duration
                self.seconds["ssgs:" + origin] += duration
                self.counts["ssgs"] += 1
            start = time.perf_counter()
            result = (
                evaluate(self.problem, genotype, schedule)
                if prefill_context is None
                else prefill_context.evaluate(genotype, schedule)
            )
            duration = time.perf_counter() - start
            self.seconds["exact"] += duration
            self.seconds["exact:" + origin] += duration
            if structural:
                self._remember(genotype, schedule, result)
        self.counts["exact"] += 1
        self.counts["exact:" + origin] += 1
        if prefill_context is not None:
            self.counts["prefill_exact_reuse"] += 1
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
        if self.candidate_sink is not None:
            self.candidate_sink(candidate)
        return candidate
