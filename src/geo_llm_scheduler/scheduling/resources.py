"""Exact half-open resource intervals shared by decoding and timing moves."""

from bisect import bisect_left, bisect_right, insort
from itertools import chain

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.utils.numeric import TOL, less


class ResourceView:
    """Invocation-local index of one immutable MS/timing pair, in operation order.

    No state is attached to the problem or Schedule. A different genotype or
    timing requires a new view, even when an operation remains on its server.
    """

    def __init__(self, problem: ProblemInstance, genotype: Genotype, schedule: Schedule) -> None:
        self.problem, self.genotype, self.schedule = problem, genotype, schedule
        items: list[list[tuple[int, float, float]]] = [[] for _ in problem.instances]
        for o, start in enumerate(schedule.starts):
            if start >= 0:
                items[genotype.ms[o]].append((o, start, start + problem.profile(o).duration))
        self.items = tuple(tuple(bucket) for bucket in items)
        self._events: dict[int, tuple[float, ...]] = {}
        self._activity: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}

    def events(self, machine: int) -> tuple[float, ...]:
        """Reuse every original boundary, including boundaries hidden by overlap."""
        if machine not in self._events:
            self._events[machine] = tuple(
                sorted({t for _, a, b in self.items[machine] for t in (a, b)})
            )
        return self._events[machine]

    def activity_bounds(self, machine: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Index half-open activity membership without merging integral segments."""
        if machine not in self._activity:
            items = self.items[machine]
            self._activity[machine] = (
                tuple(sorted(a for _, a, b in items if a < b)),
                tuple(sorted(b for _, a, b in items if a < b)),
            )
        return self._activity[machine]

    def intervals(
        self, machine: int, excluded: frozenset[int] = frozenset()
    ) -> list[tuple[int, float, float]]:
        """Return a fresh list with the reference order and exclusions."""
        return [item for item in self.items[machine] if item[0] not in excluded]

    def fits(self, operation: int, start: float, excluded: frozenset[int] = frozenset()) -> bool:
        """Check frozen competitors without regrouping floating-point sums."""
        machine = self.genotype.ms[operation]
        others = self.intervals(machine, excluded | {operation})
        return _fits_intervals(self.problem, machine, operation, start, others)

    def self_fits(self) -> dict[int, bool]:
        """``fits(o, start_o)`` for every scheduled operation, one sweep per machine.

        Within an operation's own window the breakpoints of the others are the
        machine's elementary boundaries, and the operations running at each are
        the machine's running set without ``o``. Each check sums that set in
        ``items`` order, exactly as the reference per-operation scan does.
        """
        problem = self.problem
        out: dict[int, bool] = {}
        for machine, items in enumerate(self.items):
            if not items:
                continue
            vram = problem.instances[machine].vram
            profiles = {o: problem.profile(o) for o, _, _ in items}
            begins: dict[float, list[int]] = {}
            finish: dict[float, list[int]] = {}
            for o, s, e in items:
                if s < e:
                    begins.setdefault(s, []).append(o)
                    finish.setdefault(e, []).append(o)
            bounds = sorted({t for _, a, b in items for t in (a, b)})
            # Running set at each boundary (s <= at < e), listed in ``items``
            # (operation) order like the reference scan.
            running = []
            now: set[int] = set()
            for at in bounds:
                now.difference_update(finish.get(at, ()))
                now.update(begins.get(at, ()))
                running.append([(o, profiles[o]) for o in sorted(now)])
            for o, s, e in items:
                p = profiles[o]
                ok = True
                if s < e:
                    for k in range(bisect_left(bounds, s), bisect_left(bounds, e)):
                        active = [q for x, q in running[k] if x != o]
                        if less(
                            1.0, p.compute + sum(q.compute for q in active), TOL.compute
                        ) or less(vram, p.vram + sum(q.vram for q in active), TOL.vram):
                            ok = False
                            break
                out[o] = ok
        return out

    def earliest(self, operation: int, est: float) -> float:
        """Search all original completion events against the same frozen profile."""
        machine = self.genotype.ms[operation]
        return _earliest_intervals(
            self.problem, machine, operation, est, self.intervals(machine, frozenset({operation}))
        )

    def check_binding(
        self, problem: ProblemInstance, genotype: Genotype, schedule: Schedule
    ) -> None:
        """Reject accidental reuse across any changed immutable input."""
        if (
            self.problem is not problem
            or self.genotype is not genotype
            or self.schedule is not schedule
        ):
            raise ValueError("ResourceView belongs to a different MS/timing pair")


def intervals(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    machine: int,
    excluded: frozenset[int] = frozenset(),
) -> list[tuple[int, float, float]]:
    """Read assignment from MS and return scheduled intervals on one server."""
    return [
        (o, s, s + problem.profile(o).duration)
        for o, s in enumerate(schedule.starts)
        if o not in excluded and genotype.ms[o] == machine and s >= 0
    ]


def fits(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    operation: int,
    start: float,
    excluded: frozenset[int] = frozenset(),
) -> bool:
    """Check a proposed interval against frozen operations, excluding itself."""
    machine = genotype.ms[operation]
    others = intervals(problem, genotype, schedule, machine, excluded | {operation})
    return _fits_intervals(problem, machine, operation, start, others)


def _fits_intervals(
    problem: ProblemInstance,
    machine: int,
    operation: int,
    start: float,
    others: list[tuple[int, float, float]],
) -> bool:
    """Check a frozen interval list in the reference operation summation order.

    Only intervals overlapping ``[start, end)`` can run at, or put a breakpoint
    inside, the checked window, so the others are dropped first. The kept list
    preserves the given order, so every capacity sum adds the same terms in the
    same order as a scan of the full list.
    """
    p = problem.profile(operation)
    end = start + p.duration
    near = [(s, e, problem.profile(o)) for o, s, e in others if e > start and s < end]
    points = sorted({start, end} | {t for s, e, _ in near for t in (s, e) if start < t < end})
    vram = problem.instances[machine].vram
    for a, b in zip(points, points[1:]):
        if b <= a:
            continue
        active = [q for s, e, q in near if s <= a < e]
        if less(1.0, p.compute + sum(q.compute for q in active), TOL.compute):
            return False
        if less(vram, p.vram + sum(q.vram for q in active), TOL.vram):
            return False
    return True


def earliest(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule, operation: int, est: float
) -> float:
    """Find earliest feasible interval using completion events, never time steps."""
    machine = genotype.ms[operation]
    others = intervals(problem, genotype, schedule, machine, frozenset({operation}))
    return _earliest_intervals(problem, machine, operation, est, others)


def _earliest_intervals(
    problem: ProblemInstance,
    machine: int,
    operation: int,
    est: float,
    others: list[tuple[int, float, float]],
) -> float:
    """Reuse the same frozen intervals across the unchanged completion events."""
    events = {est} | {b for _, _, b in others if b >= est}
    for t in sorted(events):
        if _fits_intervals(problem, machine, operation, t, others):
            return t
    raise ValueError("Operation cannot fit on assigned instance")


_EMPTY_SUMS = (sum(()), sum(()))  # the reference sums over no running operation


class ResourceCalendar:
    """Index scheduled prefix operations by machine in operation-index order.

    Besides ``items`` (read directly by some callers; never mutate it, use
    :meth:`commit`), each machine keeps its sorted distinct interval boundaries
    and, for the elementary segment starting at each boundary, the Compute and
    VRAM sums of the operations running there. Every sum is a fresh ``sum`` over
    the running operations in ``items`` order, exactly the expression the
    reference capacity check evaluates, so indexed checks are bit-identical.
    """

    def __init__(self, problem: ProblemInstance) -> None:
        self.problem = problem
        self.items: list[list[tuple[int, float, float]]] = [[] for _ in problem.instances]
        self._bounds: list[list[float]] = [[] for _ in problem.instances]
        self._sums: list[list[tuple[float, float]]] = [[] for _ in problem.instances]

    def commit(self, machine: int, operation: int, start: float) -> None:
        """Insert one newly scheduled operation without regrouping resource sums."""
        item = (operation, start, start + self.problem.profile(operation).duration)
        insort(self.items[machine], item)
        bounds, sums = self._bounds[machine], self._sums[machine]
        for t in (item[1], item[2]):
            i = bisect_left(bounds, t)
            if i == len(bounds) or bounds[i] != t:
                # A new boundary splits a segment; both parts keep its running set
                # until the segments covered by the new operation are recomputed.
                bounds.insert(i, t)
                sums.insert(i, sums[i - 1] if i else _EMPTY_SUMS)
        for j in range(bisect_left(bounds, item[1]), bisect_left(bounds, item[2])):
            sums[j] = self._segment_sums(machine, bounds[j])

    def _segment_sums(self, machine: int, at: float) -> tuple[float, float]:
        """Reference-order Compute and VRAM sums of the operations running at ``at``."""
        profile = self.problem.profile
        active = [profile(o) for o, s, e in self.items[machine] if s <= at < e]
        return sum(q.compute for q in active), sum(q.vram for q in active)

    def _indexed_fits(self, machine: int, operation: int, start: float) -> bool:
        """The reference check of ``operation`` at ``start`` against all committed items."""
        p = self.problem.profile(operation)
        end = start + p.duration
        if not start < end:
            return True
        bounds, sums = self._bounds[machine], self._sums[machine]
        vram = self.problem.instances[machine].vram
        first = bisect_right(bounds, start)
        for k in chain((first - 1,), range(first, bisect_left(bounds, end))):
            compute, memory = sums[k] if k >= 0 else _EMPTY_SUMS
            if less(1.0, p.compute + compute, TOL.compute):
                return False
            if less(vram, p.vram + memory, TOL.vram):
                return False
        return True

    def _uncommitted(self, machine: int, operation: int) -> bool:
        return all(item[0] != operation for item in self.items[machine])

    def fits(self, machine: int, operation: int, start: float) -> bool:
        """Reference ``_fits_intervals`` against the machine's items other than ``operation``."""
        if self._uncommitted(machine, operation):
            return self._indexed_fits(machine, operation, start)
        others = [item for item in self.items[machine] if item[0] != operation]
        return _fits_intervals(self.problem, machine, operation, start, others)

    def earliest(
        self,
        machine: int,
        operation: int,
        est: float,
        temporary: tuple[int, float, float] | None = None,
    ) -> float:
        """Return the reference event, including an optional uncommitted Prefill."""
        if temporary is None and self._uncommitted(machine, operation):
            items = self.items[machine]
            for t in sorted({est} | {b for _, _, b in items if b >= est}):
                if self._indexed_fits(machine, operation, t):
                    return t
            raise ValueError("Operation cannot fit on assigned instance")
        others = [item for item in self.items[machine] if item[0] != operation]
        if temporary is not None and temporary[0] != operation:
            insort(others, temporary)
        return _earliest_intervals(self.problem, machine, operation, est, others)


def calendar_fits(
    calendar: ResourceCalendar,
    machine: int,
    operation: int,
    start: float,
    others: list[tuple[int, float, float]],
) -> bool:
    """``_fits_intervals`` for a calendar's machine, through its index when possible.

    ``others`` must be the machine's calendar items other than ``operation`` (the
    list the reference check scans); the indexed query reads the same items.
    """
    return calendar.fits(machine, operation, start)


def est_for(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule, operation: int
) -> float:
    """Release or preceding Prefill completion plus assignment-dependent KV."""
    job = problem.jobs[operation // 2]
    if operation % 2 == 0:
        return job.release
    return schedule.end(problem, operation - 1) + job.kv_delay * (
        genotype.ms[operation] != genotype.ms[operation - 1]
    )


def union_length(items: list[tuple[float, float]]) -> float:
    """Compute union length without double-counting concurrent operations."""
    total = 0.0
    right = float("-inf")
    for a, b in sorted(items):
        total += max(0.0, b - max(a, right))
        right = max(right, b)
    return total
