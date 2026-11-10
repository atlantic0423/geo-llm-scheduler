"""F3: decomposition-guided constructive variation (research arm, default off).

With probability ``p`` an offspring slot skips crossover-plus-mutation and instead
rebuilds the subproblem's own incumbent with one targeted destroy-repair pass
(:class:`~geo_llm_scheduler.lsv2.destroy_repair.TargetedDestroyRepair`, one
proposal). Which Jobs are torn down and how the rebuild is scored follow the
subproblem's weight direction:

* Flow third (high Flow weight): ``FLOW`` (longest-waiting Jobs);
* middle third: ``MIXED`` (rank mean of the FLOW, TOU and PEAK scores);
* electricity third (low Flow weight): ``TOU`` or ``PEAK``, each with probability 1/2;
  the rebuild Tchebycheff uses the subproblem's own ``lambda``.

``direction_aware=False`` is the matched control: same probability, destruction
size and rebuild machinery, but the mode is uniform over those four modes and the
rebuild weight is fixed at ``(0.5, 0.5)``. ``neighbor_paths=True`` additionally
offers each torn-down Job the (Prefill, Decode) paths that the incumbents of the
subproblem's neighbours use for it (at most ``neighbor_extra`` new ones).

The child is committed as a genotype and goes through the normal gateway: SSGS
decode, exact evaluation, Archive, trigger and neighbour replacement are
unchanged. It keeps origin ``"structural"``, so evolution/LS accounting (the N7
share controller) books it as evolution. A slot whose rebuild fails (no feasible
reinsertion, or no proposal different from the incumbent) falls back to the
unchanged variation path and is counted. Routing uses the new stream ``F3:route``
and the rebuild uses ``F3:destroy``; both exist only when the arm is on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.lsv2.destroy_repair import TargetedDestroyRepair
from geo_llm_scheduler.moead.core import maximum
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, preference
from geo_llm_scheduler.utils.numeric import EPS_NORM
from geo_llm_scheduler.utils.rng import RNGManager

GUIDED_MODES = ("FLOW", "MIXED", "TOU", "PEAK")
DIRECTION_MODES: tuple[tuple[str, ...], ...] = (("FLOW",), ("MIXED",), ("TOU", "PEAK"))
"""Destroy modes per Preference class index (0 Flow, 1 Balanced, 2 Electricity)."""
NEUTRAL_WEIGHT = (0.5, 0.5)
REINSERTION_FAILURE = "no feasible reinsertion"


@dataclass(frozen=True)
class GuidedOptions:
    """Declared switches of one guided-variation arm."""

    probability: float = 0.2
    direction_aware: bool = True
    neighbor_paths: bool = False
    neighbor_extra: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Guided probability outside [0,1]")
        if type(self.neighbor_extra) is not int or self.neighbor_extra < 0:
            raise ValueError("Neighbour path allowance must be a non-negative integer")


@dataclass
class Cell:
    """Counters of one (direction class, destroy mode) cell of the guided path."""

    attempts: int = 0
    successes: int = 0
    fallback_no_reinsertion: int = 0
    fallback_no_proposal: int = 0
    seconds: float = 0.0  # propose + exact evaluation of successful slots
    fallback_seconds: float = 0.0  # spent on slots that fell back
    inserted: int = 0  # Archive insertions of the exact-evaluated children
    inserted_children: int = 0  # children with at least one insertion
    hv_gain: float = 0.0  # frozen-frame HV gain of those insertions
    feasible: int = 0
    gap_count: int = 0  # feasible children with a timetable record
    bill_gap: float = 0.0  # sum of (decoded exact Bill - designed timetable Bill)
    bill_gap_abs: float = 0.0
    bill_gap_rel: float = 0.0  # sum of gap / designed Bill
    flow_gap: float = 0.0  # sum of (decoded exact Flow - designed timetable Flow)
    extra_path_used: int = 0  # children whose MS uses a neighbour-offered path


@dataclass
class GuidedVariation:
    """Routes offspring slots to a direction-guided destroy-repair child."""

    config: Config
    options: GuidedOptions
    gain_probe: Callable[[], float] | None = None
    slots: int = 0
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)

    # ---------------- policy ----------------
    def mode_for(self, direction: int, rng: Any) -> str:
        """Destroy mode of one attempt; the matched control ignores the direction."""
        modes = DIRECTION_MODES[direction] if self.options.direction_aware else GUIDED_MODES
        return modes[0] if len(modes) == 1 else rng.choice(modes)

    @staticmethod
    def neighbor_paths(
        population: Sequence[Candidate], subproblem: int, neighbors: Sequence[int]
    ) -> Callable[[int], list[tuple[int, int]]]:
        """Job -> distinct neighbour paths, most shared first, then nearest first."""
        mss = [population[j].genotype.ms for j in neighbors if j != subproblem]

        def paths_of(job: int) -> list[tuple[int, int]]:
            count: dict[tuple[int, int], int] = {}
            for ms in mss:
                pair = (ms[2 * job], ms[2 * job + 1])
                count[pair] = count.get(pair, 0) + 1
            return sorted(count, key=lambda p: -count[p])

        return paths_of

    def __call__(
        self,
        population: list[Candidate],
        subproblem: int,
        neighbors: tuple[int, ...],
        lambdas: tuple[tuple[float, float], ...],
        gateway: EvaluationGateway,
        streams: RNGManager,
    ) -> Candidate | None:
        """Return the guided child of this slot, or ``None`` to use the usual variation."""
        self.slots += 1
        if streams.stream("F3:route").random() >= self.options.probability:
            return None
        direction = preference(subproblem, len(population))
        rng = streams.stream("F3:destroy")
        mode = self.mode_for(direction, rng)
        cell = self.cells.setdefault((PREFERENCE_LABELS[direction], mode), Cell())
        cell.attempts += 1
        incumbent = population[subproblem]
        ideal, top = gateway.ideal, maximum(population)
        operator = TargetedDestroyRepair(
            mode=mode,
            weight=lambdas[subproblem] if self.options.direction_aware else NEUTRAL_WEIGHT,
            scale=(max(top[0] - ideal[0], EPS_NORM), max(top[1] - ideal[1], EPS_NORM)),
            record_timetable=True,
            extra_paths=(
                self.neighbor_paths(population, subproblem, neighbors)
                if self.options.neighbor_paths
                else None
            ),
            extra_limit=self.options.neighbor_extra,
        )
        started = perf_counter()
        try:
            batch = operator.propose(gateway.problem, incumbent, 1, self.config, rng)
        except ValueError as exc:
            if REINSERTION_FAILURE not in str(exc):
                raise
            cell.fallback_no_reinsertion += 1
            cell.fallback_seconds += perf_counter() - started
            return None
        if not batch.proposals:
            cell.fallback_no_proposal += 1
            cell.fallback_seconds += perf_counter() - started
            return None
        proposal = batch.proposals[0]
        before = gateway.archive.insertions
        child = gateway.evaluate(proposal.genotype)
        cell.seconds += perf_counter() - started
        cell.successes += 1
        added = gateway.archive.insertions - before
        cell.inserted += added
        cell.inserted_children += int(added > 0)
        if added > 0 and self.gain_probe is not None:
            cell.hv_gain += self.gain_probe()
        if self.options.neighbor_paths:
            cell.extra_path_used += int(
                self._uses_offered_path(child, incumbent, population, subproblem, neighbors)
            )
        if child.evaluation.feasible:
            cell.feasible += 1
            timetable = batch.instrumentation["timetable"][0]
            gap = child.evaluation.bill - timetable["bill"]
            cell.gap_count += 1
            cell.bill_gap += gap
            cell.bill_gap_abs += abs(gap)
            cell.bill_gap_rel += gap / max(abs(timetable["bill"]), EPS_NORM)
            cell.flow_gap += child.evaluation.flow - timetable["flow"]
        return child

    @staticmethod
    def _uses_offered_path(
        child: Candidate,
        incumbent: Candidate,
        population: Sequence[Candidate],
        subproblem: int,
        neighbors: Sequence[int],
    ) -> bool:
        """Whether the child moved a Job onto a path a neighbouring incumbent uses for it."""
        a, b = child.genotype.ms, incumbent.genotype.ms
        others = [population[j].genotype.ms for j in neighbors if j != subproblem]
        for job in range(len(a) // 2):
            pair = (a[2 * job], a[2 * job + 1])
            if pair != (b[2 * job], b[2 * job + 1]) and any(
                (ms[2 * job], ms[2 * job + 1]) == pair for ms in others
            ):
                return True
        return False

    # ---------------- record ----------------
    def summary(self) -> dict:
        """JSON-ready counters per cell, per direction class and in total."""
        keys = list(Cell.__dataclass_fields__)
        cells = {f"{d}/{m}": asdict(c) for (d, m), c in sorted(self.cells.items())}
        by_direction: dict[str, dict[str, float]] = {}
        total = dict.fromkeys(keys, 0)
        for (direction, _), cell in self.cells.items():
            row = by_direction.setdefault(direction, dict.fromkeys(keys, 0))
            for key in keys:
                row[key] += getattr(cell, key)
                total[key] += getattr(cell, key)
        attempts = total["attempts"]
        return dict(
            options=asdict(self.options),
            slots=self.slots,
            attempts=attempts,
            successes=total["successes"],
            fallback_no_reinsertion=total["fallback_no_reinsertion"],
            fallback_no_proposal=total["fallback_no_proposal"],
            guided_seconds=total["seconds"] + total["fallback_seconds"],
            exact_per_second=(total["successes"] / total["seconds"] if total["seconds"] else 0.0),
            by_direction=by_direction,
            cells=cells,
        )


__all__ = [
    "DIRECTION_MODES",
    "GUIDED_MODES",
    "NEUTRAL_WEIGHT",
    "Cell",
    "GuidedOptions",
    "GuidedVariation",
]
