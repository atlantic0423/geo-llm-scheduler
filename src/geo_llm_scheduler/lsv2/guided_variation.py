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

Cost-control extensions (2026-11-11, all default off, defaults keep every draw and
record unchanged): ``destroy_count`` fixes the number of destroyed Jobs, and
``controller="switch"`` replaces the fixed probability by one that follows the
decayed HV gain per second of the guided path against the ordinary path (the
N7 form, ``p = clip(p_on (rho_g / rho_o)^gamma, p_probe, p_on)``). Path timing uses a
private clock so the engine's and the run's clocks are never read an extra time.
"""

from __future__ import annotations

import math
import time as _time
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
DEFAULT_CLOCK = _time.perf_counter  # private reference; unaffected by later patching of ``time``
BASE_OPTION_FIELDS = ("probability", "direction_aware", "neighbor_paths", "neighbor_extra")
EXTRA_OPTION_DEFAULTS: dict[str, Any] = dict(
    destroy_count=None,
    controller="fixed",
    p_probe=0.02,
    tau_seconds=300.0,
    gamma=2.0,
    warmup_seconds=5.0,
    trace_seconds=30.0,
    record_paths=False,
)


@dataclass(frozen=True)
class GuidedOptions:
    """Declared switches of one guided-variation arm.

    ``probability`` is the routing probability (``p_on`` under the switch controller).
    The cost-control fields default to the behaviour of the first F3 batch.
    """

    probability: float = 0.2
    direction_aware: bool = True
    neighbor_paths: bool = False
    neighbor_extra: int = 2
    destroy_count: int | None = None  # fixed number of destroyed Jobs (None: 8%, at least 2)
    controller: str = "fixed"  # "fixed" or "switch"
    p_probe: float = 0.02  # switch controller: lower bound and warm-up probability
    tau_seconds: float = 300.0  # switch controller: decay time constant
    gamma: float = 2.0  # switch controller: exponent of the rate ratio
    warmup_seconds: float = 5.0  # switch controller: decayed seconds each path needs
    trace_seconds: float = 30.0  # switch controller: record one point per bin
    record_paths: bool = False  # time/count the ordinary path and bin both paths by decile

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Guided probability outside [0,1]")
        if type(self.neighbor_extra) is not int or self.neighbor_extra < 0:
            raise ValueError("Neighbour path allowance must be a non-negative integer")
        if self.destroy_count is not None and (
            type(self.destroy_count) is not int or self.destroy_count < 1
        ):
            raise ValueError("Destroy count must be a positive integer")
        if self.controller not in ("fixed", "switch"):
            raise ValueError("Unknown guided controller")
        if self.controller == "switch" and not 0.0 <= self.p_probe <= self.probability:
            raise ValueError("Probe probability must lie in [0, probability]")
        if self.tau_seconds <= 0 or self.gamma <= 0 or self.warmup_seconds < 0:
            raise ValueError("Controller decay, exponent and warm-up must be positive")
        if self.trace_seconds <= 0:
            raise ValueError("Trace bins must be positive")

    @property
    def timed(self) -> bool:
        """Whether the ordinary path is observed (needed by the controller and the records)."""
        return self.controller == "switch" or self.record_paths

    def record(self) -> dict[str, Any]:
        """Configuration record: the four base fields, plus all cost fields only if any is set."""
        values = asdict(self)
        out = {k: values[k] for k in BASE_OPTION_FIELDS}
        if any(values[k] != v for k, v in EXTRA_OPTION_DEFAULTS.items()):
            out.update({k: values[k] for k in EXTRA_OPTION_DEFAULTS})
        return out


@dataclass
class PathRates:
    """Decayed HV gain and seconds of the guided (``g``) and ordinary (``o``) paths.

    Both paths decay by ``exp(-dt / tau)`` on one shared clock, as in the N7 share rule,
    so their rate ratio is not changed by decay alone.
    """

    tau: float
    gain: dict[str, float] = field(default_factory=lambda: dict(g=0.0, o=0.0))
    seconds: dict[str, float] = field(default_factory=lambda: dict(g=0.0, o=0.0))
    stamp: float = 0.0

    def _factor(self, now: float) -> float:
        return math.exp(-max(0.0, now - self.stamp) / self.tau)

    def observe(self, path: str, seconds: float, gain: float, now: float) -> None:
        """Decay both paths to ``now`` and add one child's seconds and HV gain to ``path``."""
        f = self._factor(now)
        for table in (self.gain, self.seconds):
            for key in table:
                table[key] *= f
        self.stamp = now
        self.seconds[path] += seconds
        self.gain[path] += gain

    def probability(self, now: float, options: GuidedOptions) -> tuple[float, float, float]:
        """Routing probability at ``now`` and the two decayed rates ``(p, rho_g, rho_o)``.

        Below ``warmup_seconds`` of decayed time on either path, or with a zero guided
        rate, the probability is ``p_probe``; with a zero ordinary rate and a positive
        guided rate it is ``p_on``; otherwise ``clip(p_on (rho_g / rho_o)^gamma)``.
        """
        f = self._factor(now)
        sg, so = self.seconds["g"] * f, self.seconds["o"] * f
        rho_g = self.gain["g"] / self.seconds["g"] if self.seconds["g"] > 0 else 0.0
        rho_o = self.gain["o"] / self.seconds["o"] if self.seconds["o"] > 0 else 0.0
        on, probe = options.probability, options.p_probe
        if sg < options.warmup_seconds or so < options.warmup_seconds or rho_g <= 0.0:
            return probe, rho_g, rho_o
        if rho_o <= 0.0:
            return on, rho_g, rho_o
        return min(on, max(probe, on * (rho_g / rho_o) ** options.gamma)), rho_g, rho_o


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
    # Recorded only with ``record_paths`` (see ``EXTRA_CELL_FIELDS``).
    destroyed_jobs: int = 0
    plan_bill_delta: float = 0.0  # sum of (designed timetable Bill - incumbent Bill)
    decoded_bill_delta: float = 0.0  # sum of (decoded exact Bill - incumbent Bill)
    plan_flow_delta: float = 0.0
    decoded_flow_delta: float = 0.0


EXTRA_CELL_FIELDS = (
    "destroyed_jobs",
    "plan_bill_delta",
    "decoded_bill_delta",
    "plan_flow_delta",
    "decoded_flow_delta",
)
BASE_CELL_FIELDS = tuple(f for f in Cell.__dataclass_fields__ if f not in EXTRA_CELL_FIELDS)


@dataclass
class PathBin:
    """Counters of one path in one tenth of the time budget."""

    slots: int = 0
    successes: int = 0
    seconds: float = 0.0
    inserted: int = 0
    inserted_children: int = 0
    hv_gain: float = 0.0


@dataclass
class GuidedVariation:
    """Routes offspring slots to a direction-guided destroy-repair child.

    ``gain_probe`` returns the frozen-frame HV gain of the latest evaluated candidate and
    ``terminal_probe`` whether the terminal sweep has started (its gains are not counted
    for the controller). ``timer`` is a private clock used only when the options ask for
    path timing; it is never the engine's clock.
    """

    config: Config
    options: GuidedOptions
    gain_probe: Callable[[], float] | None = None
    terminal_probe: Callable[[], bool] | None = None
    timer: Callable[[], float] = DEFAULT_CLOCK
    slots: int = 0
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    guided_bins: list[PathBin] = field(default_factory=lambda: [PathBin() for _ in range(10)])
    ordinary_bins: list[PathBin] = field(default_factory=lambda: [PathBin() for _ in range(10)])
    trace: dict[int, tuple[float, float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rates = PathRates(self.options.tau_seconds)
        self.t0 = self.timer() if self.options.timed else 0.0
        self._gateway: EvaluationGateway | None = None

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

    def routing_probability(self, now: float) -> float:
        """Probability of taking the guided path at run time ``now`` (seconds since start)."""
        if self.options.controller != "switch":
            return self.options.probability
        p, rho_g, rho_o = self.rates.probability(now, self.options)
        self.trace[int(now // self.options.trace_seconds)] = (p, rho_g, rho_o)
        return p

    # ---------------- observation ----------------
    def _decile(self, now: float) -> int:
        budget = self.config.seconds
        if not budget:
            return 0
        return min(9, max(0, int(10 * now / budget)))

    def _observe(
        self, path: str, seconds: float, gain: float, inserted: int, now: float, success: bool
    ) -> None:
        """Bin one child by budget decile and feed the controller (outside the terminal sweep)."""
        bins = self.guided_bins if path == "g" else self.ordinary_bins
        b = bins[self._decile(now)]
        b.slots += 1
        b.successes += int(success)
        b.seconds += seconds
        b.inserted += inserted
        b.inserted_children += int(inserted > 0)
        b.hv_gain += gain
        if self.options.controller == "switch" and not (
            self.terminal_probe is not None and self.terminal_probe()
        ):
            self.rates.observe(path, seconds, gain, now)

    def ordinary(self, build: Callable[[], Candidate]) -> Candidate:
        """Time and count the usual variation child (the engine's ``ordinary_path`` seam)."""
        gateway = self._gateway
        if not self.options.timed or gateway is None:
            return build()
        start, before = self.timer(), gateway.archive.insertions
        child = build()
        end = self.timer()
        added = gateway.archive.insertions - before
        gain = self.gain_probe() if self.gain_probe is not None and added > 0 else 0.0
        self._observe("o", end - start, gain, added, end - self.t0, True)
        return child

    # ---------------- the guided slot ----------------
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
        self._gateway = gateway
        timed = self.options.timed
        start = self.timer() if timed else 0.0
        if streams.stream("F3:route").random() >= self.routing_probability(start - self.t0):
            return None
        direction = preference(subproblem, len(population))
        rng = streams.stream("F3:destroy")
        mode = self.mode_for(direction, rng)
        cell = self.cells.setdefault((PREFERENCE_LABELS[direction], mode), Cell())
        cell.attempts += 1
        incumbent = population[subproblem]
        ideal, top = gateway.ideal, maximum(population)
        n_jobs = len(gateway.problem.jobs)
        count = self.options.destroy_count
        ratio = (
            TargetedDestroyRepair.destroy_ratio if count is None else min(count, n_jobs) / n_jobs
        )
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
            destroy_ratio=ratio,
        )
        started = perf_counter()
        try:
            batch = operator.propose(gateway.problem, incumbent, 1, self.config, rng)
        except ValueError as exc:
            if REINSERTION_FAILURE not in str(exc):
                raise
            cell.fallback_no_reinsertion += 1
            cell.fallback_seconds += perf_counter() - started
            self._wasted(timed, start)
            return None
        if not batch.proposals:
            cell.fallback_no_proposal += 1
            cell.fallback_seconds += perf_counter() - started
            self._wasted(timed, start)
            return None
        proposal = batch.proposals[0]
        before = gateway.archive.insertions
        child = gateway.evaluate(proposal.genotype)
        cell.seconds += perf_counter() - started
        cell.successes += 1
        added = gateway.archive.insertions - before
        cell.inserted += added
        cell.inserted_children += int(added > 0)
        gain = self.gain_probe() if self.gain_probe is not None and added > 0 else 0.0
        cell.hv_gain += gain
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
            if self.options.record_paths:
                inc = incumbent.evaluation
                cell.plan_bill_delta += timetable["bill"] - inc.bill
                cell.decoded_bill_delta += child.evaluation.bill - inc.bill
                cell.plan_flow_delta += timetable["flow"] - inc.flow
                cell.decoded_flow_delta += child.evaluation.flow - inc.flow
        if self.options.record_paths:
            cell.destroyed_jobs += int(batch.diagnostics.get("destroyed", 0))
        if timed:
            end = self.timer()
            self._observe("g", end - start, gain, added, end - self.t0, True)
        return child

    def _wasted(self, timed: bool, start: float) -> None:
        """Charge a fallen-back slot's guided seconds (no gain) to the guided path."""
        if timed:
            end = self.timer()
            self._observe("g", end - start, 0.0, 0, end - self.t0, False)

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
        """JSON-ready counters per cell, per direction class and in total.

        Arms with ``record_paths`` add the cost-control fields, the per-decile bins of
        both paths and (for the switch controller) the probability and rate trace.
        """
        keys = list(BASE_CELL_FIELDS) + (
            list(EXTRA_CELL_FIELDS) if self.options.record_paths else []
        )
        cells = {
            f"{d}/{m}": {k: getattr(c, k) for k in keys} for (d, m), c in sorted(self.cells.items())
        }
        by_direction: dict[str, dict[str, float]] = {}
        total = dict.fromkeys(keys, 0)
        for (direction, _), cell in self.cells.items():
            row = by_direction.setdefault(direction, dict.fromkeys(keys, 0))
            for key in keys:
                row[key] += getattr(cell, key)
                total[key] += getattr(cell, key)
        out = dict(
            options=self.options.record(),
            slots=self.slots,
            attempts=total["attempts"],
            successes=total["successes"],
            fallback_no_reinsertion=total["fallback_no_reinsertion"],
            fallback_no_proposal=total["fallback_no_proposal"],
            guided_seconds=total["seconds"] + total["fallback_seconds"],
            exact_per_second=(total["successes"] / total["seconds"] if total["seconds"] else 0.0),
            by_direction=by_direction,
            cells=cells,
        )
        if self.options.record_paths:
            out["deciles"] = dict(
                guided=[asdict(b) for b in self.guided_bins],
                ordinary=[asdict(b) for b in self.ordinary_bins],
            )
        if self.options.controller == "switch":
            out["controller"] = dict(
                trace_seconds=self.options.trace_seconds,
                trace={str(k): list(v) for k, v in sorted(self.trace.items())},
            )
        return out


__all__ = [
    "BASE_CELL_FIELDS",
    "DEFAULT_CLOCK",
    "DIRECTION_MODES",
    "EXTRA_CELL_FIELDS",
    "GUIDED_MODES",
    "NEUTRAL_WEIGHT",
    "Cell",
    "GuidedOptions",
    "GuidedVariation",
    "PathBin",
    "PathRates",
]
