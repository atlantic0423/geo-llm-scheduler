"""MOEA/D-structured variants of the S4-B2 local search (development screen arms).

All variants start from S4-B2 (no right shift during the search, terminal sweep
of the Archive front). Each option changes one MOEA/D-facing decision:

``ls_enabled=False``
    No LS trajectory at all; only the terminal sweep runs (control "no LS").
``terminal_ops``
    Timing-only operators (A7/A8) are removed from the Q pool by the caller and
    applied here, Archive-only, to every terminal front target after its right
    shift (heritability firewall P1).
``refine``
    ``flow_safe``/``weighted`` installs :class:`DecoderRefiner` on the gateway
    after initialization (P2a/P2b); ``weighted`` uses a weight class that the
    caller sets per MOEA/D subproblem and that LS sets per incumbent ray.
``seed_mode``
    ``duplicate`` (P3): seed from a population cluster of subproblems sharing one
    incumbent, scoring best-of-B by a cluster-end subproblem weight.
    ``region_yield`` (P4): five weight regions; LS goes to regions where its
    decayed Archive-insertion rate exceeds that of evolutionary offspring, with
    wider steps there (best-of-``wide_budget`` instead of the configured B; the
    Q state encodes remaining steps, so depth stays fixed). ``region_random`` is
    the random-region reference (uniform region, configured width). ``regions``
    sets the number of equal-width bins of the Flow weight of a front point's
    Tchebycheff ray (3: Flow end, balanced, Bill end). ``region_ratio`` keeps
    the same rates and widening rule but samples regions by comparative
    advantage, weight (LS_r + d) / (evo_r + d) with d = 5% of the largest rate,
    so LS still has a preferred region when evolution out-yields it everywhere
    (where ``region_yield`` falls back to near-uniform epsilon weights).
    ``region_bandit`` (N6) treats regions as arms of a rate bandit over LS time:
    each LS trajectory is charged to the region it was launched from, its cost is
    the LS seconds it used and its reward the in-frame HV gain of its Archive
    insertions; a region's rate is decayed reward over decayed cost (both decay
    with ``tau_seconds``, so an unvisited region keeps its last estimate).
    Untried present regions are launched first, then regions are sampled with
    weight rate + 5% of the largest rate. No widening; requires HV yields.
``share_mode``
    ``fixed`` (default): LS may use ``time_share`` of search time throughout.
    ``equimarginal`` (N7): the share follows the measured marginal value of LS
    time. LS and evolution each keep a decayed in-frame HV gain over their
    decayed seconds (``tau_seconds``); the current share is
    ``time_share * (LS rate / evolution rate) ** share_exponent`` clipped to
    ``share_bounds``, and the LS allowance integrates that share over the
    inclusive clock, so LS is front-loaded while it buys more HV per second
    than evolution and recedes when it buys less.
``yield_measure``
    ``count`` (default): a yield event is one Archive insertion. ``hv``: it is
    the insertion's HV gain over the previous Archive in the current frozen
    frame, so equal-objective timing variants and negligible steps add nothing.
    A degenerate frame (a single-point front, zero range in an objective) has no
    defined HV scale; insertions measured in it gain 0.
``trace``
    Record-only instrumentation (no decision reads it): per ``trace_seconds`` bin
    of the inclusive clock and per ``TRACE_WBINS`` bin of the Flow ray weight,
    Archive insertions and their hypervolume gain in the current frozen frame,
    split by source (evolution, LS, terminal sweep, other), plus LS seed picks
    and widened picks; the yield arms also keep each bin's last decayed rates.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from random import Random
from typing import Any, cast

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.lsv2.decoder_refine import DecoderRefiner, weight_class
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, gap_scores
from geo_llm_scheduler.lsv2.search import Point, _with_diagnostics
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, maximum
from geo_llm_scheduler.utils.rng import RNGManager

SEED_MODES = (
    "gap",
    "duplicate",
    "region_yield",
    "region_random",
    "region_ratio",
    "region_bandit",
)
YIELD_MODES = ("region_yield", "region_ratio")
HV_MODES = (*YIELD_MODES, "region_bandit")
REGION_MODES = ("region_yield", "region_random", "region_ratio", "region_bandit")
SHARE_MODES = ("fixed", "equimarginal")
SHARE_MIN_SECONDS = 5.0
YIELD_MEASURES = ("count", "hv")
TRACE_WBINS = 30
TRACE_FIELDS = (
    "evo",
    "ls",
    "terminal",
    "other",
    "evo_hv",
    "ls_hv",
    "terminal_hv",
    "other_hv",
    "seed",
    "wide",
)


@dataclass(frozen=True)
class MoeadOptions:
    """Declared MOEA/D-facing switches of one arm (defaults equal S4-B2)."""

    ls_enabled: bool = True
    terminal_ops: tuple[str, ...] = ()
    refine: str | None = None
    seed_mode: str = "gap"
    regions: int = 5
    tau_seconds: float = 120.0
    wide_budget: int = 6
    trace: bool = False
    trace_seconds: float = 30.0
    yield_measure: str = "count"
    share_mode: str = "fixed"
    share_bounds: tuple[float, float] = (0.05, 0.5)
    share_exponent: float = 0.5

    def __post_init__(self) -> None:
        if self.seed_mode not in SEED_MODES:
            raise ValueError("Unknown seed mode")
        if self.refine not in (None, "flow_safe", "weighted"):
            raise ValueError("Unknown refine mode")
        if not set(self.terminal_ops) <= {"A7", "A8"}:
            raise ValueError("Terminal operators must be timing-only A7/A8")
        if self.regions < 2 or self.tau_seconds <= 0 or self.wide_budget < 1:
            raise ValueError("Invalid region or depth settings")
        if self.trace_seconds <= 0:
            raise ValueError("Trace bins must be positive")
        if self.yield_measure not in YIELD_MEASURES:
            raise ValueError("Unknown yield measure")
        if self.yield_measure != "count" and self.seed_mode not in HV_MODES:
            raise ValueError("A yield measure needs a yield-driven seed mode")
        if self.seed_mode == "region_bandit" and self.yield_measure != "hv":
            raise ValueError("The region bandit is rewarded by HV gain")
        if self.share_mode not in SHARE_MODES:
            raise ValueError("Unknown LS share mode")
        lo, hi = self.share_bounds
        if not 0 < lo < hi < 1 or self.share_exponent <= 0:
            raise ValueError("Invalid LS share bounds or exponent")


def degenerate(frame: FrozenFrame) -> bool:
    """True when an objective has (numerically) zero range in the frozen frame."""
    return any(
        frame.nadir[k] - frame.ideal[k] <= 1e-9 * max(1.0, abs(frame.nadir[k])) for k in range(2)
    )


class MoeadMixin:
    """Overrides the LSV2 research seams; mixed in front of the S4 search class."""

    moead: MoeadOptions

    def moead_init(self, options: MoeadOptions) -> None:
        """Attach options and empty run-local state (no RNG is drawn here)."""
        self.moead = options
        self._ready = False
        self._rng: Random | None = None
        self._target_weight: tuple[float, float] | None = None
        self._base_config: object | None = None
        k = options.regions
        self._rate = dict(evo=[0.0] * k, ls=[0.0] * k)
        self._rate_time = dict(evo=0.0, ls=0.0)
        self._rate_clock: float | None = None
        self._rate_ls_seconds = 0.0
        self._last_insertions = 0
        self.moead_counts: dict[str, int] = {}
        self._last_wide = False
        self._clock_offset: float | None = None
        self._hv_frame: FrozenFrame | None = None
        self._hv_last = 0.0
        self._trace: dict[tuple[int, int, int], float] = {}
        self._trace_rates: dict[int, list[list[float]]] = {}
        self._trace_ls_seconds: dict[int, float] = {}
        self._bandit_gain = [0.0] * k
        self._bandit_time = [0.0] * k
        self._bandit_region: int | None = None
        self._bandit_clock: float | None = None
        self._bandit_mark = 0.0
        self._share: float | None = None
        self._allow = 0.0
        self._allow_clock: float | None = None
        self._mgain = dict(evo=0.0, ls=0.0)
        self._mtime = dict(evo=0.0, ls=0.0)
        self._mclock: float | None = None
        self._mmark = 0.0
        self._share_trace: dict[int, float] = {}

    def _mcount(self, name: str, value: int = 1) -> None:
        self.moead_counts[name] = self.moead_counts.get(name, 0) + value

    # ---------------- per-call setup ----------------
    def _setup(self, population: list[Candidate], gateway: EvaluationGateway, streams: RNGManager):
        self._ready = True
        self._rng = streams.stream("lsv2:moead")
        self._last_insertions = gateway.archive.insertions
        if self.moead.refine is not None:
            flows = [c.evaluation.flow for c in population]
            bills = [c.evaluation.bill for c in population]
            scales = (
                max(max(flows) - min(flows), 1e-9 * max(1.0, max(flows))),
                max(max(bills) - min(bills), 1e-9 * max(1.0, max(bills))),
            )
            gateway.refine = DecoderRefiner(gateway.problem, self.moead.refine, scales)
            gateway.refine_class = 0
        if (
            self.moead.seed_mode in HV_MODES
            or self.moead.trace
            or self.moead.share_mode == "equimarginal"
        ):
            previous = gateway.candidate_sink

            def sink(candidate: Candidate) -> None:
                self._on_candidate(candidate, gateway)
                if previous is not None:
                    previous(candidate)

            gateway.candidate_sink = sink

    def __call__(  # type: ignore[override]
        self,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        budget_elapsed: Callable[[], float],
    ) -> None:
        """Install run-level seams once, then run S4-B2 LS or only its terminal sweep."""
        if not self._ready:
            self._setup(population, gateway, streams)
        if self.moead.trace:
            self._clock_offset = budget_elapsed() - self.timer()  # type: ignore[attr-defined]
        if self.moead.ls_enabled:
            super().__call__(population, gateway, streams, budget_elapsed)  # type: ignore[misc]
            return
        now = budget_elapsed()
        base = cast(Any, self)
        if base.search_start is None:
            base.search_start = now
        if base.terminal_share is not None and base._terminal_due(now):
            base._terminal_sweep(population, gateway, streams, budget_elapsed)

    # ---------------- P1: terminal timing operators ----------------
    def _terminal_extra(
        self,
        target: Candidate,
        front: list[Point],
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        budget_elapsed: Callable[[], float],
    ) -> None:
        """Apply A7/A8 Archive-only to one terminal target; entrants are swept later."""
        if not self.moead.terminal_ops:
            return
        problem = gateway.problem
        frame = FrozenFrame.of(front)
        current = _with_diagnostics(problem, target)
        for name in self.moead.terminal_ops:
            if self._deadline(budget_elapsed()):  # type: ignore[attr-defined]
                return
            batch = self._propose(  # type: ignore[attr-defined]
                name, current, frame, streams.stream("lsv2:terminal:" + name), problem
            )
            self._mcount("terminal_calls:" + name)
            if not batch.proposals:
                continue
            context = NormalizationContext(gateway.ideal, maximum(population))
            before = gateway.archive.insertions
            execute(
                batch,
                current,
                self.config.budget,  # type: ignore[attr-defined]
                frame.weight_for(current.evaluation.objectives),
                context,
                gateway,
                "LSV2:TERMINAL_" + name,
            )
            self._mcount("terminal_entrants:" + name, gateway.archive.insertions - before)

    # ---------------- P2b: LS weight class ----------------
    def _before_execute(
        self, current: Candidate, frame: FrozenFrame, gateway: EvaluationGateway
    ) -> None:
        """LS decodes use the class of the incumbent's ray (or of the P3 target)."""
        if self.moead.refine == "weighted" and gateway.refine is not None:
            weight = self._execute_weight(frame, current)
            gateway.refine_class = weight_class(min(1.0, max(0.0, weight[1])))

    def _execute_weight(self, frame: FrozenFrame, current: Candidate) -> tuple[float, float]:
        if self._target_weight is not None:
            return self._target_weight
        return frame.weight_for(current.evaluation.objectives)

    # ---------------- P3/P4: seeds ----------------
    def _region_of(self, frame: FrozenFrame, point: Point) -> int:
        k = self.moead.regions
        return min(k - 1, int(k * frame.weight_for(point)[0]))

    def _gap_pick(self, frame: FrozenFrame, front: list[Point], allowed: list[int]) -> int:
        assert self._rng is not None and self.sampler is not None  # type: ignore[attr-defined]
        scores = gap_scores([frame.normalize(p) for p in front], self.config.end_bonus)  # type: ignore[attr-defined]
        sampler = self.sampler  # type: ignore[attr-defined]
        weights = [
            scores[i] * (sampler.discount ** sampler.failures.get(front[i], 0)) + 1e-12
            for i in allowed
        ]
        return allowed[self._rng.choices(range(len(allowed)), weights=weights)[0]]

    def _pick_index(self, frame: FrozenFrame, front: list[Point], population: list[Candidate]):
        self._target_weight = None
        self._last_wide = False
        mode = self.moead.seed_mode
        if mode == "duplicate":
            index = self._pick_duplicate(frame, front, population)
        elif mode in REGION_MODES:
            index = self._pick_region(frame, front)
        else:
            index = super()._pick_index(frame, front, population)  # type: ignore[misc]
        if self.moead.trace:
            self._trace_seed(frame, front[index])
        return index

    def _pick_duplicate(
        self, frame: FrozenFrame, front: list[Point], population: list[Candidate]
    ) -> int:
        assert self._rng is not None
        on_front = {p: i for i, p in enumerate(front)}
        clusters: dict[Point, list[int]] = {}
        for j, member in enumerate(population):
            point = member.evaluation.objectives
            if point in on_front:
                clusters.setdefault(point, []).append(j)
        shared = sorted((p, js) for p, js in clusters.items() if len(js) >= 2)
        if not shared:
            self._mcount("duplicate_fallback")
            return super()._pick_index(frame, front, population)  # type: ignore[misc]
        point, members = shared[
            self._rng.choices(range(len(shared)), weights=[len(js) for _, js in shared])[0]
        ]
        end = min(members) if self._rng.random() < 0.5 else max(members)
        self._target_weight = self.lambdas[end]  # type: ignore[attr-defined]
        self._mcount("duplicate_seeds")
        self._mcount("duplicate_cluster_size", len(members))
        return on_front[point]

    def _decay_to(self, now: float) -> None:
        if self._rate_clock is None:
            self._rate_clock = now
            self._rate_ls_seconds = self.ls_seconds  # type: ignore[attr-defined]
            return
        dt = max(0.0, now - self._rate_clock)
        ls_dt = max(0.0, self.ls_seconds - self._rate_ls_seconds)  # type: ignore[attr-defined]
        factor = math.exp(-dt / self.moead.tau_seconds)
        for source in ("evo", "ls"):
            self._rate[source] = [v * factor for v in self._rate[source]]
            self._rate_time[source] *= factor
        self._rate_time["ls"] += min(ls_dt, dt)
        self._rate_time["evo"] += max(0.0, dt - ls_dt)
        self._rate_clock, self._rate_ls_seconds = now, self.ls_seconds  # type: ignore[attr-defined]

    def _rates(self) -> tuple[list[float], list[float]]:
        evo_t, ls_t = self._rate_time["evo"], self._rate_time["ls"]
        evo = [c / evo_t if evo_t > 0 else 0.0 for c in self._rate["evo"]]
        ls = [c / ls_t if ls_t > 0 else 0.0 for c in self._rate["ls"]]
        return evo, ls

    def _on_candidate(self, candidate: Candidate, gateway: EvaluationGateway) -> None:
        """Attribute each Archive insertion to evolution or LS and to a weight region.

        The trace (when enabled) only records; the yield rates and every decision
        are the same with or without it.
        """
        insertions = gateway.archive.insertions
        inserted = insertions > self._last_insertions
        self._last_insertions = insertions
        frame = self.frame  # type: ignore[attr-defined]
        if not inserted or frame is None:
            return
        terminal = self.terminal_started is not None  # type: ignore[attr-defined]
        origin = candidate.origin
        source: str | None
        if origin == "structural":
            source = "evo"
        elif origin.startswith("LSV2:") and "RIGHTSHIFT" not in origin:
            source = "ls"
        else:
            source = None
        yield_mode = self.moead.seed_mode in YIELD_MODES
        bandit = self.moead.seed_mode == "region_bandit"
        hv_yield = (yield_mode or bandit) and self.moead.yield_measure == "hv"
        marginal = self.moead.share_mode == "equimarginal"
        gain = (
            self._insertion_gain(frame, candidate, gateway)
            if self.moead.trace or hv_yield or marginal
            else 0.0
        )
        if self.moead.trace:
            label = "terminal" if terminal else (source or "other")
            self._trace_insertion(frame, candidate, label, gain)
        if marginal and not terminal and source is not None:
            self._share_accrue(self.timer())  # type: ignore[attr-defined]
            self._mgain[source] += gain
        if bandit:
            if not terminal and source == "ls" and self._bandit_region is not None:
                self._bandit_accrue(self.timer())  # type: ignore[attr-defined]
                self._bandit_gain[self._bandit_region] += gain
            return
        if terminal or source is None or not yield_mode:
            return
        self._decay_to(self.timer())  # type: ignore[attr-defined]
        region = self._region_of(frame, candidate.evaluation.objectives)
        self._rate[source][region] += gain if hv_yield else 1.0

    # ---------------- record-only trace ----------------
    def _trace_time(self) -> int:
        now = self.timer() + (self._clock_offset or 0.0)  # type: ignore[attr-defined]
        return max(0, int(now // self.moead.trace_seconds))

    def _trace_add(self, t: int, w: int, field: str, value: float) -> None:
        key = (t, w, TRACE_FIELDS.index(field))
        self._trace[key] = self._trace.get(key, 0.0) + value

    @staticmethod
    def _wbin(frame: FrozenFrame, point: Point) -> int:
        return min(TRACE_WBINS - 1, int(TRACE_WBINS * frame.weight_for(point)[0]))

    def _insertion_gain(
        self, frame: FrozenFrame, candidate: Candidate, gateway: EvaluationGateway
    ) -> float:
        """HV gain of one insertion over the previous Archive in this frozen frame.

        The first insertion seen in a new frame is charged its exclusive
        contribution (the Archive before it is no longer available); an
        insertion whose objectives already exist in the Archive gains nothing.
        """
        points = [m.evaluation.objectives for m in gateway.archive.members]
        if degenerate(frame):
            self._hv_frame, self._hv_last = None, 0.0
            self._mcount("degenerate_frame_insertions")
            return 0.0
        hv = frame.hv(points)
        if self._hv_frame is frame:
            gain = hv - self._hv_last
        else:
            point = candidate.evaluation.objectives
            rest = [p for p in points if p != point]
            gain = hv - frame.hv(rest) if len(rest) == len(points) - 1 else 0.0
            self._mcount("trace_frame_resets")
        self._hv_frame, self._hv_last = frame, hv
        return max(0.0, gain)

    def _trace_insertion(
        self, frame: FrozenFrame, candidate: Candidate, label: str, gain: float
    ) -> None:
        """Record one insertion and its HV gain in its time and ray-weight bin."""
        t = self._trace_time()
        w = self._wbin(frame, candidate.evaluation.objectives)
        self._trace_add(t, w, label, 1.0)
        self._trace_add(t, w, label + "_hv", gain)
        self._trace_ls_seconds[t] = self.ls_seconds  # type: ignore[attr-defined]

    def _trace_seed(self, frame: FrozenFrame, point: Point) -> None:
        t = self._trace_time()
        w = self._wbin(frame, point)
        self._trace_add(t, w, "seed", 1.0)
        if self._last_wide:
            self._trace_add(t, w, "wide", 1.0)
        if self.moead.seed_mode in YIELD_MODES:
            evo, ls = self._rates()
            self._trace_rates[t] = [list(evo), list(ls)]
        elif self.moead.seed_mode == "region_bandit":
            self._trace_rates[t] = [list(self._bandit_time), self._bandit_rates()]
        self._trace_ls_seconds[t] = self.ls_seconds  # type: ignore[attr-defined]

    def trace_record(self) -> dict:
        """JSON-ready sparse trace: ``cells`` rows are [time bin, weight bin, field, value]."""
        return dict(
            seconds=self.moead.trace_seconds,
            wbins=TRACE_WBINS,
            fields=list(TRACE_FIELDS),
            cells=[[t, w, f, v] for (t, w, f), v in sorted(self._trace.items())],
            rates={str(t): r for t, r in sorted(self._trace_rates.items())},
            ls_seconds={str(t): v for t, v in sorted(self._trace_ls_seconds.items())},
        )

    def _pick_region(self, frame: FrozenFrame, front: list[Point]) -> int:
        assert self._rng is not None
        base = cast(Any, self)
        if self._base_config is None:
            self._base_config = base.config
        wide = False
        regions: dict[int, list[int]] = {}
        for i, p in enumerate(front):
            regions.setdefault(self._region_of(frame, p), []).append(i)
        present = sorted(regions)
        if self.moead.seed_mode == "region_random":
            region = present[self._rng.randrange(len(present))]
        elif self.moead.seed_mode == "region_bandit":
            region = self._bandit_pick(present)
        else:
            self._decay_to(self.timer())  # type: ignore[attr-defined]
            evo, ls = self._rates()
            top = max([*evo, *ls, 0.0])
            eps = 0.05 * top if top > 0 else 1.0
            if self.moead.seed_mode == "region_ratio":
                weights = [(ls[r] + eps) / (evo[r] + eps) for r in present]
            else:
                weights = [max(0.0, ls[r] - evo[r]) + eps for r in present]
            region = present[self._rng.choices(range(len(present)), weights=weights)[0]]
            if ls[region] > evo[region]:
                wide = True
                self._mcount("region_wide")
        self._last_wide = wide
        base.config = (
            replace(cast(Any, self._base_config), budget=self.moead.wide_budget)
            if wide
            else self._base_config
        )
        self._mcount(f"region_seed:{region}")
        return self._gap_pick(frame, front, regions[region])

    # ---------------- N7: equimarginal LS share ----------------
    def _share_accrue(self, now: float) -> None:
        """Decay both sources' HV gains and seconds to ``now``; add the elapsed
        LS and evolution seconds since the last accrual."""
        ls_now = self.ls_seconds  # type: ignore[attr-defined]
        if self._mclock is None:
            self._mclock, self._mmark = now, ls_now
            return
        dt = max(0.0, now - self._mclock)
        ls_dt = min(dt, max(0.0, ls_now - self._mmark))
        factor = math.exp(-dt / self.moead.tau_seconds)
        for source in ("evo", "ls"):
            self._mgain[source] *= factor
            self._mtime[source] *= factor
        self._mtime["ls"] += ls_dt
        self._mtime["evo"] += dt - ls_dt
        self._mclock, self._mmark = now, ls_now

    def _target_share(self) -> float:
        """``time_share * (LS HV/s / evolution HV/s) ** exponent`` within the bounds."""
        base = cast(Any, self).config.time_share
        lo, hi = self.moead.share_bounds
        self._share_accrue(self.timer())  # type: ignore[attr-defined]
        if min(self._mtime.values()) < SHARE_MIN_SECONDS:
            return min(hi, max(lo, base))
        ls_rate = self._mgain["ls"] / self._mtime["ls"]
        evo_rate = self._mgain["evo"] / self._mtime["evo"]
        if evo_rate <= 0:
            return hi if ls_rate > 0 else min(hi, max(lo, base))
        return min(hi, max(lo, base * (ls_rate / evo_rate) ** self.moead.share_exponent))

    def _ls_allowance(self, now: float) -> float:
        """Fixed share (reference) or the integral of the equimarginal share."""
        if self.moead.share_mode == "fixed":
            return super()._ls_allowance(now)  # type: ignore[misc]
        base = cast(Any, self)
        if self._allow_clock is None:
            self._allow_clock = base.search_start
            self._share = base.config.time_share
        assert self._share is not None and self._allow_clock is not None
        self._allow += self._share * max(0.0, now - self._allow_clock)
        self._allow_clock = now
        self._share = self._target_share()
        if self.moead.trace:
            self._share_trace[self._trace_time()] = self._share
        return self._allow

    # ---------------- N6: region rate bandit ----------------
    def _bandit_accrue(self, now: float) -> None:
        """Decay rewards and costs to ``now``; charge LS seconds since the last
        accrual to the region of the trajectory in progress."""
        ls_now = self.ls_seconds  # type: ignore[attr-defined]
        if self._bandit_clock is None:
            self._bandit_clock, self._bandit_mark = now, ls_now
            return
        factor = math.exp(-max(0.0, now - self._bandit_clock) / self.moead.tau_seconds)
        self._bandit_gain = [v * factor for v in self._bandit_gain]
        self._bandit_time = [v * factor for v in self._bandit_time]
        if self._bandit_region is not None:
            self._bandit_time[self._bandit_region] += max(0.0, ls_now - self._bandit_mark)
        self._bandit_clock, self._bandit_mark = now, ls_now

    def _bandit_rates(self) -> list[float]:
        """Decayed HV gain per decayed LS second of each region (0 before any use)."""
        return [g / t if t > 0 else 0.0 for g, t in zip(self._bandit_gain, self._bandit_time)]

    def _bandit_pick(self, present: list[int]) -> int:
        """Close the previous trajectory's account and choose the next launch region."""
        assert self._rng is not None
        self._bandit_accrue(self.timer())  # type: ignore[attr-defined]
        untried = [r for r in present if self._bandit_time[r] <= 0.0]
        if untried:
            region = untried[self._rng.randrange(len(untried))]
            self._mcount("bandit_explore")
        else:
            rates = self._bandit_rates()
            top = max(rates[r] for r in present)
            eps = 0.05 * top if top > 0 else 1.0
            weights = [rates[r] + eps for r in present]
            region = present[self._rng.choices(range(len(present)), weights=weights)[0]]
        self._bandit_region = region
        return region

    def moead_summary(self) -> dict:
        """JSON-ready options, counters and final decayed rates (plus the trace if on)."""
        evo, ls = self._rates()
        options = dict(
            ls_enabled=self.moead.ls_enabled,
            terminal_ops=list(self.moead.terminal_ops),
            refine=self.moead.refine,
            seed_mode=self.moead.seed_mode,
            regions=self.moead.regions,
            tau_seconds=self.moead.tau_seconds,
            wide_budget=self.moead.wide_budget,
        )
        out = dict(
            options=options, counts=dict(self.moead_counts), final_rates=dict(evo=evo, ls=ls)
        )
        if self.moead.yield_measure != "count":
            options.update(yield_measure=self.moead.yield_measure)
        if self.moead.share_mode != "fixed":
            options.update(
                share_mode=self.moead.share_mode,
                share_bounds=list(self.moead.share_bounds),
                share_exponent=self.moead.share_exponent,
            )
            out["share"] = dict(
                final=self._share,
                allowance=self._allow,
                gain=dict(self._mgain),
                seconds=dict(self._mtime),
                trace={str(t): v for t, v in sorted(self._share_trace.items())},
            )
        if self.moead.seed_mode == "region_bandit":
            out["bandit"] = dict(
                gain=list(self._bandit_gain),
                time=list(self._bandit_time),
                rates=self._bandit_rates(),
            )
        if self.moead.trace:
            options.update(trace=True, trace_seconds=self.moead.trace_seconds)
            out["trace"] = self.trace_record()
        return out


def with_moead(search: object, options: MoeadOptions) -> MoeadMixin:
    """Re-class an S4 search instance to its MOEA/D-structured variant in place."""
    base = type(search)
    variant = type("Moead" + base.__name__, (MoeadMixin, base), {})
    search.__class__ = variant
    assert isinstance(search, MoeadMixin)
    search.moead_init(options)
    return search


__all__ = [
    "HV_MODES",
    "REGION_MODES",
    "SEED_MODES",
    "SHARE_MODES",
    "YIELD_MEASURES",
    "YIELD_MODES",
    "TRACE_FIELDS",
    "TRACE_WBINS",
    "MoeadMixin",
    "degenerate",
    "MoeadOptions",
    "with_moead",
]
