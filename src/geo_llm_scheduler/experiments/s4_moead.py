"""S4-B2 with MOEA/D-structured changes: nine-arm development screen (2026-11-06).

Baseline ``M0_B2`` is S4-B2 (``s4_decouple.B2_TERMINAL_SHIFT``): no right shift
during the search, Archive-only right shift of the front in the final 5%. Every
other arm changes one declared decision (see :mod:`lsv2.moead_search`):

========================  ====================================================
``M1_NO_LS``              no LS trajectories; terminal right-shift sweep kept
``M2_P1_T05``/``M3_P1_T10``  A7/A8 removed from the Q pool, applied Archive-only
                          in the terminal sweep; terminal share 5% / 10%
``M4_P2A``                decoder-embedded Flow-safe right shift (heritable)
``M5_P2B``                ``M4`` plus weight-class Decode trade-off moves
``M6_P3``                 LS seeds from duplicate-incumbent subproblem clusters
``M7_P4``                 LS seeds and step width by per-region insertion yields
``M8_P4_RANDOM``          uniform random region, configured depth (reference)
========================  ====================================================

Judgement (user, 2026-11-06): P1/P2/P3 against ``M0_B2``; P4 against
``M1_NO_LS``; every arm is reported against both.

Combination arms (``COMBO_ARMS``, user 2026-11-06): P4 on top of P1, all with
the record-only region/time trace. ``N0_P1`` is ``M2_P1_T05`` plus the trace.

========================  ====================================================
``N0_P1``                 P1 terminal 5% (control)
``N1_P1_P4R5``            P1 + P4 region-yield seeds and width, five regions
``N2_P1_P4R3``            P1 + P4 with three regions (Flow end, balanced, Bill end)
``N3_P1_P4R3_RANDOM``     P1 + uniform random region of three (mechanism reference)
========================  ====================================================

Follow-up arms (``NEXT_ARMS``), three regions, P1 and the trace:

========================  ====================================================
``N4_P1_P4R3_HV``         ``N2`` with HV-gain yields instead of insertion counts
``N5_P1_P4R3_CA``         HV-gain yields and comparative-advantage region sampling
========================  ====================================================

Bandit arm (``BANDIT_ARMS``), three regions, P1 and the trace:

========================  ====================================================
``N6_P1_P4R3_BANDIT``     LS launch region by a decayed HV-per-LS-second rate
                          bandit (launch-region credit, no widening)
========================  ====================================================

Time-share arms (``SHARE_ARMS``), P1 and the trace; candidates, not yet run:

========================  ====================================================
``N7_P1_SHARE``           gap seeds; LS share follows LS/evolution HV per second
``N8_P1_P4R3_BANDIT_SHARE`` N6 region bandit plus the N7 equimarginal share
========================  ====================================================

Widening controls (``WIDEN_ARMS``), P1 and the trace; candidates, not yet run.
They separate where LS is launched from P4's longer Flow-end steps:

==============================  ==============================================
``N9_P1_FLOWWIDE``              gap seeds; seeds in the Flow-end third always
                                run best-of-6 steps (no yields, no regions)
``N10_P1_P4R3_BANDIT_FLOWWIDE`` N6 region bandit plus the same Flow-end widening
==============================  ==============================================

Launch-point arms (``SPACE_ARMS``, user 2026-11-08), P1 and the trace; candidates,
not yet run. Each trajectory is charged to its launch point (see
:mod:`lsv2.seed_field`):

========================  ====================================================
``R1_P1_RANDPOINT``       uniformly random Archive-front point (reference)
``G1_P1_GP``              Thompson sampling of a Gaussian-RBF field of HV per LS
                          second over (Flow ray weight, budget fraction)
``E2_P1_EQM``             five weight segments with a shared diminishing return;
                          equimarginal launch and LS share from that marginal
``GE3_P1_GP_EQM``         field over (ray weight, local recent LS seconds);
                          equimarginal LS share from its best marginal return
========================  ====================================================

LS investment-control arms (``INVEST_ARMS``, user 2026-11-09): the working
version ``N7_P1_SHARE`` (B2 + P1 + N7, fixed three steps) with only the
trajectory length rule changed (see ``continuation`` in :mod:`lsv2.moead_search`):

========================  ====================================================
``X1_EXTEND``             three steps, then one more while the last step gained
                          Archive HV (at most ten)
``X1R_RANDEXT``           three steps, then random extension at the run's
                          gaining-step rate, blind to the last step (reference)
``X2_PROBE``              one step, then continue while the last step gained
``X2L_L1``                exactly one step (reference for ``X2_PROBE``)
========================  ====================================================

Guided-variation arms (``GUIDED_ARMS``, user 2026-11-10, F3): the working version
``N7_P1_SHARE`` (the study's V0 control, unchanged) with one routed variation path
(see :mod:`lsv2.guided_variation`). Each offspring slot takes, with probability
0.2, a targeted destroy-repair rebuild of the subproblem's incumbent instead of
crossover plus mutation:

========================  ====================================================
``V1_GUIDED_DIR``         destroy mode and rebuild weight follow the subproblem
                          direction (Flow third FLOW, middle MIXED, electricity
                          third TOU/PEAK)
``V2_GUIDED_RAND``        matched control: uniform mode, fixed (0.5, 0.5) weight
``V3_GUIDED_DIR_NB``      ``V1`` plus up to two neighbour-incumbent paths per
                          rebuilt Job
========================  ====================================================

Cost-control arms (``COST_ARMS``, user 2026-11-11, from the F3 result; all build on
the neutral ``V2`` variant and also record both paths per budget decile):

============================  ================================================
``G8_GUIDED_RAND``            ``V2_GUIDED_RAND`` repeated, with the extra records
``G4_GUIDED_RAND_D4``         only four destroyed Jobs per rebuild
``GS_GUIDED_RAND_SWITCH``     probability follows the decayed HV gain per second
                              of the guided path against the ordinary path
============================  ================================================
"""

from collections.abc import Callable
from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments import s4_decouple
from geo_llm_scheduler.experiments.gap_allocation import GapAllocator, GapOptions
from geo_llm_scheduler.experiments.r1_online import R1Options, RegionalBlockMixer
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.lsv2.decoder_refine import weight_class
from geo_llm_scheduler.lsv2.guided_variation import (
    DEFAULT_CLOCK,
    GuidedOptions,
    GuidedVariation,
)
from geo_llm_scheduler.lsv2.moead_search import MoeadMixin, MoeadOptions, with_moead
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search
from geo_llm_scheduler.lsv2.terminal_gates import POLICIES
from geo_llm_scheduler.moead.core import weights

BASE = "M0_B2"
NO_LS = "M1_NO_LS"
ARMS = (
    BASE,
    NO_LS,
    "M2_P1_T05",
    "M3_P1_T10",
    "M4_P2A",
    "M5_P2B",
    "M6_P3",
    "M7_P4",
    "M8_P4_RANDOM",
)
COMBO_ARMS = ("N0_P1", "N1_P1_P4R5", "N2_P1_P4R3", "N3_P1_P4R3_RANDOM")
NEXT_ARMS = ("N4_P1_P4R3_HV", "N5_P1_P4R3_CA")
BANDIT_ARMS = ("N6_P1_P4R3_BANDIT",)
SHARE_ARMS = ("N7_P1_SHARE", "N8_P1_P4R3_BANDIT_SHARE")
WIDEN_ARMS = ("N9_P1_FLOWWIDE", "N10_P1_P4R3_BANDIT_FLOWWIDE")
SPACE_ARMS = ("R1_P1_RANDPOINT", "G1_P1_GP", "E2_P1_EQM", "GE3_P1_GP_EQM")
INVEST_ARMS = ("X1_EXTEND", "X1R_RANDEXT", "X2_PROBE", "X2L_L1")
GUIDED_ARMS = ("V1_GUIDED_DIR", "V2_GUIDED_RAND", "V3_GUIDED_DIR_NB")
COST_ARMS = (
    "G8_GUIDED_RAND",
    "G4_GUIDED_RAND_D4",
    "GS_GUIDED_RAND_SWITCH",
    "GS4_GUIDED_RAND_SWITCH_D4",
)
EXTRA_ARMS = (
    ARMS
    + COMBO_ARMS
    + NEXT_ARMS
    + BANDIT_ARMS
    + SHARE_ARMS
    + WIDEN_ARMS
    + SPACE_ARMS
    + INVEST_ARMS
    + GUIDED_ARMS
    + COST_ARMS
    + tuple(POLICIES)
    + ("R1A_GS4", "R1P_GS4")
    + ("A1_GAP_GS4", "A0_SHUF_GS4")
)
PROTOCOL = "s4-b2-moead/v1"
TIMING_OPS = ("A7", "A8")

_OPTIONS = {
    BASE: MoeadOptions(),
    NO_LS: MoeadOptions(ls_enabled=False),
    "M2_P1_T05": MoeadOptions(terminal_ops=TIMING_OPS),
    "M3_P1_T10": MoeadOptions(terminal_ops=TIMING_OPS),
    "M4_P2A": MoeadOptions(refine="flow_safe"),
    "M5_P2B": MoeadOptions(refine="weighted"),
    "M6_P3": MoeadOptions(seed_mode="duplicate"),
    "M7_P4": MoeadOptions(seed_mode="region_yield"),
    "M8_P4_RANDOM": MoeadOptions(seed_mode="region_random"),
    "N0_P1": MoeadOptions(terminal_ops=TIMING_OPS, trace=True),
    "N1_P1_P4R5": MoeadOptions(terminal_ops=TIMING_OPS, seed_mode="region_yield", trace=True),
    "N2_P1_P4R3": MoeadOptions(
        terminal_ops=TIMING_OPS, seed_mode="region_yield", regions=3, trace=True
    ),
    "N3_P1_P4R3_RANDOM": MoeadOptions(
        terminal_ops=TIMING_OPS, seed_mode="region_random", regions=3, trace=True
    ),
    "N4_P1_P4R3_HV": MoeadOptions(
        terminal_ops=TIMING_OPS,
        seed_mode="region_yield",
        regions=3,
        trace=True,
        yield_measure="hv",
    ),
    "N5_P1_P4R3_CA": MoeadOptions(
        terminal_ops=TIMING_OPS,
        seed_mode="region_ratio",
        regions=3,
        trace=True,
        yield_measure="hv",
    ),
    "N6_P1_P4R3_BANDIT": MoeadOptions(
        terminal_ops=TIMING_OPS,
        seed_mode="region_bandit",
        regions=3,
        trace=True,
        yield_measure="hv",
    ),
    "N7_P1_SHARE": MoeadOptions(terminal_ops=TIMING_OPS, trace=True, share_mode="equimarginal"),
    "N8_P1_P4R3_BANDIT_SHARE": MoeadOptions(
        terminal_ops=TIMING_OPS,
        seed_mode="region_bandit",
        regions=3,
        trace=True,
        yield_measure="hv",
        share_mode="equimarginal",
    ),
    "N9_P1_FLOWWIDE": MoeadOptions(
        terminal_ops=TIMING_OPS, regions=3, static_wide=(2,), trace=True
    ),
    "R1_P1_RANDPOINT": MoeadOptions(terminal_ops=TIMING_OPS, seed_mode="random_point", trace=True),
    "G1_P1_GP": MoeadOptions(terminal_ops=TIMING_OPS, seed_mode="gp_field", trace=True),
    "E2_P1_EQM": MoeadOptions(
        terminal_ops=TIMING_OPS,
        seed_mode="marginal_segments",
        regions=5,
        trace=True,
        share_mode="equimarginal",
        share_signal="frontier",
    ),
    "GE3_P1_GP_EQM": MoeadOptions(
        terminal_ops=TIMING_OPS,
        seed_mode="gp_marginal",
        trace=True,
        share_mode="equimarginal",
        share_signal="frontier",
    ),
    "N10_P1_P4R3_BANDIT_FLOWWIDE": MoeadOptions(
        terminal_ops=TIMING_OPS,
        seed_mode="region_bandit",
        regions=3,
        trace=True,
        yield_measure="hv",
        static_wide=(2,),
    ),
}
_INVEST = {
    "X1_EXTEND": "extend",
    "X1R_RANDEXT": "random_extend",
    "X2_PROBE": "probe",
    "X2L_L1": "single",
}
_OPTIONS.update(
    {
        arm: replace(_OPTIONS["N7_P1_SHARE"], continuation=rule, max_steps=10)
        for arm, rule in _INVEST.items()
    }
)
_GUIDED = {
    "V1_GUIDED_DIR": GuidedOptions(),
    "V2_GUIDED_RAND": GuidedOptions(direction_aware=False),
    "V3_GUIDED_DIR_NB": GuidedOptions(neighbor_paths=True),
}
_GUIDED.update(
    {
        # F3 cost-control arms (2026-11-11): the neutral V2 variant with path recording
        # (``G8`` repeats ``V2_GUIDED_RAND``; only the records differ), a fixed 4 destroyed
        # Jobs, and a probability that follows the paths' decayed HV gain per second.
        "G8_GUIDED_RAND": GuidedOptions(direction_aware=False, record_paths=True),
        "G4_GUIDED_RAND_D4": GuidedOptions(
            direction_aware=False, destroy_count=4, record_paths=True
        ),
        "GS_GUIDED_RAND_SWITCH": GuidedOptions(
            direction_aware=False, controller="switch", record_paths=True
        ),
    }
)
# GS4 (2026-11-11, single new arm): both mechanisms at once, fixed 4 destroyed Jobs
# and the switch controller, on the neutral V2 variant with path records.
_GUIDED["GS4_GUIDED_RAND_SWITCH_D4"] = GuidedOptions(
    direction_aware=False, destroy_count=4, controller="switch", record_paths=True
)
_OPTIONS.update({arm: _OPTIONS["N7_P1_SHARE"] for arm in _GUIDED})
_GUIDED.update({arm: _GUIDED["GS4_GUIDED_RAND_SWITCH_D4"] for arm in POLICIES})
_OPTIONS.update({arm: replace(_OPTIONS["N7_P1_SHARE"], terminal_gate=arm) for arm in POLICIES})
# Online R1 (2026-11-13): the GS4 working version plus regional-block mixing,
# archive-only (R1A) or also replacing population members (R1P).
_R1 = {"R1A_GS4": R1Options(mode="archive"), "R1P_GS4": R1Options(mode="population")}
_GUIDED.update({arm: _GUIDED["GS4_GUIDED_RAND_SWITCH_D4"] for arm in _R1})
_OPTIONS.update({arm: _OPTIONS["N7_P1_SHARE"] for arm in _R1})
# Front-gap reproduction allocation (2026-11-13): the GS4 working version with
# subproblem slots sampled by Archive gap area (A1) or by the same probabilities
# shuffled across subproblems (A0, matched control).
_GAP = {"A1_GAP_GS4": GapOptions(mode="gap"), "A0_SHUF_GS4": GapOptions(mode="shuffled")}
_GUIDED.update({arm: _GUIDED["GS4_GUIDED_RAND_SWITCH_D4"] for arm in _GAP})
_OPTIONS.update({arm: _OPTIONS["N7_P1_SHARE"] for arm in _GAP})
_TERMINAL = {"M3_P1_T10": 0.10}


def _check(arm: str) -> None:
    if arm not in EXTRA_ARMS:
        raise ValueError("Unknown S4-B2 MOEA/D arm")


def options(arm: str) -> MoeadOptions:
    """Declared MOEA/D-facing switches of one arm."""
    _check(arm)
    return _OPTIONS[arm]


def guided_options(arm: str) -> GuidedOptions | None:
    """Guided-variation switches of an arm; ``None`` keeps the unchanged variation path."""
    _check(arm)
    return _GUIDED.get(arm)


def terminal_share(arm: str) -> float:
    """Terminal sweep share of the inclusive budget (5% except ``M3_P1_T10``)."""
    _check(arm)
    return _TERMINAL.get(arm, s4_decouple.TERMINAL_SHARE)


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """The unchanged S4-B2 engine and inclusive wall-clock configuration."""
    _check(arm)
    return s4_decouple.arm_config(s4_decouple.TERMINAL, seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """S4-B2 search fields; P1 arms drop the timing-only operators from the Q pool."""
    _check(arm)
    base = s4_decouple.search_config(s4_decouple.TERMINAL)
    if options(arm).terminal_ops:
        base = replace(base, operators=tuple(op for op in base.operators if op not in TIMING_OPS))
    return base


def make_search(arm: str, config: Config, calibration: dict) -> LSV2Search:
    """S4-B2 search (fresh Q learner, decoupled right shift) with the arm's options."""
    search = s4_decouple.make_search(s4_decouple.TERMINAL, config, calibration)
    search.config = search_config(arm)
    search.terminal_share = terminal_share(arm)
    with_moead(search, options(arm))
    return search


def configuration_record(arm: str, config: Config, calibration: dict) -> dict:
    """Record the S4-B2 boundaries plus the arm's declared MOEA/D switches."""
    record = s4_decouple.configuration_record(s4_decouple.TERMINAL, config, calibration)
    opts = options(arm)
    record.update(
        arm=arm,
        protocol=PROTOCOL,
        lsv2=asdict(search_config(arm)),
        base=BASE,
        decoupling=dict(
            s4_decouple.decoupling(s4_decouple.TERMINAL), terminal_share=terminal_share(arm)
        ),
        moead=dict(
            ls_enabled=opts.ls_enabled,
            terminal_ops=list(opts.terminal_ops),
            refine=opts.refine,
            seed_mode=opts.seed_mode,
            regions=opts.regions,
            tau_seconds=opts.tau_seconds,
            wide_budget=opts.wide_budget,
        ),
    )
    if opts.trace:
        # Record-only; the nine-arm records stay byte-identical.
        record["moead"].update(trace=True, trace_seconds=opts.trace_seconds)
    if opts.yield_measure != "count":
        record["moead"].update(yield_measure=opts.yield_measure)
    if opts.static_wide:
        record["moead"].update(static_wide=list(opts.static_wide))
    if opts.share_signal != "average":
        record["moead"].update(share_signal=opts.share_signal)
    if opts.continuation != "fixed":
        record["moead"].update(continuation=opts.continuation, max_steps=opts.max_steps)
    if opts.share_mode != "fixed":
        record["moead"].update(
            share_mode=opts.share_mode,
            share_bounds=list(opts.share_bounds),
            share_exponent=opts.share_exponent,
        )
    guided = guided_options(arm)
    if opts.terminal_gate is not None:
        record["terminal_gate"] = dict(
            name=opts.terminal_gate, **asdict(POLICIES[opts.terminal_gate])
        )
    if guided is not None:
        # Only guided arms carry this key; every other arm's record is unchanged.
        record["guided"] = guided.record()
    if arm in _R1:
        record["r1"] = asdict(_R1[arm])
    if arm in _GAP:
        record["gap_allocation"] = asdict(_GAP[arm])
    return record


def run_arm(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    calibration: dict,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    search_sink: Callable[[LSV2Search], None] | None = None,
    guided_timer: Callable[[], float] | None = None,
    r1_sink: Callable[[RegionalBlockMixer], None] | None = None,
    gap_sink: Callable[[GapAllocator], None] | None = None,
) -> RunResult:
    """Charge initialization, learning, refinement and the terminal sweep to one run.

    For ``M5_P2B`` every offspring slot first sets the gateway refine class from
    the processed subproblem's Bill weight, so the decoder is a deterministic
    function of the genotype and the class. ``guided_timer`` replaces the private clock
    of the guided-variation recorder/controller (tests inject a deterministic clock).
    """
    search = make_search(arm, config, calibration)
    if search_sink is not None:
        search_sink(search)
    lambdas = weights(config.population)
    weighted = options(arm).refine == "weighted"

    def observe(event: str, data: dict) -> None:
        if weighted and event == "before_offspring":
            gateway = data["gateway"]
            if gateway.refine is not None:
                gateway.refine_class = weight_class(lambdas[data["subproblem"]][1])
        if observer is not None:
            observer(event, data)

    assert isinstance(search, MoeadMixin)
    guided_arm = guided_options(arm)
    guided = None
    if guided_arm is not None:
        guided = GuidedVariation(
            config,
            guided_arm,
            gain_probe=search.last_gain,
            terminal_probe=search.in_terminal,
            timer=guided_timer or DEFAULT_CLOCK,
        )
        search.guided = guided
    mixer = RegionalBlockMixer(problem, _R1[arm]) if arm in _R1 else None
    if mixer is not None and r1_sink is not None:
        r1_sink(mixer)
    allocator = GapAllocator(_GAP[arm]) if arm in _GAP else None
    if allocator is not None and gap_sink is not None:
        gap_sink(allocator)
    return run_policy(
        problem,
        "BILL_PERMUTED",
        config,
        evidence,
        observer=observe if (weighted or observer is not None) else None,
        interleave=search,
        guided_offspring=guided,
        ordinary_path=guided.ordinary if guided is not None and guided.options.timed else None,
        post_replacement=mixer if mixer is not None else allocator,
        traversal=allocator.order if allocator is not None else None,
    )


__all__ = [
    "ARMS",
    "BASE",
    "COMBO_ARMS",
    "NEXT_ARMS",
    "BANDIT_ARMS",
    "SHARE_ARMS",
    "WIDEN_ARMS",
    "SPACE_ARMS",
    "INVEST_ARMS",
    "GUIDED_ARMS",
    "COST_ARMS",
    "EXTRA_ARMS",
    "NO_LS",
    "PROTOCOL",
    "TIMING_OPS",
    "arm_config",
    "configuration_record",
    "guided_options",
    "make_search",
    "options",
    "run_arm",
    "search_config",
    "terminal_share",
]
