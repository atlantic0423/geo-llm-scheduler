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
"""

from collections.abc import Callable
from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments import s4_decouple
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.lsv2.decoder_refine import weight_class
from geo_llm_scheduler.lsv2.moead_search import MoeadMixin, MoeadOptions, with_moead
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search
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
EXTRA_ARMS = ARMS + COMBO_ARMS + NEXT_ARMS
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
}
_TERMINAL = {"M3_P1_T10": 0.10}


def _check(arm: str) -> None:
    if arm not in EXTRA_ARMS:
        raise ValueError("Unknown S4-B2 MOEA/D arm")


def options(arm: str) -> MoeadOptions:
    """Declared MOEA/D-facing switches of one arm."""
    _check(arm)
    return _OPTIONS[arm]


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
) -> RunResult:
    """Charge initialization, learning, refinement and the terminal sweep to one run.

    For ``M5_P2B`` every offspring slot first sets the gateway refine class from
    the processed subproblem's Bill weight, so the decoder is a deterministic
    function of the genotype and the class.
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
    return run_policy(
        problem,
        "BILL_PERMUTED",
        config,
        evidence,
        observer=observe if (weighted or observer is not None) else None,
        interleave=search,
    )


__all__ = [
    "ARMS",
    "BASE",
    "COMBO_ARMS",
    "NEXT_ARMS",
    "EXTRA_ARMS",
    "NO_LS",
    "PROTOCOL",
    "TIMING_OPS",
    "arm_config",
    "configuration_record",
    "make_search",
    "options",
    "run_arm",
    "search_config",
    "terminal_share",
]
