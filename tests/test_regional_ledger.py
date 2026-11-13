"""Regional ledgers: exact agreement, splice prediction, tail/window boundaries."""

import importlib.util
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest

from geo_llm_scheduler.domain.models import (
    Candidate,
    Genotype,
    Job,
    PowerModel,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.evaluation.regional import (
    combine,
    evaluate_with_ledgers,
    region_ledgers,
    regional_bill,
    splice,
    tail_windows,
    tariff_indexes,
)
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.numeric import TOL, close

spec = importlib.util.spec_from_file_location(
    "r1_offline", Path(__file__).parents[1] / "scripts/r1_offline.py"
)
assert spec is not None and spec.loader is not None
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)


def _problem(beta: float = 1.0, jobs: int = 8) -> ProblemInstance:
    # Tariff boundaries are deliberately off the 900-second grid.
    tariffs = (Tariff(0, 700, 0.2), Tariff(700, 2000, 0.9), Tariff(2000, 100000, 0.4))
    return ProblemInstance(
        tuple(
            Job(str(i), 0 if i < 4 else 650, Profile(150 + 40 * i, 0.6, 2), Profile(300, 0.5, 2))
            for i in range(jobs)
        ),
        tuple(Region(str(r), tariffs, 2 + r) for r in range(3)),
        tuple(ServingInstance(str(m), m // 2, 8, 10 + m, 100) for m in range(6)),
        PowerModel(beta, 0.8, 0.6),
    )


def _random_genotype(p: ProblemInstance, rng: random.Random, regions: list[int]) -> Genotype:
    os = [j for j in range(len(p.jobs)) for _ in range(2)]
    rng.shuffle(os)
    ms = tuple(2 * regions[j] + rng.randrange(2) for j in range(len(p.jobs)) for _ in range(2))
    return Genotype(ms, tuple(os))


def _candidate(p: ProblemInstance, g: Genotype) -> Candidate:
    schedule = decode(p, g)
    return Candidate(g, schedule, evaluate(p, g, schedule))


def _assert_matches(p: ProblemInstance, prediction, exact) -> None:
    assert close(prediction.flow, exact.flow, TOL.time)
    assert close(prediction.tou, exact.tou, TOL.cost)
    for a, b in zip(prediction.demand, exact.demand):
        assert close(a, b, TOL.cost)
    assert close(prediction.bill, exact.bill, TOL.cost)


@pytest.mark.parametrize("beta", [1.0, 0.3])
def test_own_ledgers_reproduce_exact_objectives_and_windows(beta):
    p = _problem(beta)
    indexes = tariff_indexes(p)
    rng = random.Random(20261113)
    for _ in range(40):
        regions = [rng.randrange(3) for _ in p.jobs]
        c = _candidate(p, _random_genotype(p, rng, regions))
        ledgers = region_ledgers(p, c.genotype, c.schedule)
        prediction = combine(p, indexes, ledgers)
        assert prediction.horizon == c.schedule.horizon(p)
        _assert_matches(p, prediction, c.evaluation)
        for g, windows in zip(ledgers, c.evaluation.windows):
            # Core terms are added in the reference order, so windows are bitwise equal.
            assert tail_windows(g, prediction.horizon) == windows


@pytest.mark.parametrize("beta", [1.0, 0.3])
def test_spliced_prediction_matches_decoded_exact_including_horizon_changes(beta):
    p = _problem(beta)
    indexes = tariff_indexes(p)
    rng = random.Random(912)
    horizon_changes = 0
    other_region_shift = 0
    for _ in range(60):
        regions = [rng.randrange(3) for _ in p.jobs]
        donor, recipient = (_candidate(p, _random_genotype(p, rng, regions)) for _ in range(2))
        d_led = region_ledgers(p, donor.genotype, donor.schedule)
        r_led = region_ledgers(p, recipient.genotype, recipient.schedule)
        for region in range(3):
            if not d_led[region].jobs:
                with pytest.raises(ValueError, match="equal nonempty"):
                    splice(r_led, d_led, region)
                continue
            prediction = combine(p, indexes, splice(r_led, d_led, region))
            genotype, _ = r1.mix_region(p, donor, recipient, region)
            mixed = _candidate(p, genotype)
            assert mixed.evaluation.feasible
            _assert_matches(p, prediction, mixed.evaluation)
            if not close(prediction.horizon, recipient.schedule.horizon(p), TOL.time):
                horizon_changes += 1
                untouched = [q for q in range(3) if q != region]
                if any(
                    not close(mixed.evaluation.demand[q], recipient.evaluation.demand[q], TOL.cost)
                    for q in untouched
                ):
                    other_region_shift += 1
    # The fixture must exercise the global-horizon coupling, not only easy cases.
    assert horizon_changes > 10
    assert other_region_shift > 0


def test_hand_computed_idle_tail_in_untouched_empty_region():
    p = ProblemInstance(
        (Job("a", 0, Profile(450, 1, 1), Profile(450, 1, 1)),),
        (
            Region("a", (Tariff(0, 900, 0.2), Tariff(900, 10000, 0.7)), 2),
            Region("empty", (Tariff(0, 900, 0.2), Tariff(900, 10000, 0.7)), 3),
        ),
        (ServingInstance("a", 0, 8, 10, 100), ServingInstance("empty", 1, 8, 20, 100)),
    )
    indexes = tariff_indexes(p)
    g = Genotype((0, 0), (0, 0))
    early = region_ledgers(p, g, Schedule((0, 450)))
    late = region_ledgers(p, g, Schedule((0, 900)))
    empty = early[1]
    assert empty.jobs == frozenset() and empty.end == 0 and empty.core_windows == ()
    tou_900, demand_900 = regional_bill(p, indexes[1], empty, 900)
    tou_1350, demand_1350 = regional_bill(p, indexes[1], empty, 1350)
    assert tou_900 == pytest.approx(20 * 900 * 0.2 / 3600)
    assert tou_1350 - tou_900 == pytest.approx(20 * 450 * 0.7 / 3600)
    # Fixed divisor: the partial tail window averages 10 kW, the full one 20 kW.
    assert tail_windows(empty, 1350) == (20, 10)
    assert demand_900 == demand_1350 == pytest.approx(3 * 20)
    assert combine(p, indexes, late).horizon == 1350
    _assert_matches(p, combine(p, indexes, late), evaluate(p, g, Schedule((0, 900))))


def test_window_boundaries_and_peaks_match_full_windows():
    p = _problem()
    indexes = tariff_indexes(p)
    rng = random.Random(5)
    c = _candidate(p, _random_genotype(p, rng, [0] * 4 + [1] * 4))
    for g in region_ledgers(p, c.genotype, c.schedule):
        grid = math.ceil(g.end / 900) * 900
        for horizon in {g.end, grid, grid + 1, grid + 450, grid + 900, grid + 2700, grid + 2701}:
            if horizon < g.end:
                continue
            _, demand = regional_bill(p, indexes[g.region], g, horizon)
            peak = max(tail_windows(g, horizon), default=0.0)
            assert demand == pytest.approx(p.regions[g.region].demand_rate * peak, abs=1e-12)


@pytest.mark.parametrize("beta", [1.0, 0.3])
def test_fused_evaluation_equals_reference_exactly(beta):
    p = _problem(beta)
    rng = random.Random(77)
    for _ in range(30):
        regions = [rng.randrange(3) for _ in p.jobs]
        g = _random_genotype(p, rng, regions)
        s = decode(p, g)
        result, ledgers = evaluate_with_ledgers(p, g, s)
        assert result == evaluate(p, g, s)
        assert ledgers == region_ledgers(p, g, s)
    # Infeasible: identical violation result and no ledgers.
    shifted = Schedule(tuple(t - 1 for t in s.starts))
    result, ledgers = evaluate_with_ledgers(p, g, shifted)
    assert result == evaluate(p, g, shifted) and not result.feasible and ledgers == ()
    short = ProblemInstance(
        p.jobs, tuple(Region(r.name, (Tariff(0, 10, 0.1),), 1) for r in p.regions), p.instances
    )
    with pytest.raises(ValueError, match="coverage"):
        evaluate_with_ledgers(short, g, s)


def test_invalid_inputs_are_rejected():
    p = _problem()
    indexes = tariff_indexes(p)
    rng = random.Random(1)
    c = _candidate(p, _random_genotype(p, rng, [0] * 8))
    ledgers = region_ledgers(p, c.genotype, c.schedule)
    with pytest.raises(ValueError, match="precedes"):
        regional_bill(p, indexes[0], ledgers[0], ledgers[0].end - 1)
    with pytest.raises(ValueError, match="precedes"):
        tail_windows(ledgers[0], ledgers[0].end - 1)
    with pytest.raises(ValueError, match="coverage"):
        regional_bill(p, indexes[1], ledgers[1], 1e6)
    with pytest.raises(ValueError, match="ordered"):
        combine(p, indexes, tuple(reversed(ledgers)))
    with pytest.raises(ValueError, match="partition"):
        combine(p, indexes, (ledgers[0], replace(ledgers[1], jobs=ledgers[0].jobs), ledgers[2]))
    bad = Genotype((0, 2) + c.genotype.ms[2:], c.genotype.os)
    with pytest.raises(ValueError, match="P/D"):
        region_ledgers(p, bad, c.schedule)
    assert ledgers[0].same_core(ledgers[0]) and not ledgers[0].same_core(ledgers[1])
    assert indexes[0].integral(800, 800) == 0.0
    assert indexes[0].integral(0, 2100) == pytest.approx(700 * 0.2 + 1300 * 0.9 + 100 * 0.4)
