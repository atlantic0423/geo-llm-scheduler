"""Load-dependent power: hand integrals, legacy identity, ledger equality, scoring."""

import json
import math
import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.config import Config
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
from geo_llm_scheduler.evaluation.exact import evaluate, power_segments
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.io.validation import validate_problem
from geo_llm_scheduler.lsv2.billing import PowerProfile
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.group_polish import _Search
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.scheduling.timing import legal_bounds, move

LOAD = PowerModel(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)


def test_hand_integrated_load_power(problem):
    p = replace(problem, power_model=LOAD)
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    e = evaluate(p, g, Schedule((0, 100, 0, 100)))
    assert e.feasible
    # m0: two Prefills (compute 0.5 each) on [0,100), two Decodes on [100,300).
    # Floor 0.4*90 = 36 kW; Prefill load 0.6*90*1*0.5 = 27 kW each;
    # Decode load 0.6*90*0.25*0.5 = 6.75 kW each.
    prefill_kw, decode_kw = 10 + 36 + 2 * 27, 10 + 36 + 2 * 6.75
    assert prefill_kw == pytest.approx(100)  # full Compute reaches active power exactly
    r0 = [(0, 100, prefill_kw + 10), (100, 300, decode_kw + 10)]
    assert [s[2] for s in power_segments(p, g, Schedule((0, 100, 0, 100)), 0)] == pytest.approx(
        [r0[0][2], r0[1][2]]
    )
    tou_r0 = sum((b - a) * kw for a, b, kw in r0) * 0.2 / 3600
    tou_r1 = 10 * 300 * 0.1 / 3600
    assert e.tou == pytest.approx(tou_r0 + tou_r1)
    window = sum((b - a) * kw for a, b, kw in r0) / 900
    assert e.windows[0] == pytest.approx((window,))
    assert e.demand == pytest.approx((2 * window, 3 * 10 * 300 / 900))
    legacy = evaluate(problem, g, Schedule((0, 100, 0, 100)))
    assert e.bill < legacy.bill


def test_power_never_exceeds_active_and_floor_when_light():
    job = Job("a", 0, Profile(100, 0.2, 1), Profile(100, 0.2, 1))
    region = Region("r", (Tariff(0, 1000, 1.0),), 0)
    p = ProblemInstance((job,), (region,), (ServingInstance("m", 0, 8, 10, 100),), LOAD)
    seg = power_segments(p, Genotype((0, 0), (0, 0)), Schedule((0, 100)), 0)
    assert seg[0][2] == pytest.approx(10 + 36 + 0.6 * 90 * 0.2)
    assert seg[1][2] == pytest.approx(10 + 36 + 0.6 * 90 * 0.25 * 0.2)
    assert all(kw <= 100 for _, _, kw in seg)


@pytest.fixture(scope="module")
def cases():
    base = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    out = []
    for g in initial_genotypes(base, Config(seed=3), random.Random(3))[:6]:
        out.append((g, decode(base, g)))
    return base, out


def test_beta_one_is_bit_identical_to_two_state_model(cases):
    base, members = cases
    other = replace(base, power_model=PowerModel(1.0, 0.3, 0.7))
    for g, s in members:
        assert evaluate(other, g, s) == evaluate(base, g, s)
        assert (
            PowerProfile(other, g.ms, s.starts).bill() == PowerProfile(base, g.ms, s.starts).bill()
        )


def test_ledger_bill_and_deltas_equal_exact_under_load_power(cases):
    base, members = cases
    problem = replace(base, power_model=LOAD)
    rng = random.Random(11)
    checked = 0
    for g, s in members:
        e = evaluate(problem, g, s)
        assert e.bill != evaluate(base, g, s).bill
        profile = PowerProfile(problem, g.ms, s.starts)
        assert math.isclose(profile.bill(), e.bill, rel_tol=1e-12)
        for _ in range(300):
            o = 2 * rng.randrange(len(problem.jobs))
            lo, hi = legal_bounds(problem, g, s, o)
            if hi <= lo:
                continue
            t = lo + (hi - lo) * rng.random()
            moved = move(problem, g, s, o, t)
            if moved is None:
                continue
            change = {g.ms[o]: {o: (t, t + problem.profile(o).duration)}}
            exact = evaluate(problem, g, moved)
            assert math.isclose(e.bill + profile.delta_for(change), exact.bill, rel_tol=1e-9)
            region = problem.instances[g.ms[o]].region
            assert profile.window_values(change, region) == pytest.approx(exact.windows[region])
            copy = profile.copy()
            copy.apply(change)
            assert math.isclose(copy.bill(), exact.bill, rel_tol=1e-9)
            checked += 1
    assert checked >= 15


def _busy_decode_case(model: PowerModel) -> tuple[ProblemInstance, Candidate]:
    """Job b's Prefill sits inside job a's long Decode; tariff drops at t=200."""
    jobs = (
        Job("a", 0, Profile(100, 0.5, 1), Profile(1000, 0.5, 1)),
        Job("b", 0, Profile(100, 0.5, 1), Profile(100, 0.5, 1)),
    )
    region = Region("r", (Tariff(0, 200, 0.4), Tariff(200, 5000, 0.1)), 0)
    p = ProblemInstance(jobs, (region,), (ServingInstance("m", 0, 8, 10, 100),), model)
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = Schedule((0, 100, 100, 900))
    e = evaluate(p, g, s)
    assert e.feasible
    return p, Candidate(g, s, e, "test")


def test_union_unchanged_move_is_scored_only_with_load_power():
    p, c = _busy_decode_case(PowerModel())
    legacy = _Search(p, c)
    legacy.offer({2: 300.0}, "single")
    assert legacy.stats["union_unchanged"] == 1 and not legacy.pool
    p, c = _busy_decode_case(LOAD)
    search = _Search(p, c)
    search.offer({2: 300.0}, "single")
    assert search.stats["union_unchanged"] == 0
    [(starts, (delta, _))] = search.pool.items()
    exact = evaluate(p, c.genotype, Schedule(starts))
    assert exact.feasible and exact.flow == c.evaluation.flow
    assert delta == pytest.approx(exact.bill - c.evaluation.bill) and delta < 0
    batch = FlowSafeBillMoves(right_only=True).propose(p, c, 1, Config(seed=1), random.Random(1))
    assert batch.proposals
    best = evaluate(p, batch.proposals[0].genotype, batch.proposals[0].schedule)
    assert best.bill < c.evaluation.bill and best.flow == c.evaluation.flow


def test_power_model_validation_and_json_roundtrip(tmp_path, problem):
    with pytest.raises(ValueError, match="beta"):
        validate_problem(replace(problem, power_model=PowerModel(beta=1.2)))
    with pytest.raises(ValueError, match="beta"):
        validate_problem(replace(problem, power_model=PowerModel(kappa_decode=-0.1)))
    legacy_path, load_path = tmp_path / "legacy.json", tmp_path / "load.json"
    save_instance(problem, legacy_path)
    assert "power_model" not in json.loads(legacy_path.read_text(encoding="utf-8"))
    assert load_instance(legacy_path).power_model == PowerModel()
    loaded = replace(problem, power_model=LOAD)
    save_instance(loaded, load_path)
    assert load_instance(load_path) == loaded
    data = json.loads(load_path.read_text(encoding="utf-8"))
    del data["power_model"]["kappa_decode"]
    load_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="power_model"):
        load_instance(load_path)


@pytest.mark.parametrize("accelerated", [False, True])
def test_bill4_prefix_ledger_matches_exact_under_load_power(cases, accelerated):
    from geo_llm_scheduler.experiments.initial_bill_mixture import guided_genotype

    base, _ = cases
    problem = replace(base, power_model=LOAD)
    for family in ("TOU", "DEMAND", "BILL"):
        g, evidence = guided_genotype(
            problem, family, random.Random(5), 0.25, accelerated=accelerated
        )
        e = evaluate(problem, g, decode(problem, g))
        assert math.isclose(evidence["prefix_tou"], e.tou, rel_tol=1e-10, abs_tol=1e-8)
        assert math.isclose(evidence["prefix_demand"], sum(e.demand), rel_tol=1e-10, abs_tol=1e-8)
