"""F3 cost control (2026-11-11): fixed destroy count, switch controller, path records, batch."""

import importlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest
from test_guided_variation import POWER, _Clock, _population
from test_lsv2_fast_rightshift import CAL
from test_s4_load_shift import _panel

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.mechanism_instances import mechanism_problem, panel_specs
from geo_llm_scheduler.experiments.trans_runner import digest
from geo_llm_scheduler.initialization.generators import generate
from geo_llm_scheduler.lsv2 import guided_variation as gv
from geo_llm_scheduler.lsv2.guided_variation import GuidedOptions, GuidedVariation, PathRates
from geo_llm_scheduler.moead.core import weights
from geo_llm_scheduler.utils.rng import RNGManager


def test_cost_options_validate_and_record_only_when_set():
    for bad in (
        dict(destroy_count=0),
        dict(controller="other"),
        dict(controller="switch", p_probe=0.5),  # above probability
        dict(gamma=0.0),
        dict(tau_seconds=0.0),
        dict(trace_seconds=0.0),
    ):
        with pytest.raises(ValueError):
            GuidedOptions(**bad)
    plain = GuidedOptions()
    assert plain.record() == dict(
        probability=0.2, direction_aware=True, neighbor_paths=False, neighbor_extra=2
    )
    assert not plain.timed
    switch = GuidedOptions(direction_aware=False, controller="switch", record_paths=True)
    assert switch.timed and set(switch.record()) == set(gv.BASE_OPTION_FIELDS) | set(
        gv.EXTRA_OPTION_DEFAULTS
    )
    assert s4_moead.COST_ARMS == (
        "G8_GUIDED_RAND",
        "G4_GUIDED_RAND_D4",
        "GS_GUIDED_RAND_SWITCH",
        "GS4_GUIDED_RAND_SWITCH_D4",
    )
    g8, g4, gs = (s4_moead.guided_options(a) for a in s4_moead.COST_ARMS[:3])
    assert g8 == GuidedOptions(direction_aware=False, record_paths=True)
    assert g4.destroy_count == 4 and g4.controller == "fixed" and not g4.direction_aware
    assert gs.controller == "switch" and gs.probability == 0.2 and gs.p_probe == 0.02
    assert (gs.tau_seconds, gs.gamma, gs.warmup_seconds) == (300.0, 2.0, 5.0)


def _opts(**kw):
    return GuidedOptions(direction_aware=False, controller="switch", **kw)


def test_switch_probability_warmup_boundaries_bounds_and_decay():
    o = _opts()
    r = PathRates(300.0)
    assert r.probability(0.0, o)[0] == 0.02  # nothing observed: warm-up
    r.observe("g", 4.0, 1e-3, 1.0)  # guided has only 4 s: still warm-up
    r.observe("o", 20.0, 2e-3, 1.0)
    assert r.probability(1.0, o)[0] == 0.02
    r.observe("g", 6.0, 1e-3, 2.0)  # now both have at least 5 s; ratio (2e-3/10) / (2e-3/20) = 2
    p, rho_g, rho_o = r.probability(2.0, o)
    assert rho_g > rho_o and p == 0.2  # 0.2 * 4 clipped to p_on
    # equal-rate boundary: ratio 1 gives p_on; ratio 0.5 gives 0.2 * 0.25 = 0.05
    even = PathRates(300.0)
    even.observe("g", 10.0, 1e-3, 1.0)
    even.observe("o", 10.0, 1e-3, 1.0)
    assert even.probability(1.0, o)[0] == pytest.approx(0.2)
    half = PathRates(300.0)
    half.observe("g", 10.0, 1e-3, 1.0)
    half.observe("o", 10.0, 2e-3, 1.0)
    assert half.probability(1.0, o)[0] == pytest.approx(0.05)
    low = PathRates(300.0)  # ratio 0.25 -> 0.0125 below the probe floor
    low.observe("g", 10.0, 1e-3, 1.0)
    low.observe("o", 10.0, 4e-3, 1.0)
    assert low.probability(1.0, o)[0] == 0.02
    # zero guided rate -> probe; zero ordinary rate with positive guided rate -> p_on
    none_g = PathRates(300.0)
    none_g.observe("g", 10.0, 0.0, 1.0)
    none_g.observe("o", 10.0, 1e-3, 1.0)
    assert none_g.probability(1.0, o)[0] == 0.02
    none_o = PathRates(300.0)
    none_o.observe("g", 10.0, 1e-3, 1.0)
    none_o.observe("o", 10.0, 0.0, 1.0)
    assert none_o.probability(1.0, o)[0] == 0.2
    # decay shrinks both seconds equally: a long silence returns to warm-up, rates unchanged
    q = even.probability(1.0 + 3000.0, o)
    assert q[0] == 0.02 and q[1] == pytest.approx(1e-4) and q[2] == pytest.approx(1e-4)
    # p_probe == p_on makes the controller constant
    const = _opts(p_probe=0.2)
    assert half.probability(1.0, const)[0] == 0.2 and low.probability(1.0, const)[0] == 0.2


def test_controller_probability_is_used_for_routing_and_terminal_gains_are_ignored():
    problem, gateway, members = _population(6, 6)
    now = [0.0]
    flag = [False]
    guided = GuidedVariation(
        gv.Config(seconds=100.0),
        _opts(record_paths=True),
        gain_probe=lambda: 0.5,
        terminal_probe=lambda: flag[0],
        timer=lambda: now[0],
    )
    assert guided.routing_probability(0.0) == 0.02  # warm-up
    guided._observe("g", 10.0, 1e-3, 1, 10.0, True)
    guided._observe("o", 10.0, 1e-3, 1, 10.0, True)
    assert guided.routing_probability(10.0) == pytest.approx(0.2)
    flag[0] = True  # terminal sweep: no more updates to the controller, bins still count
    seconds_before = dict(guided.rates.seconds)
    guided._observe("o", 99.0, 5.0, 3, 99.0, True)
    assert guided.rates.seconds == seconds_before
    assert guided.ordinary_bins[9].hv_gain == 5.0 and guided.ordinary_bins[1].slots == 1
    assert sorted(guided.trace) == [0, 0.0 == 0 and 0][:1]  # both points fall in 30 s bin 0
    assert guided.summary()["controller"]["trace_seconds"] == 30.0


@pytest.mark.parametrize("n", [50, 100, 200])
def test_fixed_destroy_count_is_exactly_four_at_every_size(n):
    problem = replace(mechanism_problem(panel_specs((n,))[0]), power_model=POWER)
    gateway = EvaluationGateway(problem, Archive())
    rng = Random(3)
    members = [gateway.evaluate(generate(problem, m, rng, 0.1)) for m in (0, 3, 7, 9)]
    for options, expect in (
        (
            GuidedOptions(
                probability=1.0, direction_aware=False, destroy_count=4, record_paths=True
            ),
            4,
        ),
        (
            GuidedOptions(probability=1.0, direction_aware=False, record_paths=True),
            max(2, round(0.08 * n)),
        ),
    ):
        guided = GuidedVariation(gv.Config(), options)
        streams = RNGManager(5)
        for k in range(8):
            guided(members, k % 4, (0, 1, 2, 3), weights(4), gateway, streams)
        summary = guided.summary()
        assert summary["successes"] > 0
        assert (
            sum(c["destroyed_jobs"] for c in summary["cells"].values())
            == expect * summary["successes"]
        )


def test_path_records_deltas_and_deciles_are_consistent():
    problem = replace(mechanism_problem(panel_specs((50,))[1]), power_model=POWER)
    gateway = EvaluationGateway(problem, Archive())
    rng = Random(4)
    members = [gateway.evaluate(generate(problem, m, rng, 0.1)) for m in (0, 3, 5, 7)]
    ticks = iter(range(10_000))
    guided = GuidedVariation(
        gv.Config(seconds=100.0),
        GuidedOptions(probability=1.0, direction_aware=False, record_paths=True),
        timer=lambda: next(ticks) * 0.01,
    )
    streams = RNGManager(8)
    for k in range(10):
        guided(members, k % 4, (0, 1, 2, 3), weights(4), gateway, streams)
    for k in range(5):  # ordinary slots are timed through the engine seam
        guided.ordinary(lambda: gateway.evaluate(generate(problem, 1, rng, 0.1)))
    s = guided.summary()
    inc = [m.evaluation for m in members]
    assert s["successes"] > 0 and set(s["options"]) >= set(gv.EXTRA_OPTION_DEFAULTS)
    total = {k: sum(c[k] for c in s["cells"].values()) for k in gv.EXTRA_CELL_FIELDS}
    assert total["destroyed_jobs"] > 0 and total["decoded_flow_delta"] != total["plan_flow_delta"]
    assert all(v is not None for v in total.values()) and len(inc) == 4
    g, o = s["deciles"]["guided"], s["deciles"]["ordinary"]
    assert (
        sum(b["slots"] for b in g) == s["attempts"]
        and sum(b["successes"] for b in g) == s["successes"]
    )
    assert sum(b["slots"] for b in o) == 5 and sum(b["seconds"] for b in o) > 0
    json.dumps(s, allow_nan=False)


# ---------------- whole-run behaviour under a deterministic clock ----------------
def _run(monkeypatch, arm, *, guided_timer=None, probe=None, seed=5, seconds=40, generations=3):
    clock = _Clock(0.002)
    monkeypatch.setattr(time, "perf_counter", clock)
    for name, module in tuple(sys.modules.items()):
        if name.startswith("geo_llm_scheduler.") and module is not None:
            if callable(getattr(module, "perf_counter", None)):
                monkeypatch.setattr(module, "perf_counter", clock)
    problem = replace(mechanism_problem(panel_specs((50,))[0]), power_model=POWER)
    config = replace(s4_moead.arm_config(arm, seed, seconds), generations=generations)
    searches = []

    def sink(search):
        search.timer = clock
        searches.append(search)

    if probe is not None:
        monkeypatch.setitem(s4_moead._GUIDED, arm, replace(s4_moead._GUIDED[arm], **probe))
    result = s4_moead.run_arm(
        problem,
        arm,
        config,
        CAL,
        {},
        search_sink=sink,
        guided_timer=clock if guided_timer else None,
    )
    return result, searches[0]


def _sig(result, search):
    return (
        [(c.genotype, c.schedule, c.evaluation) for c in result.gateway.archive.members],
        dict(result.gateway.counts),
        search.steps,
        search.moead_counts,
    )


def test_recording_and_a_constant_controller_leave_v2_bit_identical(monkeypatch):
    v2 = _sig(*_run(monkeypatch, "V2_GUIDED_RAND", probe=dict(probability=0.5)))
    assert v2[1]["exact"] > 100 and v2[2]
    # path timing/recording (G8) changes no result
    g8 = _sig(*_run(monkeypatch, "G8_GUIDED_RAND", probe=dict(probability=0.5)))
    assert g8 == v2
    # the switch controller with p_probe == p_on is the constant-probability arm
    gs = _sig(*_run(monkeypatch, "GS_GUIDED_RAND_SWITCH", probe=dict(probability=0.5, p_probe=0.5)))
    assert gs == v2


@pytest.mark.parametrize("arm", ["G4_GUIDED_RAND_D4", "GS_GUIDED_RAND_SWITCH"])
def test_cost_arms_repeat_exactly_and_record_both_paths(monkeypatch, arm):
    first = _run(monkeypatch, arm, guided_timer=True, probe=dict(probability=0.5, p_probe=0.05))
    second = _run(monkeypatch, arm, guided_timer=True, probe=dict(probability=0.5, p_probe=0.05))
    assert _sig(*first) == _sig(*second)
    f3 = first[1].moead_summary()["f3"]
    assert f3["successes"] > 0
    ordinary = sum(b["slots"] for b in f3["deciles"]["ordinary"])
    assert ordinary + f3["successes"] == f3["slots"]  # every slot is exactly one path
    if arm.startswith("G4"):
        assert sum(c["destroyed_jobs"] for c in f3["cells"].values()) == 4 * f3["successes"]
    else:
        probabilities = [v[0] for v in f3["controller"]["trace"].values()]
        assert probabilities and all(0.05 <= p <= 0.5 for p in probabilities)


# ---------------- batch entry ----------------
@pytest.fixture
def cost_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_cost")


def test_batch_pools_both_sizes_with_their_own_arms_and_a_reduced_matrix(
    tmp_path, monkeypatch, cost_batch
):
    b = cost_batch
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(b.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(b.base, "CAL", digest(cal))
    assert b.SEEDS == (14, 15) and b.CONTROL == "N7_P1_SHARE" and b.DESIGN == "s4-cost/v1"
    assert b.ARMS_BY_SIZE[100] == (b.CONTROL, "G4_GUIDED_RAND_D4", "GS_GUIDED_RAND_SWITCH")
    assert b.ARMS_BY_SIZE[200] == (
        b.CONTROL,
        "G8_GUIDED_RAND",
        "G4_GUIDED_RAND_D4",
        "GS_GUIDED_RAND_SWITCH",
    )
    assert b.SELECTION == {100: (2, (14, 15)), 200: (4, (14,))}
    seen, runs, solver = [], 0, 0
    for k in range(3):
        root = tmp_path / f"root{k}"
        b.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = b.validate_inputs(root)
        assert tuple(m["arms"]) == b.ARMS and m["design"] == b.DESIGN
        runs += m["runs"]
        solver += m["solver_seconds"]
        for block in m["blocks"]:
            spec = json.loads((root / "specs" / (block + ".json")).read_text())
            arms = b.ARMS_BY_SIZE[spec["n"]]
            assert spec["seconds"] == {100: 1200, 200: 2400}[spec["n"]]
            assert set(spec["order"]) == set(arms) and set(spec["config"]) == set(arms)
            replicate = int(spec["instance"].split("_r")[-1][:2])
            assert (
                replicate <= b.SELECTION[spec["n"]][0] and spec["seed"] in b.SELECTION[spec["n"]][1]
            )
            assert "run_s4_cost.py" in spec["entry"]
            assert "guided" not in spec["config"]["N7_P1_SHARE"]
            assert spec["config"]["G4_GUIDED_RAND_D4"]["guided"]["destroy_count"] == 4
            assert spec["config"]["GS_GUIDED_RAND_SWITCH"]["guided"]["controller"] == "switch"
            assert ("G8_GUIDED_RAND" in spec["config"]) == (spec["n"] == 200)
        seen += m["blocks"]
    assert len(seen) == len(set(seen)) == 80
    assert Counter(int(x.split("_")[1][1:]) for x in seen) == {100: 40, 200: 40}
    assert runs == 40 * 4 + 40 * 3 and solver == 40 * 4 * 2400 + 40 * 3 * 1200 == 528_000
