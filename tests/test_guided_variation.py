"""F3 guided variation (2026-11-10): routing, direction modes, fallbacks, accounting, batch."""

import importlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic
from test_s4_load_shift import _panel

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import PowerModel
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.mechanism_instances import mechanism_problem, panel_specs
from geo_llm_scheduler.experiments.trans_runner import digest
from geo_llm_scheduler.lsv2 import guided_variation as gv
from geo_llm_scheduler.lsv2.destroy_repair import TargetedDestroyRepair
from geo_llm_scheduler.lsv2.guided_variation import GuidedOptions, GuidedVariation
from geo_llm_scheduler.moead.core import neighborhoods, weights
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.base import ProposalBatch
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, preference
from geo_llm_scheduler.utils.rng import RNGManager

POWER = PowerModel(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)


def _population(n_jobs=6, size=6, seed=3):
    """A population of exact-evaluated random genotypes on a one-Region synthetic input."""
    from geo_llm_scheduler.initialization.generators import generate

    problem, _ = synthetic(n_jobs)
    gateway = EvaluationGateway(problem, Archive())
    rng = Random(seed)
    members = [gateway.evaluate(generate(problem, 7, rng, 0.1)) for _ in range(size)]
    return problem, gateway, members


def _guided(config=None, **options):
    return GuidedVariation(config or Config(), GuidedOptions(**options))


def test_options_validate_and_arms_are_declared_without_touching_other_records():
    with pytest.raises(ValueError):
        GuidedOptions(probability=1.5)
    with pytest.raises(ValueError):
        GuidedOptions(neighbor_extra=-1)
    assert s4_moead.GUIDED_ARMS == ("V1_GUIDED_DIR", "V2_GUIDED_RAND", "V3_GUIDED_DIR_NB")
    assert set(s4_moead.GUIDED_ARMS) <= set(s4_moead.EXTRA_ARMS)
    cfg = s4_moead.arm_config("N7_P1_SHARE", 7, 1200)
    control = s4_moead.configuration_record("N7_P1_SHARE", cfg, CAL)
    for arm in s4_moead.EXTRA_ARMS:
        record = s4_moead.configuration_record(arm, s4_moead.arm_config(arm, 7, 1200), CAL)
        assert s4_moead.arm_config(arm, 7, 1200) == cfg
        if arm in s4_moead.GUIDED_ARMS + s4_moead.COST_ARMS:
            assert record["guided"] == s4_moead.guided_options(arm).record()
            if arm in s4_moead.GUIDED_ARMS:  # first-batch record bytes unchanged
                assert set(record["guided"]) == set(gv.BASE_OPTION_FIELDS)
            else:
                assert set(record["guided"]) == set(gv.BASE_OPTION_FIELDS) | set(
                    gv.EXTRA_OPTION_DEFAULTS
                )
            rest = {k: v for k, v in record.items() if k not in ("guided", "arm")}
            assert rest == {k: v for k, v in control.items() if k != "arm"}
        else:
            assert "guided" not in record and s4_moead.guided_options(arm) is None
    assert s4_moead.guided_options("V1_GUIDED_DIR") == GuidedOptions()
    assert s4_moead.guided_options("V2_GUIDED_RAND").direction_aware is False
    assert s4_moead.guided_options("V3_GUIDED_DIR_NB").neighbor_paths is True
    assert s4_moead.guided_options("V1_GUIDED_DIR").probability == 0.2


def test_direction_modes_follow_the_preference_thirds():
    direction = _guided()
    rng = Random(1)
    modes = {
        i: {direction.mode_for(preference(i, 100), rng) for _ in range(60)} for i in (0, 33, 50, 99)
    }
    assert modes[99] == {"FLOW"} and modes[50] == {"MIXED"}  # high Flow weight / middle
    assert modes[0] == modes[33] == {"TOU", "PEAK"}  # electricity third mixes both
    assert [PREFERENCE_LABELS[preference(i, 100)] for i in (0, 33, 34, 66, 67, 99)] == [
        "Electricity",
        "Electricity",
        "Balanced",
        "Balanced",
        "Flow",
        "Flow",
    ]
    # Flow weight is the first component of lambda, so the highest index is the Flow end.
    assert weights(100)[99] == (1.0, 0.0)
    neutral = _guided(direction_aware=False)
    assert {neutral.mode_for(2, rng) for _ in range(200)} == set(gv.GUIDED_MODES)


def test_neighbour_paths_are_ranked_deduplicated_and_exclude_the_subproblem():
    problem, _, members = _population(5, 5)
    ms = [list(m.genotype.ms) for m in members]
    for k, pair in enumerate([(0, 1), (0, 1), (2, 3), (4, 4)]):  # members 1..4 for job 0
        ms[k + 1][0], ms[k + 1][1] = pair
    members = [
        replace(m, genotype=replace(m.genotype, ms=tuple(row))) for m, row in zip(members, ms)
    ]
    offered = GuidedVariation.neighbor_paths(members, 0, (0, 1, 2, 3, 4))
    assert offered(0) == [(0, 1), (2, 3), (4, 4)]  # shared first, then nearest first
    assert GuidedVariation.neighbor_paths(members, 1, (1,))(0) == []  # only itself


def test_extra_paths_are_legal_new_and_capped():
    problem, _, _ = _population(4, 2)
    op = TargetedDestroyRepair(extra_paths=lambda job: [(9, 9), (0, 0), (1, 1), (2, 2), (3, 3)])
    options = paths(problem, 0)
    chosen = [(0, 0)]
    out = op.add_extra(0, options, chosen)
    assert out is chosen and len(chosen) == 3  # two new legal paths; (9, 9) and the repeat skipped
    assert chosen[1:] == [(1, 1), (2, 2)] and all(p in options for p in chosen)
    off = TargetedDestroyRepair()
    assert off.add_extra(0, options, [(0, 0)]) == [(0, 0)]


def test_extra_paths_never_cross_regions_on_a_multi_region_input():
    problem = mechanism_problem(panel_specs((50,))[0])
    legal = paths(problem, 3)
    regions = {problem.instances[p].region for pd in legal for p in pd}
    assert len(regions) > 1
    cross = [(a, b) for a in range(len(problem.instances)) for b in range(len(problem.instances))]
    cross = [
        pd for pd in cross if problem.instances[pd[0]].region != problem.instances[pd[1]].region
    ]
    op = TargetedDestroyRepair(extra_paths=lambda job: cross + legal, extra_limit=2)
    chosen: list[tuple[int, int]] = []
    op.add_extra(3, legal, chosen)
    assert len(chosen) == 2 and all(pd in legal for pd in chosen)


def test_timetable_record_is_opt_in_and_decoded_flow_never_exceeds_it():
    problem = replace(mechanism_problem(panel_specs((50,))[1]), power_model=POWER)
    gateway = EvaluationGateway(problem, Archive())
    from geo_llm_scheduler.initialization.generators import generate

    incumbent = gateway.evaluate(generate(problem, 7, Random(5), 0.1))
    plain = TargetedDestroyRepair(mode="MIXED").propose(problem, incumbent, 1, Config(), Random(2))
    assert plain.instrumentation == {}
    batch = TargetedDestroyRepair(mode="MIXED", record_timetable=True).propose(
        problem, incumbent, 1, Config(), Random(2)
    )
    assert [p.genotype for p in batch.proposals] == [p.genotype for p in plain.proposals]
    assert batch.proposals and len(batch.instrumentation["timetable"]) == len(batch.proposals)
    exact = evaluate(
        problem,
        batch.proposals[0].genotype,
        gateway.evaluate(batch.proposals[0].genotype).schedule,
    )
    tt = batch.instrumentation["timetable"][0]
    assert exact.feasible and exact.flow <= tt["flow"] + 1e-6 and tt["bill"] > 0


def test_routing_frequency_and_stream_isolation():
    problem, gateway, members = _population(6, 6)
    guided = _guided(probability=0.2)
    lambdas, neighbors = weights(6), neighborhoods(weights(6), 3)
    streams = RNGManager(11)
    variation_before = RNGManager(11).stream("variation").getstate()
    hits = 0
    for k in range(1500):
        child = guided(members, k % 6, neighbors[k % 6], lambdas, gateway, streams)
        hits += child is not None
    attempts = sum(c.attempts for c in guided.cells.values())
    assert guided.slots == 1500 and abs(attempts - 300) < 70  # binomial(1500, 0.2): 5 sigma ~ 87
    assert guided.summary()["attempts"] == attempts and hits <= attempts
    assert "variation" not in streams.snapshot()  # the usual variation stream is never touched
    assert RNGManager(11).stream("variation").getstate() == variation_before


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_probability_extremes(probability):
    problem, gateway, members = _population(6, 6)
    guided = _guided(probability=probability)
    streams = RNGManager(2)
    out = [guided(members, k % 6, tuple(range(6)), weights(6), gateway, streams) for k in range(40)]
    attempts = sum(c.attempts for c in guided.cells.values())
    assert attempts == (0 if probability == 0 else 40)
    if probability == 0:
        assert all(c is None for c in out) and "F3:destroy" not in streams.snapshot()


def test_fallbacks_are_counted_and_other_errors_propagate(monkeypatch):
    problem, gateway, members = _population(6, 6)
    args = (members, 0, tuple(range(6)), weights(6), gateway)

    def raising(message):
        def propose(self, *a, **k):
            raise ValueError(message)

        return propose

    guided = _guided(probability=1.0)
    monkeypatch.setattr(
        TargetedDestroyRepair, "propose", raising("Removed Job has no feasible reinsertion")
    )
    assert guided(*args, RNGManager(1)) is None
    monkeypatch.setattr(TargetedDestroyRepair, "propose", lambda self, *a, **k: ProposalBatch())
    assert guided(*args, RNGManager(1)) is None
    cell = next(iter(guided.cells.values()))
    cell_total = guided.summary()
    assert cell_total["fallback_no_reinsertion"] == 1 and cell_total["fallback_no_proposal"] == 1
    assert cell_total["successes"] == 0 and cell.fallback_seconds >= 0
    exact_before = gateway.counts["exact"]
    monkeypatch.setattr(TargetedDestroyRepair, "propose", raising("something else"))
    with pytest.raises(ValueError, match="something else"):
        guided(*args, RNGManager(1))
    assert gateway.counts["exact"] == exact_before  # fallbacks consume no exact evaluation


def test_successful_child_is_structural_exact_and_recorded():
    problem = replace(mechanism_problem(panel_specs((50,))[2]), power_model=POWER)
    from geo_llm_scheduler.initialization.generators import generate

    gateway = EvaluationGateway(problem, Archive())
    rng = Random(4)
    members = [gateway.evaluate(generate(problem, m, rng, 0.1)) for m in (0, 3, 5, 7, 9, 1)]
    lam = weights(len(members))
    for options in (
        GuidedOptions(probability=1.0),
        GuidedOptions(probability=1.0, neighbor_paths=True),
    ):
        guided = GuidedVariation(Config(), options, gain_probe=lambda: 0.25)
        streams = RNGManager(9)
        exact_before = gateway.counts["exact"]
        children = [
            guided(members, k % 6, tuple(range(6)), lam, gateway, streams) for k in range(12)
        ]
        done = [c for c in children if c is not None]
        assert done and all(c.origin == "structural" for c in done)
        assert gateway.counts["exact"] - exact_before == len(done)
        for c in done:  # exact is the truth: independent re-evaluation agrees
            assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
        summary = guided.summary()
        assert summary["successes"] == len(done)
        assert (
            summary["attempts"]
            == 12
            == summary["successes"]
            + summary["fallback_no_proposal"]
            + summary["fallback_no_reinsertion"]
        )
        total_gap_n = sum(row["gap_count"] for row in summary["by_direction"].values())
        assert total_gap_n == sum(c.evaluation.feasible for c in done)
        json.dumps(summary, allow_nan=False)  # JSON-ready


# ---------------- whole-run behaviour under a deterministic clock ----------------
class _Clock:
    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def _run(monkeypatch, arm, seed=5, seconds=40, generations=3, probability=None):
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

    if probability is not None:
        monkeypatch.setitem(
            s4_moead._GUIDED, arm, replace(s4_moead._GUIDED[arm], probability=probability)
        )
    result = s4_moead.run_arm(problem, arm, config, CAL, {}, search_sink=sink)
    return result, searches[0]


def _signature(result, search):
    return (
        [(c.genotype, c.schedule, c.evaluation) for c in result.gateway.archive.members],
        {k: v for k, v in result.gateway.counts.items()},
        search.steps,
        search.moead_counts,
    )


def test_control_arm_is_untouched_and_probability_zero_matches_it(monkeypatch):
    control = _signature(*_run(monkeypatch, "N7_P1_SHARE"))
    assert control[1]["exact"] > 100 and control[2]
    zero = _signature(*_run(monkeypatch, "V1_GUIDED_DIR", probability=0.0))
    assert zero == control  # route stream only; no variation draw, no extra exact


@pytest.mark.parametrize("arm", s4_moead.GUIDED_ARMS)
def test_guided_arms_repeat_exactly_and_book_children_as_evolution(monkeypatch, arm):
    first = _run(monkeypatch, arm, probability=0.5)
    second = _run(monkeypatch, arm, probability=0.5)
    assert _signature(*first) == _signature(*second)
    f3 = first[1].moead_summary()["f3"]
    assert f3["successes"] > 5 and f3["slots"] >= f3["attempts"] >= f3["successes"]
    assert set(f3["by_direction"]) <= set(PREFERENCE_LABELS)
    guided_exact = f3["successes"]
    counts = first[0].gateway.counts
    assert counts["exact:structural"] >= guided_exact  # origin stays "structural"
    assert not any(k.startswith("exact:F3") for k in counts)
    assert f3["options"] == replace(s4_moead.guided_options(arm), probability=0.5).record()


def test_runner_writes_the_f3_record(tmp_path):
    from geo_llm_scheduler.experiments.trans_runner import run_case
    from geo_llm_scheduler.io.loaders import save_instance

    instance = tmp_path / "input.json"
    save_instance(synthetic(6)[0], instance)
    out = tmp_path / "out"
    summary = run_case(
        "V1_GUIDED_DIR",
        instance,
        out,
        seconds=6,
        calibration=CAL,
        max_generations=3,
        campaign_output=True,
    )
    assert summary["independent_exact_audit"] == "passed"
    import gzip

    with gzip.open(out / "local_search.json.gz", "rt") as f:
        data = json.load(f)
    f3 = data["moead"]["f3"]
    assert f3["slots"] > 0 and "cells" in f3
    algo = json.loads((out / "algorithm_config.json").read_text())
    assert algo["guided"]["probability"] == 0.2 and algo["guided"]["direction_aware"] is True


# ---------------- batch entry ----------------
@pytest.fixture
def guided_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_guided")


def test_batch_prepares_four_arms_on_the_panel_at_two_new_seeds(
    tmp_path, monkeypatch, guided_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(guided_batch.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(guided_batch.base, "CAL", digest(cal))
    assert guided_batch.ARMS == (
        "N7_P1_SHARE",
        "V1_GUIDED_DIR",
        "V2_GUIDED_RAND",
        "V3_GUIDED_DIR_NB",
    )
    assert guided_batch.CONTROL == "N7_P1_SHARE" and guided_batch.SEEDS == (12, 13)
    seen = []
    for k in range(3):
        root = tmp_path / f"root{k}"
        guided_batch.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = guided_batch.validate_inputs(root)
        assert m["design"] == "s4-guided/v1" and tuple(m["arms"]) == guided_batch.ARMS
        assert m["runs"] == 4 * len(m["blocks"]) and len(m["blocks"]) == 80
        for block in m["blocks"]:
            spec = json.loads((root / "specs" / (block + ".json")).read_text())
            assert spec["seed"] in (12, 13) and "run_s4_guided.py" in spec["entry"]
            assert spec["seconds"] == {50: 600, 100: 1200, 200: 2400}[spec["n"]]
            assert "guided" not in spec["config"]["N7_P1_SHARE"]
            assert spec["config"]["V1_GUIDED_DIR"]["guided"]["direction_aware"] is True
            assert spec["config"]["V2_GUIDED_RAND"]["guided"]["direction_aware"] is False
            assert spec["config"]["V3_GUIDED_DIR_NB"]["guided"]["neighbor_paths"] is True
        seen += m["blocks"]
    assert len(seen) == len(set(seen)) == 240
    assert Counter(b.rsplit("_s", 1)[1] for b in seen) == {"12": 120, "13": 120}
