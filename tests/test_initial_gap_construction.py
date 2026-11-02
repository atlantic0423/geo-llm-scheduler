"""G1 uses the observed front only and all four exact trials remain eligible."""

from dataclasses import replace
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import initial_gap_construction as g1
from geo_llm_scheduler.experiments.initial_bill_mixture import guided_genotype
from geo_llm_scheduler.experiments.s4_ablation import arm_config, configuration_record, run_arm
from geo_llm_scheduler.initialization.retention import retain_pool
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, nondominated
from geo_llm_scheduler.utils.rng import RNGManager


def test_direction_depends_on_the_gap_and_remembers_raw_endpoints():
    front = [(0, 20), (1, 19), (9, 5), (10, 0)]
    frame = FrozenFrame.of(front)
    assert g1._direction(front, frame, [], set()) == ("bill_anchor", 0, None)
    edge = (front[1], front[2])
    kind, w, pair = g1._direction(front, frame, [0], set())
    assert kind == "gap" and pair == edge
    assert w == pytest.approx(0.6 / (0.5 + 0.6))
    assert g1._direction(front, frame, [0, w], {edge})[2] == (front[2], front[3])
    # A successful split creates fresh endpoint pairs even if the old one was tried.
    changed = front[:2] + [(5, 10)] + front[2:]
    assert g1._direction(changed, FrozenFrame.of(changed), [0, w], {edge})[2] == ((1, 19), (5, 10))


def test_singleton_fallback_consumes_ten_distinct_directions_and_handles_ties():
    front = [(3.0, 4.0)]
    frame = FrozenFrame.of(front)
    used = []
    for i in range(10):
        kind, weight, edge = g1._direction(front, frame, used, set())
        assert edge is None and kind == ("grid" if i else "bill_anchor")
        used.append(weight)
    assert used[:2] == [0.0, 1.0] and sorted(used) == [i / 9 for i in range(10)]
    front = [(0, 2), (1, 1), (2, 0)]
    assert g1._direction(front, FrozenFrame.of(front), [0], set())[2] == (front[0], front[1])


def test_g1_guidance_fast_and_reference_agree():
    pb, _ = synthetic(4)
    for w in (0, 0.35, 1):
        for seed in range(3):
            fast = guided_genotype(pb, "BILL", Random(seed), w, accelerated=True, tchebycheff=True)
            ref = guided_genotype(pb, "BILL", Random(seed), w, tchebycheff=True)
            assert fast == ref


@pytest.mark.parametrize("tchebycheff,eligible", [(False, [0, 2]), (True, [1])])
def test_prefix_choice_prefers_balanced_path_under_tchebycheff(monkeypatch, tchebycheff, eligible):
    # Controlled first-job path profiles, in (Flow,Bill) normalized coordinates:
    # (0,1), (.6,.6), (1,0). The existing 5% restricted list still separates them.
    import geo_llm_scheduler.experiments.initial_bill_mixture as mixture

    pb, _ = synthetic(3)
    monkeypatch.setattr(mixture, "paths", lambda *args: [(i, i) for i in range(3)])

    def earliest(self, machine, op, est, temporary=None):
        return est + ([0, 6, 10][machine] if op % 2 else 0)

    def bill(self, additions, horizon):
        return ([10, 6, 0][next(iter(additions))], 0, {}, 0, {}, horizon)

    monkeypatch.setattr(mixture.ResourceCalendar, "earliest", earliest)
    monkeypatch.setattr(mixture.PrefixPower, "trial", bill)

    class FirstChoice(Exception):
        pass

    class InspectChoice(Random):
        def choice(self, rows):
            assert [row[2] for row in rows] == eligible
            raise FirstChoice

    with pytest.raises(FirstChoice):
        guided_genotype(
            pb, "BILL", InspectChoice(1), 0.5, accelerated=True, tchebycheff=tchebycheff
        )


def test_exact_pool_replay_and_direction_causality():
    pb, _ = synthetic(4)
    cfg = arm_config("A0_FULL_S4", 1, 30)
    previous = None
    for _ in range(2):
        e, trace = {}, []
        gw = EvaluationGateway(pb, Archive())
        gw.candidate_sink = trace.append
        rng = RNGManager(1)
        rng_before = rng.stream("variation").getstate()
        members = g1.gap_initializer(pb, cfg, e)(gw, rng)
        assert rng.stream("variation").getstate() == rng_before
        assert len(trace) == gw.counts["exact"] == len({c.genotype for c in trace}) == 140
        assert len(members) == 100 and members == retain_pool(trace, 100)
        assert all(evaluate(pb, c.genotype, c.schedule) == c.evaluation for c in trace)
        assert set(c.evaluation.objectives for c in gw.archive.members) <= {
            c.evaluation.objectives for c in members
        }
        tried = set()
        for slot, row in enumerate(e["rounds"]):
            start = 100 + 4 * slot
            online = nondominated([c.evaluation.objectives for c in trace[:start]])
            frame = FrozenFrame.of(online)
            assert row["ideal"] == frame.ideal and row["nadir"] == frame.nadir
            assert row["points"] == [c.evaluation.objectives for c in trace[start : start + 4]]
            if row["edge"]:
                assert row["edge"] in list(zip(online, online[1:])) and row["edge"] not in tried
                tried.add(row["edge"])
            else:
                assert slot == 0 or all(edge in tried for edge in zip(online, online[1:]))
            assert len(row["scores"]) == 4
        stable = (members, e["rounds"], e["initial_archive_sha256"])
        assert previous is None or previous == stable
        previous = stable


def test_real_s4_seam_and_incompatible_options():
    pb, _ = synthetic(4)
    cfg = replace(arm_config("A0_FULL_S4", 1, 30), generations=1)
    opt = dict(initial_retention="global_pool", initial_construction="g1")
    evidence = {}
    result = run_arm(pb, "A0_FULL_S4", cfg, CAL, evidence, **opt)
    assert result.offspring_count == 100 and evidence["policy"] == "g1"
    assert configuration_record("A0_FULL_S4", cfg, CAL, **opt)["initialization"] == "g1"
    for arm, opts in [
        ("A3_RANDOM_INIT", opt),
        ("A3_RANDOM_INIT", dict(initial_construction="g1")),
        ("A0_FULL_S4", dict(initial_construction="unknown")),
    ]:
        with pytest.raises(ValueError):
            configuration_record(arm, cfg, CAL, **opts)
        with pytest.raises(ValueError):
            run_arm(pb, arm, cfg, CAL, {}, **opts)


def test_duplicate_exhaustion_does_not_evaluate_more_trials(monkeypatch):
    pb, _ = synthetic(4)
    cfg = arm_config("A0_FULL_S4", 1, 30)
    from geo_llm_scheduler.experiments.initial_bill_benchmark import traced_original

    parent = traced_original(pb, cfg)[0]
    monkeypatch.setattr(g1, "guided_genotype", lambda *a, **kw: (parent.genotype, {}))
    gw = EvaluationGateway(pb, Archive())
    with pytest.raises(ValueError, match="attempt limit"):
        g1.gap_initializer(pb, cfg, {})(gw, RNGManager(1))
    assert gw.counts["exact"] == 100


def test_legacy_g1_changes_only_population_and_preserves_exact_stream():
    from collections import Counter

    from geo_llm_scheduler.experiments.initial_bill_mixture import retain_original

    pb, _ = synthetic(4)
    cfg = arm_config("A0_FULL_S4", 1, 30)
    results = {}
    for retention in ("legacy", "global_pool"):
        e, trace = {}, []
        gw = EvaluationGateway(pb, Archive())
        gw.candidate_sink = trace.append
        members = g1.gap_initializer(pb, cfg, e, retention=retention)(gw, RNGManager(1))
        results[retention] = (members, trace, e, gw.archive.members)
    legacy, trace, e, archive = results["legacy"]
    new, other_trace, other_e, other_archive = results["global_pool"]
    assert trace == other_trace and archive == other_archive and e["rounds"] == other_e["rounds"]
    assert len(legacy) == len(new) == 100 and len({c.genotype for c in legacy}) == 100
    assert legacy[:90] == retain_original(trace[:100], 10)
    assert set(Counter(c.origin for c in legacy[:90]).values()) == {9}
    for slot, row in enumerate(e["rounds"]):
        trial = trace[100 + 4 * slot : 104 + 4 * slot]
        lo = tuple(min(c.evaluation.objectives[k] for c in trial) for k in (0, 1))
        span = tuple(max(c.evaluation.objectives[k] for c in trial) - lo[k] for k in (0, 1))
        w = row["flow_weight"]

        def rank(c):
            f, b = c.evaluation.objectives
            return (
                (1 - w) * (b - lo[1]) / (span[1] or 1) + w * (f - lo[0]) / (span[0] or 1),
                c.genotype.ms,
                c.genotype.os,
            )

        assert legacy[90 + slot] == min(trial, key=rank)
    with pytest.raises(ValueError, match="Unknown"):
        g1.gap_initializer(pb, cfg, {}, retention="unknown")


def test_g1_legacy_actual_artifact_entry(tmp_path):
    import json

    from geo_llm_scheduler.experiments.trans_runner import run_case
    from geo_llm_scheduler.io.loaders import save_instance

    path = tmp_path / "instance.json"
    save_instance(synthetic(4)[0], path)
    result = run_case(
        "A0_FULL_S4",
        path,
        tmp_path / "out",
        seed=1,
        seconds=30,
        calibration=CAL,
        max_generations=1,
        campaign_output=True,
        initial_construction="g1",
        initial_retention="legacy",
    )
    assert result["independent_exact_audit"] == "passed"
    e = json.loads((tmp_path / "out/initialization.json").read_text())
    assert e["initial_retention"] == "legacy" and e["retention"]["pool_count"] == 140
