"""Paired continuation preserves original transitions and plain-engine offspring."""

import pickle
from dataclasses import replace
from random import Random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.offspring import create_offspring
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.lsv2_delayed import (
    ActionUnavailableError,
    audit_candidates,
    continue_search,
    fingerprint,
    force_step,
    opportunity_features,
    pair,
    restore,
)
from geo_llm_scheduler.experiments.lsv2_study import operator_config, study_config
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.search import A8Gate, LSV2Config, local_box
from geo_llm_scheduler.moead.core import (
    NormalizationContext,
    maximum,
    neighborhoods,
    replace_neighbors,
    weights,
)
from geo_llm_scheduler.moead.replacement import replacement_neighborhood
from geo_llm_scheduler.utils.rng import RNGManager


@pytest.fixture(scope="module")
def snapshot():
    p = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    cfg = replace(study_config("LSV2_RANDSEL", 16901, 100), population=6, neighborhood=4)
    gw = EvaluationGateway(p, Archive(), decode_cache_size=cfg.decode_cache_size)
    pop = [gw.evaluate(g) for g in initial_genotypes(p, cfg, Random(3))[:6]]
    frame = FrozenFrame.of([c.evaluation.objectives for c in gw.archive.members])
    current = gw.archive.members[0]
    return dict(
        problem=p,
        population=pop,
        current=current,
        archive=gw.archive.members,
        archive_attempts=gw.archive.attempts,
        archive_insertions=gw.archive.insertions,
        archive_peak=gw.archive.peak_size,
        ideal=gw.ideal,
        cache=gw._decoded,
        engine=cfg,
        operators=operator_config(16901, 100),
        config=LSV2Config(selector="uniform"),
        gate=A8Gate(8, 4, 0.75, 20),
        frame=frame,
        frame_id=0,
        ls_seconds=0,
        box=local_box(
            [frame.normalize(c.evaluation.objectives) for c in gw.archive.members],
            frame.normalize(current.evaluation.objectives),
        ),
        info=dict(trajectory=0, step=0, seed=current.evaluation.objectives, seed_gap=0),
    )


def test_force_matches_original_choice_and_snapshot_isolation(snapshot):
    baseline = pickle.dumps(snapshot)
    normal = restore(snapshot)
    normal.search.config = replace(normal.search.config, selector="uniform")
    s = normal.snapshot
    normal.current, _ = normal.search._step(
        s["population"],
        normal.gateway,
        RNGManager(29),
        s["frame"],
        normal.current,
        s["box"],
        s["info"],
    )
    op = normal.search.steps[-1]["operator"]
    forced = restore(snapshot)
    row = force_step(forced, op, 29)
    assert fingerprint(normal) == fingerprint(forced)
    assert row["hv_gain"] == normal.search.steps[-1]["hv_gain"]
    assert audit_candidates(forced) == forced.gateway.counts["exact"]
    assert pickle.dumps(snapshot) == baseline


@pytest.mark.parametrize("op", ["A1", "A7", "P2_FLOW", "P3"])
def test_pair_reproducible_and_telescopes(snapshot, op):
    a = pair(snapshot, op, "A7", 100, 200)
    b = pair(snapshot, op, "A7", 100, 200)
    assert a["eligible"] and a["state_sha256"] == b["state_sha256"]
    assert a["terminal_gain"] == pytest.approx(
        a["first_step"]["hv_gain"] + a["second_step"]["hv_gain"], abs=1e-12
    )
    assert a["fresh_exact"] == a["first_step"]["exact"] + a["second_step"]["exact"]
    assert snapshot["frame"].hv(a["terminal_objectives"]) - a["initial_hv"] == pytest.approx(
        a["terminal_gain"], abs=1e-12
    )


def test_gate_is_not_bypassed_or_double_counted(snapshot):
    context = restore(snapshot)
    for segment in range(3):
        for _ in range(4):
            context.search.gate.record(segment, True)
    before = pickle.dumps(context.search.gate)
    with pytest.raises(ActionUnavailableError):
        force_step(context, "A8", 1)
    assert pickle.dumps(context.search.gate) == before
    result = pair(context.snapshot, "A7", "A8", 1, 2)
    assert not result["eligible"]


def test_pair_does_not_hide_operator_errors(snapshot, monkeypatch):
    from geo_llm_scheduler.experiments import lsv2_delayed

    original = lsv2_delayed.force_step

    def broken(context, action, seed):
        if seed == 2:
            raise ValueError("operator bug")
        return original(context, action, seed)

    monkeypatch.setattr(lsv2_delayed, "force_step", broken)
    with pytest.raises(ValueError, match="operator bug"):
        pair(snapshot, "A7", "A7", 1, 2)


def test_unknown_action_and_budget_rejected(snapshot):
    with pytest.raises(ValueError):
        force_step(restore(snapshot), "BAD", 1)
    with pytest.raises(ValueError):
        continue_search(snapshot, "A7", 1, 2, 0)
    row = continue_search(snapshot, "A7", 1, 2, 1e-12)
    assert row["termination"] == "prefix_exhausted" and row["offspring"] == 0


def test_noop_engine_matches_plain_offspring_reference(snapshot):
    seed = 90
    c = restore(snapshot)
    s = c.snapshot
    cfg = replace(s["engine"], seed=seed)
    rng = RNGManager(seed)
    order = list(range(cfg.population))
    rng.stream("traversal").shuffle(order)
    lam = weights(cfg.population)
    near = neighborhoods(lam, cfg.neighborhood)
    for i in order[:3]:
        child = create_offspring(s["population"], i, near[i], cfg, c.gateway, rng)
        norm = NormalizationContext(c.gateway.ideal, maximum(s["population"]))
        targets = replacement_neighborhood(cfg.replacement_policy, i, child, near, lam, norm, rng)
        replace_neighbors(s["population"], child, targets, lam, norm, cfg.replacement_cap)
    row = continue_search(snapshot, "NOOP", 1, seed, 100, offspring_limit=3)
    assert row["state_sha256"] == fingerprint(c)
    assert row["offspring"] == 3 and row["tail_ls_steps"] == 0


def test_features_no_mutation_and_finite(snapshot):
    baseline = pickle.dumps(snapshot)
    features = opportunity_features(snapshot)
    assert 0 <= features["prefill_window_fraction"] <= 1
    assert features["seconds"] >= 0 and features["resource_wait_fraction"] >= 0
    assert pickle.dumps(snapshot) == baseline


def test_short_continuation_audits_and_terminates(snapshot):
    row = continue_search(snapshot, "NOOP", 1, 2, 0.1)
    assert row["termination"] == "time_budget"
    assert row["charged_seconds"] >= 0.1
    assert row["fresh_exact"] == row["tail_exact"] and row["fresh_exact"] > 0
    # Instrumentation may let the initial LS trajectory consume this tiny budget.
    # A valid timed continuation need not reach its first evolutionary offspring.
    assert row["offspring"] + row["tail_ls_steps"] > 0
