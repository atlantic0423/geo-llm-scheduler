"""Independent selection contrasts, scoped certificates and honest log denominators."""

import gzip
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, NO_P3, _prepared, synthetic
from test_lsv2_group_polish import joint_problem
from test_lsv2_polish_selection import CTX, FRAME, markers

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import lsv2_common_opportunity as legacy
from geo_llm_scheduler.experiments import lsv2_common_refine as reference
from geo_llm_scheduler.experiments import lsv2_eligibility_selection as study
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish
from geo_llm_scheduler.lsv2.opportunity_selection import OpportunitySelector
from geo_llm_scheduler.lsv2.polish_selection import PolishSelector
from geo_llm_scheduler.lsv2.selection_summary import summarize_selection
from geo_llm_scheduler.moead.core import weights


def test_matrix_separates_selection_from_resources_and_preserves_controls():
    assert len(study.ARMS) == 8 and len(study.PRIMARY_CONTRASTS) == 7
    assert study.search_config("S0_NONE") == reference.search_config("C0_NONE")
    base = asdict(study.search_config("S1_RANDOM"))
    for arm in study.ARMS[1:]:
        config = study.search_config(arm)
        differences = {k for k, v in asdict(config).items() if v != base[k]}
        assert differences <= {"polish_selection"}
        assert config.polish_method == "common_fast"
        assert config.time_share == 0.2 and config.polish_budget == 3
        assert config.budget == config.trajectory_steps == 3
        assert config.polish_targets == "partial" and not config.polish_pool_directions
        assert "P3" not in config.operators and {"A3", "A4", "A8", "P2_MIXED"} <= set(
            config.operators
        )
        for n, seconds in ((100, 1200), (200, 2400)):
            record = study.configuration_record(arm, 16901, seconds, CAL)
            assert record["qspec"]["gamma"] == 0.9
            assert study.arm_config(arm, 16901, seconds).seconds == seconds
            assert len(study.PANEL[n]) == 4
    # The old R matrix is deliberately not rewritten by this new study.
    assert legacy.search_config("R3_ELIGIBLE").polish_method == "common_calendar"
    for fn, args in ((study.search_config, ("bad",)), (study.arm_config, ("bad", 1, 10))):
        with pytest.raises(ValueError):
            fn(*args)


@pytest.mark.parametrize("arm", study.ARMS)
def test_study_builds_real_gateway_with_terminal_credit_before_polish(arm):
    original, pop, gateway, streams = _prepared(NO_P3)
    search = study.make_search(arm, original.engine_config, CAL, original.table, training=True)
    search.timer = original.timer
    search(pop, gateway, streams, lambda: 0)
    ref, rp, rg, rs = _prepared(NO_P3)
    ref._trajectory(rp, rg, rs, lambda: 10)
    search._trajectory(pop, gateway, streams, lambda: 10)
    assert asdict(search.table) == asdict(ref.table) and search._pending is None
    assert search.steps[-1]["q_transition"]["terminal"]
    assert not search.reused_polish.resource_calendar and not search.reused_polish.peak_switch
    summary = summarize_selection(search.polish_steps)
    assert summary["module_attempts"] == (0 if arm == "S0_NONE" else 1)
    for c in pop + gateway.archive.members:
        assert evaluate(gateway.problem, c.genotype, c.schedule) == c.evaluation


def test_lazy_random_reference_matches_legacy_across_next_trajectories():
    a, ap, ag, ar = _prepared(NO_P3)
    b, bp, bg, br = _prepared(NO_P3)
    a.config = reference.search_config("C2_FAST")
    b.config = study.search_config("S1_RANDOM")
    for _ in range(5):
        a._trajectory(ap, ag, ar, lambda: 10)
        b._trajectory(bp, bg, br, lambda: 10)
        assert ap == bp and ag.archive.members == bg.archive.members
        assert asdict(a.table) == asdict(b.table)
        assert a.polish_steps[-1]["input_schedule"] == b.polish_steps[-1]["input_schedule"]


@pytest.mark.parametrize(
    "policies", [("pool_random", "eligible_random"), ("direction", "eligible_direction")]
)
def test_no_certificates_means_same_selection_rng_and_direction_rotation(policies):
    pool = markers()
    old, new = PolishSelector(), OpportunitySelector()
    a, b = Random(10), Random(10)
    for _ in range(40):
        x = old.choose(
            policies[0], pool, pool[0], FRAME, CTX, weights(11), a, diagnose_directions=False
        )
        y = new.choose(policies[1], pool, pool[0], FRAME, CTX, weights(11), b, lambda c: False)
        assert x == y and a.getstate() == b.getstate()
        new.completed(y.candidate)


def test_direction_reference_and_fresh_are_distinguished_by_prior_visits_only():
    a = markers()[1]
    b = replace(
        a,
        schedule=Schedule((9.0, *a.schedule.starts[1:])),
        evaluation=replace(
            a.evaluation, flow=a.evaluation.flow + 0.01, tou=a.evaluation.tou + 0.01
        ),
    )
    # A single populated direction isolates representative choice from allocation.
    lambdas = ((0.5, 0.5),)
    plain, fresh = OpportunitySelector(), OpportunitySelector()
    for selector in (plain, fresh):
        selector.completed(a)
    x = plain.choose(
        "eligible_direction", [a, b], a, FRAME, CTX, lambdas, Random(1), lambda c: False
    )
    y = fresh.choose("fresh_direction", [a, b], a, FRAME, CTX, lambdas, Random(1), lambda c: False)
    assert x.candidate == a and y.candidate == b
    excluded = plain.choose(
        "eligible_direction", [a, b], a, FRAME, CTX, lambdas, Random(1), lambda c: c == a
    )
    assert excluded.candidate == b
    for policy in ("eligible_direction", "fresh_direction"):
        with pytest.raises(ValueError):
            plain.choose(policy, [a], a, FRAME, CTX, (), Random(1), lambda c: False)


def test_region_balance_skips_certified_region_even_when_it_has_least_cost():
    pool = markers()[:3]
    selector = OpportunitySelector()
    selector.charge(1, 4.0)
    selector.charge(2, 1.0)
    pick = selector.choose(
        "eligible_region", pool, pool[0], FRAME, CTX, weights(3), Random(1), lambda c: c == pool[0]
    )
    assert pick.region == 2 and pick.region_sizes == (0, 1, 1)
    selector.charge(2, 5.0)
    pick = selector.choose(
        "eligible_region", pool, pool[0], FRAME, CTX, weights(3), Random(1), lambda c: c == pool[0]
    )
    assert pick.region == 1


def test_certificate_changes_with_scope_but_not_archive_context_or_os():
    problem, c = synthetic(left_only=True)
    op = ReusedCommonPolish()
    assert not op.propose(problem, c, 3, Config(), Random(1)).proposals
    other_os = replace(c, genotype=replace(c.genotype, os=tuple(reversed(c.genotype.os))))
    assert op.known_empty(problem, other_os, 3, Config())
    op.all_targets = True
    assert not op.known_empty(problem, c, 3, Config())
    op.all_targets = False
    assert not op.known_empty(problem, c, 3, replace(Config(), seed=999))
    changed = replace(c, schedule=Schedule((1e-12, *c.schedule.starts[1:])))
    assert not op.known_empty(problem, changed, 3, Config())
    # Positive generator output remains eligible even if every Archive member
    # already dominates it; qualification never receives the Archive as input.
    problem, c = joint_problem()
    assert op.propose(problem, c, 3, Config(), Random(1)).proposals
    assert not op.known_empty(problem, c, 3, Config())


def _row(status, *, start=0.0, **kwargs):
    row = dict(
        status=status,
        seconds=0.1,
        seconds_selection=0.02,
        exact=0,
        proposals=0,
        accepted=False,
        hv_gain=0.0,
        reflow_replaced=0,
    )
    if status not in ("no_opportunity", "region_skip", "deadline_skip"):
        row.update(segment="middle", input_schedule=dict(ms=[0, 1], os=[0, 0], starts=[start, 100]))
    row.update(kwargs)
    return row


def test_summary_keeps_empty_skipped_partial_repeats_and_zero_hv_separate():
    rows = [
        _row("empty"),
        _row("empty"),
        _row("evaluated", start=2.0, exact=1, proposals=1, accepted=True, reflow_replaced=1),
        _row("no_opportunity", raw_pool_size=3, eligible_pool_size=0, certified_empty_excluded=3),
        _row("no_opportunity", raw_pool_size=3, eligible_pool_size=0, certified_empty_excluded=3),
        _row("deadline_partial", start=1e-12, proposals=3, exact=1),
        _row("deadline_after_selection", start=1.0),
        _row("deadline_skip"),
    ]
    s = summarize_selection(rows)
    assert s["module_attempts"] == 8 and s["generator_calls"] == 4 and s["complete_searches"] == 3
    assert s["empty_rate_complete"] == pytest.approx(2 / 3) and s["no_opportunity_calls"] == 2
    assert s["selected_calls"] == 5 and s["unique_selected_inputs"] == 4
    assert s["non_generated_calls"] == 4 and s["non_generated_rate"] == 0.5
    assert s["repeated_selected_calls"] == s["selections_after_complete_empty"] == 1
    assert s["certified_empty_exposures"] == 6 and s["measured_pool_calls"] == 2
    assert s["seconds_full"] == pytest.approx(0.8) and s["seconds_selected_by_region"][
        "middle"
    ] == pytest.approx(0.5)
    assert (
        s["accepted_bill_calls"] == 1
        and s["positive_hv_calls"] == 0
        and s["reflow_replacements"] == 1
    )
    assert s["exact"] == 2 and s["proposals"] == 4
    assert summarize_selection([])["empty_rate_complete"] is None
    with pytest.raises(ValueError):
        summarize_selection([_row("made_up")])
    with pytest.raises(ValueError):
        summarize_selection(
            [
                _row(
                    "no_opportunity",
                    raw_pool_size=1,
                    eligible_pool_size=0,
                    certified_empty_excluded=2,
                )
            ]
        )
    missing = _row("empty")
    del missing["input_schedule"]
    assert summarize_selection([missing])["selected_without_input_record"] == 1


def test_summary_cli_reads_compressed_log_and_leaves_original_bytes(tmp_path):
    log = tmp_path / "rows.json.gz"
    out = tmp_path / "summary.json"
    with gzip.open(log, "wt", encoding="utf-8") as stream:
        json.dump([_row("empty")], stream)
    before = log.read_bytes()
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/summarize_lsv2_selection.py", str(log), "--out", str(out)],
        cwd=repo,
        env=dict(os.environ, PYTHONPATH=str(repo / "src")),
        check=True,
    )
    assert log.read_bytes() == before and json.loads(out.read_text())["empty_rate_complete"] == 1
    failed = subprocess.run(
        [sys.executable, "scripts/summarize_lsv2_selection.py", str(log), "--out", str(log)],
        cwd=repo,
        env=dict(os.environ, PYTHONPATH=str(repo / "src")),
        capture_output=True,
    )
    assert failed.returncode != 0 and log.read_bytes() == before
