"""Bounds, counterexamples, reference identity and Archive-only gate integration."""

import importlib
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest
from test_guided_cost import _run, _sig
from test_lsv2_fast_rightshift import CAL, synthetic
from test_s4_load_shift import _panel
from test_s4_moead import _initialized

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, PowerModel, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.trans_runner import digest
from geo_llm_scheduler.lsv2.terminal_gates import POLICIES, GatePolicy, TerminalGate, opportunity
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.scheduling.resources import intervals, union_length
from geo_llm_scheduler.utils.rng import RNGManager


def test_slack_zero_and_positive_are_not_cost_certificates():
    p, c = synthetic(2)
    assert opportunity(p, c, "SHIFT") > 0
    s = Schedule((90.0, 100.0, 110.0, 120.0))
    tight = Candidate(c.genotype, s, evaluate(p, c.genotype, s))
    assert tight.evaluation.feasible and opportunity(p, tight, "SHIFT") == 0
    with pytest.raises(ValueError):
        opportunity(p, c, "unknown")


@pytest.mark.parametrize("power", [PowerModel(), PowerModel(0.4, 1.0, 0.25)])
def test_bounds_cover_actual_generated_gains(power):
    p, c = synthetic(6)
    p = replace(
        p,
        power_model=power,
        jobs=tuple(
            replace(
                j, prefill=replace(j.prefill, compute=0.4), decode=replace(j.decode, compute=0.4)
            )
            for j in p.jobs
        ),
    )
    g = replace(c.genotype, ms=tuple(0 for _ in c.genotype.ms))
    sched = Schedule(tuple(t for i in range(6) for t in (i * 10.0, 100 + i * 10.0)))
    c = replace(c, genotype=g, schedule=sched)
    c = replace(c, evaluation=evaluate(p, c.genotype, c.schedule))
    union = sum(
        union_length([(a, b) for _, a, b in intervals(p, c.genotype, c.schedule, m)])
        for m in range(len(p.instances))
    )
    pack_bound = opportunity(p, c, "A7") * union
    peak_bound = opportunity(p, c, "A8") * c.evaluation.bill
    for op, name in [(ActivePack(), "A7"), (PeakCoalition(), "A8")]:
        batch = op.propose(p, c, 10, Config(), Random(8))
        assert batch.proposals, name
        for prop in batch.proposals:
            ev = evaluate(p, prop.genotype, prop.schedule)
            assert ev.feasible
            if name == "A7":
                after = sum(
                    union_length(
                        [(a, b) for _, a, b in intervals(p, prop.genotype, prop.schedule, m)]
                    )
                    for m in range(len(p.instances))
                )
                assert union - after <= pack_bound + 1e-7
            else:
                assert sum(c.evaluation.demand) - sum(ev.demand) <= peak_bound + 1e-7


def test_capacity_saturated_union_has_no_compression_bound():
    p, c = synthetic(2)
    # Each operation consumes full Compute; total union already equals total work.
    assert opportunity(p, c, "A7") == 0
    assert not TerminalGate("TG_PACK").admit(p, c, "A7")
    assert TerminalGate("TG_SHIFT").admit(p, c, "A7")  # untouched operator


def test_cap_is_per_genotype_and_operator_not_global_shutdown():
    p, c = synthetic(2)
    gate = TerminalGate("TG_CAPPED")
    assert [gate.admit(p, c, "SHIFT") for _ in range(4)] == [True, True, True, False]
    other = replace(c, genotype=replace(c.genotype, os=tuple(reversed(c.genotype.os))))
    assert gate.admit(p, other, "SHIFT")
    assert gate.counts["SHIFT:cap_skip"] == 1


def test_quantiles_preserve_ties_and_reject_zero(monkeypatch):
    from geo_llm_scheduler.lsv2 import terminal_gates as mod

    p, c = synthetic(2)
    peers = [
        replace(
            c,
            schedule=Schedule((float(i), 100.0, 0.0, 120.0)),
            evaluation=replace(c.evaluation, flow=float(i)),
        )
        for i in range(4)
    ]
    monkeypatch.setattr(mod, "opportunity", lambda p, c, o: c.schedule.starts[0])
    gate = TerminalGate("TG_MEDIUM")
    assert [gate.admit(p, c, "SHIFT", peers) for c in peers] == [False, False, True, True]
    assert gate.admit(p, peers[2], "SHIFT", [peers[2], peers[2]])


def test_disabled_gate_is_identical_to_reference_terminal_path(monkeypatch):
    monkeypatch.setitem(POLICIES, "TG_MEDIUM", GatePolicy())
    records = []
    for arm in ("GS4_GUIDED_RAND_SWITCH_D4", "TG_MEDIUM"):
        p, gw, pop, search = _initialized(4, arm)
        streams = RNGManager(3)
        search._setup(pop, gw, streams)
        search._terminal_sweep(pop, gw, streams, lambda: 0.0)
        records.append((list(gw.archive.members), dict(gw.counts), list(pop)))
    assert records[0] == records[1]


def test_gates_do_not_change_preterminal_search_or_reference_records(monkeypatch):
    arm = "GS4_GUIDED_RAND_SWITCH_D4"
    ref = _sig(*_run(monkeypatch, arm, guided_timer=True))
    new = _sig(*_run(monkeypatch, "TG_MEDIUM", guided_timer=True))
    assert new == ref
    cfg = s4_moead.arm_config(arm, 16, 1200)
    assert "terminal_gate" not in s4_moead.configuration_record(arm, cfg, CAL)
    assert s4_moead.guided_options("TG_MEDIUM") == s4_moead.guided_options(arm)


@pytest.mark.parametrize("arm", POLICIES)
def test_terminal_gates_preserve_population_exact_and_inherited_inputs(arm):
    p, gw, pop, search = _initialized(6, arm)
    search._setup(pop, gw, RNGManager(3))
    before = list(pop)
    search._terminal_sweep(pop, gw, RNGManager(3), lambda: 0.0)
    assert pop == before
    assert search.terminal_gate.counts
    for c in gw.archive.members:
        assert c.evaluation == evaluate(p, c.genotype, c.schedule)


def test_nine_arm_matrix_covers_full_panel_and_inclusive_budgets(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    b = importlib.import_module("run_terminal_gates")
    panel, cal, _ = _panel(tmp_path, dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25))
    monkeypatch.setattr(b.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(b.base, "CAL", digest(cal))
    seen = []
    runs = seconds = 0
    for k in range(3):
        root = tmp_path / f"root{k}"
        b.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = b.validate_inputs(root)
        seen += m["blocks"]
        runs += m["runs"]
        seconds += m["solver_seconds"]
    assert len(seen) == len(set(seen)) == 120
    assert runs == 1080 and seconds == 9 * 40 * (600 + 1200 + 2400)
