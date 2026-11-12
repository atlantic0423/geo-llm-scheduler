"""Live sampling context, denominator, boundedness and fixed-work path invariance."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_guided_variation import _population
from test_lsv2_fast_rightshift import CAL
from test_s4_load_shift import _panel

from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.r1_observation import RegionalObserver
from geo_llm_scheduler.experiments.trans_runner import digest
from geo_llm_scheduler.moead.core import NormalizationContext, maximum

ARM = "GS4_GUIDED_RAND_SWITCH_D4"


def test_bounds_stage_denominators_and_post_replacement_snapshot():
    p, gw, pop = _population()
    observer = RegionalObserver(100, 17, capacity=2, recipients=2)
    observer("initialization", dict(elapsed=5))
    context = NormalizationContext(gw.ideal, maximum(pop))
    for stage in range(3):
        for k in range(20):
            data = dict(
                elapsed=6 + 30 * stage,
                candidate=pop[0],
                context=context,
                neighbors=(0, 1, 2),
                generation=k,
                subproblem=0,
            )
            observer("before_replacement", data)
            observer(
                "offspring", dict(population=pop, gateway=gw, processed=k, row={"replaced": 2})
            )
    record = observer.record()
    assert record["seen"] == [20, 20, 20]
    assert len(record["snapshots"]) == 6
    assert all(len(r["receivers"]) == 2 and r["replaced"] == 2 for r in record["snapshots"])
    json.dumps(record, allow_nan=False)
    before = record["snapshots"][0]["receivers"][0]["candidate"]
    pop[:] = pop[::-1]
    gw.archive.members.clear()
    assert observer.record()["snapshots"][0]["receivers"][0]["candidate"] == before
    assert observer.record()["snapshots"][0]["archive"]
    observer("before_replacement", {**data, "elapsed": 95})
    assert observer.seen == [20, 20, 20] and observer.pending is None
    with pytest.raises(ValueError):
        RegionalObserver(100, 0, capacity=0)


def test_gs4_fixed_work_rng_and_every_decision_match_without_sampler(monkeypatch):
    problem, _, _ = _population(8, 6)
    # A constant fake clock freezes all timing-dependent controllers. A fixed
    # generation count supplies work; observer clock reads cannot advance it.
    for name, module in tuple(sys.modules.items()):
        if name.startswith("geo_llm_scheduler.") and module is not None:
            if callable(getattr(module, "perf_counter", None)):
                monkeypatch.setattr(module, "perf_counter", lambda: 1.0)
    cfg = replace(s4_moead.arm_config(ARM, 17, 600), generations=2)
    traces, states, results = [], [], []
    for active in (False, True):
        trace, state = [], []
        sampler = RegionalObserver(600, 17)

        def observe(event, data):
            if event == "offspring":
                trace.append(data["row"].copy())
                state.append(data["streams"].snapshot())
            if active:
                sampler(event, data)

        def sink(search):
            search.timer = lambda: 1.0

        result = s4_moead.run_arm(
            problem, ARM, cfg, CAL, {}, observer=observe, search_sink=sink, guided_timer=lambda: 1.0
        )
        traces.append(trace)
        states.append(state)
        results.append((result.population, result.archive.members, dict(result.gateway.counts)))
    assert traces[0] == traces[1] and len(traces[0]) == 200
    assert states[0] == states[1] and results[0] == results[1]
    assert sampler.seen == [200, 0, 0] and len(sampler.record()["snapshots"]) == 16


def test_campaign_uses_all_120_instances_one_fresh_seed_and_default_budget(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    b = importlib.import_module("run_r1_observation")
    panel, cal, _ = _panel(tmp_path, dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25))
    monkeypatch.setattr(b.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(b.base, "CAL", digest(cal))
    keys, seconds = [], 0
    for k in range(3):
        root = tmp_path / str(k)
        b.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = b.validate_inputs(root)
        assert m["arms"] == [ARM] and m["seeds"] == [17]
        keys.extend(m["blocks"])
        seconds += m["solver_seconds"]
    assert len(keys) == len(set(keys)) == 120 and seconds == 168000


def test_offline_replay_keeps_empty_pairs_and_identical_blocks_in_denominator(monkeypatch):
    from dataclasses import asdict

    from test_r1_offline import _candidate, _problem

    from geo_llm_scheduler.domain.models import Genotype

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    replay = importlib.import_module("analyze_r1_live")
    p = _problem(0.4)
    donor = _candidate(p, Genotype((0, 0) * 3 + (2, 2) * 3, tuple(range(6)) * 2))
    other = _candidate(p, Genotype((2, 2) + (0, 0) * 2 + (2, 2) * 3, tuple(range(6)) * 2))
    snapshot = dict(
        donor=asdict(donor),
        archive=[asdict(donor)],
        front=[donor.evaluation.objectives],
        context=dict(ideal=donor.evaluation.objectives, maximum=other.evaluation.objectives),
        stage=0,
        processed=1,
        elapsed=12,
        subproblem=0,
        receivers=[
            dict(slot=i, weight=(0.5, 0.5), candidate=asdict(c))
            for i, c in enumerate((donor, other))
        ],
    )
    pairs, trials = replay.analyze_snapshot(p, snapshot)
    assert len(pairs) == 2 and [x["eligible"] for x in pairs] == [2, 0]
    assert len(trials) == 2 and all(not x["rescued"] and not x["hv_positive"] for x in trials)
    assert all(not x["nonidentical"] for x in trials)
    summary = replay.summarize(
        [
            dict(
                observer_seconds=0,
                trials=trials,
                pairs=[dict(key="x", n=6, **x) for x in pairs],
            )
        ]
    )
    assert summary["strata"][0]["pairs"] == 2
    assert summary["strata"][0]["eligible_pairs"] == 1


def test_runner_persists_observation_under_manifest(tmp_path, monkeypatch):
    from geo_llm_scheduler.experiments.trans_runner import run_case
    from geo_llm_scheduler.io.loaders import save_instance

    problem, _, _ = _population(6, 6)
    instance = tmp_path / "instance.json"
    save_instance(problem, instance)
    out = tmp_path / "run"
    result = run_case(
        ARM,
        instance,
        out,
        seed=17,
        seconds=20,
        calibration=CAL,
        max_generations=1,
        campaign_output=True,
        regional_observation=True,
    )
    assert result["independent_exact_audit"] == "passed"
    entries = json.loads((out / "manifest.json").read_text())
    entry = next(e for e in entries if e["path"] == "regional_observation.json.gz")
    assert digest(out / entry["path"]) == entry["sha256"]
