"""Regional inheritance counterexamples, boundary accounting and replay metrics."""

import gzip
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import asdict, replace
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
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.scheduling.ssgs import decode

spec = importlib.util.spec_from_file_location(
    "r1_offline", Path(__file__).parents[1] / "scripts/r1_offline.py"
)
assert spec is not None and spec.loader is not None
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)


def _problem(beta: float = 1.0) -> ProblemInstance:
    return ProblemInstance(
        tuple(
            Job(str(i), 0 if i < 3 else 900, Profile(150, 0.6, 2), Profile(300, 0.5, 2), 45)
            for i in range(6)
        ),
        tuple(
            Region(str(r), (Tariff(0, 900, 0.2), Tariff(900, 10000, 0.7)), 2 + r) for r in range(3)
        ),
        tuple(ServingInstance(str(m), m // 2, 8, 10, 100) for m in range(6)),
        PowerModel(beta, 0.8, 0.6),
    )


def _candidate(p: ProblemInstance, g: Genotype) -> Candidate:
    schedule = decode(p, g)
    return Candidate(g, schedule, evaluate(p, g, schedule))


@pytest.mark.parametrize("beta", [1.0, 0.3])
def test_region_inheritance_randomized_and_independent_billing(beta):
    p = _problem(beta)
    rng = random.Random(9912)
    for _ in range(30):
        candidates = []
        for _ in range(2):
            os = list(range(6)) * 2
            rng.shuffle(os)
            ms = tuple(2 * (j // 3) + rng.randrange(2) for j in range(6) for _ in range(2))
            candidates.append(_candidate(p, Genotype(ms, tuple(os))))
        for region in (0, 1):
            g, expected = r1.mix_region(p, *candidates, region)
            actual = decode(p, g)
            assert actual.starts == expected.starts
            c = Candidate(g, actual, evaluate(p, g, actual))
            assert c.evaluation.feasible
            r1._oracle_check(p, c)
            assert r1.region_jobs(p, g, region) == r1.region_jobs(p, candidates[1].genotype, region)


def test_empty_and_unequal_job_sets_rejected():
    p = _problem()
    c = _candidate(p, Genotype((0, 0) * 3 + (2, 2) * 3, tuple(range(6)) * 2))
    other = _candidate(p, Genotype((2, 2) + (0, 0) * 2 + (2, 2) * 3, tuple(range(6)) * 2))
    with pytest.raises(ValueError, match="equal nonempty"):
        r1.mix_region(p, c, other, 0)
    with pytest.raises(ValueError, match="equal nonempty"):
        r1.mix_region(p, c, c, 2)
    invalid = Genotype((0, 2) + (0, 0) * 2 + (2, 2) * 3, tuple(range(6)) * 2)
    with pytest.raises(ValueError, match="P/D"):
        r1.region_jobs(p, invalid, 0)


def test_timing_polished_donor_is_not_automatically_inheritable():
    p = _problem()
    c = _candidate(p, Genotype((0, 0) * 3 + (2, 2) * 3, tuple(range(6)) * 2))
    delayed = Schedule(tuple(t + 60 if o // 2 < 3 else t for o, t in enumerate(c.schedule.starts)))
    donor = replace(c, schedule=delayed, evaluation=evaluate(p, c.genotype, delayed))
    assert donor.evaluation.feasible
    g, predicted = r1.mix_region(p, donor, c, 0)
    assert decode(p, g).starts != predicted.starts


def test_horizon_changes_idle_cost_in_untouched_region_and_fixed_tail():
    p = ProblemInstance(
        (Job("a", 0, Profile(450, 1, 1), Profile(450, 1, 1)),),
        (
            Region("a", (Tariff(0, 900, 0.2), Tariff(900, 10000, 0.7)), 2),
            Region("empty", (Tariff(0, 900, 0.2), Tariff(900, 10000, 0.7)), 3),
        ),
        (ServingInstance("a", 0, 8, 10, 100), ServingInstance("empty", 1, 8, 20, 100)),
    )
    g = Genotype((0, 0), (0, 0))
    s = Schedule((0, 450))
    later = Schedule((0, 900))
    a = Candidate(g, s, evaluate(p, g, s))
    b = Candidate(g, later, evaluate(p, g, later))
    ledger_a, ledger_b = r1._oracle_check(p, a), r1._oracle_check(p, b)
    assert ledger_b[1] - ledger_a[1] == pytest.approx(20 * 450 * 0.7 / 3600)
    assert b.evaluation.windows[1] == (20, 10)
    assert a.evaluation.windows[1] == (20,)


def test_hypervolume_union_not_sum_and_reference_clipping():
    points = [(1, 3), (2, 2), (3, 1)]
    assert r1.hypervolume(points, (4, 4)) == 6
    assert r1.hypervolume(points * 2 + [(5, 0), (0, 5)], (4, 4)) == 6
    assert r1.hypervolume([(-1, 1)], (4, 4)) == 15


def test_offline_replay_pipeline_keeps_inputs_and_checks_manifests(tmp_path, monkeypatch):
    p = _problem(0.4)
    shard = tmp_path / "formal_panel_shard0"
    run = shard / "runs" / "tiny_s16" / r1.ARM
    run.mkdir(parents=True)
    inputs = shard / "instances"
    inputs.mkdir()
    instance = inputs / "tiny.json"
    save_instance(p, instance)
    first = _candidate(p, Genotype((0, 0) * 3 + (2, 2) * 3, tuple(range(6)) * 2))
    second = _candidate(p, Genotype((0, 1) * 3 + (2, 3) * 3, tuple(reversed(range(6))) * 2))
    for name in ("native_population", "archive"):
        with gzip.open(run / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
            json.dump([asdict(first), asdict(second)], stream)
    (run / "config.json").write_text(
        json.dumps(
            dict(
                instance="tiny.json",
                instance_sha256=r1._sha(instance),
                seed=16,
                source=dict(commit="fixture"),
            )
        )
    )
    (run / "algorithm_config.json").write_text(json.dumps(dict(engine=dict(neighborhood=2))))
    (run / "checkpoints.jsonl").write_text(
        json.dumps(dict(elapsed=10, front=[first.evaluation.objectives]))
    )
    sealed = list(run.iterdir())
    manifest = [dict(path=f.name, bytes=f.stat().st_size, sha256=r1._sha(f)) for f in sealed]
    (run / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "out"
    (output / "runs").mkdir(parents=True)
    result = r1._trial_run((str(run), str(inputs), str(output), 96, 9912))
    assert result["sampled"] > 0
    assert result["nonreproducible_slots"] == 0
    assert result["max_parent_start_error"] == 0
    monkeypatch.setitem(sys.modules, "r1_offline", r1)
    analysis_spec = importlib.util.spec_from_file_location(
        "analyze_r1_offline", Path(__file__).parents[1] / "scripts/analyze_r1_offline.py"
    )
    assert analysis_spec is not None and analysis_spec.loader is not None
    analysis = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis)
    checked = analysis._analyze((str(output / "runs/tiny_s16.json"), str(tmp_path)))
    assert len(checked["rows"]) == result["sampled"]
    assert checked["conservative_pre_gain"] <= result["union_pre_gain"] + 1e-9
    assert checked["final_gain"] == pytest.approx(result["union_final_gain"])
    assert analysis._bootstrap_mean([1.0, 1.0], 1) == {"mean": 1.0, "ci95": [1.0, 1.0]}
    for item in manifest:
        assert hashlib.sha256((run / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    (run / "algorithm_config.json").write_text(json.dumps(dict(engine=dict(neighborhood=1))))
    with pytest.raises(AssertionError):
        r1._trial_run((str(run), str(inputs), str(output), 96, 9912))
