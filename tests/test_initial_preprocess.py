"""Budget, frozen timing, same-parent prefixes and transactional artifact checks."""

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import (
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.initial_preprocess import ARMS, preprocess, select_cost_safe
from geo_llm_scheduler.operators.right_shift import polish
from geo_llm_scheduler.utils.numeric import identity


def parent(problem):
    """Return a feasible phenotype with nonzero right-shift slack."""
    gateway = EvaluationGateway(problem, Archive())
    return gateway.evaluate(
        Genotype((0, 1, 0, 1), (0, 1, 0, 1)), Schedule((0.0, 500.0, 200.0, 700.0))
    )


@pytest.mark.parametrize("arm", ARMS)
def test_exact_cap_freeze_and_parent_immutable(problem, arm):
    x = parent(problem)
    before = identity(x)
    y, archive, stats = preprocess(problem, x, arm, random.Random(71))
    assert identity(x) == before
    assert stats["exact"] == stats["independent_exact_verified"] <= stats["exact_budget"]
    assert y.evaluation.feasible and y.evaluation.flow <= x.evaluation.flow
    assert y.evaluation.bill <= x.evaluation.bill
    assert archive.members
    if arm.startswith("TIME"):
        assert y.genotype == x.genotype and stats["ssgs"] == 0
        assert y.schedule.starts[1::2] == x.schedule.starts[1::2]
        assert y.evaluation.flow == x.evaluation.flow
    if arm == "RAW":
        assert y is x and stats["exact"] == 0


@pytest.mark.parametrize("prefix", ("TIME", "STRUCT"))
def test_nested_budget_candidate_prefix_and_seed_replay(problem, prefix):
    x = parent(problem)
    low, _, a = preprocess(problem, x, prefix + "16", random.Random(88))
    high, _, b = preprocess(problem, x, prefix + "64", random.Random(88))
    repeated, _, c = preprocess(problem, x, prefix + "64", random.Random(88))
    assert a["ledger"] == b["ledger"][: len(a["ledger"])]
    assert identity(high) == identity(repeated) and b["ledger"] == c["ledger"]
    assert high.evaluation.bill <= low.evaluation.bill + 1e-9


def test_untruncated_timing_matches_original_polish(problem):
    x = parent(problem)
    gateway = EvaluationGateway(problem, Archive())
    expected = polish(problem, x, gateway)
    actual, _, stats = preprocess(problem, x, "TIME64", random.Random(1))
    assert not stats["cap_truncated"]
    assert stats["exact"] == gateway.counts["exact"]
    assert actual.evaluation.objectives == expected.evaluation.objectives


def test_timing_cannot_call_ssgs(problem, monkeypatch):
    x = parent(problem)

    def forbid(*args):
        raise AssertionError("Unexpected structural rebuild")

    monkeypatch.setattr("geo_llm_scheduler.engine.evaluation.decode", forbid)
    preprocess(problem, x, "TIME16", random.Random(1))


def test_cost_safe_rejects_cheaper_flow_worsening(problem):
    x = parent(problem)
    bad = replace(x, evaluation=replace(x.evaluation, flow=x.evaluation.flow + 1, tou=0.0))
    assert select_cost_safe(problem, x, [bad]) is x


def test_independent_exact_mismatch_is_hard_failure(problem, monkeypatch):
    x = parent(problem)
    monkeypatch.setattr(
        "geo_llm_scheduler.experiments.initial_preprocess.evaluate",
        lambda *args: replace(x.evaluation, tou=-1),
    )
    with pytest.raises(AssertionError, match="exact mismatch"):
        preprocess(problem, x, "TIME16", random.Random(1))


def test_unknown_and_infeasible_parent_refused(problem):
    x = parent(problem)
    with pytest.raises(ValueError):
        preprocess(problem, x, "unknown", random.Random(1))
    with pytest.raises(ValueError):
        preprocess(
            problem,
            replace(x, evaluation=replace(x.evaluation, feasible=False)),
            "RAW",
            random.Random(1),
        )


def test_complete_requires_every_file_and_matching_parent(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_initial_preprocess import FILES, complete

    assert not complete(tmp_path, "case", "RAW", {})
    folder = tmp_path / "runs/case/RAW"
    folder.mkdir(parents=True)
    (folder / "complete.json").write_text(json.dumps({"host": "foreign", "files": {}}))
    with pytest.raises(ValueError, match="hosts"):
        complete(tmp_path, "case", "RAW", {})
    assert "observations.json" in FILES


def test_real_budget_truncation_and_all_decode_times_frozen():
    p = ProblemInstance(
        tuple(Job(str(i), 0, Profile(100, 0.02, 0.1), Profile(100, 0.02, 0.1)) for i in range(20)),
        (Region("r", (Tariff(0, 1000, 1), Tariff(1000, 10000, 0.01)), 0),),
        (ServingInstance("p", 0, 16, 0, 100), ServingInstance("d", 0, 16, 0, 100)),
    )
    g = Genotype((0, 1) * 20, tuple(range(20)) * 2)
    s = Schedule(tuple(t for i in range(20) for t in (0.0, 2000.0 + 10 * i)))
    x = EvaluationGateway(p, Archive()).evaluate(g, s)
    low, _, a = preprocess(p, x, "TIME16", random.Random(1))
    high, _, b = preprocess(p, x, "TIME64", random.Random(1))
    assert a["exact"] == 16 and a["cap_truncated"]
    assert a["ledger"] == b["ledger"][:16]
    assert low.schedule.starts[1::2] == high.schedule.starts[1::2] == s.starts[1::2]


def test_small_worker_export_resume_and_hash_corruption(problem, tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_initial_preprocess as runner

    from geo_llm_scheduler.experiments.campaign.validation import file_hash
    from geo_llm_scheduler.experiments.runner import solution
    from geo_llm_scheduler.io.loaders import save_instance

    for name in ("instances", "parents", "specs", "ops/heartbeats"):
        (tmp_path / name).mkdir(parents=True)
    save_instance(problem, tmp_path / "instances/input.json")
    # Worker baseline is ordinary SSGS rather than the slack fixture.
    x = EvaluationGateway(problem, Archive()).evaluate(parent(problem).genotype)
    (tmp_path / "parents/case.json").write_text(json.dumps([solution(x)]))
    spec = {
        "key": "case",
        "methods": ARMS,
        "seed": 11001,
        "base_seed": 1,
        "tariff": "T",
        "rng_version": "test",
        "instance": "instances/input.json",
        "instance_sha256": file_hash(tmp_path / "instances/input.json"),
        "parent_population": "parents/case.json",
        "parent_population_sha256": file_hash(tmp_path / "parents/case.json"),
        "source_commit": "test",
        "source_hash": "test",
        "population": 1,
        "runner_sha256": file_hash(Path(runner.__file__)),
    }
    (tmp_path / "specs/case.json").write_text(json.dumps(spec))
    monkeypatch.setattr(runner, "canonical_source_hash", lambda: "test")
    monkeypatch.setattr(runner, "git_metadata", lambda: {"git_commit": "test", "git_dirty": False})
    runner.worker(tmp_path, "case")
    assert all(runner.complete(tmp_path, "case", arm, spec) for arm in ARMS)
    before = {arm: file_hash(tmp_path / "runs/case" / arm / "complete.json") for arm in ARMS}
    runner.worker(tmp_path, "case")
    assert before == {
        arm: file_hash(tmp_path / "runs/case" / arm / "complete.json") for arm in ARMS
    }
    (tmp_path / "runs/case/RAW/population.json").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.complete(tmp_path, "case", "RAW", spec)
