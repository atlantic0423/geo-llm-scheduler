"""Opt-in pathways share exact/archive/cap semantics and deterministic compact logs."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.p1_observation import SemanticRecorder
from geo_llm_scheduler.experiments.p2 import P2_ARMS, p2_config
from geo_llm_scheduler.experiments.p2_validation import validate_local_p2


@pytest.mark.parametrize("arm", P2_ARMS)
def test_opt_in_arms_preserve_exact_gt_archive_and_replay(problem, arm):
    config = p2_config(
        arm,
        Config(
            method="full",
            population=6,
            neighborhood=3,
            generations=2,
            trigger_mode="always",
            rl_steps=2,
            seed=19,
        ),
    )
    signatures = []
    for retain in (True, False):
        recorder = SemanticRecorder()
        audited = []

        def observer(event, data):
            recorder(event, data)
            if event == "candidate":
                c = data["candidate"]
                assert c.evaluation == evaluate(problem, c.genotype, c.schedule)
                audited.append(c)

        result = run(problem, config, retain_trace=retain, observer=observer)
        assert result.gateway.counts["exact"] == len(audited)
        assert result.offspring_count == 12
        assert len(result.trace) == (12 if retain else 0)
        assert all(c in audited for c in result.archive.members)
        signatures.append(recorder.signature(result))
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("arm", P2_ARMS)
def test_opt_in_arms_stop_exactly_at_cap(problem, arm):
    config = p2_config(
        arm,
        Config(
            method="full",
            population=6,
            neighborhood=3,
            generations=20,
            trigger_mode="always",
            rl_steps=2,
            exact_evaluation_cap=11,
            seed=23,
        ),
    )
    result = run(problem, config, retain_trace=False)
    assert result.termination_reason == "exact_evaluation_cap"
    assert result.gateway.counts["exact"] == 11 and not result.trace


def test_local_validator_shared_initialization_and_provenance(tmp_path):
    report = validate_local_p2(tmp_path / "validation", seeds=(17,))
    assert report["runs"] == 12
    assert len({r["initial_hash"] for r in report["rows"]}) == 1
    assert all(
        r["replay_equal"] and r["audited_exact"][0] == r["counts"]["exact"] for r in report["rows"]
    )
    assert not report["server_deployed"] and not report["scientific_gain_verified"]
    assert (tmp_path / "validation/validation.json").is_file()
    with pytest.raises(FileExistsError):
        validate_local_p2(tmp_path / "validation", seeds=(17,))
    with pytest.raises(ValueError):
        validate_local_p2(tmp_path / "invalid", seeds=(17, 17))


@pytest.mark.parametrize("arm", P2_ARMS)
def test_review_templates_match_single_factor_builder(arm):
    base = replace(p2_config("F6"), generations=1_000_000, seconds=600)
    assert load_config(f"configs/p2_local/{arm}.yaml") == p2_config(arm, base)
