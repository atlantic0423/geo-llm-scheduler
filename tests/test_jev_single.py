"""Single-candidate scope, complete A6 ranking, original repair and exact regressions."""

import math
import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.jev_decisions import JevSettings
from geo_llm_scheduler.experiments.jev_search import SINGLE_ARM, JevPolicy, study_config
from geo_llm_scheduler.experiments.jev_single import BACKGROUND, PROMPT_VERSION, SingleJevPolicy
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.utils.rng import RNGManager


class RankingJev:
    def __init__(self, action="A6", tied=False, invalid=False):
        self.requests = []
        self.action, self.tied, self.invalid = action, tied, invalid

    def __call__(self, payload, timeout):
        assert timeout > 0
        self.requests.append(payload)
        if self.invalid:
            return {}
        answers = {}
        for name, q in payload["questions"].items():
            if q["type"] == "noul":
                answers[name] = dict(type="noul", noul=0)
                continue
            keys = list(q["criteria"])
            if name == "action":
                probs = {k: float(k == self.action) for k in keys}
            elif payload["state"].get("decision_kind") == "a6_job_ranking":
                values = [1 if self.tied else i + 1 for i in range(len(keys))]
                probs = {k: v / sum(values) for k, v in zip(keys, values)}
            else:
                probs = {k: float(i == 0) for i, k in enumerate(keys)}
            best = max(probs, key=probs.get)
            answers[name] = dict(type="choice", choice=best, probabilities=probs, confidence=1)
        return dict(
            model=payload["model"], answers=answers, usage=dict(input_tokens=100, output_tokens=10)
        )


def many_jobs(problem, n):
    p = replace(
        problem,
        jobs=tuple(replace(problem.jobs[0], name=str(i)) for i in range(n)),
        regions=tuple(
            replace(r, tariffs=(replace(r.tariffs[0], end=1_000_000),)) for r in problem.regions
        ),
    )
    gateway = EvaluationGateway(p, Archive())
    g = Genotype((0, 1) * n, tuple(j for j in range(n) for _ in (0, 1)))
    return p, gateway, gateway.evaluate(g)


@pytest.mark.parametrize("n", [1, 20, 200])
def test_a6_one_request_selects_true_top_fraction_and_original_repair_is_exact(problem, n):
    p, gateway, x = many_jobs(problem, n)
    fake = RankingJev()
    policy = SingleJevPolicy(JevSettings(epsilon_g=0.9), 1000, 1, fake)
    cfg = study_config(SINGLE_ARM, 1, 1000)
    state = policy.snapshot(gateway, x, (0.5, 0.5))
    out = policy.candidates(6, state).propose(p, x, 1, cfg, random.Random(7))
    assert len(fake.requests) == 1 and set(
        fake.requests[0]["questions"]["selection"]["criteria"]
    ) == {str(i) for i in range(n)}
    count = min(n, max(2, math.ceil(0.1 * n)))
    selection = policy.session.selections[-1]
    assert selection["selected"] == list(reversed(range(n - count, n)))
    assert len(set(selection["selected"])) == count
    assert selection["fallback"] is None and selection["selected_count"] == count
    assert out.attempts == 1 and len(out.proposals) <= 1
    for proposal in out.proposals:
        c = gateway.evaluate(proposal.genotype)
        assert c.evaluation.feasible and c.evaluation == evaluate(p, c.genotype, c.schedule)
        for j in set(range(n)) - set(selection["selected"]):
            assert c.genotype.ms[2 * j : 2 * j + 2] == x.genotype.ms[2 * j : 2 * j + 2]
    assert x.genotype.ms == (0, 1) * n
    assert fake.requests[0]["state"]["sampling_rule"]["a6_epsilon_sampling"] is False
    assert fake.requests[0]["state"]["selected_count"] == count


@pytest.mark.parametrize("invalid,tied", [(True, False), (False, True)])
def test_a6_fallback_and_ties_replay_explicit_rng_without_truncation(problem, invalid, tied):
    p, gateway, x = many_jobs(problem, 20)
    selections = []
    for _ in range(2):
        fake = RankingJev(invalid=invalid, tied=tied)
        policy = SingleJevPolicy(JevSettings(epsilon_g=0.9), 1000, 1, fake)
        state = policy.snapshot(gateway, x, (0.5, 0.5))
        policy.candidates(6, state).propose(
            p, x, 1, study_config(SINGLE_ARM, 1, 1000), random.Random(7)
        )
        selections.append(policy.session.selections[-1])
        assert len(fake.requests) == 1
    assert selections[0] == selections[1]
    assert len(set(selections[0]["selected"])) == 2
    assert bool(selections[0]["fallback"]) == invalid


def test_a6_refuses_more_than_choice_limit_and_b6_without_sending_request(problem):
    p, gateway, x = many_jobs(problem, 256)
    fake = RankingJev()
    policy = SingleJevPolicy(JevSettings(epsilon_g=0.9), 1000, 1, fake)
    generator = policy.candidates(6, policy.snapshot(gateway, x, (0.5, 0.5)))
    with pytest.raises(ValueError, match="255"):
        generator.propose(p, x, 1, study_config(SINGLE_ARM, 1, 1000), random.Random(1))
    with pytest.raises(ValueError, match="one candidate"):
        generator.propose(p, x, 6, study_config(SINGLE_ARM, 1, 1000), random.Random(1))
    assert not fake.requests


@pytest.mark.parametrize("action", range(1, 9))
def test_all_single_actions_bound_exact_to_one_and_bind_prompt(problem, action):
    p, gateway, x = many_jobs(problem, 3)
    policy = SingleJevPolicy(JevSettings(epsilon_g=0.9), 1000, 1, RankingJev())
    cfg = study_config(SINGLE_ARM, 1, 1000)
    state = policy.snapshot(gateway, x, (0.5, 0.5))
    out = policy.candidates(action, state).propose(p, x, 1, cfg, random.Random(8))
    assert len(out.proposals) <= 1
    result = execute(
        out,
        x,
        1,
        (0.5, 0.5),
        NormalizationContext(gateway.ideal, x.evaluation.objectives),
        gateway,
        f"A{action}",
    )
    assert result.exact_evaluations <= 1
    for c in result.evaluated:
        assert c.evaluation == evaluate(p, c.genotype, c.schedule) and c.evaluation.feasible
    assert state["sampling_rule"]["candidate_budget"] == 1
    assert state["sampling_rule"]["treatment"] == SINGLE_ARM
    assert "at most ONE complete candidate" in BACKGROUND and "SIX" not in BACKGROUND
    assert policy.session.prompt_version == PROMPT_VERSION


def test_fresh_actions_stop_feedback_and_b1_do_not_change_b6_treatment(problem):
    p, gateway, x = many_jobs(problem, 3)
    fake = RankingJev(action="A1")
    policy = SingleJevPolicy(JevSettings(epsilon_g=0.9), 1000, 1, fake)
    cfg = replace(study_config(SINGLE_ARM, 1, 1000), population=6, neighborhood=2)
    _, rows = policy(
        x, 1, 0, 0, weights(6), x.evaluation.objectives, gateway, None, cfg, RNGManager(1)
    )
    assert len(rows) == 5 and all(r["exact_evaluations"] <= 1 for r in rows)
    assert sum("action" in r["questions"] for r in fake.requests) == 5
    assert sum("stop" in r["questions"] for r in fake.requests) == 4
    assert all(r["state"]["problem_contract"] == BACKGROUND for r in fake.requests)
    assert all(r["state"]["prompt_version"] == PROMPT_VERSION for r in fake.requests)
    assert study_config("JEV_E09", 1, 1000).fixed_budget == 6
    old = JevPolicy(JevSettings(epsilon_g=0.9), 1000, 1, RankingJev())
    assert "SIX" in old.session.background and old.session.prompt_version != PROMPT_VERSION
