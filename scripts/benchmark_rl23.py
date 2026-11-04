"""Fixed-work eligibility-cache benchmark; no claims about end-to-end speedup."""

import json
import random
import statistics
from pathlib import Path
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.rl23_support import StudyPolicy
from geo_llm_scheduler.initialization.generators import generate
from geo_llm_scheduler.rl.controller import Controller


def main() -> None:
    """Compare identical 200-job masks with and without one-parent cache reuse."""
    p = case_problem(200, 3, 12, 202610160023, "arrival", 0)
    gateway = EvaluationGateway(p, Archive())
    candidate = gateway.evaluate(generate(p, 0, random.Random(17), 0.1))
    cfg = Config()
    samples = {"reference": [], "cached": []}
    outputs = {}
    for repeat in range(5):
        for mode in ("reference", "cached") if repeat % 2 == 0 else ("cached", "reference"):
            policy = StudyPolicy("LEGAL_OPPORTUNITY_MASK")
            learner = policy.learner(Controller(cfg), cfg)
            start = perf_counter()
            for i in range(20000):
                if mode == "reference":
                    policy._mask_candidate = None
                policy.mask(learner, i % 42, p, candidate)
            samples[mode].append(perf_counter() - start)
            outputs[mode] = learner.allowed
    assert outputs["reference"] == outputs["cached"]
    result = dict(
        iterations=20000,
        n=200,
        samples=samples,
        medians={k: statistics.median(v) for k, v in samples.items()},
        allowed=outputs["cached"][0],
        outputs_identical=True,
        evaluations_performed=gateway.counts["exact"],
        scope="eligibility support only; not whole-run or quality benchmark",
    )
    Path("research/rl23_20261021/ELIGIBILITY_BENCHMARK.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
