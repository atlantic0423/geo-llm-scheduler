"""Near-capacity adversarial checks with both independent resource dimensions."""

from dataclasses import replace

from geo_llm_scheduler.domain.models import Genotype, Profile, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate


def test_compute_and_vram_boundaries(problem):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((0, 100, 0, 100))
    for compute, vram, feasible in ((0.5, 4, True), (0.50001, 4, False), (0.5, 4.00001, False)):
        jobs = tuple(
            replace(j, prefill=Profile(100, compute, vram), decode=Profile(200, compute, vram))
            for j in problem.jobs
        )
        assert evaluate(replace(problem, jobs=jobs), g, schedule).feasible is feasible
