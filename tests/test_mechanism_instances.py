"""Input validity, reproducibility, physical stress mechanisms and exact audit witnesses."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from random import Random

import pytest

from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.mechanism_audit import audit_problem, probe_candidates
from geo_llm_scheduler.experiments.mechanism_instances import (
    FAMILIES,
    input_features,
    mechanism_problem,
    panel_specs,
)
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.scheduling.ssgs import decode


def test_panel_balanced_independent_and_stable():
    specs = panel_specs()
    assert len(specs) == len({s.key for s in specs}) == len({s.seed for s in specs}) == 120
    assert Counter(s.jobs for s in specs) == {50: 40, 100: 40, 200: 40}
    assert set(Counter((s.jobs, s.family) for s in specs).values()) == {4}
    assert Counter(s.regions for s in specs) == {3: 60, 5: 60}
    assert len(FAMILIES) == 10
    spec = specs[0]
    assert mechanism_problem(spec) == mechanism_problem(spec)
    assert mechanism_problem(spec) != mechanism_problem(replace(spec, seed=spec.seed + 1))


@pytest.mark.parametrize("spec", panel_specs(), ids=lambda s: s.key)
def test_every_input_roundtrip_feasible_and_rightshift_structures(spec, tmp_path):
    p = mechanism_problem(spec)
    features = input_features(p)
    assert features["concurrent_prefill_jobs"] >= spec.jobs // 5
    assert features["long_decode_jobs"] > 0
    assert min(len(paths(p, j)) for j in range(spec.jobs)) >= 12
    assert all(m.phases == (0, 1) and m.vram == 16 for m in p.instances)
    for region in p.regions:
        assert all(a.end == b.start for a, b in zip(region.tariffs, region.tariffs[1:]))
        assert any(a.price > 2 * b.price for a, b in zip(region.tariffs, region.tariffs[1:]))
    rng = Random(spec.seed + 1)
    ms = tuple(m for j in range(spec.jobs) for m in rng.choice(paths(p, j)))
    order = [j for j in range(spec.jobs) for _ in (0, 1)]
    rng.shuffle(order)
    genotype = Genotype(ms, tuple(order))
    schedule = decode(p, genotype)
    assert evaluate(p, genotype, schedule).feasible
    assert schedule.horizon(p) < min(r.tariffs[-1].end for r in p.regions)
    target = tmp_path / "input.json"
    save_instance(p, target)
    assert load_instance(target) == p


def test_reject_invalid_design_and_all_inputs_physically_distinct():
    spec = panel_specs()[0]
    for kwargs in ({"jobs": 51}, {"family": "unknown"}, {"regions": 2}, {"replicate": 0}):
        with pytest.raises(ValueError):
            mechanism_problem(replace(spec, **kwargs))
    fingerprints = {
        hashlib.sha256(
            json.dumps(asdict(mechanism_problem(s)), sort_keys=True).encode()
        ).hexdigest()
        for s in panel_specs()
    }
    assert len(fingerprints) == 120
    with pytest.raises(ValueError):
        probe_candidates([])


def test_real_bill4_audit_verifies_exact_and_does_not_mutate_input():
    p = mechanism_problem(next(s for s in panel_specs() if s.family == "F09"))
    before = asdict(p)
    audit = audit_problem(p, operators=False)
    assert asdict(p) == before
    assert audit["initial_population_count"] == 100
    assert audit["initial_exact_checks"] == 140
    assert audit["sampled_front_count"] in range(1, 6)
    assert len(audit["rows"]) == audit["sampled_front_count"]
    assert audit["rightshift_witnesses"], "The frozen F09 example should retain real bill headroom"
    for row in audit["rows"]:
        assert row["feasible"] == row["proposals"] == row["dominates_parent"]
        assert row["initial_hv_gain"] >= 0
    for witness in audit["rightshift_witnesses"]:
        a, b = witness["before"], witness["after"]
        assert b["evaluation"]["tou"] + sum(b["evaluation"]["demand"]) < (
            a["evaluation"]["tou"] + sum(a["evaluation"]["demand"])
        )
        assert a["schedule"]["starts"][1::2] == b["schedule"]["starts"][1::2]
