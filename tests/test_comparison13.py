"""Full algorithm roster, paired native cases, initialization and solver counterexamples."""

import json
import random
from collections import Counter
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.classic_baselines import (
    decode_keys,
    encode_keys,
    epsilon_contributions,
    grid_repository,
    ibea_select,
    normalized_objectives,
    run_classic,
    select_leader,
    spea_fitness,
    spea_select,
    update_repository,
)
from geo_llm_scheduler.experiments.comparison13 import (
    ARMS,
    PRESSURES,
    SIZES,
    TOPOLOGIES,
    arm_config,
    balanced_order,
    campaign_specs,
    case_problem,
    initial_trials,
    initialization_factory,
)
from geo_llm_scheduler.experiments.initial_bill_benchmark import benchmark_population
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.utils.rng import RNGManager


def point(x, y, tag=0):
    return Candidate(
        Genotype((tag, tag), (0, 0)),
        Schedule((0.0, 0.0)),
        EvaluationResult(True, float(x), float(y), (0.0,), ((0.0,),)),
        str(tag),
    )


def test_roster_keeps_three_initializers_separate_from_macro_budget():
    assert len(ARMS) == len(set(ARMS)) == 13
    for arm in ARMS:
        config = arm_config(arm, 3, 50)
        assert config.population == 100 and config.fixed_budget == 6
        assert initial_trials(arm) == (2 if arm == "M11_I2" else 1 if arm == "M11_I1" else 4)
    assert arm_config("M00_I4", 1, 5).a3_region_policy == "load"
    assert arm_config("M10_I4", 1, 5).replacement_policy == "birth"
    assert arm_config("M01_I4", 1, 5).a3_region_policy == "load"
    assert arm_config("DS_I4", 1, 5).budget_policy == "severity_v2"
    assert arm_config("DC_I4", 1, 5).budget_policy == "coverage_v2"
    for arm in ARMS[8:]:
        assert not arm_config(arm, 1, 5).polish
    with pytest.raises(ValueError):
        arm_config("missing", 1, 5)
    with pytest.raises(ValueError):
        initial_trials("missing")


def test_odd_williams_position_and_directed_predecessor_balance():
    orders = [balanced_order(i) for i in range(26)]
    assert all(set(row) == set(ARMS) for row in orders)
    for i in range(13):
        assert Counter(row[i] for row in orders) == dict.fromkeys(ARMS, 2)
    adjacent = Counter((a, b) for row in orders for a, b in zip(row, row[1:]))
    assert len(adjacent) == 13 * 12 and set(adjacent.values()) == {2}


def test_matrix_rotates_size_and_has_four_complete_versions_per_parent():
    specs = campaign_specs(2)
    assert len(specs) * 13 == 8320
    assert [s["n"] for s in specs] == list(SIZES) * (len(specs) // 4)
    assert len({s["key"] for s in specs}) == len(specs)
    assert len({s["base_seed"] for s in specs}) == 160
    for base in {s["base_seed"] for s in specs}:
        assert {(s["variant"], s["seed"]) for s in specs if s["base_seed"] == base} == {
            (v, s) for v in (0, 1) for s in (16101, 16201)
        }
    assert not {s["base_seed"] for s in campaign_specs(5)} & {
        s["base_seed"] for s in campaign_specs(1, pilot=True)
    }


@pytest.mark.parametrize("r,m", TOPOLOGIES)
@pytest.mark.parametrize("pressure", PRESSURES)
def test_paired_inputs_change_only_declared_factor(r, m, pressure):
    a, b = [case_problem(20, r, m, 171, pressure, v) for v in (0, 1)]
    assert len(a.jobs) == len(b.jobs) == 20
    assert len(a.instances) == len(b.instances) == m
    assert len({tuple(t.price for t in region.tariffs) for region in a.regions}) == r
    assert a != b and a == case_problem(20, r, m, 171, pressure, 0)
    if pressure == "arrival":
        assert a.regions == b.regions and a.instances == b.instances
        assert [replace(j, release=0.0) for j in a.jobs] == [
            replace(j, release=0.0) for j in b.jobs
        ]
    elif pressure == "kv":
        assert a.regions == b.regions and a.instances == b.instances
        assert [replace(j, kv_delay=0.0) for j in a.jobs] == [
            replace(j, kv_delay=0.0) for j in b.jobs
        ]
    elif pressure == "demand":
        assert a.jobs == b.jobs and a.instances == b.instances
        assert [replace(x, demand_rate=0.0) for x in a.regions] == [
            replace(x, demand_rate=0.0) for x in b.regions
        ]
    else:
        assert a.jobs == b.jobs and a.regions == b.regions
        assert sorted(Counter(x.region for x in a.instances).values()) == sorted(
            Counter(x.region for x in b.instances).values()
        )
        assert Counter(x.region for x in a.instances)[0] > Counter(x.region for x in b.instances)[0]


@pytest.mark.parametrize("trials", (1, 2, 4))
def test_charged_initialization_matches_original_and_archives_unselected_trials(trials):
    problem = case_problem(12, 3, 12, 20, "arrival", 0)
    config = Config(seed=33)
    reference, _, _ = benchmark_population(
        problem, config, "BILL10", 1133, accelerated=True, trials=trials
    )
    gateway = EvaluationGateway(problem, Archive())
    evidence = {}
    factory = initialization_factory(problem, config, trials, 1133, evidence)
    population = factory(gateway, RNGManager(33))
    assert population == reference
    assert gateway.counts["exact"] == 100 + 10 * trials
    assert gateway.archive.attempts == 100 + 10 * trials
    assert evidence["bank_stats"]["independent_exact_verified"] == 0
    assert evidence["trials"] == trials
    assert Counter(c.origin for c in population) == {
        **{f"initial_mode_{i}": 9 for i in range(10)},
        "bill_seed_BILL": 10,
    }


def test_spea2_strength_density_and_lexicographic_truncation():
    population = [point(0, 0), point(1, 1), point(2, 2)]
    fitness, _ = spea_fitness(population)
    assert fitness[0] < 1 < fitness[1] < fitness[2]
    assert spea_select(population, 2)[0] == population[:2]
    front = [point(i, 10 - i, i) for i in (0, 1, 2, 9, 10)]
    selected, _ = spea_select(front, 3)
    assert front[0] in selected and front[-1] in selected
    with pytest.raises(ValueError):
        spea_select(front, 6)
    with pytest.raises(ValueError):
        spea_fitness([])


def test_ibea_epsilon_sign_incremental_deletion_and_scale_invariance():
    population = [point(0, 0), point(1, 1), point(2, 2)]
    matrix = epsilon_contributions(population)
    assert matrix[0][2] > matrix[2][0]
    assert ibea_select(population, 1)[0] == population[:1]
    assert normalized_objectives([point(1, 2), point(1, 5)]) == [(0.0, 0.0), (0.0, 1.0)]
    transformed = [
        point(c.evaluation.flow * 100 + 30, c.evaluation.bill * 0.1 + 9) for c in population
    ]
    assert ibea_select(transformed, 1)[0] == transformed[:1]
    with pytest.raises(ValueError):
        epsilon_contributions([])
    with pytest.raises(ValueError):
        ibea_select(population, 0)


def test_random_keys_roundtrip_legal_pd_paths_and_boundary_keys():
    problem = case_problem(6, 3, 6, 81, "kv", 1)
    config = Config(population=8, neighborhood=4, seed=2)
    for g in initial_genotypes(problem, config, RNGManager(2).stream("initialization")):
        assert decode_keys(problem, encode_keys(problem, g)) == g
    for bound in (0.0, 1.0):
        g = decode_keys(problem, [bound] * 18)
        assert Counter(g.os) == dict.fromkeys(range(6), 2)
        assert all(
            problem.instances[g.ms[2 * j]].region == problem.instances[g.ms[2 * j + 1]].region
            for j in range(6)
        )
    with pytest.raises(ValueError):
        decode_keys(problem, [0.0])
    with pytest.raises(ValueError):
        decode_keys(problem, [float("nan")] * 18)


def test_native_mopso_repository_is_bounded_nd_and_rng_explicit():
    points = [point(i, 10 - i, i) for i in range(11)] + [point(100, 100, 99)]
    a = grid_repository(points, 4, random.Random(1))
    b = grid_repository(points, 4, random.Random(1))
    assert a == b and len(a) == 4 and points[-1] not in a
    assert select_leader(a, random.Random(5)) in a
    assert update_repository(a, point(100, 100, 998), 4, random.Random(3)) == a
    better = point(-1, -1, 999)
    assert update_repository(a, better, 4, random.Random(3)) == [better]
    assert len(update_repository(a, point(5.2, 4.8, 30), 4, random.Random(3))) == 4


def test_ibea_incremental_removals_match_full_recomputed_active_losses():
    population = [
        point(x, y, i) for i, (x, y) in enumerate(((0, 9), (3, 5), (6, 3), (9, 0), (4, 7), (8, 8)))
    ]
    contribution = epsilon_contributions(population)
    selected = list(range(6))
    while len(selected) > 3:
        loss = {i: sum(contribution[j][i] for j in selected if j != i) for i in selected}
        selected.remove(max(selected, key=lambda i: (loss[i], i)))
    assert ibea_select(population, 3)[0] == [population[i] for i in selected]


@pytest.mark.parametrize("method", ("SPEA2", "EPS_IBEA", "RK_MOPSO"))
def test_classics_exact_cap_and_seeded_replay(method):
    problem = case_problem(6, 3, 6, 31, "arrival", 0)
    config = Config(
        population=8, neighborhood=4, generations=10, seed=2, exact_evaluation_cap=9, polish=False
    )
    genes = initial_genotypes(problem, config, RNGManager(2).stream("initialization"))

    def initialize(gateway, streams):
        return [gateway.evaluate(g) for g in genes]

    a, b = [run_classic(problem, config, method, initialize) for _ in (0, 1)]
    assert a.termination_reason == "exact_evaluation_cap" and a.gateway.counts["exact"] == 9
    assert a.population == b.population and a.archive.members == b.archive.members


@pytest.mark.parametrize("solver", (run, run_nsga2))
def test_initializer_cost_is_inside_total_budget_and_conflicting_input_rejected(solver):
    import time

    problem = case_problem(6, 3, 6, 31, "arrival", 0)
    config = Config(
        population=8, neighborhood=4, generations=10, seed=2, seconds=0.001, polish=False
    )
    genes = initial_genotypes(problem, config, RNGManager(2).stream("initialization"))

    def initialize(gateway, streams):
        time.sleep(0.01)
        return [gateway.evaluate(g) for g in genes]

    result = solver(problem, config, initializer=initialize)
    assert result.termination_reason == "time_budget" and result.offspring_count == 0
    assert result.elapsed >= 0.01
    with pytest.raises(ValueError, match="Choose"):
        solver(problem, config, genes, initializer=initialize)


@pytest.mark.parametrize("method", ("MOEAD", "NSGAII", "SPEA2", "EPS_IBEA", "RK_MOPSO"))
def test_all_baselines_charge_initializer_and_exact_replay(method):
    problem = case_problem(6, 3, 6, 31, "arrival", 0)
    config = Config(
        population=8,
        neighborhood=4,
        generations=2,
        seed=2,
        method="nsga2" if method == "NSGAII" else "plain",
        polish=False,
    )
    genes = initial_genotypes(problem, config, RNGManager(2).stream("initialization"))

    def initializer(gateway, streams):
        return [gateway.evaluate(g, origin="test_initial") for g in genes]

    if method == "MOEAD":
        result = run(problem, config, initializer=initializer)
    elif method == "NSGAII":
        result = run_nsga2(problem, config, initializer=initializer)
    else:
        result = run_classic(problem, config, method, initializer)
    assert result.offspring_count > 0
    assert result.gateway.counts["exact:test_initial"] == 8
    assert len(result.population) == 8
    for c in result.population + result.archive.members:
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
    if method in ("MOEAD", "NSGAII"):
        old = (
            run_nsga2(problem, config, genes) if method == "NSGAII" else run(problem, config, genes)
        )
        assert [c.genotype for c in old.population] == [c.genotype for c in result.population]


def test_campaign_input_hash_tamper_and_roundtrip(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_comparison13 as runner

    monkeypatch.setattr(
        runner, "campaign_specs", lambda *a, **kw: [campaign_specs(1, pilot=True)[0]]
    )
    manifest = runner.prepare(tmp_path, 1, pilot=True)
    assert runner.validate_inputs(tmp_path) == manifest
    assert manifest["runs"] == 13
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path, 1)
    case = tmp_path / next(f for f in manifest["files"] if f.startswith("inputs"))
    case.write_text(json.dumps({}))
    with pytest.raises(AssertionError):
        runner.validate_inputs(tmp_path)
