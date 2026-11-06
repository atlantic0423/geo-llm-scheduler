"""S4-B2 MOEA/D-structured arms: declared seams, decoder refine, seeds, terminal ops, runner."""

import gzip
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic
from test_s4_load_shift import _panel

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_decouple, s4_moead
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.trans_runner import digest, run_case
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.lsv2.decoder_refine import (
    DecoderRefiner,
    flow_safe_pass,
    tradeoff_pass,
    weight_class,
)
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, nondominated
from geo_llm_scheduler.lsv2.moead_search import MoeadMixin, MoeadOptions
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager


def test_arms_change_only_their_declared_switch():
    b2 = asdict(s4_decouple.search_config(s4_decouple.TERMINAL))
    cfg = s4_moead.arm_config(s4_moead.BASE, 7, 1200)
    assert cfg == s4_decouple.arm_config(s4_decouple.TERMINAL, 7, 1200)
    assert len(s4_moead.ARMS) == 9 and len(set(s4_moead.ARMS)) == 9
    for arm in s4_moead.ARMS:
        assert s4_moead.arm_config(arm, 7, 1200) == cfg
        fields = asdict(s4_moead.search_config(arm))
        diff = {k for k in b2 if b2[k] != fields[k]}
        opts = s4_moead.options(arm)
        assert diff == ({"operators"} if opts.terminal_ops else set())
        search = s4_moead.make_search(arm, cfg, CAL)
        assert isinstance(search, MoeadMixin) and search.moead == opts
        assert search.polish_reflow is False and search.polish_after_episode is False
        assert search.terminal_share == (0.10 if arm == "M3_P1_T10" else 0.05)
        record = s4_moead.configuration_record(arm, cfg, CAL)
        assert record["arm"] == arm and record["protocol"] == s4_moead.PROTOCOL
        assert record["decoupling"]["terminal_share"] == search.terminal_share
        assert record["moead"]["seed_mode"] == opts.seed_mode
    p1 = s4_moead.search_config("M2_P1_T05").operators
    assert "A7" not in p1 and "A8" not in p1 and set(p1) < set(b2["operators"])
    assert s4_moead.options(s4_moead.NO_LS).ls_enabled is False
    assert s4_moead.options("M5_P2B").refine == "weighted"
    with pytest.raises(ValueError):
        s4_moead.options("A1_NO_SHIFT")
    with pytest.raises(ValueError):
        MoeadOptions(terminal_ops=("A4",))
    with pytest.raises(ValueError):
        MoeadOptions(seed_mode="best")


def _genotypes(n: int, k: int = 6):
    problem = synthetic(n)[0]
    gs = initial_genotypes(problem, Config(population=20, seed=1), RNGManager(1).stream("x"))
    return problem, gs[:k]


def test_flow_safe_pass_is_deterministic_feasible_and_flow_invariant():
    problem, gs = _genotypes(8)
    for g in gs:
        s = decode(problem, g)
        base = evaluate(problem, g, s)
        r = flow_safe_pass(problem, g, s)
        assert r == flow_safe_pass(problem, g, s)
        e = evaluate(problem, g, r)
        assert e.feasible and abs(e.flow - base.flow) <= 1e-6
        assert e.bill <= base.bill + 1e-9
        assert all(a >= b - 1e-9 for a, b in zip(r.starts, s.starts))  # right moves only


def test_tradeoff_pass_respects_horizon_and_weighted_gain():
    problem, gs = _genotypes(8)
    for g in gs:
        s = flow_safe_pass(problem, g, decode(problem, g))
        base = evaluate(problem, g, s)
        scales = (max(1.0, 0.1 * base.flow), max(1e-6, 0.1 * base.bill))
        r = tradeoff_pass(problem, g, s, 5.0 / 6.0, scales)
        e = evaluate(problem, g, r)
        assert e.feasible and e.flow >= base.flow - 1e-6
        end = max(s.starts[o] + problem.profile(o).duration for o in range(1, len(s.starts), 2))
        new_end = max(r.starts[o] + problem.profile(o).duration for o in range(1, len(r.starts), 2))
        assert new_end <= end + 1e-6
        gain = (5 / 6) * (base.bill - e.bill) / scales[1] - (1 / 6) * (e.flow - base.flow) / scales[
            0
        ]
        assert gain >= -1e-9
    with pytest.raises(ValueError):
        tradeoff_pass(problem, gs[0], decode(problem, gs[0]), 1.0, (1.0, 1.0))


def test_weight_class_boundaries():
    assert [weight_class(w) for w in (0.0, 0.33, 1 / 3, 0.66, 2 / 3, 1.0)] == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        weight_class(1.5)


def test_gateway_refine_seam_caches_by_class_and_audits():
    problem, gs = _genotypes(6, 3)
    gw = EvaluationGateway(problem, Archive(), decode_cache_size=16)
    plain = gw.evaluate(gs[0])
    assert plain.schedule == decode(problem, gs[0])
    gw2 = EvaluationGateway(problem, Archive(), decode_cache_size=16)
    flows = [c.evaluation.flow for c in [plain]]
    gw2.refine = DecoderRefiner(problem, "weighted", (max(1.0, flows[0] * 0.1), 0.01))
    gw2.audit_decode_cache = True
    for cls in (0, 2, 0, 2):
        gw2.refine_class = cls
        c = gw2.evaluate(gs[0])
        assert c.evaluation == evaluate(problem, gs[0], c.schedule)
    assert gw2.counts["ssgs_cache_hits"] == 2 and len(gw2._decoded) == 2
    # Two fresh decodes plus two audited cache hits each run the refiner once.
    assert gw2.counts["refine"] == 4 and gw2.seconds["refine"] > 0
    gw2.refine_class = 0
    assert gw2.evaluate(gs[0]).schedule == flow_safe_pass(problem, gs[0], decode(problem, gs[0]))
    with pytest.raises(ValueError):
        DecoderRefiner(problem, "greedy")


def _initialized(n: int, arm: str, seconds: int = 1200):
    problem = synthetic(n)[0]
    cfg = s4_moead.arm_config(arm, 3, seconds)
    gw = EvaluationGateway(problem, Archive(), decode_cache_size=cfg.decode_cache_size)
    members = charged_initializer(problem, cfg, "BILL_MOEAD", {})(gw, RNGManager(3))
    search = s4_moead.make_search(arm, cfg, CAL)
    return problem, gw, members, search


def test_terminal_timing_ops_are_archive_only_and_exact():
    problem, gw, population, search = _initialized(6, "M2_P1_T05")
    search._setup(population, gw, RNGManager(3))
    before = [id(c) for c in population]
    search._terminal_sweep(population, gw, RNGManager(3), lambda: 0.0)
    assert [id(c) for c in population] == before
    assert search.moead_counts.get("terminal_calls:A7", 0) >= 1
    for member in gw.archive.members:
        assert member.evaluation == evaluate(problem, member.genotype, member.schedule)


def test_duplicate_and_region_seeds_follow_their_rules():
    problem, gw, population, search = _initialized(6, "M6_P3")
    search._setup(population, gw, RNGManager(3))
    by_objective = {}
    for m in gw.archive.members:
        by_objective.setdefault(m.evaluation.objectives, m)
    front = nondominated(list(by_objective))
    frame = FrozenFrame.of(front)
    # Force a duplicate cluster on the first front point.
    first = by_objective[front[0]]
    population[10] = first
    population[11] = first
    index = search._pick_index(frame, front, population)
    clusters = {}
    for j, m in enumerate(population):
        clusters.setdefault(m.evaluation.objectives, []).append(j)
    members = clusters[front[index]]
    assert len(members) >= 2
    assert search._target_weight in (search.lambdas[min(members)], search.lambdas[max(members)])
    assert search._execute_weight(frame, first) == search._target_weight
    for arm in ("M7_P4", "M8_P4_RANDOM"):
        _, gw4, pop4, s4 = _initialized(6, arm)
        s4._setup(pop4, gw4, RNGManager(5))
        s4.sampler = __import__(
            "geo_llm_scheduler.lsv2.frontier", fromlist=["SeedSampler"]
        ).SeedSampler("gap", RNGManager(5).stream("s"))
        objs = {}
        for m in gw4.archive.members:
            objs.setdefault(m.evaluation.objectives, m)
        f4 = nondominated(list(objs))
        fr4 = FrozenFrame.of(f4)
        for _ in range(10):
            i = s4._pick_index(fr4, f4, pop4)
            assert 0 <= i < len(f4) and s4._target_weight is None
        assert sum(v for k, v in s4.moead_counts.items() if k.startswith("region_seed")) == 10


@pytest.mark.parametrize("arm", s4_moead.ARMS)
def test_runner_executes_every_arm(tmp_path, arm, monkeypatch):
    monkeypatch.setattr(s4_decouple, "TERMINAL_SHARE", 0.999)
    monkeypatch.setattr(s4_moead, "_TERMINAL", {"M3_P1_T10": 0.999})
    instance = tmp_path / "input.json"
    save_instance(synthetic(4)[0], instance)
    out = tmp_path / "out"
    summary = run_case(
        arm, instance, out, seconds=6, calibration=CAL, max_generations=3, campaign_output=True
    )
    assert summary["independent_exact_audit"] == "passed" and summary["archive_size"] > 0
    record = json.loads((out / "algorithm_config.json").read_text())
    assert record["protocol"] == s4_moead.PROTOCOL and record["arm"] == arm
    with gzip.open(out / "local_search.json.gz", "rt") as f:
        data = json.load(f)
    assert data["moead"]["options"]["seed_mode"] == s4_moead.options(arm).seed_mode
    rows = data["polish"]
    assert all(r.get("phase") == "terminal" for r in rows)
    if arm == s4_moead.NO_LS:
        assert data["steps"] == []


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_moead")


def test_plan_keeps_nine_arms_on_forty_large_blocks(tmp_path, monkeypatch, runner):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(runner, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(cal))
    root = tmp_path / "full"
    runner.prepare(root, panel, cal, "example")
    m = runner.validate_inputs(root)
    assert m["runs"] == 360 and len(m["blocks"]) == 40 and m["replicates"] == 2
    sizes = []
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        sizes.append(spec["n"])
        assert set(spec["order"]) == set(s4_moead.ARMS)
        assert spec["config"]["M3_P1_T10"]["decoupling"]["terminal_share"] == 0.10
        assert "A7" not in spec["config"]["M2_P1_T05"]["lsv2"]["operators"]
    assert sizes.count(100) == 20 and sizes.count(200) == 20
    seen = []
    for k in range(3):
        shard = tmp_path / f"shard{k}"
        runner.prepare(shard, panel, cal, f"host{k}", f"{k}/3")
        seen += runner.validate_inputs(shard)["blocks"]
    assert sorted(seen) == sorted(m["blocks"])


def test_region_yield_widens_steps_only_where_ls_outyields_evolution():
    _, gw, pop, search = _initialized(6, "M7_P4")
    search._setup(pop, gw, RNGManager(9))
    from geo_llm_scheduler.lsv2.frontier import SeedSampler

    search.sampler = SeedSampler("gap", RNGManager(9).stream("s"))
    objs = {}
    for m in gw.archive.members:
        objs.setdefault(m.evaluation.objectives, m)
    front = nondominated(list(objs))
    frame = FrozenFrame.of(front)
    base_budget = search.config.budget
    search._rate_clock = search.timer()
    search._rate_ls_seconds = search.ls_seconds
    search._rate_time = dict(evo=10.0, ls=10.0)
    k = search.moead.regions
    search._rate = dict(evo=[5.0] * k, ls=[0.0] * k)  # evolution dominates everywhere
    for _ in range(5):
        search._pick_index(frame, front, pop)
        assert search.config.budget == base_budget
    search._rate = dict(evo=[0.0] * k, ls=[50.0] * k)  # LS dominates everywhere
    search._pick_index(frame, front, pop)
    assert search.config.budget == search.moead.wide_budget == 6
    assert search.moead_counts.get("region_wide", 0) >= 1
