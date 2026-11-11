"""S4-B2 MOEA/D-structured arms: declared seams, decoder refine, seeds, terminal ops, runner."""

import gzip
import importlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

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
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, SeedSampler, nondominated
from geo_llm_scheduler.lsv2.moead_search import (
    TRACE_FIELDS,
    TRACE_WBINS,
    MoeadMixin,
    MoeadOptions,
    degenerate,
)
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


@pytest.fixture
def confirm(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_p1_confirm")


def test_confirmation_runs_two_arms_on_every_unscreened_block(tmp_path, monkeypatch, confirm):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(confirm.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(confirm.base, "CAL", digest(cal))
    root = tmp_path / "full"
    confirm.prepare(root, panel, cal, "example")
    m = confirm.validate_inputs(root)
    assert m["runs"] == 400 and len(m["blocks"]) == 200 and tuple(m["arms"]) == confirm.ARMS
    assert m["design"] == confirm.DESIGN and m["level"] == "development confirmation"
    cells, primary, orders = Counter(), Counter(), set()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        cells[(spec["n"], spec["seed"])] += 1
        primary[spec["primary"]] += 1
        assert block == f"{spec['instance']}_s{spec['seed']}"
        assert spec["seconds"] == {50: 600, 100: 1200, 200: 2400}[spec["n"]]
        assert spec["config"]["M2_P1_T05"]["moead"]["terminal_ops"] == ["A7", "A8"]
        assert "A7" not in spec["config"]["M2_P1_T05"]["lsv2"]["operators"]
        assert "A7" in spec["config"]["M0_B2"]["lsv2"]["operators"]
        assert spec["config"]["M2_P1_T05"]["decoupling"]["terminal_share"] == 0.05
        orders.add(tuple(spec["order"]))
    assert cells == {
        (50, 1): 40,
        (100, 1): 20,
        (200, 1): 20,
        (50, 2): 40,
        (100, 2): 40,
        (200, 2): 40,
    }
    assert primary == {True: 120, False: 80}
    assert orders == {confirm.ARMS, tuple(reversed(confirm.ARMS))}
    screen = tmp_path / "screen"
    confirm.base.prepare(screen, panel, cal, "example")
    used = set(confirm.base.validate_inputs(screen)["blocks"])
    assert len(used) == 40 and not used & set(m["blocks"])
    assert all(confirm.screened(dict(key=b[:-3], n=int(b[4:7]), seed=1)) for b in used)
    seen = []
    for k in range(3):
        shard = tmp_path / f"shard{k}"
        confirm.prepare(shard, panel, cal, f"host{k}", f"{k}/3")
        seen += confirm.validate_inputs(shard)["blocks"]
    assert sorted(seen) == sorted(m["blocks"]) and len(seen) == len(set(seen))
    spec_path = root / "specs" / (m["blocks"][0] + ".json")
    spec = json.loads(spec_path.read_text())
    spec["seed"] = 1 if spec["seed"] == 2 else 2
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(AssertionError):
        confirm.validate_inputs(root)


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


def test_combination_arms_add_only_regions_and_the_record_only_trace():
    cfg = s4_moead.arm_config("N0_P1", 7, 1200)
    assert set(s4_moead.COMBO_ARMS) & set(s4_moead.ARMS) == set()
    assert s4_moead.EXTRA_ARMS == (
        s4_moead.ARMS
        + s4_moead.COMBO_ARMS
        + s4_moead.NEXT_ARMS
        + s4_moead.BANDIT_ARMS
        + s4_moead.SHARE_ARMS
        + s4_moead.WIDEN_ARMS
        + s4_moead.SPACE_ARMS
        + s4_moead.INVEST_ARMS
        + s4_moead.GUIDED_ARMS
        + s4_moead.COST_ARMS
    )
    p1 = s4_moead.options("M2_P1_T05")
    assert s4_moead.options("N0_P1") == replace(p1, trace=True)
    want = dict(
        N1_P1_P4R5=replace(p1, seed_mode="region_yield", trace=True),
        N2_P1_P4R3=replace(p1, seed_mode="region_yield", regions=3, trace=True),
        N3_P1_P4R3_RANDOM=replace(p1, seed_mode="region_random", regions=3, trace=True),
    )
    reference = s4_moead.configuration_record("M2_P1_T05", cfg, CAL)
    assert "trace" not in reference["moead"]
    for arm in s4_moead.COMBO_ARMS:
        opts = s4_moead.options(arm)
        assert opts == want.get(arm, opts) and opts.trace and opts.terminal_ops == ("A7", "A8")
        assert s4_moead.arm_config(arm, 7, 1200) == cfg
        assert s4_moead.search_config(arm) == s4_moead.search_config("M2_P1_T05")
        assert s4_moead.terminal_share(arm) == 0.05
        record = s4_moead.configuration_record(arm, cfg, CAL)
        assert record["moead"]["trace"] is True and record["moead"]["trace_seconds"] == 30.0
        same = {k: v for k, v in record.items() if k not in ("arm", "moead")}
        assert same == {k: v for k, v in reference.items() if k not in ("arm", "moead")}
        diff = {k for k in record["moead"] if record["moead"][k] != reference["moead"].get(k)}
        assert diff <= {"seed_mode", "regions", "trace", "trace_seconds"}
    with pytest.raises(ValueError):
        MoeadOptions(trace=True, trace_seconds=0.0)


def _fake_region_run(arm: str, trace: bool):
    """Drive one search through a fixed insertion/pick sequence with a fake clock."""
    search = s4_moead.make_search(arm, s4_moead.arm_config(arm, 3, 1200), CAL)
    search.moead_init(replace(search.moead, trace=trace))
    clock = [100.0]
    search.timer = lambda: clock[0]
    search._ready = True
    search._clock_offset = 0.0
    search._rng = RNGManager(4).stream("lsv2:moead")
    search.sampler = SeedSampler("gap", RNGManager(4).stream("s"))
    start = [(1.0, 10.0), (5.0, 5.0), (10.0, 1.0)]
    archive = SimpleNamespace(
        members=[SimpleNamespace(evaluation=SimpleNamespace(objectives=p)) for p in start],
        insertions=3,
    )
    gateway = SimpleNamespace(archive=archive)
    search._last_insertions = 3
    search.frame = FrozenFrame.of(start)
    events = [
        ("structural", (2.0, 7.0)),
        ("LSV2:A2", (6.0, 3.0)),
        ("LSV2:RIGHTSHIFT", (4.0, 6.0)),
        ("structural", (2.0, 7.0)),  # equal objectives, distinct timing: zero HV gain
        ("LSV2:A4", (8.0, 1.5)),
    ]
    picks = []
    for origin, point in events:
        clock[0] += 31.0
        search.ls_seconds += 5.0
        archive.members.append(SimpleNamespace(evaluation=SimpleNamespace(objectives=point)))
        archive.insertions += 1
        candidate = SimpleNamespace(origin=origin, evaluation=SimpleNamespace(objectives=point))
        search._on_candidate(candidate, gateway)
        front = nondominated([m.evaluation.objectives for m in archive.members])
        picks.append((search._pick_index(search.frame, front, []), search.config.budget))
    return search, picks, start, archive


@pytest.mark.parametrize(
    "arm",
    [
        *s4_moead.COMBO_ARMS,
        *s4_moead.NEXT_ARMS,
        *s4_moead.BANDIT_ARMS,
        *s4_moead.SHARE_ARMS,
        *s4_moead.WIDEN_ARMS,
        *s4_moead.SPACE_ARMS,
    ],
)
def test_trace_is_record_only_and_charges_hv_gain_by_source(arm):
    traced, picks, start, archive = _fake_region_run(arm, True)
    plain, plain_picks, _, _ = _fake_region_run(arm, False)
    assert picks == plain_picks and traced._rates() == plain._rates()
    assert traced._rate == plain._rate and traced.moead_counts.get("region_wide") == (
        plain.moead_counts.get("region_wide")
    )
    assert traced._bandit_gain == plain._bandit_gain
    assert traced._bandit_time == plain._bandit_time
    assert traced._mgain == plain._mgain and traced._mtime == plain._mtime
    assert "trace" not in plain.moead_summary() and plain._trace == {}
    summary = traced.moead_summary()
    assert summary["options"]["trace"] is True
    record = summary["trace"]
    assert record["fields"] == list(TRACE_FIELDS) and record["wbins"] == TRACE_WBINS
    totals: Counter = Counter()
    for t, w, f, v in record["cells"]:
        assert 0 <= w < TRACE_WBINS and t >= 4 and v >= 0
        totals[TRACE_FIELDS[f]] += v
    assert totals["evo"] == 2 and totals["ls"] == 2 and totals["other"] == 1
    assert totals["seed"] == 5 and totals["terminal"] == 0
    frame = traced.frame
    first = frame.hv([*start, (2.0, 7.0)]) - frame.hv(start)
    assert totals["evo_hv"] == pytest.approx(first) and first > 0
    hv_total = sum(totals[f] for f in ("evo_hv", "ls_hv", "other_hv"))
    final = [m.evaluation.objectives for m in archive.members]
    assert hv_total == pytest.approx(frame.hv(final) - frame.hv(start))
    if s4_moead.options(arm).seed_mode in ("region_yield", "region_ratio"):
        assert record["rates"] and all(
            len(r[0]) == traced.moead.regions for r in record["rates"].values()
        )
        assert totals["wide"] == traced.moead_counts.get("region_wide", 0)
    elif s4_moead.options(arm).seed_mode == "region_bandit":
        assert record["rates"] and "bandit" in summary
        assert totals["wide"] == traced.moead_counts.get("static_wide", 0)
    else:
        assert record["rates"] == {}
        assert totals["wide"] == traced.moead_counts.get("static_wide", 0)
    if not s4_moead.options(arm).static_wide:
        assert "static_wide" not in traced.moead_counts
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.parametrize(
    "arm",
    [
        *s4_moead.COMBO_ARMS,
        *s4_moead.NEXT_ARMS,
        *s4_moead.BANDIT_ARMS,
        *s4_moead.SHARE_ARMS,
        *s4_moead.WIDEN_ARMS,
        *s4_moead.SPACE_ARMS,
    ],
)
def test_runner_executes_every_combination_arm_with_trace(tmp_path, arm, monkeypatch):
    monkeypatch.setattr(s4_decouple, "TERMINAL_SHARE", 0.999)
    instance = tmp_path / "input.json"
    save_instance(synthetic(4)[0], instance)
    out = tmp_path / "out"
    summary = run_case(
        arm, instance, out, seconds=6, calibration=CAL, max_generations=3, campaign_output=True
    )
    assert summary["independent_exact_audit"] == "passed" and summary["archive_size"] > 0
    record = json.loads((out / "algorithm_config.json").read_text())
    assert record["arm"] == arm and record["moead"]["trace"] is True
    with gzip.open(out / "local_search.json.gz", "rt") as f:
        data = json.load(f)
    assert data["moead"]["options"]["regions"] == s4_moead.options(arm).regions
    assert data["moead"]["trace"]["fields"] == list(TRACE_FIELDS)


@pytest.fixture
def combo(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_p4p1")


def test_combination_batch_runs_four_arms_on_large_blocks_at_seed_three(
    tmp_path, monkeypatch, combo
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(combo.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(combo.base, "CAL", digest(cal))
    root = tmp_path / "full"
    combo.prepare(root, panel, cal, "example")
    m = combo.validate_inputs(root)
    assert m["runs"] == 320 and len(m["blocks"]) == 80 and tuple(m["arms"]) == combo.ARMS
    assert m["design"] == combo.DESIGN and m["control"] == "N0_P1"
    assert m["solver_seconds"] == 4 * (40 * 1200 + 40 * 2400)
    cells, orders, firsts = Counter(), Counter(), Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        cells[(spec["n"], spec["seed"])] += 1
        assert block == f"{spec['instance']}_s3" and spec["seed"] == 3
        assert spec["config"]["N2_P1_P4R3"]["moead"]["regions"] == 3
        assert spec["config"]["N1_P1_P4R5"]["moead"]["regions"] == 5
        assert spec["config"]["N3_P1_P4R3_RANDOM"]["moead"]["seed_mode"] == "region_random"
        assert all("A7" not in spec["config"][a]["lsv2"]["operators"] for a in combo.ARMS)
        orders[tuple(spec["order"])] += 1
        firsts[(spec["n"], spec["order"][0])] += 1
    assert cells == {(100, 3): 40, (200, 3): 40}
    assert len(orders) == 4 and set(orders.values()) == {20}
    assert set(firsts.values()) == {10}
    seen = []
    for k in range(3):
        shard = tmp_path / f"shard{k}"
        combo.prepare(shard, panel, cal, f"host{k}", f"{k}/3")
        seen += combo.validate_inputs(shard)["blocks"]
    assert sorted(seen) == sorted(m["blocks"]) and len(seen) == len(set(seen))
    assert all(combo.base.MODE[a] == {"common_fast"} for a in combo.ARMS)
    spec_path = root / "specs" / (m["blocks"][0] + ".json")
    spec = json.loads(spec_path.read_text())
    spec["config"]["N2_P1_P4R3"]["moead"]["regions"] = 5
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(AssertionError):
        combo.validate_inputs(root)


def _fixed_clock_search(arm: str):
    search = s4_moead.make_search(arm, s4_moead.arm_config(arm, 3, 1200), CAL)
    search.timer = lambda: 500.0
    search._ready = True
    search._clock_offset = 0.0
    search._rng = RNGManager(6).stream("lsv2:moead")
    search.sampler = SeedSampler("gap", RNGManager(6).stream("s"))
    start = [(1.0, 10.0), (5.0, 5.0), (10.0, 1.0)]
    archive = SimpleNamespace(
        members=[SimpleNamespace(evaluation=SimpleNamespace(objectives=p)) for p in start],
        insertions=3,
    )
    search._last_insertions = 3
    search.frame = FrozenFrame.of(start)
    return search, start, archive


def test_follow_up_arms_declare_hv_yields_and_comparative_advantage():
    n2 = s4_moead.options("N2_P1_P4R3")
    assert s4_moead.options("N4_P1_P4R3_HV") == replace(n2, yield_measure="hv")
    assert s4_moead.options("N5_P1_P4R3_CA") == replace(
        n2, seed_mode="region_ratio", yield_measure="hv"
    )
    cfg = s4_moead.arm_config("N5_P1_P4R3_CA", 7, 1200)
    for arm in s4_moead.NEXT_ARMS:
        record = s4_moead.configuration_record(arm, cfg, CAL)
        assert record["moead"]["yield_measure"] == "hv" and record["moead"]["regions"] == 3
        assert s4_moead.search_config(arm) == s4_moead.search_config("N0_P1")
    for arm in s4_moead.COMBO_ARMS:
        assert "yield_measure" not in s4_moead.configuration_record(arm, cfg, CAL)["moead"]
    with pytest.raises(ValueError):
        MoeadOptions(yield_measure="hv")
    with pytest.raises(ValueError):
        MoeadOptions(seed_mode="region_yield", yield_measure="area")


def test_hv_yield_charges_in_frame_gain_and_ignores_equal_objectives():
    rates = {}
    for arm in ("N4_P1_P4R3_HV", "N2_P1_P4R3"):
        search, start, archive = _fixed_clock_search(arm)
        gateway = SimpleNamespace(archive=archive)
        for point in ((2.0, 7.0), (2.0, 7.0)):
            archive.members.append(SimpleNamespace(evaluation=SimpleNamespace(objectives=point)))
            archive.insertions += 1
            candidate = SimpleNamespace(
                origin="structural", evaluation=SimpleNamespace(objectives=point)
            )
            search._on_candidate(candidate, gateway)
        region = search._region_of(search.frame, (2.0, 7.0))
        rates[arm] = (search._rate["evo"][region], sum(search._rate["evo"]))
        frame = search.frame
    gain = frame.hv([*start, (2.0, 7.0)]) - frame.hv(start)
    assert gain > 0
    assert rates["N4_P1_P4R3_HV"] == (pytest.approx(gain), pytest.approx(gain))
    assert rates["N2_P1_P4R3"] == (2.0, 2.0)


def test_comparative_advantage_targets_the_relatively_best_region():
    counts = {}
    for arm in ("N5_P1_P4R3_CA", "N2_P1_P4R3"):
        search, start, _ = _fixed_clock_search(arm)
        search._rate_clock = 500.0
        search._rate_ls_seconds = search.ls_seconds
        search._rate_time = dict(evo=10.0, ls=10.0)
        # Evolution out-yields LS everywhere; LS is relatively strongest in the middle.
        search._rate = dict(evo=[50.0, 50.0, 50.0], ls=[2.0, 30.0, 2.0])
        regions = [search._region_of(search.frame, p) for p in start]
        assert sorted(regions) == [0, 1, 2]
        for _ in range(300):
            search._pick_index(search.frame, start, [])
            assert search.config.budget != search.moead.wide_budget
        counts[arm] = [search.moead_counts.get(f"region_seed:{r}", 0) for r in range(3)]
    assert counts["N5_P1_P4R3_CA"][1] > 0.6 * 300
    assert max(counts["N2_P1_P4R3"]) < 0.5 * 300


@pytest.fixture
def follow(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_p4_next")


def test_follow_up_batch_runs_four_arms_on_large_blocks_at_seed_four(tmp_path, monkeypatch, follow):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(follow.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(follow.base, "CAL", digest(cal))
    root = tmp_path / "full"
    follow.prepare(root, panel, cal, "example")
    m = follow.validate_inputs(root)
    assert m["runs"] == 320 and len(m["blocks"]) == 80 and m["design"] == "s4-p1-p4-next/v1"
    assert tuple(m["arms"]) == ("N0_P1", "N2_P1_P4R3", "N4_P1_P4R3_HV", "N5_P1_P4R3_CA")
    orders = Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        assert spec["seed"] == 4 and block.endswith("_s4") and spec["n"] in (100, 200)
        assert spec["config"]["N5_P1_P4R3_CA"]["moead"]["seed_mode"] == "region_ratio"
        assert spec["config"]["N4_P1_P4R3_HV"]["moead"]["yield_measure"] == "hv"
        assert "run_s4_p4_next.py" in spec["entry"]
        orders[tuple(spec["order"])] += 1
    assert len(orders) == 4 and set(orders.values()) == {20}
    seen = []
    for k in range(3):
        shard = tmp_path / f"shard{k}"
        follow.prepare(shard, panel, cal, f"host{k}", f"{k}/3")
        seen += follow.validate_inputs(shard)["blocks"]
    assert sorted(seen) == sorted(m["blocks"]) and len(seen) == len(set(seen))


def test_bandit_arm_declares_an_hv_rate_bandit_without_widening():
    n2 = s4_moead.options("N2_P1_P4R3")
    assert s4_moead.options("N6_P1_P4R3_BANDIT") == replace(
        n2, seed_mode="region_bandit", yield_measure="hv"
    )
    cfg = s4_moead.arm_config("N6_P1_P4R3_BANDIT", 7, 1200)
    record = s4_moead.configuration_record("N6_P1_P4R3_BANDIT", cfg, CAL)
    assert record["moead"]["seed_mode"] == "region_bandit"
    assert record["moead"]["yield_measure"] == "hv"
    assert s4_moead.search_config("N6_P1_P4R3_BANDIT") == s4_moead.search_config("N0_P1")
    with pytest.raises(ValueError):
        MoeadOptions(seed_mode="region_bandit")


def _insert(search, archive, origin, point):
    archive.members.append(SimpleNamespace(evaluation=SimpleNamespace(objectives=point)))
    archive.insertions += 1
    candidate = SimpleNamespace(origin=origin, evaluation=SimpleNamespace(objectives=point))
    search._on_candidate(candidate, SimpleNamespace(archive=archive))


def test_bandit_charges_each_trajectory_to_its_launch_region_per_ls_second():
    search, start, archive = _fixed_clock_search("N6_P1_P4R3_BANDIT")
    clock = [500.0]
    search.timer = lambda: clock[0]
    frame = search.frame
    regions = [search._region_of(frame, p) for p in start]
    assert sorted(regions) == [0, 1, 2]
    launched = []
    gains = {}
    # Exploration: each present region is launched once before any rate is used.
    for step, point in enumerate(((0.5, 10.0), (9.0, 0.5), (4.0, 4.0))):
        search._pick_index(frame, start, [])
        region = search._bandit_region
        launched.append(region)
        assert search.config.budget != search.moead.wide_budget
        search.ls_seconds += 10.0 * (step + 1)
        before = frame.hv([m.evaluation.objectives for m in archive.members])
        _insert(search, archive, "LSV2:A2", point)
        gains[region] = frame.hv([m.evaluation.objectives for m in archive.members]) - before
        _insert(search, archive, "structural", (point[0] + 0.01, point[1] + 0.01))
    assert sorted(launched) == [0, 1, 2] and search.moead_counts["bandit_explore"] == 3
    search._bandit_pick([0, 1, 2])  # closes the third trajectory's account
    for step, region in enumerate(launched):
        assert search._bandit_time[region] == pytest.approx(10.0 * (step + 1))
        # Credit follows the launch region, wherever the insertion landed.
        assert search._bandit_gain[region] == pytest.approx(gains[region])
        assert gains[region] > 0
    rates = search._bandit_rates()
    clock[0] += 120.0
    search._bandit_region = None
    search._bandit_accrue(clock[0])
    assert search._bandit_rates() == pytest.approx(rates)  # decay keeps each estimate
    search._bandit_gain = [0.0, 50.0, 0.0]
    search._bandit_time = [1.0, 1.0, 1.0]
    picks = Counter(search._bandit_pick([0, 1, 2]) for _ in range(300))
    assert picks[1] > 0.8 * 300 and picks[0] > 0 and picks[2] > 0


def test_degenerate_frame_insertions_gain_nothing():
    single = FrozenFrame.of([(1.0, 1.0)])
    assert degenerate(single) and not degenerate(FrozenFrame.of([(1.0, 2.0), (2.0, 1.0)]))
    assert single.hv([(0.5, 0.5)]) > 1e20  # why the guard exists
    search, _, _ = _fixed_clock_search("N4_P1_P4R3_HV")
    search.frame = single
    archive = SimpleNamespace(
        members=[SimpleNamespace(evaluation=SimpleNamespace(objectives=(1.0, 1.0)))],
        insertions=1,
    )
    search._last_insertions = 1
    _insert(search, archive, "structural", (0.5, 0.5))
    assert sum(search._rate["evo"]) == 0.0
    assert search.moead_counts["degenerate_frame_insertions"] == 1


@pytest.fixture
def bandit_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_p4_bandit")


def test_bandit_batch_runs_two_arms_on_large_blocks_at_seed_five(
    tmp_path, monkeypatch, bandit_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(bandit_batch.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(bandit_batch.base, "CAL", digest(cal))
    root = tmp_path / "full"
    bandit_batch.prepare(root, panel, cal, "example")
    m = bandit_batch.validate_inputs(root)
    assert m["runs"] == 160 and len(m["blocks"]) == 80 and m["design"] == "s4-p1-p4-bandit/v1"
    assert tuple(m["arms"]) == ("N0_P1", "N6_P1_P4R3_BANDIT")
    assert m["solver_seconds"] == 2 * (40 * 1200 + 40 * 2400)
    orders = Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        assert spec["seed"] == 5 and block.endswith("_s5") and spec["n"] in (100, 200)
        assert spec["config"]["N6_P1_P4R3_BANDIT"]["moead"]["seed_mode"] == "region_bandit"
        assert "run_s4_p4_bandit.py" in spec["entry"]
        orders[tuple(spec["order"])] += 1
    assert len(orders) == 2 and set(orders.values()) == {40}
    seen = []
    for k in range(3):
        shard = tmp_path / f"shard{k}"
        bandit_batch.prepare(shard, panel, cal, f"host{k}", f"{k}/3")
        seen += bandit_batch.validate_inputs(shard)["blocks"]
    assert sorted(seen) == sorted(m["blocks"]) and len(seen) == len(set(seen))


def test_share_arms_declare_the_equimarginal_ls_share():
    assert s4_moead.options("N7_P1_SHARE") == replace(
        s4_moead.options("N0_P1"), share_mode="equimarginal"
    )
    assert s4_moead.options("N8_P1_P4R3_BANDIT_SHARE") == replace(
        s4_moead.options("N6_P1_P4R3_BANDIT"), share_mode="equimarginal"
    )
    cfg = s4_moead.arm_config("N7_P1_SHARE", 7, 1200)
    record = s4_moead.configuration_record("N7_P1_SHARE", cfg, CAL)
    assert record["moead"]["share_mode"] == "equimarginal"
    assert record["moead"]["share_bounds"] == [0.05, 0.5]
    assert "share_mode" not in s4_moead.configuration_record("N6_P1_P4R3_BANDIT", cfg, CAL)["moead"]
    assert s4_moead.search_config("N7_P1_SHARE") == s4_moead.search_config("N0_P1")
    for bad in (
        dict(share_mode="adaptive"),
        dict(share_bounds=(0.5, 0.2)),
        dict(share_bounds=(0.0, 0.5)),
        dict(share_exponent=0.0),
    ):
        with pytest.raises(ValueError):
            MoeadOptions(**bad)


def test_equimarginal_share_integrates_the_marginal_value_of_ls_time():
    fixed, _, _ = _fixed_clock_search("N0_P1")
    fixed.search_start = 0.0
    assert fixed._ls_allowance(40.0) == pytest.approx(0.2 * 40.0)
    search, _, _ = _fixed_clock_search("N7_P1_SHARE")
    search.search_start = 0.0
    assert search._ls_allowance(0.0) == 0.0 and search._share == pytest.approx(0.2)
    search._mtime = dict(evo=100.0, ls=100.0)
    search._mgain = dict(evo=1.0, ls=9.0)  # LS buys 9x more HV per second
    assert search._ls_allowance(10.0) == pytest.approx(2.0)
    assert search._share == pytest.approx(0.5)  # 0.2 * 3 clipped to the upper bound
    assert search._ls_allowance(20.0) == pytest.approx(7.0)
    search._mgain = dict(evo=9.0, ls=1.0)
    assert search._ls_allowance(30.0) == pytest.approx(12.0)
    assert search._share == pytest.approx(0.2 / 3)
    assert search._ls_allowance(40.0) == pytest.approx(12.0 + 10 * 0.2 / 3)
    search._mgain = dict(evo=9.0, ls=0.0)
    search._ls_allowance(50.0)
    assert search._share == pytest.approx(0.05)  # lower bound keeps LS measurable
    search._mgain = dict(evo=0.0, ls=1.0)
    search._ls_allowance(60.0)
    assert search._share == pytest.approx(0.5)
    summary = search.moead_summary()
    assert summary["options"]["share_mode"] == "equimarginal"
    assert summary["share"]["allowance"] == pytest.approx(search._allow)


def test_equimarginal_share_charges_hv_and_seconds_by_source():
    search, start, archive = _fixed_clock_search("N7_P1_SHARE")
    search.moead_init(replace(search.moead, tau_seconds=1e12))
    search._ready = True
    clock = [500.0]
    search.timer = lambda: clock[0]
    frame = search.frame
    gains = Counter()
    for origin, point, ls_dt in (
        ("structural", (2.0, 7.0), 0.0),
        ("LSV2:A2", (6.0, 3.0), 4.0),
        ("structural", (2.0, 7.0), 0.0),  # equal objectives: no gain
        ("LSV2:RIGHTSHIFT", (4.0, 6.0), 1.0),  # neither source
        ("LSV2:A4", (8.0, 1.5), 3.0),
    ):
        clock[0] += 10.0
        search.ls_seconds += ls_dt
        before = frame.hv([m.evaluation.objectives for m in archive.members])
        _insert(search, archive, origin, point)
        after = frame.hv([m.evaluation.objectives for m in archive.members])
        source = "evo" if origin == "structural" else ("ls" if "RIGHTSHIFT" not in origin else None)
        if source:
            gains[source] += after - before
    assert search._mgain["evo"] == pytest.approx(gains["evo"]) and gains["evo"] > 0
    assert search._mgain["ls"] == pytest.approx(gains["ls"]) and gains["ls"] > 0
    # The first accrual only starts the clock; afterwards LS and evolution seconds add up.
    assert search._mtime["ls"] + search._mtime["evo"] == pytest.approx(40.0)
    assert search._mtime["ls"] == pytest.approx(8.0)


@pytest.fixture
def factorial_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_p4_factorial")


def test_factorial_batch_crosses_launch_region_with_ls_share_at_seed_six(
    tmp_path, monkeypatch, factorial_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(factorial_batch.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(factorial_batch.base, "CAL", digest(cal))
    root = tmp_path / "full"
    factorial_batch.prepare(root, panel, cal, "example")
    m = factorial_batch.validate_inputs(root)
    assert m["runs"] == 400 and len(m["blocks"]) == 80
    assert m["design"] == "s4-p1-p4-factorial/v2"
    assert factorial_batch.ARMS[:3] == ("N0_P1", "N3_P1_P4R3_RANDOM", "N6_P1_P4R3_BANDIT")
    orders = Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        assert spec["seed"] == 6 and block.endswith("_s6")
        cfg = spec["config"]
        assert cfg["N3_P1_P4R3_RANDOM"]["moead"]["seed_mode"] == "region_random"
        assert cfg["N3_P1_P4R3_RANDOM"]["moead"]["regions"] == 3
        assert cfg["N7_P1_SHARE"]["moead"]["share_mode"] == "equimarginal"
        assert cfg["N8_P1_P4R3_BANDIT_SHARE"]["moead"]["seed_mode"] == "region_bandit"
        assert "share_mode" not in cfg["N6_P1_P4R3_BANDIT"]["moead"]
        assert "share_mode" not in cfg["N3_P1_P4R3_RANDOM"]["moead"]
        assert "run_s4_p4_factorial.py" in spec["entry"]
        orders[tuple(spec["order"])] += 1
    assert len(orders) == 5 and set(orders.values()) == {16}


def test_widen_arms_declare_flow_end_widening_without_yields():
    assert s4_moead.options("N9_P1_FLOWWIDE") == replace(
        s4_moead.options("N0_P1"), regions=3, static_wide=(2,)
    )
    assert s4_moead.options("N10_P1_P4R3_BANDIT_FLOWWIDE") == replace(
        s4_moead.options("N6_P1_P4R3_BANDIT"), static_wide=(2,)
    )
    cfg = s4_moead.arm_config("N9_P1_FLOWWIDE", 7, 1200)
    assert s4_moead.configuration_record("N9_P1_FLOWWIDE", cfg, CAL)["moead"]["static_wide"] == [2]
    assert (
        "static_wide" not in s4_moead.configuration_record("N6_P1_P4R3_BANDIT", cfg, CAL)["moead"]
    )
    assert s4_moead.search_config("N9_P1_FLOWWIDE") == s4_moead.search_config("N0_P1")
    for bad in (
        dict(seed_mode="region_yield", static_wide=(2,)),  # has its own widening rule
        dict(seed_mode="duplicate", static_wide=(2,)),
        dict(regions=3, static_wide=(3,)),
        dict(regions=3, static_wide=(2, 2)),
    ):
        with pytest.raises(ValueError):
            MoeadOptions(**bad)


@pytest.mark.parametrize("arm", ["N9_P1_FLOWWIDE", "N10_P1_P4R3_BANDIT_FLOWWIDE"])
def test_static_widening_widens_exactly_the_flow_end_seeds(arm):
    search, start, _ = _fixed_clock_search(arm)
    frame = search.frame
    regions = {p: search._region_of(frame, p) for p in start}
    assert sorted(regions.values()) == [0, 1, 2]
    base_budget = search.config.budget
    assert base_budget == s4_moead.search_config(arm).budget != search.moead.wide_budget
    seen = Counter()
    for _ in range(200):
        index = search._pick_index(frame, start, [])
        flow_end = regions[start[index]] == 2
        seen[flow_end] += 1
        assert search._last_wide is flow_end
        assert search.config.budget == (search.moead.wide_budget if flow_end else base_budget)
    assert seen[True] > 0 and seen[False] > 0
    assert search.moead_counts["static_wide"] == seen[True]
    assert search.moead_summary()["options"]["static_wide"] == [2]


@pytest.fixture
def widen_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_p4_widen")


def test_widen_batch_crosses_launch_region_with_flow_end_widening_at_seed_seven(
    tmp_path, monkeypatch, widen_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(widen_batch.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(widen_batch.base, "CAL", digest(cal))
    root = tmp_path / "full"
    widen_batch.prepare(root, panel, cal, "example")
    m = widen_batch.validate_inputs(root)
    assert m["runs"] == 320 and len(m["blocks"]) == 80
    assert m["design"] == "s4-p1-p4-widen/v1"
    orders = Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        assert spec["seed"] == 7 and block.endswith("_s7")
        cfg = spec["config"]
        assert cfg["N9_P1_FLOWWIDE"]["moead"]["static_wide"] == [2]
        assert cfg["N10_P1_P4R3_BANDIT_FLOWWIDE"]["moead"]["seed_mode"] == "region_bandit"
        assert "static_wide" not in cfg["N6_P1_P4R3_BANDIT"]["moead"]
        assert "run_s4_p4_widen.py" in spec["entry"]
        orders[tuple(spec["order"])] += 1
    assert len(orders) == 4 and set(orders.values()) == {20}


@pytest.fixture
def n7_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_n7_confirm")


def test_n7_confirmation_batch_pairs_control_and_share_at_seeds_seven_and_eight(
    tmp_path, monkeypatch, n7_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(n7_batch.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(n7_batch.base, "CAL", digest(cal))
    root = tmp_path / "full"
    n7_batch.prepare(root, panel, cal, "example")
    m = n7_batch.validate_inputs(root)
    assert m["runs"] == 320 and len(m["blocks"]) == 160
    assert m["design"] == "s4-p1-n7-confirm/v1" and tuple(m["seeds"]) == (7, 8)
    orders, seeds = Counter(), Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        seeds[spec["seed"]] += 1
        assert block.endswith(f"_s{spec['seed']}") and spec["n"] in (100, 200)
        cfg = spec["config"]
        assert cfg["N7_P1_SHARE"]["moead"]["share_mode"] == "equimarginal"
        assert "share_mode" not in cfg["N0_P1"]["moead"]
        assert cfg["N0_P1"]["moead"]["terminal_ops"] == ["A7", "A8"]
        assert "run_s4_n7_confirm.py" in spec["entry"]
        orders[tuple(spec["order"])] += 1
    assert seeds == {7: 80, 8: 80}
    assert len(orders) == 2 and set(orders.values()) == {80}
    shards = [
        n7_batch.base.shard_blocks(
            n7_batch.select(
                n7_batch.plan(
                    json.loads((panel / "manifest.json").read_text())["instances"],
                    json.loads(cal.read_text()),
                )
            ),
            f"{k}/3",
        )
        for k in range(3)
    ]
    assert sorted(len(s) for s in shards) == [53, 53, 54]


def test_space_arms_declare_the_launch_models():
    n0 = s4_moead.options("N0_P1")
    assert s4_moead.options("R1_P1_RANDPOINT") == replace(n0, seed_mode="random_point")
    assert s4_moead.options("G1_P1_GP") == replace(n0, seed_mode="gp_field")
    frontier = dict(share_mode="equimarginal", share_signal="frontier")
    assert s4_moead.options("E2_P1_EQM") == replace(n0, seed_mode="marginal_segments", **frontier)
    assert s4_moead.options("GE3_P1_GP_EQM") == replace(n0, seed_mode="gp_marginal", **frontier)
    cfg = s4_moead.arm_config("E2_P1_EQM", 7, 1200)
    record = s4_moead.configuration_record("E2_P1_EQM", cfg, CAL)
    assert record["moead"]["share_signal"] == "frontier"
    assert "share_signal" not in s4_moead.configuration_record("N7_P1_SHARE", cfg, CAL)["moead"]
    for arm in s4_moead.SPACE_ARMS:
        assert s4_moead.search_config(arm) == s4_moead.search_config("N0_P1")
    for bad in (
        dict(share_signal="peak"),
        dict(share_signal="frontier"),  # needs the equimarginal share
        dict(seed_mode="gp_field", share_mode="equimarginal", share_signal="frontier"),
    ):
        with pytest.raises(ValueError):
            MoeadOptions(**bad)


def test_random_point_launches_uniformly_from_the_front():
    search, start, _ = _fixed_clock_search("R1_P1_RANDPOINT")
    counts = Counter(search._pick_index(search.frame, start, []) for _ in range(900))
    assert set(counts) == {0, 1, 2} and min(counts.values()) > 250
    assert search.moead_counts["random_point_seeds"] == 900


def _run_field(search, start, gain_of, trajectories):
    """Scripted trajectories: one LS second each, HV gain set by the launch point."""
    picks = []
    for _ in range(trajectories):
        i = search._pick_index(search.frame, start, [])
        picks.append(i)
        search._traj["gain"] = gain_of(i)
        search.ls_seconds += 1.0
    return picks


def test_field_charges_trajectories_to_launch_points_and_learns_where_ls_pays():
    search, start, archive = _fixed_clock_search("G1_P1_GP")
    lams = [search.frame.weight_for(p)[0] for p in start]
    best = max(range(3), key=lambda i: lams[i])  # the Flow end pays
    picks = _run_field(search, start, lambda i: 0.02 if i == best else 0.0005, 120)
    assert search.moead_counts["field_observations"] == 119
    assert Counter(picks[-40:])[best] >= 30
    assert search.moead_counts.get("field_explore", 0) == 10  # random warm-up only
    # LS Archive insertions during an open trajectory are added to its gain.
    search._traj["gain"] = 0.0
    _insert(search, archive, "LSV2:A2", (2.0, 7.0))
    assert search._traj["gain"] > 0
    gain = search._traj["gain"]
    _insert(search, archive, "structural", (1.5, 8.0))  # evolution: not charged to LS
    assert search._traj["gain"] == gain
    record = search.moead_summary()["field"]
    assert record["mode"] == "gp_field" and len(record["mean"]) == 21 * 11
    assert len(record["trajectories"]) == 119 and record["snapshots"]
    assert json.loads(json.dumps(search.moead_summary())) == search.moead_summary()


def test_marginal_segments_launch_the_best_marginal_and_set_the_frontier_share():
    search, start, _ = _fixed_clock_search("E2_P1_EQM")
    model = search._field_model()
    segs = [model.segment(search.frame.weight_for(p)[0]) for p in start]
    best = max(range(3), key=lambda i: segs[i])
    picks = _run_field(search, start, lambda i: 0.05 if i == best else 0.001, 150)
    assert Counter(picks[-50:])[best] >= 40
    frontier = search._frontier_rate()
    present = sorted(set(segs))
    marg = model.marginals(0.05 * search._scale.current())
    assert frontier == pytest.approx(max(marg[g] for g in present)) and frontier > 0
    search._mtime = dict(evo=100.0, ls=100.0)
    search._mgain = dict(evo=1.0, ls=1.0)  # equal average rates: N7 would keep 0.2
    search._mclock = search.timer()
    want = min(0.5, max(0.05, 0.2 * (frontier / 0.01) ** 0.5))
    assert search._target_share() == pytest.approx(want)
    assert search.moead_summary()["options"]["share_signal"] == "frontier"


def test_gp_marginal_uses_local_investment_and_reports_a_frontier_rate():
    search, start, _ = _fixed_clock_search("GE3_P1_GP_EQM")
    lams = [search.frame.weight_for(p)[0] for p in start]
    best = max(range(3), key=lambda i: lams[i])
    assert search._frontier_rate() is None
    _run_field(search, start, lambda i: 0.02 if i == best else 0.0005, 60)
    z = [search._local_z(lam, search.timer()) for lam in lams]
    assert all(0 <= v < 1 for v in z) and max(z) > 0
    rate = search._frontier_rate()
    assert rate is not None and rate > 0
    record = search.moead_summary()["field"]
    assert record["mode"] == "gp_marginal" and record["snapshots"]
    snap = next(iter(record["snapshots"].values()))
    assert len(snap["mean"]) == len(snap["second"]) == 21


def _reference_frontier(search, now):
    """Pre-optimization GE3 frontier rate: scalar local investment, full posterior."""
    inputs = [(lam, search._local_z(lam, now)) for lam in search._field_lams]
    mean, _ = search._field.posterior(inputs)
    return float(search._scale.to_rate(float(max(mean))))


def test_gp_marginal_vectorized_inputs_match_the_scalar_reference():
    search, start, _ = _fixed_clock_search("GE3_P1_GP_EQM")
    clock = {"now": 500.0}
    search.timer = lambda: clock["now"]
    tau = search.moead.tau_seconds
    grid = [i / 20 for i in range(21)]
    for step in range(80):
        clock["now"] += 4.0  # launches spread over time
        _run_field(search, start, lambda i: 0.02 if i == 2 else 0.001, 1)
        if step % 9 == 0 or step > 70:
            lams = grid + search._field_lams
            now = clock["now"]
            fast = search._local_zs(lams, now)
            ref = [search._local_z(lam, now) for lam in lams]
            assert fast == pytest.approx(ref, rel=1e-12, abs=1e-15)
            if search._field.observations >= 10:
                want = _reference_frontier(search, now)
                assert search._frontier_rate() == pytest.approx(want, rel=1e-12)
    assert max(search._local_zs(search._field_lams, clock["now"])) > 0
    # Ageing past the 8 tau window prunes launches; the cache follows the launch set.
    later = clock["now"] + 8 * tau - 100.0
    assert search._local_zs(grid, later) == pytest.approx(
        [search._local_z(lam, later) for lam in grid], rel=1e-12, abs=1e-15
    )
    assert 0 < len(search._launches) < 79
    gone = clock["now"] + 9 * tau
    assert search._local_zs(grid, gone) == [0.0] * len(grid) and search._launches == []


@pytest.fixture
def space_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_space")


def _panel500(tmp_path, power):
    panel = tmp_path / "panel500"
    (panel / "instances").mkdir(parents=True)
    rows = []
    for fam in range(1, 11):
        for r in range(1, 5):
            key = f"mx_n500_f{fam:02d}_r{r:02d}"
            rel = f"instances/{key}.json"
            (panel / rel).write_text("{}")
            rows.append(dict(key=key, path=rel, jobs=500, sha256=digest(panel / rel)))
    (panel / "manifest.json").write_text(json.dumps(dict(power_model=power, instances=rows)))
    return panel


def test_space_batch_prepares_base_and_large_groups_at_seed_nine(
    tmp_path, monkeypatch, space_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    panel500 = _panel500(tmp_path, power)
    monkeypatch.setattr(space_batch.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(space_batch.base, "CAL", digest(cal))
    monkeypatch.setattr(space_batch, "PANEL500", digest(panel500 / "manifest.json"))
    assert space_batch.BUDGETS[500] == 6000 and 500 not in s4_ablation_budgets()
    root = tmp_path / "base"
    space_batch.prepare(root, panel, cal, "example", group="base")
    m = space_batch.validate_inputs(root)
    assert m["runs"] == 480 and len(m["blocks"]) == 80 and m["group"] == "base"
    assert m["design"] == "s4-space/v1" and tuple(m["seeds"]) == (9,)
    orders = Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        assert spec["seed"] == 9 and spec["n"] in (100, 200)
        cfg = spec["config"]
        assert cfg["G1_P1_GP"]["moead"]["seed_mode"] == "gp_field"
        assert cfg["E2_P1_EQM"]["moead"]["share_signal"] == "frontier"
        assert cfg["R1_P1_RANDPOINT"]["moead"]["seed_mode"] == "random_point"
        assert "run_s4_space.py" in spec["entry"]
        orders[tuple(spec["order"])] += 1
    assert len(orders) == 6
    large = tmp_path / "large"
    space_batch.prepare(large, panel500, cal, "example", shard="1/3", group="large")
    ml = space_batch.validate_inputs(large)
    assert ml["group"] == "large" and ml["sizes"] == [500]
    want = 10 * len(space_batch.LARGE_REPLICATES)
    assert len(ml["blocks"]) in (want // 3, want // 3 + 1) and ml["runs"] == 6 * len(ml["blocks"])
    folds = Counter()
    for block in ml["blocks"]:
        spec = json.loads((large / "specs" / (block + ".json")).read_text())
        assert spec["n"] == 500 and spec["seconds"] == 6000
        assert spec["calibration_fold"] in ("n200_f0", "n200_f1")
        folds[spec["calibration_fold"]] += 1
    assert set(folds) == {"n200_f0", "n200_f1"}
    reps = {int(b.split("_r")[1].split("_")[0]) for b in ml["blocks"]}
    assert reps <= set(space_batch.LARGE_REPLICATES)
    with pytest.raises(AssertionError):
        space_batch.prepare(tmp_path / "wrong", panel, cal, "example", group="large")
    small = tmp_path / "small"
    space_batch.prepare(small, panel, cal, "example", group="small")
    ms = space_batch.validate_inputs(small)
    assert ms["group"] == "small" and ms["runs"] == 240 and len(ms["blocks"]) == 40
    assert {
        json.loads((small / "specs" / (b + ".json")).read_text())["seconds"] for b in ms["blocks"]
    } == {600}


def s4_ablation_budgets():
    from geo_llm_scheduler.experiments import s4_ablation

    return s4_ablation.BUDGETS


@pytest.fixture
def ge3fix_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_ge3fix")


def test_ge3fix_batch_prepares_the_panel_and_nine_large_blocks(
    tmp_path, monkeypatch, ge3fix_batch, space_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    panel500 = _panel500(tmp_path, power)
    for mod in (ge3fix_batch, space_batch):
        monkeypatch.setattr(mod.base, "PANEL", digest(panel / "manifest.json"))
        monkeypatch.setattr(mod.base, "CAL", digest(cal))
        monkeypatch.setattr(mod, "PANEL500", digest(panel500 / "manifest.json"))
    root = tmp_path / "panel_root"
    ge3fix_batch.prepare(root, panel, cal, "example", group="panel")
    m = ge3fix_batch.validate_inputs(root)
    assert m["runs"] == 360 and len(m["blocks"]) == 120 and m["design"] == "s4-ge3fix/v1"
    assert tuple(m["arms"]) == ("N0_P1", "N7_P1_SHARE", "GE3_P1_GP_EQM")
    sizes, orders = Counter(), Counter()
    for block in m["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        sizes[spec["n"]] += 1
        orders[tuple(spec["order"])] += 1
        assert spec["seed"] == 9 and "run_s4_ge3fix.py" in spec["entry"]
        assert spec["seconds"] == {50: 600, 100: 1200, 200: 2400}[spec["n"]]
        assert spec["config"]["GE3_P1_GP_EQM"]["moead"]["seed_mode"] == "gp_marginal"
        assert spec["config"]["N7_P1_SHARE"]["moead"]["share_mode"] == "equimarginal"
    assert sizes == {50: 40, 100: 40, 200: 40} and len(orders) == 3
    # Large: f01-f09 at replicate 1 over three shards, folds as in the launch-point batch.
    old = tmp_path / "old"
    space_batch.prepare(old, panel500, cal, "example", group="large")
    old_folds = {
        b: json.loads((old / "specs" / (b + ".json")).read_text())["calibration_fold"]
        for b in space_batch.validate_inputs(old)["blocks"]
    }
    seen = []
    for k in range(3):
        large = tmp_path / f"large{k}"
        ge3fix_batch.prepare(large, panel500, cal, "example", shard=f"{k}/3", group="large")
        ml = ge3fix_batch.validate_inputs(large)
        assert ml["group"] == "large" and len(ml["blocks"]) == 3 and ml["runs"] == 9
        for b in ml["blocks"]:
            spec = json.loads((large / "specs" / (b + ".json")).read_text())
            assert spec["seconds"] == 6000 and spec["calibration_fold"] == old_folds[b]
        seen += ml["blocks"]
    assert sorted(seen) == [f"mx_n500_f{f:02d}_r01_s9" for f in range(1, 10)]
    with pytest.raises(AssertionError):
        ge3fix_batch.prepare(tmp_path / "wrong", panel, cal, "example", group="large")


# ---------------- LS investment control (2026-11-09) ----------------
def test_invest_arms_change_only_the_continuation_rule():
    n7 = s4_moead.options("N7_P1_SHARE")
    rules = dict(X1_EXTEND="extend", X1R_RANDEXT="random_extend", X2_PROBE="probe", X2L_L1="single")
    cfg = s4_moead.arm_config("X1_EXTEND", 10, 1200)
    for arm, rule in rules.items():
        assert s4_moead.options(arm) == replace(n7, continuation=rule, max_steps=10)
        assert s4_moead.search_config(arm) == s4_moead.search_config("N7_P1_SHARE")
        record = s4_moead.configuration_record(arm, cfg, CAL)
        assert record["moead"]["continuation"] == rule and record["moead"]["max_steps"] == 10
    assert "continuation" not in s4_moead.configuration_record("N7_P1_SHARE", cfg, CAL)["moead"]
    assert s4_moead.search_config("N7_P1_SHARE").trajectory_steps == 3
    with pytest.raises(ValueError):
        MoeadOptions(continuation="forever")
    with pytest.raises(ValueError):
        MoeadOptions(continuation="extend", max_steps=0)


def _lengths(arm: str, gains: list[list[float]], elapsed: float = 100.0, rng=None) -> list[int]:
    """Steps actually run when step k of trajectory j gains gains[j][k] (0 beyond the list)."""
    search, _, _ = _fixed_clock_search(arm)
    if rng is not None:
        search._extend_rng = rng
    out = []
    for g in gains:
        n = 0
        while True:
            gain = g[n] if n < len(g) else 0.0
            search.steps.append({})
            go = search._continue_trajectory(n, gain, 3, lambda: elapsed)
            n += 1
            if not go:
                break
        out.append(n)
    return out


def test_fixed_rule_runs_exactly_the_configured_steps_whatever_the_gains():
    assert _lengths("N7_P1_SHARE", [[0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1]]) == [3, 3, 3]
    search, _, _ = _fixed_clock_search("N7_P1_SHARE")
    search.steps.append({})
    search._continue_trajectory(2, 1.0, 3, lambda: 100.0)
    assert search.steps[-1] == {} and not search.moead_counts  # fixed rows stay unchanged


def test_extend_and_probe_follow_gains_up_to_the_cap_and_never_into_the_terminal_share():
    win = [1.0] * 20
    assert _lengths("X1_EXTEND", [[0, 0, 0], [0, 0, 1], [0, 0, 1, 1, 0], win]) == [3, 4, 5, 10]
    assert _lengths("X1_EXTEND", [[0, 0, 1]], elapsed=1150.0) == [3]  # 1200 s, terminal 5%
    assert _lengths("X2_PROBE", [[0], [1, 0], [1, 1, 1, 0], win]) == [1, 2, 4, 10]
    assert _lengths("X2L_L1", [[0], [1, 1, 1], win]) == [1, 1, 1]
    search, _, _ = _fixed_clock_search("X1_EXTEND")
    search.steps.append({})
    assert search._continue_trajectory(2, 1e-12, 3, lambda: 100.0) is False  # within tolerance
    assert search.steps[-1] == dict(continued=False, continue_reason="no_gain", gained=False)
    assert search.moead_counts == {"continue:no_gain": 1, "trajectory_length:3": 1}


def test_random_extension_uses_the_run_gain_rate_not_the_last_step():
    from random import Random

    # No gaining step so far: q = 0, never extends.
    assert _lengths("X1R_RANDEXT", [[0, 0, 0]] * 5, rng=Random(1)) == [3] * 5
    # Every step gains: q = 1, always extends to the cap.
    assert _lengths("X1R_RANDEXT", [[1.0] * 20] * 3, rng=Random(1)) == [10] * 3
    # A gaining history with a zero last step still extends at rate q (blind to the step).
    search, _, _ = _fixed_clock_search("X1R_RANDEXT")
    search._extend_rng = Random(3)
    search._cont_steps, search._cont_gains = 999, 999
    search.steps.append({})
    assert search._continue_trajectory(2, 0.0, 3, lambda: 100.0) is True
    assert search.steps[-1]["continue_reason"] == "random_extend"


def test_trajectory_loop_asks_the_rule_after_every_step(monkeypatch):
    from geo_llm_scheduler.lsv2 import search as search_module

    calls = []

    def fake_step(self, population, gateway, streams, frame, current, box, info):
        calls.append(info["step"])
        self.steps.append(dict(operator="A1", bandit=dict(state=(0, 1, 0))))
        return current, 1.0

    monkeypatch.setattr(search_module.LSV2Search, "_step", fake_step)
    monkeypatch.setattr(search_module, "_with_diagnostics", lambda problem, c: c)
    for arm, want in (("N7_P1_SHARE", [0, 1, 2]), ("X2L_L1", [0]), ("X1_EXTEND", list(range(10)))):
        calls.clear()
        search, start, _ = _fixed_clock_search(arm)
        members = [SimpleNamespace(evaluation=SimpleNamespace(objectives=p)) for p in start]
        gateway = SimpleNamespace(problem=None, archive=SimpleNamespace(members=members))
        search.trajectories = 0
        search.steps_since_frame = 0
        search._trajectory([], gateway, RNGManager(1), lambda: 100.0)
        assert calls == want, arm


def test_q_state_of_extension_steps_is_the_last_configured_step():
    search, _, _ = _fixed_clock_search("X1_EXTEND")
    streams = RNGManager(2)
    available = ["A1", "A2"]
    states = {}
    for step in (0, 1, 2, 3, 7):
        _, decision = search._choose_operator(0, available, streams, None, dict(step=step))
        states[step] = decision["state"]
    assert states[0][1] == 3 and states[1][1] == 2 and states[2][1] == 1
    assert states[3] == states[7] == states[2]


@pytest.mark.parametrize("arm", s4_moead.INVEST_ARMS)
def test_runner_executes_every_invest_arm_with_its_length_rule(tmp_path, arm):
    instance = tmp_path / "input.json"
    save_instance(synthetic(4)[0], instance)
    out = tmp_path / "out"
    summary = run_case(
        arm, instance, out, seconds=8, calibration=CAL, max_generations=4, campaign_output=True
    )
    assert summary["independent_exact_audit"] == "passed" and summary["archive_size"] > 0
    with gzip.open(out / "local_search.json.gz", "rt") as f:
        data = json.load(f)
    assert data["moead"]["options"]["continuation"] == s4_moead.options(arm).continuation
    lengths = Counter(r["trajectory"] for r in data["steps"])
    assert lengths and max(lengths.values()) <= 10
    if arm == "X2L_L1":
        assert set(lengths.values()) == {1}
    for r in data["steps"]:
        assert 1 <= r["q_state"][1] <= 3
        if "continued" in r and not r["continued"]:
            assert lengths[r["trajectory"]] == r["step"] + 1


@pytest.fixture
def invest_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_invest")


def test_invest_batch_prepares_five_arms_on_the_panel_at_seed_ten(
    tmp_path, monkeypatch, invest_batch
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(invest_batch.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(invest_batch.base, "CAL", digest(cal))
    seen = []
    for k in range(3):
        root = tmp_path / f"root{k}"
        invest_batch.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = invest_batch.validate_inputs(root)
        assert m["design"] == "s4-invest/v1" and tuple(m["arms"]) == invest_batch.ARMS
        assert m["runs"] == 5 * len(m["blocks"]) and len(m["blocks"]) == 40
        for block in m["blocks"]:
            spec = json.loads((root / "specs" / (block + ".json")).read_text())
            assert spec["seed"] == 10 and "run_s4_invest.py" in spec["entry"]
            assert spec["seconds"] == {50: 600, 100: 1200, 200: 2400}[spec["n"]]
            assert "continuation" not in spec["config"]["N7_P1_SHARE"]["moead"]
            assert spec["config"]["X1_EXTEND"]["moead"]["continuation"] == "extend"
            assert spec["config"]["X2L_L1"]["moead"]["continuation"] == "single"
        seen += m["blocks"]
    assert len(seen) == len(set(seen)) == 120
    assert Counter(int(b.split("_")[1][1:]) for b in seen) == {50: 40, 100: 40, 200: 40}
