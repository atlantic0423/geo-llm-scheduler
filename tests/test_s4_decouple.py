"""Right-shift decoupling arms: single-seam changes, Archive-only entrants, terminal sweep."""

import gzip
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic
from test_s4_load_shift import _panel

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_ablation, s4_decouple
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.trans_runner import ALGORITHMS, digest, run_case
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.lsv2.frontier import nondominated
from geo_llm_scheduler.lsv2.polish_selection import physical_key
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch
from geo_llm_scheduler.utils.rng import RNGManager

POLISH_FIELDS = {
    "polish_region",
    "polish_selection",
    "polish_diagnostics",
    "polish_method",
    "polish_pool_directions",
}


def test_arms_change_only_the_declared_seam():
    full = asdict(s4_ablation.search_config(s4_decouple.FULL))
    control = asdict(s4_ablation.search_config(s4_decouple.CONTROL))
    for arm in s4_decouple.EXTRA_ARMS:
        assert asdict(s4_decouple.search_config(arm)) == full
    # Without episode polish, B2 differs from the control only in polish-only fields.
    assert {k for k in control if control[k] != full[k]} <= POLISH_FIELDS
    assert s4_decouple.decoupling(s4_decouple.TERMINAL)["polish_after_episode"] is False
    assert s4_decouple.decoupling(s4_decouple.ARCHIVE)["polish_after_episode"] is True
    assert s4_decouple.decoupling(s4_decouple.BOTH) == dict(
        polish_reflow=False, polish_after_episode=True, terminal_share=0.05
    )
    cfg = s4_decouple.arm_config(s4_decouple.TERMINAL, 7, 600)
    assert cfg == s4_ablation.arm_config(s4_decouple.CONTROL, 7, 600)
    for arm in s4_decouple.EXTRA_ARMS:
        a = s4_decouple.make_search(arm, cfg, CAL)
        b = s4_decouple.make_search(arm, cfg, CAL)
        assert isinstance(a, QLearningSearch) and a.table is not b.table and not a.table.values
        assert a.spec == s4_ablation.make_search(s4_decouple.FULL, cfg, CAL).spec
        assert a.polish_reflow is False
        record = s4_decouple.configuration_record(arm, cfg, CAL)
        assert record["arm"] == arm and record["control"] == s4_decouple.CONTROL
        assert record["decoupling"] == s4_decouple.decoupling(arm)
    assert s4_decouple.make_search(s4_decouple.ARCHIVE, cfg, CAL).terminal_share is None
    assert s4_decouple.make_search(s4_decouple.TERMINAL, cfg, CAL).terminal_share == 0.05
    assert s4_decouple.make_search(s4_decouple.BOTH, cfg, CAL).polish_after_episode is True
    reference = s4_ablation.make_search(s4_decouple.FULL, cfg, CAL)
    assert reference.polish_reflow is True and reference.terminal_share is None
    assert reference.polish_after_episode is True
    assert set(ALGORITHMS).isdisjoint(s4_decouple.EXTRA_ARMS) and len(ALGORITHMS) == 14
    with pytest.raises(ValueError):
        s4_decouple.search_config(s4_decouple.CONTROL)


def test_terminal_window_starts_at_the_declared_share():
    cfg = s4_decouple.arm_config(s4_decouple.TERMINAL, 1, 600)
    search = s4_decouple.make_search(s4_decouple.TERMINAL, cfg, CAL)
    assert not search._terminal_due(569.99)
    assert search._terminal_due(570.0) and search._terminal_due(600.0)
    search.terminal_share = None
    assert not search._terminal_due(599.0)


def _initialized(n: int, seconds: int = 600):
    problem = synthetic(n)[0]
    cfg = s4_decouple.arm_config(s4_decouple.TERMINAL, 3, seconds)
    gw = EvaluationGateway(problem, Archive(), decode_cache_size=cfg.decode_cache_size)
    members = charged_initializer(problem, cfg, "BILL_MOEAD", {})(gw, RNGManager(3))
    return problem, cfg, gw, members


@pytest.mark.parametrize("arm", [s4_decouple.TERMINAL, s4_decouple.BOTH])
def test_terminal_sweep_visits_every_front_schedule_without_touching_population(arm):
    problem, cfg, gw, population = _initialized(6)
    search = s4_decouple.make_search(arm, cfg, CAL)
    config = search.config
    before = [id(c) for c in population]
    bill_before = {}
    for member in gw.archive.members:
        flow, bill = member.evaluation.objectives
        bill_before[flow] = min(bill, bill_before.get(flow, bill))
    search._terminal_sweep(population, gw, RNGManager(3), lambda: 0.0)
    assert [id(c) for c in population] == before
    rows = search.polish_steps
    assert rows and all(r["phase"] == "terminal" and r["reflow_replaced"] == 0 for r in rows)
    assert {r["status"] for r in rows} <= {"empty", "evaluated"}
    by_objective = {}
    for member in gw.archive.members:
        by_objective.setdefault(member.evaluation.objectives, member)
        assert member.evaluation == evaluate(problem, member.genotype, member.schedule)
    for point in nondominated(list(by_objective)):
        assert physical_key(by_objective[point]) in search._terminal_done
    # Same Flow points can only get cheaper (right shift keeps Flow fixed).
    for member in gw.archive.members:
        flow, bill = member.evaluation.objectives
        if flow in bill_before:
            assert bill <= bill_before[flow] + 1e-9
    calls = search.counts["terminal:polish"]
    search._terminal_sweep(population, gw, RNGManager(3), lambda: 0.0)
    assert search.counts["terminal:polish"] == calls  # no new insertion: skipped
    assert search.terminal_seconds > 0
    assert search.config is config and search.polish_reflow is False  # restored
    assert all(r.get("selection") is None for r in rows)  # sweep polishes its own target


def test_terminal_sweep_stops_at_the_deadline():
    _, cfg, gw, population = _initialized(6, seconds=600)
    search = s4_decouple.make_search(s4_decouple.TERMINAL, cfg, CAL)
    search._terminal_sweep(population, gw, RNGManager(3), lambda: 600.0)
    assert not search.polish_steps and search._terminal_signature is None


@pytest.mark.parametrize("arm", s4_decouple.EXTRA_ARMS)
def test_runner_executes_decoupled_arms_without_population_reflow(tmp_path, arm, monkeypatch):
    if arm != s4_decouple.ARCHIVE:
        # Enter the terminal window at once so a short run exercises it.
        monkeypatch.setattr(s4_decouple, "TERMINAL_SHARE", 0.999)
    instance = tmp_path / "input.json"
    save_instance(synthetic(4)[0], instance)
    out = tmp_path / "out"
    summary = run_case(
        arm, instance, out, seconds=6, calibration=CAL, max_generations=3, campaign_output=True
    )
    assert summary["independent_exact_audit"] == "passed" and summary["archive_size"] > 0
    record = json.loads((out / "algorithm_config.json").read_text())
    assert record["protocol"] == s4_decouple.PROTOCOL and record["arm"] == arm
    with gzip.open(out / "local_search.json.gz", "rt") as f:
        rows = json.load(f)["polish"]
    assert rows and all(r["reflow_replaced"] == 0 for r in rows if "reflow_replaced" in r)
    modes = {r["diagnostics"]["mode"] for r in rows if isinstance(r.get("diagnostics"), dict)}
    assert modes <= {"common_fast"}
    phases = {r.get("phase") for r in rows}
    expected = {s4_decouple.ARCHIVE: {None}, s4_decouple.TERMINAL: {"terminal"}}
    assert phases == expected.get(arm, phases) and ("terminal" in phases) == (
        arm != s4_decouple.ARCHIVE
    )


def test_terminal_arm_runs_no_right_shift_before_its_window(tmp_path):
    instance = tmp_path / "input.json"
    save_instance(synthetic(4)[0], instance)
    out = tmp_path / "out"
    run_case(
        s4_decouple.TERMINAL,
        instance,
        out,
        seconds=600,
        calibration=CAL,
        max_generations=2,
        campaign_output=True,
    )
    with gzip.open(out / "local_search.json.gz", "rt") as f:
        data = json.load(f)
    assert data["polish"] == [] and data["steps"]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_decouple")


def test_plan_rotates_four_arms_shards_and_subsets(tmp_path, monkeypatch, runner):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(runner, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(cal))
    full_root = tmp_path / "full"
    runner.prepare(full_root, panel, cal, "example")
    full = runner.validate_inputs(full_root)
    assert full["runs"] == 480 and len(full["blocks"]) == 120 and full["replicates"] == 4
    orders = {}
    for block in full["blocks"]:
        spec = json.loads((full_root / "specs" / (block + ".json")).read_text())
        orders[tuple(spec["order"])] = orders.get(tuple(spec["order"]), 0) + 1
        assert spec["config"]["A1_NO_SHIFT"]["lsv2"]["polish_region"] == "off"
        assert spec["config"]["B1_ARCHIVE_SHIFT"]["decoupling"]["polish_reflow"] is False
        assert spec["config"]["B2_TERMINAL_SHIFT"]["decoupling"]["terminal_share"] == 0.05
        assert spec["config"]["B3_ARCHIVE_TERMINAL"]["decoupling"]["polish_after_episode"]
    assert sorted(orders.values()) == [30, 30, 30, 30] and len(orders) == 4
    seen = []
    for k in range(3):
        root = tmp_path / f"shard{k}"
        runner.prepare(root, panel, cal, f"host{k}", f"{k}/3", 3)
        m = runner.validate_inputs(root)
        assert m["runs"] == 4 * len(m["blocks"]) and m["replicates"] == 3
        seen += m["blocks"]
        sizes = [json.loads((root / "specs" / (b + ".json")).read_text())["n"] for b in m["blocks"]]
        assert all(sizes.count(n) == 10 for n in (50, 100, 200))
    assert len(set(seen)) == 90 and not any(b.split("_s")[0].endswith("r04") for b in seen)
    (tmp_path / "shard0" / "specs" / (seen[0] + ".json")).write_text("{}")
    with pytest.raises(AssertionError):
        runner.validate_inputs(tmp_path / "shard0")
