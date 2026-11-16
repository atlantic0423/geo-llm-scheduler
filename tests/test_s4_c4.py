"""S4+C4 changes only the right-shift neighborhood; C4 finds envelope-knot savings."""

import gzip
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler.domain.models import (
    Candidate,
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_ablation, s4_c4
from geo_llm_scheduler.experiments.lsv2_study import operator_config
from geo_llm_scheduler.experiments.trans_runner import ALGORITHMS, digest, run_case
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch


def test_only_the_right_shift_neighborhood_differs_from_full_s4():
    control = asdict(s4_ablation.search_config(s4_c4.CONTROL))
    treated = asdict(s4_c4.search_config(s4_c4.ARM))
    assert {k for k in control if control[k] != treated[k]} == {"polish_method"}
    assert (control["polish_method"], treated["polish_method"]) == ("common_fast", "common_peak")
    cfg = s4_c4.arm_config(s4_c4.ARM, 7, 600)
    assert cfg == s4_ablation.arm_config(s4_c4.CONTROL, 7, 600)
    a = s4_c4.make_search(s4_c4.ARM, cfg, CAL)
    b = s4_c4.make_search(s4_c4.ARM, cfg, CAL)
    assert isinstance(a, QLearningSearch) and a.table is not b.table and not a.table.values
    assert a.spec == s4_ablation.make_search(s4_c4.CONTROL, cfg, CAL).spec
    record = s4_c4.configuration_record(s4_c4.ARM, cfg, CAL)
    assert record["arm"] == s4_c4.ARM and record["control"] == s4_c4.CONTROL
    assert record["lsv2"]["polish_method"] == "common_peak"
    assert s4_c4.ARM not in ALGORITHMS and len(ALGORITHMS) == 14
    with pytest.raises(ValueError):
        s4_c4.search_config(s4_c4.CONTROL)


def _knot_case() -> Candidate:
    """Demand-only bill whose two window averages cross between event shifts (delta=150)."""
    problem = ProblemInstance(
        (
            Job("A", 0, Profile(600, 0.4, 4), Profile(300, 0.3, 4)),
            Job("B", 0, Profile(500, 0.4, 4), Profile(300, 0.3, 4)),
        ),
        (Region("r", (Tariff(0, 3600, 1.0),), 10.0),),
        (ServingInstance("m", 0, 32.0, 4.0, 10.0),),
    )
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((300.0, 1600.0, 400.0, 1500.0))
    return problem, Candidate(genotype, schedule, evaluate(problem, genotype, schedule))


def test_envelope_knot_recovers_saving_missed_by_event_points():
    problem, incumbent = _knot_case()
    cfg = operator_config(1, 100)
    fast = ReusedCommonPolish().propose(problem, incumbent, 3, cfg, Random(1))
    peak = ReusedCommonPolish(peak_switch=True).propose(problem, incumbent, 3, cfg, Random(1))
    assert "peak_switch" not in fast.diagnostics["proposal_sources"]
    assert peak.diagnostics["proposal_sources"][0] == "peak_switch"
    best = peak.proposals[0]
    assert best.selected == (0, 2)
    assert best.schedule.starts[0] - incumbent.schedule.starts[0] == pytest.approx(150.0)
    after = evaluate(problem, best.genotype, best.schedule)
    assert after.feasible and after.flow == incumbent.evaluation.flow
    assert incumbent.evaluation.bill - after.bill == pytest.approx(10.0)
    fast_best = min(evaluate(problem, p.genotype, p.schedule).bill for p in fast.proposals)
    assert fast_best > after.bill + 1.0


def test_runner_executes_c4_arm_and_records_only_its_neighborhood(tmp_path):
    instance = tmp_path / "input.json"
    save_instance(synthetic(4)[0], instance)
    out = tmp_path / "out"
    summary = run_case(
        s4_c4.ARM,
        instance,
        out,
        seconds=10,
        calibration=CAL,
        max_generations=2,
        campaign_output=True,
    )
    assert summary["independent_exact_audit"] == "passed" and summary["archive_size"] > 0
    record = json.loads((out / "algorithm_config.json").read_text())
    assert record["protocol"] == s4_c4.PROTOCOL
    with gzip.open(out / "local_search.json.gz", "rt") as f:
        rows = json.load(f)["polish"]
    modes = {r["diagnostics"]["mode"] for r in rows if isinstance(r.get("diagnostics"), dict)}
    assert modes <= {"common_peak"}
    with pytest.raises(ValueError):
        run_case("S4_UNKNOWN", instance, tmp_path / "x", seconds=10, calibration=CAL)


def test_stratified_plan_keeps_original_folds_and_rejects_drift(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_s4_c4_strata")
    panel = tmp_path / "panel"
    (panel / "instances").mkdir(parents=True)
    rows, folds, index = [], {}, {}
    for n in (50, 100, 200):
        for f in range(2):
            folds[f"n{n}_f{f}"] = dict(CAL, keys=[])
        keys = sorted(
            f"mx_n{n:03d}_f{fam:02d}_r{r:02d}" for fam in range(1, 11) for r in range(1, 5)
        )
        for i, key in enumerate(keys):
            rel = f"instances/{key}.json"
            (panel / rel).write_text("{}")
            rows.append(dict(key=key, path=rel, jobs=n, sha256=digest(panel / rel)))
            index[key] = i
    (panel / "manifest.json").write_text(json.dumps(dict(instances=rows)))
    cal = tmp_path / "folds.json"
    cal.write_text(json.dumps(folds))
    monkeypatch.setattr(runner, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(cal))
    root = tmp_path / "campaign"
    runner.prepare(root, panel, cal, "example")
    manifest = runner.validate_inputs(root)
    assert manifest["runs"] == 96 and len(manifest["blocks"]) == 48
    families = {}
    for block in manifest["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        assert spec["calibration_fold"] == f"n{spec['n']}_f{index[spec['instance']] % 2}"
        families.setdefault(spec["family"], set()).add(spec["seed"])
        assert spec["config"][s4_c4.ARM]["lsv2"]["polish_method"] == "common_peak"
        assert spec["config"][s4_c4.CONTROL]["lsv2"]["polish_method"] == "common_fast"
    assert families == {"f07": {1, 2}, "f10": {1}, "f08": {1}}
    (root / "specs" / (manifest["blocks"][0] + ".json")).write_text("{}")
    with pytest.raises(AssertionError):
        runner.validate_inputs(root)
