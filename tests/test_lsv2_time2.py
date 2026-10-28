"""Paired Q/uniform masks, frozen paths and outcome-independent panel extension."""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_lsv2_ablation3 import CAL, execute

from geo_llm_scheduler.experiments import ext_hybrid, lsv2_ablation3, lsv2_budget12
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.lsv2_study import study_config
from geo_llm_scheduler.experiments.lsv2_time2 import (
    ARMS,
    BASE_BUDGETS,
    BILL,
    EXTERNAL_ARMS,
    INTERNAL_ARMS,
    LEARNING_ARMS,
    Q_FULL,
    RUN_SEED,
    arm_config,
    calibration_for,
    configuration_record,
    make_search,
    search_config,
    select_specs,
)
from geo_llm_scheduler.lsv2.qlearning import QTable


@pytest.mark.parametrize("arm", INTERNAL_ARMS)
def test_frozen_search_and_masked_uniform_are_exact(arm):
    problem = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    cfg = replace(arm_config(arm, RUN_SEED, 100), population=6, neighborhood=4)
    tables = [QTable(), QTable()]
    actual = make_search(arm, cfg, CAL, tables[0], training=True)
    old = lsv2_ablation3 if arm in LEARNING_ARMS else lsv2_budget12
    reference = old.make_search(
        arm if arm in LEARNING_ARMS else "UNIFORM_B3L3", cfg, CAL, tables[1], training=True
    )
    if arm == "UNIFORM_B3L3_NO_P3":
        reference.config = replace(
            reference.config, operators=tuple(op for op in reference.config.operators if op != "P3")
        )
    assert execute(problem, actual) == execute(problem, reference)
    assert actual.steps == reference.steps and asdict(tables[0]) == asdict(tables[1])
    assert len(actual.steps) == 6
    if arm.endswith("NO_P3"):
        assert all("P3" not in row["available"] for row in actual.steps)
    if arm not in LEARNING_ARMS:
        assert asdict(tables[0]) == dict(values={}, visits={})


def test_ten_arms_only_change_budget_and_declared_selector_mask():
    assert len(ARMS) == len(set(ARMS)) == 10
    assert len(EXTERNAL_ARMS) == 5
    for arm in ARMS:
        cfg = arm_config(arm, RUN_SEED, 600)
        old = (
            ext_hybrid.arm_config(arm, RUN_SEED, 300)
            if arm in EXTERNAL_ARMS
            else study_config(BILL, RUN_SEED, 300)
        )
        assert cfg == replace(old, seconds=600)
        record = configuration_record(arm, RUN_SEED, 600, CAL)
        assert record["engine"] == asdict(cfg)
        assert (record["qspec"] is not None) == (arm in LEARNING_ARMS)
    for suffix in ("", "_NO_P3"):
        q, uniform = search_config(Q_FULL + suffix), search_config("UNIFORM_B3L3" + suffix)
        assert q == replace(uniform, selector="bandit")
        assert q.budget == q.trajectory_steps == 3
    with pytest.raises(ValueError):
        arm_config("bad", RUN_SEED, 600)
    with pytest.raises(ValueError):
        search_config(BILL)


def panel():
    existing, pool = [], []
    for n in BASE_BUDGETS:
        for i in range(16):
            base = f"st_n{n}_d{i:02d}"
            pool.append(dict(n=n, key=base, seed=999, input=f"inputs/{base}.json"))
            if i < 8:
                existing.append(
                    dict(
                        pool[-1],
                        base_seed=base,
                        key=base + "_s16901",
                        seed=RUN_SEED,
                        panel="dev" if i < 4 else "holdout",
                    )
                )
    return existing, pool


def test_extension_does_not_depend_on_order_or_outcomes_and_never_refits():
    existing, pool = panel()
    chosen = select_specs(existing, pool)
    assert chosen == select_specs(list(reversed(existing)), list(reversed(pool)))
    assert len(chosen) == 40 and all(s["seed"] == RUN_SEED for s in chosen)
    assert all(s["base_seed"].endswith(("d08", "d09")) for s in chosen if s["panel"] == "extension")
    cals = {
        f"n{n}_f{f}": dict(CAL, keys=["training"], validation_keys=[])
        for n in BASE_BUDGETS
        for f in range(2)
    }
    for s in chosen:
        if s["panel"] == "dev":
            cals[f"n{s['n']}_f0"]["validation_keys"].append(s["key"])
        name, cal = calibration_for(s, cals, 1)
        assert cal == cals[name]
        assert name.endswith("f0" if s["panel"] == "dev" else "f1")
    cals["n20_f1"]["keys"] = ["st_n20_d08_s16901"]
    with pytest.raises(ValueError, match="evaluation"):
        calibration_for(next(s for s in chosen if s["base_seed"] == "st_n20_d08"), cals, 1)
    with pytest.raises(ValueError):
        select_specs(existing[:1], pool)


def test_prepare_seals_400_jobs_and_unchanged_calibration(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("time2_runner", scripts / "run_lsv2_time2.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    existing, pool = panel()
    old, pool_root, prior = (tmp_path / p for p in ("old", "pool", "prior"))
    cals = {
        f"n{n}_f{f}": dict(CAL, keys=["training"], validation_keys=[])
        for n in BASE_BUDGETS
        for f in range(2)
    }
    for root, specs in ((old, existing), (pool_root, pool)):
        files = {}
        for s in specs:
            runner.atomic_json(root / s["input"], {})
            s["input_sha256"] = files[s["input"]] = runner.sha(root / s["input"])
            name = "specs/" + s["key"] + ".json"
            runner.atomic_json(root / name, s)
            files[name] = runner.sha(root / name)
            if root == old:
                if s["panel"] == "dev":
                    cals[f"n{s['n']}_f0"]["validation_keys"].append(s["key"])
                out = root / "runs" / s["key"] / "EXTH_NSGAII"
                runner.atomic_json(
                    out / "summary.json",
                    dict(
                        initialization=dict(
                            initial_population_sha256="p", initial_archive_sha256="a"
                        )
                    ),
                )
                runner._seal(out, ["summary.json"])
        runner.atomic_json(root / "manifest.json", dict(files=files))
        runner.atomic_json(
            root / "manifest.sha256.json", dict(sha256=runner.sha(root / "manifest.json"))
        )
    runner.atomic_json(prior / "calibration.json", cals)
    runner.atomic_json(
        prior / "manifest.json",
        dict(files={"calibration.json": runner.sha(prior / "calibration.json")}),
    )
    runner.atomic_json(
        prior / "manifest.sha256.json", dict(sha256=runner.sha(prior / "manifest.json"))
    )
    root = tmp_path / "prepared"
    runner._prepare(
        argparse.Namespace(root=root, historical=old, pool=pool_root, prior_q=prior, host="test")
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 400 and len(manifest["blocks"]) == 40
    assert manifest["cpus"] == list(range(16)) and manifest["run_seeds"] == [16901]
    assert json.loads((root / "calibration.json").read_text()) == cals
    assert len({j["id"] for j in manifest["jobs"]}) == 400
    for job in manifest["jobs"]:
        assert job["seconds"] == BASE_BUDGETS[job["n"]] * 2
        assert job["runs"][0]["training"] == (job["arm"] in LEARNING_ARMS)
        assert runner.sha(root / job["config_file"]) == manifest["files"][job["config_file"]]
