"""Action-mask factorial, exact baseline equivalence and sealed campaign preparation."""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from random import Random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.lsv2_ablation3 import (
    ARMS,
    BASE,
    LEARNING_ARMS,
    REMOVALS,
    RUN_SEED,
    arm_config,
    configuration_record,
    make_search,
    search_config,
)
from geo_llm_scheduler.experiments.lsv2_budget12 import make_search as old_search
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager

CAL = dict(rho=0.001, h=0.01, w_cutoff=1.5, p_cutoff=0.5)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


@pytest.fixture(scope="module")
def problem():
    return case_problem(20, 3, 6, 202610160020, "arrival", 0)


def execute(problem, search):
    gateway = EvaluationGateway(problem, Archive())
    pop = [
        gateway.evaluate(g) for g in initial_genotypes(problem, search.engine_config, Random(3))[:6]
    ]
    search.timer = Clock()
    streams = RNGManager(RUN_SEED)
    search(pop, gateway, streams, lambda: 0.0)
    for _ in range(2):
        search._trajectory(pop, gateway, streams, lambda: 1.0)
    for c in pop + gateway.archive.members:
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
    return [identity(c) for c in pop + gateway.archive.members], gateway.counts


def test_exact_factorial_only_changes_operator_pool():
    assert len(ARMS) == 8 and RUN_SEED == 16901
    assert {frozenset(v) for v in REMOVALS.values()} == {
        frozenset(op for i, op in enumerate(("P3", "A4", "P2_MIXED")) if mask & (1 << i))
        for mask in range(8)
    }
    base = asdict(search_config(BASE))
    for arm in ARMS:
        cfg = asdict(search_config(arm))
        assert {k for k in base if base[k] != cfg[k]} <= {"operators"}
        assert {"A3", "A8"} <= set(cfg["operators"])
        assert cfg["budget"] == cfg["trajectory_steps"] == 3
        record = configuration_record(arm, RUN_SEED, 300, CAL)
        assert record["lsv2"] == cfg and record["qspec"]["gamma"] == 0.9
        assert arm_config(arm, RUN_SEED, 300) == arm_config(BASE, RUN_SEED, 300)
    with pytest.raises(ValueError):
        search_config("unknown")
    with pytest.raises(ValueError):
        arm_config("unknown", RUN_SEED, 300)


def test_full_control_is_identical_to_frozen_b3l3(problem):
    cfg = replace(arm_config(BASE, RUN_SEED, 100), population=6, neighborhood=4)
    tables = [QTable(), QTable()]
    searches = [
        make_search(BASE, cfg, CAL, tables[0], training=True),
        old_search(BASE, cfg, CAL, tables[1], training=True),
    ]
    assert execute(problem, searches[0]) == execute(problem, searches[1])
    assert searches[0].steps == searches[1].steps
    assert asdict(tables[0]) == asdict(tables[1])


@pytest.mark.parametrize("arm", ARMS)
def test_removed_actions_never_selected_or_bootstrapped(problem, arm):
    cfg = replace(arm_config(arm, RUN_SEED, 100), population=6, neighborhood=4)
    table = QTable()
    search = make_search(arm, cfg, CAL, table, training=True)
    execute(problem, search)
    assert len(search.steps) == 6
    for i, row in enumerate(search.steps):
        assert set(row["available"]).isdisjoint(REMOVALS[arm])
        assert row["operator"] not in REMOVALS[arm]
        trans = row["q_transition"]
        assert set(trans["next_available"]).isdisjoint(REMOVALS[arm])
        assert trans["terminal"] == (i % 3 == 2)
        assert row["exact"] <= 3 and row["q_state"][1] == 3 - i % 3
    assert all(set(v).isdisjoint(REMOVALS[arm]) for v in table.values.values())
    assert sum(sum(v.values()) for v in table.visits.values()) == 6
    # Even a stale removed action with enormous Q is excluded from the future maximum.
    table = QTable(values={"0:2:1": {"P3": 1e10, "A4": 1e10, "P2_MIXED": 1e10, "A3": 2}})
    mask = search_config(BASE + "_NO_P3_A4_P2M").operators
    trans = table.update((0, 3, 1), "A3", 0, (0, 2, 1), mask, search.spec, training=True)
    assert trans["bootstrap"] == 2


@pytest.fixture(scope="module")
def runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "ablation3_runner", scripts / "run_lsv2_ablation3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(scripts))


def test_prepare_reuses_calibration_and_old_seed_and_seals_configs(tmp_path, runner):
    old, prior = tmp_path / "old", tmp_path / "q"
    files, cals = {}, {}
    for n in (20, 50, 100, 200):
        for i in range(4):
            key = f"st_n{n}_d{i:02d}_s16901"
            name = f"inputs/st_n{n}_d{i:02d}.json"
            p = old / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
            files[name] = runner.sha(p)
            s = dict(
                key=key,
                seed=RUN_SEED,
                panel="dev",
                n=n,
                input=name,
                input_sha256=files[name],
                seconds=300,
            )
            sn = "specs/" + key + ".json"
            runner.atomic_json(old / sn, s)
            files[sn] = runner.sha(old / sn)
            cals[key] = dict(CAL, keys=["other"], validation_keys=[key])
            out = old / "runs" / key / "EXTH_NSGAII"
            runner.atomic_json(
                out / "summary.json",
                dict(
                    initialization=dict(initial_population_sha256="p", initial_archive_sha256="a")
                ),
            )
            runner._seal(out, ["summary.json"])
    runner.atomic_json(old / "manifest.json", dict(files=files))
    runner.atomic_json(old / "manifest.sha256.json", dict(sha256=runner.sha(old / "manifest.json")))
    runner.atomic_json(prior / "calibration.json", cals)
    runner.atomic_json(
        prior / "manifest.json",
        dict(files={"calibration.json": runner.sha(prior / "calibration.json")}),
    )
    runner.atomic_json(
        prior / "manifest.sha256.json", dict(sha256=runner.sha(prior / "manifest.json"))
    )
    root = tmp_path / "new"
    runner._prepare(argparse.Namespace(root=root, historical=old, prior_q=prior, host="test"))
    m = json.loads((root / "manifest.json").read_text())
    assert len(m["jobs"]) == 128 and len(m["blocks"]) == 16
    assert m["run_seeds"] == [16901]
    assert json.loads((root / "calibration.json").read_text()) == cals
    assert len({j["id"] for j in m["jobs"]}) == 128
    for job in m["jobs"]:
        assert job["runs"][0]["training"] == (job["arm"] in LEARNING_ARMS)
        assert runner.sha(root / job["config_file"]) == m["files"][job["config_file"]]
