"""Budget12 configuration isolation, real terminal/mask behavior and reference equivalence."""

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
from geo_llm_scheduler.experiments import ext_hybrid
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.lsv2_budget12 import (
    ARMS,
    EXTERNAL_ARMS,
    INTERNAL_ARMS,
    LEARNING_ARMS,
    RUN_SEED,
    arm_config,
    configuration_record,
    make_search,
    search_config,
)
from geo_llm_scheduler.experiments.lsv2_qscreen import make_search as original_search
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.lsv2.qlearning import QSpec, QTable, encode_state
from geo_llm_scheduler.lsv2.search import LSV2Config
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


def test_exact_authorized_matrix_and_config_differences():
    assert len(ARMS) == len(set(ARMS)) == 12
    assert len(LEARNING_ARMS) == 6 and RUN_SEED == 16901
    assert not any("G9_B1L1" in a for a in ARMS)
    for arm in INTERNAL_ARMS:
        cfg = search_config(arm)
        original = LSV2Config(selector=cfg.selector)
        differences = {k for k, v in asdict(cfg).items() if v != asdict(original)[k]}
        allowed = {"operators"} if arm.endswith("NOA8") else {"budget", "trajectory_steps"}
        assert differences <= allowed
        assert cfg.time_share == 0.20
        assert ("A8" in cfg.operators) != arm.endswith("NOA8")
        record = configuration_record(arm, RUN_SEED, 300, CAL)
        assert record["lsv2"] == asdict(cfg)
    for arm in EXTERNAL_ARMS:
        assert arm_config(arm, RUN_SEED, 300) == ext_hybrid.arm_config(arm, RUN_SEED, 300)
        assert configuration_record(arm, RUN_SEED, 300, {})["initialization"] == ext_hybrid.POLICY
    with pytest.raises(ValueError):
        search_config("Q1_NET_G9_B1L1")
    with pytest.raises(ValueError):
        arm_config("unknown", RUN_SEED, 300)


@pytest.mark.parametrize(
    "steps,gate,size", [(5, True, 30), (3, True, 18), (1, True, 6), (5, False, 15)]
)
def test_natural_state_cardinality(steps, gate, size):
    masks = (["A7"], ["A7", "A8"]) if gate else (["A7"],)
    states = {
        encode_state(QSpec("Q1_NET", 0.9), seg, k, mask, 0, trajectory_steps=steps)
        for seg in range(3)
        for k in range(steps)
        for mask in masks
    }
    assert len(states) == size
    assert {s[1] for s in states} == set(range(1, steps + 1))
    with pytest.raises(ValueError):
        encode_state(QSpec("Q1_NET", 0.9), 0, steps, [], 0, trajectory_steps=steps)


@pytest.mark.parametrize("arm", INTERNAL_ARMS)
def test_real_trajectories_masks_candidates_and_terminal(problem, arm):
    cfg = replace(arm_config(arm, RUN_SEED, 100), population=6, neighborhood=4)
    table = QTable()
    search = make_search(arm, cfg, CAL, table, training=arm in LEARNING_ARMS)
    execute(problem, search)
    length = search.config.trajectory_steps
    assert len(search.steps) == 2 * length
    for i, row in enumerate(search.steps):
        assert row["exact"] <= search.config.budget
        assert row["operator"] in row["available"]
        if arm.endswith("NOA8"):
            assert "A8" not in row["available"]
        if arm in LEARNING_ARMS:
            assert row["q_state"][1] == length - i % length
            assert row["q_state"][2] == int("A8" in row["available"])
            transition = row["q_transition"]
            assert transition["terminal"] == (i % length == length - 1)
            if transition["terminal"]:
                assert transition["bootstrap"] == 0
            else:
                assert transition["next_state"] == search.steps[i + 1]["q_state"]
                assert transition["next_available"] == search.steps[i + 1]["available"]
    assert sum(sum(r.values()) for r in table.visits.values()) == (
        2 * length if arm in LEARNING_ARMS else 0
    )


@pytest.mark.parametrize(
    "new,old",
    [("Q1_NET_G9_B6L5", "Q1_NET_G9"), ("Q1_NET_G0_B6L5", "Q1_NET_G0"), ("UNIFORM_B6L5", "UNIFORM")],
)
def test_reference_same_candidates_updates_and_rng(problem, new, old):
    cfg = replace(arm_config(new, RUN_SEED, 100), population=6, neighborhood=4)
    tables = [QTable(), QTable()]
    searches = [
        make_search(new, cfg, CAL, tables[0], training=new in LEARNING_ARMS),
        original_search(old, cfg, CAL, tables[1], training=new in LEARNING_ARMS),
    ]
    assert execute(problem, searches[0]) == execute(problem, searches[1])
    assert searches[0].steps == searches[1].steps
    assert asdict(tables[0]) == asdict(tables[1])


def test_l1_discount_is_irrelevant_and_noa8_does_not_bootstrap_removed_action(problem):
    cfg = replace(arm_config("Q1_NET_G0_B1L1", RUN_SEED, 100), population=6, neighborhood=4)
    results = []
    for gamma in (0.0, 0.9):
        q = QTable()
        search = make_search("Q1_NET_G0_B1L1", cfg, CAL, q, training=True)
        search.spec = replace(search.spec, gamma=gamma)
        results.append((execute(problem, search), asdict(q)))
    assert results[0] == results[1]
    q = QTable(values={"0:4:0": {"A8": 1e9, "A7": 2.0}})
    transition = q.update(
        (0, 5, 0),
        "A7",
        0,
        (0, 4, 0),
        search_config("Q1_NET_G9_B6L5_NOA8").operators,
        QSpec("Q1_NET", 0.9),
        training=True,
    )
    assert transition["bootstrap"] == 2.0


@pytest.fixture(scope="module")
def runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "budget12_runner", scripts / "run_lsv2_budget12.py"
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
    assert len(m["jobs"]) == 192 and len(m["blocks"]) == 16
    assert m["run_seeds"] == [16901]
    assert json.loads((root / "calibration.json").read_text()) == cals
    assert len({j["id"] for j in m["jobs"]}) == 192
    for job in m["jobs"]:
        assert job["runs"][0]["training"] == (job["arm"] in LEARNING_ARMS)
        assert runner.sha(root / job["config_file"]) == m["files"][job["config_file"]]
