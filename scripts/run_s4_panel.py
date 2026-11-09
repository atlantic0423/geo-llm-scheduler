"""Prepare and execute the sealed fourteen-configuration seed-one campaign."""

import argparse
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from run_lsv2 import source_hash
from slot_assignment_dispatch import dispatch

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import ext_hybrid, s4_ablation
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.lsv2_time2 import calibration_for
from geo_llm_scheduler.experiments.trans_common import SearchContext
from geo_llm_scheduler.experiments.trans_runner import ALGORITHMS, NATIVE, digest, run_case
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager

PANEL = "1c17b9edadc5f2f334390c95cd0ab2106c323a4c7f5032153a1b46bf79854484"
CAL = "45d79eb46cefb2ec4ca28d62554e5e2529547c5739f4c3ac9ab94526c4c2cf9c"
SEED = 1
REPO = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _policy(arm: str) -> str:
    if arm in ext_hybrid.ARMS:
        return ext_hybrid.POLICY
    return "random" if arm == "A3_RANDOM_INIT" else "bill4"


def _config(arm: str, seconds: int, cal: dict) -> dict:
    if arm in s4_ablation.ARMS:
        return s4_ablation.configuration_record(
            arm, s4_ablation.arm_config(arm, SEED, seconds), cal
        )
    if arm in ext_hybrid.ARMS:
        return asdict(ext_hybrid.arm_config(arm, SEED, seconds))
    return dict(
        name=arm,
        seed=SEED,
        seconds=seconds,
        population=1000 if arm == "PSLGP_H_PD" else 100,
        wait_genes=True,
        max_generations=None,
    )


def prepare(root: Path, panel: Path, calibration: Path, host: str) -> None:
    """Seal all inputs, outcome-independent size/fold mapping, commands and settings."""
    assert digest(panel / "manifest.json") == PANEL and digest(calibration) == CAL
    records = sorted(_read(panel / "manifest.json")["instances"], key=lambda r: r["key"])
    assert Counter(r["jobs"] for r in records) == {50: 40, 100: 40, 200: 40}
    root.mkdir(parents=True, exist_ok=False)
    folds = _read(calibration)
    files = {}
    for p, name in [
        (calibration, "calibration.json"),
        (panel / "manifest.json", "panel_manifest.json"),
    ]:
        shutil.copyfile(p, root / name)
        files[name] = digest(root / name)
    blocks = []
    groups = Counter()
    for i, row in enumerate(records):
        key, n = row["key"], row["jobs"]
        assert key not in blocks and Path(row["path"]).as_posix() == f"instances/{key}.json"
        blocks.append(key)
        target = root / row["path"]
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(panel / row["path"], target)
        assert digest(target) == row["sha256"]
        files[row["path"]] = row["sha256"]
        label, cal = calibration_for(dict(panel="frozen120", key=key, n=n), folds, groups[n])
        groups[n] += 1
        t = s4_ablation.BUDGETS[n]
        spec = dict(
            key=key,
            n=n,
            seed=SEED,
            input=row["path"],
            input_sha256=row["sha256"],
            seconds=t,
            calibration_fold=label,
            order=ALGORITHMS[i % 14 :] + ALGORITHMS[: i % 14],
            config={a: _config(a, t, cal) for a in ALGORITHMS},
            entry=f"scripts/run_s4_panel.py worker --root ROOT --key {key} --arm ARM --cpu CPU",
        )
        path = root / "specs" / (key + ".json")
        atomic_json(path, spec)
        files[path.relative_to(root).as_posix()] = digest(path)
    m = dict(
        source=source_hash(),
        commit=_head(),
        files=files,
        arms=ALGORITHMS,
        blocks=blocks,
        runs=1680,
        seed=SEED,
        workers=24,
        host=host,
        window_seconds=72 * 3600,
        solver_seconds=sum(40 * t * 14 for t in s4_ablation.BUDGETS.values()),
    )
    atomic_json(root / "manifest.json", m)
    atomic_json(root / "manifest.sha256.json", dict(sha256=digest(root / "manifest.json")))


def validate_inputs(root: Path) -> dict:
    """Reject any matrix, policy, source or input drift before work is dispatched."""
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    m = _read(root / "manifest.json")
    assert m["commit"] == _head() and m["source"] == source_hash()
    assert m["seed"] == SEED and tuple(m["arms"]) == ALGORITHMS and len(set(ALGORITHMS)) == 14
    assert m["runs"] == 1680 and len(set(m["blocks"])) == 120 and m["solver_seconds"] == 2352000
    assert (
        digest(root / "panel_manifest.json") == PANEL and digest(root / "calibration.json") == CAL
    )
    for name, sha in m["files"].items():
        assert digest(root / name) == sha, name
    folds = _read(root / "calibration.json")
    for key in m["blocks"]:
        s = _read(root / "specs" / (key + ".json"))
        assert s["seed"] == SEED and s["seconds"] == s4_ablation.BUDGETS[s["n"]]
        assert set(s["order"]) == set(ALGORITHMS) and len(s["order"]) == 14
        for arm in ALGORITHMS:
            expected = json.loads(
                json.dumps(_config(arm, s["seconds"], folds[s["calibration_fold"]]))
            )
            assert s["config"][arm] == expected
    return m


def preflight(root: Path, key: str) -> None:
    """Fingerprint all shared initializers and independently audit physical candidates."""
    m = _read(root / "manifest.json")
    s = _read(root / "specs" / (key + ".json"))
    assert digest(root / s["input"]) == s["input_sha256"]
    problem = load_instance(root / s["input"])
    checks = {}
    for policy, arm in [
        ("bill4", "BILL_MOEAD"),
        ("random", "EXT_MOEAD"),
        (ext_hybrid.POLICY, "EXTH_MOEAD"),
    ]:
        cfg = (
            ext_hybrid.arm_config(arm, SEED, s["seconds"])
            if arm == "EXTH_MOEAD"
            else s4_ablation.arm_config("A0_FULL_S4", SEED, s["seconds"])
        )
        gw = EvaluationGateway(problem, Archive(), decode_cache_size=cfg.decode_cache_size)
        evidence = {}
        make = (
            ext_hybrid.hybrid_initializer(problem, cfg, evidence)
            if arm == "EXTH_MOEAD"
            else charged_initializer(problem, cfg, arm, evidence)
        )
        members = make(gw, RNGManager(SEED))
        assert len(members) == 100 and evidence["policy"] == policy
        sample = members[::20] + list(gw.archive.members[:3])
        for c in sample:
            assert c.evaluation.feasible and c.evaluation == evaluate(
                problem, c.genotype, c.schedule
            )
        checks[policy] = {
            k: evidence[k] for k in ("initial_population_sha256", "initial_archive_sha256")
        }
        checks[policy]["fresh_exact"] = len(sample)
    ctx = SearchContext(problem, SEED, s["seconds"])
    for c in ctx.initialize(4):
        assert c.candidate.evaluation == evaluate(
            problem, c.candidate.genotype, c.candidate.schedule
        )
    path = root / "preflight" / (key + ".json")
    atomic_json(
        path,
        dict(
            source=m["source"],
            input_sha256=s["input_sha256"],
            host=socket.gethostname(),
            policies=checks,
            native_key_probe_exact=4,
        ),
    )
    atomic_json(path.with_suffix(".sha256.json"), dict(sha256=digest(path)))


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Verify only complete, sealed runs with the declared provenance and budget."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").is_file():
        return False
    seal = _read(out / "complete.json")
    assert seal["source"] == source and seal["key"] == key and seal["arm"] == arm
    assert digest(out / "manifest.json") == seal["manifest"]
    for record in json.loads((out / "manifest.json").read_text()):
        p = out / record["path"]
        assert p.stat().st_size == record["bytes"] and digest(p) == record["sha256"]
    return True


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Run one sealed paired task, cross-check initializer fingerprints, seal artifacts."""
    os.sched_setaffinity(0, {cpu})
    m = _read(root / "manifest.json")
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    specpath = root / "specs" / (key + ".json")
    assert (
        m["source"] == source_hash()
        and m["commit"] == _head()
        and digest(specpath) == m["files"][f"specs/{key}.json"]
    )
    s = _read(specpath)
    assert (
        arm in s["order"] and s["seed"] == SEED and digest(root / s["input"]) == s["input_sha256"]
    )
    pp = root / "preflight" / (key + ".json")
    assert digest(pp) == _read(pp.with_suffix(".sha256.json"))["sha256"]
    pref = _read(pp)
    assert (
        pref["host"] == socket.gethostname() == m["host"]
        and pref["input_sha256"] == s["input_sha256"]
    )
    assert digest(root / "calibration.json") == CAL
    cal = _read(root / "calibration.json")[s["calibration_fold"]]
    out = root / "runs" / key / arm
    summary = run_case(
        arm,
        root / s["input"],
        out,
        seed=SEED,
        seconds=s["seconds"],
        calibration=cal if arm in s4_ablation.ARMS else None,
        campaign_output=True,
    )
    cfg = _read(out / "config.json")
    assert (
        cfg["source"]["commit"] == m["commit"]
        and cfg["source"]["status"] == ""
        and cfg["instance_sha256"] == s["input_sha256"]
    )
    assert (
        cfg["seed"] == SEED
        and not cfg["engineering_override"]
        and cfg["seconds_including_initialization"] == s["seconds"]
    )
    assert (
        summary["independent_exact_audit"] == "passed"
        and summary["elapsed_seconds"] >= s["seconds"] - 0.05
    )
    if arm not in NATIVE:
        assert _read(out / "algorithm_config.json") == s["config"][arm]
        init = _read(out / "initialization.json")
        for field in ("initial_population_sha256", "initial_archive_sha256"):
            assert init[field] == pref["policies"][_policy(arm)][field]
    if arm == s4_ablation.DIRECT:
        assert _read(out / "q_table.json") is None
        with gzip.open(out / "local_search.json.gz", "rt") as fh:
            ls = json.load(fh)
        assert (
            not ls["steps"]
            and ls["polish"]
            and all(
                r.get("source") == "endpoint" or r["status"] == "deadline_skip"
                for r in ls["polish"]
            )
        )
    atomic_json(
        out / "complete.json",
        dict(
            source=m["source"],
            key=key,
            arm=arm,
            manifest=digest(out / "manifest.json"),
            spec=digest(specpath),
        ),
    )


def main() -> None:
    """Prepare, validate, preflight, run one case or dispatch on 24 distinct physical cores."""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "validate", "preflight", "dispatch", "worker"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--panel", type=Path)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--host", default="neu01-2288H-V6")
    p.add_argument("--key")
    p.add_argument("--arm", choices=ALGORITHMS)
    p.add_argument("--cpu", type=int)
    p.add_argument("--cpus")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.panel, a.calibration, a.host)
    elif a.action == "validate":
        print(validate_inputs(a.root)["runs"])
    elif a.action == "preflight":
        preflight(a.root, a.key)
    elif a.action == "worker":
        worker(a.root, a.key, a.arm, a.cpu)
    else:
        cores = [int(c) for c in a.cpus.split(",")]
        topo = {
            tuple(
                (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
                for f in ("physical_package_id", "core_id")
            )
            for c in cores
        }
        assert len(cores) == len(topo) == 24 and set(cores) <= os.sched_getaffinity(0)
        result = dispatch(
            a.root, sys.modules[__name__], cores, Path(__file__).resolve(), interval=2
        )
        print(json.dumps(result), flush=True)
        if result["state"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
