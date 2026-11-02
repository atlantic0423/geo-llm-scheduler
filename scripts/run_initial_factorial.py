"""Sealed S4 construction × retention developer screening, with unchanged budgets."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from run_lsv2 import source_hash
from run_s4_panel import CAL, PANEL
from run_s4_panel import complete as complete
from slot_assignment_dispatch import dispatch

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_ablation
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.initial_gap_construction import gap_initializer
from geo_llm_scheduler.experiments.initial_pool_retention import pool_initializer
from geo_llm_scheduler.experiments.runner import digest as object_digest
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.experiments.trans_runner import digest, run_case
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager

OPTIONS = {
    "I0_BILL4_LEGACY": dict(initial_construction="bill4", initial_retention="legacy"),
    "I1_BILL4_POOL": dict(initial_construction="bill4", initial_retention="global_pool"),
    "I2_G1_LEGACY": dict(initial_construction="g1", initial_retention="legacy"),
    "I3_G1_POOL": dict(initial_construction="g1", initial_retention="global_pool"),
}
ARMS = tuple(OPTIONS)
REPO = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _config(arm: str, spec: dict, calibration: dict) -> dict:
    cfg = s4_ablation.arm_config("A0_FULL_S4", spec["seed"], spec["seconds"])
    return s4_ablation.configuration_record("A0_FULL_S4", cfg, calibration, **OPTIONS[arm])


def prepare(root: Path, reference: Path, selection: Path, host: str) -> None:
    """Freeze the selected prior instances and their original seed/calibration folds."""
    plan = _read(selection)
    keys = plan["blocks"]
    assert (
        keys
        and len(keys) == len(set(keys))
        and set(keys) <= set(_read(reference / "manifest.json")["blocks"])
    )
    assert digest(reference / "panel_manifest.json") == PANEL
    assert digest(reference / "calibration.json") == CAL
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for src, name in [
        (reference / "calibration.json", "calibration.json"),
        (reference / "panel_manifest.json", "panel_manifest.json"),
        (selection, "selection.json"),
    ]:
        shutil.copyfile(src, root / name)
        files[name] = digest(root / name)
    folds = _read(root / "calibration.json")
    seconds = 0
    for i, key in enumerate(keys):
        old = _read(reference / "specs" / (key + ".json"))
        target = root / old["input"]
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(reference / old["input"], target)
        assert digest(target) == old["input_sha256"]
        files[old["input"]] = old["input_sha256"]
        spec = {
            k: old[k]
            for k in ("key", "n", "seed", "input", "input_sha256", "seconds", "calibration_fold")
        }
        assert spec["seed"] == 1 and spec["seconds"] == s4_ablation.BUDGETS[spec["n"]]
        spec.update(
            order=ARMS[i % 4 :] + ARMS[: i % 4],
            config={a: _config(a, spec, folds[spec["calibration_fold"]]) for a in ARMS},
            entry=f"scripts/run_initial_factorial.py worker --root ROOT --key {key} --arm ARM --cpu CPU",
        )
        path = root / "specs" / (key + ".json")
        atomic_json(path, spec)
        files[path.relative_to(root).as_posix()] = digest(path)
        seconds += spec["seconds"] * 4
    m = dict(
        source=source_hash(),
        commit=_head(),
        files=files,
        arms=ARMS,
        blocks=keys,
        runs=4 * len(keys),
        seed=1,
        host=host,
        solver_seconds=seconds,
        window_seconds=plan["window_seconds"],
        deadline_epoch=plan["deadline_epoch"],
        classification="development_diagnostic_screening",
        options=OPTIONS,
    )
    atomic_json(root / "manifest.json", m)
    atomic_json(root / "manifest.sha256.json", dict(sha256=digest(root / "manifest.json")))


def validate_inputs(root: Path) -> dict:
    """Verify sealed source, matrix, options, original input hashes and exact config."""
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    m = _read(root / "manifest.json")
    assert m["source"] == source_hash() and m["commit"] == _head()
    assert tuple(m["arms"]) == ARMS and m["options"] == OPTIONS and m["seed"] == 1
    assert m["blocks"] == _read(root / "selection.json")["blocks"]
    assert len(set(m["blocks"])) * 4 == m["runs"]
    assert (
        digest(root / "panel_manifest.json") == PANEL and digest(root / "calibration.json") == CAL
    )
    for name, sha in m["files"].items():
        assert digest(root / name) == sha, name
    folds = _read(root / "calibration.json")
    total = 0
    for key in m["blocks"]:
        s = _read(root / "specs" / (key + ".json"))
        assert s["key"] == key and s["seed"] == 1 and s["seconds"] == s4_ablation.BUDGETS[s["n"]]
        assert len(s["order"]) == 4 and set(s["order"]) == set(ARMS)
        assert s["input_sha256"] == digest(root / s["input"])
        for arm in ARMS:
            assert s["config"][arm] == json.loads(
                json.dumps(_config(arm, s, folds[s["calibration_fold"]]))
            )
        total += s["seconds"] * 4
    assert total == m["solver_seconds"]
    return m


def preflight(root: Path, key: str) -> None:
    """Audit all four initial populations, same-construction streams and fresh exact."""
    m = validate_inputs(root)
    s = _read(root / "specs" / (key + ".json"))
    pb = load_instance(root / s["input"])
    cfg = s4_ablation.arm_config("A0_FULL_S4", s["seed"], s["seconds"])
    checks = {}
    for arm, opt in OPTIONS.items():
        e: dict = {}
        gw = EvaluationGateway(pb, Archive(), decode_cache_size=cfg.decode_cache_size)
        trace: list = []
        gw.candidate_sink = trace.append
        if opt["initial_construction"] == "g1":
            make = gap_initializer(pb, cfg, e, retention=opt["initial_retention"])
        elif opt["initial_retention"] == "global_pool":
            make = pool_initializer(pb, cfg, e)
        else:
            make = charged_initializer(pb, cfg, "BILL_MOEAD", e)
        members = make(gw, RNGManager(cfg.seed))
        assert len(members) == 100 and len(trace) == gw.counts["exact"] == 140
        sample = members[::20] + list(gw.archive.members[:3])
        for c in sample:
            assert c.evaluation.feasible and c.evaluation == evaluate(pb, c.genotype, c.schedule)
        checks[arm] = {k: e[k] for k in ("initial_population_sha256", "initial_archive_sha256")}
        checks[arm].update(
            stream=object_digest([solution(c) for c in trace]),
            originals=object_digest([solution(c) for c in trace[:100]]),
            fresh_exact=len(sample),
            initialization_seconds=e["initialization_seconds"],
        )
    for a, b in [(ARMS[0], ARMS[1]), (ARMS[2], ARMS[3])]:
        assert checks[a]["stream"] == checks[b]["stream"]
        assert checks[a]["initial_archive_sha256"] == checks[b]["initial_archive_sha256"]
    assert len({c["originals"] for c in checks.values()}) == 1
    path = root / "preflight" / (key + ".json")
    atomic_json(
        path,
        dict(
            source=m["source"],
            host=socket.gethostname(),
            input_sha256=s["input_sha256"],
            policies=checks,
        ),
    )
    atomic_json(path.with_suffix(".sha256.json"), dict(sha256=digest(path)))


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Run one complete S4 with the declared initializer and sealed diagnostics."""
    os.sched_setaffinity(0, {cpu})
    m = validate_inputs(root)
    s = _read(root / "specs" / (key + ".json"))
    pp = root / "preflight" / (key + ".json")
    assert digest(pp) == _read(pp.with_suffix(".sha256.json"))["sha256"]
    check = _read(pp)
    assert check["source"] == m["source"] and check["host"] == socket.gethostname() == m["host"]
    assert check["input_sha256"] == s["input_sha256"]
    out = root / "runs" / key / arm
    summary = run_case(
        "A0_FULL_S4",
        root / s["input"],
        out,
        seed=s["seed"],
        seconds=s["seconds"],
        calibration=_read(root / "calibration.json")[s["calibration_fold"]],
        campaign_output=True,
        **OPTIONS[arm],
    )
    assert (
        summary["independent_exact_audit"] == "passed"
        and summary["elapsed_seconds"] >= s["seconds"] - 0.05
    )
    assert _read(out / "algorithm_config.json") == s["config"][arm]
    cfg = _read(out / "config.json")
    assert cfg["source"]["commit"] == m["commit"] and cfg["source"]["status"] == ""
    assert cfg["instance_sha256"] == s["input_sha256"] and not cfg["engineering_override"]
    init = _read(out / "initialization.json")
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert init[field] == check["policies"][arm][field]
    atomic_json(
        out / "complete.json",
        dict(
            source=m["source"],
            key=key,
            arm=arm,
            manifest=digest(out / "manifest.json"),
            spec=digest(root / "specs" / (key + ".json")),
        ),
    )


def main() -> None:
    """Prepare, validate, audit or dispatch the explicit four-factorial campaign."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "validate", "preflight", "dispatch", "worker"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--host", default="neu01-2288H-V6")
    p.add_argument("--key")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--cpu", type=int)
    p.add_argument("--cpus")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.reference, a.selection, a.host)
    elif a.action == "validate":
        print(validate_inputs(a.root)["runs"])
    elif a.action == "preflight":
        preflight(a.root, a.key)
    elif a.action == "worker":
        worker(a.root, a.key, a.arm, a.cpu)
    else:
        import time

        cores = [int(c) for c in a.cpus.split(",")]
        topo = {
            tuple(
                (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
                for f in ("physical_package_id", "core_id")
            )
            for c in cores
        }
        assert len(cores) == len(topo) and set(cores) <= os.sched_getaffinity(0)
        m = validate_inputs(a.root)
        # The dispatcher drains normally before this absolute deadline. No running job is cut short.
        if not (a.root / "start.json").exists():
            atomic_json(
                a.root / "start.json",
                dict(
                    started=time.time(),
                    deadline=m["deadline_epoch"],
                    dispatch="flat_ops_v1",
                    parallel=len(cores),
                ),
            )
        result = dispatch(
            a.root, sys.modules[__name__], cores, Path(__file__).resolve(), interval=2
        )
        print(json.dumps(result), flush=True)
        if result["state"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
