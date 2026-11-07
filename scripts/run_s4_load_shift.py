"""Prepare and execute S4 versus S4 without right shift under load-dependent power.

Inputs are the frozen 120-instance panel's derived copy that only adds
``power_model`` (beta 0.4, kappa 1 / 0.25). Both arms share each block's
instance, seed, calibration fold and BILL4 initializer; arm order alternates.
Development screen: it asks whether the right shift helps once power depends on
the running load, not which algorithm is best.
"""

import argparse
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

from run_lsv2 import source_hash
from slot_assignment_dispatch import dispatch

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import PowerModel
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_ablation
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.lsv2_time2 import calibration_for
from geo_llm_scheduler.experiments.trans_runner import digest, run_case
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager

PANEL = "d21eb49116b089d410163265800b00ee2512174f80cc5226322c8b1fb0fad555"
CAL = "45d79eb46cefb2ec4ca28d62554e5e2529547c5739f4c3ac9ab94526c4c2cf9c"
LOAD = PowerModel(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
ARMS = ("A0_FULL_S4", "A1_NO_SHIFT")
SEEDS = (1,)
RUNS = 2 * 120 * len(SEEDS)
MODE: dict[str, set[str]] = {"A0_FULL_S4": {"common_fast"}, "A1_NO_SHIFT": set()}
REPO = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _config(arm: str, seed: int, seconds: int, cal: dict) -> dict:
    return s4_ablation.configuration_record(arm, s4_ablation.arm_config(arm, seed, seconds), cal)


def plan(records: list[dict], folds: dict) -> list[dict]:
    """Every panel instance and seed; calibration folds follow the original mapping."""
    groups: Counter = Counter()
    blocks = []
    for row in sorted(records, key=lambda r: r["key"]):
        key, n = row["key"], row["jobs"]
        label, _ = calibration_for(dict(panel="frozen120", key=key, n=n), folds, groups[n])
        groups[n] += 1
        for seed in SEEDS:
            blocks.append(
                dict(block=f"{key}_s{seed}", key=key, n=n, seed=seed, fold=label, row=row)
            )
    return blocks


def shard_blocks(blocks: list[dict], shard: str = "0/1") -> list[tuple[int, dict]]:
    """Deterministic balanced shard ``k/n`` of the full plan, with each block's global index.

    Blocks are dealt round-robin after sorting by budget (longest first) and name, so
    every shard gets nearly the same number of blocks of each size. ``0/1`` is the full
    plan. The global index keeps the full design's alternating arm order.
    """
    k, n = (int(x) for x in shard.split("/"))
    assert 0 <= k < n
    dealt = sorted(
        enumerate(blocks), key=lambda ib: (-s4_ablation.BUDGETS[ib[1]["n"]], ib[1]["block"])
    )
    return sorted((ib for j, ib in enumerate(dealt) if j % n == k), key=lambda ib: ib[0])


def prepare(root: Path, panel: Path, calibration: Path, host: str, shard: str = "0/1") -> None:
    """Seal inputs, the declared power model, fold mapping, commands and arm settings."""
    assert digest(panel / "manifest.json") == PANEL and digest(calibration) == CAL
    panel_manifest = _read(panel / "manifest.json")
    assert PowerModel(**panel_manifest["power_model"]) == LOAD
    records = panel_manifest["instances"]
    assert Counter(r["jobs"] for r in records) == {50: 40, 100: 40, 200: 40}
    folds = _read(calibration)
    blocks = plan(records, folds)
    assert 2 * len(blocks) == RUNS
    selected = shard_blocks(blocks, shard)
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for p, name in [
        (calibration, "calibration.json"),
        (panel / "manifest.json", "panel_manifest.json"),
    ]:
        shutil.copyfile(p, root / name)
        files[name] = digest(root / name)
    for i, b in selected:
        row = b["row"]
        target = root / row["path"]
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(panel / row["path"], target)
        assert digest(target) == row["sha256"]
        files[row["path"]] = row["sha256"]
        t = s4_ablation.BUDGETS[b["n"]]
        cal = folds[b["fold"]]
        spec = dict(
            key=b["block"],
            instance=b["key"],
            family=b["key"].split("_")[2],
            n=b["n"],
            seed=b["seed"],
            input=row["path"],
            input_sha256=row["sha256"],
            seconds=t,
            calibration_fold=b["fold"],
            power_model=panel_manifest["power_model"],
            order=list(ARMS if i % 2 == 0 else ARMS[::-1]),
            config={a: _config(a, b["seed"], t, cal) for a in ARMS},
            entry=f"scripts/run_s4_load_shift.py worker --root ROOT --key {b['block']} --arm ARM --cpu CPU",
        )
        path = root / "specs" / (b["block"] + ".json")
        atomic_json(path, spec)
        files[path.relative_to(root).as_posix()] = digest(path)
    m = dict(
        source=source_hash(),
        commit=_head(),
        files=files,
        arms=ARMS,
        blocks=[b["block"] for _, b in selected],
        seeds=SEEDS,
        runs=2 * len(selected),
        shard=shard,
        host=host,
        level="development screen",
        window_seconds=20 * 3600,
        power_model=panel_manifest["power_model"],
        solver_seconds=sum(2 * s4_ablation.BUDGETS[b["n"]] for _, b in selected),
    )
    atomic_json(root / "manifest.json", m)
    atomic_json(root / "manifest.sha256.json", dict(sha256=digest(root / "manifest.json")))


def validate_inputs(root: Path) -> dict:
    """Reject any matrix, policy, power-model, source or input drift before dispatch."""
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    m = _read(root / "manifest.json")
    assert m["commit"] == _head() and m["source"] == source_hash()
    assert tuple(m["arms"]) == ARMS and m["runs"] == 2 * len(set(m["blocks"]))
    assert PowerModel(**m["power_model"]) == LOAD
    assert (
        digest(root / "panel_manifest.json") == PANEL and digest(root / "calibration.json") == CAL
    )
    for name, sha in m["files"].items():
        assert digest(root / name) == sha, name
    folds = _read(root / "calibration.json")
    full = plan(_read(root / "panel_manifest.json")["instances"], folds)
    expected = {b["block"] for _, b in shard_blocks(full, m.get("shard", "0/1"))}
    assert expected == set(m["blocks"]) and len(m["blocks"]) == len(expected)
    for key in m["blocks"]:
        s = _read(root / "specs" / (key + ".json"))
        assert s["seconds"] == s4_ablation.BUDGETS[s["n"]] and set(s["order"]) == set(ARMS)
        assert s["seed"] in SEEDS and PowerModel(**s["power_model"]) == LOAD
        for arm in ARMS:
            want = json.loads(
                json.dumps(_config(arm, s["seed"], s["seconds"], folds[s["calibration_fold"]]))
            )
            assert s["config"][arm] == want
    return m


def preflight(root: Path, key: str) -> None:
    """Fingerprint the shared BILL4 initializer under load power; audit a sample."""
    m = _read(root / "manifest.json")
    s = _read(root / "specs" / (key + ".json"))
    assert digest(root / s["input"]) == s["input_sha256"]
    problem = load_instance(root / s["input"])
    assert problem.power_model == LOAD
    cfg = s4_ablation.arm_config(ARMS[0], s["seed"], s["seconds"])
    gw = EvaluationGateway(problem, Archive(), decode_cache_size=cfg.decode_cache_size)
    evidence: dict = {}
    members = charged_initializer(problem, cfg, "BILL_MOEAD", evidence)(gw, RNGManager(s["seed"]))
    assert len(members) == 100 and evidence["policy"] == "bill4"
    sample = members[::20] + list(gw.archive.members[:3])
    for c in sample:
        assert c.evaluation.feasible and c.evaluation == evaluate(problem, c.genotype, c.schedule)
    path = root / "preflight" / (key + ".json")
    atomic_json(
        path,
        dict(
            source=m["source"],
            input_sha256=s["input_sha256"],
            host=socket.gethostname(),
            bill4={k: evidence[k] for k in ("initial_population_sha256", "initial_archive_sha256")},
            fresh_exact=len(sample),
        ),
    )
    atomic_json(path.with_suffix(".sha256.json"), dict(sha256=digest(path)))


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Verify only complete, sealed runs with the declared provenance."""
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


def polish_modes(out: Path) -> set[str]:
    """Right-shift neighborhoods actually used by the run's evaluated or empty calls."""
    path = out / "local_search.json.gz"
    if not path.exists():
        return set()
    with gzip.open(path, "rt") as fh:
        rows = json.load(fh).get("polish", [])
    return {r["diagnostics"]["mode"] for r in rows if isinstance(r.get("diagnostics"), dict)}


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Run one sealed paired task, cross-check initializer, power model and neighborhood."""
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
    assert arm in s["order"] and digest(root / s["input"]) == s["input_sha256"]
    assert load_instance(root / s["input"]).power_model == LOAD
    pp = root / "preflight" / (key + ".json")
    assert digest(pp) == _read(pp.with_suffix(".sha256.json"))["sha256"]
    pref = _read(pp)
    assert pref["host"] == socket.gethostname() == m["host"]
    assert pref["input_sha256"] == s["input_sha256"]
    assert digest(root / "calibration.json") == CAL
    cal = _read(root / "calibration.json")[s["calibration_fold"]]
    out = root / "runs" / key / arm
    summary = run_case(
        arm,
        root / s["input"],
        out,
        seed=s["seed"],
        seconds=s["seconds"],
        calibration=cal,
        campaign_output=True,
    )
    cfg = _read(out / "config.json")
    assert cfg["source"]["commit"] == m["commit"] and cfg["source"]["status"] == ""
    assert cfg["instance_sha256"] == s["input_sha256"] and cfg["seed"] == s["seed"]
    assert not cfg["engineering_override"]
    assert cfg["seconds_including_initialization"] == s["seconds"]
    assert summary["independent_exact_audit"] == "passed"
    assert summary["elapsed_seconds"] >= s["seconds"] - 0.05
    assert _read(out / "algorithm_config.json") == s["config"][arm]
    init = _read(out / "initialization.json")
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert init[field] == pref["bill4"][field]
    assert polish_modes(out) <= MODE[arm]
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
    """Prepare, validate, preflight, run one case or dispatch on distinct physical cores."""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "validate", "preflight", "dispatch", "worker"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--panel", type=Path)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--host", default="neu01-2288H-V6")
    p.add_argument("--key")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--cpu", type=int)
    p.add_argument("--cpus")
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.panel, a.calibration, a.host, a.shard)
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
        assert len(cores) == len(topo) >= 1 and set(cores) <= os.sched_getaffinity(0)
        result = dispatch(
            a.root, sys.modules[__name__], cores, Path(__file__).resolve(), interval=2
        )
        print(json.dumps(result), flush=True)
        if result["state"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
