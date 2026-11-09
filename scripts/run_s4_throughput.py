"""Prepare and execute the S4 throughput HV batch (user, 2026-11-09).

The level 1-3 throughput rewrites return bit-identical schedules, exact results and
LS billing values (``research/throughput_20261109``) and complete about 1.4-2x more
exact evaluations in the same wall time. This batch asks whether that throughput
becomes final HV. Both arms run the working version ``N7_P1_SHARE`` (B2 + P1 + N7)
from the same commit, input, seed and budget:

- ``C0_REF``: the frozen 582237f scheduling / exact / LS-billing code, entered
  through ``reference_runtime`` (control);
- ``C0F``: the current accelerated code.

Frozen 50/100/200-job panel (120 blocks), seed 11 (never used before), budgets
600/1200/2400 s including initialization. Judgement and analysis are fixed in the
launch record before any result. Preflight and completion checks are those of
``run_s4_moead``; the worker maps each arm label to the algorithm and engine.
"""

import argparse
import hashlib
import importlib
import json
import os
import pkgutil
import socket
import sys
from collections import Counter
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import run_s4_moead as base
from run_lsv2 import source_hash
from slot_assignment_dispatch import dispatch

import geo_llm_scheduler
from geo_llm_scheduler.domain.models import PowerModel
from geo_llm_scheduler.experiments import s4_ablation
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.trans_runner import digest, run_case
from geo_llm_scheduler.io.loaders import load_instance

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.throughput_20261109.reference_runtime import reference_runtime  # noqa: E402

ALGORITHM = "N7_P1_SHARE"
ARMS = ("C0_REF", "C0F")
ENGINE = {"C0_REF": "reference_582237f", "C0F": "accelerated"}
CONTROL = "C0_REF"
SEEDS = (11,)
GROUPS = {"panel": (50, 100, 200)}
GROUP_BLOCKS = {"panel": 120}
PANEL_SIZES = (50, 100, 200)
DESIGN = "s4-throughput/v1"
BUDGETS = dict(s4_ablation.BUDGETS)
REFERENCE_DIR = REPO / "research" / "throughput_20261109"
base.MODE.update({a: {"common_fast"} for a in ARMS})

complete = base.complete
preflight = base.preflight


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_hash() -> str:
    """Bind the frozen reference code and its runtime switch (outside ``source_hash``)."""
    paths = sorted(REFERENCE_DIR.glob("frozen/*.py")) + [REFERENCE_DIR / "reference_runtime.py"]
    return hashlib.sha256(
        json.dumps(
            [
                (p.relative_to(REPO).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                for p in paths
            ]
        ).encode()
    ).hexdigest()


def import_all() -> list[str]:
    """Import every package module, so the reference switch also reaches modules the
    run would otherwise import lazily (their bindings are fixed at import time).

    A module whose third-party dependency is not installed (e.g. plotting) cannot be
    imported by the run either; it is skipped and its name returned.
    """
    skipped = []
    for m in pkgutil.walk_packages(geo_llm_scheduler.__path__, "geo_llm_scheduler."):
        try:
            importlib.import_module(m.name)
        except ModuleNotFoundError as exc:
            if exc.name is None or exc.name.split(".")[0] == "geo_llm_scheduler":
                raise
            skipped.append(m.name)
    return skipped


def engine_for(arm: str) -> AbstractContextManager[None]:
    """The control runs inside ``reference_runtime``; the accelerated arm needs no switch."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}")
    return reference_runtime() if arm == CONTROL else nullcontext()


def _order(index: int) -> list[str]:
    """Alternate the arms' start order by block index."""
    k = index % len(ARMS)
    return list(ARMS[k:] + ARMS[:k])


def plan(records: list[dict], folds: dict, group: str = "panel") -> list[dict]:
    """Every panel block at the batch seed."""
    assert group == "panel" and base.SEEDS == (1,)
    return [
        dict(b, block=f"{b['key']}_s{seed}", seed=seed)
        for b in base.plan(records, folds)
        for seed in SEEDS
    ]


def select(blocks: list[dict], group: str) -> list[dict]:
    """The blocks whose job count belongs to the group."""
    return [b for b in blocks if b["n"] in GROUPS[group]]


def shard_blocks(blocks: list[dict], shard: str = "0/1") -> list[tuple[int, dict]]:
    """Deal blocks to shards longest budget first (as ``run_s4_moead.shard_blocks``)."""
    k, n = (int(x) for x in shard.split("/"))
    assert 0 <= k < n
    dealt = sorted(enumerate(blocks), key=lambda ib: (-BUDGETS[ib[1]["n"]], ib[1]["block"]))
    return sorted((ib for j, ib in enumerate(dealt) if j % n == k), key=lambda ib: ib[0])


def _check_panel(panel: Path, group: str) -> dict:
    manifest = _read(panel / "manifest.json")
    assert group == "panel" and PowerModel(**manifest["power_model"]) == base.LOAD
    assert digest(panel / "manifest.json") == base.PANEL
    assert Counter(r["jobs"] for r in manifest["instances"]) == {n: 40 for n in PANEL_SIZES}
    return manifest


def _config(seed: int, seconds: int, cal: dict) -> dict:
    return base._config(ALGORITHM, seed, seconds, cal)


def prepare(
    root: Path, panel: Path, calibration: Path, host: str, shard: str = "0/1", group: str = "panel"
) -> None:
    """Seal inputs, the declared power model, fold mapping, commands and both arms' settings."""
    assert group in GROUPS and digest(calibration) == base.CAL
    panel_manifest = _check_panel(panel, group)
    folds = _read(calibration)
    chosen = select(plan(panel_manifest["instances"], folds, group), group)
    assert len(chosen) == GROUP_BLOCKS[group]
    selected = shard_blocks(chosen, shard)
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for p, name in [
        (calibration, "calibration.json"),
        (panel / "manifest.json", "panel_manifest.json"),
    ]:
        target = root / name
        target.write_bytes(p.read_bytes())
        files[name] = digest(target)
    for i, b in selected:
        row = b["row"]
        target = root / row["path"]
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes((panel / row["path"]).read_bytes())
        assert digest(target) == row["sha256"]
        files[row["path"]] = row["sha256"]
        t = BUDGETS[b["n"]]
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
            order=_order(i),
            algorithm=ALGORITHM,
            engine=ENGINE,
            config={a: _config(b["seed"], t, cal) for a in ARMS},
            entry=(
                "scripts/run_s4_throughput.py worker --root ROOT "
                f"--key {b['block']} --arm ARM --cpu CPU"
            ),
        )
        path = root / "specs" / (b["block"] + ".json")
        atomic_json(path, spec)
        files[path.relative_to(root).as_posix()] = digest(path)
    m = dict(
        source=source_hash(),
        reference=reference_hash(),
        commit=base._head(),
        design=DESIGN,
        group=group,
        sizes=list(GROUPS[group]),
        files=files,
        arms=ARMS,
        engine=ENGINE,
        algorithm=ALGORITHM,
        control=CONTROL,
        blocks=[b["block"] for _, b in selected],
        seeds=SEEDS,
        runs=len(ARMS) * len(selected),
        shard=shard,
        host=host,
        level="development screen",
        window_seconds=48 * 3600,
        power_model=panel_manifest["power_model"],
        solver_seconds=sum(len(ARMS) * BUDGETS[b["n"]] for _, b in selected),
    )
    atomic_json(root / "manifest.json", m)
    atomic_json(root / "manifest.sha256.json", dict(sha256=digest(root / "manifest.json")))


def validate_inputs(root: Path) -> dict:
    """Reject any matrix, engine, power-model, source, reference or input drift."""
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    m = _read(root / "manifest.json")
    group = m["group"]
    assert m["commit"] == base._head() and m["source"] == source_hash()
    assert m["reference"] == reference_hash() and m["engine"] == ENGINE
    assert m["design"] == DESIGN and tuple(m["seeds"]) == SEEDS and group in GROUPS
    assert tuple(m["arms"]) == ARMS and m["runs"] == len(ARMS) * len(set(m["blocks"]))
    assert m["algorithm"] == ALGORITHM and PowerModel(**m["power_model"]) == base.LOAD
    assert digest(root / "panel_manifest.json") == base.PANEL
    assert digest(root / "calibration.json") == base.CAL
    for name, sha in m["files"].items():
        assert digest(root / name) == sha, name
    folds = _read(root / "calibration.json")
    records = _read(root / "panel_manifest.json")["instances"]
    chosen = select(plan(records, folds, group), group)
    expected = {b["block"] for _, b in shard_blocks(chosen, m["shard"])}
    assert expected == set(m["blocks"]) and len(m["blocks"]) == len(expected)
    for key in m["blocks"]:
        s = _read(root / "specs" / (key + ".json"))
        assert s["seconds"] == BUDGETS[s["n"]] and sorted(s["order"]) == sorted(ARMS)
        assert s["n"] in GROUPS[group] and s["seed"] in SEEDS
        assert PowerModel(**s["power_model"]) == base.LOAD
        assert s["algorithm"] == ALGORITHM and s["engine"] == ENGINE
        want = json.loads(
            json.dumps(_config(s["seed"], s["seconds"], folds[s["calibration_fold"]]))
        )
        assert all(s["config"][arm] == want for arm in ARMS)
    return m


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Run one sealed task on the arm's engine; the same checks as ``run_s4_moead.worker``."""
    os.sched_setaffinity(0, {cpu})
    m = _read(root / "manifest.json")
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    specpath = root / "specs" / (key + ".json")
    assert (
        m["source"] == source_hash()
        and m["reference"] == reference_hash()
        and m["commit"] == base._head()
        and digest(specpath) == m["files"][f"specs/{key}.json"]
    )
    s = _read(specpath)
    assert arm in s["order"] and digest(root / s["input"]) == s["input_sha256"]
    assert load_instance(root / s["input"]).power_model == base.LOAD
    pp = root / "preflight" / (key + ".json")
    assert digest(pp) == _read(pp.with_suffix(".sha256.json"))["sha256"]
    pref = _read(pp)
    assert pref["host"] == socket.gethostname() == m["host"]
    assert pref["input_sha256"] == s["input_sha256"]
    assert digest(root / "calibration.json") == base.CAL
    cal = _read(root / "calibration.json")[s["calibration_fold"]]
    out = root / "runs" / key / arm
    import_all()  # both arms, before any switch
    with engine_for(arm):
        summary = run_case(
            ALGORITHM,
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
    assert base.polish_modes(out) <= base.MODE[arm]
    atomic_json(
        out / "complete.json",
        dict(
            source=m["source"],
            key=key,
            arm=arm,
            engine=ENGINE[arm],
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
    p.add_argument("--group", choices=tuple(GROUPS), default="panel")
    p.add_argument("--key")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--cpu", type=int)
    p.add_argument("--cpus")
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.panel, a.calibration, a.host, a.shard, a.group)
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
