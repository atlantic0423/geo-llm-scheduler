"""Prepare and execute the F3 cost-control batch (user, 2026-11-11), reduced matrix.

The first F3 batch (``run_s4_guided``) showed that rebuilding the incumbent in the
variation step helps at 50 and 100 jobs and loses about 12% HV at 200 jobs, where each
guided child costs far more wall time than an ordinary child yet is rarely inserted.
This batch tests two ways to control that cost, both on the neutral (direction-free)
variant ``V2``:

- ``G4_GUIDED_RAND_D4``: only four Jobs are destroyed per rebuild;
- ``GS_GUIDED_RAND_SWITCH``: the routing probability follows the decayed HV gain per
  second of the guided path against the ordinary path (``p`` between 0.02 and 0.2);
- ``G8_GUIDED_RAND`` (200 jobs only): ``V2_GUIDED_RAND`` repeated with the extra
  records, the matched control for both; ``N7_P1_SHARE`` (V0) is the working version.

One root, both sizes: 100 jobs run V0, G4, GS and 200 jobs run V0, G8, G4, GS. Because
the batch must finish within about three hours, the matrix is reduced from
``NEXT_STEP_SPEC.md`` (user, 2026-11-11): 200 jobs use all 40 instances at
seed 14 (40 blocks); 100 jobs use replicates 1-2 (20 instances) at seeds 14 and 15
(40 blocks); budgets 1200 and 2400 s including initialization. Frozen load-power panel.
Judgement and analysis follow the launch record. Preflight, worker and completion
checks are those of ``run_s4_moead``.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import run_s4_moead as base
from run_lsv2 import source_hash
from slot_assignment_dispatch import dispatch

from geo_llm_scheduler.domain.models import PowerModel
from geo_llm_scheduler.experiments import s4_ablation, s4_moead
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.trans_runner import digest

CONTROL = "N7_P1_SHARE"
ARMS_BY_SIZE = {
    100: (CONTROL, "G4_GUIDED_RAND_D4", "GS_GUIDED_RAND_SWITCH"),
    200: (CONTROL, "G8_GUIDED_RAND", "G4_GUIDED_RAND_D4", "GS_GUIDED_RAND_SWITCH"),
}
ARMS = tuple(dict.fromkeys(a for arms in ARMS_BY_SIZE.values() for a in arms))
# size -> (largest instance replicate number used, seeds)
SELECTION = {100: (4, (16, 17)), 200: (4, (16, 17))}
SEEDS = (16, 17)
GROUPS = {"panel": (100, 200)}
GROUP_BLOCKS = {"panel": 160}
PANEL_SIZES = (50, 100, 200)
DESIGN = "s4-cost/v1"
assert set(ARMS) <= set(s4_moead.EXTRA_ARMS)
BUDGETS = dict(s4_ablation.BUDGETS)
base.MODE.update({a: {"common_fast"} for a in ARMS})

__all__ = [
    "ARMS",
    "ARMS_BY_SIZE",
    "CONTROL",
    "SELECTION",
    "DESIGN",
    "GROUPS",
    "SEEDS",
    "atomic_json",
    "complete",
    "plan",
    "prepare",
    "select",
    "validate_inputs",
    "worker",
]

complete = base.complete
preflight = base.preflight
worker = base.worker


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _order(index: int, size: int) -> list[str]:
    """Rotate the size's arms' start order cyclically by block index."""
    arms = ARMS_BY_SIZE[size]
    k = index % len(arms)
    return list(arms[k:] + arms[:k])


def plan(records: list[dict], folds: dict, group: str = "panel") -> list[dict]:
    """The selected panel blocks: per size, replicates up to a cap and that size's seeds."""
    assert group == "panel" and base.SEEDS == (1,)
    out = []
    for b in base.plan(records, folds):
        if b["n"] not in SELECTION:
            continue
        replicate_cap, seeds = SELECTION[b["n"]]
        if int(b["key"].split("_r")[-1][:2]) > replicate_cap:
            continue
        out += [dict(b, block=f"{b['key']}_s{seed}", seed=seed) for seed in seeds]
    return out


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


def prepare(
    root: Path, panel: Path, calibration: Path, host: str, shard: str = "0/1", group: str = "panel"
) -> None:
    """Seal inputs, the declared power model, fold mapping, commands and all arms' settings."""
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
            order=_order(i, b["n"]),
            config={a: base._config(a, b["seed"], t, cal) for a in ARMS_BY_SIZE[b["n"]]},
            entry=(
                f"scripts/run_s4_cost.py worker --root ROOT --key {b['block']} --arm ARM --cpu CPU"
            ),
        )
        path = root / "specs" / (b["block"] + ".json")
        atomic_json(path, spec)
        files[path.relative_to(root).as_posix()] = digest(path)
    m = dict(
        source=source_hash(),
        commit=base._head(),
        design=DESIGN,
        group=group,
        sizes=list(GROUPS[group]),
        files=files,
        arms=ARMS,
        control=CONTROL,
        blocks=[b["block"] for _, b in selected],
        seeds=SEEDS,
        runs=sum(len(ARMS_BY_SIZE[b["n"]]) for _, b in selected),
        shard=shard,
        host=host,
        level="development screen",
        window_seconds=48 * 3600,
        power_model=panel_manifest["power_model"],
        solver_seconds=sum(len(ARMS_BY_SIZE[b["n"]]) * BUDGETS[b["n"]] for _, b in selected),
    )
    atomic_json(root / "manifest.json", m)
    atomic_json(root / "manifest.sha256.json", dict(sha256=digest(root / "manifest.json")))


def validate_inputs(root: Path) -> dict:
    """Reject any matrix, policy, power-model, source or input drift before dispatch."""
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    m = _read(root / "manifest.json")
    group = m["group"]
    assert m["commit"] == base._head() and m["source"] == source_hash()
    assert m["design"] == DESIGN and tuple(m["seeds"]) == SEEDS and group in GROUPS
    assert tuple(m["arms"]) == ARMS
    assert PowerModel(**m["power_model"]) == base.LOAD
    assert digest(root / "panel_manifest.json") == base.PANEL
    assert digest(root / "calibration.json") == base.CAL
    for name, sha in m["files"].items():
        assert digest(root / name) == sha, name
    folds = _read(root / "calibration.json")
    records = _read(root / "panel_manifest.json")["instances"]
    chosen = select(plan(records, folds, group), group)
    expected = {b["block"] for _, b in shard_blocks(chosen, m["shard"])}
    assert expected == set(m["blocks"]) and len(m["blocks"]) == len(expected)
    total_runs = 0
    for key in m["blocks"]:
        s = _read(root / "specs" / (key + ".json"))
        arms = ARMS_BY_SIZE[s["n"]]
        total_runs += len(arms)
        assert s["seconds"] == BUDGETS[s["n"]] and set(s["order"]) == set(arms)
        assert len(s["order"]) == len(arms) and s["n"] in GROUPS[group]
        assert s["seed"] in SEEDS and PowerModel(**s["power_model"]) == base.LOAD
        for arm in arms:
            want = json.loads(
                json.dumps(base._config(arm, s["seed"], s["seconds"], folds[s["calibration_fold"]]))
            )
            assert s["config"][arm] == want
    assert m["runs"] == total_runs
    return m


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
