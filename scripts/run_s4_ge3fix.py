"""Prepare and execute the S4 GE3 rerun after the share-controller fix (8c6ebc1).

The launch-point batch (``run_s4_space``, seed 9) found GE3 far behind, but a
profile showed its share controller recomputing the local investment of every
front point at each allowance check, which halved its evaluations. After the fix
the user asked (2026-11-08) to rerun GE3 against two references only:

- ``N0_P1``: gap seeds over the whole front, fixed 20% LS share (control);
- ``N7_P1_SHARE``: gap seeds with the average-rate equimarginal share (same share
  mechanism family as GE3, so GE3 against N7 isolates the learned launch points);
- ``GE3_P1_GP_EQM``: field over (ray weight, local LS investment) with the share
  set by its best marginal return.

Two block groups, prepared as separate roots so each can run on its own cores:

- ``panel``: every 50/100/200-job instance of the frozen 120-input panel;
- ``large``: the extended 500-job inputs, 6000 seconds each, Q calibration from the
  200-job folds, families ``LARGE_FAMILIES`` at replicate 1 only. The count was
  fixed by the user's 08:00 deadline before any result (never by performance);
  the fold mapping equals the earlier launch-point batch.

Seed 9 as in that batch (same blocks, so the fixed GE3 is comparable block by
block). Judgement and analysis are fixed in the launch record before any result.
Preflight, worker and completion checks are those of ``run_s4_moead``.
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

ARMS = ("N0_P1", "N7_P1_SHARE", "GE3_P1_GP_EQM")
CONTROL = "N0_P1"
SEEDS = (9,)
GROUPS = {"panel": (50, 100, 200), "large": (500,)}
LARGE_REPLICATES = (1,)
LARGE_FAMILIES = tuple(range(1, 10))  # deadline 2026-11-09 08:00 (user): f01-f09
GROUP_BLOCKS = {"panel": 120, "large": len(LARGE_FAMILIES) * len(LARGE_REPLICATES)}
PANEL_SIZES = (50, 100, 200)
# Load-power copy of the 40 extended 500-job inputs (manifest SHA256).
PANEL500 = "0f1138afbb889e263de499a8198070b3be6c4ab9133ff680914b9761153fb36b"
LARGE_SECONDS = 6000
DESIGN = "s4-ge3fix/v1"
assert set(ARMS) <= set(s4_moead.EXTRA_ARMS)
# Batch-local budgets: the shared table stays as historical scripts expect it.
BUDGETS = {**s4_ablation.BUDGETS, 500: LARGE_SECONDS}
base.MODE.update({a: {"common_fast"} for a in ARMS})

__all__ = [
    "ARMS",
    "CONTROL",
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


def _order(index: int) -> list[str]:
    """Rotate the arms' start order cyclically by block index."""
    k = index % len(ARMS)
    return list(ARMS[k:] + ARMS[:k])


def plan_large(records: list[dict], folds: dict) -> list[dict]:
    """Kept 500-job blocks; the 200-job folds alternate as in the launch-point batch.

    That batch kept replicate 1 of all ten families and alternated the folds over
    their sorted keys; dropping family 10 leaves every kept block's fold unchanged.
    """
    assert all(row["jobs"] == 500 for row in records)
    kept = [r for r in records if int(r["key"].rsplit("_r", 1)[1]) in LARGE_REPLICATES]
    blocks = []
    for i, row in enumerate(sorted(kept, key=lambda r: r["key"])):
        if int(row["key"].split("_")[2][1:]) not in LARGE_FAMILIES:
            continue
        label = f"n200_f{i % 2}"
        assert row["key"] not in folds[label]["keys"]
        for seed in SEEDS:
            blocks.append(
                dict(
                    block=f"{row['key']}_s{seed}",
                    key=row["key"],
                    n=500,
                    seed=seed,
                    fold=label,
                    row=row,
                )
            )
    return blocks


def plan(records: list[dict], folds: dict, group: str) -> list[dict]:
    """Every block of a group at the batch seed."""
    if group == "large":
        return plan_large(records, folds)
    assert base.SEEDS == (1,)
    return [
        dict(b, block=f"{b['key']}_s{seed}", seed=seed)
        for b in base.plan(records, folds)
        for seed in SEEDS
    ]


def select(blocks: list[dict], group: str) -> list[dict]:
    """The blocks whose job count belongs to the group."""
    return [b for b in blocks if b["n"] in GROUPS[group]]


def shard_blocks(blocks: list[dict], shard: str = "0/1") -> list[tuple[int, dict]]:
    """``run_s4_moead.shard_blocks`` with the batch-local budgets (500 jobs included)."""
    k, n = (int(x) for x in shard.split("/"))
    assert 0 <= k < n
    dealt = sorted(enumerate(blocks), key=lambda ib: (-BUDGETS[ib[1]["n"]], ib[1]["block"]))
    return sorted((ib for j, ib in enumerate(dealt) if j % n == k), key=lambda ib: ib[0])


def _check_panel(panel: Path, group: str) -> dict:
    manifest = _read(panel / "manifest.json")
    assert PowerModel(**manifest["power_model"]) == base.LOAD
    counts = Counter(r["jobs"] for r in manifest["instances"])
    if group == "large":
        assert digest(panel / "manifest.json") == PANEL500
        assert counts == {500: 40}
    else:
        assert digest(panel / "manifest.json") == base.PANEL
        assert counts == {n: 40 for n in PANEL_SIZES}
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
            order=_order(i),
            config={a: base._config(a, b["seed"], t, cal) for a in ARMS},
            entry=(
                f"scripts/run_s4_ge3fix.py worker --root ROOT --key {b['block']} --arm ARM --cpu CPU"
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
    """Reject any matrix, policy, power-model, source or input drift before dispatch."""
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    m = _read(root / "manifest.json")
    group = m["group"]
    assert m["commit"] == base._head() and m["source"] == source_hash()
    assert m["design"] == DESIGN and tuple(m["seeds"]) == SEEDS and group in GROUPS
    assert tuple(m["arms"]) == ARMS and m["runs"] == len(ARMS) * len(set(m["blocks"]))
    assert PowerModel(**m["power_model"]) == base.LOAD
    want_panel = PANEL500 if group == "large" else base.PANEL
    assert digest(root / "panel_manifest.json") == want_panel
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
        assert s["seconds"] == BUDGETS[s["n"]] and set(s["order"]) == set(ARMS)
        assert len(s["order"]) == len(ARMS) and s["n"] in GROUPS[group]
        assert s["seed"] in SEEDS and PowerModel(**s["power_model"]) == base.LOAD
        for arm in ARMS:
            want = json.loads(
                json.dumps(base._config(arm, s["seed"], s["seconds"], folds[s["calibration_fold"]]))
            )
            assert s["config"][arm] == want
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
