"""Prepare and execute the S4 launch-point comparison (G1/E2/GE3 against N0, random, N7).

Six arms, all with P1 (A7/A8 out of the Q pool, Archive-only in the final 5%) and
the record-only trace (user, 2026-11-08):

- ``N0_P1``: gap seeds over the whole front, fixed 20% LS share (control);
- ``R1_P1_RANDPOINT``: uniformly random front point, fixed share;
- ``G1_P1_GP``: Thompson sampling of a Gaussian-RBF field over (ray weight, time);
- ``E2_P1_EQM``: equimarginal weight segments with a diminishing return and the
  LS share set by that marginal return;
- ``GE3_P1_GP_EQM``: field over (ray weight, local LS investment) with the share
  set by its best marginal return;
- ``N7_P1_SHARE``: gap seeds with the average-rate equimarginal share (reference
  that separates where LS goes from how much LS time is spent).

Two block groups, prepared as separate roots so each can run on its own cores:

- ``base``: every 100/200-job instance of the frozen 120-input panel (80 blocks);
- ``small`` (user 2026-11-08, if time allows): every 50-job instance of the frozen
  panel (40 blocks, 600 seconds), run on the base cores after the base group;
- ``large``: extended 500-job instances (same generator and seed formula), 6000
  seconds each, Q calibration from the 200-job folds (user decision). To finish by
  the user's deadline the large group keeps only the replicates in
  ``LARGE_REPLICATES`` of every family, fixed before any result (never by solver
  performance); replicates 1-2 have three regions and 3-4 five.

Seed 9 (unused by any earlier S4 batch). Judgement and analysis are fixed in the
batch's launch record before any result. Preflight, worker and completion checks
are those of ``run_s4_moead``.
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

ARMS = (
    "N0_P1",
    "R1_P1_RANDPOINT",
    "G1_P1_GP",
    "E2_P1_EQM",
    "GE3_P1_GP_EQM",
    "N7_P1_SHARE",
)
CONTROL = "N0_P1"
SEEDS = (9,)
GROUPS = {"base": (100, 200), "large": (500,), "small": (50,)}
LARGE_REPLICATES = (1,)  # deadline 2026-11-08 23:00 (user): one replicate per family
GROUP_BLOCKS = {"base": 80, "large": 10 * len(LARGE_REPLICATES), "small": 40}
PANEL_SIZES = (50, 100, 200)
# Load-power copy of the 40 extended 500-job inputs (manifest SHA256).
PANEL500 = "0f1138afbb889e263de499a8198070b3be6c4ab9133ff680914b9761153fb36b"
LARGE_SECONDS = 6000
DESIGN = "s4-space/v1"
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
    """Kept 500-job blocks; the 200-job calibration folds alternate over their sorted keys."""
    assert all(row["jobs"] == 500 for row in records)
    kept = [r for r in records if int(r["key"].rsplit("_r", 1)[1]) in LARGE_REPLICATES]
    blocks = []
    for i, row in enumerate(sorted(kept, key=lambda r: r["key"])):
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
    root: Path, panel: Path, calibration: Path, host: str, shard: str = "0/1", group: str = "base"
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
                f"scripts/run_s4_space.py worker --root ROOT --key {b['block']} --arm ARM --cpu CPU"
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
    p.add_argument("--group", choices=tuple(GROUPS), default="base")
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
