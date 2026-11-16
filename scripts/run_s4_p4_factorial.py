"""Prepare and execute the S4 P4 confirmation and space x time factorial batch.

Five arms, all with P1 (A7/A8 out of the Q pool, Archive-only in the final 5%)
and the record-only trace. The 2 x 2 factorial crosses where LS is launched
with how much LS time is spent; the random-region arm is the same-seed
reference for the rate bandit (user, 2026-11-07):

- ``N0_P1``: gap seeds over the whole front, fixed 20% LS share (control);
- ``N3_P1_P4R3_RANDOM``: uniform random region of three, fixed share;
- ``N6_P1_P4R3_BANDIT``: region rate bandit, fixed share;
- ``N7_P1_SHARE``: gap seeds, equimarginal LS share;
- ``N8_P1_P4R3_BANDIT_SHARE``: region rate bandit and equimarginal share.

Blocks: every 100/200-job instance of the frozen panel at seed 6 (unused by any
earlier S4 batch): 80 blocks, 400 runs. Judgement and analysis are fixed in the
batch's launch record before any result. Development screen; it does not change
the default algorithm. Preflight, worker and completion checks are those of
``run_s4_moead``.
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
    "N3_P1_P4R3_RANDOM",
    "N6_P1_P4R3_BANDIT",
    "N7_P1_SHARE",
    "N8_P1_P4R3_BANDIT_SHARE",
)
CONTROL = "N0_P1"
SEEDS = (6,)
SIZES = (100, 200)
PANEL_SIZES = (50, 100, 200)
SELECTED_BLOCKS = 80
DESIGN = "s4-p1-p4-factorial/v2"
assert set(ARMS) <= set(s4_moead.EXTRA_ARMS)
base.MODE.update({a: {"common_fast"} for a in ARMS})

__all__ = [
    "ARMS",
    "CONTROL",
    "DESIGN",
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


def plan(records: list[dict], folds: dict) -> list[dict]:
    """Every panel instance at the batch seed; folds follow the screen's per-instance mapping."""
    assert base.SEEDS == (1,)
    return [
        dict(b, block=f"{b['key']}_s{seed}", seed=seed)
        for b in base.plan(records, folds)
        for seed in SEEDS
    ]


def select(blocks: list[dict]) -> list[dict]:
    """The 100/200-job blocks."""
    return [b for b in blocks if b["n"] in SIZES]


def prepare(root: Path, panel: Path, calibration: Path, host: str, shard: str = "0/1") -> None:
    """Seal inputs, the declared power model, fold mapping, commands and all arms' settings."""
    assert digest(panel / "manifest.json") == base.PANEL and digest(calibration) == base.CAL
    panel_manifest = _read(panel / "manifest.json")
    assert PowerModel(**panel_manifest["power_model"]) == base.LOAD
    records = panel_manifest["instances"]
    assert Counter(r["jobs"] for r in records) == {n: 40 for n in PANEL_SIZES}
    folds = _read(calibration)
    chosen = select(plan(records, folds))
    assert len(chosen) == SELECTED_BLOCKS
    selected = base.shard_blocks(chosen, shard)
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
            order=_order(i),
            config={a: base._config(a, b["seed"], t, cal) for a in ARMS},
            entry=(
                f"scripts/run_s4_p4_factorial.py worker --root ROOT --key {b['block']} --arm ARM --cpu CPU"
            ),
        )
        path = root / "specs" / (b["block"] + ".json")
        atomic_json(path, spec)
        files[path.relative_to(root).as_posix()] = digest(path)
    m = dict(
        source=source_hash(),
        commit=base._head(),
        design=DESIGN,
        files=files,
        arms=ARMS,
        control=CONTROL,
        blocks=[b["block"] for _, b in selected],
        seeds=SEEDS,
        runs=len(ARMS) * len(selected),
        shard=shard,
        host=host,
        level="development screen",
        sizes=SIZES,
        window_seconds=20 * 3600,
        power_model=panel_manifest["power_model"],
        solver_seconds=sum(len(ARMS) * s4_ablation.BUDGETS[b["n"]] for _, b in selected),
    )
    atomic_json(root / "manifest.json", m)
    atomic_json(root / "manifest.sha256.json", dict(sha256=digest(root / "manifest.json")))


def validate_inputs(root: Path) -> dict:
    """Reject any matrix, policy, power-model, source or input drift before dispatch."""
    assert digest(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    m = _read(root / "manifest.json")
    assert m["commit"] == base._head() and m["source"] == source_hash()
    assert m["design"] == DESIGN and tuple(m["seeds"]) == SEEDS
    assert tuple(m["arms"]) == ARMS and m["runs"] == len(ARMS) * len(set(m["blocks"]))
    assert PowerModel(**m["power_model"]) == base.LOAD
    assert (
        digest(root / "panel_manifest.json") == base.PANEL
        and digest(root / "calibration.json") == base.CAL
    )
    for name, sha in m["files"].items():
        assert digest(root / name) == sha, name
    folds = _read(root / "calibration.json")
    chosen = select(plan(_read(root / "panel_manifest.json")["instances"], folds))
    expected = {b["block"] for _, b in base.shard_blocks(chosen, m["shard"])}
    assert expected == set(m["blocks"]) and len(m["blocks"]) == len(expected)
    for key in m["blocks"]:
        s = _read(root / "specs" / (key + ".json"))
        assert s["seconds"] == s4_ablation.BUDGETS[s["n"]] and set(s["order"]) == set(ARMS)
        assert len(s["order"]) == len(ARMS) and s["n"] in SIZES
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
