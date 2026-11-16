"""Prepare and execute the S4-B2 P1 terminal-5% confirmation batch (load-dependent power).

Two arms from the nine-arm screen, unchanged: ``M0_B2`` and ``M2_P1_T05``
(:mod:`geo_llm_scheduler.experiments.s4_moead`). Blocks are every frozen-panel
instance (50/100/200 jobs, 40 each) at seeds 1 and 2, minus the 40 blocks the
screen already used (100/200 jobs, replicates 1–2, seed 1): 80 + 120 = 200
blocks, 400 runs. Judgement (user, 2026-11-06): P1 against B2 only. Primary
analysis: the 120 blocks with 100/200 jobs; the 80 blocks with 50 jobs are a
secondary generalization check. Development confirmation; it does not change
the default algorithm.

Preflight, worker and completion checks are those of ``run_s4_moead`` (same
initializer fingerprint, exact audit, sealing and provenance rules).
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

ARMS = (s4_moead.BASE, "M2_P1_T05")
SEEDS = (1, 2)
SIZES = (50, 100, 200)
PRIMARY_SIZES = (100, 200)
FULL_BLOCKS = 120 * len(SEEDS)
SELECTED_BLOCKS = FULL_BLOCKS - 40
DESIGN = "s4-b2-p1-t05-confirmation/v1"

__all__ = [
    "ARMS",
    "DESIGN",
    "SEEDS",
    "atomic_json",
    "complete",
    "plan",
    "prepare",
    "screened",
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
    """Alternate the two arms' start order by block index."""
    k = index % len(ARMS)
    return list(ARMS[k:] + ARMS[:k])


def _replicate(block: dict) -> int:
    return int(block["key"].split("_r")[-1][:2])


def plan(records: list[dict], folds: dict) -> list[dict]:
    """Every panel instance at every seed; folds follow the screen's per-instance mapping."""
    assert base.SEEDS == (1,)
    blocks = []
    for b in base.plan(records, folds):
        for seed in SEEDS:
            blocks.append(dict(b, block=f"{b['key']}_s{seed}", seed=seed))
    return blocks


def screened(block: dict) -> bool:
    """True for a block the nine-arm screen ran (100/200 jobs, replicate ≤ 2, seed 1)."""
    return block["seed"] == 1 and block["n"] in base.SIZES and _replicate(block) <= 2


def select(blocks: list[dict]) -> list[dict]:
    """All blocks not already used by the screen."""
    return [b for b in blocks if not screened(b)]


def prepare(root: Path, panel: Path, calibration: Path, host: str, shard: str = "0/1") -> None:
    """Seal inputs, the declared power model, fold mapping, commands and both arms' settings."""
    assert digest(panel / "manifest.json") == base.PANEL and digest(calibration) == base.CAL
    panel_manifest = _read(panel / "manifest.json")
    assert PowerModel(**panel_manifest["power_model"]) == base.LOAD
    records = panel_manifest["instances"]
    assert Counter(r["jobs"] for r in records) == {n: 40 for n in SIZES}
    folds = _read(calibration)
    blocks = plan(records, folds)
    assert len(blocks) == FULL_BLOCKS
    chosen = select(blocks)
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
            primary=b["n"] in PRIMARY_SIZES,
            input=row["path"],
            input_sha256=row["sha256"],
            seconds=t,
            calibration_fold=b["fold"],
            power_model=panel_manifest["power_model"],
            order=_order(i),
            config={a: base._config(a, b["seed"], t, cal) for a in ARMS},
            entry=(
                "scripts/run_s4_p1_confirm.py worker --root ROOT --key "
                f"{b['block']} --arm ARM --cpu CPU"
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
        blocks=[b["block"] for _, b in selected],
        seeds=SEEDS,
        runs=len(ARMS) * len(selected),
        shard=shard,
        host=host,
        level="development confirmation",
        primary_sizes=PRIMARY_SIZES,
        excluded="nine-arm screen blocks: 100/200 jobs, replicates 1-2, seed 1",
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
        assert s["seed"] in SEEDS and PowerModel(**s["power_model"]) == base.LOAD
        assert s["primary"] == (s["n"] in PRIMARY_SIZES)
        assert not screened(dict(key=s["instance"], n=s["n"], seed=s["seed"]))
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
