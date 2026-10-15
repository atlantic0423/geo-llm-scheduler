"""Disjoint calibration and concurrent target-resource pilot before formal T0."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.experiments.auxiliary_instances import PRESSURES, paired_instance
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.runner import digest
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.macrosearch.trajectory_budget import ARM_SPECS

from .campaign import SIZES, _initial, make_config
from .common import sha, source, validate_complete


def technical_specs(control: Path, mode: str) -> list[dict]:
    """Freeze separate pilot/calibration parents, never counted as formal observations."""
    specs = []
    hosts = json.loads((control / "hosts.json").read_text())
    (control / "inputs").mkdir(parents=True, exist_ok=True)
    count = 32 if mode == "calibration" else 48
    for i in range(count):
        node = i % 2
        jobs = SIZES[i // 8] if mode == "calibration" else 200
        pressure = PRESSURES[(i // 2) % 4]
        seed = 771017 + i
        arm = "W1" if mode == "calibration" else list(ARM_SPECS)[i % 18]
        problem = paired_instance(jobs, 3, 12, pressure, i % 2, random.Random(202610990000 + i))
        path = control / "inputs" / f"{mode}-{i}.json"
        save_instance(problem, path)
        cfg = make_config(arm, seed, 90 if mode == "calibration" else 180, {})
        specs.append(
            {
                "id": f"{mode}-{i}-{arm}",
                "case": f"{mode}-{i}",
                "parent": f"{mode}-{i}",
                "host": hosts[node],
                "node": node,
                "arm": arm,
                "jobs": jobs,
                "input": path.relative_to(control).as_posix(),
                "input_sha256": sha(path),
                "initial_config": asdict(cfg),
                "config": asdict(cfg),
                "calibration": mode == "calibration",
            }
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        specs = [f.result() for f in [pool.submit(_initial, spec, control) for spec in specs]]
    for spec in specs:
        del spec["initial_config"]
    atomic_json(control / f"{mode}_specs.json", specs)
    return specs


def execute_technical(control: Path, output: Path, mode: str, node: int) -> None:
    """Run one bounded technical wave at real target concurrency and retain all failures."""
    if socket.gethostname() != json.loads((control / "hosts.json").read_text())[node]:
        raise ValueError("Wrong target host")
    specs = json.loads((control / f"{mode}_specs.json").read_text())
    provenance = json.loads((control / "source.json").read_text())
    own = [s for s in specs if s["node"] == node]
    (control / "specs").mkdir(exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    def one(spec: dict) -> dict:
        path = control / "specs" / f"{spec['id']}.json"
        atomic_json(path, spec)
        with (output / f"{spec['id']}.log").open("w") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.auxiliary_walk_20261016.worker",
                    "--spec",
                    str(path),
                    "--control",
                    str(control),
                    "--output",
                    str(output / spec["id"]),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
        if process.returncode:
            raise RuntimeError(f"Technical worker failed: {spec['id']} exit={process.returncode}")
        marker = validate_complete(output / spec["id"], spec, provenance)
        return {"id": spec["id"], "marker": marker}

    begin = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            results = list(pool.map(one, own))
    except Exception as error:
        atomic_json(output / f"{mode}_failure.json", {"error": str(error), "time": time.time()})
        raise
    atomic_json(
        output / f"{mode}_complete.json", {"results": results, "begin": begin, "end": time.time()}
    )


def thresholds(values: list[float]) -> list[float]:
    """Use pooled independent calibration order statistics; keep exact zero/tie bins."""
    if not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError("Need finite calibrated signals in [0,1]")
    ordered = sorted(values)
    return [ordered[(len(ordered) - 1) // 3], ordered[2 * (len(ordered) - 1) // 3]]


def freeze_calibration(control: Path, outputs: list[Path]) -> dict:
    """Verify all calibration runs and freeze seven mapped cutoff pairs."""
    specs = json.loads((control / "calibration_specs.json").read_text())
    provenance = json.loads((control / "source.json").read_text())
    signals = {arm: [] for arm in ("D1", "D2", "D3", "D4", "D5", "D6")}
    evidence = []
    for spec in specs:
        root = outputs[spec["node"]] / spec["id"]
        marker = validate_complete(root, spec, provenance)
        diag = json.loads((root / "diagnostics.json").read_text())
        for arm in signals:
            signals[arm].extend(diag["signals"][arm])
        evidence.append({"id": spec["id"], "marker_hash": digest(marker)})
    cutoffs = {arm: thresholds(values) for arm, values in signals.items()}
    for arm in ("C0", "L1", "A1"):
        cutoffs[arm] = cutoffs["D2"]
    atomic_json(
        control / "calibration_evidence.json",
        {
            "runs": evidence,
            "sample_counts": {a: len(v) for a, v in signals.items()},
            "cutoffs": cutoffs,
            "rule": "pooled per-run bounded reservoir order statistics 1/3 and 2/3; <=lo ->3, <=hi ->6, else10; reverse D6; zero/tie bins retained",
        },
    )
    atomic_json(control / "cutoffs.json", cutoffs)
    return cutoffs


def main() -> None:
    """Build or execute an independent technical calibration/resource wave."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "calibration", "pilot"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--node", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "build":
        atomic_json(args.control / "source.json", source())
        for mode in ("calibration", "pilot"):
            technical_specs(args.control, mode)
    else:
        execute_technical(args.control, args.output, args.mode, args.node)


if __name__ == "__main__":
    main()
