"""E15 resumable stage supervisor: paired instances, pilots, screening and validation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import BinaryIO

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json, free_disk_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.e15_instances import e15_generator_manifest, generate_e15_pair
from geo_llm_scheduler.experiments.e15_worker import validate_e15_result
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.runner import source_digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.utils.rng import RNGManager

REPO = Path(__file__).resolve().parents[1]
SEEDS = (101, 202, 303)
STAGES = ("A_GEN", "A_TIME", "B_GEN", "B_TIME", "C_EVAL", "D_GEN", "D_TIME")
ARMS_A = ("F6", "C2", "S2", "SEQ")
ARMS_B = ("M0", "N0", "M1", "N1")
EXACT_CAP = 20100
MIN_FREE_GB = 30.0


def commit() -> str:
    """Read the frozen source revision for manifest and worker checks."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _base_ids(stage: str) -> tuple[int, ...]:
    return (
        tuple(range(1001, 1013))
        if stage == "A"
        else (tuple(range(2001, 2021)) if stage == "B" else tuple(range(3001, 3013)))
    )


def prepare(root: Path, *, small: bool = False) -> dict:
    """Freeze all disjoint H/T instances and shared initial populations once."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    source = source_digest()
    revision = commit()
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if (frozen["source_hash"], frozen["commit"], frozen["small"]) != (source, revision, small):
            raise ValueError("Frozen E15 source or matrix changed")
        for entry in frozen["instances"].values():
            if file_hash(Path(entry["path"])) != entry["hash"]:
                raise ValueError("Frozen instance changed")
        for entry in frozen["initial"].values():
            if file_hash(Path(entry["path"])) != entry["hash"]:
                raise ValueError("Frozen shared initialization changed")
        return frozen
    instances: dict[str, dict] = {}
    initial: dict[str, dict] = {}
    for stage in ("A", "B", "D", "P50", "P100"):
        ids = (9001,) if stage.startswith("P") else _base_ids(stage)
        if small and not stage.startswith("P"):
            ids = ids[:1]
        jobs = 100 if stage in ("D", "P100") else 50
        for base in ids:
            pair = generate_e15_pair(jobs, base)
            for variant, problem in zip(("H", "T"), pair):
                if stage.startswith("P") and variant == "T":
                    continue
                key = f"{stage}{base}{variant}"
                instance_path = root / "instances" / f"{key}.json"
                instance_path.parent.mkdir(parents=True, exist_ok=True)
                save_instance(problem, instance_path)
                instances[key] = {
                    "path": str(instance_path),
                    "hash": file_hash(instance_path),
                    "pair_id": f"{stage}{base}",
                    "variant": variant,
                    "jobs": jobs,
                    "base_seed": base,
                }
                for seed in SEEDS if not stage.startswith("P") else (101,):
                    initial_path = root / "initial_populations" / f"{key}_a{seed}.json"
                    config = replace(
                        load_config(REPO / "configs" / "default.yaml"),
                        seed=seed,
                        instance=str(instance_path),
                    )
                    genotypes = initial_genotypes(
                        load_instance(instance_path),
                        config,
                        RNGManager(seed).stream("initialization"),
                    )
                    atomic_json(initial_path, [asdict(g) for g in genotypes])
                    initial[f"{key}:{seed}"] = {
                        "path": str(initial_path),
                        "hash": file_hash(initial_path),
                    }
    frozen = {
        "commit": revision,
        "source_hash": source,
        "small": small,
        "generator": e15_generator_manifest(),
        "instances": instances,
        "initial": initial,
        "algorithm_seeds": SEEDS,
        "stage_D_rule": "conservative precommitted four-arm replication",
        "exact_cap": EXACT_CAP,
        "trace_rule": "seed 101 and base seed divisible by 3; all arms/protocols/variants",
        "created_epoch": time.time(),
    }
    atomic_json(path, frozen)
    return frozen


def _config(
    path: str, seed: int, arm: str, protocol: str, pilot: dict, selected: str | None
) -> Config:
    base = load_config(REPO / "configs" / "default.yaml")
    policy = {"F6": "fixed", "C2": "coverage_v2", "S2": "severity_v2", "SEQ": "sequential"}.get(
        arm, "fixed"
    )
    method = (
        "plain"
        if arm == "M0"
        else "nsga2"
        if arm == "N0"
        else ("nsga2_memetic" if arm == "N1" else "full")
    )
    if arm in ("M1", "N1"):
        if selected is None:
            raise ValueError("Stage B/D requires Stage A selection")
        policy = {"C2": "coverage_v2", "S2": "severity_v2", "SEQ": "sequential"}[selected]
    if arm in ARMS_A:
        method = "full"
    jobs = 100 if protocol.startswith("D_") or protocol == "P100" else 50
    seconds = (
        pilot.get("T100_wallclock" if jobs == 100 else "T50_wallclock")
        if protocol.endswith("TIME")
        else None
    )
    generations = 1_000_000 if seconds is not None or protocol == "C_EVAL" else 200
    if protocol.startswith("P"):
        generations = 200
    return replace(
        base,
        instance=path,
        seed=seed,
        method=method,
        controller="qlearning",
        trigger_mode="strict",
        trigger_delta=0.0,
        trigger_quality_gate=True,
        rl_steps=5,
        polish_mode="trajectory",
        budget_policy=policy,
        fixed_budget=6,
        action_mask_policy="none",
        generations=generations,
        seconds=seconds,
        exact_evaluation_cap=EXACT_CAP if protocol == "C_EVAL" else None,
    )


def make_spec(
    root: Path,
    manifest: dict,
    stage: str,
    arm: str,
    base: int,
    variant: str,
    seed: int,
    pilot: dict,
    selected: str | None,
) -> JobSpec:
    """Build one hash-identifiable run with frozen shared initialization."""
    family = stage[0] if stage[0] in "ABD" else "B"
    key = f"{family}{base}{variant}"
    item = manifest["instances"][key]
    start = manifest["initial"][f"{key}:{seed}"]
    config = _config(item["path"], seed, arm, stage, pilot, selected)
    if manifest["small"]:
        config = replace(
            config,
            generations=2 if stage.endswith("GEN") else 10000,
            seconds=0.5 if stage.endswith("TIME") else None,
            exact_evaluation_cap=config.population + 10 if stage == "C_EVAL" else None,
        )
    return JobSpec(
        stage,
        arm,
        base * 10 + int(variant == "T"),
        seed,
        asdict(config),
        item["path"],
        item["hash"],
        start["path"],
        start["hash"],
        manifest["source_hash"],
        manifest["commit"],
    )


def stage_specs(
    root: Path, manifest: dict, stage: str, pilot: dict, selected: str | None
) -> list[JobSpec]:
    """Generate the precommitted complete matrix for one dependency stage."""
    family = "A" if stage.startswith("A") else "D" if stage.startswith("D") else "B"
    ids = _base_ids(family)
    if manifest["small"]:
        ids = ids[:1]
    arms = ARMS_A if family == "A" else ARMS_B
    return [
        make_spec(root, manifest, stage, arm, base, variant, seed, pilot, selected)
        for base in ids
        for variant in ("H", "T")
        for seed in SEEDS
        for arm in arms
    ]


def _memory_fraction() -> float:
    """Read non-reclaimable cgroup working-set fraction, or zero off Linux."""
    limit = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    stat = Path("/sys/fs/cgroup/memory.stat")
    if not (limit.exists() and current.exists() and stat.exists()):
        return 0.0
    maximum = limit.read_text().strip()
    if maximum == "max":
        return 0.0
    fields = dict(line.split() for line in stat.read_text().splitlines())
    working = max(0, int(current.read_text()) - int(fields.get("inactive_file", "0")))
    return working / int(maximum)


def _memory_ok() -> bool:
    """Dispatch only below the frozen 78% cgroup working-set gate."""
    return _memory_fraction() < 0.78


def _quarantine(root: Path, spec: JobSpec) -> None:
    output = root / "runs" / spec.key
    paths = [
        output,
        output.with_suffix(".failure.json"),
        *output.parent.glob(f".{output.name}.attempt-*"),
    ]
    target = root / "quarantine" / f"{spec.key}_{int(time.time() * 1000)}"
    for path in paths:
        if path.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target / path.name))


def _eta(root: Path, stage: str, pending: int, active: dict, workers: int) -> dict:
    completed = []
    by_arm: dict[str, list[float]] = defaultdict(list)
    for file in (root / "runs").glob("*/summary.json"):
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
            completed.append(row["elapsed"])
            by_arm[file.parent.name.split("__")[1]].append(row["elapsed"])
        except (OSError, KeyError, ValueError):
            continue
    typical = median(completed[-96:]) if completed else 180.0
    ordered = sorted(completed[-96:])
    p75 = ordered[int(0.75 * (len(ordered) - 1))] if ordered else typical
    remaining = {"A": 576, "B": 960, "C": 480, "D": 576}
    if (root / "manifest.json").exists() and json.loads((root / "manifest.json").read_text())[
        "small"
    ]:
        remaining = {"A": 48, "B": 48, "C": 24, "D": 48}
    for key, prefix in (("A", "A_"), ("B", "B_"), ("C", "C_"), ("D", "D_")):
        done = sum(1 for p in (root / "runs").glob(f"{prefix}*/*summary.json"))
        remaining[key] = max(0, remaining[key] - done)
    rates = {key: round(n * p75 / max(workers, 1)) for key, n in remaining.items()}
    result = {f"eta_stage_{key}": seconds for key, seconds in rates.items()}
    result.update(
        {
            "eta_mandatory_total_seconds": sum(rates.values()),
            "recent_median_seconds": typical,
            "recent_p75_seconds": p75,
            "per_arm_median_seconds": {k: median(v) for k, v in by_arm.items()},
            "workers": workers,
            "stage": stage,
            "pending_current": pending,
            "running_current": len(active),
            "updated_epoch": time.time(),
        }
    )
    atomic_json(root / "eta.json", result)
    return result


def run_batch(root: Path, specs: list[JobSpec], workers: int, stage: str) -> bool:
    """Resume completed cells, quarantine incomplete cells and retry failures twice."""
    spec_by_key = {s.key: s for s in specs}
    pending = []
    for spec in specs:
        spec_file = root / "specs" / f"{spec.key}.json"
        if spec_file.exists():
            prior = JobSpec(**json.loads(spec_file.read_text(encoding="utf-8")))
            if prior.input_hash != spec.input_hash:
                raise ValueError(f"Frozen spec changed: {spec.key}")
        else:
            atomic_json(spec_file, asdict(spec))
        valid, _ = validate_e15_result(root / "runs" / spec.key, spec)
        if not valid:
            _quarantine(root, spec)
            pending.append(spec)
    active: dict[str, tuple[subprocess.Popen, BinaryIO]] = {}
    retries: dict[str, int] = defaultdict(int)
    failed: dict[str, str] = {}
    last_eta = 0.0
    memory_stalls = 0
    peak_fraction = 0.0
    while pending or active:
        stopping = (root / "stop.request").exists()
        if stopping and not active:
            atomic_json(
                root / "status.json",
                {
                    "stage": "stopped",
                    "pending": len(pending),
                    "running": {},
                    "updated_epoch": time.time(),
                },
            )
            return False
        if free_disk_gb(root) < MIN_FREE_GB:
            atomic_json(
                root / "incident.json",
                {
                    "reason": "disk_guard",
                    "stage": stage,
                    "free_gb": free_disk_gb(root),
                    "epoch": time.time(),
                },
            )
            raise RuntimeError("Mandatory result disk safety line reached")
        peak_fraction = max(peak_fraction, _memory_fraction())
        if pending and not stopping and len(active) < workers and not _memory_ok():
            memory_stalls += 1
        while pending and not stopping and len(active) < workers and _memory_ok():
            spec = pending.pop(0)
            log_path = root / "logs" / f"{spec.key}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream: BinaryIO = log_path.open("ab")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "geo_llm_scheduler.experiments.e15_worker",
                    str(root / "specs" / f"{spec.key}.json"),
                    str(root / "runs" / spec.key),
                ],
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                env={
                    **os.environ,
                    "PYTHONPATH": str(REPO / "src"),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
            )
            active[spec.key] = process, stream
        for key, (process, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del active[key]
            spec = spec_by_key[key]
            valid, reason = validate_e15_result(root / "runs" / key, spec)
            if code != 0 or not valid:
                if retries[key] < 2:
                    retries[key] += 1
                    _quarantine(root, spec)
                    pending.append(spec)
                    time.sleep(min(5 * retries[key], 10))
                else:
                    failed[key] = f"exit={code}; validation={reason}"
        now = time.time()
        atomic_json(
            root / "status.json",
            {
                "stage": stage,
                "updated_epoch": now,
                "supervisor_pid": os.getpid(),
                "pending": len(pending),
                "running": {k: p.pid for k, (p, _) in active.items()},
                "completed": len(specs) - len(pending) - len(active) - len(failed),
                "planned": len(specs),
                "failed": failed,
                "memory_gate_blocked_polls": memory_stalls,
                "peak_working_set_fraction": peak_fraction,
            },
        )
        atomic_json(root / "heartbeat.json", {"epoch": now, "pid": os.getpid(), "stage": stage})
        if now - last_eta > 30:
            _eta(root, stage, len(pending), active, workers)
            last_eta = now
        if pending or active:
            time.sleep(2)
    if failed:
        atomic_json(root / "failure.json", {"stage": stage, "failed": failed})
        return False
    return all(validate_e15_result(root / "runs" / s.key, s)[0] for s in specs)


def _metric_rows(root: Path, specs: list[JobSpec]) -> list[dict]:
    """Compute metrics with one pooled front and scale per instance/protocol."""
    groups: dict[tuple[str, int], list[JobSpec]] = defaultdict(list)
    for spec in specs:
        groups[(spec.stage, spec.instance_seed)].append(spec)
    rows = []
    for (stage, instance), group in groups.items():
        objectives = {}
        for spec in group:
            path = root / "runs" / spec.key
            objectives[spec.key] = [
                tuple(p)
                for p in json.loads((path / "archive_objectives.json").read_text(encoding="utf-8"))
            ]
        pooled = [p for points in objectives.values() for p in points]
        lo = tuple(min(p[i] for p in pooled) for i in range(2))
        hi = tuple(max(p[i] for p in pooled) for i in range(2))

        def norm(point: tuple[float, float]) -> tuple[float, float]:
            return (
                (point[0] - lo[0]) / max(hi[0] - lo[0], 1e-12),
                (point[1] - lo[1]) / max(hi[1] - lo[1], 1e-12),
            )

        ordered_front = sorted({norm(p) for p in pooled})
        front = []
        best_y = float("inf")
        for point in ordered_front:
            if point[1] < best_y:
                front.append(point)
                best_y = point[1]
        for spec in group:
            summary = json.loads((root / "runs" / spec.key / "summary.json").read_text())
            points = [norm(p) for p in objectives[spec.key]]
            rows.append(
                {
                    "stage": stage,
                    "instance": instance,
                    "base": instance // 10,
                    "tariff": "T" if instance % 10 else "H",
                    "seed": spec.algorithm_seed,
                    "arm": spec.arm,
                    "hv": hypervolume(points, (1.1, 1.1)),
                    "igd_plus": igd_plus(points, front),
                    "flow_extreme": min(p[0] for p in objectives[spec.key]),
                    "electricity_extreme": min(p[1] for p in objectives[spec.key]),
                    "elapsed": summary["elapsed"],
                    "exact": summary["counts"]["exact"],
                    "archive_size": summary["archive_size"],
                    "archive_peak_size": summary["archive_peak_size"],
                }
            )
    return rows


def select_adaptive(root: Path, specs: list[JobSpec]) -> str:
    """Select one adaptive using base-clustered paired HV across both protocols."""
    rows = _metric_rows(root, specs)
    atomic_json(root / "analysis" / "stageA_metrics.json", rows)
    by_cell = {(r["stage"], r["base"], r["tariff"], r["seed"], r["arm"]): r for r in rows}
    differences: dict[str, list[float]] = {a: [] for a in ARMS_A[1:]}
    for arm in ARMS_A[1:]:
        for stage in ("A_GEN", "A_TIME"):
            for base in sorted({r["base"] for r in rows}):
                cell_diffs = []
                for tariff in ("H", "T"):
                    for seed in SEEDS:
                        key = stage, base, tariff, seed
                        cell_diffs.append(by_cell[(*key, arm)]["hv"] - by_cell[(*key, "F6")]["hv"])
                differences[arm].append(sum(cell_diffs) / len(cell_diffs))
    scores = {a: median(v) for a, v in differences.items()}
    selected = max(ARMS_A[1:], key=lambda a: (scores[a], a))
    record = {
        "selected": selected,
        "adaptive_beats_fixed": scores[selected] > 0,
        "paired_base_median_hv_difference": scores,
        "rule": "max median base-paired HV difference, stable name tie-break; independent Stage B",
        "complete_cells": len(rows),
        "source_commit": commit(),
    }
    path = root / "selected_adaptive_macrosearch.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Frozen adaptive selection changed")
    atomic_json(path, record)
    return selected


def campaign(root: Path, workers: int, *, small: bool = False) -> None:
    """Run all mandatory E15 stages and final analysis, resuming validated cells."""
    if workers < 1:
        raise ValueError("Positive workers required")
    manifest = prepare(root, small=small)
    pilot_path = root / "resource_plan.json"
    if not pilot_path.exists():
        raise ValueError("Stage 0 pilot must freeze resource plan before launch")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot["source_commit"] != manifest["commit"] or pilot["worker_count"] != workers:
        raise ValueError("Frozen pilot/source/worker mismatch")
    selected: str | None = None
    all_specs: list[JobSpec] = []
    for stage in STAGES:
        if stage == "B_GEN":
            selected = select_adaptive(root, all_specs)
        specs = stage_specs(root, manifest, stage, pilot, selected)
        if not run_batch(root, specs, workers, stage):
            raise RuntimeError(f"Stage {stage} failed; mandatory campaign remains incomplete")
        all_specs.extend(specs)
    from e15_analysis import analyze  # local script module, excluded from solver dependencies

    analyze(root, all_specs)
    if not all(validate_e15_result(root / "runs" / s.key, s)[0] for s in all_specs):
        raise RuntimeError("Final mandatory completeness check failed")
    atomic_json(
        root / "finished.json",
        {
            "complete": True,
            "source_commit": commit(),
            "mandatory_runs": len(all_specs),
            "stage_counts": {stage: sum(s.stage == stage for s in all_specs) for stage in STAGES},
            "finished_epoch": time.time(),
        },
    )
    atomic_json(
        root / "status.json",
        {
            "stage": "complete",
            "complete": True,
            "mandatory_runs": len(all_specs),
            "updated_epoch": time.time(),
        },
    )


def watchdog(root: Path, workers: int, small: bool = False) -> None:
    """Independent restart loop; only a validated complete marker ends supervision."""
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    lock = (root / "watchdog.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
    except BlockingIOError as error:
        raise RuntimeError("Another E15 watchdog already owns this campaign") from error
    atomic_json(root / "watchdog.json", {"pid": os.getpid(), "started_epoch": time.time()})
    while not (root / "stop.request").exists():
        finished = root / "finished.json"
        if finished.exists() and json.loads(finished.read_text()).get("complete"):
            return
        with (root / "logs" / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(REPO / "scripts" / "run_e15_campaign.py"),
                    "run",
                    "--root",
                    str(root),
                    "--workers",
                    str(workers),
                    *(["--small"] if small else []),
                ],
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env={
                    **os.environ,
                    "PYTHONPATH": str(REPO / "src"),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
            )
            atomic_json(
                root / "supervisor.json", {"pid": process.pid, "started_epoch": time.time()}
            )
            code = process.wait()
        if finished.exists() and json.loads(finished.read_text()).get("complete"):
            return
        if (root / "failure.json").exists() or (root / "incident.json").exists():
            atomic_json(root / "watchdog_failure.json", {"exit": code, "epoch": time.time()})
            return
        time.sleep(30)


def main() -> None:
    """CLI: prepare, run, watchdog, status, stop and pilot."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "pilot", "run", "watchdog", "status", "stop")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        print(json.dumps({"instances": len(prepare(root, small=args.small)["instances"])}))
    elif args.command == "pilot":
        from e15_pilot import run_pilot

        run_pilot(root, args.small)
    elif args.command == "run":
        try:
            campaign(root, args.workers, small=args.small)
        except Exception as error:
            if not (root / "stop.request").exists():
                atomic_json(
                    root / "failure.json",
                    {"type": type(error).__name__, "message": str(error), "epoch": time.time()},
                )
            raise
    elif args.command == "watchdog":
        watchdog(root, args.workers, args.small)
    elif args.command == "stop":
        atomic_json(root / "stop.request", {"epoch": time.time()})
    else:
        print(
            (root / "status.json").read_text() if (root / "status.json").exists() else "not started"
        )


if __name__ == "__main__":
    main()
