"""Resumable 200-generation campaign for paired baselines, action masks and budgets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash, validate_result
from geo_llm_scheduler.experiments.generation20_instances import generate_generation20_instance
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.runner import source_digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.utils.rng import RNGManager

REPO = Path(__file__).resolve().parents[1]
DEV_INSTANCES = (81, 82, 83)
VALID_INSTANCES = tuple(range(111, 121))
DEV_SEEDS = (101, 202, 303)
VALID_SEEDS = (101, 202, 303, 404, 505)
MASKS = {"M0": "none", "M1": "no_a6", "M2": "no_a4a5", "M3": "no_a4a5a6"}
CUTOFFS = {"B1": (1.0, 2.0), "B2": (0.75, 1.5), "B3": (0.5, 1.25)}


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _base(instance_path: Path, seed: int, arm: str) -> Config:
    config = load_config(REPO / "configs" / "default.yaml")
    method = "plain" if arm == "V1" else "nsga2" if arm == "V2" else "full"
    return replace(
        config,
        instance=str(instance_path),
        output="outputs",
        seed=seed,
        generations=200,
        seconds=None,
        method=method,
        controller="qlearning",
        trigger_mode="strict",
        trigger_delta=0.0,
        trigger_quality_gate=True,
        rl_steps=5,
        polish_mode="trajectory",
        budget_policy="fixed",
        fixed_budget=6,
        action_mask_policy="none",
    )


def _configured(base: Config, arm: str, selected: dict[str, str] | None = None) -> Config:
    if arm in MASKS:
        return replace(base, action_mask_policy=MASKS[arm])
    if arm in CUTOFFS:
        return replace(base, budget_policy="severity", severity_budget_cutoffs=CUTOFFS[arm])
    if arm in ("V1", "V2", "V3"):
        return base
    if selected is None:
        raise ValueError("Validation treatment requires frozen development selections")
    mask = MASKS[selected["mask"]]
    cutoffs = CUTOFFS[selected["budget"]]
    if arm == "V4":
        return replace(base, action_mask_policy=mask)
    if arm == "V5":
        return replace(base, budget_policy="severity", severity_budget_cutoffs=cutoffs)
    if arm == "V6":
        return replace(
            base, action_mask_policy=mask, budget_policy="severity", severity_budget_cutoffs=cutoffs
        )
    raise ValueError(f"Unknown arm {arm}")


def prepare(root: Path) -> dict:
    """Freeze source, instances and paired initial populations once for the campaign."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    source, commit = source_digest(), _commit()
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest["source_hash"], manifest["commit"]) != (source, commit):
            raise ValueError("Campaign source changed; refusing mixed-version resume")
        for item in manifest["instances"].values():
            if file_hash(Path(item["path"])) != item["hash"]:
                raise ValueError("Frozen instance changed")
        for item in manifest["initial"].values():
            if file_hash(Path(item["path"])) != item["hash"]:
                raise ValueError("Frozen initial population changed")
        return manifest
    instances = {}
    initial = {}
    for seed in DEV_INSTANCES + VALID_INSTANCES:
        kind = "heterogeneous_prices" if seed >= 116 else "mixed"
        problem = generate_generation20_instance(seed, kind == "heterogeneous_prices")
        instance_path = root / "instances" / f"{kind}_i{seed}.json"
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        save_instance(problem, instance_path)
        instances[str(seed)] = {
            "path": str(instance_path),
            "hash": file_hash(instance_path),
            "kind": kind,
        }
        for algorithm_seed in DEV_SEEDS if seed in DEV_INSTANCES else VALID_SEEDS:
            output = root / "initial" / f"i{seed}_a{algorithm_seed}.json"
            config = _base(instance_path, algorithm_seed, "V3")
            genotypes = initial_genotypes(
                load_instance(instance_path),
                config,
                RNGManager(algorithm_seed).stream("initialization"),
            )
            atomic_json(output, [asdict(g) for g in genotypes])
            initial[f"{seed}:{algorithm_seed}"] = {"path": str(output), "hash": file_hash(output)}
    manifest = {
        "commit": commit,
        "source_hash": source,
        "generations": 200,
        "instances": instances,
        "initial": initial,
        "development": {
            "instances": DEV_INSTANCES,
            "seeds": DEV_SEEDS,
            "arms": tuple(MASKS) + tuple(CUTOFFS),
        },
        "validation": {
            "instances": VALID_INSTANCES,
            "seeds": VALID_SEEDS,
            "arms": tuple(f"V{i}" for i in range(1, 7)),
        },
        "heterogeneous_convention": {
            "hardware": "same three instances per region; no heterogeneous hardware",
            "region_price_multipliers": (0.8, 1.0, 1.2),
            "region_price_segment_rotations": (0, 1, 2),
            "region_demand_cny_kw": (4.0, 5.0, 6.0),
            "mixed_generator": "unchanged 50-job D02 convention; seeds 111..120",
        },
    }
    atomic_json(path, manifest)
    return manifest


def make_spec(
    root: Path,
    manifest: dict,
    stage: str,
    arm: str,
    instance: int,
    seed: int,
    selected: dict[str, str] | None = None,
) -> JobSpec:
    """Create a hash-identifiable job sharing initial genotypes within its paired cell."""
    item = manifest["instances"][str(instance)]
    start = manifest["initial"][f"{instance}:{seed}"]
    config = _configured(_base(Path(item["path"]), seed, arm), arm, selected)
    return JobSpec(
        stage,
        arm,
        instance,
        seed,
        asdict(config),
        item["path"],
        item["hash"],
        start["path"],
        start["hash"],
        manifest["source_hash"],
        manifest["commit"],
    )


def _spec_file(root: Path, spec: JobSpec) -> Path:
    path = root / "specs" / f"{spec.key}.json"
    if path.exists():
        prior = JobSpec(**json.loads(path.read_text(encoding="utf-8")))
        if prior.input_hash != spec.input_hash:
            raise ValueError(f"Changed frozen job: {spec.key}")
    else:
        atomic_json(path, asdict(spec))
    return path


def _quarantine(root: Path, spec: JobSpec) -> None:
    output = root / "runs" / spec.key
    failure = output.with_suffix(".failure.json")
    attempts = list(output.parent.glob(f".{output.name}.attempt-*"))
    if not output.exists() and not failure.exists() and not attempts:
        return
    target = root / "quarantine" / f"{spec.key}_{int(time.time() * 1000)}"
    target.mkdir(parents=True)
    if output.exists():
        shutil.move(str(output), str(target / output.name))
    if failure.exists():
        shutil.move(str(failure), str(target / failure.name))
    for attempt in attempts:
        shutil.move(str(attempt), str(target / attempt.name))


def _memory_ok() -> bool:
    limit_path = Path("/sys/fs/cgroup/memory.max")
    used_path = Path("/sys/fs/cgroup/memory.current")
    if not limit_path.exists() or not used_path.exists():
        return True
    limit = limit_path.read_text().strip()
    if limit == "max":
        return True
    # Large trace writes remain in reclaimable page cache. Use the cgroup's
    # non-reclaimable footprint for dispatch rather than stalling on cached files.
    stat_path = Path("/sys/fs/cgroup/memory.stat")
    stat = dict(line.split() for line in stat_path.read_text().splitlines())
    inactive_file = int(stat.get("inactive_file", "0"))
    working_set = max(0, int(used_path.read_text()) - inactive_file)
    return working_set < int(limit) * 0.78


def _status(
    root: Path, stage: str, pending: int, running: dict, succeeded: int, failed: dict
) -> None:
    atomic_json(
        root / "status.json",
        {
            "updated_epoch": time.time(),
            "pid": os.getpid(),
            "stage": stage,
            "pending": pending,
            "running": {k: p.pid for k, (p, _) in running.items()},
            "succeeded_this_session": succeeded,
            "failed": failed,
            "deadline_epoch": json.loads((root / "launch.json").read_text())["deadline_epoch"],
        },
    )


def run_batch(root: Path, specs: list[JobSpec], workers: int, stage: str) -> bool:
    """Run independent workers with complete-artifact resume and bounded retries."""
    pending = []
    for spec in specs:
        _spec_file(root, spec)
        output = root / "runs" / spec.key
        valid, reason = validate_result(output, spec)
        if valid:
            continue
        if output.exists() or output.with_suffix(".failure.json").exists():
            _quarantine(root, spec)
        pending.append(spec)
    active: dict[str, tuple[subprocess.Popen, object]] = {}
    failures: dict[str, str] = {}
    retries: dict[str, int] = {}
    success = 0
    deadline = json.loads((root / "launch.json").read_text())["deadline_epoch"]
    while pending or active:
        while (
            pending
            and len(active) < workers
            and time.time() < deadline
            and _memory_ok()
            and not (root / "stop.request").exists()
        ):
            spec = pending.pop(0)
            output = root / "runs" / spec.key
            log = root / "logs" / f"{spec.key}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("ab")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "geo_llm_scheduler.experiments.campaign.worker",
                    str(root / "specs" / f"{spec.key}.json"),
                    str(output),
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
            active[spec.key] = (process, stream)
        for key, (process, stream) in list(active.items()):
            result = process.poll()
            if result is None:
                continue
            stream.close()
            del active[key]
            spec = next(s for s in specs if s.key == key)
            valid, reason = validate_result(root / "runs" / key, spec)
            if result == 0 and valid:
                success += 1
            elif retries.get(key, 0) < 2 and time.time() < deadline:
                retries[key] = retries.get(key, 0) + 1
                _quarantine(root, spec)
                pending.append(spec)
            else:
                failures[key] = f"exit={result}; {reason}"
        _status(root, stage, len(pending), active, success, failures)
        if pending or active:
            time.sleep(2)
        if pending and not active and (time.time() >= deadline or (root / "stop.request").exists()):
            break
    return not pending and not failures


def _points(root: Path, spec: JobSpec) -> list[tuple[float, float]]:
    with (root / "runs" / spec.key / "objectives.csv").open(encoding="utf-8", newline="") as handle:
        return [(float(r["flow_seconds"]), float(r["bill_cny"])) for r in csv.DictReader(handle)]


def metric_rows(root: Path, specs: list[JobSpec], scope: str) -> list[dict]:
    """Evaluate complete cells using one pooled objective scale per instance."""
    rows = []
    for instance in sorted({s.instance_seed for s in specs}):
        group = [s for s in specs if s.instance_seed == instance]
        if not all(validate_result(root / "runs" / s.key, s)[0] for s in group):
            continue
        points = {s.key: _points(root, s) for s in group}
        all_points = [p for arr in points.values() for p in arr]
        ideal = [min(p[k] for p in all_points) for k in (0, 1)]
        scale = [max(max(p[k] for p in all_points) - ideal[k], 1e-12) for k in (0, 1)]
        normalized = {
            key: [((p[0] - ideal[0]) / scale[0], (p[1] - ideal[1]) / scale[1]) for p in arr]
            for key, arr in points.items()
        }
        front = []
        best_bill = float("inf")
        for p in sorted(set(p for arr in normalized.values() for p in arr)):
            if p[1] < best_bill:
                front.append(p)
                best_bill = p[1]
        for spec in group:
            summary = json.loads((root / "runs" / spec.key / "summary.json").read_text())
            arr = normalized[spec.key]
            rows.append(
                {
                    "scope": scope,
                    "key": spec.key,
                    "instance": instance,
                    "algorithm_seed": spec.algorithm_seed,
                    "arm": spec.arm,
                    "hv": hypervolume(arr, (1.1, 1.1)),
                    "igd_plus": igd_plus(arr, front),
                    "elapsed": summary["elapsed"],
                    "exact": summary["counts"].get("exact", 0),
                    "archive_size": summary["archive_size"],
                }
            )
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select(root: Path, rows: list[dict]) -> dict[str, str]:
    """Choose one nonbaseline treatment per factor by equal-instance mean HV."""

    def score(arm: str) -> tuple[float, float]:
        by_instance = []
        for instance in DEV_INSTANCES:
            matching = [r for r in rows if r["instance"] == instance and r["arm"] == arm]
            if len(matching) != len(DEV_SEEDS):
                raise ValueError(f"Incomplete development arm {arm}, instance {instance}")
            by_instance.append(
                (
                    sum(r["hv"] for r in matching) / len(matching),
                    sum(r["igd_plus"] for r in matching) / len(matching),
                )
            )
        return (
            sum(v[0] for v in by_instance) / len(by_instance),
            -sum(v[1] for v in by_instance) / len(by_instance),
        )

    mask = max(("M1", "M2", "M3"), key=lambda a: (*score(a), a))
    budget = max(CUTOFFS, key=lambda a: (*score(a), a))
    selected = {
        "mask": mask,
        "budget": budget,
        "selection": "development only",
        "baseline_hv": score("M0")[0],
        "mask_hv": score(mask)[0],
        "budget_hv": score(budget)[0],
    }
    path = root / "selected.json"
    if path.exists() and json.loads(path.read_text()) != selected:
        raise ValueError("Frozen development selection changed")
    atomic_json(path, selected)
    return selected


def campaign(root: Path, workers: int, hours: float) -> None:
    """Run the 63 development and 300 validation cells, preserving resume state."""
    if workers < 1 or hours <= 0:
        raise ValueError("Positive workers and hours required")
    lock = root / "pipeline.pid"
    if lock.exists():
        prior = int(lock.read_text())
        if Path(f"/proc/{prior}").exists() and prior != os.getpid():
            raise RuntimeError(f"Campaign already running with PID {prior}")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    try:
        launch = root / "launch.json"
        if not launch.exists():
            atomic_json(
                launch,
                {
                    "start_epoch": time.time(),
                    "deadline_epoch": time.time() + hours * 3600,
                    "requested_workers": workers,
                    "hours": hours,
                },
            )
        manifest = prepare(root)
        dev = [
            make_spec(root, manifest, "development", arm, i, a)
            for i in DEV_INSTANCES
            for a in DEV_SEEDS
            for arm in (*MASKS, *CUTOFFS)
        ]
        baseline = [
            make_spec(root, manifest, "validation", arm, i, a)
            for i in VALID_INSTANCES
            for a in VALID_SEEDS
            for arm in ("V1", "V2", "V3")
        ]
        # Interleave independent baselines while screening proceeds.
        queue = []
        for offset in range(max(len(dev), len(baseline))):
            if offset < len(dev):
                queue.append(dev[offset])
            if offset < len(baseline):
                queue.append(baseline[offset])
        if not run_batch(root, queue, workers, "development_and_baselines"):
            return
        dev_rows = metric_rows(root, dev, "development")
        _write_rows(root / "analysis" / "development_metrics.csv", dev_rows)
        selected = select(root, dev_rows)
        treatment = [
            make_spec(root, manifest, "validation", arm, i, a, selected)
            for i in VALID_INSTANCES
            for a in VALID_SEEDS
            for arm in ("V4", "V5", "V6")
        ]
        if not run_batch(root, treatment, workers, "validation_treatments"):
            return
        validation = metric_rows(root, baseline + treatment, "validation")
        _write_rows(root / "analysis" / "validation_metrics.csv", validation)
        atomic_json(
            root / "finished.json",
            {
                "finished_epoch": time.time(),
                "complete": len(dev) == len(dev_rows) and len(validation) == 300,
                "development": len(dev_rows),
                "validation": len(validation),
                "selected": selected,
                "commit": manifest["commit"],
            },
        )
        _status(root, "complete", 0, {}, len(dev_rows) + len(validation), {})
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


def main() -> None:
    """CLI entry point for the detached server pipeline or a read-only status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status", "prepare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hours", type=float, default=20)
    args = parser.parse_args()
    if args.command == "run":
        campaign(args.root, args.workers, args.hours)
    elif args.command == "prepare":
        print(json.dumps(prepare(args.root)["development"], ensure_ascii=False))
    else:
        path = args.root / "status.json"
        print(path.read_text(encoding="utf-8") if path.exists() else "not started")


if __name__ == "__main__":
    main()
