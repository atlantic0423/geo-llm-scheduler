"""Frozen new-instance, actual-structure D1 campaign with local safe queues."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import (
    alive,
    atomic_json,
    available_memory_gb,
    free_disk_gb,
    keep_awake,
    process_peak_rss_gb,
)
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.campaign.worker import TUPLE_FIELDS
from geo_llm_scheduler.experiments.d1 import extract_intent
from geo_llm_scheduler.experiments.d1_campaign import campaign_lease
from geo_llm_scheduler.experiments.d1_realpath_observation import (
    WINDOWS,
    RealPathObserver,
    allocate_routes,
    retained_gains,
)
from geo_llm_scheduler.experiments.d1_routing import GROUPS, routing_probe
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.p2_worker import campaign_path, canonical_source_hash
from geo_llm_scheduler.experiments.runner import digest, git_metadata, solution
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.numeric import TOL, close
from geo_llm_scheduler.utils.rng import RNGManager

MASTER_SEED = 2026100403
FORMAL_BASES = {50: tuple(range(830001, 830009)), 100: tuple(range(840001, 840009))}
SEEDS = (1101, 2202, 3303)
MODULE = "geo_llm_scheduler.experiments.d1_realpath"


def frozen_manifest(root: Path) -> dict:
    """Read immutable identity, refusing source, protocol or input drift."""
    expected = json.loads((root / "manifest.sha256.json").read_text(encoding="utf-8"))
    if file_hash(root / "manifest.json") != expected["manifest.json"]:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if canonical_source_hash() != manifest["source_hash"]:
        raise ValueError("Frozen runtime source changed")
    for name, sha in manifest["input_files"].items():
        if file_hash(campaign_path(root, name)) != sha:
            raise ValueError(f"Frozen input changed: {name}")
    return manifest


def prepare_realpath(root: Path, repository: Path, pilot_seconds: float | None = None) -> dict:
    """Freeze sixteen unseen bases, paired tariffs and three seeds; pilot uses old bases."""
    provenance = git_metadata(repository)
    if not provenance["git_commit"] or provenance["git_dirty"]:
        raise ValueError("Freeze a clean source commit first")
    if root.exists():
        raise FileExistsError("Existing campaigns are never overwritten")
    if pilot_seconds is not None and pilot_seconds <= 0:
        raise ValueError("Positive pilot duration required")
    for folder in ("inputs", "specs", "samples", "probes", "ops", "logs", "analysis"):
        (root / folder).mkdir(parents=True)
    config = load_config(repository / "configs/p2_local/F6.yaml")
    source_hash = canonical_source_hash()
    keys = []
    worker_seconds = 0.0
    for n in (50, 100):
        bases = (
            ((810001,) if n == 50 else (820001,)) if pilot_seconds is not None else FORMAL_BASES[n]
        )
        for base in bases:
            pair = generate_e15_pair(n, base)
            for tariff, problem in zip(("H", "T"), pair):
                save_instance(problem, root / "inputs" / f"n{n}_i{base}_{tariff}.json")
            for seed in (6606, 7707) if pilot_seconds is not None else SEEDS:
                initial = f"inputs/n{n}_i{base}_a{seed}_initial.json"
                atomic_json(
                    root / initial,
                    [
                        asdict(g)
                        for g in initial_genotypes(
                            pair[0],
                            replace(config, seed=seed),
                            RNGManager(seed).stream("initialization"),
                        )
                    ],
                )
                for tariff in ("H", "T"):
                    key = f"n{n}_i{base}_{tariff}_a{seed}"
                    instance = f"inputs/n{n}_i{base}_{tariff}.json"
                    seconds = (pilot_seconds if pilot_seconds is not None else 600.0) * (
                        1 if n == 50 else 2
                    )
                    spec = {
                        "key": key,
                        "jobs": n,
                        "base_seed": base,
                        "tariff": tariff,
                        "algorithm_seed": seed,
                        "instance": instance,
                        "initial": initial,
                        "instance_hash": file_hash(root / instance),
                        "initial_hash": file_hash(root / initial),
                        "source_commit": provenance["git_commit"],
                        "source_hash": source_hash,
                        "config": asdict(
                            replace(
                                config,
                                seed=seed,
                                seconds=seconds,
                                generations=1_000_000,
                                instance=instance,
                                output=f"samples/{key}",
                            )
                        ),
                        "quota_seconds": 1.0 if n == 50 else 3.0,
                    }
                    atomic_json(root / "specs" / f"{key}.json", spec)
                    keys.append(key)
                    worker_seconds += seconds
    RNGManager(MASTER_SEED).stream("D1R:source_order").shuffle(keys)
    manifest = {
        "schema": 1,
        "kind": "D1_realpath_v1",
        "pilot": pilot_seconds is not None,
        "source_commit": provenance["git_commit"],
        "host": platform.node(),
        "source_hash": source_hash,
        "master_seed": MASTER_SEED,
        "keys": keys,
        "bases": sorted(
            {json.loads((root / "specs" / f"{k}.json").read_text())["base_seed"] for k in keys}
        ),
        "groups": GROUPS,
        "windows": WINDOWS,
        "reservoir_cap_per_phase_preference": 16,
        "source_workers": 6,
        "probe_workers": 2,
        "source_worker_hours": worker_seconds / 3600,
        "max_cases": len(keys) * 3 * 3 * 16,
        "scope": "New-base real structural calls; offline mechanism confirmation, not online MOEA/D benefit",
        "source_rule": "variation uses first sampled parent a; structural A1-A6 use fixed step incumbent",
        "random_pool": "within run/phase/caller-preference; positive waits and changed genotype only; match exact gate count",
        "normalization": "own-arm candidate-updated ideal for actual selection; common union-updated ideal for retained-output comparison; frozen comparison sensitivity",
        "aggregation": "population-event weights within phase; equal three phases; equal six tariff/seed runs per base; independent bases equal",
        "protocol_sha256": file_hash(repository / "docs/experiments/d1_realpath.md"),
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for folder in ("inputs", "specs")
            for p in sorted((root / folder).glob("*.json"))
        },
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"manifest.json": file_hash(root / "manifest.json")})
    return manifest


def validate_realpath_job(root: Path, role: str, key: str) -> tuple[bool, str]:
    """Verify complete identity and exact artifact hashes before reuse."""
    try:
        if role not in ("samples", "probes"):
            return False, "unknown role"
        spec = json.loads(campaign_path(root, f"specs/{key}.json").read_text(encoding="utf-8"))
        output = campaign_path(root, f"{role}/{key}")
        marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if marker["spec_hash"] != digest(spec) or set(marker["files"]) != {
            "data.json",
            "summary.json",
        }:
            return False, "identity mismatch"
        if any(file_hash(output / n) != h for n, h in marker["files"].items()):
            return False, "artifact hash mismatch"
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if (
            summary["source_commit"] != spec["source_commit"]
            or summary["source_hash"] != spec["source_hash"]
        ):
            return False, "source mismatch"
        if role == "samples" and (
            summary["termination_reason"] != "time_budget"
            or summary["engine_seconds"] < spec["config"]["seconds"]
        ):
            return False, "incomplete source budget"
        if role == "probes" and summary["sample_sha256"] != file_hash(
            root / "samples" / key / "data.json"
        ):
            return False, "source sample changed"
        return True, "complete"
    except (OSError, ValueError, KeyError, TypeError):
        return False, "missing or unreadable evidence"


def audit_case(problem: ProblemInstance, row: dict) -> int:
    """Independently re-decode base and re-evaluate every recorded output/proposal."""
    target = Genotype(tuple(row["target"]["ms"]), tuple(row["target"]["os"]))
    rebuilt = decode(problem, target)
    if rebuilt.starts != tuple(row["base_starts"]):
        raise ValueError("Base SSGS replay mismatch")
    checks = 0
    records = [
        {"starts": row["base_starts"], "objectives": row["base_objectives"], "feasible": True}
    ]
    for arm in row["groups"].values():
        records.append({"starts": arm["starts"], "objectives": arm["objectives"], "feasible": True})
        records.extend(arm["proposals"])
    for item in records:
        evaluation = evaluate(problem, target, Schedule(tuple(item["starts"])))
        if evaluation.feasible != item["feasible"] or any(
            not close(a, b, TOL.time if m == 0 else TOL.cost)
            for m, (a, b) in enumerate(zip(evaluation.objectives, item["objectives"]))
        ):
            raise ValueError("Independent exact mismatch")
        checks += 1
    return checks


def execute_realpath_job(root: Path, role: str, key: str) -> dict:
    """Run one complete source or paired real-target block, with bounded artifacts."""
    manifest = frozen_manifest(root)
    if role not in ("samples", "probes") or key not in manifest["keys"]:
        raise ValueError("Unknown frozen job")
    provenance = git_metadata()
    if provenance["git_dirty"] or provenance["git_commit"] != manifest["source_commit"]:
        raise ValueError("Worker requires frozen clean commit")
    if manifest["host"] != platform.node():
        raise ValueError(
            "Host migration requires a fresh whole paired campaign; preserve old evidence"
        )
    spec = json.loads((root / "specs" / f"{key}.json").read_text(encoding="utf-8"))
    output = root / role / key
    temporary = root / role / f".{key}.attempt-{os.getpid()}"
    if output.exists() or temporary.exists():
        raise FileExistsError("Existing evidence must be preserved and reviewed")
    temporary.mkdir()
    problem = load_instance(root / spec["instance"])
    begun = time.perf_counter()
    last_beat = -10.0

    def beat(extra: dict) -> None:
        atomic_json(
            root / "ops/heartbeats" / f"{role}_{key}.json",
            {
                "pid": os.getpid(),
                "timestamp": time.time(),
                "role": role,
                "key": key,
                "rss_peak_gib": process_peak_rss_gb(),
                **extra,
            },
        )

    beat({"state": "starting"})
    summary = {
        "key": key,
        "role": role,
        "jobs": spec["jobs"],
        "base_seed": spec["base_seed"],
        "tariff": spec["tariff"],
        "algorithm_seed": spec["algorithm_seed"],
        "source_commit": spec["source_commit"],
        "source_hash": spec["source_hash"],
        "host": platform.node(),
    }
    if role == "samples":
        values = dict(spec["config"])
        for field in TUPLE_FIELDS:
            values[field] = tuple(values[field])
        config = Config(**values)
        assert config.seconds is not None
        initial = [
            Genotype(tuple(g["ms"]), tuple(g["os"]))
            for g in json.loads((root / spec["initial"]).read_text(encoding="utf-8"))
        ]
        observer = RealPathObserver(
            problem,
            float(config.seconds),
            config.seed,
            manifest["reservoir_cap_per_phase_preference"],
        )

        def observe(event: str, payload: dict) -> None:
            nonlocal last_beat
            observer(event, payload)
            if event == "offspring" and payload["elapsed"] - last_beat >= 10:
                beat(
                    {
                        "state": "sampling",
                        "elapsed": payload["elapsed"],
                        "events": observer.events,
                        "offspring": payload["processed"],
                        "exact": payload["gateway"].counts["exact"],
                    }
                )
                last_beat = payload["elapsed"]

        result = run(problem, config, initial, retain_trace=False, observer=observe)
        audit_start = time.perf_counter()
        for candidate in result.population + result.archive.members:
            if evaluate(problem, candidate.genotype, candidate.schedule) != candidate.evaluation:
                raise ValueError("Final F6 exact mismatch")
        data = observer.export()
        data["population"] = [solution(c) for c in result.population]
        data["archive"] = [solution(c) for c in result.archive.members]
        summary.update(
            engine_seconds=result.elapsed,
            termination_reason=result.termination_reason,
            offspring=result.offspring_count,
            exact=result.gateway.counts["exact"],
            cases=len(data["rows"]),
            natural_counts=data["counts"],
            observer_seconds=result.observer_seconds,
            reservoir_seconds=observer.observer_seconds,
            audit_seconds=time.perf_counter() - audit_start,
        )
    else:
        if not validate_realpath_job(root, "samples", key)[0]:
            raise ValueError("Incomplete source evidence")
        sample_path = root / "samples" / key / "data.json"
        raw = json.loads(sample_path.read_text(encoding="utf-8"))
        sources = [
            {
                **r,
                "source": restore_candidate(r["source"]),
                "target": Genotype(tuple(r["target"]["ms"]), tuple(r["target"]["os"])),
            }
            for r in raw["rows"]
        ]
        tick = time.perf_counter()
        intents = [extract_intent(problem, r["source"]) for r in sources]
        allocation = allocate_routes(
            sources, intents, RNGManager(MASTER_SEED).stream(f"D1R:allocation:{key}")
        )
        planning_seconds = time.perf_counter() - tick
        rows = []
        audit_checks = 0
        audit_seconds = 0.0
        replay_seconds = 0.0
        replays = []
        replayed_stages: set[int] = set()
        for i, source in enumerate(sources):
            context = NormalizationContext(
                tuple(source["context"]["ideal"]), tuple(source["context"]["maximum"])
            )
            seed = (
                RNGManager(MASTER_SEED).stream(f"D1R:probe:{key}:{source['event']}").getrandbits(32)
            )
            row = routing_probe(
                problem,
                source["source"],
                source["target"],
                tuple(source["weight"]),
                context,
                seed,
                allocation["routes"][i],
                spec["quota_seconds"],
                allow_unchanged=True,
                normalization="updated",
            )
            if (row["groups"]["CONDITIONAL"]["route"] == "TRUE") != source["natural_gate"]:
                raise ValueError("Natural gate replay mismatch")
            row.update(
                {
                    k: source[k]
                    for k in (
                        "event",
                        "stage",
                        "preference",
                        "subproblem",
                        "generation",
                        "path",
                        "step",
                        "parents",
                        "bucket_events",
                        "bucket_sampled",
                        "probability",
                    )
                }
            )
            row["retained_gains"] = retained_gains(row)
            row["frozen_retained_gains"] = retained_gains(row, updated=False)
            for m, metric in enumerate(("flow", "bill")):
                row[f"{metric}_gains"] = {
                    g: (row["base_objectives"][m] - arm["objectives"][m])
                    / max(row["base_objectives"][m], 1e-12)
                    for g, arm in row["groups"].items()
                }
            tick = time.perf_counter()
            audit_checks += audit_case(problem, row)
            audit_seconds += time.perf_counter() - tick
            if source["stage"] not in replayed_stages:
                replayed_stages.add(source["stage"])
                tick = time.perf_counter()
                replay = routing_probe(
                    problem,
                    source["source"],
                    source["target"],
                    tuple(source["weight"]),
                    context,
                    seed,
                    allocation["routes"][i],
                    spec["quota_seconds"],
                    allow_unchanged=True,
                    normalization="updated",
                )
                fields = (
                    "route",
                    "starts",
                    "objectives",
                    "attempts",
                    "exact",
                    "failed_projection",
                    "duplicates",
                    "on_time_exact",
                    "selection_context",
                )
                match = all(
                    all(row["groups"][g][f] == replay["groups"][g][f] for f in fields)
                    for g in GROUPS
                )
                binding = any(
                    a["overshoot_seconds"] > 0
                    for result in (row, replay)
                    for a in result["groups"].values()
                )
                if not match and not binding:
                    raise ValueError("Non-time policy replay mismatch")
                replay_seconds += time.perf_counter() - tick
                replays.append(
                    {
                        "stage": source["stage"],
                        "event": source["event"],
                        "match": match,
                        "budget_binding": binding,
                        "exact": sum(a["exact"] for a in replay["groups"].values()),
                    }
                )
            rows.append(row)
            if time.perf_counter() - begun - last_beat >= 10:
                beat(
                    {
                        "state": "probing",
                        "elapsed": time.perf_counter() - begun,
                        "cases": len(rows),
                        "total_cases": len(sources),
                    }
                )
                last_beat = time.perf_counter() - begun
        data = {
            "rows": rows,
            "allocation": allocation["blocks"],
            "source_counts": raw["counts"],
            "buckets": raw["buckets"],
            "replays": replays,
        }
        summary.update(
            cases=len(rows),
            planning_seconds=planning_seconds,
            audit_seconds=audit_seconds,
            audit_checks=audit_checks,
            sample_sha256=file_hash(sample_path),
            true_counts={g: sum(r["groups"][g]["route"] == "TRUE" for r in rows) for g in GROUPS},
            replay_seconds=replay_seconds,
            replayed_cases=len(replays),
            replay_budget_binding=sum(r["budget_binding"] for r in replays),
        )
        if summary["true_counts"]["CONDITIONAL"] != summary["true_counts"]["RANDOM"]:
            raise ValueError("Mismatched route frequency")
    summary.update(elapsed=time.perf_counter() - begun, rss_peak_gib=process_peak_rss_gb())
    atomic_json(temporary / "data.json", data)
    atomic_json(temporary / "summary.json", summary)
    atomic_json(
        temporary / "complete.json",
        {
            "spec_hash": digest(spec),
            "files": {n: file_hash(temporary / n) for n in ("data.json", "summary.json")},
        },
    )
    temporary.rename(output)
    valid, reason = validate_realpath_job(root, role, key)
    if not valid:
        raise ValueError(reason)
    beat({"state": "complete", "elapsed": summary["elapsed"]})
    return summary


def run_realpath(root: Path, source_workers: int = 6, probe_workers: int = 2) -> dict:
    """Drain on stop/failure; never auto-retry or mix damaged paired evidence."""
    if (
        source_workers < 1
        or probe_workers < 1
        or source_workers > 6
        or probe_workers > 2
        or source_workers + probe_workers > (os.cpu_count() or 1)
    ):
        raise ValueError("Invalid frozen worker allocation")
    manifest = frozen_manifest(root)
    with campaign_lease(root / "ops/supervisor.lock"):
        status_path = root / "ops/status.json"
        if status_path.exists() and any(
            alive(x["pid"]) for x in json.loads(status_path.read_text())["active"]
        ):
            raise RuntimeError("Prior workers still alive")
        if list((root / "ops/failures").glob("*.json")) or any(
            any((root / r).glob(".*.attempt-*")) for r in ("samples", "probes")
        ):
            raise ValueError("Unfinished failure evidence requires explicit review")
        done: dict[str, set[str]] = {r: set() for r in ("samples", "probes")}
        for role in done:
            for key in manifest["keys"]:
                if (root / role / key).exists():
                    valid, reason = validate_realpath_job(root, role, key)
                    if not valid:
                        raise ValueError(reason)
                    done[role].add(key)
        started = time.time()
        atomic_json(
            root / "ops" / f"epoch_{time.time_ns()}.json",
            {
                "pid": os.getpid(),
                "host": platform.node(),
                "started": started,
                "source_workers": source_workers,
                "probe_workers": probe_workers,
                "manifest_sha256": file_hash(root / "manifest.json"),
            },
        )
        active: list[dict] = []
        failed = False
        status = {"pid": os.getpid(), "active": []}
        keep_awake(True)
        try:
            while any(len(done[r]) != len(manifest["keys"]) for r in done):
                for item in list(active):
                    code = item["process"].poll()
                    if code is None:
                        if time.time() - item["started"] > item["deadline_seconds"]:
                            if sys.platform == "win32":
                                subprocess.run(
                                    ["taskkill", "/PID", str(item["pid"]), "/T", "/F"],
                                    check=False,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )
                            else:
                                item["process"].terminate()
                            item["process"].wait(timeout=10)
                            failed = True
                            atomic_json(
                                root / "ops/failures" / f"{item['role']}_{item['key']}.json",
                                {"reason": "owned worker hard timeout", "pid": item["pid"]},
                            )
                        continue
                    active.remove(item)
                    valid, reason = validate_realpath_job(root, item["role"], item["key"])
                    if code != 0 or not valid:
                        failed = True
                        atomic_json(
                            root / "ops/failures" / f"{item['role']}_{item['key']}.json",
                            {"exit_code": code, "reason": reason},
                        )
                    else:
                        done[item["role"]].add(item["key"])
                stopping = failed or (root / "stop.request").exists()
                memory, disk = available_memory_gb(), free_disk_gb(root)
                guarded = memory < 4 or disk < 10
                if not stopping and not guarded:
                    for role, cap in (("samples", source_workers), ("probes", probe_workers)):
                        running = {x["key"] for x in active if x["role"] == role}
                        for key in manifest["keys"]:
                            if len(running) >= cap:
                                break
                            if (
                                key in done[role]
                                or key in running
                                or (role == "probes" and key not in done["samples"])
                            ):
                                continue
                            spec = json.loads(
                                (root / "specs" / f"{key}.json").read_text(encoding="utf-8")
                            )
                            flags = (
                                (
                                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                    | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                                )
                                if sys.platform == "win32"
                                else 0
                            )
                            with (root / "logs" / f"{role}_{key}.log").open("ab") as log:
                                process = subprocess.Popen(
                                    [
                                        sys.executable,
                                        "-m",
                                        MODULE,
                                        "worker",
                                        "--root",
                                        str(root),
                                        "--role",
                                        role,
                                        "--key",
                                        key,
                                    ],
                                    cwd=root,
                                    env={
                                        **os.environ,
                                        "PYTHONUTF8": "1",
                                        "OMP_NUM_THREADS": "1",
                                        "OPENBLAS_NUM_THREADS": "1",
                                        "MKL_NUM_THREADS": "1",
                                    },
                                    stdin=subprocess.DEVNULL,
                                    stdout=log,
                                    stderr=log,
                                    creationflags=flags,
                                )
                            active.append(
                                {
                                    "role": role,
                                    "key": key,
                                    "pid": process.pid,
                                    "process": process,
                                    "started": time.time(),
                                    "deadline_seconds": spec["config"]["seconds"] * 2 + 300
                                    if role == "samples"
                                    else 1800,
                                }
                            )
                            running.add(key)
                state = (
                    "failed_draining"
                    if failed
                    else "draining"
                    if stopping
                    else "resource_wait"
                    if guarded
                    else "running"
                )
                status = {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "state": state,
                    "timestamp": time.time(),
                    "started": started,
                    "source_workers": source_workers,
                    "probe_workers": probe_workers,
                    "completed_samples": len(done["samples"]),
                    "completed_probes": len(done["probes"]),
                    "total_samples": len(manifest["keys"]),
                    "total_probes": len(manifest["keys"]),
                    "available_memory_gib": memory,
                    "free_disk_gib": disk,
                    "active": [{k: v for k, v in x.items() if k != "process"} for x in active],
                }
                atomic_json(status_path, status)
                if stopping and not active:
                    break
                time.sleep(2)
            complete = all(len(done[r]) == len(manifest["keys"]) for r in done)
            final: dict = {
                "state": "complete" if complete else "failed" if failed else "paused",
                "timestamp": time.time(),
                "elapsed": time.time() - started,
                "completed_samples": len(done["samples"]),
                "completed_probes": len(done["probes"]),
                "active": [],
            }
            if complete:
                from geo_llm_scheduler.experiments.d1_realpath_analysis import analyse_realpath

                analyse_realpath(root)
            atomic_json(root / "ops/exit.json", final)
            atomic_json(status_path, {**status, **final})
            return final
        finally:
            keep_awake(False)


def main() -> None:
    """Expose explicit local prepare/run/worker commands with preserved failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pilot-seconds", type=float)
    parser.add_argument("--source-workers", type=int, default=6)
    parser.add_argument("--probe-workers", type=int, default=2)
    parser.add_argument("--role", choices=("samples", "probes"))
    parser.add_argument("--key")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare_realpath(root, args.repository.resolve(), args.pilot_seconds)
        print(
            json.dumps(
                {
                    "root": str(root),
                    "runs": len(result["keys"]),
                    "source_worker_hours": result["source_worker_hours"],
                }
            )
        )
    elif args.command == "run":
        print(json.dumps(run_realpath(root, args.source_workers, args.probe_workers)))
    else:
        try:
            if args.role is None or args.key is None:
                raise ValueError("Worker role and key required")
            print(json.dumps(execute_realpath_job(root, args.role, args.key)))
        except Exception:
            atomic_json(
                root / "ops/failures" / f"{args.role}_{args.key}.json",
                {"timestamp": time.time(), "traceback": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    main()
