"""Frozen local D1 stage sampling and paired probes with bounded historical memory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterator

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.domain.models import Genotype
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
from geo_llm_scheduler.experiments.d1 import ARMS, paired_probe, perturb_structure
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.p1_observation import restore_candidate, sample_population
from geo_llm_scheduler.experiments.p2_worker import campaign_path, canonical_source_hash
from geo_llm_scheduler.experiments.runner import digest, git_metadata, solution
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.utils.rng import RNGManager

FRACTIONS = (0.1, 0.5, 0.9)


@contextmanager
def campaign_lease(path: Path) -> Iterator[None]:
    """Hold an exclusive OS-released lock, surviving supervisor crashes safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare(root: Path, repository: Path, pilot_seconds: float | None = None) -> dict:
    """Freeze 80 F6 source runs and 240 stage probes; never overwrite a campaign."""
    if (root / "manifest.json").exists():
        raise FileExistsError("Campaign already frozen")
    provenance = git_metadata(repository)
    if not provenance["git_commit"] or provenance["git_dirty"]:
        raise ValueError("Freeze requires a clean commit")
    if pilot_seconds is not None and pilot_seconds <= 0:
        raise ValueError("Pilot duration must be positive")
    for directory in ("inputs", "specs", "samples", "probes", "checkpoints", "ops", "logs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    config = load_config(repository / "configs/p2_local/F6.yaml")
    source_hash = canonical_source_hash()
    sample_keys: list[str] = []
    probe_keys: list[str] = []
    worker_seconds = 0.0
    for n, first in ((50, 810001), (100, 820001)):
        for base in range(first, first + (1 if pilot_seconds is not None else 4)):
            pair = generate_e15_pair(n, base)
            for tariff, problem in zip(("H", "T"), pair):
                save_instance(problem, root / "inputs" / f"n{n}_i{base}_{tariff}.json")
            for seed in (6606,) if pilot_seconds is not None else (1101, 2202, 3303, 4404, 5505):
                initialization = initial_genotypes(
                    pair[0], replace(config, seed=seed), RNGManager(seed).stream("initialization")
                )
                initial = f"inputs/n{n}_i{base}_a{seed}_initial.json"
                atomic_json(root / initial, [asdict(g) for g in initialization])
                for tariff in ("H", "T"):
                    key = f"n{n}_i{base}_{tariff}_a{seed}"
                    instance = f"inputs/n{n}_i{base}_{tariff}.json"
                    seconds = (
                        pilot_seconds * (1 if n == 50 else 2)
                        if pilot_seconds is not None
                        else (1800.0 if n == 50 else 3600.0)
                    )
                    spec = {
                        "key": key,
                        "jobs": n,
                        "base_seed": base,
                        "tariff": tariff,
                        "algorithm_seed": seed,
                        "instance": instance,
                        "instance_hash": file_hash(root / instance),
                        "initial": initial,
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
                        "fractions": FRACTIONS,
                        "probe_quota_seconds": 1.0 if n == 50 else 3.0,
                        "intent_cap_seconds": 900.0,
                    }
                    atomic_json(root / "specs" / f"{key}.json", spec)
                    sample_keys.append(key)
                    worker_seconds += seconds
                    for stage in range(len(FRACTIONS)):
                        probe_keys.append(f"{key}_s{stage}")
    RNGManager(20261003).stream("D1:source_order").shuffle(sample_keys)
    manifest = {
        "schema": 1,
        "kind": "D1_pilot" if pilot_seconds is not None else "D1_overnight_80",
        "source_commit": provenance["git_commit"],
        "source_hash": source_hash,
        "samples": sample_keys,
        "probes": probe_keys,
        "arms": ARMS,
        "source_worker_hours": worker_seconds / 3600,
        "source_workers": 6,
        "probe_workers": 2,
        "fractions": FRACTIONS,
        "exact_cap_per_arm": 3,
        "projection_scales": (0.25, 0.5, 1.0),
        "magnitudes": "1 job and ceil(0.1*N) jobs",
        "minimum_source_wall_hours": worker_seconds / 3600 / 6,
        "scope": "8 development bases; offline D1 residual waits only; not online MOEA/D benefit",
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for folder in ("inputs", "specs")
            for p in sorted((root / folder).glob("*.json"))
        },
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"sha256": file_hash(root / "manifest.json")})
    return manifest


def validate_job(root: Path, role: str, key: str) -> tuple[bool, str]:
    """Verify role, frozen spec, source provenance and every atomic result artifact."""
    try:
        marker = json.loads((root / role / key / "complete.json").read_text(encoding="utf-8"))
        sample_key = key if role == "samples" else key.rsplit("_s", 1)[0]
        spec = json.loads((root / "specs" / f"{sample_key}.json").read_text(encoding="utf-8"))
        if marker["role"] != role or marker["key"] != key or marker["spec_hash"] != digest(spec):
            return False, "identity mismatch"
        if marker["source_hash"] != spec["source_hash"]:
            return False, "source mismatch"
        if any(file_hash(campaign_path(root, p)) != h for p, h in marker["files"].items()):
            return False, "artifact mismatch"
        required = {f"{role}/{key}/summary.json", f"{role}/{key}/data.json"}
        required.update(
            {f"checkpoints/{key}_s{stage}.json" for stage in range(len(FRACTIONS))}
            if role == "samples"
            else {f"checkpoints/{key}.json"}
        )
        if required != set(marker["files"]):
            return False, "incomplete artifacts"
        summary = json.loads((root / role / key / "summary.json").read_text(encoding="utf-8"))
        if (
            summary["source_hash"] != spec["source_hash"]
            or summary["source_commit"] != spec["source_commit"]
            or summary["host"] != marker["host"]
        ):
            return False, "summary provenance mismatch"
        if role == "samples" and (
            summary["termination_reason"] != "time_budget"
            or summary["elapsed"] < spec["config"]["seconds"]
            or summary["offspring"] < 1
            or summary["checkpoints"] != 3
        ):
            return False, "source run ended before the frozen budget"
        if role == "probes" and summary["quartets"] + summary["skipped"] != summary["planned"]:
            return False, "probe accounting mismatch"
        return True, "complete"
    except (OSError, ValueError, KeyError, TypeError):
        return False, "missing/unreadable result"


def execute(root: Path, role: str, key: str) -> None:
    """Execute one source or whole-stage probe, publishing only verified artifacts."""
    if role not in ("samples", "probes"):
        raise ValueError("Unknown worker role")
    sample_key = key if role == "samples" else key.rsplit("_s", 1)[0]
    spec = json.loads(campaign_path(root, f"specs/{sample_key}.json").read_text(encoding="utf-8"))
    provenance = git_metadata()
    if (
        provenance["git_dirty"]
        or provenance["git_commit"] != spec["source_commit"]
        or canonical_source_hash() != spec["source_hash"]
    ):
        raise ValueError("Worker requires the clean frozen source")
    for path, sha in (
        (spec["instance"], spec["instance_hash"]),
        (spec["initial"], spec["initial_hash"]),
    ):
        if file_hash(campaign_path(root, path)) != sha:
            raise ValueError("Frozen input mismatch")
    output = campaign_path(root, f"{role}/{key}")
    temporary = campaign_path(root, f"{role}/.{key}.attempt-{os.getpid()}")
    if output.exists() or temporary.exists():
        raise FileExistsError("Supervisor must recover unfinished output before launch")
    temporary.mkdir(parents=True)
    problem = load_instance(root / spec["instance"])
    begun = time.perf_counter()
    heartbeat = root / "ops/heartbeats" / f"{role}_{key}.json"

    def beat(values: dict) -> None:
        atomic_json(
            heartbeat,
            {
                "pid": os.getpid(),
                "host": platform.node(),
                "timestamp": time.time(),
                "role": role,
                "key": key,
                "rss_peak_gib": process_peak_rss_gb(),
                **values,
            },
        )

    beat({"state": "starting", "elapsed": 0.0})
    checkpoint_files: list[str] = []
    contents: dict
    if role == "samples":
        data = dict(spec["config"])
        for field in TUPLE_FIELDS:
            data[field] = tuple(data[field])
        config = Config(**data)
        assert config.seconds is not None
        budget_seconds = config.seconds
        initial = [
            Genotype(tuple(g["ms"]), tuple(g["os"]))
            for g in json.loads((root / spec["initial"]).read_text(encoding="utf-8"))
        ]
        sample_rng = RNGManager(config.seed).stream("D1:observation")
        last_beat = -10.0

        def observe(event: str, context: dict) -> None:
            nonlocal last_beat
            if event != "offspring":
                return
            elapsed = context["elapsed"]
            for stage, fraction in enumerate(FRACTIONS):
                name = f"checkpoints/{key}_s{stage}.json"
                if elapsed >= fraction * budget_seconds and name not in checkpoint_files:
                    population = context["population"]
                    sampled = sample_population(population, sample_rng)
                    payload = {
                        "key": key,
                        "stage": stage,
                        "fraction": fraction,
                        "elapsed": elapsed,
                        "processed": context["processed"],
                        "spec_hash": digest(spec),
                        "source_hash": spec["source_hash"],
                        "ideal": context["gateway"].ideal,
                        "maximum": maximum(population),
                        "sample": [
                            {
                                **row,
                                "candidate": solution(population[row["index"]])
                                if row["index"] is not None
                                else None,
                            }
                            for row in sampled
                        ],
                    }
                    atomic_json(root / name, payload)
                    checkpoint_files.append(name)
            if elapsed - last_beat >= 10:
                beat(
                    {
                        "state": "sampling",
                        "elapsed": elapsed,
                        "exact": context["gateway"].counts["exact"],
                        "offspring": context["processed"],
                        "checkpoints": len(checkpoint_files),
                    }
                )
                last_beat = elapsed

        result = run(problem, config, initial, retain_trace=False, observer=observe)
        tick = time.perf_counter()
        for candidate in result.population + result.archive.members:
            if evaluate(problem, candidate.genotype, candidate.schedule) != candidate.evaluation:
                raise ValueError("Final source candidate failed exact re-evaluation")
        summary = {
            "elapsed": result.elapsed,
            "termination_reason": result.termination_reason,
            "offspring": result.offspring_count,
            "counts": result.gateway.counts,
            "checkpoints": len(checkpoint_files),
            "verification_seconds": time.perf_counter() - tick,
        }
        contents = {
            "population": [solution(c) for c in result.population],
            "archive": [solution(c) for c in result.archive.members],
        }
    else:
        checkpoint = root / f"checkpoints/{key}.json"
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload["spec_hash"] != digest(spec) or payload["source_hash"] != spec["source_hash"]:
            raise ValueError("Checkpoint does not match the frozen source")
        context = NormalizationContext(tuple(payload["ideal"]), tuple(payload["maximum"]))
        streams = RNGManager(spec["algorithm_seed"])
        rows: list[dict] = []
        skipped = planned = 0
        for sample_index, sample in enumerate(payload["sample"]):
            if sample["candidate"] is None:
                skipped += 6
                planned += 6
                rows.append(
                    {"missing_source": True, "sample_index": sample_index, "sample": sample}
                )
                continue
            source = restore_candidate(sample["candidate"])
            weight = weights(spec["config"]["population"])[sample["index"]]
            for kind in ("OS", "INSTANCE", "REGION"):
                for magnitude in (1, max(2, (len(problem.jobs) + 9) // 10)):
                    planned += 1
                    label = f"{key}:{sample_index}:{kind}:{magnitude}"
                    seed = streams.stream(f"D1:case:{label}").getrandbits(63)
                    tick = time.perf_counter()
                    target = perturb_structure(
                        problem,
                        source.genotype,
                        kind,
                        magnitude,
                        RNGManager(seed).stream("D1:structure"),
                    )
                    perturbation_seconds = time.perf_counter() - tick
                    metadata = {
                        "sample_index": sample_index,
                        "preference": sample["stratum"],
                        "kind": kind,
                        "magnitude": magnitude,
                        "seed": seed,
                        "perturbation_seconds": perturbation_seconds,
                    }
                    if target is None:
                        skipped += 1
                        rows.append({**metadata, "unchanged_structure": True})
                    else:
                        rows.append(
                            {
                                **metadata,
                                **paired_probe(
                                    problem,
                                    source,
                                    target,
                                    weight,
                                    context,
                                    seed,
                                    spec["probe_quota_seconds"],
                                    spec["intent_cap_seconds"],
                                ),
                            }
                        )
                    beat(
                        {
                            "state": "probing",
                            "elapsed": time.perf_counter() - begun,
                            "cases": planned,
                            "skipped": skipped,
                        }
                    )
        checkpoint_files.append(f"checkpoints/{key}.json")
        summary = {
            "quartets": sum("arms" in r for r in rows),
            "skipped": skipped,
            "planned": planned,
            "elapsed": time.perf_counter() - begun,
            "source_fraction": payload["fraction"],
            "source_elapsed": payload["elapsed"],
        }
        contents = {"rows": rows, "checkpoint_sha256": file_hash(checkpoint)}
    summary.update(
        host=platform.node(),
        rss_peak_gib=process_peak_rss_gb(),
        source_commit=spec["source_commit"],
        source_hash=spec["source_hash"],
        total_worker_seconds=time.perf_counter() - begun,
    )
    atomic_json(temporary / "summary.json", summary)
    atomic_json(temporary / "data.json", contents)
    # Marker references final relative paths; publish the directory atomically,
    # then validate. The supervisor will refuse to reuse an invalid marker.
    atomic_json(
        temporary / "complete.json",
        {
            "role": role,
            "key": key,
            "spec_hash": digest(spec),
            "source_hash": spec["source_hash"],
            "host": platform.node(),
            "files": {
                **{
                    f"{role}/{key}/{name}": file_hash(temporary / name)
                    for name in ("summary.json", "data.json")
                },
                **{name: file_hash(root / name) for name in checkpoint_files},
            },
        },
    )
    temporary.rename(output)
    valid, reason = validate_job(root, role, key)
    if not valid:
        raise ValueError(reason)
    beat({"state": "complete", **summary})


def recover_job(root: Path, role: str, key: str) -> bool:
    """Reuse valid completions; quarantine interrupted attempts without deleting evidence."""
    if (root / "ops/failures" / f"{role}_{key}.json").exists():
        raise RuntimeError("Recorded failure needs review before resume")
    output = root / role / key
    if output.exists():
        valid, reason = validate_job(root, role, key)
        if not valid:
            raise RuntimeError(f"Invalid result {role}/{key}: {reason}")
        return True
    attempts = list((root / role).glob(f".{key}.attempt-*"))
    if attempts:
        destination = root / "quarantine" / f"{role}_{key}_{time.time_ns()}"
        destination.mkdir(parents=True)
        if role == "samples":
            for stage in range(len(FRACTIONS)):
                probe_key = f"{key}_s{stage}"
                checkpoint = root / f"checkpoints/{probe_key}.json"
                if checkpoint.exists():
                    checkpoint.rename(destination / checkpoint.name)
                old_probe = root / "probes" / probe_key
                if old_probe.exists():
                    old_probe.rename(destination / old_probe.name)
        for path in attempts:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Recovery escaped the campaign")
            path.rename(destination / path.name)
    return False


def analyse(root: Path, manifest: dict) -> dict:
    """Aggregate paired differences by independent base, retaining stage and cost strata."""
    if not all(validate_job(root, "samples", key)[0] for key in manifest["samples"]):
        raise ValueError("Analysis requires complete source budgets and checkpoint hashes")
    records: list[dict] = []
    for key in manifest["probes"]:
        if not validate_job(root, "probes", key)[0]:
            raise ValueError("Analysis requires all complete paired probe jobs")
        sample_key = key.rsplit("_s", 1)[0]
        spec = json.loads((root / "specs" / f"{sample_key}.json").read_text(encoding="utf-8"))
        data = json.loads((root / "probes" / key / "data.json").read_text(encoding="utf-8"))
        for row in data["rows"]:
            if "arms" not in row:
                continue
            denominator = max(row["base_scalar"], 1e-12)
            true = row["arms"]["TRUE"]
            records.append(
                {
                    "base": spec["base_seed"],
                    "jobs": spec["jobs"],
                    "tariff": spec["tariff"],
                    "seed": spec["algorithm_seed"],
                    "stage": key.rsplit("_s", 1)[1],
                    "preference": row["preference"],
                    "kind": row["kind"],
                    "magnitude": row["magnitude"],
                    "true_gain": true["scalar_gain"] / denominator,
                    "true_minus_shuffled": (row["arms"]["SHUFFLED"]["scalar"] - true["scalar"])
                    / denominator,
                    "true_minus_a7": (row["arms"]["FRESH_A7"]["scalar"] - true["scalar"])
                    / denominator,
                    "true_seconds": true["seconds"],
                    "a7_seconds": row["arms"]["FRESH_A7"]["seconds"],
                    "true_overshoot": true["overshoot_seconds"],
                    "a7_overshoot": row["arms"]["FRESH_A7"]["overshoot_seconds"],
                    "intent_nonzero": true["intent_nonzero"],
                }
            )
    directory = root / "analysis"
    directory.mkdir(exist_ok=True)
    if records:
        with (directory / "paired_cases.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    by_base = []
    metrics = ("true_gain", "true_minus_shuffled", "true_minus_a7", "true_seconds", "a7_seconds")
    for base in sorted({r["base"] for r in records}):
        subset = [r for r in records if r["base"] == base]
        by_base.append(
            {
                "base": base,
                "cases": len(subset),
                **{m: sum(r[m] for r in subset) / len(subset) for m in metrics},
            }
        )
    report = {
        "status": "complete",
        "cases": len(records),
        "independent_bases": len(by_base),
        "base_means": by_base,
        "scope": "Development mechanism evidence only; samples/seeds are not independent bases.",
        "matched_cost": "Same per-arm quota; late candidates excluded; actual cost and overshoot retained.",
        "representation": "Positive frozen-profile residual waits only; no full TOU/peak intent claim.",
        "source_commit": manifest["source_commit"],
        "timestamp": time.time(),
    }
    atomic_json(directory / "report.json", report)
    return report


def supervise(root: Path, source_workers: int = 6, probe_workers: int = 2) -> None:
    """Run source and probe queues, drain on stop or failure, and finish paired aggregation."""
    if (
        source_workers < 1
        or probe_workers < 1
        or source_workers + probe_workers > (os.cpu_count() or 1)
    ):
        raise ValueError("Invalid local worker allocation")
    with campaign_lease(root / "ops/supervisor.lock"):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            file_hash(root / "manifest.json")
            != json.loads((root / "manifest.sha256.json").read_text())["sha256"]
            or canonical_source_hash() != manifest["source_hash"]
            or any(file_hash(root / p) != h for p, h in manifest["input_files"].items())
        ):
            raise ValueError("Frozen campaign/source changed")
        if manifest["kind"] != "D1_pilot" and source_workers > manifest["source_workers"]:
            raise ValueError("Worker increase would violate the minimum wall-clock allocation")
        status_file = root / "ops/status.json"
        if status_file.exists() and any(
            alive(x["pid"]) for x in json.loads(status_file.read_text())["active"]
        ):
            raise RuntimeError("Prior workers are still alive")
        done: dict[str, set[str]] = {role: set() for role in ("samples", "probes")}
        for role in done:
            for key in manifest[role]:
                if recover_job(root, role, key):
                    done[role].add(key)
        active: list[dict] = []
        failed = False
        started = time.time()
        atomic_json(
            root / "ops" / f"epoch_{time.time_ns()}.json",
            {
                "started": started,
                "pid": os.getpid(),
                "host": platform.node(),
                "source_workers": source_workers,
                "probe_workers": probe_workers,
                "manifest_sha256": file_hash(root / "manifest.json"),
            },
        )
        keep_awake(True)
        try:
            while len(done["samples"]) < len(manifest["samples"]) or len(done["probes"]) < len(
                manifest["probes"]
            ):
                for item in list(active):
                    code = item["process"].poll()
                    role, key = item["role"], item["key"]
                    if code is None:
                        if time.time() - item["started"] > item["deadline_seconds"]:
                            item["process"].terminate()
                            atomic_json(
                                root / "ops/failures" / f"{role}_{key}.json",
                                {
                                    "reason": "owned worker exceeded hard watchdog",
                                    "pid": item["pid"],
                                },
                            )
                            failed = True
                        continue
                    active.remove(item)
                    valid, reason = validate_job(root, role, key)
                    if code != 0 or not valid:
                        atomic_json(
                            root / "ops/failures" / f"{role}_{key}.json",
                            {"exit_code": code, "reason": reason, "timestamp": time.time()},
                        )
                        failed = True
                    else:
                        done[role].add(key)
                stopping = failed or (root / "stop.request").exists()
                memory, disk = available_memory_gb(), free_disk_gb(root)
                guarded = memory < 4 or disk < 20
                if not stopping and not guarded:
                    for role, cap in (("samples", source_workers), ("probes", probe_workers)):
                        running = {item["key"] for item in active if item["role"] == role}
                        for key in manifest[role]:
                            if len(running) >= cap:
                                break
                            if key in done[role] or key in running:
                                continue
                            if role == "probes" and not (root / f"checkpoints/{key}.json").exists():
                                continue
                            spec_key = key if role == "samples" else key.rsplit("_s", 1)[0]
                            spec = json.loads((root / "specs" / f"{spec_key}.json").read_text())
                            flags = (
                                subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
                                if sys.platform == "win32"
                                else 0
                            )
                            with (root / "logs" / f"{role}_{key}.log").open("ab") as log:
                                process = subprocess.Popen(
                                    [
                                        sys.executable,
                                        "-m",
                                        "geo_llm_scheduler.experiments.d1_campaign",
                                        "worker",
                                        "--root",
                                        str(root),
                                        "--role",
                                        role,
                                        "--key",
                                        key,
                                    ],
                                    cwd=root,
                                    env=dict(
                                        os.environ,
                                        OMP_NUM_THREADS="1",
                                        OPENBLAS_NUM_THREADS="1",
                                        MKL_NUM_THREADS="1",
                                        PYTHONUTF8="1",
                                    ),
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
                                    "deadline_seconds": spec["config"]["seconds"] * 3 + 600
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
                atomic_json(
                    status_file,
                    {
                        "pid": os.getpid(),
                        "host": platform.node(),
                        "timestamp": time.time(),
                        "started": started,
                        "state": state,
                        "source_workers": source_workers,
                        "probe_workers": probe_workers,
                        "completed_samples": len(done["samples"]),
                        "total_samples": len(manifest["samples"]),
                        "completed_probes": len(done["probes"]),
                        "total_probes": len(manifest["probes"]),
                        "available_memory_gib": memory,
                        "free_disk_gib": disk,
                        "active": [
                            {k: v for k, v in item.items() if k != "process"} for item in active
                        ],
                    },
                )
                if stopping and not active:
                    break
                time.sleep(2)
            complete = all(len(done[r]) == len(manifest[r]) for r in done)
            if complete:
                analyse(root, manifest)
            final = {
                "state": "complete" if complete else "failed" if failed else "paused",
                "timestamp": time.time(),
                "started": started,
                "elapsed": time.time() - started,
                "completed_samples": len(done["samples"]),
                "completed_probes": len(done["probes"]),
            }
            atomic_json(root / "ops/exit.json", final)
            old = json.loads(status_file.read_text()) if status_file.exists() else {}
            atomic_json(status_file, {**old, **final, "active": []})
        finally:
            keep_awake(False)


def main() -> None:
    """Prepare, supervise or execute a frozen local D1 campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-seconds", type=float)
    parser.add_argument("--source-workers", type=int, default=6)
    parser.add_argument("--probe-workers", type=int, default=2)
    parser.add_argument("--role", choices=("samples", "probes"))
    parser.add_argument("--key")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.operation == "prepare":
        result = prepare(root, Path(__file__).resolve().parents[3], args.pilot_seconds)
        print(
            json.dumps({k: v for k, v in result.items() if k != "input_files"}, ensure_ascii=False)
        )
    elif args.operation == "run":
        supervise(root, args.source_workers, args.probe_workers)
    else:
        try:
            if args.role is None or args.key is None:
                raise ValueError("Worker needs a role and key")
            execute(root, args.role, args.key)
        except Exception:
            atomic_json(
                root / "ops/failures" / f"{args.role}_{args.key}.json",
                {"pid": os.getpid(), "timestamp": time.time(), "traceback": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    main()
