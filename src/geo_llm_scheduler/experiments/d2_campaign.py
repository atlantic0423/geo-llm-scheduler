"""Frozen resumable ten-hour-plus D2 opportunity and D1 cost campaign."""

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
from geo_llm_scheduler.experiments.d1_campaign import campaign_lease
from geo_llm_scheduler.experiments.d2_observation import FRACTIONS, OpportunityObserver
from geo_llm_scheduler.experiments.d2_probe import d1_cost_twins, probe_panel
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.p2_worker import campaign_path, canonical_source_hash
from geo_llm_scheduler.experiments.runner import digest, git_metadata, solution
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.utils.rng import RNGManager

MASTER_SEED = 2026100404
BASES = {50: tuple(range(850001, 850013)), 100: tuple(range(860001, 860013))}
SEEDS = (1101, 2202, 3303)
MODULE = "geo_llm_scheduler.experiments.d2_campaign"
PROTOCOL = "docs/experiments/d2_opportunity.md"


def frozen_manifest(root: Path) -> dict:
    """Refuse source, protocol, manifest or frozen input drift."""
    expected = json.loads((root / "manifest.sha256.json").read_text(encoding="utf-8"))
    if file_hash(root / "manifest.json") != expected["manifest.json"]:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if canonical_source_hash() != manifest["source_hash"]:
        raise ValueError("Frozen runtime source changed")
    repository = Path(__file__).resolve().parents[3]
    if file_hash(repository / PROTOCOL) != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed")
    if (
        manifest["kind"] == "D4_peak_diagnosis_v1"
        and file_hash(repository / "docs/experiments/d4_peak_diagnosis.md")
        != manifest["d4_protocol_sha256"]
    ):
        raise ValueError("D4 protocol changed")
    if (
        manifest["kind"] == "D3_response_v1"
        and file_hash(repository / "docs/experiments/d3_response.md")
        != manifest["d3_protocol_sha256"]
    ):
        raise ValueError("D3 protocol changed")
    for name, sha in manifest["input_files"].items():
        if file_hash(campaign_path(root, name)) != sha:
            raise ValueError(f"Frozen input changed: {name}")
    return manifest


def prepare_opportunity(
    root: Path,
    repository: Path,
    pilot_seconds: float | None = None,
    study: str = "D2",
    node: int = 0,
) -> dict:
    """Freeze 24 unseen bases, split by base before labels, with 144 source runs."""
    if study not in ("D2", "D4", "D3") or node not in (0, 1):
        raise ValueError("Unknown observation study")
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
    development: list[int] = []
    heldout: list[int] = []
    for n in (50, 100):
        formal_bases = (
            BASES[n]
            if study == "D2"
            else tuple(range(875001 if n == 50 else 885001, 875017 if n == 50 else 885017))
        )
        pilot_base = (
            (810001 if n == 50 else 820001) if study == "D2" else (874001 if n == 50 else 884001)
        )
        if study == "D3":
            formal_bases = tuple(
                range(895001 if n == 50 else 905001, 895025 if n == 50 else 905025)
            )
            pilot_base = (894001 if n == 50 else 904001) + node
        bases = (pilot_base,) if pilot_seconds else formal_bases
        split_at = 16 if study == "D3" else 8
        development.extend(bases[:split_at])
        heldout.extend(bases[split_at:])
        if study == "D3" and not pilot_seconds:
            bases = tuple(b for i, b in enumerate(bases) if i % 2 == node)
        for base in bases:
            pair = generate_e15_pair(n, base)
            for tariff, problem in zip(("H", "T"), pair):
                save_instance(problem, root / "inputs" / f"n{n}_i{base}_{tariff}.json")
            for seed in (6606, 7707, 8808) if pilot_seconds else SEEDS:
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
                    seconds = (pilot_seconds or 1800.0) * (1 if n == 50 else 2)
                    spec = {
                        "key": key,
                        "jobs": n,
                        "base_seed": base,
                        "tariff": tariff,
                        "split": "development" if base in development else "heldout",
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
                    }
                    atomic_json(root / "specs" / f"{key}.json", spec)
                    keys.append(key)
                    worker_seconds += seconds
    RNGManager(MASTER_SEED).stream("D2:source_order").shuffle(keys)
    manifest: dict = {
        "schema": 1,
        "kind": {"D2": "D2_opportunity_v1", "D4": "D4_peak_diagnosis_v1", "D3": "D3_response_v1"}[
            study
        ],
        "pilot": pilot_seconds is not None,
        "source_commit": provenance["git_commit"],
        "source_hash": source_hash,
        "master_seed": MASTER_SEED,
        "keys": keys,
        "development_bases": development,
        "heldout_bases": heldout,
        "fractions": FRACTIONS,
        "pool_size": 6,
        "source_workers": 10,
        "probe_workers": 2,
        "source_worker_hours": worker_seconds / 3600,
        "minimum_source_wall_hours": worker_seconds / 3600 / 10,
        "panels": len(keys) * 9,
        "max_action_rows": len(keys) * 9 * 6 * 9,
        "scope": "Prospective fixed-mate/neighbor-donor predictive screening; not online benefit"
        if study == "D2"
        else "Offline A8 failure diagnosis; no online pruning or quality claim",
        "budget": "B=3 unique complete exact per source/action; costs fully recorded",
        "protocol_sha256": file_hash(repository / PROTOCOL),
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for folder in ("inputs", "specs")
            for p in sorted((root / folder).glob("*.json"))
        },
    }
    if study == "D4":
        protocol_copy = root / "inputs/d4_peak_diagnosis.md"
        protocol_copy.write_bytes(
            (repository / "docs/experiments/d4_peak_diagnosis.md").read_bytes()
        )
        manifest["input_files"]["inputs/d4_peak_diagnosis.md"] = file_hash(protocol_copy)
        manifest["d4_protocol_sha256"] = file_hash(protocol_copy)
    if study == "D3":
        protocol_copy = root / "inputs/d3_response.md"
        protocol_copy.write_bytes((repository / "docs/experiments/d3_response.md").read_bytes())
        manifest["input_files"]["inputs/d3_response.md"] = file_hash(protocol_copy)
        manifest.update(
            d3_protocol_sha256=file_hash(protocol_copy),
            node=node,
            nodes=2,
            source_workers=8,
            probe_workers=2,
            minimum_source_wall_hours=worker_seconds / 3600 / 8,
            scope="Pre-search direction selection from six identical neighbor slots; same raw A1-A8 pool; no online HV claim",
            max_action_rows=len(keys) * 9 * 6 * 8,
        )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"manifest.json": file_hash(root / "manifest.json")})
    return manifest


def validate_opportunity_job(root: Path, role: str, key: str) -> tuple[bool, str]:
    """Validate all output and panel hashes plus provenance and full source budget."""
    try:
        if role not in ("samples", "probes"):
            return False, "unknown role"
        spec = json.loads((root / "specs" / f"{key}.json").read_text(encoding="utf-8"))
        output = root / role / key
        marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if marker["spec_hash"] != digest(spec) or set(marker["files"]) != {
            "data.json",
            "summary.json",
        }:
            return False, "identity mismatch"
        if any(file_hash(output / n) != h for n, h in marker["files"].items()):
            return False, "artifact hash mismatch"
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if summary["source_commit"] != spec["source_commit"] or (
            summary["source_hash"] != spec["source_hash"]
        ):
            return False, "source mismatch"
        if role == "samples" and (
            summary["termination_reason"] != "time_budget"
            or summary["engine_seconds"] < spec["config"]["seconds"]
            or summary["panels"] != 9
        ):
            return False, "incomplete source budget or phases"
        if role == "probes":
            sample = root / "samples" / key
            source_summary = json.loads((sample / "summary.json").read_text(encoding="utf-8"))
            if summary["sample_sha256"] != file_hash(sample / "data.json") or (
                summary["host"] != source_summary["host"] or summary["panels"] != 9
            ):
                return False, "source/host mismatch"
            for name, sha in summary["panel_files"].items():
                if file_hash(campaign_path(root, name)) != sha:
                    return False, "panel checkpoint changed"
        return True, "complete"
    except (OSError, ValueError, KeyError, TypeError):
        return False, "missing/unreadable evidence"


def execute_opportunity_job(root: Path, role: str, key: str) -> dict:
    """Publish source or whole paired panel outputs transactionally."""
    manifest = frozen_manifest(root)
    if role not in ("samples", "probes") or key not in manifest["keys"]:
        raise ValueError("Unknown job")
    provenance = git_metadata()
    if provenance["git_dirty"] or provenance["git_commit"] != manifest["source_commit"]:
        raise ValueError("Worker requires clean frozen source")
    spec = json.loads((root / "specs" / f"{key}.json").read_text(encoding="utf-8"))
    output = root / role / key
    temporary = root / role / f".{key}.attempt-{os.getpid()}"
    if output.exists() or temporary.exists():
        raise FileExistsError("Recover unfinished output before launching")
    temporary.mkdir(parents=True)
    begun = time.perf_counter()
    summary = {
        "key": key,
        "host": platform.node(),
        "source_hash": spec["source_hash"],
        "source_commit": spec["source_commit"],
    }
    problem = load_instance(root / spec["instance"])
    config_values = dict(spec["config"])
    for field in TUPLE_FIELDS:
        config_values[field] = tuple(config_values[field])
    config = Config(**config_values)

    def beat(values: dict) -> None:
        atomic_json(
            root / "ops/heartbeats" / f"{role}_{key}.json",
            {
                "pid": os.getpid(),
                "host": platform.node(),
                "timestamp": time.time(),
                **values,
            },
        )

    if role == "samples":
        assert config.seconds is not None
        initial = [
            Genotype(tuple(g["ms"]), tuple(g["os"]))
            for g in json.loads((root / spec["initial"]).read_text(encoding="utf-8"))
        ]
        observer = OpportunityObserver(problem, config.seconds, config.seed, config.neighborhood)
        last_beat = -10.0

        def callback(event: str, values: dict) -> None:
            nonlocal last_beat
            observer(event, values)
            if event == "offspring" and values["elapsed"] - last_beat >= 10:
                beat(
                    {
                        "state": "sampling",
                        "elapsed": values["elapsed"],
                        "exact": values["gateway"].counts["exact"],
                        "offspring": values["processed"],
                        "panels": len(observer.panels),
                    }
                )
                last_beat = values["elapsed"]

        result = run(problem, config, initial, retain_trace=False, observer=callback)
        data = observer.export()
        tick = time.perf_counter()
        for candidate in result.population + result.archive.members:
            if evaluate(problem, candidate.genotype, candidate.schedule) != candidate.evaluation:
                raise ValueError("Final source failed exact")
        data.update(
            population=[solution(c) for c in result.population],
            archive=[solution(c) for c in result.archive.members],
        )
        summary.update(
            engine_seconds=result.elapsed,
            termination_reason=result.termination_reason,
            offspring=result.offspring_count,
            exact=result.gateway.counts["exact"],
            panels=len(observer.panels),
            observer_seconds=result.observer_seconds,
            final_audit_checks=len(result.population) + len(result.archive.members),
            final_audit_seconds=time.perf_counter() - tick,
        )
    else:
        valid, reason = validate_opportunity_job(root, "samples", key)
        if not valid:
            raise ValueError(reason)
        sample = root / "samples" / key
        source_summary = json.loads((sample / "summary.json").read_text(encoding="utf-8"))
        if source_summary["host"] != platform.node():
            raise ValueError("Migrate unfinished paired jobs explicitly before continuing")
        sample_sha = file_hash(sample / "data.json")
        source = json.loads((sample / "data.json").read_text(encoding="utf-8"))
        panels = []
        checkpoint_files = {}
        for index, panel in enumerate(source["panels"]):
            name = f"ops/panel_checkpoints/{key}/{panel['panel']}.json"
            checkpoint = root / name
            checkpoint_sha = checkpoint.with_suffix(".sha256.json")
            if checkpoint.exists():
                expected = json.loads(checkpoint_sha.read_text(encoding="utf-8"))
                if file_hash(checkpoint) != expected["sha256"]:
                    raise ValueError("Partial panel hash mismatch")
                record = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (
                    record["spec_hash"] != digest(spec)
                    or record["sample_sha256"] != sample_sha
                    or (record["host"] != platform.node())
                ):
                    raise ValueError("Partial panel identity mismatch")
            else:
                seed = config.seed + index * 1000003
                if manifest["kind"] == "D4_peak_diagnosis_v1":
                    from geo_llm_scheduler.experiments.d4_peak import probe_peak_panel

                    diagnostic = probe_peak_panel(problem, panel, config, seed)
                    twins: dict = {"rows": []}
                elif manifest["kind"] == "D3_response_v1":
                    from geo_llm_scheduler.experiments.d3_response import probe_response_panel

                    diagnostic = probe_response_panel(problem, panel, config, seed)
                    twins = {"rows": []}
                else:
                    diagnostic = probe_panel(problem, panel, config, seed)
                    twins = d1_cost_twins(problem, panel, config, seed)
                record = {
                    "spec_hash": digest(spec),
                    "sample_sha256": sample_sha,
                    "host": platform.node(),
                    "d2": diagnostic,
                    "d1": twins,
                }
                atomic_json(checkpoint, record)
                atomic_json(checkpoint_sha, {"sha256": file_hash(checkpoint)})
            checkpoint_files[name] = file_hash(checkpoint)
            checkpoint_files[name.replace(".json", ".sha256.json")] = file_hash(checkpoint_sha)
            panels.append(record)
            beat(
                {"state": "probing", "panels": len(panels), "elapsed": time.perf_counter() - begun}
            )
        data = {"panels": panels, "sample_sha256": sample_sha}
        summary.update(
            panels=len(panels),
            sample_sha256=sample_sha,
            panel_files=checkpoint_files,
            rows=sum(len(p["d2"]["rows"]) for p in panels),
            audit_checks=sum(p["d2"]["audit_checks"] for p in panels),
            d1_binding=sum(r["binding"] for p in panels for r in p["d1"]["rows"]),
        )
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
    valid, reason = validate_opportunity_job(root, role, key)
    if not valid:
        raise ValueError(reason)
    beat({"state": "complete", "elapsed": summary["elapsed"]})
    return summary


def recover_interrupted(root: Path, migrate: bool = False) -> dict:
    """Quarantine interrupted attempts; never erase or retry recorded crashes.

    Migration preserves completed paired jobs and restarts unfinished pairs on
    the new host. All moves are verified to remain inside the campaign.
    """
    manifest = frozen_manifest(root)
    with campaign_lease(root / "ops/supervisor.lock"):
        status_path = root / "ops/status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if any(alive(x["pid"]) for x in status.get("active", [])):
            raise ValueError("Owned workers still alive")
        if list((root / "ops/failures").glob("*.json")):
            raise ValueError("Recorded crashes require explicit review, not automatic retry")
        destination = root / "ops/quarantine" / str(time.time_ns())
        paths_to_move = list((root / "samples").glob(".*.attempt-*")) + list(
            (root / "probes").glob(".*.attempt-*")
        )
        if migrate:
            for key in manifest["keys"]:
                if (root / "probes" / key).exists():
                    valid, reason = validate_opportunity_job(root, "probes", key)
                    if not valid:
                        raise ValueError(reason)
                    continue
                for path in (root / "samples" / key, root / "ops/panel_checkpoints" / key):
                    if path.exists():
                        paths_to_move.append(path)
        moved = []
        for path in paths_to_move:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Recovery path escapes campaign")
            target = destination / path.relative_to(root)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Quarantine path escapes campaign")
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            moved.append(str(target.relative_to(root)))
        result = {"host": platform.node(), "migrate": migrate, "quarantined": moved}
        atomic_json(root / "ops" / f"recovery_{time.time_ns()}.json", result)
        return result


def run_opportunity(root: Path, source_workers: int = 10, probe_workers: int = 2) -> dict:
    """Drain on stop/failure; never auto-retry or mix damaged paired evidence."""
    if (
        source_workers < 1
        or probe_workers < 1
        or source_workers > 10
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
                    valid, reason = validate_opportunity_job(root, role, key)
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
                    valid, reason = validate_opportunity_job(root, item["role"], item["key"])
                    if code != 0 or not valid:
                        failed = True
                        failure_path = root / "ops/failures" / f"{item['role']}_{item['key']}.json"
                        prior = (
                            json.loads(failure_path.read_text(encoding="utf-8"))
                            if failure_path.exists()
                            else {}
                        )
                        atomic_json(
                            failure_path,
                            {**prior, "exit_code": code, "reason": reason},
                        )
                    else:
                        done[item["role"]].add(item["key"])
                stopping = failed or (root / "stop.request").exists()
                memory, disk = available_memory_gb(), free_disk_gb(root)
                cgroup = Path("/sys/fs/cgroup")
                if (
                    manifest["kind"] in ("D4_peak_diagnosis_v1", "D3_response_v1")
                    and (cgroup / "memory.max").exists()
                ):
                    limit = (cgroup / "memory.max").read_text().strip()
                    if limit != "max":
                        memory = min(
                            memory,
                            (int(limit) - int((cgroup / "memory.current").read_text())) / 1024**3,
                        )
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
                                    else 7200,
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
                from geo_llm_scheduler.experiments.d2_analysis import analyse_opportunity

                try:
                    if manifest.get("kind") == "D4_peak_diagnosis_v1":
                        from geo_llm_scheduler.experiments.d4_peak import analyse_peak

                        analyse_peak(root)
                    elif manifest.get("kind") == "D3_response_v1":
                        from geo_llm_scheduler.experiments.d3_response import analyse_response

                        analyse_response(root)
                    else:
                        analyse_opportunity(root)
                except Exception:
                    atomic_json(
                        root / "ops/failures/analysis.json",
                        {"host": platform.node(), "traceback": traceback.format_exc()},
                    )
                    final.update(state="failed", execution_complete=True, analysis_complete=False)
            atomic_json(root / "ops/exit.json", final)
            atomic_json(status_path, {**status, **final})
            return final
        finally:
            keep_awake(False)


def main() -> None:
    """Expose freeze, run, recovery and explicit migration commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "recover", "migrate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pilot-seconds", type=float)
    parser.add_argument("--study", choices=("D2", "D4", "D3"), default="D2")
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--source-workers", type=int, default=10)
    parser.add_argument("--probe-workers", type=int, default=2)
    parser.add_argument("--role", choices=("samples", "probes"))
    parser.add_argument("--key")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare_opportunity(
            root, args.repository.resolve(), args.pilot_seconds, args.study, args.node
        )
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
        result = run_opportunity(root, args.source_workers, args.probe_workers)
        print(json.dumps(result))
        if result["state"] == "failed":
            raise SystemExit(1)
    elif args.command in ("recover", "migrate"):
        print(json.dumps(recover_interrupted(root, args.command == "migrate")))
    else:
        try:
            result = execute_opportunity_job(root, args.role, args.key)
            print(json.dumps(result))
        except Exception:
            atomic_json(
                root / "ops/failures" / f"{args.role}_{args.key}.json",
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
