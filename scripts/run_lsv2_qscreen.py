"""Sealed online-from-zero Q screen; independent per-run tables and fixed budgets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from run_lsv2 import preflight, sha, source_hash, write_candidates, write_json_gz

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.lsv2_qscreen import (
    LEARNING_ARMS,
    REFERENCES,
    calibrate,
    make_search,
    run_screen,
)
from geo_llm_scheduler.experiments.lsv2_study import study_config
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.qlearning import QTable

RUN_SEEDS = (18101, 18102)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare(args: argparse.Namespace) -> None:
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    old = args.historical
    prior = _read(old / "manifest.json")
    assert sha(old / "manifest.json") == _read(old / "manifest.sha256.json")["sha256"]
    specs = [_read(p) for p in sorted((old / "specs").glob("*.json"))]
    dev = [s for s in specs if s["panel"] == "dev"]
    assert len(dev) == 16
    with gzip.open(args.steps, "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with args.features.open(encoding="utf-8-sig") as f:
        features = list(csv.DictReader(f))
    files, blocks, jobs, calibrations = {}, [], [], {}
    for s in dev:
        old_spec = "specs/" + s["key"] + ".json"
        assert sha(old / old_spec) == prior["files"][old_spec]
        assert sha(old / s["input"]) == s["input_sha256"] == prior["files"][s["input"]]
        target = root / s["input"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / s["input"], target)
        files[s["input"]] = sha(target)
        for seed in RUN_SEEDS:
            key = s["base_seed"] + "_s" + str(seed)
            spec = dict(s, key=key, seed=seed, host=args.host)
            path = root / "specs" / (key + ".json")
            atomic_json(path, spec)
            files[path.relative_to(root).as_posix()] = sha(path)
            blocks.append(key)
    for n in (20, 50, 100, 200):
        instances = sorted([s for s in dev if s["n"] == n], key=lambda s: s["key"])
        assert len(instances) == 4
        for fold in (0, 1):
            train, valid = instances[fold::2], instances[1 - fold :: 2]
            calibration = calibrate(rows, features, [s["key"] for s in train])
            calibration["validation_keys"] = [s["key"] for s in valid]
            calibrations[f"n{n}_f{fold}"] = calibration
            # Historical calibration is fixed from OTHER instances; no Q pretraining.
            for s in valid:
                for seed in RUN_SEEDS:
                    key = s["base_seed"] + "_s" + str(seed)
                    for arm in LEARNING_ARMS:
                        jobs.append(
                            dict(
                                id=f"{key}_{arm}",
                                n=n,
                                fold=fold,
                                arm=arm,
                                calibration=calibration,
                                runs=[dict(key=key, training=True)],
                                seconds=s["seconds"],
                                learning_mode="online_from_zero",
                            )
                        )
        for s in instances:
            for seed in RUN_SEEDS:
                key = s["base_seed"] + "_s" + str(seed)
                for arm in REFERENCES:
                    jobs.append(
                        dict(
                            id=f"{key}_{arm}",
                            n=n,
                            fold=None,
                            arm=arm,
                            calibration={},
                            runs=[dict(key=key, training=False)],
                            seconds=s["seconds"],
                        )
                    )
    assert sum(len(j["runs"]) for j in jobs) == 416 and len(jobs) == 416
    atomic_json(root / "calibration.json", calibrations)
    files["calibration.json"] = sha(root / "calibration.json")
    manifest = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source=source_hash(),
        host=args.host,
        files=files,
        blocks=blocks,
        jobs=jobs,
        cpus=list(range(32)),
        child_address_limit_bytes=3 * 1024**3,
        source_evidence=dict(random_steps=sha(args.steps), features=sha(args.features)),
        budgets={20: 300, 50: 600, 100: 1200, 200: 2400},
        run_seeds=RUN_SEEDS,
        learning_mode="online_from_zero",
        pretraining_runs=0,
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    print(json.dumps(dict(jobs=len(jobs), runs=416, calibrations=calibrations)))


def _validate(root: Path) -> dict:
    manifest = _read(root / "manifest.json")
    assert sha(root / "manifest.json") == _read(root / "manifest.sha256.json")["sha256"]
    assert manifest["source"] == source_hash() and manifest["host"] == socket.gethostname()
    source = Path(__file__).resolve().parents[1]
    assert (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        == manifest["commit"]
    )
    for name, digest in manifest["files"].items():
        assert sha(root / name) == digest, name
    return manifest


def _sealed(path: Path) -> dict | None:
    marker = path / "complete.json"
    if not marker.exists():
        return None
    record = _read(marker)
    for name, value in record["files"].items():
        assert (path / name).stat().st_size == value["bytes"]
        assert sha(path / name) == value["sha256"]
    return record


def _seal(out: Path, names: list[str], **extra: object) -> None:
    atomic_json(
        out / "complete.json",
        dict(
            **extra,
            files={
                name: dict(bytes=(out / name).stat().st_size, sha256=sha(out / name))
                for name in names
            },
        ),
    )


def _run_one(root: Path, out: Path, job: dict, item: dict, table: QTable, manifest: dict) -> None:
    key, arm = item["key"], job["arm"]
    spec = _read(root / "specs" / (key + ".json"))
    expected = _read(root / "preflight" / (key + ".json"))
    assert expected["source"] == manifest["source"]
    assert expected["input_sha256"] == spec["input_sha256"]
    assert sha(root / spec["input"]) == spec["input_sha256"]
    cfg = study_config("LSV2", spec["seed"], spec["seconds"])
    search = make_search(arm, cfg, job["calibration"], table, training=item["training"])
    out.mkdir(parents=True, exist_ok=True)
    before = asdict(table)
    assert before == dict(values={}, visits={}), "No pretrained Q table permitted"
    atomic_json(out / "q_before.json", before)
    atomic_json(
        out / "config.json",
        dict(
            spec=spec,
            arm=arm,
            engine=asdict(cfg),
            lsv2=asdict(search.config),
            qspec=asdict(search.spec) if hasattr(search, "spec") else None,
            calibration=job["calibration"],
            training=item["training"],
            commit=manifest["commit"],
        ),
    )
    events: list = []
    snapshots: list = []
    last, inserted = -10.0, 0
    evidence: dict = {}
    clock = time.perf_counter()
    thresholds = [spec["seconds"] * f for f in (0.1, 0.25, 0.5, 0.75, 0.9)]

    def observe(event: str, data: dict) -> None:
        nonlocal last, inserted
        gw = data.get("gateway")
        if gw is None:
            return
        elapsed = time.perf_counter() - clock
        if event == "candidate" and gw.archive.insertions != inserted:
            inserted = gw.archive.insertions
            events.append((gw.counts["exact"], elapsed, *data["candidate"].evaluation.objectives))
        if event == "initial_ready":
            for field in ("initial_population_sha256", "initial_archive_sha256"):
                assert evidence[field] == expected["policies"]["bill4"]["evidence"][field]
            write_candidates(out / "initial.json.gz", data["population"])
        if event == "initial_ready" or thresholds and elapsed >= thresholds[0]:
            if event != "initial_ready":
                thresholds.pop(0)
            snapshots.append(
                dict(
                    elapsed=elapsed,
                    exact=gw.counts["exact"],
                    objectives=[c.evaluation.objectives for c in gw.archive.members],
                )
            )
        if elapsed - last >= 10 or event == "initial_ready":
            atomic_json(
                out / "heartbeat.json",
                dict(
                    pid=os.getpid(),
                    updated=time.time(),
                    elapsed=elapsed,
                    exact=gw.counts["exact"],
                    steps=len(search.steps),
                    training=item["training"],
                    arm=arm,
                    key=key,
                ),
            )
            last = elapsed

    problem = load_instance(root / spec["input"])
    result = run_screen(problem, cfg, evidence, search, observer=observe)
    assert result.termination_reason == "time_budget" and result.offspring_count > 0
    assert (
        search.steps and all("q_transition" in r for r in search.steps)
        if arm in LEARNING_ARMS
        else search.steps
    )
    if not item["training"]:
        assert asdict(table) == before
    audit_tick = time.perf_counter()
    for c in result.population + result.archive.members:
        assert c.evaluation.feasible and evaluate(problem, c.genotype, c.schedule) == c.evaluation
    audit_seconds = time.perf_counter() - audit_tick
    snapshots.append(
        dict(
            elapsed=result.elapsed,
            exact=result.gateway.counts["exact"],
            objectives=[c.evaluation.objectives for c in result.archive.members],
        )
    )
    write_candidates(out / "population.json.gz", result.population)
    write_candidates(out / "archive.json.gz", result.archive.members)
    write_json_gz(out / "lsv2_steps.json.gz", search.steps)
    write_json_gz(out / "archive_events.json.gz", events)
    atomic_json(out / "anytime.json", snapshots)
    atomic_json(out / "q_after.json", asdict(table))
    atomic_json(
        out / "summary.json",
        dict(
            key=key,
            arm=arm,
            training=item["training"],
            learning_mode="online_from_zero" if arm in LEARNING_ARMS else "reference",
            elapsed=result.elapsed,
            audit_seconds=audit_seconds,
            termination=result.termination_reason,
            offspring=result.offspring_count,
            counts=result.gateway.counts,
            initialization=evidence,
            lsv2=search.summary(),
            final_exact_verified=len(result.population) + len(result.archive.members),
        ),
    )
    names = [
        "config.json",
        "initial.json.gz",
        "population.json.gz",
        "archive.json.gz",
        "lsv2_steps.json.gz",
        "archive_events.json.gz",
        "anytime.json",
        "q_before.json",
        "q_after.json",
        "summary.json",
    ]
    _seal(out, names, commit=manifest["commit"], key=key, arm=arm, training=item["training"])


def _worker(args: argparse.Namespace) -> None:
    import fcntl
    import resource

    root = args.root
    manifest = _validate(root)
    limit = manifest["child_address_limit_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    job = next(j for j in manifest["jobs"] if j["id"] == args.job)
    out = root / "results" / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert len(job["runs"]) == 1, "Each job owns exactly one fresh Q table"
    table = QTable()
    completed = []
    for i, item in enumerate(job["runs"]):
        name = f"{i:02d}_{item['key']}"
        run = out / name
        marker = _sealed(run)
        if marker is None:
            _run_one(root, run, job, item, table, manifest)
        else:
            assert marker["commit"] == manifest["commit"] and marker["arm"] == job["arm"]
            assert marker["key"] == item["key"] and marker["training"] == item["training"]
            assert _read(run / "q_before.json") == asdict(table)
        table = QTable(**_read(run / "q_after.json"))
        completed.append(name + "/complete.json")
        atomic_json(
            out / "progress.json",
            dict(completed=i + 1, total=len(job["runs"]), updated=time.time()),
        )
    _seal(
        out,
        completed,
        commit=manifest["commit"],
        finished=time.time(),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def _dispatch(args: argparse.Namespace) -> None:
    root = args.root
    manifest = _validate(root)
    pending = sorted(manifest["jobs"], key=lambda j: (-j["seconds"], j["id"]))
    active: dict = {}
    done, failed = [], []
    atomic_json(
        root / "start.json", dict(started=time.time(), pid=os.getpid(), commit=manifest["commit"])
    )
    while pending or active:
        for cpu, (proc, job, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[cpu]
            if code or _sealed(root / "results" / job["id"]) is None:
                failed.append(dict(job=job["id"], returncode=code))
            else:
                done.append(job["id"])
        memory = (
            int(
                next(
                    s.split()[1]
                    for s in Path("/proc/meminfo").read_text().splitlines()
                    if s.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        disk = shutil.disk_usage(root).free
        if not failed:
            for cpu in manifest["cpus"]:
                if not pending or cpu in active or memory < 220 * 1024**3 or disk < 220 * 1024**3:
                    continue
                job = pending.pop(0)
                out = root / "results" / job["id"]
                out.mkdir(parents=True, exist_ok=True)
                if _sealed(out) is not None:
                    # Validate every run too; job marker only seals their manifests.
                    for i, item in enumerate(job["runs"]):
                        assert _sealed(out / f"{i:02d}_{item['key']}") is not None
                    done.append(job["id"])
                    continue
                log = (out / "worker.log").open("a")
                cmd = [
                    "taskset",
                    "-c",
                    str(cpu),
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--root",
                    str(root),
                    "--job",
                    job["id"],
                ]
                active[cpu] = (
                    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT),
                    job,
                    log,
                )
        atomic_json(
            root / "status.json",
            dict(
                updated=time.time(),
                done=len(done),
                total=len(manifest["jobs"]),
                pending=len(pending),
                failed=failed,
                mem_available=memory,
                disk_free=disk,
                active=[dict(cpu=c, pid=p.pid, job=j["id"]) for c, (p, j, _) in active.items()],
            ),
        )
        if failed and not active:
            break
        if pending or active:
            time.sleep(3)
    atomic_json(
        root / "exit.json",
        dict(
            state="failed" if failed else "complete",
            done=len(done),
            failed=failed,
            finished=time.time(),
        ),
    )
    if failed:
        raise RuntimeError(str(failed))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "worker", "dispatch", "validate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--host", default="neu01-2288H-V6")
    parser.add_argument("--job")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            _prepare(args)
        elif args.mode == "preflight":
            _validate(args.root)
            preflight(args.root, args.job)
        elif args.mode == "validate":
            _validate(args.root)
        else:
            (_worker if args.mode == "worker" else _dispatch)(args)
    except Exception:
        atomic_json(
            args.root / "failures" / ((args.job or args.mode) + ".json"),
            dict(time=time.time(), error=traceback.format_exc()),
        )
        raise


if __name__ == "__main__":
    _main()
