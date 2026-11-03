"""Sealed twelve-arm B/L and A8 screen, matched to historical seed 16901."""

from __future__ import annotations

import argparse
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

from run_lsv2 import preflight as bill_preflight
from run_lsv2 import sha, source_hash, write_candidates, write_json_gz

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import ext_hybrid
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.lsv2_budget12 import (
    ARMS,
    EXTERNAL_ARMS,
    LEARNING_ARMS,
    RUN_SEED,
    arm_config,
    configuration_record,
    make_search,
)
from geo_llm_scheduler.experiments.lsv2_qscreen import run_screen
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager

RUN_SEEDS = (RUN_SEED,)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare(args: argparse.Namespace) -> None:
    root, old, prior_root = args.root, args.historical, args.prior_q
    root.mkdir(parents=True, exist_ok=False)
    prior = _read(old / "manifest.json")
    assert sha(old / "manifest.json") == _read(old / "manifest.sha256.json")["sha256"]
    qm = _read(prior_root / "manifest.json")
    assert sha(prior_root / "manifest.json") == _read(prior_root / "manifest.sha256.json")["sha256"]
    assert sha(prior_root / "calibration.json") == qm["files"]["calibration.json"]
    calibrations = _read(prior_root / "calibration.json")
    dev = [_read(p) for p in sorted((old / "specs").glob("*.json"))]
    dev = [s for s in dev if s["panel"] == "dev"]
    assert len(dev) == 16
    files, blocks, jobs, history = {}, [], [], {}
    for index, spec in enumerate(dev):
        key = spec["key"]
        assert spec["seed"] == RUN_SEED and key.endswith("_s16901")
        name = "specs/" + key + ".json"
        assert sha(old / name) == prior["files"][name]
        assert sha(old / spec["input"]) == spec["input_sha256"] == prior["files"][spec["input"]]
        target = root / spec["input"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / spec["input"], target)
        files[spec["input"]] = sha(target)
        spec = dict(spec, host=args.host)
        atomic_json(root / name, spec)
        files[name] = sha(root / name)
        blocks.append(key)
        matched = [c for c in calibrations.values() if key in c["validation_keys"]]
        assert len(matched) == 1 and key not in matched[0]["keys"]
        # Persist the original three-baseline initialization fingerprints, not results for fitting.
        old_run = old / "runs" / key / "EXTH_NSGAII"
        seal = _read(old_run / "complete.json")
        summary = old_run / "summary.json"
        assert sha(summary) == seal["files"]["summary.json"]["sha256"]
        hist = _read(summary)["initialization"]
        history[key] = {k: hist[k] for k in ("initial_population_sha256", "initial_archive_sha256")}
        order = ARMS[index % len(ARMS) :] + ARMS[: index % len(ARMS)]
        for arm in order:
            cal = matched[0] if arm in LEARNING_ARMS else {}
            config_name = "configs/" + key + "_" + arm + ".json"
            atomic_json(
                root / config_name, configuration_record(arm, RUN_SEED, spec["seconds"], cal)
            )
            files[config_name] = sha(root / config_name)
            jobs.append(
                dict(
                    id=f"{key}_{arm}",
                    arm=arm,
                    n=spec["n"],
                    calibration=cal,
                    runs=[dict(key=key, training=arm in LEARNING_ARMS)],
                    seconds=spec["seconds"],
                    config_file=config_name,
                    order_index=len(jobs),
                )
            )
    assert len(jobs) == 192
    for name, data in (
        ("calibration.json", calibrations),
        ("historical_initialization.json", history),
    ):
        atomic_json(root / name, data)
        files[name] = sha(root / name)
    manifest = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source=source_hash(),
        host=args.host,
        files=files,
        blocks=blocks,
        jobs=jobs,
        arms=ARMS,
        cpus=list(range(32)),
        child_address_limit_bytes=3 * 1024**3,
        run_seeds=RUN_SEEDS,
        learning_mode="online_from_zero",
        pretraining_runs=0,
        historical_manifest_sha256=sha(old / "manifest.json"),
        calibration_manifest_sha256=sha(prior_root / "manifest.json"),
        budgets={20: 300, 50: 600, 100: 1200, 200: 2400},
        classification="development_screen",
        pilot="waived_by_user; correctness_preflight_retained",
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    print(json.dumps(dict(jobs=len(jobs), seeds=RUN_SEEDS, arms=ARMS)))


def _preflight(root: Path, key: str) -> None:
    manifest = _validate(root)
    result = bill_preflight(root, key)
    spec = _read(root / "specs" / (key + ".json"))
    problem = load_instance(root / spec["input"])
    cfg = arm_config("EXTH_MOEAD", spec["seed"], spec["seconds"])
    gw = EvaluationGateway(problem, Archive())
    captured = []
    gw.candidate_sink = captured.append
    evidence = {}
    members = ext_hybrid.hybrid_initializer(problem, cfg, evidence)(gw, RNGManager(cfg.seed))
    assert len(members) == len(captured) == gw.counts["exact"] == 100
    for candidate in captured:
        assert candidate.evaluation.feasible
        assert decode(problem, candidate.genotype) == candidate.schedule
        assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
    expected = _read(root / "historical_initialization.json")[key]
    assert all(evidence[k] == v for k, v in expected.items()), (
        "Historical hybrid initialization mismatch"
    )
    result["policies"][ext_hybrid.POLICY] = dict(evidence=evidence, fresh_exact_verified=100)
    result["historical_hybrid_match"] = True
    result["config_checks"] = sum(j["runs"][0]["key"] == key for j in manifest["jobs"])
    assert result["config_checks"] == 12
    atomic_json(root / "preflight" / (key + ".json"), result)
    atomic_json(
        root / "preflight" / (key + ".sha256.json"),
        dict(sha256=sha(root / "preflight" / (key + ".json"))),
    )


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
    assert tuple(manifest["arms"]) == ARMS and manifest["run_seeds"] == [RUN_SEED]
    assert len(manifest["jobs"]) == 192 and len(manifest["blocks"]) == 16
    for job in manifest["jobs"]:
        spec = _read(root / "specs" / (job["runs"][0]["key"] + ".json"))
        assert spec["seed"] == RUN_SEED and spec["panel"] == "dev"
        expected = configuration_record(
            job["arm"], spec["seed"], spec["seconds"], job["calibration"]
        )
        assert _read(root / job["config_file"]) == json.loads(json.dumps(expected))
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
    assert (
        sha(root / "preflight" / (key + ".json"))
        == _read(root / "preflight" / (key + ".sha256.json"))["sha256"]
    )
    expected = _read(root / "preflight" / (key + ".json"))
    assert expected["source"] == manifest["source"]
    assert expected["input_sha256"] == spec["input_sha256"]
    assert sha(root / spec["input"]) == spec["input_sha256"]
    cfg = arm_config(arm, spec["seed"], spec["seconds"])
    external = arm in EXTERNAL_ARMS
    search = (
        None
        if external
        else make_search(arm, cfg, job["calibration"], table, training=item["training"])
    )
    steps = [] if search is None else search.steps
    policy = ext_hybrid.POLICY if external else "bill4"
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
            lsv2=None if search is None else asdict(search.config),
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
                assert evidence[field] == expected["policies"][policy]["evidence"][field]
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
                    steps=len(steps),
                    training=item["training"],
                    arm=arm,
                    key=key,
                ),
            )
            last = elapsed

    problem = load_instance(root / spec["input"])
    if external:
        result = ext_hybrid.run_arm(problem, cfg, arm, evidence, observer=observe)
    else:
        assert search is not None
        result = run_screen(problem, cfg, evidence, search, observer=observe)
    assert result.termination_reason == "time_budget" and result.offspring_count > 0
    if not external:
        assert steps
    if arm in LEARNING_ARMS:
        assert all("q_transition" in r for r in steps)
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
    write_json_gz(out / "lsv2_steps.json.gz", steps)
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
            lsv2=None if search is None else search.summary(),
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
    pending = sorted(manifest["jobs"], key=lambda j: (-j["seconds"], j["order_index"]))
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
    parser.add_argument("--prior-q", type=Path)
    parser.add_argument("--host", default="neu01-2288H-V6")
    parser.add_argument("--job")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            _prepare(args)
        elif args.mode == "preflight":
            _validate(args.root)
            _preflight(args.root, args.job)
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
