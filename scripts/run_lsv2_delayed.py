"""Sealed, resumable workers for the LSV2 delayed-reward diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from geo_llm_scheduler.experiments.lsv2_delayed import (
    continue_search,
    opportunity_features,
    pair,
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write(p: Path, value: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _worker(args: argparse.Namespace) -> None:
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    manifest = _read(args.manifest)
    job = next(j for j in manifest["jobs"] if j["id"] == args.job)
    source = Path(__file__).resolve().parents[1]
    assert (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        == manifest["commit"]
    )
    path = args.input / job["key"] / job["snapshot"]
    assert _sha(path) == job["sha256"]
    with gzip.open(path, "rb") as f:
        snapshot = pickle.load(f)
    baseline = pickle.dumps(snapshot, protocol=5)
    out = args.output / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    features = opportunity_features(snapshot)
    _write(out / "features.json", features)
    previous = {
        (r["repeat"], r["operator"]): r
        for r in [
            json.loads(line) for line in (path.parent / "replay.jsonl").read_text().splitlines()
        ]
        if r["snapshot"] == job["snapshot"]
    }
    available = snapshot["meta"]["available"]
    cases = []
    seed0 = int.from_bytes(hashlib.sha256((job["key"] + job["snapshot"]).encode()).digest()[:8])
    for rep in range(manifest["repeats"]):
        for action in available:
            cases.append(dict(kind="pair", first=action, second="A7", repeat=rep))
            if action != "A7":
                cases.append(dict(kind="pair", first="A7", second=action, repeat=rep))
        for action in available + ["NOOP"]:
            cases.append(dict(kind="tail", first=action, repeat=rep))
    random.Random(seed0 + 999999).shuffle(cases)
    _write(
        out / "start.json",
        dict(
            pid=os.getpid(),
            started=time.time(),
            job=job,
            cases=len(cases),
            commit=manifest["commit"],
        ),
    )
    exact = 0
    completed = 0
    for case in cases:
        name = f"{case['kind']}_{case['repeat']}_{case['first']}_{case.get('second', '')}.json"
        dest = out / "trials" / name
        if dest.exists():
            result = _read(dest)
        else:
            seed = seed0 + 1000 + case["repeat"]
            follow = seed0 + 100000 + case["repeat"]
            tick = time.perf_counter()
            if case["kind"] == "pair":
                result = pair(snapshot, case["first"], case["second"], seed, follow)
            else:
                result = continue_search(snapshot, case["first"], seed, follow, job["seconds"])
            result.update(
                case,
                key=job["key"],
                snapshot=job["snapshot"],
                n=job["n"],
                phase=job["phase"],
                segment=job["segment"],
                seed=seed,
                follow_seed=follow,
                worker_seconds=time.perf_counter() - tick,
            )
            one = result.get("first_step")
            if one is not None:
                old = previous[case["repeat"], case["first"]]
                assert math.isclose(one["hv_gain"], old["hv_gain"], rel_tol=1e-10, abs_tol=1e-12)
                assert one["exact"] == old["exact"] and one["chain"] == old["chain"]
                result["old_prefix_match"] = True
            assert pickle.dumps(snapshot, protocol=5) == baseline
            _write(dest, result)
        exact += result["fresh_exact"]
        completed += 1
        _write(
            out / "heartbeat.json",
            dict(
                pid=os.getpid(),
                updated=time.time(),
                done=completed,
                total=len(cases),
                fresh_exact=exact,
                last=name,
            ),
        )
    files = [
        p
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name not in ("worker.log", "complete.json")
    ]
    record = dict(
        job=job,
        finished=time.time(),
        trials=completed,
        fresh_exact=exact,
        files=[
            dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=_sha(p))
            for p in files
        ],
    )
    if os.name == "posix":
        record["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _write(out / "complete.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "files"}), flush=True)


def _dispatch(args: argparse.Namespace) -> None:
    manifest = _read(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    pending = sorted(manifest["jobs"], key=lambda j: (-j["seconds"], j["id"]))
    active = {}
    done = []
    failed = []
    cpus = manifest["cpus"]
    _write(
        args.output / "start.json",
        dict(started=time.time(), pid=os.getpid(), commit=manifest["commit"], jobs=len(pending)),
    )
    while pending or active:
        for cpu, (proc, job, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[cpu]
            if code or not (args.output / job["id"] / "complete.json").exists():
                failed.append(dict(job=job["id"], returncode=code))
                pending = []
            else:
                done.append(job["id"])
        available = (
            int(
                next(
                    s.split()[1]
                    for s in Path("/proc/meminfo").read_text().splitlines()
                    if s.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        for cpu in cpus:
            if failed or not pending or cpu in active or available < 220 * 1024**3:
                continue
            job = pending.pop(0)
            out = args.output / job["id"]
            out.mkdir(parents=True, exist_ok=True)
            if (out / "complete.json").exists():
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
                "--manifest",
                str(args.manifest),
                "--input",
                str(args.input),
                "--output",
                str(args.output),
                "--job",
                job["id"],
            ]
            active[cpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), job, log)
        _write(
            args.output / "status.json",
            dict(
                updated=time.time(),
                done=len(done),
                total=len(manifest["jobs"]),
                pending=len(pending),
                failed=failed,
                mem_available=available,
                active=[dict(cpu=cpu, pid=p.pid, job=j["id"]) for cpu, (p, j, _) in active.items()],
            ),
        )
        if active or pending:
            time.sleep(3)
    _write(
        args.output / "exit.json",
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
    parser.add_argument("mode", choices=["worker", "dispatch"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job")
    args = parser.parse_args()
    try:
        (_worker if args.mode == "worker" else _dispatch)(args)
    except Exception:
        path = args.output / (args.job or "supervisor") / "failure.json"
        _write(path, dict(time=time.time(), error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    _main()
