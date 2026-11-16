"""Operations-only flat dispatch of unchanged, independently audited workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"


def cpu_order(allowed: set[int], topology: dict[int, tuple[str, str]], parallel: int) -> list[int]:
    """Use each allowed physical core before using its SMT sibling."""
    groups: dict[tuple[str, str], list[int]] = {}
    for cpu in sorted(allowed):
        groups.setdefault(topology[cpu], []).append(cpu)
    order = [v[0] for v in groups.values()] + [c for v in groups.values() for c in v[1:]]
    if not 1 <= parallel <= min(96, len(order)):
        raise ValueError("Requested parallelism exceeds96 or allowed CPUs")
    return order[:parallel]


def pending_tasks(root: Path, manifest: dict, runner: Any) -> list[tuple[str, str, int]]:
    """Preserve frozen size and Williams arm order, skipping validated runs only."""
    tasks = []
    for key in manifest["blocks"]:
        spec = json.loads((root / "specs" / (key + ".json")).read_text())
        assert set(spec["order"]) == set(manifest["arms"]) and len(spec["order"]) == len(
            manifest["arms"]
        )
        for arm in spec["order"]:
            if not runner.complete(root, key, arm, manifest["source"]):
                tasks.append((key, arm, spec["seconds"]))
    assert len({(k, a) for k, a, _ in tasks}) == len(tasks)
    return tasks


def orphan_workers(root: Path) -> list[int]:
    """Detect existing original workers before any recovery or dispatch."""
    found = []
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = p.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"worker" in args and str(root).encode() in args:
            found.append(int(p.parent.name))
    return found


def resources_available(root: Path, active_count: int = 0) -> bool:
    """Retain at least 200GiB memory and disk with conservative dispatch guards and cap dispatch below cgroup high."""
    available = (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    # Reserve worst-case worker growth plus16 bounded HTTP children, even when
    # current RSS is small. Re-read this before EVERY admission, not once per wave.
    growth = (active_count + 1) * 2 * 1024**3 + 32 * 1024**3
    memory_ok = True
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            p = Path("/sys/fs/cgroup" + line[3:]) / "memory.current"
            if p.exists():
                memory_ok = int(p.read_text()) + growth < 360000000000
    return (
        available > 220 * 1024**3 + growth
        and shutil.disk_usage(root).free > 220 * 1024**3
        and memory_ok
    )


def dispatch(
    root: Path, runner: Any, cpus: list[int], worker_entry: Path, interval: float = 1.0
) -> dict:
    """Run unchanged workers; lock, audit, drain on failure/pause, and retain attempts."""
    import fcntl

    lock = (root / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = runner.validate_inputs(root)
    assert not list(root.glob("runs/*/*/failure.json")) and not (root / "failure.json").exists(), (
        "Recorded failure; manual review required"
    )
    assert not orphan_workers(root), "Existing workers; refusing duplicate dispatch"
    for key in manifest["blocks"]:
        check = json.loads((root / "preflight" / (key + ".json")).read_text())
        spec = json.loads((root / "specs" / (key + ".json")).read_text())
        assert check["source"] == manifest["source"] and check["host"] == manifest["host"]
        assert check["input_sha256"] == spec["input_sha256"]
    pending = deque(pending_tasks(root, manifest, runner))
    for key, arm, _ in pending:
        out = root / "runs" / key / arm
        if out.exists():
            target = root / "quarantine" / (key + "_" + arm + "_" + str(time.time_ns()))
            target.parent.mkdir(exist_ok=True)
            out.rename(target)
    start_path = root / "start.json"
    if not start_path.exists():
        runner.atomic_json(
            start_path,
            dict(
                started=time.time(),
                deadline=time.time() + manifest["window_seconds"],
                dispatch="flat_ops_v1",
                parallel=len(cpus),
            ),
        )
    start = json.loads(start_path.read_text())
    active: dict[int, dict] = {}
    done = manifest["runs"] - len(pending)
    failed = False
    while pending or active:
        stopped = (root / "stop.request").exists() or failed
        if stopped and not active:
            break
        for slot, cpu in enumerate(cpus):
            if slot in active or stopped or not pending:
                continue
            if not resources_available(root, len(active)):
                continue
            key, arm, seconds = pending[0]
            if time.time() + seconds + 900 > start["deadline"]:
                continue
            pending.popleft()
            log_path = root / "logs" / (key + "_" + arm + ".log")
            log_path.parent.mkdir(exist_ok=True)
            log = log_path.open("a")
            command = [
                sys.executable,
                str(worker_entry),
                "worker",
                "--root",
                str(root),
                "--key",
                key,
                "--arm",
                arm,
                "--cpu",
                str(cpu),
            ]
            try:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                log.close()
                failed = stopped = True
                runner.atomic_json(
                    root / "failure.json",
                    dict(key=key, arm=arm, dispatch_error=repr(exc), updated=time.time()),
                )
                break
            active[slot] = dict(
                child=child,
                log=log,
                key=key,
                arm=arm,
                cpu=cpu,
                started=time.time(),
                limit=seconds + 1800,
            )
        for slot, item in list(active.items()):
            child = item["child"]
            if child.poll() is None and time.time() - item["started"] > item["limit"]:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child.poll() is not None:
                item["log"].close()
                try:
                    assert child.returncode == 0, "Worker exit " + str(child.returncode)
                    assert runner.complete(root, item["key"], item["arm"], manifest["source"]), (
                        "Missing complete marker"
                    )
                    done += 1
                except Exception as exc:
                    failed = True
                    runner.atomic_json(
                        root / "failure.json",
                        dict(
                            key=item["key"],
                            arm=item["arm"],
                            code=child.returncode,
                            validation_error=repr(exc),
                            updated=time.time(),
                        ),
                    )
                del active[slot]
        runner.atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                updated=time.time(),
                started=start["started"],
                deadline=start["deadline"],
                done=done,
                total=manifest["runs"],
                dispatch="flat_ops_v1",
                parallel=len(cpus),
                pending_runs=len(pending),
                pending_blocks=len({k for k, _, _ in pending}),
                failure=failed,
                active=[
                    dict(
                        pid=x["child"].pid,
                        key=x["key"],
                        arm=x["arm"],
                        cpu=x["cpu"],
                        started=x["started"],
                    )
                    for x in active.values()
                ],
            ),
        )
        if pending and not active and time.time() + pending[0][2] + 900 > start["deadline"]:
            break
        time.sleep(interval)
    state = (
        "failed"
        if failed
        else "stopped"
        if (root / "stop.request").exists()
        else "complete"
        if done == manifest["runs"]
        else "window_complete"
    )
    result = dict(state=state, updated=time.time(), done=done, total=manifest["runs"])
    runner.atomic_json(root / "exit.json", result)
    lock.close()
    return result


def main() -> None:
    """Run only the authorized campaign roots with the frozen original worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=96)
    args = parser.parse_args()
    assert args.root.resolve() in {(ROOT / "formal").resolve(), (ROOT / "pilot96").resolve()}
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
    import run_llm_flash as runner

    assert not subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    ).strip()
    secret_path = os.environ.get("LLM_SECRET_FILE")
    if secret_path:
        secret = Path(secret_path)
        if secret.stat().st_mode & 0o077:
            raise PermissionError("Secret file must be owner-only")
        os.environ["DEEPSEEK_API_KEY"] = secret.read_text().strip()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("LLM credential is not configured")
    allowed = set(os.sched_getaffinity(0))
    topology = {
        c: tuple(
            (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
            for f in ("physical_package_id", "core_id")
        )
        for c in allowed
    }
    cpus = cpu_order(allowed, topology, args.parallel)
    print(
        json.dumps(dispatch(args.root, runner, cpus, SOURCE / "scripts/run_llm_flash.py")),
        flush=True,
    )


if __name__ == "__main__":
    main()
