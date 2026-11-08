"""Operations-only flat dispatch of slot-assignment workers, longest budgets first."""

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


def cpu_order(
    allowed: set[int],
    topology: dict[int, tuple[str, str]],
    parallel: int,
    excluded: frozenset[int] = frozenset(),
) -> list[int]:
    """Use each allowed physical core before its SMT sibling; skip excluded cores entirely.

    Excluding one logical CPU removes its whole physical core, so another study
    pinned there never shares a core with these workers.
    """
    busy = {topology[c] for c in excluded if c in topology}
    groups: dict[tuple[str, str], list[int]] = {}
    for cpu in sorted(allowed):
        if topology[cpu] not in busy:
            groups.setdefault(topology[cpu], []).append(cpu)
    order = [v[0] for v in groups.values()] + [c for v in groups.values() for c in v[1:]]
    if not 1 <= parallel <= min(96, len(order)):
        raise ValueError("Requested parallelism exceeds96 or allowed CPUs")
    return order[:parallel]


def pending_tasks(root: Path, manifest: dict, runner: Any) -> list[tuple[str, str, int]]:
    """Longest budgets first, stable within frozen Williams order; skip validated runs."""
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
    return sorted(tasks, key=lambda task: -task[2])


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


CONTAINER_LIMIT = 32 * 1024**3


def memory_gate(
    host_available: int, disk_free: int, cgroup_current: int | None, cgroup_max: int | None
) -> bool:
    """Decide whether one more worker may start.

    Shared host (no cgroup limit, or a unit limit above 32GiB): keep the original
    rule, at least 220GiB host memory and disk free and the unit below 360GB.
    Small container (cgroup ``memory.max`` at most 32GiB): the container limit is
    the binding constraint, so dispatch only while the cgroup uses under 70% of it;
    host memory must still exceed 8GiB and disk 50GiB.
    """
    if cgroup_max is not None and cgroup_max <= CONTAINER_LIMIT:
        return (
            cgroup_current is not None
            and cgroup_current < 0.7 * cgroup_max
            and host_available > 8 * 1024**3
            and disk_free > 50 * 1024**3
        )
    unit_ok = cgroup_current is None or cgroup_current < 360000000000
    return host_available > 220 * 1024**3 and disk_free > 220 * 1024**3 and unit_ok


def charged_memory(current: int, stat: str) -> int:
    """Non-reclaimable cgroup charge: ``memory.current`` minus file cache plus ``shmem``.

    Active and inactive file cache are both reclaimable under the container limit.
    Linux may keep files read or written by earlier batches in ``active_file`` for
    hours, which would otherwise hold a small container above the gate with almost
    no process memory. ``shmem`` is counted inside ``file`` but cannot be reclaimed,
    so it is added back (the E15 rule).
    """
    fields = {}
    for line in stat.splitlines():
        parts = line.split()
        if len(parts) == 2:
            fields[parts[0]] = int(parts[1])
    return max(0, current - fields.get("file", 0) + fields.get("shmem", 0))


def resources_available(root: Path) -> bool:
    """Read host memory, disk and this process's cgroup, then apply :func:`memory_gate`.

    In a small container (limit at most 32GiB) the cgroup charge excludes reclaimable
    file cache (:func:`charged_memory`); the shared-host rule reads ``memory.current``.
    """
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
    current = limit = None
    stat = ""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            base = Path("/sys/fs/cgroup" + line[3:])
            if (base / "memory.current").exists():
                current = int((base / "memory.current").read_text())
            if (base / "memory.max").exists():
                text = (base / "memory.max").read_text().strip()
                limit = None if text == "max" else int(text)
            if (base / "memory.stat").exists():
                stat = (base / "memory.stat").read_text()
    if current is not None and limit is not None and limit <= CONTAINER_LIMIT and stat:
        current = charged_memory(current, stat)
    return memory_gate(available, shutil.disk_usage(root).free, current, limit)


def slot_limit(root: Path, slots: int) -> int:
    """Active-slot cap read from ``root/parallel.limit`` at every dispatch tick.

    Absent file: every slot (historical behaviour). A readable integer is clamped
    to ``[0, slots]``; unreadable or non-integer content gives 0, so a damaged file
    pauses new starts instead of over-committing memory. Running workers are never
    touched; lowering the cap only stops new starts until enough slots finish.
    """
    path = root / "parallel.limit"
    if not path.exists():
        return slots
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError):
        return 0
    return max(0, min(slots, value))


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
        resources = resources_available(root)
        limit = slot_limit(root, len(cpus))
        for slot, cpu in enumerate(cpus):
            if slot in active or stopped or not resources or not pending:
                continue
            if len(active) >= limit:
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
                limit=limit,
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
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--exclude-cpus", default="")
    args = parser.parse_args()
    assert args.root.resolve() in {(ROOT / name).resolve() for name in ("formal", "pilot", "long")}
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
    import run_slot_assignment as runner

    assert not subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    ).strip()
    allowed = set(os.sched_getaffinity(0))
    topology = {
        c: tuple(
            (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
            for f in ("physical_package_id", "core_id")
        )
        for c in allowed
    }
    excluded = frozenset(int(c) for c in args.exclude_cpus.split(",") if c.strip())
    cpus = cpu_order(allowed, topology, args.parallel, excluded)
    print(
        json.dumps(dispatch(args.root, runner, cpus, SOURCE / "scripts/run_slot_assignment.py")),
        flush=True,
    )


if __name__ == "__main__":
    main()
