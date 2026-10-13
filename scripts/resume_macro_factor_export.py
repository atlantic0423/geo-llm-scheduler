"""Resume frozen MacroSearch jobs with a bounded, serialized post-search export stage."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SEARCH_BYTES = 256 * 1024**2
EXPORT_BYTES = 1024**3


def export_memory(group: Path) -> tuple[int, int]:
    """Return free bytes and conservatively reclaimable clean inactive file pages."""
    free = int((group / "memory.max").read_text()) - int((group / "memory.current").read_text())
    stat = {
        key: int(value)
        for key, value in (
            line.split() for line in (group / "memory.stat").read_text().splitlines()
        )
    }
    clean_inactive = max(0, stat["inactive_file"] - stat["file_dirty"] - stat["file_writeback"])
    return free, clean_inactive


def install_child_limits(original: Callable[[], None]) -> None:
    """Keep the frozen search soft limit; reserve permission for post-search export only."""
    import resource

    setter = resource.setrlimit

    def set_limit(kind: int, limits: tuple[int, int]) -> None:
        if kind == resource.RLIMIT_AS and limits == (SEARCH_BYTES, SEARCH_BYTES):
            setter(kind, (SEARCH_BYTES, EXPORT_BYTES))
        else:
            setter(kind, limits)

    resource.setrlimit = set_limit
    try:
        original()
        assert resource.getrlimit(resource.RLIMIT_AS) == (SEARCH_BYTES, EXPORT_BYTES)
    finally:
        resource.setrlimit = setter


def worker_command(command: list[str], wrapper: Path, source: Path) -> list[str]:
    """Redirect only the exact frozen factor worker invocation to this audited wrapper."""
    if len(command) == 5 and command[1:3] == [
        "-m",
        "geo_llm_scheduler.experiments.macro_factor_worker",
    ]:
        return [command[0], str(wrapper), "worker", str(source), *command[3:]]
    return command


def export_gate(root: Path, key: str) -> Any:
    """Hold one shared export lease after the engine ends, respecting memory and watchdogs."""
    import fcntl
    import resource

    from geo_llm_scheduler.experiments.campaign.support import atomic_json

    if resource.getrlimit(resource.RLIMIT_AS) != (SEARCH_BYTES, EXPORT_BYTES):
        raise RuntimeError("Unexpected worker search limit")
    stream = (root.parent / "post_search_export.lock").open("a+b")
    started = time.time()
    spec = json.loads((root / "specs" / f"{key}.json").read_text())
    # Leave time for the unchanged final exact verification/export before its owned watchdog.
    max_wait = min(1800.0, spec["config"]["seconds"] * 2)
    while True:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            group = Path("/sys/fs/cgroup")
            available, reclaimable = export_memory(group)
            if available + reclaimable >= 1.5 * 1024**3:
                break
            fcntl.flock(stream, fcntl.LOCK_UN)
        except BlockingIOError:
            pass
        if time.time() - started > max_wait:
            stream.close()
            raise TimeoutError("Post-search export resource gate timed out")
        time.sleep(0.5)
    resource.setrlimit(resource.RLIMIT_AS, (EXPORT_BYTES, EXPORT_BYTES))
    atomic_json(
        root / "ops" / "post_search_exports" / f"{key}.json",
        {
            "pid": os.getpid(),
            "timestamp": time.time(),
            "search_soft_bytes": SEARCH_BYTES,
            "export_soft_bytes": EXPORT_BYTES,
            "gate_wait_seconds": time.time() - started,
            "available_bytes_before_export": available,
            "reclaimable_clean_inactive_file_bytes": reclaimable,
            "wrapper": str(Path(__file__).resolve()),
        },
    )
    return stream


def worker_main(source: Path, root: Path, key: str) -> None:
    """Execute unchanged frozen search, then unchanged exact/export under the export lease."""
    sys.path.insert(0, str(source / "src"))
    from geo_llm_scheduler.experiments import macro_factor_worker as worker

    original_run = worker.run
    original_argv = sys.argv
    lease = None

    def run_and_gate(*args: Any, **kwargs: Any) -> Any:
        nonlocal lease
        result = original_run(*args, **kwargs)
        lease = export_gate(root, key)
        return result

    worker.run = run_and_gate
    sys.argv = [str(Path(worker.__file__)), str(root), key]
    try:
        worker.main()
    finally:
        worker.run = original_run
        sys.argv = original_argv
        if lease is not None:
            lease.close()


def main() -> None:
    """Wrap only this experiment's workers; leave the frozen source and matrix untouched."""
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_main(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    wrapper = Path(__file__).resolve()
    original_popen = subprocess.Popen

    def spawn(command: list[str], *positional: Any, **kwargs: Any) -> Any:
        rewritten = worker_command(command, wrapper, source)
        if rewritten != command:
            original_limits = kwargs["preexec_fn"]
            kwargs["preexec_fn"] = lambda: install_child_limits(original_limits)
        return original_popen(rewritten, *positional, **kwargs)

    subprocess.Popen = spawn  # type: ignore[assignment]
    sys.path.insert(0, str(source / "src"))
    sys.argv = [
        str(source / "scripts/run_macro_factor_campaign.py"),
        "run",
        "--root",
        str(args.root.resolve()),
        "--workers",
        str(args.workers),
        "--deadline",
        str(args.deadline),
    ]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        subprocess.Popen = original_popen


if __name__ == "__main__":
    main()
