"""Detached watchdog that restarts a failed supervisor without terminal ownership."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.support import (
    atomic_json,
    detached_process,
    keep_awake,
)


def watch(repo: Path, root: Path) -> None:
    """Maintain one supervisor until its final marker or a bounded fatal restart limit."""
    atomic_json(
        root / "watchdog.pid.json",
        {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()},
    )
    try:
        keep_awake(True)
    except (OSError, AttributeError) as error:
        atomic_json(
            root / "keep_awake_warning.json",
            {"time_utc": datetime.now(timezone.utc).isoformat(), "reason": str(error)},
        )
    failures = 0
    try:
        while not (root / "finished.json").exists():
            log = root / "logs" / "supervisor.log"
            process = detached_process(
                [
                    sys.executable,
                    "-m",
                    "geo_llm_scheduler.experiments.campaign.scheduler",
                    str(repo),
                    str(root),
                ],
                repo,
                log,
            )
            atomic_json(
                root / "supervisor.pid.json",
                {
                    "pid": process.pid,
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "restart_count": failures,
                },
            )
            while process.poll() is None:
                heartbeat = root / "heartbeat.json"
                if heartbeat.exists():
                    try:
                        data = json.loads(heartbeat.read_text(encoding="utf-8"))
                        if data.get("supervisor_pid") == process.pid and (
                            time.time() - data["updated_epoch"] > 180
                        ):
                            process.terminate()
                            break
                    except (OSError, ValueError, KeyError):
                        pass
                time.sleep(10)
            if (root / "finished.json").exists():
                break
            failures += 1
            atomic_json(
                root / "watchdog_state.json",
                {
                    "status": "restarting",
                    "attempt": failures,
                    "last_exit_code": process.poll(),
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if failures >= 10:
                atomic_json(
                    root / "watchdog_state.json",
                    {
                        "status": "fatal",
                        "reason": "supervisor restart limit exhausted",
                        "attempt": failures,
                    },
                )
                break
            time.sleep(min(180, 5 * 2 ** min(failures - 1, 5)))
    finally:
        try:
            keep_awake(False)
        except (OSError, AttributeError):
            pass
        active = repo / "outputs" / "campaign_60h" / "active.json"
        try:
            if json.loads(active.read_text(encoding="utf-8")).get("campaign_id") == root.name:
                active.unlink()
        except (FileNotFoundError, OSError, ValueError):
            pass


def main() -> None:
    """Read repository and campaign root from the parent CLI invocation."""
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python -m geo_llm_scheduler.experiments.campaign.watchdog REPO ROOT"
        )
    watch(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
