"""Single-command detached 60-hour campaign lifecycle CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.model import load_settings
from geo_llm_scheduler.experiments.campaign.support import (
    alive,
    atomic_json,
    detached_process,
    free_disk_gb,
    wait_for,
)

REPO = Path(__file__).resolve().parents[1]
CAMPAIGNS = REPO / "outputs" / "campaign_60h"
ACTIVE = CAMPAIGNS / "active.json"


def active_root() -> Path | None:
    """Resolve the active campaign, or the newest historical campaign for read-only status."""
    if ACTIVE.exists():
        data = json.loads(ACTIVE.read_text(encoding="utf-8"))
        return CAMPAIGNS / data["campaign_id"]
    launched: list[tuple[float, Path]] = []
    if CAMPAIGNS.exists():
        for path in CAMPAIGNS.glob("*/launch.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # In-process mini DAG tests also create launch.json, but are not
                # user-started campaigns and must not hijack status/resume.
                if "development_smoke" in data:
                    launched.append((float(data["started_epoch"]), path.parent))
            except (OSError, ValueError, KeyError):
                continue
    return max(launched, default=(0.0, None))[1]


def _git_clean_main() -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=REPO, text=True
    ).strip()
    if branch != "main" or status.strip() or head != remote:
        raise RuntimeError("Formal campaign requires clean main at origin/main merge SHA")


def _launch(root: Path) -> dict[str, object]:
    if (root / "finished.json").exists():
        raise RuntimeError("Completed campaign cannot be resumed")
    prior = root / "watchdog.pid.json"
    if prior.exists():
        pid = json.loads(prior.read_text(encoding="utf-8"))["pid"]
        if alive(pid):
            return status(root)
    supervisor = root / "supervisor.pid.json"
    if supervisor.exists() and alive(json.loads(supervisor.read_text(encoding="utf-8"))["pid"]):
        raise RuntimeError("Supervisor is still alive; refusing to create a second watchdog")
    process = detached_process(
        [
            sys.executable,
            "-m",
            "geo_llm_scheduler.experiments.campaign.watchdog",
            str(REPO),
            str(root),
        ],
        REPO,
        root / "logs" / "watchdog.log",
    )
    active = {"campaign_id": root.name, "watchdog_pid": process.pid}
    atomic_json(ACTIVE, active)
    ok = wait_for(
        root / "heartbeat.json",
        lambda data: (
            (root / "campaign_manifest.json").exists() and data.get("queued_or_running", 0) > 0
        ),
        20,
    )
    if not ok or not alive(process.pid) or not (root / "campaign_manifest.json").exists():
        raise RuntimeError(
            "Detached campaign did not reach verified heartbeat/manifest in 20 seconds; "
            f"inspect {root / 'logs' / 'watchdog.log'}"
        )
    return status(root)


def start(config_path: Path, development: bool = False) -> dict[str, object]:
    """Create one frozen campaign and verify an independently running watchdog."""
    settings = load_settings(config_path)
    if not development:
        _git_clean_main()
    if free_disk_gb(REPO) < settings.min_free_disk_gb:
        raise OSError("Not enough free disk for formal campaign")
    for other in CAMPAIGNS.glob("*/watchdog.pid.json"):
        if alive(json.loads(other.read_text(encoding="utf-8"))["pid"]):
            if not ACTIVE.exists() or other.parent.name != active_root().name:
                raise RuntimeError(f"Another active campaign watchdog exists: {other.parent.name}")
    if ACTIVE.exists():
        previous = active_root()
        if previous and not (previous / "finished.json").exists():
            return _launch(previous)
    campaign_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = CAMPAIGNS / campaign_id
    root.mkdir(parents=True, exist_ok=False)
    now = time.time()
    atomic_json(
        root / "launch.json",
        {
            "campaign_id": campaign_id,
            "started_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "started_epoch": now,
            "deadline_epoch": now + settings.duration_hours * 3600,
            "settings": asdict(settings),
            "development_smoke": development,
        },
    )
    return _launch(root)


def status(root: Path | None = None) -> dict[str, object]:
    """Read persistent process and progress files without changing campaign state."""
    target = root or active_root()
    if target is None:
        return {"status": "not_started"}
    result: dict[str, object] = {"campaign_id": target.name, "output_root": str(target)}
    for name, key in (
        ("watchdog.pid.json", "watchdog"),
        ("supervisor.pid.json", "supervisor"),
        ("heartbeat.json", "heartbeat"),
        ("launch.json", "launch"),
        ("finished.json", "finished"),
    ):
        path = target / name
        if path.exists():
            result[key] = json.loads(path.read_text(encoding="utf-8"))
    result["watchdog_alive"] = (
        alive(result.get("watchdog", {}).get("pid"))
        if (isinstance(result.get("watchdog"), dict))
        else False
    )
    result["supervisor_alive"] = (
        alive(result.get("supervisor", {}).get("pid"))
        if (isinstance(result.get("supervisor"), dict))
        else False
    )
    heartbeat = result.get("heartbeat")
    if isinstance(heartbeat, dict):
        counts = heartbeat.get("counts", {})
        result["current_stage"] = heartbeat.get("stage")
        result["heartbeat_age_s"] = round(
            max(0.0, time.time() - float(heartbeat["updated_epoch"])), 1
        )
        result["completed"] = counts.get("SUCCEEDED", 0)
        result["running"] = counts.get("RUNNING", 0)
        result["waiting"] = counts.get("READY", 0) + counts.get("FAILED_RETRYABLE", 0)
        result["failed"] = counts.get("FAILED_PERMANENT", 0)
    result["log"] = str(target / "logs" / "supervisor.log")
    return result


def stop() -> dict[str, object]:
    """Request a safe boundary stop; preserve all complete and partial artifacts."""
    root = active_root()
    if root is None or (root / "finished.json").exists():
        return {"status": "not_running"}
    atomic_json(root / "stop.request", {"time_utc": datetime.now(timezone.utc).isoformat()})
    return {"status": "stop_requested", "campaign_id": root.name}


def main() -> int:
    """Dispatch start, resume, status and stop commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "resume", "status", "stop"))
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--config", default="configs/campaigns/60h_freeze.yaml")
    parser.add_argument("--allow-development", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command in ("start", "resume") and not args.detach:
        parser.error("start/resume require --detach")
    if args.command == "start":
        result = start(REPO / args.config, args.allow_development)
    elif args.command == "resume":
        root = active_root()
        if root is None:
            raise RuntimeError("No prior campaign to resume")
        result = _launch(root)
    elif args.command == "stop":
        result = stop()
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
