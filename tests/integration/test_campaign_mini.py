"""A real tiny campaign automatically unlocks every mandatory stage and E13."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.model import CampaignSettings
from geo_llm_scheduler.experiments.campaign.scheduler import Supervisor
from geo_llm_scheduler.experiments.campaign.support import atomic_json


def test_mini_campaign_end_to_end():
    repo = Path(__file__).resolve().parents[2]
    campaign_parent = repo / "outputs" / "campaign_60h"
    campaign_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="test-mini-", dir=campaign_parent, ignore_cleanup_errors=True
    ) as directory:
        root = Path(directory)
        settings = replace(
            CampaignSettings(),
            duration_hours=0.05,
            grace_minutes=1,
            heartbeat_seconds=1,
            dependency_poll_seconds=1,
            screening_seconds=0.2,
            final_seconds=0.2,
            jobs_per_instance=3,
            population=6,
            screening_instance_seeds=(71,),
            final_instance_seeds=(71,),
            algorithm_seeds=(101,),
            max_workers=2,
            memory_reserve_gb=0.1,
            memory_per_worker_gb=0.3,
            min_free_disk_gb=0.01,
            e13_extra_seed_start=9999,
            e13_extra_seed_stop=9999,
            d03_raw_root="outputs/nonexistent_d03_raw_for_test",
        )
        now = time.time()
        atomic_json(
            root / "launch.json",
            {
                "campaign_id": root.name,
                "started_epoch": now,
                "deadline_epoch": now + settings.duration_hours * 3600,
                "settings": asdict(settings),
            },
        )
        log = root / "mini.log"
        with log.open("wb") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "geo_llm_scheduler.experiments.campaign.scheduler",
                    str(repo),
                    str(root),
                ],
                cwd=repo,
                stdout=handle,
                stderr=handle,
            )
            deadline = time.monotonic() + 65
            complete = False
            while time.monotonic() < deadline and process.poll() is None:
                database = root / "state.sqlite3"
                if database.exists():
                    with closing(sqlite3.connect(database, timeout=5)) as connection:
                        try:
                            rows = connection.execute(
                                "SELECT stage,state FROM jobs WHERE stage IN "
                                "('e13_moead','e13_nsga2','e13_full')"
                            ).fetchall()
                        except sqlite3.OperationalError:
                            rows = []
                    complete = len(rows) == 3 and all(state == "SUCCEEDED" for _, state in rows)
                    if complete:
                        atomic_json(root / "stop.request", {"test": True})
                        break
                time.sleep(0.2)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        error_log = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        assert complete, error_log
        assert process.returncode == 0, error_log
        assert (root / "locks" / "final_full_config.lock.json").is_file()
        assert (root / "e14_threshold" / "e14_offline_shortlist.json").is_file()
        assert (root / "e13_hv_igd.csv").is_file()
        assert json.loads((root / "campaign_summary.json").read_text())["e13_paired_cells"] == 1
        report = (root / "reports" / "FINAL_CAMPAIGN_REPORT.md").read_text(encoding="utf-8")
        assert "分阶段完成情况" in report
        assert "下游临时选择与回退" in report
        assert "并非统计显著性结论" in report


def test_mini_campaign_supervisor_in_process():
    """Exercise the real dependency DAG while measuring supervisor code coverage."""
    repo = Path(__file__).resolve().parents[2]
    campaign_parent = repo / "outputs" / "campaign_60h"
    campaign_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="test-mini-inprocess-", dir=campaign_parent, ignore_cleanup_errors=True
    ) as directory:
        root = Path(directory)
        settings = replace(
            CampaignSettings(),
            duration_hours=0.05,
            grace_minutes=1,
            heartbeat_seconds=1,
            dependency_poll_seconds=1,
            screening_seconds=0.2,
            final_seconds=0.2,
            jobs_per_instance=3,
            population=6,
            screening_instance_seeds=(71,),
            final_instance_seeds=(71,),
            algorithm_seeds=(101,),
            max_workers=2,
            memory_reserve_gb=0.1,
            memory_per_worker_gb=0.3,
            min_free_disk_gb=0.01,
            e13_extra_seed_start=9999,
            e13_extra_seed_stop=9999,
            d03_raw_root="outputs/nonexistent_d03_raw_for_test",
        )
        now = time.time()
        atomic_json(
            root / "launch.json",
            {
                "campaign_id": root.name,
                "started_epoch": now,
                "deadline_epoch": now + settings.duration_hours * 3600,
                "settings": asdict(settings),
            },
        )
        with ThreadPoolExecutor(max_workers=1) as pool:

            def run_supervisor() -> None:
                Supervisor(repo, root).run()

            future = pool.submit(run_supervisor)
            deadline = time.monotonic() + 65
            complete = False
            while time.monotonic() < deadline and not future.done():
                database = root / "state.sqlite3"
                if database.exists():
                    with closing(sqlite3.connect(database, timeout=5)) as connection:
                        try:
                            rows = connection.execute(
                                "SELECT stage,state FROM jobs WHERE stage IN "
                                "('e13_moead','e13_nsga2','e13_full')"
                            ).fetchall()
                        except sqlite3.OperationalError:
                            rows = []
                    complete = len(rows) == 3 and all(state == "SUCCEEDED" for _, state in rows)
                    if complete:
                        atomic_json(root / "stop.request", {"test": True})
                        break
                time.sleep(0.2)
            if not complete:
                atomic_json(root / "stop.request", {"test": True})
            future.result(timeout=30)
        assert complete
        assert json.loads((root / "campaign_summary.json").read_text())["e13_paired_cells"] == 1
