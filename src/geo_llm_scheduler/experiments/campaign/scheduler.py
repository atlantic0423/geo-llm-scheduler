"""Dependency-aware supervisor with durable jobs, retries and provisional freezes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.experiments.campaign.aggregation import final_aggregate, stage_aggregate
from geo_llm_scheduler.experiments.campaign.model import CampaignSettings, JobSpec
from geo_llm_scheduler.experiments.campaign.planning import (
    config_for_arm,
    freeze_initial,
    freeze_inputs,
    make_spec,
    resource_pilot,
)
from geo_llm_scheduler.experiments.campaign.state_store import TERMINAL, StateStore
from geo_llm_scheduler.experiments.campaign.support import (
    alive,
    atomic_json,
    available_memory_gb,
    free_disk_gb,
    safe_worker_count,
)
from geo_llm_scheduler.experiments.campaign.thresholds import offline_shortlist
from geo_llm_scheduler.experiments.campaign.validation import file_hash, validate_result

STAGES = (
    "e14_online",
    "e11_trigger",
    "e12_delta",
    "e09_search",
    "e06_macrosearch",
    "e01_controller",
    "e04_polish",
    "e13_full",
)
SELECTION_FILES = {
    "e14_online": "selected_thresholds.json",
    "e11_trigger": "selected_trigger_mode.json",
    "e12_delta": "selected_trigger_delta.json",
    "e09_search": "selected_l_rl.json",
    "e06_macrosearch": "selected_macrosearch.json",
    "e01_controller": "selected_controller.json",
    "e04_polish": "selected_polish_mode.json",
}
FALLBACKS = {
    "e14_online": "baseline",
    "e11_trigger": "preference",
    "e12_delta": "0.1",
    "e09_search": "5",
    "e06_macrosearch": "fixed",
    "e01_controller": "qlearning",
    "e04_polish": "trajectory",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stage_arms(stage: str, root: Path, settings: CampaignSettings) -> tuple[str, ...]:
    """Expose the frozen arm list for one approved experiment stage."""
    if stage == "e14_online":
        data = json.loads((root / "e14_threshold" / "e14_offline_shortlist.json").read_text())
        return tuple(data["candidate_thresholds"])
    if stage == "e11_trigger":
        return "fixed", "strict", "preference"
    if stage == "e12_delta":
        return "0", "0.05", "0.1", "0.2", "0.5", "off"
    if stage == "e09_search":
        return "3", "5", "10"
    if stage == "e06_macrosearch":
        return (
            ("fixed", "severity", "coverage", "static", "random")
            if (settings.include_optional_macro_arms)
            else ("fixed", "severity", "coverage")
        )
    if stage == "e01_controller":
        return "random", "bandit", "qlearning"
    if stage == "e04_polish":
        return "none", "step", "trajectory"
    if stage == "e13_full":
        return ("full",)
    raise ValueError(f"Unknown stage: {stage}")


def _selected(root: Path) -> dict[str, Any]:
    selections = root / "selections"
    result: dict[str, Any] = {}
    mapping = {
        "selected_thresholds.json": "thresholds",
        "selected_trigger_mode.json": "trigger_mode",
        "selected_trigger_delta.json": "trigger_delta",
        "selected_l_rl.json": "rl_steps",
        "selected_macrosearch.json": "budget_policy",
        "selected_controller.json": "controller",
        "selected_polish_mode.json": "polish_mode",
    }
    for name, key in mapping.items():
        path = selections / name
        if path.exists():
            result[key] = json.loads(path.read_text(encoding="utf-8"))["value"]
    if result.get("trigger_delta") == "off":
        result["trigger_delta"] = 0.1
        result["trigger_quality_gate"] = False
    return result


def _config_from_json(data: dict[str, Any]) -> Config:
    for key in (
        "mutation_weights",
        "severity_thresholds",
        "budgets",
        "static_budgets",
        "enabled_operators",
    ):
        data[key] = tuple(data[key])
    return Config(**data)


class Supervisor:
    """Own all state transitions; workers only write isolated run artifacts."""

    def __init__(self, repo: Path, root: Path):
        self.repo, self.root = repo, root
        self.settings = CampaignSettings(**_settings_from_manifest(root))
        self.store = StateStore(root / "state.sqlite3")
        self.manifest: dict[str, Any] = {}
        self.resource_plan: dict[str, Any] = {}
        self.children: dict[str, subprocess.Popen[bytes]] = {}
        self.deadline_utc = float(json.loads((root / "launch.json").read_text())["deadline_epoch"])
        self.monotonic_deadline = time.monotonic() + max(0, self.deadline_utc - time.time())
        self.grace_end = self.monotonic_deadline + 60 * self.settings.grace_minutes
        self.last_heartbeat = 0.0
        self.verified_successes: set[str] = set()

    def _heartbeat(self, stage: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_heartbeat < min(5, self.settings.heartbeat_seconds):
            return
        counts: dict[str, int] = {}
        for row in self.store.rows():
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        atomic_json(
            self.root / "heartbeat.json",
            {
                "updated_utc": _utc(),
                "updated_epoch": time.time(),
                "supervisor_pid": os.getpid(),
                "stage": stage,
                "counts": counts,
                "deadline_utc": datetime.fromtimestamp(self.deadline_utc, timezone.utc).isoformat(),
                "queued_or_running": sum(counts.get(k, 0) for k in ("READY", "RUNNING")),
                "free_disk_gb": round(free_disk_gb(self.repo), 2),
                "available_memory_gb": round(available_memory_gb(), 2),
            },
        )
        self.last_heartbeat = now

    def _preflight(self) -> None:
        self._heartbeat("stage00_preflight")
        self.manifest = freeze_inputs(self.repo, self.root, self.settings)
        self.resource_plan = resource_pilot(self.repo, self.root, self.settings)
        atomic_json(
            self.root / "stage00_preflight" / "stage00_preflight.done.json",
            {
                "status": "SUCCEEDED",
                "source_commit": self.manifest["source_commit"],
                "source_hash": self.manifest["source_hash"],
                "settings_hash": self.manifest["settings_hash"],
                "resource_plan": "resource_plan.json",
                "time_utc": _utc(),
            },
        )
        for stage, arm in (("e13_moead", "moead"), ("e13_nsga2", "nsga2")):
            self._queue_stage(stage, (arm,))
        self._offline()
        self._heartbeat("stage00_ready", force=True)

    def _offline(self) -> None:
        path = self.root / "e14_threshold" / "e14_offline_shortlist.json"
        if path.exists():
            return
        try:
            data = offline_shortlist(self.repo / self.settings.d03_raw_root)
        except (OSError, ValueError) as error:
            data = {
                "status": "fallback_no_usable_d03_raw",
                "reason": str(error),
                "candidate_thresholds": {"baseline": [0.2] * 6},
                "statistically_concluded": False,
            }
        atomic_json(path, data)

    def _queue_stage(self, stage: str, arms: tuple[str, ...]) -> None:
        selected = _selected(self.root)
        base = load_config(self.repo / self.settings.base_config)
        cells = (
            self.manifest["final_cells"]
            if stage.startswith("e13_")
            else self.manifest["screening_cells"]
        )
        threshold_data = (
            json.loads(
                (self.root / "e14_threshold" / "e14_offline_shortlist.json").read_text(
                    encoding="utf-8"
                )
            )
            if stage == "e14_online"
            else None
        )
        for instance_seed, algorithm_seed in cells:
            for arm in arms:
                if stage == "e13_full":
                    lock = json.loads(
                        (self.root / "locks" / "final_full_config.lock.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    config = _config_from_json(dict(lock["config"]))
                elif stage == "e13_moead":
                    config = replace(base, method="plain", polish=False)
                elif stage == "e13_nsga2":
                    config = replace(base, method="nsga2", polish=False)
                else:
                    config = config_for_arm(
                        base,
                        stage,
                        arm,
                        selected,
                        threshold_data["candidate_thresholds"] if threshold_data else None,
                    )
                    if stage in ("e11_trigger", "e12_delta", "e09_search"):
                        config = replace(config, budget_policy="fixed", fixed_budget=6)
                config = replace(
                    config,
                    population=self.settings.population,
                    neighborhood=min(config.neighborhood, self.settings.population),
                    generations=100000,
                    seconds=(
                        self.settings.final_seconds
                        if stage.startswith("e13_")
                        else self.settings.screening_seconds
                    ),
                    seed=algorithm_seed,
                    instance=self.manifest["instances"][str(instance_seed)]["path"],
                    output=(self.root / stage / "runs").relative_to(self.repo).as_posix(),
                )
                spec = make_spec(self.manifest, stage, arm, instance_seed, algorithm_seed, config)
                self.store.add(spec)
                atomic_json(self.root / "specs" / f"{spec.key}.json", asdict(spec))

    def _stage_done(self, stage: str) -> bool:
        rows = self.store.rows(stage)
        return bool(rows) and all(row["state"] in TERMINAL for row in rows)

    def _publish_selection(self, stage: str) -> None:
        output = self.root / "selections" / SELECTION_FILES[stage]
        if output.exists():
            return
        arms = stage_arms(stage, self.root, self.settings)
        summary = stage_aggregate(self.root, stage, arms, FALLBACKS[stage], self.store)
        selected = summary["selected"]
        value: Any = selected["selected_arm"]
        if stage == "e14_online":
            data = json.loads(
                (self.root / "e14_threshold" / "e14_offline_shortlist.json").read_text()
            )
            value = data["candidate_thresholds"][value]
        elif stage == "e12_delta" and value != "off":
            value = float(value)
        elif stage == "e09_search":
            value = int(value)
        atomic_json(
            output,
            {
                "stage": stage,
                "value": value,
                "selected_arm": selected["selected_arm"],
                "fallback_used": selected["fallback_used"],
                "ambiguous": selected["ambiguity"],
                "reason": selected["reason"],
                "evidence": f"{stage}/stage_summary.json",
                "time_utc": _utc(),
                "provisional_downstream_only": True,
            },
        )

    def _freeze_final(self) -> None:
        path = self.root / "locks" / "final_full_config.lock.json"
        if path.exists():
            return
        selected = _selected(self.root)
        config = config_for_arm(
            load_config(self.repo / self.settings.base_config), "e13_full", "full", selected
        )
        evidence = {stage: f"selections/{name}" for stage, name in SELECTION_FILES.items()}
        fallback = {
            stage: json.loads((self.root / evidence[stage]).read_text())["fallback_used"]
            for stage in SELECTION_FILES
        }
        lock = {
            "campaign_id": self.root.name,
            "source_commit": self.manifest["source_commit"],
            "source_hash": self.manifest["source_hash"],
            "config": asdict(config),
            "selected_parameters": selected,
            "selection_evidence": evidence,
            "fallback_flags": fallback,
            "selection_time_utc": _utc(),
            "provisional_downstream_only": True,
        }
        atomic_json(path, lock)
        atomic_json(self.root / "locks" / "final_full_config.json", asdict(config))
        import yaml

        (self.root / "locks" / "final_full_config.yaml").write_text(
            yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=True), encoding="utf-8"
        )

    def _advance_dag(self) -> str:
        self._offline()
        previous: str | None = None
        for stage in STAGES:
            if (
                previous is not None
                and not (self.root / "selections" / SELECTION_FILES[previous]).exists()
            ):
                return f"waiting:{previous}"
            if stage == "e12_delta" and _selected(self.root).get("trigger_mode") == "fixed":
                skip = self.root / "selections" / "e12_skipped.json"
                if not skip.exists():
                    atomic_json(
                        skip, {"reason": "selected fixed trigger has no scalar-quality gate"}
                    )
                    atomic_json(
                        self.root / "selections" / SELECTION_FILES[stage],
                        {
                            "stage": stage,
                            "value": 0.1,
                            "selected_arm": "not_applicable",
                            "fallback_used": False,
                            "ambiguous": False,
                            "reason": "not applicable to fixed trigger",
                            "time_utc": _utc(),
                        },
                    )
                previous = stage
                continue
            if stage == "e13_full":
                if not (self.root / "locks" / "final_full_config.lock.json").exists():
                    self._freeze_final()
            arms = stage_arms(stage, self.root, self.settings)
            if not self.store.rows(stage):
                self._queue_stage(stage, arms)
                return stage
            if stage != "e13_full":
                if not self._stage_done(stage):
                    return stage
                self._publish_selection(stage)
            elif not self._stage_done(stage):
                return stage
            previous = stage
        return "e13_core_complete"

    def _quarantine(self, key: str) -> None:
        row = self.store.get(key)
        output = self.root / row["stage"] / "runs" / key
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        for path in (
            output,
            output.with_suffix(".failure.json"),
            *output.parent.glob(f".{key}.attempt-*"),
        ):
            if path.exists():
                destination = self.root / "quarantine" / f"{key}__{stamp}__{path.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))

    @staticmethod
    def _stop_child(child: subprocess.Popen[bytes]) -> int | None:
        """Reap a worker before touching its artifact directory."""
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        else:
            child.wait()
        return child.returncode

    def _fail(self, row: dict[str, Any], reason: str, exit_code: int | None) -> None:
        child = self.children.pop(row["key"], None)
        if child is not None:
            exit_code = self._stop_child(child)
        self._quarantine(row["key"])
        attempts = row["attempt"]
        if attempts <= self.settings.max_job_retries:
            self.store.transition(
                row["key"],
                "FAILED_RETRYABLE",
                pid=None,
                slot=None,
                reason=reason,
                exit_code=exit_code,
                ready_after=time.time() + (60 if attempts == 1 else 180),
            )
        else:
            self.store.transition(
                row["key"],
                "FAILED_PERMANENT",
                pid=None,
                slot=None,
                reason=reason,
                exit_code=exit_code,
            )

    def _reconcile(self) -> None:
        for row in self.store.rows():
            if row["state"] == "SUCCEEDED" and row["key"] in self.verified_successes:
                continue
            spec = JobSpec(**json.loads(row["spec_json"]))
            output = self.root / row["stage"] / "runs" / row["key"]
            valid, reason = validate_result(output, spec)
            if row["state"] == "SUCCEEDED":
                if row["key"] not in self.verified_successes:
                    if not valid:
                        self._quarantine(row["key"])
                        self.store.transition(
                            row["key"],
                            "READY",
                            pid=None,
                            slot=None,
                            reason=f"success artifact invalidated:{reason}",
                        )
                    else:
                        self.verified_successes.add(row["key"])
                continue
            if valid:
                child = self.children.get(row["key"])
                if child is not None:
                    try:
                        completed_exit_code = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        continue
                    self.children.pop(row["key"], None)
                    if completed_exit_code != 0:
                        self._fail(row, "worker-nonzero-exit-after-artifact", completed_exit_code)
                        continue
                elif row["state"] == "RUNNING" and alive(row["pid"]):
                    # A previous supervisor launched this worker. Its exit code is
                    # unavailable here, so wait for process death and the marker.
                    continue
                self.store.transition(
                    row["key"],
                    "SUCCEEDED",
                    pid=None,
                    slot=None,
                    elapsed=json.loads((output / "summary.json").read_text())["elapsed"],
                    result_hash=file_hash(output / "summary.json"),
                    reason="validated",
                )
                self.verified_successes.add(row["key"])
                continue
            if row["state"] == "RUNNING":
                process = self.children.get(row["key"])
                exit_code = process.poll() if process else None
                budget = JobSpec(**json.loads(row["spec_json"])).config["seconds"]
                if time.time() - row["updated_at"] > max(3 * budget, budget + 900):
                    self._fail(row, "worker-hard-timeout", exit_code)
                    continue
                if exit_code is None and alive(row["pid"]):
                    continue
                self._fail(row, f"worker-exit-or-incomplete:{reason}", exit_code)
            elif row["state"] == "FAILED_RETRYABLE" and time.time() >= row["ready_after"]:
                self.store.transition(row["key"], "READY", pid=None, slot=None)
            elif row["state"] == "READY" and output.exists():
                self._quarantine(row["key"])

    def _launch(self, row: dict[str, Any], slot: int) -> None:
        spec_path = self.root / "specs" / f"{row['key']}.json"
        output = self.root / row["stage"] / "runs" / row["key"]
        log = self.root / "logs" / "workers" / f"{row['key']}.attempt{row['attempt'] + 1}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        with log.open("ab", buffering=0) as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "geo_llm_scheduler.experiments.campaign.worker",
                    str(spec_path),
                    str(output),
                ],
                cwd=self.repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                close_fds=True,
            )
        if sys.platform == "win32":
            import ctypes

            pinned = ctypes.windll.kernel32.SetProcessAffinityMask(  # type: ignore[attr-defined]
                int(process._handle),  # type: ignore[attr-defined]
                1 << (slot * 2),
            )
            if not pinned:
                process.terminate()
                process.wait(timeout=10)
                raise OSError("Could not enforce equal one-CPU worker affinity")
        elif hasattr(os, "sched_setaffinity"):
            allowed = sorted(os.sched_getaffinity(0))
            if slot >= len(allowed):
                process.terminate()
                process.wait(timeout=10)
                raise OSError("Insufficient allowed CPUs for pinned worker slot")
            try:
                os.sched_setaffinity(process.pid, {allowed[slot]})
            except OSError:
                process.terminate()
                process.wait(timeout=10)
                raise
        self.children[row["key"]] = process
        self.store.transition(
            row["key"],
            "RUNNING",
            pid=process.pid,
            slot=slot,
            attempt=row["attempt"] + 1,
            reason=None,
            ready_after=0,
        )

    def _fill_workers(self) -> None:
        if time.monotonic() >= self.monotonic_deadline:
            return
        if free_disk_gb(self.repo) < self.settings.min_free_disk_gb:
            atomic_json(
                self.root / "disk_incident.json",
                {
                    "time_utc": _utc(),
                    "reason": "low disk; paused new workers",
                    "free_gb": free_disk_gb(self.repo),
                },
            )
            return
        plan_workers = int(self.resource_plan["workers"])
        allowed_cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        current_cap = safe_worker_count(
            allowed_cpus,
            available_memory_gb(),
            self.settings.memory_reserve_gb,
            float(self.resource_plan["worker_memory_budget_gb"]),
            min(plan_workers, self.settings.max_workers),
        )
        running = self.store.rows()
        occupied_slots = {row["slot"] for row in running if row["state"] == "RUNNING"}
        slots = [i for i in range(current_cap) if i not in occupied_slots]
        ready = [row for row in running if row["state"] == "READY"]
        priorities = {
            "e14_online": 0,
            "e11_trigger": 1,
            "e12_delta": 2,
            "e09_search": 3,
            "e06_macrosearch": 4,
            "e01_controller": 5,
            "e04_polish": 6,
            "e13_full": 7,
            "e13_moead": 8,
            "e13_nsga2": 8,
            "e13_extra": 9,
        }
        ready.sort(key=lambda row: (priorities.get(row["stage"], 20), row["key"]))
        recent = [
            row["elapsed"]
            for row in running
            if row["state"] == "SUCCEEDED" and row["elapsed"] is not None
        ]
        p90 = sorted(recent)[int(0.9 * (len(recent) - 1))] if recent else 0.0
        if len(slots) >= 2:
            baseline_progress = {
                stage: sum(
                    row["stage"] == stage and row["state"] in ("RUNNING", "SUCCEEDED")
                    for row in running
                )
                for stage in ("e13_moead", "e13_nsga2")
            }
            baselines = [row for row in ready if row["stage"] in baseline_progress]
            baseline = min(
                baselines,
                key=lambda row: (baseline_progress[row["stage"]], row["key"]),
                default=None,
            )
            selector = next(
                (row for row in ready if row["stage"] not in ("e13_moead", "e13_nsga2")), None
            )
            if baseline and selector:
                ready = [baseline, selector] + [
                    row for row in ready if row not in (baseline, selector)
                ]
        for slot, row in zip(slots, ready):
            spec = JobSpec(**json.loads(row["spec_json"]))
            estimate = max(float(spec.config["seconds"]), p90)
            if self.monotonic_deadline - time.monotonic() < 1.2 * estimate + 60:
                continue
            self._launch(row, slot)

    def _extra_e13(self) -> None:
        if not all(self._stage_done(stage) for stage in ("e13_moead", "e13_nsga2", "e13_full")):
            return
        if time.monotonic() + 3 * self.settings.final_seconds >= self.monotonic_deadline:
            return
        if any(row["state"] not in TERMINAL for row in self.store.rows("e13_extra")):
            return
        used = {row["algorithm_seed"] for row in self.store.rows("e13_extra")}
        seed = next(
            (
                s
                for s in range(
                    self.settings.e13_extra_seed_start, self.settings.e13_extra_seed_stop + 1
                )
                if s not in used
            ),
            None,
        )
        if seed is None:
            return
        base = load_config(self.repo / self.settings.base_config)
        lock = json.loads((self.root / "locks" / "final_full_config.lock.json").read_text())
        for instance_seed in self.settings.final_instance_seeds:
            instance = self.manifest["instances"][str(instance_seed)]
            initial_path, initial_hash = freeze_initial(
                self.repo,
                self.root,
                instance_seed,
                seed,
                self.repo / instance["path"],
                self.settings,
            )
            self.manifest["initial_populations"][f"{instance_seed}:{seed}"] = {
                "path": initial_path,
                "sha256": initial_hash,
            }
            for arm in ("moead", "nsga2", "full"):
                if arm == "full":
                    config = _config_from_json(dict(lock["config"]))
                else:
                    config = replace(
                        base, method="plain" if arm == "moead" else "nsga2", polish=False
                    )
                config = replace(
                    config,
                    population=self.settings.population,
                    neighborhood=min(config.neighborhood, self.settings.population),
                    generations=100000,
                    seconds=self.settings.final_seconds,
                    seed=seed,
                    instance=instance["path"],
                    output=(self.root / "e13_extra" / "runs").relative_to(self.repo).as_posix(),
                )
                spec = make_spec(self.manifest, "e13_extra", arm, instance_seed, seed, config)
                self.store.add(spec)
                atomic_json(self.root / "specs" / f"{spec.key}.json", asdict(spec))
        atomic_json(self.root / "campaign_manifest.json", self.manifest)

    def run(self) -> None:
        """Run until deadline/stop, then aggregate and release the watchdog."""
        try:
            self._preflight()
            while True:
                self._reconcile()
                stopping = (self.root / "stop.request").exists()
                stage = "stopping" if stopping else self._advance_dag()
                self._heartbeat(stage)
                if stopping:
                    if not any(row["state"] == "RUNNING" for row in self.store.rows()):
                        break
                elif time.monotonic() < self.monotonic_deadline:
                    if stage == "e13_core_complete":
                        self._extra_e13()
                    self._fill_workers()
                else:
                    for row in self.store.rows():
                        if row["state"] in ("READY", "FAILED_RETRYABLE"):
                            self.store.transition(row["key"], "CANCELLED_AT_DEADLINE")
                    if not any(row["state"] == "RUNNING" for row in self.store.rows()):
                        break
                    if time.monotonic() > self.grace_end:
                        for row in self.store.rows():
                            if row["state"] == "RUNNING":
                                child = self.children.get(row["key"])
                                if child:
                                    self._stop_child(child)
                                    self.children.pop(row["key"], None)
                                self.store.transition(
                                    row["key"], "CANCELLED_AT_DEADLINE", reason="grace exhausted"
                                )
                        break
                time.sleep(min(5, self.settings.heartbeat_seconds))
            finish_reason = (
                "stopped_by_request"
                if (self.root / "stop.request").exists()
                else "deadline_reached"
                if time.monotonic() >= self.monotonic_deadline
                else "complete"
            )
            result = final_aggregate(self.root, self.store, finish_reason)
            atomic_json(
                self.root / "finished.json",
                {
                    "time_utc": _utc(),
                    "status": result["status"],
                    "summary": "campaign_summary.json",
                },
            )
            self._heartbeat("finished")
        finally:
            self.store.close()


def _settings_from_manifest(root: Path) -> dict[str, Any]:
    data = json.loads((root / "launch.json").read_text(encoding="utf-8"))["settings"]
    return {
        **data,
        "screening_instance_seeds": tuple(data["screening_instance_seeds"]),
        "final_instance_seeds": tuple(data["final_instance_seeds"]),
        "algorithm_seeds": tuple(data["algorithm_seeds"]),
    }


def main() -> None:
    """Supervisor process entry point used only by the detached watchdog."""
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python -m geo_llm_scheduler.experiments.campaign.scheduler REPO ROOT"
        )
    Supervisor(Path(sys.argv[1]), Path(sys.argv[2])).run()


if __name__ == "__main__":
    main()
