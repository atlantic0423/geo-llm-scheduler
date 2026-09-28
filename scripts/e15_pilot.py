"""Independent 50/100-job wall-clock and 24-versus-32 worker pilot."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json, free_disk_gb


def run_pilot(root: Path, small: bool = False) -> dict:
    """Freeze worker count, T50 and T100 before any formal outcome is inspected."""
    from run_e15_campaign import _config, prepare, run_batch

    manifest = prepare(root, small=small)
    plan_file = root / "resource_plan.json"
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if plan["source_commit"] != manifest["commit"]:
            raise ValueError("Frozen pilot belongs to another source commit")
        return plan
    pilot_specs = []
    for label, jobs in (("P50", 50), ("P100", 100)):
        key = f"{label}9001H"
        item = manifest["instances"][key]
        start = manifest["initial"][f"{key}:101"]
        for arm in ("M0", "N0", "F6"):
            config = _config(item["path"], 101, arm, label, {}, None)
            if small:
                config = replace(config, generations=2)
            pilot_specs.append(
                JobSpec(
                    label,
                    arm,
                    90010,
                    101,
                    asdict(config),
                    item["path"],
                    item["hash"],
                    start["path"],
                    start["hash"],
                    manifest["source_hash"],
                    manifest["commit"],
                )
            )
    if not run_batch(root, pilot_specs, 6, "pilot_wallclock"):
        raise RuntimeError("Wall-clock pilot failed")
    runtimes = {}
    for spec in pilot_specs:
        summary = json.loads((root / "runs" / spec.key / "summary.json").read_text())
        runtimes[f"{spec.stage}:{spec.arm}"] = summary["elapsed"]
    t50 = math.ceil(1.1 * max(v for k, v in runtimes.items() if k.startswith("P50:")))
    t100 = math.ceil(1.1 * max(v for k, v in runtimes.items() if k.startswith("P100:")))
    key = "P509001H"
    item = manifest["instances"][key]
    start = manifest["initial"][f"{key}:101"]
    activation_config = replace(
        _config(item["path"], 101, "C2", "P50", {}, None), generations=2 if small else 20
    )
    activation = JobSpec(
        "ACT",
        "C2",
        90010,
        101,
        asdict(activation_config),
        item["path"],
        item["hash"],
        start["path"],
        start["hash"],
        manifest["source_hash"],
        manifest["commit"],
    )
    if not run_batch(root, [activation], 1, "pilot_coverage_activation"):
        raise RuntimeError("Coverage activation pilot failed")
    activation_summary = json.loads((root / "runs" / activation.key / "summary.json").read_text())
    activation_budgets = activation_summary["budget_counts"]
    if not small and not all(activation_budgets.get(str(n), 0) > 0 for n in (3, 6, 10)):
        raise RuntimeError(f"Coverage-v2 3/6/10 activation missing: {activation_budgets}")
    # A short identical workload measures aggregate throughput, not research quality.
    throughput = {}
    resources = {}
    for count in (24, 32):
        key = "P509001H"
        item = manifest["instances"][key]
        start = manifest["initial"][f"{key}:101"]
        specs = []
        for index in range(32):
            config = replace(
                _config(item["path"], 101, "M0", "P50", {}, None), generations=2 if small else 10
            )
            specs.append(
                JobSpec(
                    f"THR{count}",
                    f"W{index:02d}",
                    90010,
                    101,
                    asdict(config),
                    item["path"],
                    item["hash"],
                    start["path"],
                    start["hash"],
                    manifest["source_hash"],
                    manifest["commit"],
                )
            )
        began = time.perf_counter()
        if not run_batch(root, specs, count, f"pilot_throughput_{count}"):
            raise RuntimeError(f"{count}-worker pilot failed")
        throughput[str(count)] = 32 / max(time.perf_counter() - began, 1e-9)
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        resources[str(count)] = {
            "memory_gate_blocked_polls": status["memory_gate_blocked_polls"],
            "peak_working_set_fraction": status["peak_working_set_fraction"],
        }
    worker_count = (
        32
        if (
            throughput["32"] >= throughput["24"] * 1.10
            and resources["32"]["memory_gate_blocked_polls"] == 0
            and resources["32"]["peak_working_set_fraction"] < 0.70
        )
        else 24
    )
    plan = {
        "source_commit": manifest["commit"],
        "source_hash": manifest["source_hash"],
        "worker_count": worker_count,
        "memory_gate": 0.78,
        "T50_wallclock": t50,
        "T100_wallclock": t100,
        "pilot_runtime_seconds": runtimes,
        "coverage_activation_budget_counts": activation_budgets,
        "throughput_runs_per_second": throughput,
        "throughput_resource_evidence": resources,
        "disk_free_gb_after_pilot": free_disk_gb(root),
        "disk_plan": "compact summary all runs; full trace precommitted 1/9 stratum; 30 GiB guard",
        "created_epoch": time.time(),
        "small": small,
    }
    atomic_json(plan_file, plan)
    return plan
