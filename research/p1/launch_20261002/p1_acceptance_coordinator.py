"""Coordinate isolated pilots, cross-version replay and the manifest-bound P1 gate."""

import hashlib
import json
import shutil
import time
from pathlib import Path

base = Path("/workspace/zhouhanyu")
control = base / "p1_acceptance_control_20261002"
roots = [base / f"campaign_p1_20261002_node{n}" for n in (0, 1)]
start = time.time()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def state(phase, **extra):
    atomic(
        control / "coordinator_state.json",
        {"phase": phase, "time": time.time(), "elapsed": time.time() - start, **extra},
    )


def request(node, phase, workers, tag):
    atomic(control / f"request_node{node}.json", {"phase": phase, "workers": workers, "tag": tag})


def wait(node, tag, maximum=16000):
    deadline = time.time() + maximum
    path = control / f"done_{tag}_node{node}.json"
    while not path.exists():
        if time.time() > deadline:
            raise RuntimeError(f"Acceptance wave timeout: {tag}")
        time.sleep(5)
    value = json.loads(path.read_text())
    if not value["success"]:
        raise RuntimeError(f"Acceptance wave failed: {tag}: {value}")
    return value


def batch(node, phase):
    return json.loads((roots[node] / "ops" / f"batch_{phase}.json").read_text())


try:
    for node in (0, 1):
        manifest_path = roots[node] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name in ("p1_acceptance_node.py", "p1_acceptance_coordinator.py"):
            shutil.copy2(base / name, roots[node] / "ops" / name)
        manifest["ops_coordinator_sha256"] = hashlib.sha256(
            (base / "p1_acceptance_coordinator.py").read_bytes()
        ).hexdigest()
        atomic(manifest_path, manifest)
    for node in (0, 1):
        for workers in (1, 2, 4, 8):
            tag = f"single{node}_thr{workers}"
            state("isolated_throughput", node=node, workers=workers, tag=tag)
            request(node, f"THR{workers}", workers, tag)
            wait(node, tag, 2400)
            record = batch(node, f"THR{workers}")
            if record["peak_fraction"] >= 0.70 or not record["success"]:
                raise RuntimeError("Single-node resource gate failed")
    state("dual_node_throughput", workers_per_node=8)
    for node in (0, 1):
        request(node, "DB8", 8, "dual8")
    for node in (0, 1):
        wait(node, "dual8", 2400)
        if batch(node, "DB8")["peak_fraction"] >= 0.70:
            raise RuntimeError("Dual-node memory gate failed")
    state("fixed200_replay_and_long_rss", maximum_combined_workers_per_node=8)
    for node in (0, 1):
        request(node, "ACCEPT", 8, "acceptance")
    for node in (0, 1):
        wait(node, "acceptance", 16000)
    p0_count = 0
    summaries = []
    for node in (0, 1):
        manifest = json.loads((roots[node] / "manifest.json").read_text())
        p0 = [s for s in manifest["specs"] if s["stage"] == "P0R1"]
        if len(p0) != 12:
            raise RuntimeError("P0 matrix incomplete")
        for item in p0:
            first = roots[node] / "runs" / item["key"]
            second = roots[node] / "runs" / item["key"].replace("P0R1", "P0R2")
            a = json.loads((first / "semantic_signature.json").read_text())
            b = json.loads((second / "semantic_signature.json").read_text())
            if a != b or a["offspring"] != 20000 or a["reason"] != "generation_limit":
                raise RuntimeError(f"P0 deterministic mismatch: {item['key']}")
            p0_count += 2
        for arm in ("F6", "M0"):
            item = next(s for s in manifest["specs"] if s["stage"] == "RSS" and s["arm"] == arm)
            summary = json.loads((roots[node] / "runs" / item["key"] / "summary.json").read_text())
            if summary["elapsed"] < 1200 or summary["termination_reason"] != "time_budget":
                raise RuntimeError("RSS did not cover the formal P1 budget")
            summaries.append(summary)
        for phase in ("P0", "RSS"):
            record = batch(node, phase)
            if record["peak_fraction"] >= 0.70 or not record["success"]:
                raise RuntimeError("Long-running memory or completion gate failed")
    legacy_checks = []
    manifest = json.loads((roots[0] / "manifest.json").read_text())
    for arm in ("F6", "M0", "N0"):
        item = next(
            s
            for s in manifest["specs"]
            if s["stage"] == "P0R1" and s["arm"] == arm and s["base_seed"] == 710001
        )
        legacy = json.loads((roots[0] / "ops" / f"legacy_{arm}.json").read_text())
        modern = json.loads(
            (roots[0] / "runs" / item["key"] / "semantic_signature.json").read_text()
        )
        if legacy != modern:
            raise RuntimeError(f"Legacy path mismatch: {arm}")
        legacy_checks.append(arm)
    workers_by_node = []
    throughput = {}
    for node in (0, 1):
        rows = {str(w): batch(node, f"THR{w}") for w in (1, 2, 4, 8)}
        rates = {w: 8 / max(r["elapsed"], 1e-9) for w, r in rows.items()}
        dual = batch(node, "DB8")
        double_rate = 8 / max(dual["elapsed"], 1e-9)
        # Use eight only when it improves aggregate throughput over four.
        chosen = 8 if rates["8"] >= 1.1 * rates["4"] and double_rate >= 0.9 * rates["4"] else 4
        workers_by_node.append(chosen)
        throughput[str(node)] = {"single_rates": rates, "dual8_rate": double_rate, "chosen": chosen}
    peak = max(s["peak_rss_gb"] for s in summaries)
    if 8 * peak * 1.25 + 4 > 32 * 0.70:
        raise RuntimeError("RSS worst-arm plus reserve exceeds the per-node gate")
    source = base / "p1_source_20261002"
    import os
    import subprocess

    py = "/workspace/envs/geo-llm-py313/bin/python"
    env = {**os.environ, "PYTHONPATH": str(source / "src")}
    current = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    source_hash = subprocess.check_output(
        [
            py,
            "-c",
            "from geo_llm_scheduler.experiments.runner import source_digest; print(source_digest())",
        ],
        env=env,
        text=True,
    ).strip()
    assert current == "3ec2355fbc43993375daab06ec0ea39fb9d86991"
    assert source_hash == "55e8137c4af517718394e43b96c477ac77b99cf6dc52b8844a9792ef90193ed7"
    worst_diagnostic = max(s["diagnostics"]["elapsed"] for s in summaries)
    eta_hours = max((240 * (900 + worst_diagnostic * 1.5 + 30)) / w / 3600 for w in workers_by_node)
    # Stop if the conservative budget cannot finish before the migration reserve.
    latest_safe = 1790956800  # 2026-10-03 00:00 Asia/Shanghai: 24h migration reserve
    if time.time() + eta_hours * 3600 > latest_safe:
        raise RuntimeError("Conservative migration window cannot accommodate P1")
    acceptance = {
        "accepted": True,
        "P0_runs": p0_count,
        "legacy_200gen_arms": legacy_checks,
        "long_RSS_seconds": 1200,
        "max_worker_peak_rss_gb": peak,
        "RSS_extra_validation": {"polish_and_checkpoint_validation_separate": True},
        "workers_by_node": workers_by_node,
        "throughput": throughput,
        "source_commit": current,
        "source_hash": source_hash,
        "PR": 19,
        "CI": "Python 3.11/3.13 PR and merged main checks passed",
        "expiry_verified": False,
        "migration_target": "2026-10-04",
        "conservative_remaining_hours": eta_hours,
        "worst_pilot_diagnostic_seconds": worst_diagnostic,
        "accepted_at": time.time(),
    }
    atomic(control / "acceptance.json", acceptance)
    for node in (0, 1):
        gate = {
            **acceptance,
            "node": node,
            "manifest_sha256": hashlib.sha256(
                (roots[node] / "manifest.json").read_bytes()
            ).hexdigest(),
        }
        atomic(roots[node] / "ops" / "launch_gate.json", gate)
    state("P1_launch", workers_by_node=workers_by_node, estimated_hours=eta_hours)
    for node in (0, 1):
        request(node, "P1", workers_by_node[node], "formal_p1")
    deadline = time.time() + 90
    ready = set()
    while len(ready) < 2 and time.time() < deadline:
        for node in (0, 1):
            path = roots[node] / "status.json"
            if path.exists():
                value = json.loads(path.read_text())
                if value["phase"] == "P1" and value["active"] and not value["failed"]:
                    ready.add(node)
        time.sleep(3)
    if len(ready) != 2:
        raise RuntimeError("P1 dispatch failed to become active on both nodes")
    for node in (0, 1):
        # The dispatcher is already running P1; it exits after that batch drains.
        request(node, "EXIT", 0, "exit_after_p1")
    state("P1_active", workers_by_node=workers_by_node, estimated_hours=eta_hours)
except Exception as error:
    state("failed", error=type(error).__name__, message=str(error))
    for node in (0, 1):
        atomic(roots[node] / "stop.request", {"acceptance_error": str(error), "time": time.time()})
    raise
