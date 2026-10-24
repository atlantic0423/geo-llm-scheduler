"""Twelve-run, four-paired-block English-only population-LS comparison."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import socket
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from run_llm_flash import preflight, sha, source_hash, write_candidates

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.round_prompts import BACKGROUND, VERSION
from geo_llm_scheduler.experiments.round_search import ARM, ARMS, run_policy, study_config
from geo_llm_scheduler.io.loaders import load_instance

REQUIRED = (
    "config.json",
    "summary.json",
    "initial.json.gz",
    "population.json.gz",
    "archive.json.gz",
    "qtable.json",
    "anytime.json",
    "diagnostics.json",
    "steps.jsonl.gz",
    "policy_calls.jsonl.gz",
    "rounds.json",
    "global_prompt.txt",
)
ORIGINALS = ("r0_n100_R2M12_kv_v0_s16101", "r0_n200_R2M12_kv_v0_s16101")
SEEDS = (17401, 27401)


def prepare(root: Path, historical: Path) -> dict:
    """Copy the original two SHA-bound inputs into exactly four paired seed blocks."""
    if root.exists():
        raise FileExistsError("New experiment root required")
    old = json.loads((historical / "manifest.json").read_text(encoding="utf-8"))
    assert old["commit"] == "b148c4addd207fe30a6b2a43d953a04afc35d3a1"
    assert (
        sha(historical / "manifest.json")
        == json.loads((historical / "manifest.sha256.json").read_text())["sha256"]
    )
    root.mkdir(parents=True)
    files, blocks = {}, []
    for original in ORIGINALS:
        spec_path = historical / "specs" / (original + ".json")
        assert sha(spec_path) == old["files"][spec_path.relative_to(historical).as_posix()]
        source = json.loads(spec_path.read_text())
        input_path = historical / source["input"]
        assert sha(input_path) == source["input_sha256"] == old["files"][source["input"]]
        target = root / source["input"]
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(input_path, target)
        files[source["input"]] = sha(target)
        for seed in SEEDS:
            key = original.rsplit("_s", 1)[0] + "_s" + str(seed)
            spec = {
                k: source[k]
                for k in (
                    "n",
                    "regions",
                    "machines",
                    "pressure",
                    "variant",
                    "base_seed",
                    "seconds",
                    "input",
                    "input_sha256",
                )
            }
            spec.update(
                key=key,
                seed=seed,
                host=socket.gethostname(),
                old_key=original,
                historical_spec_sha256=sha(spec_path),
            )
            atomic_json(root / "specs" / (key + ".json"), spec)
            files["specs/" + key + ".json"] = sha(root / "specs" / (key + ".json"))
            blocks.append(key)
    manifest = dict(
        source=source_hash(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        host=socket.gethostname(),
        arms=list(ARMS),
        blocks=blocks,
        runs=12,
        files=files,
        seeds=list(SEEDS),
        prompt_version=VERSION,
        clock_basis="search_excluding_synchronous_api_blocking",
        entry_policy="population_reentry20",
        fraction=0.2,
        language="English only",
        note="Two original physical inputs, two seeds each; four paired blocks, not four independent bases",
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    return manifest


def complete(root: Path, key: str, arm: str) -> bool:
    """Verify full artifacts, source, exact evidence, initial fingerprints and both clocks."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").exists():
        return False
    m = json.loads((root / "manifest.json").read_text())
    s = json.loads((root / "specs" / (key + ".json")).read_text())
    marker = json.loads((out / "complete.json").read_text())
    assert marker["source"] == m["source"] == source_hash()
    assert marker["host"] == m["host"] == s["host"] == socket.gethostname()
    assert marker["input_sha256"] == s["input_sha256"] == sha(root / s["input"])
    assert marker["spec_sha256"] == sha(root / "specs" / (key + ".json"))
    assert marker["arm"] == arm and marker["key"] == key
    assert set(marker["files"]) == set(REQUIRED)
    for name, record in marker["files"].items():
        assert (out / name).stat().st_size == record["size"] and sha(out / name) == record["sha256"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["termination"] == "time_budget" and summary["offspring"] > 0
    assert summary["population"] == 100
    assert summary["final_exact_verified"] == summary["population"] + summary["archive"]
    cfg = study_config(arm, s["seed"], s["seconds"])
    assert json.loads((out / "config.json").read_text()) == json.loads(json.dumps(asdict(cfg)))
    init = summary["initialization"]
    expected = json.loads((root / "preflight" / (key + ".json")).read_text())["policies"]["bill4"][
        "evidence"
    ]
    assert all(
        init[field] == expected[field]
        for field in ("initial_population_sha256", "initial_archive_sha256")
    )
    wall, search, blocked = (
        summary[k] for k in ("production_wall_s", "search_elapsed_s", "api_blocked_s")
    )
    assert abs(wall - search - blocked) < 1e-6 and search >= s["seconds"]
    if arm == ARM:
        e = init["round_llm"]
        assert set(e["valid_by_task"]) == {
            "ROUND_SELECT",
            "STEP_ACTION",
            "STEP_TARGET",
            "STEP_STOP",
        }
        assert all(e["valid_by_task"][k] > 0 for k in e["valid_by_task"])
        assert abs(sum(b - a for a, b in e["intervals"]) - blocked) < 1e-6
        assert all(
            r["quota"] == min(r["pool_size"], 20) and len(r["selected"]) == r["quota"]
            for r in e["rounds"]
        )
    else:
        assert blocked == 0
    assert (out / "global_prompt.txt").read_text(encoding="utf-8") == BACKGROUND
    return True


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Execute one frozen run, then independently exact-audit every final schedule."""
    import fcntl
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    os.sched_setaffinity(0, {cpu})
    lock_path = root / "locks" / (key + "_" + arm + ".lock")
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == source_hash()
    spec_path = root / "specs" / (key + ".json")
    assert sha(spec_path) == manifest["files"][spec_path.relative_to(root).as_posix()]
    spec = json.loads(spec_path.read_text())
    assert sha(root / spec["input"]) == spec["input_sha256"]
    out = root / "runs" / key / arm
    out.mkdir(parents=True, exist_ok=False)
    cfg = study_config(arm, spec["seed"], spec["seconds"])
    atomic_json(out / "config.json", asdict(cfg))
    (out / "global_prompt.txt").write_text(BACKGROUND, encoding="utf-8")
    steps = gzip.open(out / "steps.jsonl.gz", "wt", encoding="utf-8")
    policies = gzip.open(out / "policy_calls.jsonl.gz", "wt", encoding="utf-8")
    started = time.time()
    state = dict(exact=0, search_elapsed_s=0.0, production_wall_s=0.0, api_blocked_s=0.0)
    last = [-10.0]
    snapshots = []
    thresholds = [spec["seconds"] * f for f in (0.1, 0.25, 0.5, 0.75, 0.9)]

    def heartbeat(phase: str) -> None:
        atomic_json(
            out / "heartbeat.json",
            dict(
                pid=os.getpid(),
                cpu=cpu,
                host=socket.gethostname(),
                updated=time.time(),
                phase=phase,
                **state,
            ),
        )

    def observe(event: str, data: dict) -> None:
        if event == "rl_step":
            row = {
                k: v
                for k, v in data.items()
                if k
                not in ("population", "gateway", "controller", "streams", "stagnation", "context")
            }
            steps.write(json.dumps(row, allow_nan=False) + "\n")
            steps.flush()
        gateway = data.get("gateway")
        if gateway is None:
            return
        wall = data.get("elapsed", 0.0)
        elapsed = data.get("search_elapsed_s", wall)
        state.update(
            exact=gateway.counts["exact"],
            search_elapsed_s=elapsed,
            production_wall_s=wall,
            api_blocked_s=data.get("api_blocked_s", 0.0),
        )
        if event == "initial_ready":
            write_candidates(out / "initial.json.gz", data["population"])
        if event == "initial_ready" or thresholds and elapsed >= thresholds[0]:
            target = "initialization" if event == "initial_ready" else thresholds.pop(0)
            snapshots.append(
                dict(
                    target=target,
                    search_elapsed_s=elapsed,
                    wall_elapsed_s=wall,
                    objectives=[c.evaluation.objectives for c in gateway.archive.members],
                )
            )
        if time.time() - last[0] >= 10 or event in ("initial_ready", "round_complete"):
            heartbeat(event)
            last[0] = time.time()

    def sink(record: dict) -> None:
        policies.write(json.dumps(record, allow_nan=False) + "\n")
        policies.flush()
        state["search_elapsed_s"] = record["search_elapsed_s"]
        heartbeat("api_response_" + record["request"]["task"])

    try:
        if arm == ARM:
            secret = Path(os.environ["LLM_SECRET_FILE"])
            assert secret.is_file() and secret.stat().st_uid == os.getuid()
            assert secret.stat().st_mode & 0o077 == 0
            os.environ["DEEPSEEK_API_KEY"] = secret.read_text().strip()
        problem = load_instance(root / spec["input"])
        evidence: dict = {}
        result = run_policy(problem, arm, cfg, evidence, observer=observe, request_sink=sink)
        steps.close()
        policies.close()
        audit = time.perf_counter()
        for candidate in result.population + result.archive.members:
            assert candidate.evaluation.feasible
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
        audit_s = time.perf_counter() - audit
        blocked = evidence.get("round_llm", {}).get("api_blocked_s", 0.0)
        search = result.elapsed - blocked
        write_candidates(out / "population.json.gz", result.population)
        write_candidates(out / "archive.json.gz", result.archive.members)
        atomic_json(
            out / "qtable.json",
            dict(
                q=result.controller.q,
                visits=result.controller.visits,
                updates=result.controller.updates,
                selections=result.controller.selections,
            ),
        )
        snapshots.append(
            dict(
                target="final",
                search_elapsed_s=search,
                wall_elapsed_s=result.elapsed,
                objectives=[c.evaluation.objectives for c in result.archive.members],
            )
        )
        atomic_json(out / "anytime.json", snapshots)
        atomic_json(
            out / "diagnostics.json",
            dict(counts=dict(result.gateway.counts), seconds=dict(result.gateway.seconds)),
        )
        atomic_json(out / "rounds.json", evidence.get("round_llm", {}).get("rounds", []))
        atomic_json(
            out / "summary.json",
            dict(
                arm=arm,
                seed=spec["seed"],
                host=socket.gethostname(),
                source=manifest["source"],
                commit=manifest["commit"],
                started=started,
                completed=time.time(),
                requested_search_s=spec["seconds"],
                production_wall_s=result.elapsed,
                search_elapsed_s=search,
                api_blocked_s=blocked,
                audit_s=audit_s,
                initialization=evidence,
                offspring=result.offspring_count,
                termination=result.termination_reason,
                population=len(result.population),
                archive=len(result.archive.members),
                final_exact_verified=len(result.population) + len(result.archive.members),
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            ),
        )
        atomic_json(
            out / "complete.json",
            dict(
                arm=arm,
                key=key,
                source=manifest["source"],
                host=socket.gethostname(),
                input_sha256=spec["input_sha256"],
                spec_sha256=sha(spec_path),
                files={
                    name: dict(size=(out / name).stat().st_size, sha256=sha(out / name))
                    for name in REQUIRED
                },
            ),
        )
        assert complete(root, key, arm)
    except BaseException:
        steps.close()
        policies.close()
        atomic_json(
            out / "failure.json", dict(updated=time.time(), traceback=traceback.format_exc())
        )
        raise


def main() -> None:
    """Expose prepare/preflight/worker/validate operations for the frozen matrix."""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "preflight", "worker", "validate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--historical", type=Path)
    p.add_argument("--key")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--cpu", type=int, default=0)
    a = p.parse_args()
    if a.action == "prepare":
        assert a.historical is not None
        print(json.dumps(prepare(a.root, a.historical)))
    elif a.action == "preflight":
        assert a.key is not None
        print(json.dumps(preflight(a.root, a.key)))
    elif a.action == "worker":
        assert a.key is not None and a.arm is not None
        worker(a.root, a.key, a.arm, a.cpu)
    else:
        m = json.loads((a.root / "manifest.json").read_text())
        assert all(complete(a.root, key, arm) for key in m["blocks"] for arm in m["arms"])
        print(json.dumps(dict(accepted=True, runs=12)))


if __name__ == "__main__":
    main()
