"""Four-arm Jev comparison on original SHA-bound development inputs; opt-in only."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import run_llm_flash as common

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.jev_decisions import JevSettings
from geo_llm_scheduler.experiments.jev_prompts import BACKGROUND_EN, PROMPT_VERSION, PROMPTS
from geo_llm_scheduler.experiments.jev_search import SINGLE_ARM
from geo_llm_scheduler.experiments.jev_search import run_policy as run_jev
from geo_llm_scheduler.experiments.jev_search import study_config as jev_config
from geo_llm_scheduler.experiments.jev_single import BACKGROUND as SINGLE_BACKGROUND
from geo_llm_scheduler.experiments.jev_single import PROMPT_VERSION as SINGLE_PROMPT_VERSION
from geo_llm_scheduler.experiments.jev_single import RANK_EN
from geo_llm_scheduler.experiments.llm_search import run_policy as run_control
from geo_llm_scheduler.experiments.llm_search import study_config as control_config
from geo_llm_scheduler.io.loaders import load_instance

ARMS = ("BILL_MOEAD", "MAIN_Q", "JEV_E05", "JEV_E09")
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
    "prompts_en.json",
)
sha = common.sha
source_hash = common.source_hash
atomic_json = common.atomic_json
preflight = common.preflight


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep controls unchanged and use the accepted per-step Jev implementation."""
    if arm not in (*ARMS, SINGLE_ARM):
        raise ValueError("Unknown comparison arm")
    return (jev_config if arm.startswith("JEV_") else control_config)(arm, seed, seconds)


def balanced_order(index: int) -> list[str]:
    """Use the four even-treatment Williams orders, independent of outcomes."""
    return [ARMS[(j + index) % 4] for j in (0, 1, 3, 2)]


def budget_bound(runs: int, max_tokens: int, calibration_reserve_usd: float = 0.1) -> float:
    """Preallocate each run's full cap, including unknown-usage reservations."""
    if (
        type(runs) is not int
        or type(max_tokens) is not int
        or runs < 1
        or max_tokens < 262144
        or not math.isfinite(calibration_reserve_usd)
        or calibration_reserve_usd < 0
    ):
        raise ValueError("Invalid prepaid campaign envelope")
    return runs * max_tokens * 0.042 / 1_000_000 + calibration_reserve_usd


def prepare(
    root: Path,
    historical: Path,
    *,
    parents_per_scale: int = 4,
    seeds: tuple[int, ...] = (17401, 27401),
    max_tokens: int = 4_000_000,
    spend_cap_usd: float = 4.5,
) -> dict:
    """Seal whole parent/variant/seed blocks only; refuse an unaffordable matrix."""
    if not math.isfinite(spend_cap_usd) or spend_cap_usd <= 0:
        raise ValueError("Positive finite campaign spend cap required")
    if root.exists():
        raise FileExistsError("Campaign root must be new")
    if parents_per_scale not in (1, 2, 3, 4) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid balanced panel selection")
    old = json.loads((historical / "manifest.json").read_text())
    assert old["commit"] == "b148c4addd207fe30a6b2a43d953a04afc35d3a1"
    assert (
        sha(historical / "manifest.json")
        == json.loads((historical / "manifest.sha256.json").read_text())["sha256"]
    )
    assert len(old["blocks"]) == 32 and old["runs"] == 1056
    originals = []
    parents: dict[int, list[int]] = {}
    for key in old["blocks"]:
        path = historical / "specs" / (key + ".json")
        assert sha(path) == old["files"][path.relative_to(historical).as_posix()]
        spec = json.loads(path.read_text())
        parents.setdefault(spec["n"], [])
        if spec["base_seed"] not in parents[spec["n"]]:
            parents[spec["n"]].append(spec["base_seed"])
        originals.append((spec, sha(path)))
    assert set(parents) == {20, 50, 100, 200} and all(len(v) == 4 for v in parents.values())
    chosen = {b for values in parents.values() for b in sorted(values)[:parents_per_scale]}
    originals = [(s, h) for s, h in originals if s["base_seed"] in chosen]
    assert len(originals) == len(chosen) * 2
    upper = budget_bound(len(originals) * len(seeds) * 2, max_tokens)
    if upper > spend_cap_usd:
        raise ValueError(f"Preallocated API bound {upper:.6f} exceeds cap {spend_cap_usd}")
    root.mkdir(parents=True)
    files, blocks = {}, []
    for seed in seeds:
        for original, original_sha in originals:
            source = historical / original["input"]
            assert sha(source) == original["input_sha256"] == old["files"][original["input"]]
            target = root / original["input"]
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            assert sha(target) == original["input_sha256"]
            key = original["key"].rsplit("_s", 1)[0] + "_s" + str(seed)
            spec = {
                k: original[k]
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
                historical_spec_sha256=original_sha,
                host=socket.gethostname(),
                order=balanced_order(len(blocks)),
            )
            path = root / "specs" / (key + ".json")
            atomic_json(path, spec)
            files[spec["input"]] = sha(target)
            files[path.relative_to(root).as_posix()] = sha(path)
            blocks.append(key)
    manifest = dict(
        source=source_hash(),
        host=socket.gethostname(),
        arms=list(ARMS),
        blocks=blocks,
        runs=len(blocks) * 4,
        files=files,
        commit=subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        historical_manifest_sha256=sha(historical / "manifest.json"),
        independent_parents=len(chosen),
        seeds=list(seeds),
        initialization="90+10_BILL4_P100_charged",
        window_seconds=8 * 3600,
        memory_reserve_bytes=200 * 1024**3,
        max_input_tokens=max_tokens,
        spend_cap_usd=spend_cap_usd,
        preallocated_upper_usd=upper,
        model="jev-1.14.0",
        long_context=True,
        api_concurrency=4,
        prompt_version=PROMPT_VERSION,
        treatment_validity="at_least_one_valid_action; report all coverage and guard hits separately",
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    return manifest


def prepare_pilot(
    root: Path,
    historical: Path,
    *,
    original_key: str = "r0_n200_R2M12_kv_v0_s16101",
    seed: int = 17401,
    seconds: float = 1200,
    max_tokens: int = 40_000_000,
    spend_cap_usd: float = 2.0,
    single_candidate: bool = False,
) -> dict:
    """Seal the original n200 pilot, optionally running only the declared B1 treatment."""
    if root.exists():
        raise FileExistsError("Campaign root must be new")
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(spend_cap_usd):
        raise ValueError("Invalid pilot duration or spend cap")
    if type(seed) is not int or seed < 0:
        raise ValueError("Explicit nonnegative seed required")
    upper = budget_bound(1, max_tokens)
    if upper > spend_cap_usd:
        raise ValueError("Pilot token allocation exceeds spend cap")
    old = json.loads((historical / "manifest.json").read_text())
    assert old["commit"] == "b148c4addd207fe30a6b2a43d953a04afc35d3a1"
    assert (
        sha(historical / "manifest.json")
        == json.loads((historical / "manifest.sha256.json").read_text())["sha256"]
    )
    assert original_key in old["blocks"] and len(old["blocks"]) == 32
    source_spec = historical / "specs" / (original_key + ".json")
    assert sha(source_spec) == old["files"][source_spec.relative_to(historical).as_posix()]
    original = json.loads(source_spec.read_text())
    assert original["n"] == 200 and original["variant"] == 0
    source_input = historical / original["input"]
    assert sha(source_input) == original["input_sha256"] == old["files"][original["input"]]
    key = original_key.rsplit("_s", 1)[0] + "_s" + str(seed)
    arms = [SINGLE_ARM] if single_candidate else ["BILL_MOEAD", "MAIN_Q", "JEV_E09"]
    spec = {
        k: original[k]
        for k in (
            "n",
            "regions",
            "machines",
            "pressure",
            "variant",
            "base_seed",
            "input",
            "input_sha256",
        )
    }
    spec.update(
        key=key,
        seed=seed,
        seconds=seconds,
        old_key=original_key,
        historical_spec_sha256=sha(source_spec),
        host=socket.gethostname(),
        order=arms,
    )
    root.mkdir(parents=True)
    target = root / spec["input"]
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(source_input, target)
    path = root / "specs" / (key + ".json")
    atomic_json(path, spec)
    manifest = dict(
        kind="single_n200_b1_top10" if single_candidate else "single_n200_e09_pilot",
        source=source_hash(),
        host=socket.gethostname(),
        arms=arms,
        blocks=[key],
        runs=len(arms),
        files={spec["input"]: sha(target), path.relative_to(root).as_posix(): sha(path)},
        commit=subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        historical_manifest_sha256=sha(historical / "manifest.json"),
        independent_parents=1,
        seeds=[seed],
        initialization="90+10_BILL4_P100_charged",
        window_seconds=3600,
        memory_reserve_bytes=200 * 1024**3,
        max_input_tokens=max_tokens,
        spend_cap_usd=spend_cap_usd,
        preallocated_upper_usd=upper,
        model="jev-1.14.0",
        long_context=True,
        api_concurrency=1,
        prompt_version=SINGLE_PROMPT_VERSION if single_candidate else PROMPT_VERSION,
        epsilon_g=0.9,
        interpretation="single-input/single-seed exploratory pilot; report all fallbacks and API coverage",
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    return manifest


def validate_inputs(root: Path) -> dict:
    """Reject changed source, panel, fee allocation, host, specs or inputs."""
    m = json.loads((root / "manifest.json").read_text())
    assert (
        sha(root / "manifest.json")
        == json.loads((root / "manifest.sha256.json").read_text())["sha256"]
    )
    assert m["source"] == source_hash() and m["host"] == socket.gethostname()
    expected = (
        (SINGLE_ARM,)
        if m.get("kind") == "single_n200_b1_top10"
        else ("BILL_MOEAD", "MAIN_Q", "JEV_E09")
        if m.get("kind") == "single_n200_e09_pilot"
        else ARMS
    )
    assert tuple(m["arms"]) == expected and len(set(m["blocks"])) == len(m["blocks"])
    assert m["runs"] == len(m["blocks"]) * len(expected)
    jev_count = sum(a.startswith("JEV_") for a in expected)
    assert (
        budget_bound(len(m["blocks"]) * jev_count, m["max_input_tokens"])
        == m["preallocated_upper_usd"]
        <= m["spend_cap_usd"]
    )
    for name, expected in m["files"].items():
        assert sha(root / name) == expected, name
    if m.get("kind") in ("single_n200_e09_pilot", "single_n200_b1_top10"):
        assert len(m["blocks"]) == 1 and m["epsilon_g"] == 0.9
        spec = json.loads((root / "specs" / (m["blocks"][0] + ".json")).read_text())
        assert spec["n"] == 200 and spec["variant"] == 0
    return m


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Validate immutable artifacts, common charged initialization and real Jev usage."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").exists():
        return False
    spec_path = root / "specs" / (key + ".json")
    spec = json.loads(spec_path.read_text())
    mark = json.loads((out / "complete.json").read_text())
    assert mark["key"] == key and mark["arm"] == arm and mark["source"] == source
    assert mark["host"] == spec["host"] == socket.gethostname()
    assert mark["spec_sha256"] == sha(spec_path)
    assert mark["input_sha256"] == spec["input_sha256"] == sha(root / spec["input"])
    assert set(mark["files"]) == set(REQUIRED)
    for name, record in mark["files"].items():
        assert (out / name).stat().st_size == record["size"] and sha(out / name) == record["sha256"]
    s = json.loads((out / "summary.json").read_text())
    assert s["source"] == source and s["arm"] == arm and s["seed"] == spec["seed"]
    assert json.loads((out / "config.json").read_text()) == json.loads(
        json.dumps(asdict(arm_config(arm, spec["seed"], spec["seconds"])))
    )
    assert s["termination"] == "time_budget" and s["offspring"] > 0
    assert s["population"] == 100 and s["final_exact_verified"] == 100 + s["archive"]
    check = json.loads((root / "preflight" / (key + ".json")).read_text())
    assert check["source"] == source and check["input_sha256"] == spec["input_sha256"]
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert s["initialization"][field] == check["policies"]["bill4"]["evidence"][field]
    q = json.loads((out / "qtable.json").read_text())
    assert len(q["q"]) == 42 and all(len(row) == 8 for row in q["q"])
    if arm.startswith("JEV_"):
        j = s["initialization"]["jev"]
        manifest = json.loads((root / "manifest.json").read_text())
        expected = JevSettings(
            epsilon_g=0.5 if arm == "JEV_E05" else 0.9,
            max_input_tokens=manifest["max_input_tokens"],
            long_context=True,
        )
        assert j["settings"] == asdict(expected)
        assert j["provider_transport"] == "official_https_with_shared_queue"
        assert j["prompt_version"] == (
            SINGLE_PROMPT_VERSION if arm == SINGLE_ARM else PROMPT_VERSION
        )
        assert j["jev_treatment_valid"] and j["valid_action_requests"] > 0
        assert not any(any(row) for row in q["updates"])
        with gzip.open(out / "policy_calls.jsonl.gz", "rt", encoding="utf-8") as stream:
            calls = [json.loads(line) for line in stream]
        assert (
            len(calls) == j["requests"] and sum(c["accepted"] for c in calls) == j["valid_requests"]
        )
        assert (
            sum(c["reserved_input_tokens"] for c in calls) == j["billed_or_reserved_input_tokens"]
        )
        assert j["billed_or_reserved_input_tokens"] <= manifest["max_input_tokens"]
    return True


def queued_request(payload: dict, timeout: float) -> dict:
    """Bound shared HTTP concurrency; charge queueing to the caller's existing clock."""
    import fcntl

    from geo_llm_scheduler.experiments.jev_decisions import isolated_request

    deadline = time.perf_counter() + timeout
    directory = Path(os.environ["JEV_REQUEST_LOCK_DIR"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = None
    try:
        while handle is None:
            for slot in range(4):
                candidate = (directory / f"slot{slot}.lock").open("a")
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    handle = candidate
                    break
                except BlockingIOError:
                    candidate.close()
            if time.perf_counter() >= deadline:
                raise TimeoutError("Jev queue deadline")
            if handle is None:
                time.sleep(min(0.025, max(0, deadline - time.perf_counter())))
        return isolated_request(
            payload, max(0.001, deadline - time.perf_counter()), long_context=True
        )
    finally:
        if handle:
            handle.close()


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Run one charged policy, stream calls/steps, then fresh-exact audit every final member."""
    import fcntl
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    os.sched_setaffinity(0, {cpu})
    (root / "locks").mkdir(exist_ok=True)
    with (root / "locks" / (key + "_" + arm + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _worker(root, key, arm, cpu)


def _worker(root: Path, key: str, arm: str, cpu: int) -> None:
    manifest = validate_inputs(root)
    spec_path = root / "specs" / (key + ".json")
    spec = json.loads(spec_path.read_text())
    out = root / "runs" / key / arm
    out.mkdir(parents=True, exist_ok=False)
    cfg = arm_config(arm, spec["seed"], spec["seconds"])
    atomic_json(out / "config.json", asdict(cfg))
    atomic_json(
        out / "prompts_en.json",
        dict(
            version=SINGLE_PROMPT_VERSION if arm == SINGLE_ARM else PROMPT_VERSION,
            background=SINGLE_BACKGROUND if arm == SINGLE_ARM else BACKGROUND_EN,
            questions={k: v[0] for k, v in PROMPTS.items()}
            | ({"a6_job_ranking": RANK_EN} if arm == SINGLE_ARM else {}),
        ),
    )
    evidence: dict = {}
    snapshots: list = []
    started, clock, last = time.time(), time.perf_counter(), -10.0
    thresholds = [spec["seconds"] * f for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
    step_counts = dict(completed=0, valid_action=0, fallback_action=0)
    with (
        gzip.open(out / "steps.jsonl.gz", "wt", encoding="utf-8") as steps,
        gzip.open(out / "policy_calls.jsonl.gz", "wt", encoding="utf-8") as calls,
    ):

        def sink(record: dict) -> None:
            calls.write(json.dumps(record, allow_nan=False) + "\n")
            calls.flush()

        def observe(event: str, data: dict) -> None:
            nonlocal last
            if event == "rl_step":
                row = {
                    k: v
                    for k, v in data.items()
                    if k
                    not in (
                        "population",
                        "gateway",
                        "controller",
                        "streams",
                        "stagnation",
                        "context",
                    )
                }
                steps.write(json.dumps(row, allow_nan=False) + "\n")
                step_counts["completed"] += 1
                step_counts["valid_action"] += int(data.get("action_source") == "jev_current_step")
                step_counts["fallback_action"] += int(
                    data.get("action_source") == "uniform_fallback"
                )
            gateway = data.get("gateway")
            if gateway is None:
                return
            elapsed = time.perf_counter() - clock
            if event == "initial_ready":
                common.write_candidates(out / "initial.json.gz", data["population"])
            if event == "initial_ready" or thresholds and elapsed >= thresholds[0]:
                target = "initialization" if event == "initial_ready" else thresholds.pop(0)
                snapshots.append(
                    dict(
                        target=target,
                        elapsed=elapsed,
                        objectives=[c.evaluation.objectives for c in gateway.archive.members],
                    )
                )
            if elapsed - last >= 10 or event == "initial_ready":
                atomic_json(
                    out / "heartbeat.json",
                    dict(
                        pid=os.getpid(),
                        host=socket.gethostname(),
                        updated=time.time(),
                        elapsed=elapsed,
                        exact=gateway.counts["exact"],
                        archive=len(gateway.archive.members),
                        step_counts=step_counts,
                    ),
                )
                last = elapsed

        try:
            problem = load_instance(root / spec["input"])
            clock = time.perf_counter()
            if arm.startswith("JEV_"):
                settings = JevSettings(
                    epsilon_g=0.5 if arm == "JEV_E05" else 0.9,
                    max_input_tokens=manifest["max_input_tokens"],
                    long_context=True,
                )
                result = run_jev(
                    problem,
                    arm,
                    cfg,
                    evidence,
                    observer=observe,
                    settings=settings,
                    requester=queued_request,
                    request_sink=sink,
                )
            else:
                result = run_control(problem, arm, cfg, evidence, observer=observe)
            audit = time.perf_counter()
            for candidate in result.population + result.archive.members:
                assert (
                    candidate.evaluation.feasible
                    and evaluate(problem, candidate.genotype, candidate.schedule)
                    == candidate.evaluation
                )
            audit_seconds = time.perf_counter() - audit
            common.write_candidates(out / "population.json.gz", result.population)
            common.write_candidates(out / "archive.json.gz", result.archive.members)
            atomic_json(
                out / "qtable.json",
                dict(
                    q=result.controller.q,
                    updates=result.controller.updates,
                    visits=result.controller.visits,
                    selections=result.controller.selections,
                ),
            )
            snapshots.append(
                dict(
                    target="final",
                    elapsed=result.elapsed,
                    objectives=[c.evaluation.objectives for c in result.archive.members],
                )
            )
            atomic_json(out / "anytime.json", snapshots)
            atomic_json(
                out / "diagnostics.json",
                dict(
                    counts=dict(result.gateway.counts),
                    seconds=dict(result.gateway.seconds),
                    step_counts=step_counts,
                ),
            )
            # Full request bodies are already streamed; do not duplicate them in summary.
            if "jev" in evidence:
                evidence["jev"].pop("requests_evidence", None)
                evidence["jev"]["provider_transport"] = "official_https_with_shared_queue"
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
                    pid=os.getpid(),
                    cpu=cpu,
                    requested_seconds=spec["seconds"],
                    production_seconds=result.elapsed,
                    audit_seconds=audit_seconds,
                    initialization=evidence,
                    offspring=result.offspring_count,
                    termination=result.termination_reason,
                    population=len(result.population),
                    archive=len(result.archive.members),
                    final_exact_verified=len(result.population) + len(result.archive.members),
                    step_counts=step_counts,
                    counts=dict(result.gateway.counts),
                ),
            )
        except BaseException:
            atomic_json(
                out / "failure.json", dict(updated=time.time(), traceback=traceback.format_exc())
            )
            raise
    atomic_json(
        out / "complete.json",
        dict(
            arm=arm,
            key=key,
            source=manifest["source"],
            host=socket.gethostname(),
            input_sha256=spec["input_sha256"],
            spec_sha256=sha(spec_path),
            files={n: dict(size=(out / n).stat().st_size, sha256=sha(out / n)) for n in REQUIRED},
        ),
    )
    try:
        assert complete(root, key, arm, manifest["source"])
    except BaseException:
        atomic_json(
            out / "failure.json", dict(updated=time.time(), traceback=traceback.format_exc())
        )
        raise


def main() -> None:
    """Require explicit affordable preparation; never launch a campaign by import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "prepare-pilot", "preflight", "worker", "validate", "dispatch"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--parents-per-scale", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17401, 27401])
    parser.add_argument("--spend-cap-usd", type=float, default=4.5)
    parser.add_argument("--max-input-tokens", type=int, default=4_000_000)
    parser.add_argument("--key")
    parser.add_argument("--arm", choices=(*ARMS, SINGLE_ARM))
    parser.add_argument("--single-candidate", action="store_true")
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=96)
    parser.add_argument("--seconds", type=float, default=1200)
    args = parser.parse_args()
    if args.action == "prepare":
        print(
            json.dumps(
                prepare(
                    args.root,
                    args.historical_root,
                    parents_per_scale=args.parents_per_scale,
                    seeds=tuple(args.seeds),
                    spend_cap_usd=args.spend_cap_usd,
                    max_tokens=args.max_input_tokens,
                )
            )
        )
    elif args.action == "prepare-pilot":
        if len(args.seeds) != 1:
            raise ValueError("Pilot requires one explicit seed")
        print(
            json.dumps(
                prepare_pilot(
                    args.root,
                    args.historical_root,
                    seed=args.seeds[0],
                    seconds=args.seconds,
                    max_tokens=args.max_input_tokens,
                    spend_cap_usd=args.spend_cap_usd,
                    single_candidate=args.single_candidate,
                )
            )
        )
    elif args.action == "preflight":
        print(json.dumps(preflight(args.root, args.key)))
    elif args.action == "worker":
        worker(args.root, args.key, args.arm, args.cpu)
    elif args.action == "validate":
        m = validate_inputs(args.root)
        print(
            json.dumps(
                dict(
                    validated=sum(
                        complete(args.root, k, a, m["source"])
                        for k in m["blocks"]
                        for a in m["arms"]
                    ),
                    total=m["runs"],
                )
            )
        )
    else:
        import llm_flash_dispatch as dispatcher

        secret = Path(os.environ["JEV_SECRET_FILE"])
        if secret.stat().st_mode & 0o077:
            raise PermissionError("Credential must be owner-only")
        os.environ["TYPESAFE_API_KEY"] = secret.read_text().strip()
        if not os.environ["TYPESAFE_API_KEY"]:
            raise ValueError("Empty Jev credential")
        allowed = set(os.sched_getaffinity(0))
        topology = {
            c: tuple(
                (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
                for f in ("physical_package_id", "core_id")
            )
            for c in allowed
        }
        cpus = dispatcher.cpu_order(allowed, topology, args.parallel)
        print(
            json.dumps(dispatcher.dispatch(args.root, sys.modules[__name__], cpus, Path(__file__)))
        )


if __name__ == "__main__":
    main()
