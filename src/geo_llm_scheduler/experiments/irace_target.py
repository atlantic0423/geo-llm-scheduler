"""Auditable irace target: immutable inputs, full inner budget, fixed final HV."""

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_search_v4 import run_components
from geo_llm_scheduler.experiments.irace_space import decode_configuration
from geo_llm_scheduler.experiments.metrics import hypervolume
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.search import LSV2Search


def sha256(path: Path) -> str:
    """Hash raw file bytes for input and result identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_spec(points: list[tuple[float, float]], source: dict) -> dict:
    """Freeze an instance frame from a declared reference run, rejecting zero ranges."""
    if not points or any(not math.isfinite(x) for p in points for x in p):
        raise ValueError("A finite nonempty reference front is required")
    lower = [min(p[k] for p in points) for k in range(2)]
    upper = [max(p[k] for p in points) for k in range(2)]
    if any(b <= a for a, b in zip(lower, upper)):
        raise ValueError("Degenerate reference scale; cannot freeze HV")
    return dict(
        objectives=["flow", "bill"], lower=lower, upper=upper, reference=[1.1, 1.1], source=source
    )


def score_front(points: list[tuple[float, float]], metric: dict) -> float:
    """Negative normalized HV; outside-reference points contribute zero, with no clipping."""
    lo, hi, ref = metric["lower"], metric["upper"], metric["reference"]
    if metric["objectives"] != ["flow", "bill"] or any(len(v) != 2 for v in (lo, hi, ref)):
        raise ValueError("Invalid metric dimensions or objective order")
    if any(not math.isfinite(x) for v in (lo, hi, ref) for x in v) or any(
        b <= a for a, b in zip(lo, hi)
    ):
        raise ValueError("Invalid frozen metric bounds")
    normalized = []
    for point in points:
        if len(point) != 2 or not all(math.isfinite(x) for x in point):
            raise ValueError("Invalid objective vector")
        p = tuple((point[k] - lo[k]) / (hi[k] - lo[k]) for k in range(2))
        if all(p[k] <= ref[k] for k in range(2)):
            normalized.append(p)
    return -hypervolume(normalized, tuple(ref))


def validate_case(path: Path) -> dict:
    """Verify frozen per-instance inputs before every candidate run."""
    case = json.loads(path.read_text(encoding="utf-8"))
    for name in ("instance", "calibration", "metric"):
        item = path.parent / case[name]["path"]
        if sha256(item) != case[name]["sha256"]:
            raise ValueError(f"Frozen {name} checksum mismatch")
    problem = load_instance(path.parent / case["instance"]["path"])
    if case["purpose"] == "formal":
        if case["seconds"] != {50: 600, 100: 1200, 200: 2400}[len(problem.jobs)]:
            raise ValueError("Formal case must retain the full inclusive budget")
        if case.get("generations") is not None:
            raise ValueError("Formal case cannot use a generation cap")
    elif case["purpose"] != "smoke":
        raise ValueError("Unknown case purpose")
    return case


def evaluate_request(request: dict, case_path: Path, run_dir: Path) -> dict:
    """Run a configuration once; fail loudly and preserve all diagnostics on errors.

    Existing completed records can be replayed only for exactly the same request
    and frozen case. Explicit recovery preserves incomplete attempts before retry.
    """
    case = validate_case(case_path)
    if request.get("seal"):
        from geo_llm_scheduler.experiments.irace_campaign import verify_seal

        verify_seal(Path(request["seal"]), case_path)
    elif case["purpose"] == "formal":
        raise ValueError("Formal evaluation requires a frozen campaign seal")
    spec = decode_configuration(request["configuration"])
    seed = request["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    identity = json.loads(
        json.dumps(dict(configuration=asdict(spec), seed=seed, case_sha256=sha256(case_path)))
    )
    if (
        run_dir.exists()
        and not (run_dir / "complete.json").exists()
        and request.get("retry_incomplete")
    ):
        prior = run_dir / "request.json"
        if (
            not prior.exists()
            or json.loads(prior.read_text(encoding="utf-8"))["identity"] != identity
        ):
            raise ValueError("Incomplete attempt identity mismatch")
        preserved = run_dir.with_name(
            run_dir.name + ".incomplete_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        )
        if preserved.parent.resolve() != run_dir.parent.resolve():
            raise ValueError("Recovery destination escaped the run directory")
        run_dir.rename(preserved)
    if run_dir.exists():
        completed = run_dir / "complete.json"
        if not completed.exists():
            raise ValueError("Incomplete attempt exists; inspect and use an explicit new attempt")
        record = json.loads(completed.read_text(encoding="utf-8"))
        if record["identity"] != json.loads(json.dumps(identity)):
            raise ValueError("Run identity mismatch")
        if sha256(run_dir / "result.json") != record["result_sha256"]:
            raise ValueError("Completed artifact checksum mismatch")
        return record
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "request.json", dict(request=request, identity=identity))
    atomic_json(run_dir / "status.json", {"status": "running"})
    tick = perf_counter()
    try:
        problem = load_instance(case_path.parent / case["instance"]["path"])
        calibration = json.loads(
            (case_path.parent / case["calibration"]["path"]).read_text(encoding="utf-8")
        )
        metric = json.loads((case_path.parent / case["metric"]["path"]).read_text(encoding="utf-8"))
        evidence: dict = {}
        searches: list[LSV2Search] = []
        result = run_components(
            problem,
            spec,
            seed,
            case["seconds"],
            calibration,
            evidence,
            search_sink=searches.append,
            generations=case.get("generations"),
        )
        # Independent exact audit is post-run, recorded separately from the search budget.
        audit_tick = perf_counter()
        for c in result.archive.members:
            exact = evaluate(problem, c.genotype, c.schedule)
            if not exact.feasible or exact.objectives != c.evaluation.objectives:
                raise ValueError("Final archive failed independent exact audit")
        cost = score_front([c.evaluation.objectives for c in result.archive.members], metric)
        payload = dict(
            evidence=evidence,
            archive=[solution(c) for c in result.archive.members],
            population=[solution(c) for c in result.population],
            local_search=searches[0].summary(),
            counts=dict(result.gateway.counts),
            termination=result.termination_reason,
            solver_seconds=result.elapsed,
            post_audit_seconds=perf_counter() - audit_tick,
            cost=cost,
            identity=identity,
        )
        atomic_json(run_dir / "result.json", payload)
        record = dict(
            identity=identity,
            cost=cost,
            time=perf_counter() - tick,
            result_sha256=sha256(run_dir / "result.json"),
        )
        atomic_json(run_dir / "complete.json", record)
        atomic_json(run_dir / "status.json", {"status": "complete"})
        return record
    except Exception as error:
        atomic_json(run_dir / "status.json", {"status": "failed", "error": repr(error)})
        raise
