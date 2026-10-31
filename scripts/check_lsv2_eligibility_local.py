"""Finite local gateway replay and prepared configuration files; never dispatches a campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import lsv2_eligibility_selection as study
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.lsv2.selection_summary import summarize_selection
from geo_llm_scheduler.utils.rng import RNGManager


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Replay fixed calls on archived physical schedules, with real exact and reflow.

    The 100-slot technical population repeats at most twelve historical inputs.
    This skips BILL4/evolution/Q episodes deliberately: it checks the new external
    selector path only, while integration tests cover unchanged terminal credit.
    Wall times and changing pools make this unsuitable for quality comparisons.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=12)
    args = parser.parse_args()
    if args.calls < 1:
        raise ValueError("At least one technical replay call is required")
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.historical
    manifest = _read(source / "manifest.json")
    assert _sha(source / "manifest.json") == _read(source / "manifest.sha256.json")["sha256"]
    expected_keys = {f"st_n{n}_d{i:02d}_s16901" for n, ids in study.PANEL.items() for i in ids}
    jobs = [j for j in manifest["jobs"] if j["arm"] == "C1_COMMON"]
    assert {j["runs"][0]["key"] for j in jobs} == expected_keys and len(jobs) == 8
    configs, checks, input_hashes, rows = {}, {}, {}, []
    for job in jobs:
        key = job["runs"][0]["key"]
        spec_name = "specs/" + key + ".json"
        assert _sha(source / spec_name) == manifest["files"][spec_name]
        spec = _read(source / spec_name)
        assert (
            spec["seed"] == study.RUN_SEED and spec["seconds"] == {100: 1200, 200: 2400}[spec["n"]]
        )
        assert (
            _sha(source / spec["input"]) == manifest["files"][spec["input"]] == spec["input_sha256"]
        )
        assert (
            key in job["calibration"]["validation_keys"] and key not in job["calibration"]["keys"]
        )
        problem = load_instance(source / spec["input"])
        run = source / "results" / job["id"] / ("00_" + key)
        log = run / "rightshift_steps.json.gz"
        assert _sha(log) == _read(run / "complete.json")["files"][log.name]["sha256"]
        input_hashes[str(log.relative_to(source))] = _sha(log)
        input_hashes[spec["input"]] = _sha(source / spec["input"])
        with gzip.open(log, "rt", encoding="utf-8") as stream:
            historical = [
                r for r in json.load(stream) if "input_schedule" in r and "diagnostics" in r
            ]
        assert historical, f"No archived physical inputs in {key}"
        indexes = sorted({round(i * (len(historical) - 1) / 11) for i in range(12)})
        for arm in study.ARMS:
            cfg = study.arm_config(arm, study.RUN_SEED, spec["seconds"])
            record = study.configuration_record(
                arm, study.RUN_SEED, spec["seconds"], job["calibration"]
            )
            name = "configs/" + key + "_" + arm + ".json"
            atomic_json(args.out / name, record)
            configs[name] = _sha(args.out / name)
            gateway = EvaluationGateway(problem, Archive())
            seeds = []
            for i in indexes:
                v = historical[i]["input_schedule"]
                c = gateway.evaluate(
                    Genotype(tuple(v["ms"]), tuple(v["os"])), Schedule(tuple(v["starts"]))
                )
                assert c.evaluation.feasible
                seeds.append(c)
            pop = [seeds[i % len(seeds)] for i in range(cfg.population)]
            frame = FrozenFrame.of([c.evaluation.objectives for c in seeds])
            search = study.make_search(arm, cfg, job["calibration"], QTable(), training=True)
            streams = RNGManager(study.RUN_SEED)
            audited = []
            gateway.candidate_sink = audited.append
            if arm != "S0_NONE":
                for call in range(args.calls):
                    search._polish(
                        seeds[call % len(seeds)], frame, call, pop, gateway, streams, lambda: 1.0
                    )
                    assert search._pending is None and not search.table.values
                    assert (
                        not search.reused_polish.resource_calendar
                        and not search.reused_polish.peak_switch
                    )
            for c in audited + pop + gateway.archive.members:
                assert (
                    c.evaluation.feasible
                    and evaluate(problem, c.genotype, c.schedule) == c.evaluation
                )
            assert all(r["exact"] <= 3 for r in search.polish_steps)
            summary = summarize_selection(search.polish_steps)
            result_name = "replay/" + key + "_" + arm + ".json"
            atomic_json(args.out / result_name, search.polish_steps)
            checks[result_name] = _sha(args.out / result_name)
            rows.append(dict(key=key, arm=arm, fresh_exact=len(audited), summary=summary))
        print(key, flush=True)
    for name, expected in input_hashes.items():
        assert _sha(source / name) == expected
    repo = Path(__file__).resolve().parents[1]
    result = dict(
        accepted=True,
        kind="local_functional_replay_only",
        online_campaign_started=False,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        ),
        study=study.PROTOCOL,
        primary_contrasts=study.PRIMARY_CONTRASTS,
        planned_runs=64,
        total_solver_seconds=115200,
        run_seed=study.RUN_SEED,
        configuration_sha256=configs,
        replay_sha256=checks,
        input_sha256=input_hashes,
        calls_per_enabled_arm=args.calls,
        rows=rows,
        limitations=[
            "prepared configurations are not a deployed/preflight campaign",
            "technical population repeats historical snapshots; no online quality comparison",
            "timing-driven region choices are not guaranteed identical across replay runs",
        ],
    )
    atomic_json(args.out / "LOCAL_CHECK.json", result)
    print(
        json.dumps(
            dict(
                accepted=True,
                cells=len(rows),
                configs=len(configs),
                calls=sum(r["summary"]["module_attempts"] for r in rows),
                fresh_exact=sum(r["fresh_exact"] for r in rows),
            )
        )
    )


if __name__ == "__main__":
    main()
