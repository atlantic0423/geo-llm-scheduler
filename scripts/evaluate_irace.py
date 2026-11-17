"""Run paired validation of race elites, select per arm, then evaluate frozen finalists."""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_search_v4 import Components
from geo_llm_scheduler.experiments.irace_campaign import verify_seal
from geo_llm_scheduler.experiments.irace_space import decode_configuration
from geo_llm_scheduler.experiments.irace_target import evaluate_request, sha256


def candidates(root: Path) -> list[dict]:
    """Read the top candidate of each of six races; deduplicate without backfilling."""
    result = [dict(id="default", arm="default", configuration={})]
    for arm in ("numeric", "joint"):
        seen = set()
        for repetition in range(1, 4):
            path = root / f"{arm}_rep{repetition}" / "elites.csv"
            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            if not rows:
                raise ValueError(f"Empty elite export: {path}")
            for row in rows[:1]:
                config = {k: v for k, v in row.items() if not k.startswith(".")}
                decode_configuration(config)
                key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
                if key not in seen:
                    result.append(dict(id=f"{arm}_{key}", arm=arm, configuration=config))
                    seen.add(key)
    return result


def rank_finalists(configs: list[dict], scores: dict[str, list[float]]) -> list[dict]:
    """Select by balanced paired mean cost, complexity ties, and fallback to default.

    Caller supplies equal seeds per instance and equal instances per size, so the
    flat mean equals the prescribed seed/instance/size hierarchical mean.
    """
    lengths = {len(scores[c["id"]]) for c in configs}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Complete paired scores are required")
    if any(not math.isfinite(v) for values in scores.values() for v in values):
        raise ValueError("Nonfinite validation score")
    chosen = [c for c in configs if c["arm"] == "default"]
    baseline = chosen[0]
    default = asdict(Components())
    means = {c["id"]: sum(scores[c["id"]]) / len(scores[c["id"]]) for c in configs}
    for arm in ("numeric", "joint"):
        group = [c for c in configs if c["arm"] == arm]
        if not group:
            raise ValueError(f"Missing arm: {arm}")
        best = min(means[c["id"]] for c in group)
        tied = [c for c in group if means[c["id"]] <= best + 1e-12]
        winner = min(
            tied,
            key=lambda c: (
                sum(
                    v != default[k]
                    for k, v in asdict(decode_configuration(c["configuration"])).items()
                ),
                c["id"],
            ),
        )
        if means[winner["id"]] >= means[baseline["id"]] - 1e-12:
            winner = dict(id=f"{arm}_fallback_default", arm=arm, configuration={}, fallback=True)
        chosen.append(winner)
    return chosen


def main() -> None:
    """Plan or run full-budget paired evaluation; locked evaluation needs complete validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=("validation", "evaluation_locked"))
    parser.add_argument(
        "--run", action="store_true", help="Without this flag only freeze and print the task matrix"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    seal = root / "seal.json"
    verify_seal(seal)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if args.phase == "validation":
        configs = candidates(root)
    else:
        selection = root / "validation" / "selection.json"
        completed = json.loads((root / "validation" / "complete.json").read_text(encoding="utf-8"))
        if completed["selection_sha256"] != sha256(selection) or completed["seal_sha256"] != sha256(
            seal
        ):
            raise ValueError("Validation selection identity mismatch")
        configs = json.loads(selection.read_text(encoding="utf-8"))
    cases = [c for c in manifest["cases"] if c["split"] == args.phase]
    if not cases:
        raise ValueError("No cases in requested split")
    seeds = [27101, 27102, 27103] if args.phase == "validation" else list(range(37101, 37111))
    identity = dict(
        phase=args.phase, seal_sha256=sha256(seal), configurations=configs, cases=cases, seeds=seeds
    )
    output = root / args.phase
    output.mkdir(exist_ok=True)
    matrix = output / "matrix.json"
    if matrix.exists() and json.loads(matrix.read_text(encoding="utf-8")) != identity:
        raise ValueError("Evaluation task matrix changed")
    atomic_json(matrix, identity)
    print(f"{len(configs)} configurations x {len(cases)} instances x {len(seeds)} seeds")
    if not args.run:
        return
    scores: dict[str, list[float]] = {c["id"]: [] for c in configs}
    records = []
    for c in cases:
        for seed in seeds:
            for config in configs:
                run_dir = output / "runs" / config["id"] / Path(c["path"]).parent.name / str(seed)
                record = evaluate_request(
                    dict(configuration=config["configuration"], seed=seed, seal=str(seal)),
                    Path(c["path"]),
                    run_dir,
                )
                scores[config["id"]].append(record["cost"])
                records.append(
                    dict(configuration=config["id"], case=c["path"], seed=seed, cost=record["cost"])
                )
                atomic_json(
                    output / "progress.json",
                    dict(completed=len(records), total=len(configs) * len(cases) * len(seeds)),
                )
    atomic_json(output / "scores.json", dict(scores=scores, records=records))
    complete = dict(seal_sha256=sha256(seal), scores_sha256=sha256(output / "scores.json"))
    if args.phase == "validation":
        atomic_json(output / "selection.json", rank_finalists(configs, scores))
        complete["selection_sha256"] = sha256(output / "selection.json")
    atomic_json(output / "complete.json", complete)
    print(output / "scores.json")


if __name__ == "__main__":
    main()
