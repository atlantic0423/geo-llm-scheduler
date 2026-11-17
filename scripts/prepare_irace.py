"""Freeze a local irace dataset, reference frames, categorical space and scenarios."""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_search_v4 import Components, run_components
from geo_llm_scheduler.experiments.irace_space import default_table, parameter_text
from geo_llm_scheduler.experiments.irace_target import metric_spec, sha256
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance


def prepare(root: Path, split_path: Path, calibration_path: Path, *, smoke: bool = False) -> None:
    """Freeze all supplied cases; resume reference preparation only with identical inputs.

    Metric reference runs use the unchanged default and one complete generation.
    They are separately charged setup runs, not performance evidence. Inner formal
    tuning always uses the full 600/1200/2400 seconds without generation caps.
    """
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    folds = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    identity = dict(
        split_sha256=sha256(split_path),
        calibration_sha256=sha256(calibration_path),
        purpose="smoke" if smoke else "formal",
    )
    lock = root / "preparation.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != identity:
        raise ValueError("Preparation input identity changed")
    atomic_json(lock, identity)
    rows = sorted(split["instances"], key=lambda r: r["instance_id"])
    if smoke:
        rows = [r for r in rows if r["split"] == "configuration"][:3]
    counters: Counter = Counter()
    cases = []
    for row in rows:
        key, n = row["instance_id"], row["jobs"]
        original = Path(row["path"])
        if sha256(original) != row["sha256"]:
            raise ValueError(f"Instance hash mismatch: {key}")
        fold = f"n{n}_f{counters[n] % 2}"
        counters[n] += 1
        cal = folds[fold]
        if key in cal.get("keys", []):
            raise ValueError("Evaluation instance overlaps calibration fitting set")
        directory = root / "cases" / key
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / "instance.json"
        shutil.copyfile(original, input_path)
        atomic_json(directory / "calibration.json", cal)
        reference_path = directory / "reference.json"
        if not reference_path.exists():
            problem = load_instance(input_path)
            evidence: dict = {}
            baseline = run_components(
                problem,
                Components(),
                20261117,
                {50: 600, 100: 1200, 200: 2400}[n],
                cal,
                evidence,
                generations=1,
            )
            atomic_json(
                reference_path,
                dict(
                    input_sha256=sha256(input_path),
                    calibration_sha256=sha256(directory / "calibration.json"),
                    reference_protocol="default_one_generation_setup_only",
                    elapsed=baseline.elapsed,
                    termination=baseline.termination_reason,
                    archive=[solution(c) for c in baseline.archive.members],
                    evidence=evidence,
                ),
            )
        baseline_data = json.loads(reference_path.read_text(encoding="utf-8"))
        if baseline_data["input_sha256"] != sha256(input_path) or baseline_data[
            "calibration_sha256"
        ] != sha256(directory / "calibration.json"):
            raise ValueError("Reference identity mismatch")
        points = [
            (c["evaluation"]["flow"], c["evaluation"]["tou"] + sum(c["evaluation"]["demand"]))
            for c in baseline_data["archive"]
        ]
        metric = metric_spec(
            points,
            dict(protocol=baseline_data["reference_protocol"], sha256=sha256(reference_path)),
        )
        atomic_json(directory / "metric.json", metric)
        case = dict(
            purpose=identity["purpose"],
            instance_id=key,
            split=row["split"],
            seconds=2.0 if smoke else {50: 600, 100: 1200, 200: 2400}[n],
            calibration_fold=fold,
        )
        for name in ("instance", "calibration", "metric"):
            item = directory / f"{name}.json"
            case[name] = dict(path=item.name, sha256=sha256(item))
        atomic_json(directory / "case.json", case)
        cases.append(
            dict(
                path=str(directory / "case.json"),
                sha256=sha256(directory / "case.json"),
                jobs=n,
                family=row["family"],
                replicate=row["replicate"],
                split=row["split"],
            )
        )
        print(f"Prepared {len(cases)}/{len(rows)}: {key}", flush=True)
    # Interleave sizes per family/replicate: every 3-instance block covers all three sizes.
    training = sorted(
        (c for c in cases if c["split"] == "configuration"),
        key=lambda c: (c["family"], c["replicate"], c["jobs"]),
    )
    (root / "training.txt").write_text(
        "\n".join(c["path"] for c in training) + "\n", encoding="utf-8"
    )
    repo = Path(__file__).resolve().parents[1]
    for mode in ("numeric", "joint"):
        (root / f"parameters_{mode}.txt").write_text(parameter_text(mode), encoding="utf-8")
        (root / f"default_{mode}.txt").write_text(default_table(mode), encoding="utf-8")
        settings = dict(
            python=sys.executable,
            src=str(repo / "src"),
            target=str(repo / "scripts" / "irace_target.py"),
            parameters=str(root / f"parameters_{mode}.txt"),
            defaults=str(root / f"default_{mode}.txt"),
            instances=str(root / "training.txt"),
            output=str(root / f"{mode}_seed1"),
            seed=17101,
            max_experiments=48 if smoke else 2400,
            block_size=1 if smoke else 3,
            first_test=2 if smoke else 5,
        )
        if smoke:
            settings.update(nb_configurations=4, nb_iterations=1, min_survival=2)
        atomic_json(root / f"campaign_{mode}.json", settings)
    atomic_json(
        root / "manifest.json",
        dict(
            **identity,
            cases=cases,
            reference_cost_seconds=sum(
                json.loads((Path(c["path"]).parent / "reference.json").read_text(encoding="utf-8"))[
                    "elapsed"
                ]
                for c in cases
            ),
            interpretation="Prepared inputs, not completed configuration experiments",
        ),
    )


def main() -> None:
    """Prepare a frozen formal panel or explicitly marked small integration smoke test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.root, args.split, args.calibration, smoke=args.smoke)


if __name__ == "__main__":
    main()
