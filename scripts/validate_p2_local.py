"""Validate opt-in arms and independently compare defaults with the frozen base."""

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.p2_validation import validate_local_p2
from geo_llm_scheduler.experiments.runner import digest
from geo_llm_scheduler.experiments.synthetic import synthetic
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager

BASE_SHA = "3ec2355fbc43993375daab06ec0ea39fb9d86991"
NEW_FIELDS = (
    "offspring_policy",
    "clone_probability",
    "replacement_policy",
    "a7_representative_policy",
)


def _legacy_check(output: Path) -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    archive = subprocess.check_output(["git", "archive", BASE_SHA, "src"], cwd=root)
    checks = []
    with tempfile.TemporaryDirectory(prefix="p2_base_", dir=output) as tmp:
        baseline = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            handle.extractall(baseline, filter="data")
        problem = synthetic(6, 2, 2, 91)
        instance = output / "baseline_instance.json"
        save_instance(problem, instance)
        for method in ("plain", "full", "nsga2"):
            config = Config(
                method=method,
                population=6,
                neighborhood=3,
                generations=2,
                seed=91,
                trigger_mode="always",
                rl_steps=2,
            )
            initial = initial_genotypes(problem, config, RNGManager(91).stream("initialization"))
            initial_path = output / f"baseline_initial_{method}.json"
            initial_path.write_text(json.dumps([asdict(g) for g in initial]), encoding="utf-8")
            config_data = asdict(config)
            for field in NEW_FIELDS:
                del config_data[field]
            spec_path = output / f"baseline_spec_{method}.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "config": config_data,
                        "instance_path": str(instance),
                        "initial_path": str(initial_path),
                    }
                ),
                encoding="utf-8",
            )
            signatures = []
            for name, source in (("old", baseline / "src"), ("new", root / "src")):
                target = output / f"baseline_{method}_{name}.json"
                env = {**os.environ, "PYTHONPATH": str(source)}
                subprocess.run(
                    [
                        sys.executable,
                        str(root / "scripts/p1_legacy_probe.py"),
                        str(spec_path),
                        str(target),
                    ],
                    env=env,
                    cwd=root,
                    check=True,
                )
                signatures.append(json.loads(target.read_text(encoding="utf-8")))
            if signatures[0] != signatures[1]:
                raise AssertionError(f"Defaults differ from frozen baseline: {method}")
            checks.append(
                {
                    "method": method,
                    "equal": True,
                    "signature_hash": digest(signatures[0]),
                    "exact": signatures[0]["counts"]["exact"],
                }
            )
    return checks


def main() -> None:
    """Write bounded replay/ground-truth/baseline evidence, without remote actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/p2_local_validation"))
    args = parser.parse_args()
    output = args.output.resolve()
    report = validate_local_p2(output)
    report["baseline_commit"] = BASE_SHA
    report["baseline_checks"] = _legacy_check(output)
    (output / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "runs": report["runs"],
                "replay_equal": True,
                "audited_exact": sum(sum(r["audited_exact"]) for r in report["rows"]),
                "baseline_checks": report["baseline_checks"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
