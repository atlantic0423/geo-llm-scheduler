"""Five-generation BILL+MOEA/D checkpoint probes on ten predeclared medium inputs.

This is a saturation diagnostic, not a matched-time algorithm comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geo_llm_scheduler.archive.pareto import Archive  # noqa: E402
from geo_llm_scheduler.engine.evaluation import EvaluationGateway  # noqa: E402
from geo_llm_scheduler.experiments.lsv2_eligibility_selection import PANEL  # noqa: E402
from geo_llm_scheduler.experiments.lsv2_study import operator_config  # noqa: E402
from geo_llm_scheduler.experiments.mechanism_audit import (  # noqa: E402
    audit_problem,
    probe_candidates,
)
from geo_llm_scheduler.experiments.slot_assignment import run_arm, study_config  # noqa: E402
from geo_llm_scheduler.io.loaders import load_instance  # noqa: E402
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish  # noqa: E402
from geo_llm_scheduler.lsv2.frontier import FrozenFrame  # noqa: E402
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search  # noqa: E402
from geo_llm_scheduler.utils.rng import RNGManager  # noqa: E402


def _one(root: str, key: str) -> str:
    base = Path(root)
    problem = load_instance(base / "instances" / (key + ".json"))
    prior = json.loads((base / "audits" / (key + ".json")).read_text(encoding="utf-8"))
    frame = FrozenFrame(tuple(prior["hv_frame"]["ideal"]), tuple(prior["hv_frame"]["nadir"]))
    cfg = replace(study_config("BILL_RANDOM", 16901, 600), seconds=None, generations=5)
    evidence: dict = {}
    tick = perf_counter()
    result = run_arm(problem, "BILL_RANDOM", cfg, evidence)
    assert (
        evidence["initial_population_sha256"]
        == prior["initialization"]["initial_population_sha256"]
    )
    front = [c.evaluation.objectives for c in result.archive.members]
    opcfg = operator_config(16901, 600)
    search = LSV2Search(LSV2Config(budget=3, trajectory_steps=3), cfg, opcfg)
    streams = RNGManager(1016901)
    rows, points, witnesses = [], list(front), []
    for index, c in enumerate(probe_candidates(result.archive.members)):
        for name in ("RIGHTSHIFT", "A1", "A2", "A3", "A4", "A5", "A7", "A8", "P2_FLOW", "P2_MIXED"):
            rng = streams.stream(f"checkpoint:{index}:{name}")
            batch = (
                ReusedCommonPolish().propose(problem, c, 3, opcfg, rng)
                if name == "RIGHTSHIFT"
                else search._propose(name, c, frame, rng, problem)
            )
            gateway = EvaluationGateway(problem, Archive())
            trials = [gateway.evaluate(p.genotype, p.schedule, name) for p in batch.proposals]
            feasible = [t for t in trials if t.evaluation.feasible]
            new = [t.evaluation.objectives for t in feasible]
            points.extend(new)
            if name == "RIGHTSHIFT":
                assert all(
                    t.evaluation.feasible
                    and t.evaluation.flow == c.evaluation.flow
                    and t.evaluation.bill < c.evaluation.bill
                    for t in trials
                )
                if trials:
                    witnesses.append(
                        {
                            "before": asdict(c),
                            "after": asdict(min(trials, key=lambda t: t.evaluation.bill)),
                        }
                    )
            rows.append(
                {
                    "parent": index,
                    "operator": name,
                    "proposals": len(trials),
                    "feasible": len(feasible),
                    "hv_gain": frame.hv_gain(front, new),
                }
            )
    output = {
        "status": "complete",
        "key": key,
        "generations": 5,
        "offspring": result.offspring_count,
        "elapsed_s": perf_counter() - tick,
        "termination_reason": result.termination_reason,
        "initial_population_sha256": evidence["initial_population_sha256"],
        "input_sha256": prior["input_sha256"],
        "baseline_front": front,
        "baseline_hv_gain_from_initial": frame.hv(front) - prior["initial_hv"],
        "probe_hv_gain_from_baseline": frame.hv_gain(front, points),
        "rows": rows,
        "rightshift_witnesses": witnesses,
    }
    (base / "baseline_probes" / (key + ".json")).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return key


def _historical(source: Path, output: Path, key: str) -> str:
    path = source / (key + ".json")
    result = audit_problem(load_instance(path), operators=False)
    result["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    target = output / "historical_probes" / path.name
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return key


def main() -> None:
    """Audit one predeclared 100-job input per family, alternating 3/5 regions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-inputs", type=Path)
    args = parser.parse_args()
    (args.output / "baseline_probes").mkdir(exist_ok=True)
    keys = [f"mx_n100_f{i:02d}_r{1 if i % 2 else 3:02d}" for i in range(1, 11)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_one, str(args.output), key)
            for key in keys
            if not (args.output / "baseline_probes" / (key + ".json")).exists()
        ]
        if args.historical_inputs:
            (args.output / "historical_probes").mkdir(exist_ok=True)
            for n, designs in PANEL.items():
                for i in designs:
                    key = f"st_n{n}_d{i:02d}"
                    path = args.output / "historical_probes" / (key + ".json")
                    if path.exists():
                        old = json.loads(path.read_text(encoding="utf-8"))
                        assert (
                            old["input_sha256"]
                            == hashlib.sha256(
                                (args.historical_inputs / path.name).read_bytes()
                            ).hexdigest()
                        )
                    else:
                        futures.append(
                            pool.submit(_historical, args.historical_inputs, args.output, key)
                        )
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
