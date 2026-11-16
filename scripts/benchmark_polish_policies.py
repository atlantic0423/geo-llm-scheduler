"""Paired fixed-work exact-reuse benchmark on SHA-bound component33 parents."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.operators.polish_policies import policy_polish
from geo_llm_scheduler.operators.right_shift import PolishStats
from geo_llm_scheduler.utils.numeric import identity


def file_sha(path: Path) -> str:
    """Read the full file SHA256 using bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Compare each policy with/without engineering reuse, never conflate policies."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.historical
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["commit"] == "b148c4addd207fe30a6b2a43d953a04afc35d3a1"
    assert (
        file_sha(root / "manifest.json")
        == json.loads((root / "manifest.sha256.json").read_text())["sha256"]
    )
    rows = []
    for index, key in enumerate(manifest["blocks"][:4]):
        spec_path = root / "specs" / (key + ".json")
        assert file_sha(spec_path) == manifest["files"][spec_path.relative_to(root).as_posix()]
        spec = json.loads(spec_path.read_text())
        assert file_sha(root / spec["input"]) == spec["input_sha256"]
        problem = load_instance(root / spec["input"])
        path = root / "runs" / key / "FULL_CURRENT_I4" / "initial.json.gz"
        marker = json.loads((path.parent / "complete.json").read_text())
        assert file_sha(path) == marker["files"][path.name]["sha256"]
        parents = json.load(gzip.open(path, "rt"))
        for rank in (0, 90):
            record = parents[rank]
            g = Genotype(tuple(record["genotype"]["ms"]), tuple(record["genotype"]["os"]))
            s = Schedule(tuple(record["schedule"]["starts"]))
            parent = Candidate(g, s, evaluate(problem, g, s))
            assert parent.evaluation.feasible
            for repeat in range(args.repeats):
                for policy in ("reference_reuse", "two_focus", "sequential_scan"):
                    paired = {}
                    # Alternate execution order; benchmark work is identical
                    # within each pair, including every Archive opportunity.
                    for reuse in (
                        (False, True) if (index + rank + repeat) % 2 == 0 else (True, False)
                    ):
                        gateway = EvaluationGateway(problem, Archive())
                        gateway.archive.consider(parent)
                        gateway.ideal = parent.evaluation.objectives
                        candidates = []
                        gateway.candidate_sink = candidates.append
                        stats = PolishStats()
                        started = perf_counter()
                        result = policy_polish(problem, parent, gateway, policy, stats, reuse=reuse)
                        wall = perf_counter() - started
                        assert result.evaluation == evaluate(
                            problem, result.genotype, result.schedule
                        )
                        paired[reuse] = dict(
                            result=result,
                            candidates=candidates,
                            archive=gateway.archive.identity_keys(),
                            stats=stats,
                            wall=wall,
                            counts=dict(gateway.counts),
                        )
                    old, new = paired[False], paired[True]
                    assert identity(old["result"]) == identity(new["result"])
                    assert (
                        old["candidates"] == new["candidates"]
                        and old["archive"] == new["archive"]
                        and old["stats"] == new["stats"]
                    )
                    row = dict(
                        key=key,
                        n=spec["n"],
                        parent_index=rank,
                        repeat=repeat,
                        policy=policy,
                        full_seconds=old["wall"],
                        reuse_seconds=new["wall"],
                        ratio=new["wall"] / old["wall"],
                        candidates=old["stats"].candidates,
                        exact=old["stats"].exact,
                        accepted=old["stats"].accepted,
                        parent_bill=parent.evaluation.bill,
                        result_bill=new["result"].evaluation.bill,
                        all_candidates_bitwise_equal=True,
                        all_archive_equal=True,
                        parent_audits=new["counts"].get("polish_parent_audits", 0),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    (args.output / "benchmark.json").write_text(
                        json.dumps(dict(complete=False, rows=rows), indent=2), encoding="utf-8"
                    )
    (args.output / "benchmark.json").write_text(
        json.dumps(
            dict(
                complete=True,
                scope="fixed-work engineering benchmark on8 reused component33 initial solutions, not online quality or a holdout",
                rows=rows,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
