"""End-to-end fixture for the read-only regional-ledger reconciliation script."""

import gzip
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from geo_llm_scheduler.domain.models import (
    Candidate,
    Genotype,
    Job,
    PowerModel,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.trans_runner import digest
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.scheduling.ssgs import decode

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem() -> ProblemInstance:
    tariffs = (Tariff(0, 700, 0.2), Tariff(700, 2000, 0.9), Tariff(2000, 100000, 0.4))
    return ProblemInstance(
        tuple(
            Job(str(i), 0 if i < 3 else 650, Profile(150 + 50 * i, 0.6, 2), Profile(300, 0.5, 2))
            for i in range(6)
        ),
        tuple(Region(str(r), tariffs, 2 + r) for r in range(2)),
        tuple(ServingInstance(str(m), m // 2, 8, 10 + m, 100) for m in range(4)),
        PowerModel(0.4, 0.8, 0.6),
    )


def _candidate(p: ProblemInstance, rng: random.Random, regions: list[int]) -> Candidate:
    os = [j for j in range(len(p.jobs)) for _ in range(2)]
    rng.shuffle(os)
    ms = tuple(2 * regions[j] + rng.randrange(2) for j in range(len(p.jobs)) for _ in range(2))
    g = Genotype(ms, tuple(os))
    s = decode(p, g)
    return Candidate(g, s, evaluate(p, g, s), "fixture")


def test_reconcile_run_agrees_with_exact_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "r1_offline", _load("r1_offline"))
    module = _load("reconcile_r1_incremental")
    p = _problem()
    rng = random.Random(3)
    shard = tmp_path / "formal_panel_shard0"
    run = shard / "runs" / "tiny_s17" / module.ARM
    run.mkdir(parents=True)
    (shard / "instances").mkdir()
    instance = shard / "instances" / "tiny.json"
    save_instance(p, instance)
    regions = [0, 0, 1, 1, 0, 1]
    snapshots = []
    for stage in range(3):
        donor = _candidate(p, rng, regions)
        receivers = [_candidate(p, rng, regions) for _ in range(2)]
        front = sorted({c.evaluation.objectives for c in [donor, *receivers]})
        snapshots.append(
            dict(
                stage=stage,
                subproblem=stage,
                donor=asdict(donor),
                front=[list(x) for x in front],
                context=dict(
                    ideal=[min(x[k] for x in front) for k in range(2)],
                    maximum=[max(x[k] for x in front) for k in range(2)],
                ),
                receivers=[
                    dict(candidate=asdict(c), weight=[0.3 + 0.4 * i, 0.7 - 0.4 * i], slot=i)
                    for i, c in enumerate(receivers)
                ],
            )
        )
    with gzip.open(run / "regional_observation.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(dict(snapshots=snapshots), stream)
    (run / "config.json").write_text(
        json.dumps(
            dict(
                instance="tiny.json",
                instance_sha256=digest(instance),
                source=dict(commit="fixture"),
            )
        )
    )
    sealed = list(run.iterdir())
    manifest = [dict(path=f.name, bytes=f.stat().st_size, sha256=digest(f)) for f in sealed]
    (run / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "out"
    out.mkdir()
    done = module.reconcile_run((str(run), str(out)))
    assert done["trials"] == 3 * 2 * 2
    report = json.loads((out / "tiny_s17.json").read_text(encoding="utf-8"))
    for row in report["rows"]:
        assert abs(row["flow_error"]) < 1e-6 and abs(row["bill_error"]) < 1e-9
        assert row["horizon_error"] == 0
        assert row["exact_improve"] == row["predicted_improve"]
        assert row["exact_hv"] == row["predicted_hv"]
        assert not row["neutral"] or row["neutral_exact"]
    summary = module.summarize([report])["6"]
    assert summary["trials"] == 12 and summary["missed_by_screen"] == 0
    assert summary["improve_disagree"] == summary["hv_disagree"] == 0
    assert 0 <= summary["screen_pass_rate"] <= 1
    # A tampered sealed file is rejected before any replay.
    (run / "config.json").write_text("{}")
    with pytest.raises((AssertionError, KeyError)):
        module.reconcile_run((str(run), str(out)))
