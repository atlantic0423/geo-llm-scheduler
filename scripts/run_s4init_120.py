"""Eight baselines with exact S4 BILL4 initialization on the frozen seed-one panel."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from run_lsv2 import source_hash
from slot_assignment_dispatch import dispatch

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import ext_hybrid, s4_ablation
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.trans_hybrid import HybridSearchContext
from geo_llm_scheduler.experiments.trans_runner import (
    S4INIT_EXTERNAL,
    S4INIT_NATIVE,
    digest,
    run_case,
)
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager

PANEL = "1c17b9edadc5f2f334390c95cd0ab2106c323a4c7f5032153a1b46bf79854484"
ARMS = (*S4INIT_EXTERNAL, *S4INIT_NATIVE)
REPO = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def config(arm: str, seconds: int) -> dict:
    if arm in S4INIT_EXTERNAL:
        return asdict(ext_hybrid.arm_config(S4INIT_EXTERNAL[arm], 1, seconds))
    return dict(
        name=arm, seed=1, seconds=seconds, population=100, wait_genes=True, max_generations=None
    )


def prepare(root: Path, panel: Path, host: str, reference: Path) -> None:
    """Freeze source, inputs, matrix, per-task configuration and time budgets."""
    assert digest(panel / "manifest.json") == PANEL
    rows = sorted(read(panel / "manifest.json")["instances"], key=lambda r: r["key"])
    assert Counter(r["jobs"] for r in rows) == {50: 40, 100: 40, 200: 40}
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(panel / "manifest.json", root / "panel_manifest.json")
    files = {"panel_manifest.json": digest(root / "panel_manifest.json")}
    blocks = []
    for i, row in enumerate(rows):
        key, n = row["key"], row["jobs"]
        assert key not in blocks and Path(row["path"]).as_posix() == f"instances/{key}.json"
        blocks.append(key)
        target = root / row["path"]
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(panel / row["path"], target)
        assert digest(target) == row["sha256"]
        files[row["path"]] = row["sha256"]
        seconds = s4_ablation.BUDGETS[n]
        old = reference / "runs" / key / "A0_FULL_S4"
        old_config, old_init = read(old / "config.json"), read(old / "initialization.json")
        assert old_config["instance_sha256"] == row["sha256"] and old_config["seed"] == 1
        assert old_config["seconds_including_initialization"] == seconds
        old_files = {x["path"]: x["sha256"] for x in read(old / "manifest.json")}
        assert digest(old / "initialization.json") == old_files["initialization.json"]
        expected_init = {
            k: old_init[k] for k in ("initial_population_sha256", "initial_archive_sha256")
        }
        spec = dict(
            key=key,
            n=n,
            seed=1,
            input=row["path"],
            input_sha256=row["sha256"],
            seconds=seconds,
            order=ARMS[i % 8 :] + ARMS[: i % 8],
            config={a: config(a, seconds) for a in ARMS},
            initializer="s4_bill4_exact",
            reference_initialization=expected_init,
            entry="scripts/run_s4init_120.py worker --root ROOT --key KEY --arm ARM --cpu CPU",
        )
        p = root / "specs" / (key + ".json")
        atomic_json(p, spec)
        files[p.relative_to(root).as_posix()] = digest(p)
    m = dict(
        source=source_hash(),
        commit=head(),
        files=files,
        arms=ARMS,
        blocks=blocks,
        runs=960,
        seed=1,
        workers=43,
        host=host,
        window_seconds=72 * 3600,
        solver_seconds=sum(40 * t * 8 for t in s4_ablation.BUDGETS.values()),
    )
    atomic_json(root / "manifest.json", m)
    atomic_json(root / "manifest.sha256.json", {"sha256": digest(root / "manifest.json")})


def validate_inputs(root: Path) -> dict:
    """Verify all frozen files, configurations, instance sizes and task counts."""
    assert digest(root / "manifest.json") == read(root / "manifest.sha256.json")["sha256"]
    m = read(root / "manifest.json")
    assert m["source"] == source_hash() and m["commit"] == head() and tuple(m["arms"]) == ARMS
    assert m["runs"] == 960 and len(set(m["blocks"])) == 120 and m["solver_seconds"] == 1344000
    assert digest(root / "panel_manifest.json") == PANEL
    for name, sha in m["files"].items():
        assert digest(root / name) == sha, name
    for key in m["blocks"]:
        s = read(root / "specs" / (key + ".json"))
        assert s["seed"] == 1 and s["seconds"] == s4_ablation.BUDGETS[s["n"]]
        assert len(s["order"]) == 8 and set(s["order"]) == set(ARMS)
        for arm in ARMS:
            assert s["config"][arm] == json.loads(json.dumps(config(arm, s["seconds"])))
    return m


def preflight(root: Path, key: str) -> None:
    """Independently audit the common 100 members and continuous-key replay."""
    m, s = read(root / "manifest.json"), read(root / "specs" / (key + ".json"))
    assert digest(root / s["input"]) == s["input_sha256"]
    problem = load_instance(root / s["input"])
    cfg = ext_hybrid.arm_config("EXTH_MOEAD", 1, s["seconds"])
    gw = EvaluationGateway(problem, Archive(), decode_cache_size=cfg.decode_cache_size)
    evidence: dict = {}
    members = ext_hybrid.s4_initializer(problem, cfg, evidence)(gw, RNGManager(1))
    assert evidence["policy"] == "bill4" and evidence["original_parent_count"] == 100
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert evidence[field] == s["reference_initialization"][field], (key, field)
    sample = members[::20] + list(gw.archive.members[:3])
    for c in sample:
        assert c.evaluation == evaluate(problem, c.genotype, c.schedule)
    ctx = HybridSearchContext(problem, seed=1, seconds=s["seconds"], s4_init=True)
    replay = ctx.initialize(100)
    assert (
        ctx.hybrid_evidence[0]["initial_population_sha256"] == evidence["initial_population_sha256"]
    )
    assert ctx.hybrid_evidence[0]["initial_archive_sha256"] == evidence["initial_archive_sha256"]
    for c in replay[::20]:
        new = ctx.evaluate(c.genes).candidate
        assert (
            new.genotype == c.candidate.genotype
            and new.schedule.starts == c.candidate.schedule.starts
        )
        assert new.evaluation == c.candidate.evaluation
    p = root / "preflight" / (key + ".json")
    atomic_json(
        p,
        dict(
            source=m["source"],
            input_sha256=s["input_sha256"],
            host=socket.gethostname(),
            initial_population_sha256=evidence["initial_population_sha256"],
            initial_archive_sha256=evidence["initial_archive_sha256"],
            fresh_exact=len(sample),
            replay=5,
        ),
    )
    atomic_json(p.with_suffix(".sha256.json"), {"sha256": digest(p)})


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Check sealed outputs before treating a task as completed."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").is_file():
        return False
    seal = read(out / "complete.json")
    assert seal["source"] == source and seal["key"] == key and seal["arm"] == arm
    assert digest(out / "manifest.json") == seal["manifest"]
    for item in json.loads((out / "manifest.json").read_text()):
        p = out / item["path"]
        assert p.stat().st_size == item["bytes"] and digest(p) == item["sha256"]
    return True


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Execute one seed-one task, verify initial fingerprints, then seal results."""
    os.sched_setaffinity(0, {cpu})
    m = read(root / "manifest.json")
    assert digest(root / "manifest.json") == read(root / "manifest.sha256.json")["sha256"]
    sp = root / "specs" / (key + ".json")
    assert (
        m["source"] == source_hash()
        and m["commit"] == head()
        and digest(sp) == m["files"][f"specs/{key}.json"]
    )
    s = read(sp)
    assert arm in s["order"] and digest(root / s["input"]) == s["input_sha256"]
    pp = root / "preflight" / (key + ".json")
    assert digest(pp) == read(pp.with_suffix(".sha256.json"))["sha256"]
    pre = read(pp)
    assert (
        pre["host"] == socket.gethostname() == m["host"]
        and pre["input_sha256"] == s["input_sha256"]
    )
    out = root / "runs" / key / arm
    summary = run_case(
        arm, root / s["input"], out, seed=1, seconds=s["seconds"], campaign_output=True
    )
    c = read(out / "config.json")
    assert c["source"]["commit"] == m["commit"] and c["source"]["status"] == ""
    assert (
        c["seed"] == 1
        and not c["engineering_override"]
        and c["seconds_including_initialization"] == s["seconds"]
    )
    assert (
        summary["independent_exact_audit"] == "passed"
        and summary["elapsed_seconds"] >= s["seconds"] - 0.05
    )
    init = read(out / "initialization.json")
    if arm in S4INIT_NATIVE:
        init = init[0]
    else:
        assert read(out / "algorithm_config.json") == s["config"][arm]
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert init[field] == pre[field], (arm, field)
    atomic_json(
        out / "complete.json",
        dict(
            source=m["source"],
            key=key,
            arm=arm,
            manifest=digest(out / "manifest.json"),
            spec=digest(sp),
        ),
    )


def main() -> None:
    """Prepare, preflight, validate, execute or dispatch the frozen campaign."""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "validate", "preflight", "worker", "dispatch"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--panel", type=Path)
    p.add_argument("--host", default="neu01-2288H-V6")
    p.add_argument("--reference", type=Path)
    p.add_argument("--key")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--cpu", type=int)
    p.add_argument("--cpus")
    a = p.parse_args()
    if a.action == "prepare":
        if a.reference is None or a.panel is None:
            p.error("prepare requires --panel and --reference")
        prepare(a.root, a.panel, a.host, a.reference)
    elif a.action == "validate":
        print(validate_inputs(a.root)["runs"])
    elif a.action == "preflight":
        preflight(a.root, a.key)
    elif a.action == "worker":
        worker(a.root, a.key, a.arm, a.cpu)
    else:
        cpus = [int(c) for c in a.cpus.split(",")]
        topo = {
            tuple(
                (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
                for f in ("physical_package_id", "core_id")
            )
            for c in cpus
        }
        assert len(cpus) == len(topo) <= 43 and set(cpus) <= os.sched_getaffinity(0)
        result = dispatch(a.root, sys.modules[__name__], cpus, Path(__file__).resolve(), interval=2)
        print(json.dumps(result), flush=True)
        if result["state"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
