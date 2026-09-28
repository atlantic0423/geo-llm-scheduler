"""E15 frozen matrix, completed skip and incomplete quarantine integration."""

import json

from geo_llm_scheduler.experiments.e15_worker import validate_e15_result
from scripts import run_e15_campaign as campaign


def test_cgroup_gate_excludes_active_file_cache_but_counts_shared_memory():
    maximum = 32 * 1024**3
    current = 31 * 1024**3
    fields = {"file": str(29 * 1024**3), "inactive_file": str(4 * 1024**3), "shmem": str(1024**3)}
    # A cache-heavy host remains dispatchable: 2 GiB non-file + 1 GiB shared.
    assert campaign._charged_memory_fraction(current, maximum, fields) == 3 / 32
    assert campaign._charged_memory_fraction(maximum, maximum, {"file": "0"}) == 1


def test_e15_shared_initialization_resume_and_quarantine(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "MIN_FREE_GB", 0.0)
    root = tmp_path / "e15"
    manifest = campaign.prepare(root, small=True)
    pilot = {"T50_wallclock": 1, "T100_wallclock": 1}
    specs = campaign.stage_specs(root, manifest, "A_GEN", pilot, None)
    assert len(specs) == 24
    paired = [
        s
        for s in specs
        if s.instance_seed == specs[0].instance_seed and s.algorithm_seed == specs[0].algorithm_seed
    ]
    assert len({s.initial_hash for s in paired}) == 1
    assert len({s.config["method"] for s in paired}) == 1
    first = specs[0]
    assert campaign.run_batch(root, [first], 1, "mini")
    output = root / "runs" / first.key
    valid, reason = validate_e15_result(output, first)
    assert valid, reason
    marker = json.loads((output / "complete.json").read_text())
    assert campaign.run_batch(root, [first], 1, "mini_resume")
    assert json.loads((output / "complete.json").read_text()) == marker
    (output / "summary.json").write_text("{}", encoding="utf-8")
    assert campaign.run_batch(root, [first], 1, "mini_repair")
    assert validate_e15_result(output, first)[0]
    assert list((root / "quarantine").iterdir())
