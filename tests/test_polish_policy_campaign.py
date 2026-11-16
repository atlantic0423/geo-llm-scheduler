"""Frozen original-instance binding and run-level provenance for the five-arm campaign."""

import importlib.util
import json
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments.right_shift_policies import ARMS, policy_config

spec = importlib.util.spec_from_file_location(
    "policy_campaign", Path(__file__).parents[1] / "scripts/run_polish_policies.py"
)
campaign = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(campaign)


def historical_fixture(root):
    (root / "inputs").mkdir(parents=True)
    (root / "specs").mkdir()
    blocks = []
    files = {}
    for i in range(32):
        n = (20, 50, 100, 200)[i % 4]
        key = f"case{i}_s16101"
        input_path = root / "inputs" / f"b{i % 16}_v{i // 16}.json"
        input_path.write_text('{"original":' + str(i) + "}")
        s = dict(
            key=key,
            n=n,
            seed=16101,
            seconds={20: 300, 50: 600, 100: 1200, 200: 2400}[n],
            regions=3,
            machines=12,
            pressure="arrival",
            variant=i // 16,
            base_seed=i % 16,
            input=input_path.relative_to(root).as_posix(),
            input_sha256=campaign.sha(input_path),
        )
        p = root / "specs" / (key + ".json")
        campaign.atomic_json(p, s)
        blocks.append(key)
        files[p.relative_to(root).as_posix()] = campaign.sha(p)
        files[s["input"]] = s["input_sha256"]
    campaign.atomic_json(
        root / "manifest.json",
        dict(
            commit="b148c4addd207fe30a6b2a43d953a04afc35d3a1", blocks=blocks, runs=1056, files=files
        ),
    )
    campaign.atomic_json(
        root / "manifest.sha256.json", dict(sha256=campaign.sha(root / "manifest.json"))
    )


def test_original_bytes_64_blocks_two_seeds_and_all_five_arms(tmp_path):
    old = tmp_path / "old"
    historical_fixture(old)
    root = tmp_path / "new"
    m = campaign.prepare(root, old)
    assert len(m["blocks"]) == 64 and m["runs"] == 320 and m["arms"] == list(ARMS)
    assert campaign.validate_inputs(root) == m
    specs = [json.loads((root / "specs" / (k + ".json")).read_text()) for k in m["blocks"]]
    assert {s["seed"] for s in specs} == {16101, 26101}
    assert [s["n"] for s in specs[:8]] == [20, 50, 100, 200] * 2
    assert len(campaign.queue(root, m)) == len(set(campaign.queue(root, m))) == 320
    for s in specs:
        assert (root / s["input"]).read_bytes() == (old / s["input"]).read_bytes()
    with pytest.raises(FileExistsError):
        campaign.prepare(root, old)
    (root / specs[0]["input"]).write_text("changed")
    with pytest.raises(AssertionError):
        campaign.validate_inputs(root)


def test_pilot_stays_separate_and_historical_corruption_is_rejected(tmp_path):
    old = tmp_path / "old"
    historical_fixture(old)
    m = campaign.prepare(tmp_path / "pilot", old, pilot=True)
    assert m["kind"] == "pilot" and m["runs"] == 20
    assert m["sampling_probabilities"] == dict(conditional_ls=0.05, conditional_no_ls=0.10)
    (old / "inputs/b0_v0.json").write_text("corrupt")
    with pytest.raises(AssertionError):
        campaign.prepare(tmp_path / "bad", old)


def test_configuration_only_polish_placement_varies():
    full = [policy_config(a, 16101, 300) for a in ARMS if a != "BILL_MOEAD"]
    assert all(c == full[0] for c in full)
    assert not full[0].polish and full[0].rl_steps == 5 and full[0].fixed_budget == 6
    assert full[0].a3_region_policy == "tariff" and full[0].replacement_policy == "permuted"
    assert policy_config("BILL_MOEAD", 16101, 300).method == "plain"
