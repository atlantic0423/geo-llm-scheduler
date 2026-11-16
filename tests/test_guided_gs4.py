"""F3 GS4 (2026-11-11): fixed four destroyed Jobs plus the switch controller, one batch arm."""

import importlib
import json
from collections import Counter
from pathlib import Path

import pytest
from test_guided_cost import _run
from test_lsv2_fast_rightshift import CAL
from test_s4_load_shift import _panel

from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.trans_runner import digest
from geo_llm_scheduler.lsv2.guided_variation import GuidedOptions

GS4 = "GS4_GUIDED_RAND_SWITCH_D4"


def test_gs4_is_exactly_the_union_of_the_two_mechanisms():
    o = s4_moead.guided_options(GS4)
    assert o == GuidedOptions(
        direction_aware=False, destroy_count=4, controller="switch", record_paths=True
    )
    assert o.probability == 0.2 and o.p_probe == 0.02
    assert (o.tau_seconds, o.gamma, o.warmup_seconds) == (300.0, 2.0, 5.0)
    assert GS4 in s4_moead.COST_ARMS and GS4 in s4_moead.EXTRA_ARMS
    cfg = s4_moead.arm_config(GS4, 7, 1200)
    assert cfg == s4_moead.arm_config("N7_P1_SHARE", 7, 1200)  # same engine and budget
    rec = s4_moead.configuration_record(GS4, cfg, CAL)
    assert rec["guided"]["destroy_count"] == 4 and rec["guided"]["controller"] == "switch"
    assert rec["arm"] == GS4


def test_gs4_repeats_exactly_and_destroys_four_jobs(monkeypatch):
    kw = dict(guided_timer=True, probe=dict(probability=0.5, p_probe=0.05))
    first, search_a = _run(monkeypatch, GS4, **kw)
    second, search_b = _run(monkeypatch, GS4, **kw)
    sig = lambda r, s: (  # noqa: E731
        [(c.genotype, c.schedule, c.evaluation) for c in r.gateway.archive.members],
        dict(r.gateway.counts),
        s.steps,
        s.moead_counts,
    )
    assert sig(first, search_a) == sig(second, search_b)
    f3 = search_a.moead_summary()["f3"]
    assert f3["successes"] > 0
    assert sum(c["destroyed_jobs"] for c in f3["cells"].values()) == 4 * f3["successes"]
    ordinary = sum(b["slots"] for b in f3["deciles"]["ordinary"])
    assert ordinary + f3["successes"] == f3["slots"]  # every slot is exactly one path
    trace = f3["controller"]["trace"]
    assert trace and all(0.05 <= v[0] <= 0.5 for v in trace.values())


@pytest.fixture
def gs4_batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_gs4")


def test_gs4_batch_is_one_arm_on_the_first_batchs_instances_and_seeds(
    tmp_path, monkeypatch, gs4_batch
):
    b = gs4_batch
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(b.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(b.base, "CAL", digest(cal))
    assert b.ARMS == (GS4,) and b.SEEDS == (12, 13) and b.DESIGN == "s4-gs4/v1"
    assert b.ARMS_BY_SIZE == {100: (GS4,), 200: (GS4,)}
    assert b.SELECTION == {100: (4, (12, 13)), 200: (4, (12, 13))}
    runs = solver = 0
    seen = []
    for k in range(3):
        root = tmp_path / f"root{k}"
        b.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = b.validate_inputs(root)
        assert tuple(m["arms"]) == (GS4,)
        runs += m["runs"]
        solver += m["solver_seconds"]
        for block in m["blocks"]:
            spec = json.loads((root / "specs" / (block + ".json")).read_text())
            assert spec["order"] == [GS4] and spec["seed"] in (12, 13)
            assert spec["seconds"] == {100: 1200, 200: 2400}[spec["n"]]
            assert "run_s4_gs4.py" in spec["entry"]
            assert spec["config"][GS4]["guided"]["destroy_count"] == 4
        seen += m["blocks"]
    assert len(seen) == len(set(seen)) == 160
    assert Counter(int(x.split("_")[1][1:]) for x in seen) == {100: 80, 200: 80}
    assert runs == 160 and solver == 80 * 1200 + 80 * 2400 == 288_000
