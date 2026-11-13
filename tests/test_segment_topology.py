"""Front-segment isolation (S3) and its interleaved-island matched control (I3)."""

import pytest
from test_guided_cost import _run

from geo_llm_scheduler.engine import run as engine_run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.segment_topology import (
    SegmentOptions,
    SegmentTopology,
    groups,
    segmented,
)
from geo_llm_scheduler.lsv2 import search as lsv2_search
from geo_llm_scheduler.moead.core import neighborhoods, weights


def test_groups_and_neighborhoods_stay_inside_their_segment():
    lam = weights(100)
    g = groups(100, SegmentOptions())
    assert [len(x) for x in g] == [33, 34, 33] and g[0] == tuple(range(33))
    gi = groups(100, SegmentOptions(mode="interleaved"))
    assert gi[1] == tuple(range(1, 100, 3)) and sorted(sum(gi, ())) == list(range(100))
    for opts, parts in ((SegmentOptions(), g), (SegmentOptions(mode="interleaved"), gi)):
        nb, pools = segmented(lam, 20, opts)
        for part in parts:
            for i in part:
                assert set(nb[i]) <= set(part) and len(nb[i]) == 20 and nb[i][0] == i
                assert pools[i] == part
    one = SegmentOptions(segments=1)
    assert segmented(lam, 20, one)[0] == neighborhoods(lam, 20)  # one group = default topology
    assert segmented(lam, 20, one)[1][5] == tuple(range(100))
    with pytest.raises(ValueError):
        SegmentOptions(mode="other")
    with pytest.raises(ValueError):
        groups(5, SegmentOptions())


def test_single_segment_topology_reproduces_the_control_exactly(monkeypatch):
    sig = lambda r: [(c.genotype, c.schedule, c.evaluation) for c in r.gateway.archive.members]  # noqa: E731
    control, _ = _run(monkeypatch, "GS4_GUIDED_RAND_SWITCH_D4")
    monkeypatch.setitem(s4_moead._SEG, "S3_SEG_GS4", SegmentOptions(segments=1))
    seam, _ = _run(monkeypatch, "S3_SEG_GS4")
    assert sig(seam) == sig(control) and dict(seam.gateway.counts) == dict(control.gateway.counts)


@pytest.mark.parametrize("arm", ["S3_SEG_GS4", "I3_ISL_GS4"])
def test_segmented_runs_repeat_and_never_replace_across_segments(monkeypatch, arm):
    parts = groups(100, s4_moead._SEG[arm])
    seen = []
    for module in (engine_run, lsv2_search):
        original = module.replace_neighbors

        def spy(population, candidate, neighbors, *args, _original=original, **kw):
            seen.append(tuple(neighbors))
            return _original(population, candidate, neighbors, *args, **kw)

        monkeypatch.setattr(module, "replace_neighbors", spy)
    made = []

    class Capture(SegmentTopology):
        def __init__(self, options):
            super().__init__(options)
            made.append(self)

    monkeypatch.setattr(s4_moead, "SegmentTopology", Capture)
    first, search = _run(monkeypatch, arm)
    second, _ = _run(monkeypatch, arm)
    sig = lambda r: [(c.genotype, c.evaluation) for c in r.gateway.archive.members]  # noqa: E731
    assert sig(first) == sig(second)
    assert seen and all(any(set(nb) <= set(p) for p in parts) for nb in seen)
    assert search.neighbors == made[0].neighbors and made[0].options.mode in (
        "contiguous",
        "interleaved",
    )
    problem = first.gateway.problem
    for c in first.population:
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
    record = s4_moead.configuration_record(arm, s4_moead.arm_config(arm, 7, 1200), {})
    assert record["segment_topology"] == dict(mode=made[0].options.mode, segments=3)


def test_segment_batch_seals_three_arms_three_new_seeds(tmp_path, monkeypatch):
    import importlib
    import json
    from pathlib import Path

    from test_s4_load_shift import _panel

    from geo_llm_scheduler.experiments.trans_runner import digest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    b = importlib.import_module("run_segment")
    panel, cal, _ = _panel(tmp_path, dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25))
    monkeypatch.setattr(b.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(b.base, "CAL", digest(cal))
    runs, seen = 0, []
    for k in range(3):
        root = tmp_path / f"root{k}"
        b.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = b.validate_inputs(root)
        runs += m["runs"]
        seen += m["blocks"]
        spec = json.loads((root / "specs" / (m["blocks"][0] + ".json")).read_text())
        assert set(spec["order"]) == set(b.ARMS) and spec["seed"] in (21, 22, 23)
        assert "run_segment.py" in spec["entry"]
    assert len(seen) == len(set(seen)) == 360 and runs == 1080
