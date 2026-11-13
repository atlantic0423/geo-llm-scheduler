"""Front-gap reproduction allocation (A1) and its shuffled matched control (A0)."""

import pytest
from test_guided_cost import _run

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.gap_allocation import (
    GapAllocator,
    GapOptions,
    capped,
    ray_weights,
)
from geo_llm_scheduler.moead.core import weights


def test_ray_weights_follow_gap_area_and_skip_duplicates():
    front = [(0.0, 1.0), (0.1, 0.3), (1.0, 0.0)]  # gaps 0.07 (Flow side) and 0.27
    lam = weights(11)
    w = ray_weights(front, lam, [False] * 11, endpoint_bonus=0.0)
    assert sum(w) == pytest.approx(0.34)
    # rays with lambda_F/lambda_B < 0.3/0.1 cross the large Bill-side gap
    big = [i for i, (f, b) in enumerate(lam) if 0 < f and b > 0 and f / b < 3]
    assert sum(w[i] for i in big) == pytest.approx(0.27)
    dup = [i == big[0] for i in range(11)]
    w2 = ray_weights(front, lam, dup, endpoint_bonus=0.0)
    assert w2[big[0]] == 0 and sum(w2) == pytest.approx(0.34)
    assert ray_weights(front[:1], lam, [False] * 11) == [1.0] * 11
    with pytest.raises(ValueError):
        GapOptions(mode="other")
    c = capped([0.6, 0.2, 0.1, 0.1] + [0.0] * 16, 0.25)
    assert max(c) <= 0.25 + 1e-12 and sum(c) == pytest.approx(1.0)


@pytest.mark.parametrize("arm", ["A1_GAP_GS4", "A0_SHUF_GS4"])
def test_allocated_runs_repeat_exactly_and_keep_exact_members(monkeypatch, arm):
    made = []

    class Capture(GapAllocator):
        def __init__(self, options):
            super().__init__(options)
            made.append(self)

    monkeypatch.setattr(s4_moead, "GapAllocator", Capture)
    first, _ = _run(monkeypatch, arm)
    second, _ = _run(monkeypatch, arm)
    sig = lambda r: [(c.genotype, c.evaluation) for c in r.gateway.archive.members]  # noqa: E731
    assert sig(first) == sig(second)
    a = made[0]
    assert a.options.mode == ("gap" if arm.startswith("A1") else "shuffled")
    assert a.counts["generations"] >= 1 and a.counts["slots"] > 0
    assert 1 / 100 < max(a.max_probability) <= 0.05 + 1e-12  # non-uniform, capped
    problem = first.gateway.problem
    for c in first.population:
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
    record = s4_moead.configuration_record(arm, s4_moead.arm_config(arm, 7, 1200), {})
    assert record["gap_allocation"]["mode"] == a.options.mode


def test_gap_batch_seals_three_arms_two_new_seeds(tmp_path, monkeypatch):
    import importlib
    import json
    from pathlib import Path

    from test_s4_load_shift import _panel

    from geo_llm_scheduler.experiments.trans_runner import digest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    b = importlib.import_module("run_gap_alloc")
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
        assert set(spec["order"]) == set(b.ARMS) and spec["seed"] in (19, 20)
        assert "run_gap_alloc.py" in spec["entry"]
    assert len(seen) == len(set(seen)) == 240 and runs == 720
