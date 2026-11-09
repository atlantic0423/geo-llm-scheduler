"""Throughput rewrites (2026-11-09) against the frozen 582237f code: bit-identical results.

Covers the overlap-filtered ``_fits_intervals``, the incremental ``ResourceCalendar``
index, SSGS without per-operation schedule copies, the one-sweep exact resource
check, the per-machine power sweep and the LS ``PowerProfile`` memoization; and a
whole working-version run under a deterministic clock.
"""

import hashlib
import sys
import time
from dataclasses import replace
from random import Random

import pytest

from geo_llm_scheduler.domain.models import (
    Genotype,
    Job,
    PowerModel,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.evaluation import exact
from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.mechanism_instances import mechanism_problem, panel_specs
from geo_llm_scheduler.initialization.generators import generate
from geo_llm_scheduler.lsv2 import billing
from geo_llm_scheduler.scheduling import resources, ssgs
from research.throughput_20261109 import reference_runtime as reference
from research.throughput_20261109.frozen import billing as old_billing
from research.throughput_20261109.frozen import exact as old_exact
from research.throughput_20261109.frozen import resources as old_resources
from research.throughput_20261109.frozen import ssgs as old_ssgs

POWER = PowerModel(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
CAL = dict(rho=0.001, h=0.01, w_cutoff=1.5)


def test_frozen_sources_are_the_582237f_git_blobs():
    blobs = {
        "resources.py": "56fbc3d8efc0674ee013528eca010e8e6b010da9",
        "ssgs.py": "a8bf55b7d716ef6f5c6a26bad696f63abacd72bc",
        "exact.py": "fb8bd9eafd6f04b4f7136242c79c09a840b30dc7",
        "billing.py": "71d1a878ef44c07b8803f163da7a179dd36e0235",
    }
    for module in (old_resources, old_ssgs, old_exact, old_billing):
        data = open(module.__file__, "rb").read()
        name = module.__file__.replace("\\", "/").rsplit("/", 1)[-1]
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == blobs[name]
    # The frozen SSGS/exact resolve resource helpers to the frozen copy only.
    assert old_ssgs.ResourceCalendar is old_resources.ResourceCalendar
    assert old_exact.ResourceView is old_resources.ResourceView


def test_reference_runtime_rebinds_and_restores_every_live_binding():
    live = (ssgs.decode, exact.evaluate, resources._fits_intervals, billing.PowerProfile)
    with reference.reference_runtime():
        assert ssgs.decode is old_ssgs.decode and exact.evaluate is old_exact.evaluate
        assert resources._fits_intervals is old_resources._fits_intervals
        assert billing.PowerProfile is old_billing.PowerProfile
    assert (ssgs.decode, exact.evaluate, resources._fits_intervals, billing.PowerProfile) == live


def tight_problem(rng: Random, jobs: int = 14, power: PowerModel = POWER) -> ProblemInstance:
    """Coarse durations and near-capacity fractions: coincident ends and tolerance edges."""
    eps = (-2e-10, 0.0, 2e-10, 4e-9)
    computes = (0.25, 0.5, 1 / 3, 0.2, 0.75)

    def profile() -> Profile:
        return Profile(
            rng.choice((50.0, 100.0, 150.0, 100.1)),
            min(1.0, rng.choice(computes) + rng.choice(eps)),
            rng.choice((2.0, 4.0, 4.0000000004, 3.9999999998)),
        )

    return ProblemInstance(
        tuple(
            Job(
                f"j{i}",
                rng.choice((0.0, 0.0, 25.0, 50.1)),
                profile(),
                profile(),
                rng.choice((0.0, 30.0)),
            )
            for i in range(jobs)
        ),
        (
            Region("r0", (Tariff(0, 900, 0.3), Tariff(900, 50000, 0.1)), 2.0),
            Region("r1", (Tariff(0, 50000, 0.2),), 3.0),
        ),
        (
            ServingInstance("m0", 0, 8.0, 10, 100),
            ServingInstance("m1", 0, 8.0, 12, 90),
            ServingInstance("m2", 1, 8.0, 10, 100),
        ),
        power,
    )


def random_genotype(problem: ProblemInstance, rng: Random) -> Genotype:
    ms = [0] * problem.operation_count
    for j in range(len(problem.jobs)):
        region = rng.randrange(len(problem.regions))
        for o in (2 * j, 2 * j + 1):
            ms[o] = rng.choice(problem.eligible(o, region))
    os_ = [j for j in range(len(problem.jobs)) for _ in range(2)]
    rng.shuffle(os_)
    return Genotype(tuple(ms), tuple(os_))


@pytest.mark.parametrize("seed", range(6))
def test_fits_intervals_matches_reference_on_shuffled_lists(seed):
    rng = Random(seed)
    problem = tight_problem(rng)
    grid = (0.0, 25.0, 50.0, 50.1, 100.0, 100.1, 150.0, 200.2)
    for _ in range(400):
        o = rng.randrange(problem.operation_count)
        others = []
        for x in rng.sample(range(problem.operation_count), rng.randrange(0, 9)):
            if x != o:
                s = rng.choice(grid)
                others.append((x, s, s + problem.profile(x).duration))
        rng.shuffle(others)  # the reference sums in list order, sorted or not
        machine, start = rng.randrange(3), rng.choice(grid)
        assert resources._fits_intervals(problem, machine, o, start, others) == (
            old_resources._fits_intervals(problem, machine, o, start, others)
        )


@pytest.mark.parametrize("seed", range(6))
def test_calendar_index_matches_reference_through_random_commits(seed):
    rng = Random(100 + seed)
    problem = tight_problem(rng, jobs=16)
    new, old = resources.ResourceCalendar(problem), old_resources.ResourceCalendar(problem)
    order = list(range(problem.operation_count))
    rng.shuffle(order)
    grid = (0.0, 25.0, 50.0, 50.1, 100.0, 150.0, 300.0)
    for step, o in enumerate(order):
        machine = rng.randrange(3)
        est = rng.choice(grid)
        for probe in rng.sample(order, 4):  # committed or not, any machine
            m = rng.randrange(3)
            t = rng.choice(grid)
            assert new.fits(m, probe, t) == old_resources._fits_intervals(
                problem, m, probe, t, [i for i in old.items[m] if i[0] != probe]
            )
            try:
                expected = old.earliest(m, probe, t)
            except ValueError:
                with pytest.raises(ValueError):
                    new.earliest(m, probe, t)
            else:
                assert new.earliest(m, probe, t) == expected
        temp = (order[-1], 50.0, 50.0 + problem.profile(order[-1]).duration)
        if step % 3 == 0 and temp[0] != o:
            assert new.earliest(machine, o, est, temporary=temp) == old.earliest(
                machine, o, est, temporary=temp
            )
        try:
            start = old.earliest(machine, o, est)
        except ValueError:
            continue
        assert new.earliest(machine, o, est) == start
        new.commit(machine, o, start)
        old.commit(machine, o, start)
        assert new.items == old.items


@pytest.mark.parametrize("seed", range(8))
def test_decode_matches_reference_on_tight_inputs(seed):
    rng = Random(200 + seed)
    problem = tight_problem(rng, jobs=18)
    for _ in range(15):
        g = random_genotype(problem, rng)
        try:
            expected = old_ssgs.decode(problem, g)
        except ValueError:
            with pytest.raises(ValueError):
                ssgs.decode(problem, g)
            continue
        assert ssgs.decode(problem, g) == expected


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize("seed", range(6))
def test_exact_matches_reference_on_random_and_violating_timings(seed, legacy):
    rng = Random(300 + seed)
    problem = tight_problem(rng, power=PowerModel() if legacy else POWER)
    grid = (0.0, 0.0, 25.0, 50.0, 50.1, 100.0, 130.0, 260.0)
    for _ in range(40):
        g = random_genotype(problem, rng)
        timing = [rng.choice(grid) for _ in range(problem.operation_count)]
        if rng.random() < 0.3:
            timing[rng.randrange(len(timing))] = -1.0  # unscheduled: per-operation path
        schedule = Schedule(tuple(timing))
        assert exact.evaluate(problem, g, schedule) == old_exact.evaluate(problem, g, schedule)
        assert exact.violations(problem, g, schedule) == old_exact.violations(problem, g, schedule)
        try:
            decoded = old_ssgs.decode(problem, g)
        except ValueError:
            continue
        assert exact.evaluate(problem, g, decoded) == old_exact.evaluate(problem, g, decoded)
        for region in range(len(problem.regions)):
            assert exact.power_segments(problem, g, decoded, region) == (
                old_exact.power_segments(problem, g, decoded, region)
            )


@pytest.mark.parametrize("seed", range(6))
def test_power_profile_totals_deltas_and_copies_match_reference(seed):
    rng = Random(400 + seed)
    problem = tight_problem(rng)
    g = random_genotype(problem, rng)
    starts = [rng.choice((0.0, 100.0, 450.0, 899.5, 1800.0, 2700.25)) for _ in g.ms]
    new = billing.PowerProfile(problem, g.ms, starts)
    old = old_billing.PowerProfile(problem, g.ms, starts)

    def same(a, b):
        assert (a.tou, a.windows, a.unions, a.bill(), a.horizon) == (
            b.tou,
            b.windows,
            b.unions,
            b.bill(),
            b.horizon,
        )

    same(new, old)
    for _ in range(25):
        changes = {}
        for o in rng.sample(range(len(starts)), 3):
            s = rng.choice((0.0, 50.0, 900.0, 1350.5, 2000.0))
            changes.setdefault(g.ms[o], {})[o] = (
                None if rng.random() < 0.2 else (s, s + problem.profile(o).duration)
            )
        assert new.delta_for(changes) == old.delta_for(changes)
        region = rng.randrange(len(problem.regions))
        assert new.window_values(changes, region) == old.window_values(changes, region)
        if rng.random() < 0.5:
            new.apply(changes)
            old.apply(changes)
            same(new, old)
        horizon = new.horizon + rng.choice((0.0, 450.0, 2000.0))
        same(new.with_horizon(horizon), old.with_horizon(horizon))
        same(new.copy(), old.copy())


@pytest.mark.parametrize("spec", panel_specs()[:2], ids=lambda s: s.key)
def test_panel_inputs_decode_and_evaluate_identically(spec):
    problem = replace(mechanism_problem(spec), power_model=POWER)
    rng = Random(spec.seed)
    for mode in (0, 7, 9):
        g = generate(problem, mode, rng, 0.1)
        new = ssgs.decode(problem, g)
        assert new == old_ssgs.decode(problem, g)
        assert exact.evaluate(problem, g, new) == old_exact.evaluate(problem, g, new)


def _without_measurements(value):
    """Drop measured durations (real CPU time is outside any fake clock); keep decisions."""
    if isinstance(value, dict):
        return {
            k: _without_measurements(v)
            for k, v in value.items()
            if not any(word in k for word in ("seconds", "elapsed", "cpu"))
        }
    if isinstance(value, list):
        return [_without_measurements(v) for v in value]
    return value


class FakeClock:
    """Deterministic clock: every reading advances by a fixed step."""

    def __init__(self, step: float) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _run_with_clock(monkeypatch, problem, arm, use_reference):
    clock = FakeClock(0.002)
    monkeypatch.setattr(time, "perf_counter", clock)
    for name, module in tuple(sys.modules.items()):
        if name.startswith("geo_llm_scheduler.") and module is not None:
            if callable(getattr(module, "perf_counter", None)):
                monkeypatch.setattr(module, "perf_counter", clock)
    config = replace(s4_moead.arm_config(arm, 5, 40), generations=3)
    searches = []

    def sink(search):
        search.timer = clock
        searches.append(search)

    evidence: dict = {}
    if use_reference:
        with reference.reference_runtime():
            result = s4_moead.run_arm(problem, arm, config, CAL, evidence, search_sink=sink)
    else:
        result = s4_moead.run_arm(problem, arm, config, CAL, evidence, search_sink=sink)
    search = searches[0]
    return (
        [(c.genotype, c.schedule, c.evaluation) for c in result.gateway.archive.members],
        dict(result.gateway.counts),
        search.steps,
        search.moead_counts,
        _without_measurements(evidence),
    )


def test_working_version_whole_run_is_identical_under_a_deterministic_clock(monkeypatch):
    spec = panel_specs()[0]
    problem = replace(mechanism_problem(spec), power_model=POWER)
    new = _run_with_clock(monkeypatch, problem, "N7_P1_SHARE", False)
    old = _run_with_clock(monkeypatch, problem, "N7_P1_SHARE", True)
    assert new[1]["exact"] > 100 and new[2], "the run must exercise evolution and LS"
    assert new == old


# ---------------- HV batch entry (2026-11-09) ----------------
@pytest.fixture
def throughput_batch(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib

    return importlib.import_module("run_s4_throughput")


def test_batch_arms_share_the_algorithm_and_only_the_control_uses_the_reference(throughput_batch):
    b = throughput_batch
    assert b.ARMS == ("C0_REF", "C0F") and b.CONTROL == "C0_REF" and b.SEEDS == (11,)
    skipped = b.import_all()
    assert all(sys.modules.get(n) is None for n in skipped)
    assert "geo_llm_scheduler.experiments.s4_moead" in sys.modules
    with b.engine_for("C0_REF"):
        assert ssgs.decode is old_ssgs.decode and exact.evaluate is old_exact.evaluate
        assert billing.PowerProfile is old_billing.PowerProfile
    with b.engine_for("C0F"):
        assert ssgs.decode is not old_ssgs.decode and exact.evaluate is not old_exact.evaluate
    with pytest.raises(ValueError):
        b.engine_for("N7_P1_SHARE")
    assert len(b.reference_hash()) == 64


def test_batch_prepares_two_engines_on_the_panel_at_seed_eleven(
    tmp_path, monkeypatch, throughput_batch
):
    import json
    from collections import Counter

    from test_s4_load_shift import _panel

    from geo_llm_scheduler.experiments.trans_runner import digest

    b = throughput_batch
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(b.base, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(b.base, "CAL", digest(cal))
    seen = []
    for k in range(3):
        root = tmp_path / f"root{k}"
        b.prepare(root, panel, cal, "example", shard=f"{k}/3")
        m = b.validate_inputs(root)
        assert m["design"] == "s4-throughput/v1" and tuple(m["arms"]) == b.ARMS
        assert m["runs"] == 2 * len(m["blocks"]) == 80 and m["reference"] == b.reference_hash()
        for block in m["blocks"]:
            spec = json.loads((root / "specs" / (block + ".json")).read_text())
            assert spec["seed"] == 11 and "run_s4_throughput.py" in spec["entry"]
            assert spec["seconds"] == {50: 600, 100: 1200, 200: 2400}[spec["n"]]
            assert spec["config"]["C0_REF"] == spec["config"]["C0F"]
            assert spec["algorithm"] == "N7_P1_SHARE" and sorted(spec["order"]) == sorted(b.ARMS)
        seen += m["blocks"]
    assert len(seen) == len(set(seen)) == 120
    assert Counter(int(x.split("_")[1][1:]) for x in seen) == {50: 40, 100: 40, 200: 40}
    # Any drift in the frozen reference is rejected before dispatch.
    monkeypatch.setattr(b, "reference_hash", lambda: "0" * 64)
    with pytest.raises(AssertionError):
        b.validate_inputs(tmp_path / "root0")
