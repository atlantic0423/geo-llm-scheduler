"""Learned LS launch-point models (research arms G1/E2/GE3)."""

import math
from random import Random

import numpy as np
import pytest

from geo_llm_scheduler.lsv2.seed_field import (
    DiminishingSegments,
    RateScale,
    RBFField,
    investment_kernel,
    local_investment,
    local_investment_many,
)


def test_rate_scale_freezes_at_the_median_of_the_first_positive_rates():
    scale = RateScale(warmup=5)
    assert scale.current() == 1.0
    for r in (0.0, 0.0, 4.0, 1.0, 3.0):
        scale.observe(r)
    assert scale.scale is None and scale.current() == 3.0  # median of 1, 3, 4 so far
    for r in (2.0, 5.0, 100.0):
        scale.observe(r)
    assert scale.scale == 3.0  # frozen at the 5th positive rate: 1,2,3,4,5
    scale.observe(1000.0)
    assert scale.current() == 3.0
    for r in (0.0, 0.5, 3.0, 30.0):
        assert scale.to_rate(scale.to_y(r)) == pytest.approx(r)
    assert scale.to_y(-1.0) == 0.0 and scale.to_rate(-5.0) == 0.0
    with pytest.raises(ValueError):
        RateScale(0)


def _peak(lam: float, t: float) -> float:
    # High return at the Flow end early, flat late: the shape seen in the trace data.
    return 2.0 * math.exp(-((lam - 0.9) ** 2) / 0.02) * (1.0 - t) + 0.2


def test_rbf_field_recovers_a_smooth_peak_and_shrinks_uncertainty_near_data():
    rng = Random(3)
    field = RBFField(tau=1e9)
    for k in range(600):
        lam, t = rng.random(), rng.random() * 0.5
        field.update((lam, t), _peak(lam, t) + rng.gauss(0, 0.1), now=float(k))
    grid = [(i / 20, 0.2) for i in range(21)]
    mean, sd = field.posterior(grid)
    assert abs(grid[int(np.argmax(mean))][0] - 0.9) <= 0.1
    assert max(abs(m - _peak(x, t)) for (x, t), m in zip(grid, mean)) < 0.35
    _, sd_far = field.posterior([(0.5, 0.95)])
    assert sd_far[0] > sd.mean()  # no data late in the budget
    a = field.sample(grid, Random(7))
    b = field.sample(grid, Random(7))
    assert np.array_equal(a, b) and a.shape == (21,)
    assert field.observations == 600 and field.size == 72


def test_rbf_field_forgets_old_data():
    field = RBFField(tau=10.0)
    for k in range(50):
        field.update((0.2, 0.5), 3.0, now=float(k))
    for k in range(50):
        field.update((0.2, 0.5), 0.0, now=200.0 + k)
    mean, _ = field.posterior([(0.2, 0.5)])
    assert mean[0] < 0.3  # the early level 3 has decayed away
    with pytest.raises(ValueError):
        RBFField(centers=(1, 6))
    with pytest.raises(ValueError):
        RBFField(noise=0.0)


def test_diminishing_segments_recover_slope_and_launch_the_best_marginal():
    rng = Random(5)
    model = DiminishingSegments(segments=3, tau=1e9)
    true_a, true_b = [math.log(1.0), math.log(4.0), math.log(2.0)], 0.3
    for k in range(900):
        seg = k % 3
        x = rng.random() * 6.0
        model.observe(seg, x, true_a[seg] - true_b * x + rng.gauss(0, 0.05), now=float(k))
    a, b = model.fit()
    assert b == pytest.approx(true_b, abs=0.02)
    assert a == pytest.approx(true_a, abs=0.05)
    model.invested = [0.0, 0.0, 0.0]
    marg = model.marginals(floor=0.0)
    assert int(np.argmax(marg)) == 1
    # Heavy recent investment in segment 1 lowers its marginal below segment 2.
    model.invested = [0.0, 4.0, 0.0]
    marg = model.marginals(floor=0.0)
    assert int(np.argmax(marg)) == 2
    assert marg[1] == pytest.approx(4.0 * math.exp(-true_b * 4.0), rel=0.1)


def test_diminishing_segments_clip_a_negative_slope_and_fill_untried_segments():
    model = DiminishingSegments(segments=4, tau=1e9)
    for x in (0.0, 1.0, 2.0, 3.0):  # rate grows with investment: no diminishing return
        model.observe(0, x, 0.5 * x, now=x)
        model.observe(1, x, 1.0 + 0.5 * x, now=x)
    a, b = model.fit()
    assert b == 0.0
    assert a[2] == a[3] == pytest.approx((a[0] + a[1]) / 2)
    model.charge(2, 3.0, now=10.0)
    assert model.invested[2] == 3.0
    model.tau = 1.0
    model.decay(now=20.0)
    assert model.invested[2] < 1e-3
    assert model.segment(0.0) == 0 and model.segment(0.999) == 3 and model.segment(1.0) == 3
    with pytest.raises(ValueError):
        DiminishingSegments(segments=1)


def test_local_investment_is_a_gaussian_kernel_sum():
    launches = [(0.9, 2.0), (0.1, 1.0)]
    assert local_investment(launches, 0.9, 0.08) == pytest.approx(
        2.0 + math.exp(-0.5 * (0.8 / 0.08) ** 2)
    )
    assert local_investment([], 0.5, 0.08) == 0.0


def test_local_investment_many_matches_the_scalar_reference():
    rng = Random(5)
    launches = [(rng.random(), rng.uniform(0.0, 3.0)) for _ in range(400)]
    lams = [i / 30 for i in range(31)] + [launches[0][0]]
    fast = local_investment_many(lams, launches, 0.08)
    ref = [local_investment(launches, lam, 0.08) for lam in lams]
    assert fast.shape == (len(lams),)
    assert fast.tolist() == pytest.approx(ref, rel=1e-12, abs=1e-15)
    assert local_investment_many(lams, [], 0.08).tolist() == [0.0] * len(lams)
    kernel = investment_kernel([0.5], [0.5, 0.58], 0.08)
    assert kernel.shape == (1, 2) and kernel[0].tolist() == pytest.approx([1.0, math.exp(-0.5)])
    with pytest.raises(ValueError):
        investment_kernel([0.5], [0.5], 0.0)


def _manual_mean(field: RBFField, points: list[tuple[float, float]]) -> np.ndarray:
    """Uncached posterior mean straight from the stored data terms."""
    ybar = field._ysum / field._n
    precision = field.alpha * np.eye(field.size) + field._A / field.noise
    mean_w = np.linalg.solve(precision, (field._b - ybar * field._s) / field.noise)
    return ybar + field.features(points) @ mean_w


def test_rbf_mean_is_cached_until_the_data_change():
    field = RBFField(tau=100.0)
    pts = [(0.1, 0.2), (0.7, 0.9), (0.5, 0.5)]
    field.update((0.2, 0.3), 1.0, 0.0)
    m1 = field.mean(pts)
    assert np.array_equal(m1, field.posterior(pts)[0]) and np.array_equal(
        m1, _manual_mean(field, pts)
    )
    assert np.array_equal(field.mean(pts), m1)  # served from the cache
    field.update((0.8, 0.8), 3.0, 10.0)  # new data invalidate the cache
    m2 = field.mean(pts)
    assert not np.allclose(m1, m2) and np.array_equal(m2, _manual_mean(field, pts))
    field.decay(400.0)  # decay shrinks data against the prior: the mean moves
    m3 = field.mean(pts)
    assert not np.allclose(m2, m3) and np.array_equal(m3, _manual_mean(field, pts))
    field.decay(400.0)  # no time elapsed: cache stays valid
    assert np.array_equal(field.mean(pts), m3)
