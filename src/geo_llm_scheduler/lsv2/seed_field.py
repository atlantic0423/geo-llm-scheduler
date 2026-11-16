"""Learned LS launch-point models over the Archive front (research arms G1/E2/GE3).

Every LS trajectory is charged to the front point it was launched from: its
reward is the in-frame HV gain of its LS Archive insertions and its cost the LS
seconds it used, so one observation is ``rate = gain / seconds`` at the launch
point's Tchebycheff ray weight :math:`\\lambda_F`.

* :class:`RBFField` is Bayesian linear regression on a fixed two-dimensional
  Gaussian radial-basis grid over ``[0, 1]^2`` - a finite-basis approximation of
  a Gaussian-process posterior. Data terms decay with ``exp(-dt / tau)`` so the
  field follows the moving front, and Thompson samples of the field at candidate
  points choose the next launch point. G1 uses the inputs
  :math:`(\\lambda_F, t)`; GE3 uses :math:`(\\lambda_F, z)` with ``z`` the local
  recent LS investment around :math:`\\lambda_F`, so the field is a marginal
  return: what one more trajectory buys there given what was recently spent.
* :class:`DiminishingSegments` (E2) splits :math:`\\lambda_F` into equal
  segments and fits ``log(rate + floor) = a_s - b x_s`` by decayed weighted
  least squares, where ``x_s`` is the segment's recent decayed LS seconds and
  ``b >= 0`` is shared. The segment with the highest current marginal return
  ``exp(a_s - b x_s) - floor`` is launched next (equimarginal allocation).

Rates are transformed with ``y = log1p(rate / scale)``; ``scale`` is the median
of the first :data:`SCALE_WARMUP` positive rates and is then frozen, so targets
are comparable over a run. All randomness comes from an explicit ``Random``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from random import Random

import numpy as np

SCALE_WARMUP = 20


class RateScale:
    """Freeze a rate scale from the first positive observations; map rates to ``y``."""

    def __init__(self, warmup: int = SCALE_WARMUP) -> None:
        if warmup < 1:
            raise ValueError("warmup must be positive")
        self.warmup = warmup
        self._seen: list[float] = []
        self.scale: float | None = None

    def observe(self, rate: float) -> None:
        """Record a rate until the scale is frozen (zero rates do not set it)."""
        if self.scale is None and rate > 0:
            self._seen.append(rate)
            if len(self._seen) >= self.warmup:
                ordered = sorted(self._seen)
                self.scale = ordered[len(ordered) // 2]

    def current(self) -> float:
        """The frozen scale, or the median of the positive rates seen so far (1 if none)."""
        if self.scale is not None:
            return self.scale
        if not self._seen:
            return 1.0
        ordered = sorted(self._seen)
        return ordered[len(ordered) // 2]

    def to_y(self, rate: float) -> float:
        """``log1p(rate / scale)`` (rates are non-negative)."""
        return math.log1p(max(0.0, rate) / self.current())

    def to_rate(self, y: float) -> float:
        """Inverse of :meth:`to_y` (never negative)."""
        return max(0.0, math.expm1(y)) * self.current()


class RBFField:
    """Decayed Bayesian linear regression on a Gaussian RBF grid over ``[0, 1]^2``.

    The prediction is ``ybar + phi(u, v) @ w`` with ``ybar`` the decayed mean target,
    weights ``w ~ N(mu, P^-1)``, prior precision ``alpha`` and noise variance ``noise``.
    """

    def __init__(
        self,
        centers: tuple[int, int] = (12, 6),
        widths: tuple[float, float] = (0.12, 0.25),
        alpha: float = 1.0,
        noise: float = 0.25,
        tau: float = 240.0,
    ) -> None:
        if min(centers) < 2 or min(widths) <= 0 or alpha <= 0 or noise <= 0 or tau <= 0:
            raise ValueError("Invalid RBF field settings")
        gu, gv = np.linspace(0.0, 1.0, centers[0]), np.linspace(0.0, 1.0, centers[1])
        self._c = np.array([(a, b) for a in gu for b in gv])
        self._w = np.array(widths, dtype=float)
        self.alpha, self.noise, self.tau = alpha, noise, tau
        m = len(self._c)
        self._A = np.zeros((m, m))
        self._b = np.zeros(m)
        self._s = np.zeros(m)
        self._n = 0.0
        self._ysum = 0.0
        self._clock: float | None = None
        self.observations = 0
        self._version = 0  # bumped whenever the data terms change
        self._cache: tuple[int, float, np.ndarray, np.ndarray] | None = None

    @property
    def size(self) -> int:
        """Number of basis functions."""
        return len(self._c)

    def features(self, points: Sequence[tuple[float, float]]) -> np.ndarray:
        """``len(points) x size`` Gaussian RBF features (inputs clipped to ``[0, 1]``)."""
        x = np.clip(np.asarray(points, dtype=float).reshape(-1, 2), 0.0, 1.0)
        d = (x[:, None, :] - self._c[None, :, :]) / self._w
        return np.exp(-0.5 * np.sum(d * d, axis=2))

    def decay(self, now: float) -> None:
        """Discount all data terms by ``exp(-(now - last) / tau)``."""
        if self._clock is not None and now > self._clock:
            f = math.exp(-(now - self._clock) / self.tau)
            self._A *= f
            self._b *= f
            self._s *= f
            self._n *= f
            self._ysum *= f
            self._version += 1
        self._clock = now if self._clock is None else max(self._clock, now)

    def update(self, point: tuple[float, float], y: float, now: float) -> None:
        """Add one observation ``y`` at ``point`` after decaying to ``now``."""
        self.decay(now)
        phi = self.features([point])[0]
        self._A += np.outer(phi, phi)
        self._b += y * phi
        self._s += phi
        self._n += 1.0
        self._ysum += y
        self.observations += 1
        self._version += 1

    def _posterior(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Posterior weight mean and precision, recomputed only after the data change."""
        if self._cache is not None and self._cache[0] == self._version:
            return self._cache[1], self._cache[2], self._cache[3]
        ybar = self._ysum / self._n if self._n > 0 else 0.0
        precision = self.alpha * np.eye(self.size) + self._A / self.noise
        mean_w = np.linalg.solve(precision, (self._b - ybar * self._s) / self.noise)
        self._cache = (self._version, ybar, mean_w, precision)
        return ybar, mean_w, precision

    def mean(self, points: Sequence[tuple[float, float]]) -> np.ndarray:
        """Posterior mean of the field at ``points`` (same values as ``posterior()[0]``)."""
        ybar, mean_w, _ = self._posterior()
        return ybar + self.features(points) @ mean_w

    def posterior(self, points: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the field at ``points``."""
        ybar, mean_w, precision = self._posterior()
        phi = self.features(points)
        cov_w = np.linalg.inv(precision)
        mean = ybar + phi @ mean_w
        var = np.einsum("ij,jk,ik->i", phi, cov_w, phi)
        return mean, np.sqrt(np.maximum(var, 0.0))

    def sample(self, points: Sequence[tuple[float, float]], rng: Random) -> np.ndarray:
        """One joint Thompson sample of the field at ``points`` (explicit RNG)."""
        ybar, mean_w, precision = self._posterior()
        phi = self.features(points)
        cov_w = np.linalg.inv(precision)
        mean = ybar + phi @ mean_w
        cov = phi @ cov_w @ phi.T + 1e-9 * np.eye(len(phi))
        gen = np.random.default_rng(rng.getrandbits(63))
        return gen.multivariate_normal(mean, cov, method="cholesky")


class DiminishingSegments:
    """Equal :math:`\\lambda_F` segments with a shared diminishing-return slope (E2).

    ``log(rate + floor) = a_s - b x_s``: ``x_s`` is the segment's decayed recent LS
    seconds when the trajectory was launched. Ridge-regularised weighted least squares
    with decayed observation weights; ``b`` is clipped to be non-negative.
    """

    def __init__(self, segments: int = 5, tau: float = 120.0, ridge: float = 1e-3) -> None:
        if segments < 2 or tau <= 0 or ridge <= 0:
            raise ValueError("Invalid segment model settings")
        self.k, self.tau, self.ridge = segments, tau, ridge
        self.invested = [0.0] * segments
        self._G = np.zeros((segments + 1, segments + 1))
        self._h = np.zeros(segments + 1)
        self._clock: float | None = None
        self.counts = [0] * segments

    def segment(self, lam: float) -> int:
        """Equal-width segment of a Flow ray weight in ``[0, 1]``."""
        return min(self.k - 1, max(0, int(self.k * lam)))

    def decay(self, now: float) -> None:
        """Discount investments and observation weights to ``now``."""
        if self._clock is not None and now > self._clock:
            f = math.exp(-(now - self._clock) / self.tau)
            self.invested = [x * f for x in self.invested]
            self._G *= f
            self._h *= f
        self._clock = now if self._clock is None else max(self._clock, now)

    def charge(self, seg: int, seconds: float, now: float) -> None:
        """Add LS seconds spent from segment ``seg``."""
        self.decay(now)
        self.invested[seg] += max(0.0, seconds)

    def observe(self, seg: int, invested_at_launch: float, log_rate: float, now: float) -> None:
        """Record one trajectory: its segment, the segment's investment at launch, its log rate."""
        self.decay(now)
        row = np.zeros(self.k + 1)
        row[seg], row[-1] = 1.0, -invested_at_launch
        self._G += np.outer(row, row)
        self._h += log_rate * row
        self.counts[seg] += 1

    def fit(self) -> tuple[list[float], float]:
        """Intercepts ``a_s`` (untried segments get the mean of the tried) and slope ``b >= 0``."""
        reg = self.ridge * np.eye(self.k + 1)
        theta = np.linalg.solve(self._G + reg, self._h)
        a, b = theta[:-1], float(theta[-1])
        if b < 0:
            sub = self._G[:-1, :-1] + reg[:-1, :-1]
            a = np.linalg.solve(sub, self._h[:-1])
            b = 0.0
        tried = [s for s in range(self.k) if self.counts[s] > 0]
        mean_a = float(np.mean([a[s] for s in tried])) if tried else 0.0
        return [float(a[s]) if self.counts[s] > 0 else mean_a for s in range(self.k)], b

    def marginals(self, floor: float) -> list[float]:
        """Current marginal return of each segment, ``exp(a_s - b x_s) - floor`` (>= 0)."""
        a, b = self.fit()
        return [max(0.0, math.exp(a[s] - b * self.invested[s]) - floor) for s in range(self.k)]


def local_investment(launches: Sequence[tuple[float, float]], lam: float, width: float) -> float:
    """Kernel-weighted recent LS seconds ``sum c_j exp(-(lam - lam_j)^2 / (2 width^2))``."""
    return sum(c * math.exp(-0.5 * ((lam - x) / width) ** 2) for x, c in launches)


def investment_kernel(
    lams: Sequence[float] | np.ndarray, xs: Sequence[float] | np.ndarray, width: float
) -> np.ndarray:
    """``len(lams) x len(xs)`` Gaussian kernel ``exp(-(lam_i - x_j)^2 / (2 width^2))``."""
    if width <= 0:
        raise ValueError("Kernel width must be positive")
    d = (np.asarray(lams, dtype=float)[:, None] - np.asarray(xs, dtype=float)[None, :]) / width
    return np.exp(-0.5 * d * d)


def local_investment_many(
    lams: Sequence[float], launches: Sequence[tuple[float, float]], width: float
) -> np.ndarray:
    """Vectorized :func:`local_investment` at every ``lam`` (equal up to summation rounding)."""
    if not launches:
        return np.zeros(len(lams))
    xs, cs = zip(*launches, strict=True)
    return investment_kernel(lams, xs, width) @ np.asarray(cs, dtype=float)


__all__ = [
    "SCALE_WARMUP",
    "DiminishingSegments",
    "RBFField",
    "RateScale",
    "investment_kernel",
    "local_investment",
    "local_investment_many",
]
