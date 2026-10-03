"""Small, explicit paired-cluster tools for exploratory experiment analysis."""

from __future__ import annotations

import math

import numpy as np


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the original hypothesis order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("P-values must be finite and between zero and one")
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(order)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_cluster_summary(
    values: list[float],
    strata: list[int],
    rng: np.random.Generator,
    bootstrap_samples: int = 20000,
    permutation_samples: int = 100000,
) -> dict[str, float | int | list[float] | None]:
    """Describe independent cluster effects with stratified bootstrap and sign flips.

    Each supplied value is already one equally weighted base-instance effect.
    Bootstrap keeps the observed number of bases in each job-size stratum.
    The two-sided Monte Carlo sign-flip test assumes independent, symmetric null
    cluster differences; it is exploratory, not proof of random-label assignment.
    No zero-difference cluster is discarded. All randomness uses the supplied RNG.
    """
    if (
        not values
        or len(values) != len(strata)
        or not all(math.isfinite(x) for x in values)
        or bootstrap_samples < 1
        or permutation_samples < 1
    ):
        raise ValueError("Require finite matched clusters and positive resampling counts")
    data = np.asarray(values, dtype=float)
    groups = [data[np.asarray(strata) == s] for s in sorted(set(strata))]
    sums = np.zeros(bootstrap_samples)
    for group in groups:
        indices = rng.integers(0, len(group), size=(bootstrap_samples, len(group)))
        sums += group[indices].sum(axis=1)
    interval = np.quantile(sums / len(data), (0.025, 0.975))
    observed = abs(float(data.mean()))
    extreme = 0
    for offset in range(0, permutation_samples, 4096):
        count = min(4096, permutation_samples - offset)
        signs = 2 * rng.integers(0, 2, size=(count, len(data))) - 1
        simulated = np.abs((signs * data).mean(axis=1))
        extreme += int(np.count_nonzero(simulated >= observed - 1e-14))
    deviation = float(data.std(ddof=1)) if len(data) > 1 else 0.0
    return {
        "n_bases": len(data),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "ci95": [float(x) for x in interval],
        "p_signflip": (extreme + 1) / (permutation_samples + 1),
        "positive_bases": int(np.count_nonzero(data > 1e-12)),
        "negative_bases": int(np.count_nonzero(data < -1e-12)),
        "tie_bases": int(np.count_nonzero(np.abs(data) <= 1e-12)),
        "standardized_paired_effect": float(data.mean()) / deviation if deviation else None,
    }
