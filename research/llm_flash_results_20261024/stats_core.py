"""Metrics and paired inference reused from verified cb8babc7 with the new frozen seed."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.metrics import hypervolume

SIZES = (20, 50, 100, 200)
ROOT_SEED = 2026102401


def nondominated(points: list) -> list[tuple[float, float]]:
    """Return exact unique minimization points, without an arbitrary epsilon filter."""
    result: list[tuple[float, float]] = []
    best = math.inf
    for x, y in sorted(set(map(tuple, points))):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite objective")
        if y < best:
            result.append((x, y))
            best = y
    return result


def volume(points: list) -> float:
    """Compute union area in the frozen 1.1 reference rectangle."""
    return hypervolume([tuple(p) for p in points if max(p) <= 1.1], (1.1, 1.1))


def volume_audit(points: list) -> float:
    """Independently integrate horizontal strips rather than vertical rectangles."""
    front = nondominated([p for p in points if max(p) <= 1.1])
    return math.fsum(
        (1.1 - x) * ((front[i - 1][1] if i else 1.1) - y) for i, (x, y) in enumerate(front)
    )


def distance(points: list, reference: list, plus: bool = True) -> float:
    """Compute IGD+/IGD using bounded NumPy reference chunks."""
    a, r = np.asarray(points), np.asarray(reference)
    if not len(a) or not len(r):
        raise ValueError("Empty front")
    total = 0.0
    for i in range(0, len(r), 64):
        delta = a[None, :, :] - r[i : i + 64, None, :]
        if plus:
            delta = np.maximum(delta, 0)
        total += np.sqrt(np.square(delta).sum(axis=2)).min(axis=1).sum()
    return float(total / len(r))


def coverage(a: list, b: list) -> float:
    """Fraction of unique B points weakly dominated by A, using a 2-D sweep."""
    x, y = np.asarray(nondominated(a)), np.asarray(nondominated(b))
    indexes = np.searchsorted(x[:, 0], y[:, 0] + 1e-12, side="right") - 1
    return float(np.mean((indexes >= 0) & (x[np.maximum(indexes, 0), 1] <= y[:, 1] + 1e-12)))


def holm(values: list[float]) -> list[float]:
    """Adjust exactly one predeclared complete family and preserve its order."""
    answer = [0.0] * len(values)
    previous = 0.0
    for rank, i in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        previous = max(previous, (len(values) - rank) * values[i])
        answer[i] = min(1.0, previous)
    return answer


def inference(values: list[float], sizes: list[int], seed: int) -> dict:
    """Use 16 parents, stratified bootstrap and the frozen Monte Carlo sign-flip."""
    assert len(values) == 16 and all(sizes.count(n) == 4 for n in SIZES)
    assert all(math.isfinite(x) for x in values)
    a = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = np.zeros(20000)
    for n in SIZES:
        z = a[np.asarray(sizes) == n]
        boots += z[rng.integers(len(z), size=(20000, len(z)))].mean(1) / 4
    effect = float(a.mean())
    hits = 0
    for i in range(0, 100000, 4096):
        signs = rng.integers(0, 2, size=(min(4096, 100000 - i), 16)) * 2 - 1
        hits += int((np.abs((signs * a).mean(1)) >= abs(effect) - 1e-14).sum())
    exact = 0
    for i in range(0, 65536, 4096):
        signs = (
            2
            * ((np.arange(i, i + 4096, dtype=np.uint32)[:, None] >> np.arange(16)) & 1).astype(
                float
            )
            - 1
        )
        exact += int((np.abs((signs * a).mean(1)) >= abs(effect) - 1e-14).sum())
    sd = float(a.std(ddof=1))
    return dict(
        mean=effect,
        ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
        p=(hits + 1) / 100001,
        p_exact_audit=exact / 65536,
        positive=int((a > 1e-12).sum()),
        negative=int((a < -1e-12).sum()),
        standardized_paired_effect=effect / sd if sd else None,
        n_parents=16,
        seed=seed,
        per_scale={str(n): float(a[np.asarray(sizes) == n].mean()) for n in SIZES},
    )


def stored_points(path: Path) -> tuple[list, int]:
    """Read saved feasible solutions, without claiming a fresh exact evaluation."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        records = json.load(f)
    assert records and all(r["evaluation"]["feasible"] for r in records)
    points = [
        (r["evaluation"]["flow"], r["evaluation"]["tou"] + sum(r["evaluation"]["demand"]))
        for r in records
    ]
    return nondominated(points), len(records)


def write_table(out: Path, name: str, rows: list[dict]) -> None:
    """Save complete scalar observations as UTF-8 CSV."""
    if not rows:
        raise ValueError("Missing table rows")
    with (out / (name + ".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
