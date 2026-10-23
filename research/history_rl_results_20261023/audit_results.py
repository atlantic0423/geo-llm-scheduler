"""Independent table, statistical-family and raw-evidence audits for history RL."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    """Load a complete UTF-8 audit table."""
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit(data: Path, output: Path) -> dict:
    """Reconstruct parent means and all 42 inferences independently from run CSVs."""
    stats = json.loads((output / "statistics.json").read_text())
    runs = read_csv(output / "run_metrics.csv")
    parents = read_csv(output / "parent_metrics.csv")
    assert len(runs) == 768 and len(parents) == 16 * 12
    lookup = {(int(r["base_seed"]), r["arm"]): r for r in parents}
    bases = sorted({b for b, _ in lookup})
    for (base, arm), row in lookup.items():
        items = [r for r in runs if int(r["base_seed"]) == base and r["arm"] == arm]
        assert len(items) == 4
        for metric in ("hv", "igd_plus", "generations", "step_action_share", "polish_share"):
            assert abs(float(row[metric]) - sum(float(x[metric]) for x in items) / 4) < 1e-11
    contrasts = read_csv(output / "parent_contrasts.csv")
    audited = []
    for row in stats["statistics"]:
        sign = 1 if row["metric"] == "hv" else -1
        values = np.array(
            [
                sign
                * (
                    float(lookup[(b, row["left"])][row["metric"]])
                    - float(lookup[(b, row["right"])][row["metric"]])
                )
                for b in bases
            ]
        )
        saved = {
            int(r["base_seed"]): float(r["effect"])
            for r in contrasts
            if r["family"] == row["family"]
            and r["contrast"] == row["contrast"]
            and r["metric"] == row["metric"]
        }
        assert np.allclose(values, [saved[b] for b in bases], atol=1e-13, rtol=0)
        assert abs(values.mean() - row["mean"]) < 1e-13
        rng = np.random.default_rng(row["seed"])
        boots = np.zeros(20000)
        for n in (20, 50, 100, 200):
            z = values[[int(lookup[(b, row["left"])]["n"]) == n for b in bases]]
            assert len(z) == 4
            sample = rng.integers(0, 4, (20000, 4))
            boots += z[sample].sum(axis=1) / 16
        signs = rng.integers(0, 2, (100000, 16)) * 2 - 1
        p = (
            1 + np.sum(np.abs(np.sum(signs * values, axis=1) / 16) >= abs(values.mean()) - 1e-14)
        ) / 100001
        assert p == row["p"]
        assert np.allclose(np.quantile(boots, [0.025, 0.975]), row["ci95"], atol=1e-13, rtol=0)
        exact = []
        for bit in range(65536):
            value = sum((1 if bit & (1 << i) else -1) * x for i, x in enumerate(values)) / 16
            exact.append(abs(value) >= abs(values.mean()) - 1e-14)
        assert np.mean(exact) == row["p_exact_audit"]
        audited.append((row["family"], row["p"], row["p_holm"]))
    for family, expected in (("primary", 20), ("plain", 22)):
        items = [x for x in audited if x[0] == family]
        assert len(items) == expected
        p = np.array([x[1] for x in items])
        order = np.argsort(p, kind="stable")
        corrected = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
        assert np.allclose(corrected, [items[i][2] for i in order], atol=1e-15, rtol=0)
    # Bind a saved final front to its original stored solution fields independently.
    fronts = json.loads((output / "normalized_fronts.json").read_text())
    references = json.loads((output / "references.json").read_text())
    checked = 0
    for name, front in list(fronts.items())[:: max(1, len(fronts) // 32)]:
        key, arm = name.split("|")
        ref = references[front["reference"]]
        with gzip.open(data / "raw/formal/runs" / key / arm / "archive.json.gz", "rt") as f:
            records = json.load(f)
        points = [
            [
                (r["evaluation"]["flow"] - ref["ideal"][0]) / ref["span"][0],
                (r["evaluation"]["tou"] + sum(r["evaluation"]["demand"]) - ref["ideal"][1])
                / ref["span"][1],
            ]
            for r in records
        ]
        assert all(
            any(abs(x - a) < 1e-11 and abs(y - b) < 1e-11 for a, b in points)
            for x, y in front["archive"]
        )
        checked += 1
    manifest = []
    for p in sorted(output.iterdir()):
        if p.is_file() and p.name not in ("INDEPENDENT_AUDIT.json",):
            with p.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            manifest.append(dict(name=p.name, size=p.stat().st_size, sha256=digest))
    result = dict(
        accepted=True,
        run_rows=768,
        parent_rows=192,
        statistics_independently_replayed=42,
        full_sign_patterns=65536,
        holm_families=2,
        raw_front_bindings_sample=checked,
        outputs=manifest,
        fresh_full_exact_rerun=False,
    )
    (output / "INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.data, args.output)
