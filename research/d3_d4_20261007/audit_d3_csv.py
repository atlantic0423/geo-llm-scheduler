"""Independent CSV statistical audit plus original frozen local completion validation."""

import csv
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.d2_campaign import frozen_manifest, validate_opportunity_job

BASE = Path(sys.argv[1])
out = BASE / "d3/analysis"

validation = []
for node in (0, 1):
    root = out.parent / "raw" / f"formal_node{node}"
    m = frozen_manifest(root)
    for key in m["keys"]:
        for role in ("samples", "probes"):
            ok, reason = validate_opportunity_job(root, role, key)
            assert ok, reason
    validation.append(
        dict(
            node=node,
            source=m["source_commit"],
            keys=len(m["keys"]),
            input_files=len(m["input_files"]),
        )
    )
rows = list(csv.DictReader((out / "heldout_decisions.csv").open(newline="")))
policies = ("GEOMETRIC", "RANDOM", "RGAIN", "BASE", "RESPONSE", "SHUFFLED", "POST_RGAIN")
group = defaultdict(list)
for row in rows:
    group[row["base"]].append(row)
assert len(group) == 16 and all(len(v) == 432 for v in group.values())
means = {
    b: {p: float(np.mean([float(r[p]) for r in records])) for p in policies}
    for b, records in sorted(group.items())
}
tests = {}
for control in ("BASE", "RGAIN", "SHUFFLED"):
    delta = np.array([v["RESPONSE"] - v[control] for v in means.values()])
    rng = np.random.default_rng(20261006034)
    boot = delta[rng.integers(16, size=(20000, 16))].mean(1)
    # Independent vectorized enumeration of all signs, matching original summation tolerance.
    signs = np.array(list(itertools.product((-1, 1), repeat=16)))
    p = float((np.abs(signs @ delta / 16) >= abs(float(delta.mean())) - 1e-14).mean())
    tests["RESPONSE-" + control] = dict(
        mean=float(delta.mean()), ci=np.quantile(boot, [0.025, 0.975]).tolist(), p=p
    )
order = sorted(tests, key=lambda k: tests[k]["p"])
prior = 0
for i, k in enumerate(order):
    prior = max(prior, (3 - i) * tests[k]["p"])
    tests[k]["p_holm"] = min(1, prior)
original = json.loads((out / "report.json").read_text())
for k, t in tests.items():
    for f in ("mean", "ci", "p", "p_holm"):
        assert np.allclose(t[f], original["primary"][k][f], rtol=1e-12, atol=1e-14), (k, f)
assert all(float(r["POST_RGAIN"]) >= float(r[p]) - 1e-14 for r in rows for p in policies)
subgroups = {}
for axis in ("jobs", "action", "stage", "preference", "tariff"):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        label = {
            "jobs": r["key"].split("_")[0],
            "action": "A" + r["action"],
            "stage": r["panel"].split("_")[0],
            "preference": r["panel"].split("_")[1],
            "tariff": r["key"].split("_")[2],
        }[axis]
        grouped[label][r["base"]].append(r)
    subgroups[axis] = {
        label: {
            p: float(np.mean([np.mean([float(r[p]) for r in rr]) for rr in bb.values()]))
            for p in policies
        }
        for label, bb in grouped.items()
    }
same = {
    p: float(np.mean([r["RESPONSE_index"] == r[p + "_index"] for r in rows]))
    for p in ("BASE", "RGAIN", "SHUFFLED")
}
oracle = {
    p: original["policy_means"]["POST_RGAIN"] - original["policy_means"][p]
    for p in ("BASE", "RGAIN", "SHUFFLED")
}
receipt = dict(
    status="original_local_validators_and_independent_csv_statistics_passed",
    validation=validation,
    decisions=len(rows),
    bases=16,
    tests=tests,
    index_agreement=same,
    empirical_oracle_headroom=oracle,
    subgroups=subgroups,
    interpretation="All exploratory strata unadjusted; frozen whole-test unchanged. No heldout refit or retuning.",
    csv_sha256=hashlib.sha256((out / "heldout_decisions.csv").read_bytes()).hexdigest(),
)
(out / "independent_verification.json").write_text(json.dumps(receipt, indent=2))
print(
    json.dumps(
        {
            k: receipt[k]
            for k in ("status", "decisions", "index_agreement", "empirical_oracle_headroom")
        }
    )
)
