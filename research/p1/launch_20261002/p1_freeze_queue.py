"""Freeze the deployed P1 matrix in paired blocks and alternate 50/100-job blocks."""

import hashlib
import json
from pathlib import Path

from geo_llm_scheduler.utils.rng import RNGManager

base = Path("/workspace/zhouhanyu")
for node in (0, 1):
    root = base / f"campaign_p1_20261002_node{node}"
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    original = manifest["specs"]
    other = [s for s in original if s["stage"] != "P1"]
    groups = {}
    for item in original:
        if item["stage"] == "P1":
            key = (item["jobs"], item["base_seed"], item["algorithm_seed"], item["tariff"])
            groups.setdefault(key, []).append(item)
    assert len(groups) == 120 and all(
        {s["arm"] for s in items} == {"F6", "M0"} for items in groups.values()
    )
    rng = RNGManager(20261001).stream(f"P1:paired-queue:{node}")
    queues = {n: [k for k in groups if k[0] == n] for n in (50, 100)}
    for queue in queues.values():
        rng.shuffle(queue)
    assert len(queues[50]) == len(queues[100]) == 60
    ordered = []
    for small, large in zip(queues[50], queues[100]):
        for key in (small, large):
            block = list(groups[key])
            rng.shuffle(block)
            ordered.extend(block)
    assert sorted(s["key"] for s in ordered) == sorted(
        s["key"] for s in original if s["stage"] == "P1"
    )
    manifest["specs"] = other + ordered
    manifest["P1_queue"] = {
        "policy": "paired base/tariff/seed blocks; alternate 50/100 jobs; randomized arm order",
        "seed": 20261001,
        "node": node,
        "blocks": 120,
        "ops_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    tmp.replace(path)
    (root / "ops" / "p1_freeze_queue.py").write_bytes(Path(__file__).read_bytes())
print("two paired, balanced 240-run P1 queues frozen without changing any job spec")
