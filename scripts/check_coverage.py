"""Enforce overall and explicitly scoped algorithm-core line coverage gates."""

import json
from pathlib import Path

data = json.loads(Path("outputs/coverage.json").read_text(encoding="utf-8"))
core = {
    "evaluation",
    "scheduling",
    "moead",
    "archive",
    "operators",
    "macrosearch",
    "rl",
    "engine",
    "diagnostics",
}
covered = total = 0
for name, entry in data["files"].items():
    parts = name.replace("\\", "/").split("/")
    if any(part in core for part in parts):
        covered += entry["summary"]["covered_lines"]
        total += entry["summary"]["num_statements"]
core_percent = 100 * covered / total
overall = data["totals"]["percent_covered"]
print(f"Overall: {overall:.2f}% (required 80%); core: {core_percent:.2f}% (required 90%)")
assert overall >= 80 and core_percent >= 90
