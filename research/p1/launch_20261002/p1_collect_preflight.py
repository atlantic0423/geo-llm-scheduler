"""Collect completed P1 preflight evidence without touching active formal runs."""

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

base = Path("/workspace/zhouhanyu")
control = base / "p1_acceptance_control_20261002"
acceptance = json.loads((control / "acceptance.json").read_text())
assert acceptance["accepted"]
assert json.loads((control / "coordinator_state.json").read_text())["phase"] == "P1_active"
selected = {}
plan_sha = json.loads((base / "campaign_p1_20261002_node0" / "manifest.json").read_text())[
    "plan_sha256"
]
for p in base.glob("*.json"):
    if hashlib.sha256(p.read_bytes()).hexdigest() == plan_sha:
        selected[str(p.relative_to(base))] = p
assert selected, "Frozen plan not found"
for node in (0, 1):
    root = base / f"campaign_p1_20261002_node{node}"
    for folder in ("inputs", "specs"):
        for p in (root / folder).rglob("*"):
            if p.is_file():
                selected[str(p.relative_to(base))] = p
    selected[str((root / "manifest.json").relative_to(base))] = root / "manifest.json"
    for p in (root / "ops").rglob("*"):
        if p.is_file() and "P1" not in p.name:
            selected[str(p.relative_to(base))] = p
    for run in (root / "runs").iterdir():
        if run.name.startswith("P1"):
            continue
        marker = json.loads((run / "complete.json").read_text())
        for name, entry in marker["files"].items():
            assert (run / name).stat().st_size == entry["size"]
            assert hashlib.sha256((run / name).read_bytes()).hexdigest() == entry["sha256"]
        for p in run.rglob("*"):
            if p.is_file():
                selected[str(p.relative_to(base))] = p
    for p in (root / "logs").glob("*"):
        if (
            p.is_file()
            and "P1" not in p.name
            and "formal_p1" not in p.name
            and p.name != "acceptance_node.log"
        ):
            selected[str(p.relative_to(base))] = p
    selected[str((root / "status.json").relative_to(base))] = root / "status.json"
for name in (
    "acceptance.json",
    "coordinator_state.json",
    "coordinator.log",
    "historical_input_check.json",
    "freeze_validation.json",
):
    p = control / name
    if p.exists():
        selected[str(p.relative_to(base))] = p
for p in control.glob("done_*.json"):
    selected[str(p.relative_to(base))] = p
rows = []
payloads = {}
for name, p in sorted(selected.items()):
    data = p.read_bytes()
    payloads[name] = data
    rows.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
manifest = {
    "scope": "Completed P0/replay/RSS/throughput evidence and frozen P1 inputs; formal P1 results remain active",
    "source_commit": acceptance["source_commit"],
    "collected_at": time.time(),
    "files": rows,
    "acceptance": acceptance,
}
out = control / "p1_preflight_20261002.tar.gz"

with tarfile.open(out, "w:gz") as tar:
    for name, data in payloads.items():
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    data = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    info = tarfile.TarInfo("preflight_manifest.json")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
manifest_path = control / "preflight_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
checksum = hashlib.sha256(out.read_bytes()).hexdigest()
(control / "SHA256SUMS").write_text(f"{checksum}  {out.name}\n")
print(
    json.dumps(
        {
            "archive": str(out),
            "size": out.stat().st_size,
            "sha256": checksum,
            "files": len(rows),
            "manifest": str(manifest_path),
        }
    )
)
