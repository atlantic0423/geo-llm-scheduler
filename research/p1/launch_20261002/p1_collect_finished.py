"""Package drained P1 campaigns and split their verified evidence for delivery."""

import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.p1_worker import validate_spec

base = Path(os.environ.get("P1_DELIVERY_BASE", "/workspace/zhouhanyu"))
part_bytes = int(os.environ.get("P1_PART_MIB", "500")) * 1024 * 1024
assert 1024 * 1024 <= part_bytes <= 500 * 1024 * 1024
control = base / "p1_acceptance_control_20261002"
out = base / "p1_delivery_20261002"
out.mkdir(exist_ok=True)
acceptance = json.loads((control / "acceptance.json").read_text())
assert acceptance["accepted"]
members = {}
completion = []
for node in (0, 1):
    done = json.loads((control / f"done_formal_p1_node{node}.json").read_text())
    root = base / f"campaign_p1_20261002_node{node}"
    manifest = json.loads((root / "manifest.json").read_text())
    planned = [s for s in manifest["specs"] if s["stage"] == "P1"]
    assert len(planned) == 240
    valid = []
    for item in planned:
        spec = JobSpec(**json.loads(Path(item["path"]).read_text()))
        if validate_spec(root / "runs" / item["key"], spec):
            valid.append(item["key"])
    completion.append(
        {
            "node": node,
            "planned": 240,
            "completed": len(valid),
            "batch_success": done["success"],
            "completed_keys": valid,
        }
    )
    for p in root.rglob("*"):
        if p.is_file():
            assert p.resolve().is_relative_to(base.resolve())
            members[p.relative_to(base).as_posix()] = p
for p in control.rglob("*"):
    if p.is_file() and p.suffix not in (".gz", ".tar") and p.name != "SHA256SUMS":
        members[p.relative_to(base).as_posix()] = p
plan_sha = json.loads((base / "campaign_p1_20261002_node0" / "manifest.json").read_text())[
    "plan_sha256"
]
for p in base.glob("*.json"):
    if hashlib.sha256(p.read_bytes()).hexdigest() == plan_sha:
        members[p.relative_to(base).as_posix()] = p
rows = []
for name, p in sorted(members.items()):
    h = hashlib.sha256()
    with p.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            h.update(data)
    rows.append({"path": name, "size": p.stat().st_size, "sha256": h.hexdigest()})
record = {
    "scope": "P1 raw sampling, diagnostics, frozen inputs, preflight and failure evidence; analysis pending",
    "source_commit": acceptance["source_commit"],
    "collected_at": time.time(),
    "complete": all(x["completed"] == 240 and x["batch_success"] for x in completion),
    "completion": completion,
    "members": rows,
}
archive = out / "p1_results_20261002.tar"
with tarfile.open(archive.with_suffix(".partial"), "w") as tar:
    for row in rows:
        tar.add(members[row["path"]], arcname=row["path"], recursive=False)
    data = json.dumps(record, ensure_ascii=False, indent=2).encode()
    info = tarfile.TarInfo("artifact_manifest.json")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
archive.with_suffix(".partial").replace(archive)
whole = hashlib.sha256()
parts = []
with archive.open("rb") as stream:
    index = 0
    while True:
        first = stream.read(1024 * 1024)
        if not first:
            break
        name = archive.name + f".part{index:03d}"
        h = hashlib.sha256()
        size = 0
        with (out / name).open("wb") as dest:
            data = first
            while data:
                dest.write(data)
                h.update(data)
                whole.update(data)
                size += len(data)
                if size >= part_bytes:
                    break
                data = stream.read(1024 * 1024)
        parts.append({"name": name, "size": size, "sha256": h.hexdigest()})
        index += 1
record.update(
    archive_name=archive.name,
    archive_sha256=whole.hexdigest(),
    archive_size=archive.stat().st_size,
    part_capacity_bytes=part_bytes,
    parts=parts,
)
manifest_path = out / "p1_results_manifest.json"
manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
checksums = "".join(f"{p['sha256']}  {p['name']}\n" for p in parts)
checksums += f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  {manifest_path.name}\n"
(out / "SHA256SUMS").write_text(checksums)
print(
    json.dumps(
        {
            "directory": str(out),
            "complete": record["complete"],
            "completed": sum(x["completed"] for x in completion),
            "parts": len(parts),
            "archive_sha256": record["archive_sha256"],
        }
    )
)
