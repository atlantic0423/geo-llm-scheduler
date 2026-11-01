"""Fetch pinned upstream archives, verify hashes and retain attributed algorithm sources.

Raw code/PDFs belong in ignored outputs, not in the project's Python package.
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def main() -> None:
    """Download pinned code archives and safely extract auditable source subsets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/experiments/trans_sources.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = []
    for source in sources["archives"]:
        repo, commit = source["repo"], source["commit"]
        name = repo.split("/")[-1]
        archive_path = args.output / f"{name}-{commit}.zip"
        if not archive_path.exists():
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "geo-llm-scheduler-research"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError("Downloaded archive checksum mismatch")
            archive_path.write_bytes(data)
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Existing archive checksum mismatch; original preserved")
        files = []
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                parts = path.parts[1:]
                if member.is_dir() or not parts:
                    continue
                if path.is_absolute() or ".." in parts or ":" in member.filename:
                    raise ValueError("Unsafe archive path")
                relative = PurePosixPath(*parts)
                rel = str(relative)
                if name == "PlatEMO":
                    keep = (
                        any(
                            f"/{s}/" in rel
                            for s in ("CMOEA-CD", "CMOEA-AOP", "CMOFLCS", "Utility functions")
                        )
                        or rel == "PlatEMO/Metrics/HV.m"
                        or len(parts) == 1
                    )
                else:
                    keep = rel.startswith("src/") or "license" in rel.lower() or len(parts) == 1
                if not keep:
                    continue
                target = args.output / name / Path(*parts)
                data = archive.read(member)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                files.append(
                    {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                )
        manifest.append({**source, "files": files})
        print(f"{name}: archive verified, {len(files)} files retained", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
