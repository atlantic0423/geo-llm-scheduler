"""Package immutable campaign files, then extract and verify every original byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash a file without loading the complete result into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def pack(root: Path, output: Path) -> None:
    """Create independent gzip shards and an external SHA256 file manifest."""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Use the existing manifest; do not replace a recovery snapshot")
    paths = sorted(p for p in root.rglob("*") if p.is_file())

    def describe(path: Path) -> dict:
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(describe, paths))
    shards: list[list[dict]] = [[]]
    size = 0
    for entry in files:
        if size + entry["bytes"] > 4 * 1024**3 and shards[-1]:
            shards.append([])
            size = 0
        shards[-1].append(entry)
        size += entry["bytes"]

    def compress(item: tuple[int, list[dict]]) -> dict:
        number, entries = item
        path = output / f"raw_part_{number:02d}.tar.gz"
        with tarfile.open(path, "w:gz", compresslevel=1) as archive:
            for entry in entries:
                archive.add(root / entry["path"], arcname=entry["path"], recursive=False)
        if path.stat().st_size >= 2_000_000_000:
            raise ValueError("Compressed shard exceeds release asset limit")
        return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        assets = list(pool.map(compress, enumerate(shards, 1)))
    finished = json.loads((root / "finished.json").read_text(encoding="utf-8"))
    manifest = {
        "campaign": root.name,
        "source_commit": finished["commit"],
        "files": files,
        "assets": assets,
        "total_bytes": sum(entry["bytes"] for entry in files),
        "file_count": len(files),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksum_paths = [output / asset["name"] for asset in assets] + [output / "manifest.json"]
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in checksum_paths), encoding="utf-8"
    )
    print(json.dumps({"files": len(files), "raw_bytes": manifest["total_bytes"], "assets": assets}))


def unpack(assets: Path, root: Path, report: Path) -> None:
    """Verify transfer, safely extract without overwriting, and verify all files."""
    checksums = dict(
        (name, value)
        for value, name in (
            line.split(maxsplit=1)
            for line in (assets / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        )
    )
    if sha256(assets / "manifest.json") != checksums.get("manifest.json"):
        raise ValueError("Manifest checksum failed")
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    root.mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        path = assets / asset["name"]
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError(f"Transfer checksum failed: {path.name}")
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.name not in expected or not member.isfile():
                    raise ValueError(f"Unexpected archive member: {member.name}")
                target = (root / member.name).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError("Archive path escapes campaign directory")
                if target.exists():
                    if sha256(target) != expected[member.name]["sha256"]:
                        raise ValueError(f"Refusing to overwrite changed file: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                assert source is not None
                with source, target.open("xb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
        print(f"Verified and extracted {path.name}", flush=True)
    failures = []
    for name, entry in expected.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            failures.append(name)
        elif sha256(path) != entry["sha256"]:
            failures.append(name)
    result = {
        "complete": not failures,
        "verified_files": len(expected) - len(failures),
        "failures": failures,
        "raw_bytes": manifest["total_bytes"],
        "manifest_sha256": sha256(assets / "manifest.json"),
        "source_commit": manifest["source_commit"],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    if failures:
        raise ValueError("Recovered files failed verification")


def main() -> None:
    """Run the server pack step or the local verified recovery step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "pack":
        pack(args.root, args.assets)
    else:
        if args.report is None:
            parser.error("unpack requires --report")
        unpack(args.assets, args.root, args.report)


if __name__ == "__main__":
    main()
