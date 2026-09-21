"""Create a reproducible source ZIP and SHA256 manifest."""

import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
excluded = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "outputs",
    "snapshots",
    "htmlcov",
    "build",
    "dist",
}
files = sorted(
    p
    for p in root.rglob("*")
    if p.is_file()
    and not any(x in excluded or x.endswith(".egg-info") for x in p.relative_to(root).parts)
    and p.name != ".coverage"
)
manifest = {
    p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
}
digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
out = root / "snapshots"
out.mkdir(exist_ok=True)
target = out / ("geo_llm_scheduler_" + digest[:12] + ".zip")
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    info = zipfile.ZipInfo("SOURCE_MANIFEST.json", (2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, json.dumps(manifest, indent=2))
    for p in files:
        info = zipfile.ZipInfo(p.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, p.read_bytes())
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "zip": str(target),
            "source_hash": digest,
            "zip_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "files": len(files),
        }
    )
)
