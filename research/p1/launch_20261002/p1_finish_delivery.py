"""One-time background completion delivery; no algorithm retries or credential logging."""

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

folder = Path("E:/LLM调度/worktrees/moead-p1-20261001/outputs/p1_results_20261002")
folder.mkdir(parents=True, exist_ok=True)
aliases = ["geo-llm-exp-20260930", "geo-llm-e16-20260930-20002"]
remote_base = "/workspace/zhouhanyu"
remote_dir = remote_base + "/p1_delivery_20261002"
python = "/workspace/envs/geo-llm-py313/bin/python"
deadline = 1790956800
poll_code = """from pathlib import Path
import json
p=Path('/workspace/zhouhanyu/p1_acceptance_control_20261002')
print(json.dumps({'ready':all((p/f'done_formal_p1_node{n}.json').exists() for n in (0,1))}))
"""
state_path = folder / "delivery_state.json"
try:
    while time.time() < deadline:
        live_alias = None
        for alias in list(aliases):
            r = subprocess.run(
                ["ssh", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, python, "-"],
                input=poll_code,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if r.returncode:
                aliases.remove(alias)
                continue
            live_alias = alias
            ready = json.loads(r.stdout)["ready"]
            break
        if live_alias is None:
            raise RuntimeError(
                "Both verified SSH endpoints became unreachable; no stale-endpoint retries"
            )
        state_path.write_text(
            json.dumps(
                {"phase": "waiting_for_batch_drain", "time": time.time(), "live_alias": live_alias},
                indent=2,
            ),
            encoding="utf-8",
        )
        if ready:
            break
        time.sleep(60)
    else:
        raise RuntimeError("Migration reserve reached before both P1 batches drained")
    state_path.write_text(
        json.dumps({"phase": "packaging", "time": time.time()}, indent=2), encoding="utf-8"
    )
    command = [
        "ssh",
        "-q",
        "-o",
        "BatchMode=yes",
        live_alias,
        "env",
        "PYTHONPATH=" + remote_base + "/p1_source_20261002/src",
        "P1_PART_MIB=500",
        python,
        remote_base + "/p1_acceptance_control_20261002/p1_collect_finished.py",
    ]
    subprocess.run(command, check=True, timeout=7200)
    for name in ("p1_results_manifest.json", "SHA256SUMS"):
        subprocess.run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                live_alias + ":" + remote_dir + "/" + name,
                str(folder / name),
            ],
            check=True,
            timeout=120,
        )
    manifest = json.loads((folder / "p1_results_manifest.json").read_text(encoding="utf-8"))
    assert shutil.disk_usage(folder).free > manifest["archive_size"] * 2 + 5 * 1024**3
    for part in manifest["parts"]:
        local = folder / part["name"]
        valid = False
        if local.exists() and local.stat().st_size == part["size"]:
            h = hashlib.sha256()
            with local.open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    h.update(data)
            valid = h.hexdigest() == part["sha256"]
        if not valid:
            subprocess.run(
                [
                    "scp",
                    "-q",
                    "-o",
                    "BatchMode=yes",
                    live_alias + ":" + remote_dir + "/" + part["name"],
                    str(local),
                ],
                check=True,
                timeout=1800,
            )
        h = hashlib.sha256()
        with local.open("rb") as stream:
            while data := stream.read(1024 * 1024):
                h.update(data)
        assert local.stat().st_size == part["size"] and h.hexdigest() == part["sha256"]
    archive = folder / manifest["archive_name"]
    whole = hashlib.sha256()
    with archive.with_suffix(".partial").open("wb") as dest:
        for part in manifest["parts"]:
            with (folder / part["name"]).open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    whole.update(data)
                    dest.write(data)
    assert whole.hexdigest() == manifest["archive_sha256"]
    archive.with_suffix(".partial").replace(archive)
    expected = {x["path"]: x for x in manifest["members"]}
    checked = set()
    with tarfile.open(archive, "r|") as tar:
        for entry in tar:
            if entry.name == "artifact_manifest.json":
                continue
            item = expected[entry.name]
            assert entry.isfile() and entry.size == item["size"]
            h = hashlib.sha256()
            stream = tar.extractfile(entry)
            assert stream is not None
            while data := stream.read(1024 * 1024):
                h.update(data)
            assert h.hexdigest() == item["sha256"]
            checked.add(entry.name)
    assert checked == set(expected)
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        h = hashlib.sha256()
        with (folder / name).open("rb") as stream:
            while data := stream.read(1024 * 1024):
                h.update(data)
        assert h.hexdigest() == digest
    state_path.write_text(
        json.dumps(
            {
                "phase": "local_verified",
                "time": time.time(),
                "complete": manifest["complete"],
                "members": len(checked),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("p1_release.py")), str(folder), "full"],
        check=True,
        timeout=14400,
    )
    state_path.write_text(
        json.dumps(
            {
                "phase": "local_and_release_verified",
                "time": time.time(),
                "complete": manifest["complete"],
                "members": len(checked),
                "Notion_result_sync": "pending next research analysis",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
except Exception as error:
    state_path.write_text(
        json.dumps(
            {
                "phase": "blocked",
                "time": time.time(),
                "error": type(error).__name__,
                "message": str(error),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    raise
