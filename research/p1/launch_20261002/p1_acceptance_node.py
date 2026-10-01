"""Frozen operational dispatcher for the P1 resource/replay acceptance waves."""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

base = Path("/workspace/zhouhanyu")
source = base / "p1_source_20261002"
node = int(os.environ["P1_NODE"])
root = base / f"campaign_p1_20261002_node{node}"
control = base / "p1_acceptance_control_20261002"
control.mkdir(exist_ok=True)
py = "/workspace/envs/geo-llm-py313/bin/python"
env = {
    **os.environ,
    "PYTHONPATH": str(source / "src"),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
completed = set()


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data))
    os.replace(temp, path)


while True:
    atomic(
        control / f"node{node}_alive.json",
        {"pid": os.getpid(), "host": platform.node(), "time": time.time()},
    )
    request = control / f"request_node{node}.json"
    if not request.exists():
        time.sleep(2)
        continue
    task = json.loads(request.read_text())
    tag = task["tag"]
    if tag in completed:
        time.sleep(2)
        continue
    if task["phase"] == "EXIT":
        break
    commands = []
    if task["phase"] == "ACCEPT":
        for phase, workers in [("P0", 5 if node == 0 else 6), ("RSS", 2)]:
            commands.append(
                [
                    py,
                    "scripts/run_p1_campaign.py",
                    "batch",
                    "--root",
                    str(root),
                    "--phase",
                    phase,
                    "--workers",
                    str(workers),
                ]
            )
        if node == 0:
            manifest = json.loads((root / "manifest.json").read_text())
            for arm in ["F6", "M0", "N0"]:
                item = next(
                    s
                    for s in manifest["specs"]
                    if s["stage"] == "P0R1" and s["arm"] == arm and s["base_seed"] == 710001
                )
                commands.append(
                    [
                        py,
                        str(source / "scripts/p1_legacy_probe.py"),
                        item["path"],
                        str(root / "ops" / f"legacy_{arm}.json"),
                    ]
                )
    else:
        commands = [
            [
                py,
                "scripts/run_p1_campaign.py",
                "batch",
                "--root",
                str(root),
                "--phase",
                task["phase"],
                "--workers",
                str(task["workers"]),
            ]
        ]
    processes = []
    for index, command in enumerate(commands):
        selected_env = dict(env)
        if task["phase"] == "ACCEPT" and index >= 2:
            selected_env["PYTHONPATH"] = str(base / "p1_reference_3c59ada" / "src")
            # Old trace snapshots are deliberately restricted to one process at a time.
            continue
        log = (root / "logs" / f"accept_{tag}_{index}.log").open("wb")
        p = subprocess.Popen(
            command,
            cwd=source,
            env=selected_env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        processes.append((p, command))
        log.close()
    legacy_exit = []
    if task["phase"] == "ACCEPT" and node == 0:
        for index, command in enumerate(commands[2:]):
            old_env = {**env, "PYTHONPATH": str(base / "p1_reference_3c59ada" / "src")}
            with (root / "logs" / f"legacy_{index}.log").open("wb") as log:
                r = subprocess.run(command, cwd=source, env=old_env, stdout=log, stderr=log)
            legacy_exit.append(r.returncode)
            if r.returncode:
                break
    while any(p.poll() is None for p, _ in processes):
        atomic(
            control / f"node{node}_alive.json",
            {
                "pid": os.getpid(),
                "host": platform.node(),
                "time": time.time(),
                "tag": tag,
                "children": [p.pid for p, _ in processes if p.poll() is None],
            },
        )
        time.sleep(5)
    result = {
        "node": node,
        "tag": tag,
        "commands": [cmd for _, cmd in processes],
        "exits": [p.returncode for p, _ in processes],
        "legacy_exits": legacy_exit,
        "success": all(p.returncode == 0 for p, _ in processes)
        and all(x == 0 for x in legacy_exit),
        "finished": time.time(),
    }
    atomic(control / f"done_{tag}_node{node}.json", result)
    completed.add(tag)
    if not result["success"]:
        break
