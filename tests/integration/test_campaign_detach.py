"""A child spawned by a short-lived parent survives parent process exit."""

import subprocess
import sys
import time


def test_detach_survives_parent_exit(tmp_path):
    marker = tmp_path / "survived.txt"
    log = tmp_path / "child.log"
    child_code = (
        f"import pathlib,time; time.sleep(0.35); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    parent_code = (
        "import pathlib,sys; "
        "from geo_llm_scheduler.experiments.campaign.support import detached_process; "
        f"detached_process([sys.executable,'-c',{child_code!r}],"
        f"pathlib.Path({str(tmp_path)!r}),pathlib.Path({str(log)!r}))"
    )
    parent = subprocess.run(
        [sys.executable, "-c", parent_code],
        check=True,
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert parent.returncode == 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.read_text() == "alive"
