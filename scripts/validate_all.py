"""Run the project's local quality gates without hiding command failures."""

import os
import subprocess
import sys
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)
# pytest's pythonpath setting does not reach worker subprocesses. Bind every
# gate and its descendants to this worktree, even if another checkout is editable-installed.
gate_env = os.environ.copy()
gate_env["PYTHONPATH"] = (
    str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + gate_env.get("PYTHONPATH", "")
)
for args in (
    ["ruff", "check", "."],
    ["ruff", "format", "--check", "src", "tests", "scripts"],
    ["mypy", "src"],
    [
        "pytest",
        "-q",
        "--cov=geo_llm_scheduler",
        "--cov-report=json:outputs/coverage.json",
        "--cov-report=",
    ],
):
    subprocess.run([sys.executable, "-m", *args], check=True, env=gate_env)
subprocess.run([sys.executable, "scripts/check_coverage.py"], check=True, env=gate_env)
