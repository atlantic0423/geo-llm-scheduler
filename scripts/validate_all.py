"""Run the project's local quality gates without hiding command failures."""

import subprocess
import sys
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)
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
    subprocess.run([sys.executable, "-m", *args], check=True)
subprocess.run([sys.executable, "scripts/check_coverage.py"], check=True)
