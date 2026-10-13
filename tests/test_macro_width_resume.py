"""Ensure the operational quota wrapper never weakens a frozen resource guard."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/resume_macro_width.py"))
CAMPAIGN = Path("/workspace/owner/macro_width_20261012")
COMMAND = ["du", "-sb", str(CAMPAIGN.parent)]
ENOENT = (
    f"du: cannot access '{CAMPAIGN}/formal_node0/ops/heartbeats/"
    ".n50_example.json.123.tmp': No such file or directory\n"
)


@pytest.mark.parametrize(
    "stderr",
    [
        "",
        ENOENT.replace("No such file or directory", "Permission denied"),
        ENOENT.replace("macro_width_20261012", "other_project"),
        ENOENT.replace(".123.tmp", ".123.json"),
        ENOENT + "du: unrelated failure\n",
    ],
)
def test_error_match_excludes_unrelated_and_permanent_failures(stderr: str) -> None:
    assert not MODULE["transient_heartbeat_error"](stderr, CAMPAIGN)


def test_success_and_unrelated_commands_preserve_output() -> None:
    calls = []

    def original(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "1234\t/workspace/owner\n"

    assert (
        MODULE["quota_check_output"](original, CAMPAIGN, COMMAND, text=True)
        == "1234\t/workspace/owner\n"
    )
    assert calls[0][1]["stderr"] == subprocess.PIPE
    other = ["git", "rev-parse", "HEAD"]
    MODULE["quota_check_output"](original, CAMPAIGN, other, text=True)
    assert calls[1] == ((other,), {"text": True})


def test_atomic_replacement_retries_to_an_exact_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE["time"], "sleep", lambda _: None)
    attempts = 0

    def original(*args: object, **kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.CalledProcessError(
                1, COMMAND, output="100\t/workspace/owner\n", stderr=ENOENT
            )
        return "4567\t/workspace/owner\n"

    assert (
        MODULE["quota_check_output"](original, CAMPAIGN, COMMAND, text=True)
        == "4567\t/workspace/owner\n"
    )
    assert attempts == 2


@pytest.mark.parametrize("text", [True, False])
def test_persistent_transience_closes_existing_quota_guard(
    text: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE["time"], "sleep", lambda _: None)
    attempts = 0

    def original(*args: object, **kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        raise subprocess.CalledProcessError(1, COMMAND, stderr=ENOENT if text else ENOENT.encode())

    output = MODULE["quota_check_output"](original, CAMPAIGN, COMMAND, text=text)
    assert isinstance(output, str if text else bytes)
    quota_left = 100 * 1024**3 - int(output.split()[0])
    assert quota_left < 2 * 1024**3
    assert attempts == 4


def test_permission_failure_is_not_silently_retried() -> None:
    error = subprocess.CalledProcessError(
        1, COMMAND, stderr=ENOENT.replace("No such file or directory", "Permission denied")
    )

    def original(*args: object, **kwargs: object) -> str:
        raise error

    with pytest.raises(subprocess.CalledProcessError) as caught:
        MODULE["quota_check_output"](original, CAMPAIGN, COMMAND, text=True)
    assert caught.value is error
