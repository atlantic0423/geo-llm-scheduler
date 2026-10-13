"""Operational isolation and post-search resource-boundary regression tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/resume_macro_factor_export.py"
SPEC = importlib.util.spec_from_file_location("factor_export_resume", PATH)
assert SPEC and SPEC.loader
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def test_exact_worker_redirect() -> None:
    command = ["python", "-m", "geo_llm_scheduler.experiments.macro_factor_worker", "r", "k"]
    assert wrapper.worker_command(command, Path("w"), Path("s")) == [
        "python",
        "w",
        "worker",
        "s",
        "r",
        "k",
    ]


@pytest.mark.parametrize("inactive,dirty,writing,expected", [(500, 100, 80, 320), (100, 80, 80, 0)])
def test_export_memory_excludes_dirty_and_writeback(
    tmp_path: Path, inactive: int, dirty: int, writing: int, expected: int
) -> None:
    (tmp_path / "memory.max").write_text("1000")
    (tmp_path / "memory.current").write_text("600")
    (tmp_path / "memory.stat").write_text(
        f"inactive_file {inactive}\nfile_dirty {dirty}\nfile_writeback {writing}\n"
    )
    assert wrapper.export_memory(tmp_path) == (400, expected)


@pytest.mark.parametrize(
    "command",
    [
        ["git", "rev-parse", "HEAD"],
        ["du", "-sb", "r"],
        ["python", "-m", "geo_llm_scheduler.experiments.macro_width_worker", "r", "k"],
        ["python", "-m", "geo_llm_scheduler.experiments.macro_factor_worker", "r"],
    ],
)
def test_unrelated_process_is_unchanged(command: list[str]) -> None:
    assert wrapper.worker_command(command, Path("w"), Path("s")) is command


def test_search_finishes_before_export_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_llm_scheduler.experiments import macro_factor_worker as worker

    events = []
    result = object()
    lease = SimpleNamespace(close=lambda: events.append("close"))

    def run(*args: object, **kwargs: object) -> object:
        events.append(("search", args, kwargs))
        return result

    def gate(*args: object) -> object:
        events.append(("gate", args))
        return lease

    def execute() -> None:
        assert worker.run("instance", seconds=1200) is result
        events.append("unchanged-export")

    monkeypatch.setattr(worker, "run", run)
    monkeypatch.setattr(worker, "main", execute)
    monkeypatch.setattr(wrapper, "export_gate", gate)
    wrapper.worker_main(Path("s"), Path("r"), "k")
    assert events == [
        ("search", ("instance",), {"seconds": 1200}),
        ("gate", (Path("r"), "k")),
        "unchanged-export",
        "close",
    ]
    assert worker.run is run


def test_failed_search_never_opens_export_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from geo_llm_scheduler.experiments import macro_factor_worker as worker

    def failure(*args: object, **kwargs: object) -> None:
        raise ValueError("real algorithm failure")

    def forbidden(*args: object) -> None:
        pytest.fail("export gate called before successful search")

    monkeypatch.setattr(worker, "run", failure)
    monkeypatch.setattr(worker, "main", lambda: worker.run())
    monkeypatch.setattr(wrapper, "export_gate", forbidden)
    with pytest.raises(ValueError, match="real algorithm failure"):
        wrapper.worker_main(Path("s"), Path("r"), "k")
    assert worker.run is failure


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="Linux resource API")
def test_only_expected_search_limit_is_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    import resource

    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))
    monkeypatch.setattr(
        resource, "getrlimit", lambda kind: (wrapper.SEARCH_BYTES, wrapper.EXPORT_BYTES)
    )

    def limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (wrapper.SEARCH_BYTES, wrapper.SEARCH_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))

    original = resource.setrlimit
    wrapper.install_child_limits(limits)
    assert calls == [
        (resource.RLIMIT_AS, (wrapper.SEARCH_BYTES, wrapper.EXPORT_BYTES)),
        (resource.RLIMIT_NOFILE, (512, 512)),
    ]
    assert resource.setrlimit is original
