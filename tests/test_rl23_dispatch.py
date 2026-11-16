"""Operations regression tests with real subprocess exit, pause, and corruption."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

entry = Path(__file__).parents[1] / "scripts/rl23_dispatch.py"
spec = importlib.util.spec_from_file_location("flat", entry)
flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat)


def test_cpu_order_and_guard():
    assert flat.cpu_order(
        {0, 1, 2, 3}, {0: ("0", "0"), 1: ("0", "0"), 2: ("0", "1"), 3: ("0", "1")}, 4
    ) == [0, 2, 1, 3]
    assert len(flat.cpu_order(set(range(96)), {i: ("0", str(i % 48)) for i in range(96)}, 96)) == 96
    with pytest.raises(ValueError):
        flat.cpu_order({0}, {0: ("0", "0")}, 2)


class Runner:
    def __init__(self, root):
        self.manifest = dict(
            blocks=["k1", "k2"],
            arms=["A", "B"],
            runs=4,
            source="frozen",
            host="fake",
            window_seconds=3600,
        )
        (root / "preflight").mkdir()
        (root / "specs").mkdir()
        for k in self.manifest["blocks"]:
            self.atomic_json(
                root / "specs" / (k + ".json"),
                dict(order=["B", "A"], seconds=1, input_sha256="input"),
            )

            self.atomic_json(
                root / "preflight" / (k + ".json"),
                dict(source="frozen", host="fake", input_sha256="input"),
            )

    def validate_inputs(self, root):
        return self.manifest

    @staticmethod
    def complete(root, key, arm, source):
        p = root / "runs" / key / arm / "complete.json"
        if p.exists():
            assert json.loads(p.read_text()) == dict(source=source)
            return True
        return False

    @staticmethod
    def atomic_json(path, data):
        path.write_text(json.dumps(data))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    runner = Runner(tmp_path)
    monkeypatch.setattr(flat, "resources_available", lambda _: True)
    monkeypatch.setattr(flat, "orphan_workers", lambda _: [])
    entry = tmp_path / "worker.py"
    entry.write_text("""import sys,json,time
from pathlib import Path
a=sys.argv
r=Path(a[a.index('--root')+1]); k=a[a.index('--key')+1]; arm=a[a.index('--arm')+1]
o=r/'runs'/k/arm; o.mkdir(parents=True,exist_ok=True)
if (r/'inject_failure').exists() and arm=='B': sys.exit(7)
if (r/'inject_pause').exists(): (r/'stop.request').touch()
time.sleep(.05)
(o/'complete.json').write_text(json.dumps(dict(source='frozen')))
""")
    return tmp_path, runner, entry


@pytest.mark.skipif(os.name != "posix", reason="Linux flock/subprocess operations")
def test_frozen_order_skip_and_full_completion(fixture):
    r, runner, entry = fixture
    assert flat.pending_tasks(r, runner.manifest, runner) == [
        ("k1", "B", 1),
        ("k1", "A", 1),
        ("k2", "B", 1),
        ("k2", "A", 1),
    ]
    assert flat.dispatch(r, runner, [0, 1], entry, 0.01)["state"] == "complete"
    assert flat.pending_tasks(r, runner.manifest, runner) == []
    assert flat.dispatch(r, runner, [0, 1], entry, 0.01)["done"] == 4


@pytest.mark.skipif(os.name != "posix", reason="Linux flock/subprocess operations")
def test_failure_stops_dispatch_drains_and_no_automatic_retry(fixture):
    r, runner, entry = fixture
    (r / "inject_failure").touch()
    result = flat.dispatch(r, runner, [0, 1], entry, 0.01)
    assert result["state"] == "failed" and result["done"] == 1
    assert not (r / "runs/k2").exists()
    with pytest.raises(AssertionError, match="Recorded failure"):
        flat.dispatch(r, runner, [0], entry, 0.01)


@pytest.mark.skipif(os.name != "posix", reason="Linux flock/subprocess operations")
def test_pause_drains_without_dispatching_pending(fixture):
    r, runner, entry = fixture
    (r / "inject_pause").touch()
    result = flat.dispatch(r, runner, [0], entry, 0.01)
    assert result["state"] == "stopped" and result["done"] == 1
    assert not (r / "runs/k2").exists()


@pytest.mark.skipif(os.name != "posix", reason="Linux flock/subprocess operations")
def test_orphans_rejected_and_incomplete_attempt_preserved(fixture, monkeypatch):
    r, runner, entry = fixture
    p = r / "runs/k1/B"
    p.mkdir(parents=True)
    (p / "evidence").write_text("interrupted")
    monkeypatch.setattr(flat, "orphan_workers", lambda _: [123])
    with pytest.raises(AssertionError, match="Existing workers"):
        flat.dispatch(r, runner, [0], entry, 0.01)
    assert (p / "evidence").exists()
    monkeypatch.setattr(flat, "orphan_workers", lambda _: [])
    assert flat.dispatch(r, runner, [0, 1], entry, 0.01)["state"] == "complete"
    assert next((r / "quarantine").glob("*/evidence")).read_text() == "interrupted"


def test_forged_complete_and_duplicate_order_rejected(fixture):
    r, runner, entry = fixture
    p = r / "runs/k1/B"
    p.mkdir(parents=True)
    (p / "complete.json").write_text("{}")
    with pytest.raises(AssertionError):
        flat.pending_tasks(r, runner.manifest, runner)
    (p / "complete.json").unlink()
    runner.atomic_json(r / "specs/k1.json", dict(order=["A", "A"], seconds=1))
    with pytest.raises(AssertionError):
        flat.pending_tasks(r, runner.manifest, runner)
