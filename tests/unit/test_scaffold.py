"""Phase 0 installation, configuration and documentation gates."""

import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_packages_import():
    for name in ("domain", "evaluation", "archive", "macrosearch", "rl", "engine"):
        assert importlib.import_module("geo_llm_scheduler." + name)


def test_working_config():
    data = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    assert data["budgets"] == [3, 6, 10]
    assert len(data["severity_thresholds"]) == 6


def test_markdown_formula_delimiters():
    forbidden = [chr(92) + x for x in ("(", ")", "[", "]")]
    for path in ROOT.rglob("*.md"):
        if any(p.startswith(".") for p in path.relative_to(ROOT).parts):
            continue
        value = path.read_text(encoding="utf-8")
        assert not any(x in value for x in forbidden), str(path)
