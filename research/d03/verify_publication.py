"""Validate the public D03 results bundle against its frozen formal aggregates."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
REPORT = ROOT / "docs" / "reports" / "d03_results_figure_analysis.md"
CHECKSUMS = PACKAGE / "checksums.sha256"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV as named rows for independent count checks."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_checksums() -> None:
    """Write deterministic checksums for all curated data and figure files."""
    files = sorted(
        path
        for folder in ("data", "figures")
        for path in (PACKAGE / folder).rglob("*")
        if path.is_file()
    )
    CHECKSUMS.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PACKAGE).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def verify() -> None:
    """Assert bundle completeness, numerical reconciliation, link integrity and hashes."""
    state = json.loads((PACKAGE / "data" / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["formal_complete"] == state["formal_expected"] == 57
    assert state["type_i"]["runs"] == 42 and state["type_ii"]["runs"] == 15
    assert state["type_i"]["steps"] == 257010 and state["type_ii"]["steps"] == 95050
    assert state["freeze_commit"] == "2526ad1afc425b4d6ff847055eebd4785744acc8"

    suite = csv_rows(PACKAGE / "data" / "analysis" / "suite.csv")
    assert {row["suite"] for row in suite} == {"D03-I", "D03-II"}
    assert {row["suite"]: int(row["runs"]) for row in suite} == {"D03-I": 42, "D03-II": 15}
    assert {row["suite"]: int(row["generations"]) for row in suite} == {
        "D03-I": 50,
        "D03-II": 50,
    }
    cells = csv_rows(PACKAGE / "data" / "analysis" / "state_action.csv")
    for name, steps in (("D03-I", 257010), ("D03-II", 95050)):
        assert sum(int(row["n"]) for row in cells if row["suite"] == name) == steps

    assert len(csv_rows(PACKAGE / "data" / "manifests" / "type_i_evaluation.csv")) == 14
    assert len(csv_rows(PACKAGE / "data" / "manifests" / "type_ii_evaluation.csv")) == 5
    figures = PACKAGE / "figures"
    png = {path.relative_to(figures).with_suffix("") for path in figures.rglob("*.png")}
    pdf = {path.relative_to(figures).with_suffix("") for path in figures.rglob("*.pdf")}
    assert len(png) == len(pdf) == 41 and png == pdf

    report = REPORT.read_text(encoding="utf-8")
    assert len(re.findall(r"^### (?:I-|II-|H-|S\d|M\d)", report, re.MULTILINE)) == 41
    for stem in png:
        assert f"{stem.as_posix()}.png" in report
        assert f"{stem.as_posix()}.pdf" in report
    for markdown in (REPORT, PACKAGE / "README.md", PACKAGE / "data" / "D03_SUMMARY.md"):
        body = markdown.read_text(encoding="utf-8")
        assert "\\(" not in body and "\\)" not in body
        assert "\\[" not in body and "\\]" not in body
        for target in re.findall(r"!?\[[^]]+\]\(([^)]+)\)", body):
            if "://" not in target and not target.startswith("#"):
                assert (markdown.parent / target).resolve().is_file(), (markdown, target)

    for path in (PACKAGE / "data").rglob("*"):
        if path.suffix in {".json", ".csv", ".md"}:
            body = path.read_text(encoding="utf-8")
            assert "E:/LLM" not in body and "E:\\" not in body and "C:\\" not in body, path
    lines = CHECKSUMS.read_text(encoding="utf-8").splitlines()
    expected = sum(
        1
        for folder in ("data", "figures")
        for path in (PACKAGE / folder).rglob("*")
        if path.is_file()
    )
    assert len(lines) == expected
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert sha256(PACKAGE / rel) == digest, rel
    print(f"D03 publication verified: 57/57 runs, 41 PNG/PDF pairs, {len(lines)} hashed files")


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-checksums"]:
        write_checksums()
    elif sys.argv[1:]:
        raise SystemExit("usage: python research/d03/verify_publication.py [--write-checksums]")
    verify()
