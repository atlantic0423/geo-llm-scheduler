"""Crash-safe SQLite checkpoint for one campaign supervisor."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from geo_llm_scheduler.experiments.campaign.model import JobSpec

TERMINAL = {"SUCCEEDED", "FAILED_PERMANENT", "SKIPPED", "CANCELLED_AT_DEADLINE"}


class StateStore:
    """Persist job state with durable SQLite commits after every transition."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              key TEXT PRIMARY KEY, stage TEXT NOT NULL, arm TEXT NOT NULL,
              instance_seed INTEGER NOT NULL, algorithm_seed INTEGER NOT NULL,
              spec_json TEXT NOT NULL, input_hash TEXT NOT NULL,
              state TEXT NOT NULL, pid INTEGER, slot INTEGER, attempt INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL, updated_at REAL NOT NULL,
              ready_after REAL NOT NULL DEFAULT 0, exit_code INTEGER,
              reason TEXT, elapsed REAL, result_hash TEXT
            );
            CREATE INDEX IF NOT EXISTS jobs_stage_state ON jobs(stage, state);
            """
        )
        self.connection.commit()

    def add(self, spec: JobSpec) -> None:
        """Add a job once; reject a same-key input change instead of reusing results."""
        old = self.connection.execute(
            "SELECT input_hash FROM jobs WHERE key=?", (spec.key,)
        ).fetchone()
        if old:
            if old["input_hash"] != spec.input_hash:
                raise ValueError(f"Job input hash mismatch: {spec.key}")
            return
        now = time.time()
        self.connection.execute(
            """INSERT INTO jobs(key,stage,arm,instance_seed,algorithm_seed,spec_json,
               input_hash,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                spec.key,
                spec.stage,
                spec.arm,
                spec.instance_seed,
                spec.algorithm_seed,
                json.dumps(asdict(spec), ensure_ascii=False, sort_keys=True),
                spec.input_hash,
                "READY",
                now,
                now,
            ),
        )
        self.connection.commit()

    def transition(self, key: str, state: str, **changes: Any) -> None:
        """Atomically record a legal job transition and its diagnostic fields."""
        if state not in {
            "PENDING",
            "WAITING_DEPENDENCY",
            "READY",
            "RUNNING",
            "SUCCEEDED",
            "FAILED_RETRYABLE",
            "FAILED_PERMANENT",
            "SKIPPED",
            "FALLBACK_SELECTED",
            "CANCELLED_AT_DEADLINE",
        }:
            raise ValueError(f"Unknown job state: {state}")
        permitted = {
            "pid",
            "slot",
            "attempt",
            "ready_after",
            "exit_code",
            "reason",
            "elapsed",
            "result_hash",
        }
        if set(changes) - permitted:
            raise ValueError("Unknown checkpoint field")
        parts = ["state=?", "updated_at=?", *(f"{key}=?" for key in changes)]
        values = [state, time.time(), *changes.values(), key]
        cursor = self.connection.execute(f"UPDATE jobs SET {', '.join(parts)} WHERE key=?", values)
        if cursor.rowcount != 1:
            raise KeyError(key)
        self.connection.commit()

    def rows(self, stage: str | None = None) -> list[dict[str, Any]]:
        """Return a detached snapshot so callers cannot mutate DB state implicitly."""
        query = "SELECT * FROM jobs" + (" WHERE stage=?" if stage is not None else "")
        values = (stage,) if stage is not None else ()
        return [dict(row) for row in self.connection.execute(query, values)]

    def get(self, key: str) -> dict[str, Any]:
        """Read one job, raising when its identity is unknown."""
        row = self.connection.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return dict(row)

    def close(self) -> None:
        """Flush and release this process's database connection."""
        self.connection.close()
