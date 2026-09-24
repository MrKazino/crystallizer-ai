"""Single SQLite connection manager for all internal state tables.

WAL mode, a busy timeout, ``synchronous=FULL`` for journal durability, foreign keys on, and
versioned migrations. Only internal modules use it; tools can never write inside ``state_dir``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

BUSY_TIMEOUT_MS = 5000

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        text TEXT NOT NULL,
        tags TEXT NOT NULL,
        source_task TEXT,
        created_at TEXT NOT NULL,
        archived INTEGER NOT NULL DEFAULT 0,
        summary_of TEXT NOT NULL DEFAULT '[]'
    );
    CREATE TABLE journal (
        key TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        step_index INTEGER NOT NULL,
        action_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        result TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE task_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        task_id TEXT NOT NULL,
        from_state TEXT NOT NULL,
        to_state TEXT NOT NULL,
        reason TEXT NOT NULL,
        ts TEXT NOT NULL
    );
    CREATE TABLE shadow_obs (
        skill_key TEXT NOT NULL,
        occurrence TEXT NOT NULL,
        passed INTEGER NOT NULL,
        unsafe INTEGER NOT NULL,
        ts TEXT NOT NULL,
        PRIMARY KEY (skill_key, occurrence)
    );
    CREATE TABLE live_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        skill_key TEXT NOT NULL,
        occurrence TEXT NOT NULL,
        passed INTEGER NOT NULL,
        ts TEXT NOT NULL,
        UNIQUE (skill_key, occurrence)
    );
    """,
)


class Database:
    """Owns the one connection to ``state.db``."""

    def __init__(self, path: Path) -> None:
        """Open (creating if needed) the database at ``path`` and apply migrations."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(
            str(path), timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._depth = 0
        self._migrate()

    def _migrate(self) -> None:
        self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = int(row["v"]) if row is not None and row["v"] is not None else 0
        for number, script in enumerate(MIGRATIONS, start=1):
            if number <= current:
                continue
            with self.transaction():
                for statement in script.split(";"):
                    if statement.strip():
                        self._conn.execute(statement)
                self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (number,))

    @property
    def schema_version(self) -> int:
        """The applied migration number."""
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        return int(row["v"])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the block in one ``BEGIN IMMEDIATE`` transaction (re-entrant)."""
        if self._depth > 0:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self._depth = 0
            self._conn.execute("ROLLBACK")
            raise
        self._depth = 0
        self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute one statement (auto-committed unless inside :meth:`transaction`)."""
        return self._conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Execute a query and return all rows."""
        return list(self._conn.execute(sql, tuple(params)).fetchall())

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def __enter__(self) -> Database:
        """Return self."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on exit."""
        self.close()
