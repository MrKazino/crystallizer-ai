"""SQLite project memory: decisions, notes, facts and task records.

``query`` scores entries by keyword overlap (fraction of query words present) plus a recency
bonus that halves every ``half_life_days``; ties break by id, so results are deterministic.
``summarize`` compacts old entries into one model-written note and archives the originals.
Nothing is ever deleted. Text is redacted before it is stored.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime

from pydantic import Field

from crystallizer.clock import Clock, iso, parse_iso
from crystallizer.db import Database
from crystallizer.models import ModelClient, encode_request
from crystallizer.redaction import Redactor
from crystallizer.schemas import Completion, MemoryEntry, MemoryKind, Message, Strict

_WORD = re.compile(r"[a-z0-9]+")
RECENCY_WEIGHT = 0.1


class ScoredEntry(Strict):
    """A memory entry with its relevance score."""

    entry: MemoryEntry
    score: float = Field(ge=0)


def words(text: str) -> set[str]:
    """Lower-case alphanumeric words of ``text``."""
    return set(_WORD.findall(text.lower()))


class Memory:
    """Project memory backed by the ``memory`` table."""

    def __init__(
        self, db: Database, clock: Clock, redactor: Redactor, half_life_days: float = 30.0
    ) -> None:
        """Bind to ``db``."""
        self._db = db
        self._clock = clock
        self._redactor = redactor
        self._half_life = half_life_days

    def add(
        self,
        kind: MemoryKind,
        text: str,
        tags: Iterable[str] = (),
        source_task: str | None = None,
        summary_of: Sequence[int] = (),
    ) -> MemoryEntry:
        """Store a redacted entry and return it."""
        cursor = self._db.execute(
            "INSERT INTO memory (kind, text, tags, source_task, created_at, archived, summary_of)"
            " VALUES (?, ?, ?, ?, ?, 0, ?)",
            (
                kind.value,
                self._redactor.redact(text),
                json.dumps(sorted(set(tags))),
                source_task,
                iso(self._clock.now()),
                json.dumps(list(summary_of)),
            ),
        )
        entry_id = cursor.lastrowid
        if entry_id is None:
            raise RuntimeError("sqlite did not return a row id")
        return self.get(entry_id)

    def get(self, entry_id: int) -> MemoryEntry:
        """Return one entry (raises ``KeyError`` if missing)."""
        rows = self._db.query("SELECT * FROM memory WHERE id = ?", (entry_id,))
        if not rows:
            raise KeyError(entry_id)
        return _entry(rows[0])

    def entries(self, include_archived: bool = False) -> list[MemoryEntry]:
        """Return entries ordered by id."""
        sql = "SELECT * FROM memory"
        if not include_archived:
            sql += " WHERE archived = 0"
        return [_entry(row) for row in self._db.query(sql + " ORDER BY id")]

    def max_id(self) -> int:
        """Return the highest entry id (0 when empty); used as the checkpoint snapshot id."""
        rows = self._db.query("SELECT COALESCE(MAX(id), 0) AS m FROM memory")
        return int(rows[0]["m"])

    def query(self, text: str, k: int, now: datetime | None = None) -> list[ScoredEntry]:
        """Return the top ``k`` non-archived entries by overlap plus recency."""
        if k <= 0:
            return []
        moment = now or self._clock.now()
        query_words = words(text)
        scored: list[ScoredEntry] = []
        for entry in self.entries():
            overlap = (
                len(query_words & words(entry.text)) / len(query_words) if query_words else 0.0
            )
            age_days = max((moment - parse_iso(entry.created_at)).total_seconds(), 0.0) / 86400
            recency = 0.5 ** (age_days / self._half_life)
            score = round(overlap + RECENCY_WEIGHT * recency, 12)
            scored.append(ScoredEntry(entry=entry, score=score))
        scored.sort(key=lambda item: (-item.score, item.entry.id))
        return scored[:k]

    def summarize(
        self, older_than: datetime, model: ModelClient, tier: str = "small", max_tokens: int = 512
    ) -> tuple[MemoryEntry | None, Completion | None]:
        """Compact entries created before ``older_than`` into one note; archive the originals."""
        old = [entry for entry in self.entries() if parse_iso(entry.created_at) < older_than]
        if not old:
            return None, None
        payload = {"type": "summarize", "entries": [entry.text for entry in old]}
        listing = "\n".join(f"- ({entry.kind.value}) {entry.text}" for entry in old)
        messages = [
            Message(role="system", content="Summarize these project memory entries concisely."),
            Message(role="user", content=f"{listing}\n{encode_request(payload)}"),
        ]
        completion = model.complete(messages, tier, max_tokens)
        ids = [entry.id for entry in old]
        with self._db.transaction():
            summary = self.add(MemoryKind.NOTE, completion.text, ["summary"], None, ids)
            placeholders = ",".join("?" * len(ids))
            self._db.execute(
                f"UPDATE memory SET archived = 1 WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            )
        return summary, completion


def _entry(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        id=int(row["id"]),
        kind=MemoryKind(row["kind"]),
        text=str(row["text"]),
        tags=list(json.loads(row["tags"])),
        source_task=row["source_task"],
        created_at=str(row["created_at"]),
        archived=bool(row["archived"]),
        summary_of=list(json.loads(row["summary_of"])),
    )
