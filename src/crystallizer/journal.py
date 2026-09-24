"""Write-ahead idempotency journal for tool actions.

Key: ``(run_id, task_id, attempt, step_index, action_hash)``. The attempt is part of the key so
a deliberate retry re-executes, while a resume of the same attempt never repeats a side effect.

Protocol: ``started`` is committed before the tool runs and ``completed`` (with the redacted
result) after. On resume:

* ``completed``: the recorded result is reused and the tool is not called;
* ``started`` only (the process died mid-action, outcome unknown, "in doubt"): the action is
  re-executed only when the caller allows it (idempotent actions, or a human approved);
  otherwise :class:`InDoubtActionError` is raised.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from crystallizer.clock import Clock, iso
from crystallizer.db import Database
from crystallizer.errors import InDoubtActionError
from crystallizer.faults import NO_FAULTS, FaultInjector
from crystallizer.hashing import digest
from crystallizer.schemas import Action, JournalEntry, JournalStatus, Strict, ToolResult


class JournalOutcome(Strict):
    """Result of a journaled execution."""

    result: ToolResult
    replayed: bool = False
    redone: bool = False
    key: str = Field(min_length=1)


def action_hash(action: Action) -> str:
    """Stable hash of an action."""
    return digest(action)


class Journal:
    """Idempotency journal backed by the ``journal`` table."""

    def __init__(self, db: Database, clock: Clock, faults: FaultInjector = NO_FAULTS) -> None:
        """Bind to ``db``."""
        self._db = db
        self._clock = clock
        self._faults = faults

    @staticmethod
    def key(run_id: str, task_id: str, attempt: int, step_index: int, action: Action) -> str:
        """Return the idempotency key for one action."""
        return digest([run_id, task_id, attempt, step_index, action_hash(action)])

    def lookup(self, key: str) -> JournalEntry | None:
        """Return the entry for ``key`` if one exists."""
        rows = self._db.query("SELECT * FROM journal WHERE key = ?", (key,))
        if not rows:
            return None
        row = rows[0]
        result = ToolResult.model_validate_json(row["result"]) if row["result"] else None
        return JournalEntry(
            key=row["key"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            attempt=int(row["attempt"]),
            step_index=int(row["step_index"]),
            action_hash=row["action_hash"],
            status=JournalStatus(row["status"]),
            result=result,
            updated_at=row["updated_at"],
        )

    def execute(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt: int,
        step_index: int,
        action: Action,
        run: Callable[[], ToolResult],
        allow_redo: bool,
    ) -> JournalOutcome:
        """Run ``action`` at most once per key; see the module docstring for the protocol."""
        key = self.key(run_id, task_id, attempt, step_index, action)
        existing = self.lookup(key)
        redone = False
        if existing is not None and existing.status is JournalStatus.COMPLETED:
            if existing.result is None:
                raise InDoubtActionError(f"journal entry {key[:12]} has no recorded result")
            return JournalOutcome(result=existing.result, replayed=True, key=key)
        if existing is not None:
            if not allow_redo:
                raise InDoubtActionError(
                    f"action {action.tool} (task {task_id}, step {step_index}) started before a "
                    "crash and its outcome is unknown"
                )
            redone = True
        else:
            self._db.execute(
                "INSERT INTO journal (key, run_id, task_id, attempt, step_index, action_hash,"
                " status, result, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    key,
                    run_id,
                    task_id,
                    attempt,
                    step_index,
                    action_hash(action),
                    JournalStatus.STARTED.value,
                    iso(self._clock.now()),
                ),
            )
        self._faults.hit("journal.after_begin")
        result = run()
        self._faults.hit("journal.after_execute")
        self._db.execute(
            "UPDATE journal SET status = ?, result = ?, updated_at = ? WHERE key = ?",
            (JournalStatus.COMPLETED.value, result.model_dump_json(), iso(self._clock.now()), key),
        )
        return JournalOutcome(result=result, redone=redone, key=key)

    def entries(self, run_id: str | None = None) -> list[JournalEntry]:
        """Return entries (optionally for one run) in insertion order."""
        sql = "SELECT key FROM journal"
        params: tuple[str, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        keys = [row["key"] for row in self._db.query(sql + " ORDER BY rowid", params)]
        return [entry for key in keys if (entry := self.lookup(key)) is not None]
